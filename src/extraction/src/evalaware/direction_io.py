"""Load r̂ from whatever the step 2 owner produces, and validate it.

Deliberately tolerant about *format* and strict about *content*. We do not
control how the SVD step saves its output, so accept `.npz`, `.npy` and `.pt`,
and both `(n_layers, d)` and `(d,)` shapes. But everything that would silently
corrupt a downstream number — non-finite values, wrong dimension, non-unit
norm, an unrecorded layer mapping — is checked and reported.

The one thing that cannot be validated from the file alone is the **sign**.
`numpy.linalg.svd` returns v₁ only up to sign, so a flipped r̂ is a perfectly
well-formed unit vector that happens to point the wrong way. Step 3a detects
that empirically (AUROC near 0 rather than near 1) and says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .validation import assert_finite, check

__all__ = ["LoadedDirections", "load_directions", "load_direction_metadata"]


@dataclass
class LoadedDirections:
    """r̂ for one or more layers, plus whatever provenance we could recover."""

    vectors: np.ndarray                 # (n_layers, d) float64, unit rows
    layers: list[int]                   # layer id for each row
    dim: int
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    renormalized: bool = False

    def for_layer(self, layer: int) -> np.ndarray:
        check(
            layer in self.layers,
            f"no direction for layer {layer}. Available: {self.layers}. "
            f"If step 2 produced directions for a different layer set than was "
            f"captured, the two must be reconciled explicitly -- do not assume "
            f"row order matches.",
        )
        return self.vectors[self.layers.index(layer)]

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source_path,
            "n_layers": len(self.layers),
            "layers": self.layers,
            "dim": self.dim,
            "renormalized": self.renormalized,
            "metadata": self.metadata,
        }


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(path)

    if suffix == ".npz":
        with np.load(path) as data:
            keys = list(data.keys())
            for candidate in ("r_hat", "r", "direction", "directions", "v1", "arr_0"):
                if candidate in keys:
                    return data[candidate]
            check(
                len(keys) == 1,
                f"{path}: expected a key named one of r_hat/r/direction/v1, or a "
                f"single array. Found {keys}. Rename the key or say which to use.",
            )
            return data[keys[0]]

    if suffix in (".pt", ".pth", ".bin"):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                f"{path} is a torch checkpoint but torch is not installed.\n"
                f"  pip install torch\n"
                f"Or ask the step 2 owner for a .npz instead -- numpy formats "
                f"avoid this dependency entirely."
            ) from exc
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for candidate in ("r_hat", "r", "direction", "directions", "v1"):
                if candidate in obj:
                    obj = obj[candidate]
                    break
            else:
                keys = list(obj.keys())
                check(
                    len(keys) == 1,
                    f"{path}: torch dict with keys {keys}; expected one of "
                    f"r_hat/r/direction/v1 or a single entry.",
                )
                obj = obj[keys[0]]
        # bf16 has ~3 decimal digits; cosines need better than that.
        return obj.detach().float().cpu().numpy()

    raise ValueError(
        f"{path}: unsupported format {suffix!r}. Expected .npz, .npy or .pt."
    )


def load_direction_metadata(path: str | Path) -> dict[str, Any]:
    """Read a sidecar JSON if the step 2 owner supplied one."""
    path = Path(path)
    for candidate in (
        path.with_suffix(".json"),
        path.parent / f"{path.stem}_meta.json",
        path.parent / "direction_meta.json",
    ):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return {}


def load_directions(
    path: str | Path,
    layers: list[int] | None = None,
    expected_dim: int = 8192,
    norm_tol: float = 1e-4,
) -> LoadedDirections:
    """Load and validate r̂.

    ``layers`` gives the layer id for each row when the file does not record it.
    Required for a 2-D array unless the metadata supplies ``layers``.
    """
    path = Path(path)
    check(path.exists(), f"direction file not found: {path}")

    arr = np.asarray(_load_array(path), dtype=np.float64)
    assert_finite(arr, f"directions from {path.name}")
    meta = load_direction_metadata(path)

    if arr.ndim == 1:
        arr = arr[None, :]
        resolved = layers or meta.get("layers") or [0]
        check(
            len(resolved) == 1,
            f"{path}: got a single direction of shape {arr.shape} but "
            f"{len(resolved)} layers were specified.",
        )
    else:
        check(
            arr.ndim == 2,
            f"{path}: expected 1-D or 2-D, got shape {arr.shape}.",
        )
        resolved = layers or meta.get("layers")
        check(
            resolved is not None,
            f"{path}: array is 2-D with {arr.shape[0]} rows but no layer mapping "
            f"was provided and none is recorded in metadata. Pass layers=[...] "
            f"or ask step 2 for the sidecar JSON. Guessing row order is exactly "
            f"the mistake that produces a silently wrong result.",
        )
        check(
            len(resolved) == arr.shape[0],
            f"{path}: {arr.shape[0]} direction rows but {len(resolved)} layer "
            f"ids given. These must correspond one-to-one.",
        )

    # A transposed (d, n_layers) array is a plausible mistake; catch it.
    if arr.shape[1] != expected_dim and arr.shape[0] == expected_dim:
        raise AssertionError(
            f"{path}: shape {arr.shape} looks transposed -- dimension "
            f"{expected_dim} is on axis 0, expected axis 1. Pass "
            f"arr.T to the saver, or confirm the intended layout."
        )
    check(
        arr.shape[1] == expected_dim,
        f"{path}: direction dimension is {arr.shape[1]}, expected {expected_dim} "
        f"(the model's hidden size). Wrong model, or wrong array.",
    )

    norms = np.linalg.norm(arr, axis=1)
    check(
        float(norms.min()) > 0,
        f"{path}: at least one direction is the zero vector.",
    )
    renormalized = bool(np.any(np.abs(norms - 1.0) > norm_tol))
    if renormalized:
        import warnings
        warnings.warn(
            f"{path}: directions were not unit norm "
            f"(min {norms.min():.6f}, max {norms.max():.6f}); renormalizing. "
            f"Projections are scale-invariant for AUROC, so this does not change "
            f"the OOD result -- but Phase 2's r̂r̂ᵀ projection assumes unit norm, "
            f"so the step 2 owner should fix it at source.",
            stacklevel=2,
        )
        arr = arr / norms[:, None]

    return LoadedDirections(
        vectors=arr,
        layers=[int(l) for l in resolved],
        dim=int(arr.shape[1]),
        source_path=str(path),
        metadata=meta,
        renormalized=renormalized,
    )
