"""Phase 1 directions -> a per-block edit plan.

Three jobs, in order:

1. **Load** the per-layer r-hat set produced by Phase 1 step 2, validating that
   each is a finite unit vector of the model's hidden size.

2. **Justify applying an edit across depth.** Proposal 3.3 applies one
   orthogonalization at many layers, which assumes the residual stream shares a
   consistent basis with depth. That assumption is testable:
   ``cos(r_hat^(l), r_hat^(l'))`` for every layer pair. Judged against the
   random-pair baseline, never against zero -- in 8192 dimensions a cosine of
   0.05 is enormous, and eyeballing the matrix would call it nothing. If the
   matrix is not stable across the intended range, the edit is restricted to the
   band that is. See PHASE2_HANDOFF.md section 1: this matrix was never computed
   in Phase 1 and is the prerequisite for a multi-layer edit.

3. **Resolve which block gets which direction.**

The off-by-one, stated explicitly because it is invisible and consequential
-------------------------------------------------------------------------
Phase 1 captured ``hidden_states[l]``. In HuggingFace, ``hidden_states[0]`` is
the embedding output and ``hidden_states[l]`` is the output of block ``l-1``
(0-indexed). So r-hat^(l) describes the residual stream *after* block ``l-1``,
and block ``b`` writes into residual position ``b+1``.

Under the default ``write_target`` convention, block ``b`` is therefore
orthogonalized against the direction measured at the position it writes into,
``b+1``, snapped to the nearest layer Phase 1 actually measured. The
``block_index`` convention instead reads the band as literal block indices. Both
are defensible; what is not defensible is leaving the choice implicit, so the
resolved block -> layer table is written into the manifest.

Directions are per-block rather than a single pooled r-hat (see the Phase 2
planning decisions). ``PerBlockDirections`` is the only thing ``surgery`` needs,
which is also the seam where the section 5 control conditions -- a random
direction, or a band that failed step 3a -- would plug in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from . import REPO  # noqa: F401  -- side effect: puts evalaware on sys.path

# isort: split
from evalaware.extraction import cross_layer_cosine_matrix, stable_band
from evalaware.validation import (
    assert_finite,
    assert_shape,
    assert_unit_norm,
    check,
    cosine,
    random_pair_cosine_baseline,
)

__all__ = [
    "DirectionSet",
    "BandReport",
    "BlockAssignment",
    "PerBlockDirections",
    "LayerConvention",
    "DEFAULT_DIRECTIONS_DIR",
    "HANDOFF_BAND",
    "load_direction_set",
    "analyze_cross_layer",
    "resolve_band",
    "assign_blocks",
    "assignment_consistency",
]

LayerConvention = Literal["write_target", "block_index"]

DEFAULT_DIRECTIONS_DIR = REPO / "outputs" / "directions"

# PHASE2_HANDOFF.md section 1. Step 3a found the direction generalizes across
# prompt templates and transfers between the two cue regimes only in this band.
# Layer 20 *inverts* (the two cues drive it in opposite directions) and layer 24
# has the highest in-sample effect size in the entire sweep with zero
# cross-regime transfer -- picking an edit layer by in-sample strength picks 24
# and orthogonalizes against one cue's surface form.
HANDOFF_BAND: tuple[int, int] = (36, 76)

_FILENAME_RE = re.compile(r"^r_hat_L(\d+)_(?P<pooling>.+)\.npz$")


@dataclass(frozen=True)
class DirectionSet:
    """The per-layer r-hat family, keyed by the layer it was measured at."""

    vectors: dict[int, np.ndarray]
    pooling: str
    key: str
    source_dir: Path

    @property
    def layers(self) -> list[int]:
        return sorted(self.vectors)

    @property
    def dim(self) -> int:
        return int(next(iter(self.vectors.values())).shape[0])

    def nearest_layer(self, target: int, within: tuple[int, int] | None = None) -> int:
        """Measured layer closest to ``target``, optionally restricted to a band.

        Ties break toward the lower layer so the mapping is deterministic.
        """
        candidates = self.layers
        if within is not None:
            lo, hi = within
            candidates = [ell for ell in candidates if lo <= ell <= hi]
            check(
                bool(candidates),
                f"nearest_layer: no measured layer inside band {within}. Measured "
                f"layers are {self.layers}.",
            )
        return min(candidates, key=lambda ell: (abs(ell - target), ell))


@dataclass
class BandReport:
    """Cross-layer stability of r-hat, and the band the edit will be applied to."""

    layers: list[int]
    cos_matrix: np.ndarray
    threshold: float
    baseline: dict[str, float]
    stability_multiplier: float
    direction_mode: str
    stable_band: tuple[int, int] | None
    handoff_band: tuple[int, int]
    chosen_band: tuple[int, int]
    band_criterion: str
    band_criterion_value: float
    band_min_cosine: float
    handoff_band_passes: bool
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "layers": self.layers,
            "threshold": round(self.threshold, 6),
            "random_pair_abs_p99": round(self.baseline["abs_p99"], 6),
            "stability_multiplier": self.stability_multiplier,
            "direction_mode": self.direction_mode,
            "stable_band": list(self.stable_band) if self.stable_band else None,
            "handoff_band": list(self.handoff_band),
            "chosen_band": list(self.chosen_band),
            "band_criterion": self.band_criterion,
            "band_criterion_value": round(self.band_criterion_value, 6),
            "band_min_pairwise_cosine": round(self.band_min_cosine, 6),
            "handoff_band_passes": self.handoff_band_passes,
            "cos_matrix": [
                [round(float(c), 6) for c in row] for row in self.cos_matrix
            ],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class BlockAssignment:
    """Which measured direction a transformer block is orthogonalized against."""

    block: int
    residual_index: int
    source_layer: int
    convention: str

    def summary(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "residual_index": self.residual_index,
            "source_layer": self.source_layer,
            "convention": self.convention,
        }


@dataclass
class PerBlockDirections:
    """The edit plan: block index -> unit direction, plus how it was derived.

    ``surgery`` consumes only ``for_block`` and ``blocks``; everything else is
    provenance for the manifest.
    """

    assignments: list[BlockAssignment]
    vectors: dict[int, np.ndarray]          # block -> (d,) float64 unit vector
    band: tuple[int, int]
    convention: str
    pooling: str
    key: str

    @property
    def blocks(self) -> list[int]:
        return sorted(self.vectors)

    def for_block(self, block: int) -> np.ndarray:
        check(
            block in self.vectors,
            f"no direction assigned to block {block}; assigned blocks are "
            f"{self.blocks[0]}..{self.blocks[-1]}",
        )
        return self.vectors[block]

    def summary(self) -> dict[str, Any]:
        return {
            "band": list(self.band),
            "convention": self.convention,
            "pooling": self.pooling,
            "direction_key": self.key,
            "n_blocks": len(self.vectors),
            "block_range": [self.blocks[0], self.blocks[-1]],
            "assignments": [a.summary() for a in self.assignments],
        }


def load_direction_set(
    directory: str | Path = DEFAULT_DIRECTIONS_DIR,
    pooling: str = "last_prompt_token",
    key: str = "r_hat",
) -> DirectionSet:
    """Load every ``r_hat_L{layer}_{pooling}.npz`` in ``directory``.

    ``key`` selects which vector to read. Phase 1 also stored the per-regime
    variants ``r_hat_explicit`` / ``r_hat_woodlabs`` and the ``v1_svd_check``
    alongside the primary ``r_hat``.
    """
    directory = Path(directory)
    check(directory.is_dir(), f"directions directory not found: {directory}")

    vectors: dict[int, np.ndarray] = {}
    for path in sorted(directory.glob("r_hat_L*.npz")):
        m = _FILENAME_RE.match(path.name)
        if m is None or m.group("pooling") != pooling:
            continue
        layer = int(m.group(1))
        with np.load(path) as data:
            check(
                key in data.files,
                f"{path.name}: no array {key!r} (has {sorted(data.files)})",
            )
            vec = np.asarray(data[key], dtype=np.float64)
        assert_shape(vec, (None,), f"{key}[L{layer}]")
        assert_finite(vec, f"{key}[L{layer}]")
        assert_unit_norm(vec, f"{key}[L{layer}]")
        check(layer not in vectors, f"duplicate direction for layer {layer}")
        vectors[layer] = vec

    check(
        len(vectors) >= 2,
        f"found {len(vectors)} direction file(s) for pooling {pooling!r} in "
        f"{directory}. Expected the Phase 1 sweep (one per captured layer). "
        f"Check the pooling suffix -- Phase 1 only ever captured "
        f"'last_prompt_token' (PHASE2_HANDOFF.md section 6).",
    )
    dims = {int(v.shape[0]) for v in vectors.values()}
    check(len(dims) == 1, f"directions have inconsistent dimensions: {sorted(dims)}")

    return DirectionSet(vectors=vectors, pooling=pooling, key=key, source_dir=directory)


def _longest_local_run(
    mat: np.ndarray, layers: list[int], threshold: float, min_width: int
) -> tuple[int, int] | None:
    """Longest run of consecutive measured layers whose *neighbours* all agree.

    The local analogue of ``stable_band``, which requires every pair in the block
    to clear the threshold. See ``analyze_cross_layer`` for when each applies.
    """
    best: tuple[int, int] | None = None
    best_width = min_width - 1
    start = 0
    for i in range(len(layers)):
        if i > 0 and mat[i - 1, i] < threshold:
            start = i
        width = i - start + 1
        if width > best_width:
            best_width = width
            best = (layers[start], layers[i])
    return best


def analyze_cross_layer(
    dset: DirectionSet,
    stability_multiplier: float = 20.0,
    min_width: int = 3,
    handoff_band: tuple[int, int] = HANDOFF_BAND,
    direction_mode: Literal["per_block", "shared"] = "per_block",
) -> BandReport:
    """Build ``cos(r_hat^(l), r_hat^(l'))`` and decide which band may be edited.

    The threshold is ``stability_multiplier`` times the 99th percentile of |cos|
    between random unit vectors in the same dimension. Phase 1's config uses the
    same convention (``Config.stability_threshold_multiplier``), so both stages
    judge cosines on one scale.

    **What has to hold depends on how the direction is applied**, and conflating
    the two is the easy mistake here.

    ``shared``
        One r-hat is applied at every depth. Then the *whole band* must share a
        basis: every pair of layers in it must clear the threshold. This is
        ``stable_band``, and it is the literal reading of proposal 3.3.

    ``per_block``
        Each block is orthogonalized against its own nearest measured direction.
        Nothing is being transported across the full band, so band-wide agreement
        is not required. What must hold is *local*: a block's direction is snapped
        to a measured layer up to two positions away, so adjacent measured layers
        must agree well enough that the snap is harmless. Requiring the stricter
        band-wide criterion here would discard layers for failing a test the edit
        never relies on.

    The band-wide minimum is reported either way, because a low value is a real
    finding about the model -- it says the residual basis rotates with depth --
    even when it does not gate this particular edit.
    """
    check(
        direction_mode in ("per_block", "shared"),
        f"unknown direction_mode {direction_mode!r}",
    )
    mat, layers = cross_layer_cosine_matrix(dset.vectors)
    baseline = random_pair_cosine_baseline(dset.dim)
    threshold = stability_multiplier * baseline["abs_p99"]

    index = {ell: i for i, ell in enumerate(layers)}
    notes: list[str] = []

    def _members(band: tuple[int, int]) -> list[int]:
        members = [ell for ell in layers if band[0] <= ell <= band[1]]
        check(
            len(members) >= 2,
            f"band {band} contains {len(members)} measured layer(s); need at "
            f"least 2 to assess stability.",
        )
        return members

    def _pairwise_min(band: tuple[int, int]) -> float:
        idx = [index[ell] for ell in _members(band)]
        return float(mat[np.ix_(idx, idx)].min())

    def _adjacent_min(band: tuple[int, int]) -> float:
        idx = [index[ell] for ell in _members(band)]
        return float(min(mat[a, b] for a, b in zip(idx, idx[1:])))

    criterion = _adjacent_min if direction_mode == "per_block" else _pairwise_min
    if direction_mode == "per_block":
        found = _longest_local_run(mat, layers, threshold, min_width)
        criterion_name = "min adjacent-layer cos"
    else:
        found = stable_band(mat, layers, threshold, min_width=min_width)
        criterion_name = "min pairwise cos"

    handoff_score = criterion(handoff_band)
    handoff_passes = handoff_score >= threshold

    if handoff_passes:
        chosen = handoff_band
        notes.append(
            f"[{direction_mode}] handoff band {handoff_band} is stable: "
            f"{criterion_name} {handoff_score:.4f} >= threshold {threshold:.4f}"
        )
    elif found is not None:
        # Restrict to the stable band, per proposal 3.3, but never widen beyond
        # the handoff band -- outside it the direction failed step 3a's
        # cross-regime transfer check regardless of how self-consistent it looks.
        lo = max(found[0], handoff_band[0])
        hi = min(found[1], handoff_band[1])
        check(
            lo < hi,
            f"the empirically stable band {found} does not overlap the step-3a "
            f"band {handoff_band}. There is no defensible band to edit: the "
            f"layers whose directions agree with each other are exactly the "
            f"layers whose directions failed cross-regime transfer. Stop and "
            f"re-examine Phase 1 rather than editing anyway.",
        )
        chosen = (lo, hi)
        notes.append(
            f"[{direction_mode}] handoff band {handoff_band} FAILED "
            f"({criterion_name} {handoff_score:.4f} < {threshold:.4f}); "
            f"restricted to {chosen} (intersection with stable band {found})"
        )
    else:
        raise AssertionError(
            f"no run of {min_width}+ consecutive measured layers is stable above "
            f"{threshold:.4f} (= {stability_multiplier} x random-pair |cos| p99 "
            f"{baseline['abs_p99']:.4f}) under the {direction_mode!r} criterion. "
            f"The residual stream does not share a basis across depth at this "
            f"scale. Handoff band scored {handoff_score:.4f}."
        )

    pairwise_min = _pairwise_min(chosen)
    if pairwise_min < threshold:
        # Not a failure under per_block, but it is the substantive finding about
        # the model and must not be buried: one pooled direction would be badly
        # misaligned at the ends of this band.
        notes.append(
            f"NOTE: the basis rotates across {chosen} -- min pairwise cos "
            f"{pairwise_min:.4f} is below {threshold:.4f} even though adjacent "
            f"layers agree. A single shared r-hat would be misaligned at the "
            f"band edges; per-block directions are what make this band editable."
        )

    # PHASE2_HANDOFF.md section 3: block structure in this matrix has an innocent
    # explanation to rule out first. The base model is a NAS architecture whose
    # blocks are heterogeneous, and a long contiguous run of them has no
    # attention at all.
    notes.append(
        "if the matrix is block-structured rather than smoothly banded, check "
        "the architecture before concluding the basis rotates: blocks 42-51 and "
        "53-70 have attention.no_op=true, so most of this band is MLP-only depth"
    )

    return BandReport(
        layers=layers,
        cos_matrix=mat,
        threshold=threshold,
        baseline=baseline,
        stability_multiplier=stability_multiplier,
        direction_mode=direction_mode,
        stable_band=found,
        handoff_band=handoff_band,
        chosen_band=chosen,
        band_criterion=criterion_name,
        band_criterion_value=criterion(chosen),
        band_min_cosine=pairwise_min,
        handoff_band_passes=handoff_passes,
        notes=notes,
    )


def resolve_band(spec: str, report: BandReport) -> tuple[int, int]:
    """Turn a ``--band`` argument into an inclusive layer range.

    ``"auto"`` takes the band the stability analysis chose. Anything else must be
    ``"LO-HI"`` and is used verbatim, so a deliberate override is possible, but
    it is checked against the analysis and refuses to silently edit an unstable
    range.
    """
    if spec == "auto":
        return report.chosen_band

    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", spec)
    check(m is not None, f"--band must be 'auto' or 'LO-HI', got {spec!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    check(lo < hi, f"--band {spec!r}: lower bound must be below upper bound")

    members = [ell for ell in report.layers if lo <= ell <= hi]
    check(
        len(members) >= 2,
        f"--band {spec!r} contains {len(members)} measured layer(s) "
        f"({report.layers}); need at least 2.",
    )
    index = {ell: i for i, ell in enumerate(report.layers)}
    idx = [index[ell] for ell in members]
    sub = report.cos_matrix[np.ix_(idx, idx)]
    # Judge the override by the same criterion the analysis used, so --band and
    # --direction-mode cannot disagree about what "stable" means.
    if report.direction_mode == "per_block":
        score = float(min(sub[i, i + 1] for i in range(len(idx) - 1)))
    else:
        score = float(sub.min())
    check(
        score >= report.threshold,
        f"--band {spec!r} has {report.band_criterion} {score:.4f}, below the "
        f"stability threshold {report.threshold:.4f}. Applying a direction at "
        f"depths whose bases disagree is exactly what proposal 3.3's cross-layer "
        f"check exists to prevent. Use --band auto, or widen the threshold "
        f"deliberately via stability_threshold_multiplier.",
    )
    return (lo, hi)


def assign_blocks(
    dset: DirectionSet,
    band: tuple[int, int],
    n_blocks: int,
    convention: LayerConvention = "write_target",
) -> PerBlockDirections:
    """Map each transformer block in ``band`` to the direction it is edited against.

    See the module docstring for why ``write_target`` shifts by one.
    """
    check(
        convention in ("write_target", "block_index"),
        f"unknown layer convention {convention!r}",
    )
    lo, hi = band
    check(0 <= lo < hi, f"band must be an increasing non-negative pair, got {band}")

    shift = 1 if convention == "write_target" else 0
    blocks = [b for b in range(n_blocks) if lo <= b + shift <= hi]
    check(
        bool(blocks),
        f"band {band} under convention {convention!r} selects no blocks of the "
        f"{n_blocks}-block model.",
    )

    assignments: list[BlockAssignment] = []
    vectors: dict[int, np.ndarray] = {}
    for b in blocks:
        residual_index = b + shift
        source = dset.nearest_layer(residual_index, within=band)
        assignments.append(
            BlockAssignment(
                block=b,
                residual_index=residual_index,
                source_layer=source,
                convention=convention,
            )
        )
        vec = dset.vectors[source]
        assert_unit_norm(vec, f"direction for block {b} (from L{source})")
        vectors[b] = vec

    return PerBlockDirections(
        assignments=assignments,
        vectors=vectors,
        band=band,
        convention=convention,
        pooling=dset.pooling,
        key=dset.key,
    )


def assignment_consistency(plan: PerBlockDirections) -> dict[str, float]:
    """How much the per-block directions disagree with each other.

    With per-block directions the shared-basis assumption is weakened but not
    removed -- adjacent blocks still write into a stream that later blocks read.
    Reporting the spread makes the residual assumption visible.
    """
    unique = {a.source_layer: plan.vectors[a.block] for a in plan.assignments}
    layers = sorted(unique)
    cosines = [
        cosine(unique[a], unique[b])
        for i, a in enumerate(layers)
        for b in layers[i + 1:]
    ]
    check(bool(cosines), "assignment_consistency: fewer than 2 distinct directions")
    return {
        "n_distinct_directions": float(len(layers)),
        "min_pairwise_cos": float(np.min(cosines)),
        "mean_pairwise_cos": float(np.mean(cosines)),
        "max_pairwise_cos": float(np.max(cosines)),
    }
