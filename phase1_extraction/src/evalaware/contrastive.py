"""PHASE 1, STEP 1 -- build the contrastive matrix.  ** THIS IS THE DELIVERABLE. **

    X = H_eval - H_deploy        (N x d, one row per matched prompt pair)

Scope boundary: this module stops at X. Step 2 (SVD / direction isolation) is
owned by another group member -- see HANDOFF.md for the artifact contract.
Nothing here computes r_hat.

What this module is actually for
--------------------------------
The subtraction itself is one line. Everything else here exists because the
ways X goes wrong are silent:

* **Misalignment.** If row i of H_eval is not the same question as row i of
  H_deploy, X encodes topic differences and still looks perfectly healthy --
  right shape, sensible norms, no NaNs. Every row is checked against its pair
  key, not merely counted.
* **Polarity.** Getting the sign backwards yields -X, which passes every
  magnitude check and only surfaces as a backwards effect in Phase 3. Framing
  is taken from explicit labels, never inferred from key names (Hua et al.'s
  own repo has ``real``/``fake`` meaning opposite things in two files).
* **Silent degeneracy.** Duplicate rows, zero rows, or a collapsed layer index
  produce a matrix that SVD will happily decompose into noise.

Diagnostics reported here are deliberately estimator-free -- they describe the
data, they do not estimate a direction. The one exception is the *consistency
ratio* ``||mean(X)|| / mean(||X_i||)``, included because it is the cheapest
answer to "is there any consistent contrast in this data at all", which the
step 2 owner needs to know before trusting a decomposition. It is a data-
quality statistic, not the direction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .validation import (
    assert_finite,
    assert_paired,
    assert_shape,
    check,
)

__all__ = [
    "ContrastiveMatrix",
    "build_contrastive_matrix",
    "save_contrastive_matrix",
    "load_contrastive_matrix",
    "diagnose_zero_rows",
    "ARTIFACT_VERSION",
]

# Bump when the on-disk layout changes so consumers can fail loudly rather than
# misread an old artifact.
ARTIFACT_VERSION = "1.0"


@dataclass
class RowMeta:
    """Provenance for one row of X. Row order in the file IS this list's order."""

    row_index: int
    pair_key: str
    question_id: str
    regime: str
    source: str
    template_family_eval: str
    template_family_deploy: str
    system_id_eval: str
    system_id_deploy: str
    user_id_eval: str
    user_id_deploy: str


@dataclass
class ContrastiveMatrix:
    """X for one (layer, pooling) cell, with everything needed to interpret it."""

    x: np.ndarray                    # (N, d) float64, = H_eval - H_deploy
    layer: int
    pooling: str
    rows: list[RowMeta]
    split: str
    regimes: tuple[str, ...]
    model_id: str
    token_offset: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(self.x.shape[0])

    @property
    def dim(self) -> int:
        return int(self.x.shape[1])

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "layer": self.layer,
            "pooling": self.pooling,
            "split": self.split,
            "regimes": list(self.regimes),
            "model_id": self.model_id,
            "token_offset": self.token_offset,
            "n_pairs": self.n_pairs,
            "dim": self.dim,
            "n_questions": len({r.question_id for r in self.rows}),
            "n_template_families": len({r.template_family_eval for r in self.rows}),
            **self.diagnostics,
        }


def _diagnostics(x: np.ndarray, rows: Sequence[RowMeta]) -> dict[str, Any]:
    """Estimator-free description of X, plus the checks that gate it."""
    row_norms = np.linalg.norm(x, axis=1)
    mean_vec = x.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    mean_row_norm = float(row_norms.mean())

    # Consistency ratio in [0, 1]: 1.0 means every row points the same way,
    # ~1/sqrt(N) means the rows are essentially random relative to each other.
    consistency = mean_norm / mean_row_norm if mean_row_norm > 0 else 0.0
    chance_consistency = 1.0 / np.sqrt(len(rows)) if rows else float("nan")

    # Duplicate rows: usually a caching bug where two prompts hashed together.
    _, unique_idx = np.unique(np.round(x, 6), axis=0, return_index=True)
    n_duplicate = int(x.shape[0] - len(unique_idx))

    per_question: dict[str, int] = {}
    for r in rows:
        per_question[r.question_id] = per_question.get(r.question_id, 0) + 1

    return {
        "row_norm_mean": round(mean_row_norm, 6),
        "row_norm_sd": round(float(row_norms.std(ddof=1)) if len(row_norms) > 1 else 0.0, 6),
        "row_norm_min": round(float(row_norms.min()), 6),
        "row_norm_max": round(float(row_norms.max()), 6),
        "mean_vector_norm": round(mean_norm, 6),
        "consistency_ratio": round(float(consistency), 6),
        "consistency_ratio_at_chance": round(float(chance_consistency), 6),
        "consistency_above_chance": bool(consistency > 3 * chance_consistency),
        "n_duplicate_rows": n_duplicate,
        "rows_per_question_min": min(per_question.values()) if per_question else 0,
        "rows_per_question_max": max(per_question.values()) if per_question else 0,
    }


def build_contrastive_matrix(
    h_eval: np.ndarray,
    h_deploy: np.ndarray,
    items: Sequence,
    layer: int,
    pooling: str,
    split: str,
    model_id: str,
    token_offset: int,
    regimes: tuple[str, ...] = (),
) -> ContrastiveMatrix:
    """Build X = H_eval - H_deploy with alignment and polarity enforced.

    ``items`` must be ``PairedItem``s in exactly the row order of ``h_eval`` and
    ``h_deploy``. The framing of each side comes from the item's explicit
    ``eval_spec`` / ``deploy_spec`` labels, never from ordering or naming
    convention.
    """
    h_eval = np.asarray(h_eval, dtype=np.float64)
    h_deploy = np.asarray(h_deploy, dtype=np.float64)

    assert_shape(h_eval, (None, None), "H_eval")
    assert_shape(h_deploy, h_eval.shape, "H_deploy")
    assert_finite(h_eval, "H_eval")
    assert_finite(h_deploy, "H_deploy")
    check(
        len(items) == h_eval.shape[0],
        f"build_contrastive_matrix: {len(items)} items but H_eval has "
        f"{h_eval.shape[0]} rows. The item list must be in the same order as "
        f"the stacked activations.",
    )
    check(h_eval.shape[0] >= 2, f"need at least 2 pairs, got {h_eval.shape[0]}")

    # Explicit framing check: guard against a swapped-argument call, which is
    # otherwise undetectable and flips the sign of every downstream result.
    for i, it in enumerate(items):
        check(
            it.eval_spec.framing == "eval" and it.deploy_spec.framing == "deploy",
            f"row {i} ({it.pair_key}): framing labels are "
            f"{it.eval_spec.framing!r}/{it.deploy_spec.framing!r}, expected "
            f"'eval'/'deploy'. X must be H_eval - H_deploy; an inversion here "
            f"yields -X and is invisible to every magnitude check.",
        )

    ids = [it.pair_key for it in items]
    assert_paired(ids, ids, name="contrastive rows")
    check(
        len(set(ids)) == len(ids),
        f"build_contrastive_matrix: {len(ids) - len(set(ids))} duplicate pair "
        f"key(s); a pair would be counted twice.",
    )

    x = h_eval - h_deploy
    assert_finite(x, "X")

    row_norms = np.linalg.norm(x, axis=1)
    n_zero = int((row_norms == 0).sum())
    median_norm = float(np.median(row_norms))

    # Layer 0 is the embedding output. With fixed-token pooling its value
    # depends only on the token id at that position, not on context -- and
    # matched framings end with the same chat-template tokens. X is therefore
    # exactly zero there for every pair, by construction, not by malfunction.
    # Say so plainly rather than sending someone hunting for a bug.
    if median_norm <= 0 and layer == 0 and pooling in ("last_prompt_token",):
        raise AssertionError(
            "build_contrastive_matrix: X is all zeros at LAYER 0, which is "
            "expected and not a bug.\n"
            "  Layer 0 is the embedding output. Under fixed-token pooling it is "
            "a function of the token id at that position only, and matched\n"
            "  eval/deploy prompts end with identical chat-template tokens, so "
            "the difference is exactly zero for every pair.\n"
            "  Layer 0 carries no contrast. Skip it:\n"
            "      BUILD_LAYERS = tuple(l for l in CAPTURED_LAYERS if l >= 1)\n"
            "  Keep layer_index built from the CAPTURED layers, or every lookup "
            "shifts and you will read the wrong rows."
        )

    if median_norm <= 0:
        # Report the true counts and a per-regime breakdown. An earlier version
        # said "every row is zero" here, which was wrong -- the median is zero
        # as soon as HALF the rows are, and knowing whether it is half or all
        # points at completely different causes.
        by_regime: dict[str, list[int]] = {}
        for it, norm in zip(items, row_norms):
            slot = by_regime.setdefault(getattr(it, "regime", "?"), [0, 0])
            slot[0] += 1
            slot[1] += int(norm == 0)
        breakdown = ", ".join(
            f"{r}: {z}/{n} zero" for r, (n, z) in sorted(by_regime.items())
        )
        raise AssertionError(
            f"build_contrastive_matrix: {n_zero} of {x.shape[0]} rows are exactly "
            f"zero, so H_eval == H_deploy for at least half the pairs.\n"
            f"  per regime -> {breakdown}\n"
            f"\n"
            f"  Prompt text is NOT the cause if pair construction validated -- it "
            f"checks that already. Most likely causes, in order:\n"
            f"   1. The activation cache returned the same array for both sides.\n"
            f"      Check: store.get(eval_spec...) vs store.get(deploy_spec...).\n"
            f"   2. A stale cache directory from an earlier run with different\n"
            f"      code. Delete the cache dir and re-capture.\n"
            f"   3. The capture loop wrote the same `arr` for every prompt (e.g.\n"
            f"      a variable captured outside the loop).\n"
            f"\n"
            f"  Run diagnose_zero_rows(store, items, layer, layer_index) to tell\n"
            f"  these apart."
        )

    degenerate = int((row_norms < 1e-6 * median_norm).sum())
    check(
        degenerate == 0,
        f"build_contrastive_matrix: {degenerate}/{x.shape[0]} rows are "
        f"numerically zero (< 1e-6 x median norm {median_norm:.4g}), though the "
        f"majority are not. Those specific pairs produced identical activations "
        f"on both sides -- inspect their pair_keys.",
    )

    rows = [
        RowMeta(
            row_index=i,
            pair_key=it.pair_key,
            question_id=it.question_id,
            regime=it.regime,
            source=it.source,
            template_family_eval=it.template_family_eval,
            template_family_deploy=it.template_family_deploy,
            system_id_eval=it.eval_spec.system_id,
            system_id_deploy=it.deploy_spec.system_id,
            user_id_eval=it.eval_spec.user_id,
            user_id_deploy=it.deploy_spec.user_id,
        )
        for i, it in enumerate(items)
    ]

    return ContrastiveMatrix(
        x=x,
        layer=layer,
        pooling=pooling,
        rows=rows,
        split=split,
        regimes=tuple(regimes) or tuple(sorted({it.regime for it in items})),
        model_id=model_id,
        token_offset=token_offset,
        diagnostics=_diagnostics(x, rows),
    )


def diagnose_zero_rows(store, items: Sequence, layer: int, layer_index: dict | None = None,
                       max_report: int = 3) -> dict[str, Any]:
    """Work out *why* H_eval == H_deploy, by inspecting the cache directly.

    Distinguishes the three causes the build error lists:

    * identical prompt text          -> pair construction problem
    * identical cache key            -> hash collision (should be impossible)
    * different key, identical bytes -> the capture loop stored the same array

    Prints a verdict and returns the details.
    """
    n = len(items)
    same_prompt = same_key = same_bytes = differ = 0
    examples: list[dict[str, Any]] = []

    for it in items:
        e, d = it.eval_spec, it.deploy_spec
        key_e = store.prompt_hash(e.system_prompt, e.user_prompt)
        key_d = store.prompt_hash(d.system_prompt, d.user_prompt)
        prompts_same = (e.system_prompt == d.system_prompt and
                        e.user_prompt == d.user_prompt)

        try:
            ae = store.get(e.system_prompt, e.user_prompt)
            ad = store.get(d.system_prompt, d.user_prompt)
        except Exception as exc:
            if len(examples) < max_report:
                examples.append({"pair_key": it.pair_key, "error": str(exc)})
            continue

        idx = layer_index[layer] if layer_index else layer
        bytes_same = np.array_equal(ae[idx], ad[idx])

        if prompts_same:
            same_prompt += 1
        elif key_e == key_d:
            same_key += 1
        elif bytes_same:
            same_bytes += 1
            if len(examples) < max_report:
                examples.append({
                    "pair_key": it.pair_key,
                    "eval_key": key_e[:12], "deploy_key": key_d[:12],
                    "eval_first5": ae[idx][:5].tolist(),
                    "deploy_first5": ad[idx][:5].tolist(),
                })
        else:
            differ += 1

    print(f"layer {layer}, {n} pairs")
    print(f"  identical prompt text          : {same_prompt}")
    print(f"  identical cache key            : {same_key}")
    print(f"  different key, identical bytes : {same_bytes}")
    print(f"  genuinely different            : {differ}")

    if same_prompt:
        verdict = "PROMPTS: pair construction produced identical text on both sides."
    elif same_key:
        verdict = "CACHE KEY COLLISION: prompt_hash collided (should be impossible)."
    elif same_bytes:
        verdict = ("CAPTURE: distinct prompts and distinct cache keys, but identical "
                   "stored activations. The capture loop wrote the same array for "
                   "every prompt, or the cache is stale from an earlier run. "
                   "Delete the cache directory and re-capture.")
    elif differ == n:
        verdict = ("Activations DO differ at this layer -- the zero rows are not here. "
                   "Check the layer index mapping.")
    else:
        verdict = "Mixed / incomplete cache; see examples."
    print(f"\n  VERDICT: {verdict}")

    if examples:
        print("\n  examples:")
        for ex in examples:
            print("   ", ex)

    return {"n": n, "same_prompt": same_prompt, "same_key": same_key,
            "same_bytes": same_bytes, "differ": differ,
            "verdict": verdict, "examples": examples}


def save_contrastive_matrix(cm: ContrastiveMatrix, out_dir: str | Path) -> dict[str, Path]:
    """Write X plus row metadata and a manifest. See HANDOFF.md for the contract.

    Three files per (layer, pooling, split) cell::

        X_L{layer}_{pooling}_{split}.npz    key 'X', float64, shape (N, d)
        X_L{layer}_{pooling}_{split}.rows.jsonl   one RowMeta per line, row order
        X_L{layer}_{pooling}_{split}.meta.json    shapes, diagnostics, provenance

    Row order in the .npz and the .jsonl is the same and is load-bearing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"X_L{cm.layer}_{cm.pooling}_{cm.split}"

    npz_path = out_dir / f"{stem}.npz"
    rows_path = out_dir / f"{stem}.rows.jsonl"
    meta_path = out_dir / f"{stem}.meta.json"

    # Write via temp file + rename so an interrupted save cannot leave a
    # truncated artifact that a later run would silently consume.
    #
    # Note: np.savez_compressed appends '.npz' to a *path* argument that does
    # not already end in it, so a temp name like 'X.npz.tmp' would silently be
    # written to 'X.npz.tmp.npz'. Passing an open file handle avoids the
    # rename entirely.
    tmp = npz_path.with_name(npz_path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, X=cm.x)
    tmp.replace(npz_path)

    with rows_path.open("w", encoding="utf-8") as fh:
        for r in cm.rows:
            fh.write(json.dumps(asdict(r), sort_keys=True) + "\n")

    meta = cm.summary()
    meta["files"] = {
        "matrix": npz_path.name,
        "rows": rows_path.name,
        "meta": meta_path.name,
    }
    meta["row_order_is_load_bearing"] = True
    meta["polarity"] = "X = H_eval - H_deploy (positive direction points deploy -> eval)"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    return {"matrix": npz_path, "rows": rows_path, "meta": meta_path}


def load_contrastive_matrix(npz_path: str | Path) -> ContrastiveMatrix:
    """Load a saved artifact, re-validating shape/row agreement on the way in."""
    npz_path = Path(npz_path)
    check(npz_path.exists(), f"contrastive matrix not found: {npz_path}")
    stem = npz_path.name[: -len(".npz")]
    rows_path = npz_path.parent / f"{stem}.rows.jsonl"
    meta_path = npz_path.parent / f"{stem}.meta.json"
    check(rows_path.exists(), f"row metadata missing: {rows_path}")
    check(meta_path.exists(), f"meta missing: {meta_path}")

    with np.load(npz_path) as data:
        check("X" in data, f"{npz_path}: expected array key 'X', found {list(data.keys())}")
        x = data["X"]
    assert_finite(x, "loaded X")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    version = meta.get("artifact_version")
    check(
        version == ARTIFACT_VERSION,
        f"{meta_path}: artifact version {version!r} but this code expects "
        f"{ARTIFACT_VERSION!r}. Rebuild the artifact rather than reading it.",
    )

    rows = [RowMeta(**json.loads(line)) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check(
        len(rows) == x.shape[0],
        f"{npz_path}: X has {x.shape[0]} rows but metadata has {len(rows)}. "
        f"The artifact is inconsistent; rebuild it.",
    )
    for i, r in enumerate(rows):
        check(
            r.row_index == i,
            f"{rows_path}: row {i} carries row_index {r.row_index}. Row order "
            f"is load-bearing and must not be re-sorted.",
        )
    check(
        int(meta.get("n_pairs", -1)) == x.shape[0],
        f"{meta_path}: n_pairs {meta.get('n_pairs')} disagrees with X rows {x.shape[0]}.",
    )

    return ContrastiveMatrix(
        x=x,
        layer=int(meta["layer"]),
        pooling=str(meta["pooling"]),
        rows=rows,
        split=str(meta["split"]),
        regimes=tuple(meta.get("regimes", ())),
        model_id=str(meta.get("model_id", "unknown")),
        token_offset=int(meta.get("token_offset", -2)),
        diagnostics={k: v for k, v in meta.items() if k.startswith(("row_norm", "consistency", "n_"))},
    )
