"""PHASE 1, STEP 3a — out-of-distribution generalization.  ** DELIVERABLE. **

The question: does r̂ encode *evaluation awareness*, or did it memorize the
surface form of the templates it was estimated on?

Test: project held-out activations onto r̂ and ask whether that one scalar
separates eval-framed from deploy-framed prompts, on templates never seen during
extraction.

Four things this module insists on, each because the obvious version of the
check is misleading without it:

**A random-direction null.** "AUROC 0.78" means nothing without knowing what an
arbitrary direction scores on the same data. In high dimensions with correlated
activations, random directions do not always sit at 0.5. We measure the null
rather than assuming it.

**Question-clustered bootstrap.** Rows sharing a question are not independent —
the same coding problem appears under many templates. Resampling rows instead of
questions gives CIs that are far too narrow. Follows Hua et al. Appendix I.2.

**Paired statistics.** The data is paired, so the strongest test is whether the
eval member of each pair projects above *its own* deploy partner. Reported
alongside the unpaired AUROC.

**Cross-regime transfer.** The proposal requires r̂ to replicate under both the
explicit and Wood Labs cues before any surgery. A direction that separates only
the regime it came from is that cue's surface form, not awareness. Reported per
regime, never only pooled — pooling can hide a total failure on one cue.

Sign note: r̂ is only determined up to sign by SVD. An AUROC near 0 rather than
near 1 means a flipped sign, not absent signal, and is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..validation import assert_finite, assert_shape, assert_unit_norm, check

__all__ = ["OODResult", "RegimeResult", "run_ood_check", "auroc", "random_direction_null"]


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC via Mann-Whitney U, with correct tie handling (ties count 0.5)."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    check(pos.size > 0 and neg.size > 0, "auroc: both classes must be non-empty.")

    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)

    srt = combined[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1

    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _clustered_bootstrap(statistic, cluster_ids: Sequence[str], n_boot: int, seed: int):
    """Percentile CI, resampling whole clusters (questions) with replacement."""
    clusters = sorted(set(cluster_ids))
    idx: dict[str, list[int]] = {c: [] for c in clusters}
    for i, c in enumerate(cluster_ids):
        idx[c].append(i)

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(len(clusters), size=len(clusters), replace=True)
        rows: list[int] = []
        for p in picked:
            rows.extend(idx[clusters[p]])
        try:
            vals.append(statistic(np.asarray(rows, dtype=int)))
        except Exception:
            continue           # degenerate resample (e.g. one class empty)

    check(
        len(vals) >= max(20, n_boot // 10),
        f"clustered bootstrap: only {len(vals)}/{n_boot} resamples valid with "
        f"{len(clusters)} clusters. Too few questions for a stable CI.",
    )
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


@dataclass
class RegimeResult:
    """Separation within one cue regime."""

    regime: str
    n_pairs: int
    n_questions: int
    auroc: float
    auroc_ci: tuple[float, float]
    paired_win_rate: float
    paired_win_rate_ci: tuple[float, float]
    cohens_d: float

    @property
    def separates(self) -> bool:
        """Lower CI bound must clear chance on both statistics."""
        return self.auroc_ci[0] > 0.5 and self.paired_win_rate_ci[0] > 0.5

    def summary(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "n_pairs": self.n_pairs,
            "n_questions": self.n_questions,
            "auroc": round(self.auroc, 4),
            "auroc_ci95": [round(c, 4) for c in self.auroc_ci],
            "paired_win_rate": round(self.paired_win_rate, 4),
            "paired_win_rate_ci95": [round(c, 4) for c in self.paired_win_rate_ci],
            "cohens_d": round(self.cohens_d, 4),
            "separates": self.separates,
        }


@dataclass
class OODResult:
    layer: int
    pooling: str
    pooled: RegimeResult
    by_regime: dict[str, RegimeResult]
    null_auroc_mean: float
    null_auroc_p95: float
    n_null_directions: int
    sign_flipped: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def beats_null(self) -> bool:
        """Must beat the measured random-direction null, not merely 0.5."""
        return self.pooled.auroc_ci[0] > self.null_auroc_p95

    @property
    def transfers_across_regimes(self) -> bool:
        """Every regime present must separate. Pooling can mask one failing."""
        return len(self.by_regime) > 1 and all(r.separates for r in self.by_regime.values())

    @property
    def passes(self) -> bool:
        return self.pooled.separates and self.beats_null

    def summary(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "pooling": self.pooling,
            "pooled": self.pooled.summary(),
            "by_regime": {k: v.summary() for k, v in self.by_regime.items()},
            "null_auroc_mean": round(self.null_auroc_mean, 4),
            "null_auroc_p95": round(self.null_auroc_p95, 4),
            "n_null_directions": self.n_null_directions,
            "beats_null": self.beats_null,
            "transfers_across_regimes": self.transfers_across_regimes,
            "sign_flipped": self.sign_flipped,
            "passes": self.passes,
            "notes": self.notes,
        }


def random_direction_null(
    h_eval: np.ndarray,
    h_deploy: np.ndarray,
    n_directions: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """AUROC distribution for random unit directions on the same data.

    This is the bar r̂ has to clear. Assuming 0.5 is not safe: activations are
    correlated and the two classes differ in more ways than the framing, so an
    arbitrary direction can score above chance for uninteresting reasons.

    Returns (mean, 95th percentile).
    """
    h_eval = np.asarray(h_eval, dtype=np.float64)
    h_deploy = np.asarray(h_deploy, dtype=np.float64)
    d = h_eval.shape[1]
    rng = np.random.default_rng(seed)

    scores = np.empty(n_directions)
    for i in range(n_directions):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        scores[i] = auroc(h_eval @ v, h_deploy @ v)

    # Directions are sign-arbitrary, so fold: what matters is separation
    # magnitude, whichever way it points.
    folded = np.maximum(scores, 1.0 - scores)
    return float(folded.mean()), float(np.percentile(folded, 95))


def _one_regime(
    regime: str,
    proj_eval: np.ndarray,
    proj_deploy: np.ndarray,
    question_ids: Sequence[str],
    n_boot: int,
    seed: int,
) -> RegimeResult:
    diffs = proj_eval - proj_deploy
    point_auroc = auroc(proj_eval, proj_deploy)
    point_win = float((diffs > 0).mean() + 0.5 * (diffs == 0).mean())

    pooled_sd = np.sqrt((proj_eval.var(ddof=1) + proj_deploy.var(ddof=1)) / 2.0)
    d = float((proj_eval.mean() - proj_deploy.mean()) / pooled_sd) if pooled_sd > 0 else 0.0

    return RegimeResult(
        regime=regime,
        n_pairs=int(proj_eval.size),
        n_questions=len(set(question_ids)),
        auroc=point_auroc,
        auroc_ci=_clustered_bootstrap(
            lambda r: auroc(proj_eval[r], proj_deploy[r]), question_ids, n_boot, seed),
        paired_win_rate=point_win,
        paired_win_rate_ci=_clustered_bootstrap(
            lambda r: float((diffs[r] > 0).mean() + 0.5 * (diffs[r] == 0).mean()),
            question_ids, n_boot, seed + 1),
        cohens_d=d,
    )


def run_ood_check(
    r_hat: np.ndarray,
    h_eval: np.ndarray,
    h_deploy: np.ndarray,
    question_ids: Sequence[str],
    regimes: Sequence[str],
    layer: int,
    pooling: str = "last_prompt_token",
    n_boot: int = 2000,
    n_null: int = 200,
    seed: int = 0,
    auto_fix_sign: bool = True,
) -> OODResult:
    """Project held-out activations onto r̂ and test separation.

    ``h_eval`` / ``h_deploy`` must be row-aligned (row i is the same question
    under both framings) and drawn from the ``ood_holdout`` split.
    ``regimes[i]`` labels row i so transfer can be reported per cue.
    """
    r_hat = np.asarray(r_hat, dtype=np.float64).ravel()
    assert_unit_norm(r_hat, f"r_hat (OOD, layer {layer})")
    h_eval = np.asarray(h_eval, dtype=np.float64)
    h_deploy = np.asarray(h_deploy, dtype=np.float64)

    assert_shape(h_eval, (None, r_hat.size), "H_eval (holdout)")
    assert_shape(h_deploy, h_eval.shape, "H_deploy (holdout)")
    assert_finite(h_eval, "H_eval (holdout)")
    assert_finite(h_deploy, "H_deploy (holdout)")
    check(
        len(question_ids) == h_eval.shape[0] == len(regimes),
        f"run_ood_check: {len(question_ids)} question ids, {len(regimes)} regime "
        f"labels, {h_eval.shape[0]} rows -- all must match.",
    )

    notes: list[str] = []
    proj_eval = h_eval @ r_hat
    proj_deploy = h_deploy @ r_hat

    # Sign is not recoverable from the file; detect it empirically.
    sign_flipped = False
    if auroc(proj_eval, proj_deploy) < 0.5:
        sign_flipped = True
        notes.append(
            "AUROC below 0.5 before correction: r_hat points deployment-ward. "
            "SVD fixes v1 only up to sign. Separation magnitude is unaffected, "
            "but step 2 should apply `if v1 @ X.mean(0) < 0: v1 = -v1` at source "
            "-- Phase 2's edit direction depends on getting this right."
        )
        if auto_fix_sign:
            r_hat = -r_hat
            proj_eval, proj_deploy = -proj_eval, -proj_deploy

    pooled = _one_regime("all", proj_eval, proj_deploy, question_ids, n_boot, seed)

    by_regime: dict[str, RegimeResult] = {}
    for reg in sorted(set(regimes)):
        mask = np.array([r == reg for r in regimes])
        if mask.sum() < 4:
            notes.append(f"regime {reg!r} skipped: only {int(mask.sum())} pairs.")
            continue
        by_regime[reg] = _one_regime(
            reg, proj_eval[mask], proj_deploy[mask],
            [q for q, m in zip(question_ids, mask) if m], n_boot, seed + 2)

    null_mean, null_p95 = random_direction_null(h_eval, h_deploy, n_null, seed)

    if len(by_regime) > 1:
        failing = [k for k, v in by_regime.items() if not v.separates]
        if failing:
            notes.append(
                f"r_hat does NOT separate framings in regime(s) {failing}. The "
                f"proposal requires replication under both cues before surgery -- "
                f"a direction specific to one cue is that cue's surface form, not "
                f"evaluation awareness."
            )

    return OODResult(
        layer=layer,
        pooling=pooling,
        pooled=pooled,
        by_regime=by_regime,
        null_auroc_mean=null_mean,
        null_auroc_p95=null_p95,
        n_null_directions=n_null,
        sign_flipped=sign_flipped,
        notes=notes,
    )
