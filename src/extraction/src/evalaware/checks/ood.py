"""Step 3a: out-of-distribution generalization check.

The question this answers: does r_hat encode *evaluation awareness*, or did it
memorize the surface form of the templates it was estimated on?

Test: project held-out activations onto r_hat, then ask whether that single
scalar separates eval-framed from deploy-framed prompts built from templates
never seen during extraction. We report AUROC because it is threshold-free and
invariant to the arbitrary scale of the projection.

Two details that matter:

* **Paired AUROC.** Our data is paired, so we report both the standard
  unpaired AUROC and the paired win-rate (the fraction of pairs where the eval
  member projects higher than its own deploy member). The paired statistic is
  the more powerful test and is what the design actually supports.
* **Clustered bootstrap.** Rows sharing a question are not independent. CIs
  resample *questions*, not rows, following Hua et al. Appendix I.2. Naive
  row-level bootstrap would give CIs that are far too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..validation import assert_finite, assert_shape, assert_unit_norm, check

__all__ = ["OODResult", "run_ood_check", "auroc"]


@dataclass
class OODResult:
    layer: int
    pooling: str
    n_pairs: int
    n_questions: int
    auroc: float
    auroc_ci: tuple[float, float]
    paired_win_rate: float
    paired_win_rate_ci: tuple[float, float]
    cohens_d: float
    mean_proj_eval: float
    mean_proj_deploy: float

    @property
    def passes(self) -> bool:
        """Gate: the lower CI bound must clear chance on both statistics.

        Using the CI bound rather than the point estimate means a direction
        that looks good but is not measurably better than chance does not pass.
        """
        return self.auroc_ci[0] > 0.5 and self.paired_win_rate_ci[0] > 0.5

    def summary(self) -> dict:
        return {
            "layer": self.layer,
            "pooling": self.pooling,
            "n_pairs": self.n_pairs,
            "n_questions": self.n_questions,
            "auroc": round(self.auroc, 4),
            "auroc_ci95": [round(c, 4) for c in self.auroc_ci],
            "paired_win_rate": round(self.paired_win_rate, 4),
            "paired_win_rate_ci95": [round(c, 4) for c in self.paired_win_rate_ci],
            "cohens_d": round(self.cohens_d, 4),
            "passes": self.passes,
        }


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic, with correct tie handling.

    Ties contribute 0.5, which matters when projections are quantized or when
    many rows are identical.
    """
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    check(pos.size > 0 and neg.size > 0, "auroc: both classes must be non-empty.")
    combined = np.concatenate([pos, neg])
    # Average ranks so tied values share a rank.
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    sorted_vals = combined[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _clustered_bootstrap(
    statistic,
    cluster_ids: Sequence[str],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI, resampling whole clusters (questions) with replacement."""
    clusters = sorted(set(cluster_ids))
    idx_by_cluster: dict[str, list[int]] = {c: [] for c in clusters}
    for i, c in enumerate(cluster_ids):
        idx_by_cluster[c].append(i)

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(len(clusters), size=len(clusters), replace=True)
        rows: list[int] = []
        for p in picked:
            rows.extend(idx_by_cluster[clusters[p]])
        try:
            vals.append(statistic(np.asarray(rows, dtype=int)))
        except Exception:
            # A resample can be degenerate (e.g. one class empty). Skip it
            # rather than crash; the count is checked below.
            continue
    check(
        len(vals) >= max(20, n_boot // 10),
        f"clustered bootstrap: only {len(vals)}/{n_boot} resamples were valid. "
        f"Too few clusters ({len(clusters)}) for a stable CI -- add questions.",
    )
    return (
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def run_ood_check(
    r_hat: np.ndarray,
    h_eval_ood: np.ndarray,
    h_deploy_ood: np.ndarray,
    question_ids: Sequence[str],
    layer: int,
    pooling: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> OODResult:
    """Project held-out activations onto r_hat and test separation.

    ``h_eval_ood`` / ``h_deploy_ood`` must be row-aligned (row i is the same
    question under both framings), and must come from templates in the
    ``ood_holdout`` split.
    """
    r_hat = np.asarray(r_hat, dtype=np.float64).ravel()
    assert_unit_norm(r_hat, "r_hat (OOD check)")
    h_eval_ood = np.asarray(h_eval_ood, dtype=np.float64)
    h_deploy_ood = np.asarray(h_deploy_ood, dtype=np.float64)
    assert_shape(h_eval_ood, (None, r_hat.size), "H_eval_ood")
    assert_shape(h_deploy_ood, h_eval_ood.shape, "H_deploy_ood")
    assert_finite(h_eval_ood, "H_eval_ood")
    assert_finite(h_deploy_ood, "H_deploy_ood")
    check(
        len(question_ids) == h_eval_ood.shape[0],
        f"run_ood_check: {len(question_ids)} question ids for "
        f"{h_eval_ood.shape[0]} rows.",
    )

    proj_eval = h_eval_ood @ r_hat
    proj_deploy = h_deploy_ood @ r_hat

    point_auroc = auroc(proj_eval, proj_deploy)
    diffs = proj_eval - proj_deploy
    # Ties count as half a win, consistent with the AUROC convention.
    point_win = float((diffs > 0).mean() + 0.5 * (diffs == 0).mean())

    pooled_sd = np.sqrt((proj_eval.var(ddof=1) + proj_deploy.var(ddof=1)) / 2.0)
    d = float((proj_eval.mean() - proj_deploy.mean()) / pooled_sd) if pooled_sd > 0 else 0.0

    auroc_ci = _clustered_bootstrap(
        lambda rows: auroc(proj_eval[rows], proj_deploy[rows]),
        question_ids, n_boot, seed,
    )
    win_ci = _clustered_bootstrap(
        lambda rows: float((diffs[rows] > 0).mean() + 0.5 * (diffs[rows] == 0).mean()),
        question_ids, n_boot, seed + 1,
    )

    return OODResult(
        layer=layer,
        pooling=pooling,
        n_pairs=int(h_eval_ood.shape[0]),
        n_questions=len(set(question_ids)),
        auroc=point_auroc,
        auroc_ci=auroc_ci,
        paired_win_rate=point_win,
        paired_win_rate_ci=win_ci,
        cohens_d=d,
        mean_proj_eval=float(proj_eval.mean()),
        mean_proj_deploy=float(proj_deploy.mean()),
    )
