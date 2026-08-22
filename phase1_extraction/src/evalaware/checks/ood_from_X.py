"""PHASE 1, STEP 3a — OOD generalization, computed directly from X.

Why this exists rather than the activation-based version in ``ood.py``:

The held-out activations were never captured, so we cannot project H_eval and
H_deploy separately. But the *paired* statistic does not need them::

    proj_eval_i - proj_deploy_i = (h_eval_i - h_deploy_i) · r̂ = X_i · r̂

so every paired quantity is recoverable from X alone. And the paired statistic
is the one that matters here: on planted-direction fixtures it reached a 0.99
win-rate where unpaired AUROC sat at 0.75, because each question contributes a
large shared content term that swamps the between-class comparison.

What is *not* recoverable from X: unpaired AUROC, which needs the two classes
separately. It is omitted rather than approximated.

Two modes, and the distinction is the whole point:

``in_sample``
    Project X onto a direction that was estimated from all of X. Measures
    whether the direction describes the data it came from. Cannot answer the
    OOD question — reported for completeness and clearly labelled.

``leave_family_out``
    Re-estimate the direction from all template families except one, then test
    on the held-out family. This is a genuine OOD test over templates, which is
    what proposal §3.2 asks for, and it needs no new activations.

``cross_regime``
    The strongest version: estimate on every family of one cue regime, test on
    the other. A direction that fails here is the surface form of one cue, not
    evaluation awareness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..validation import assert_finite, assert_unit_norm, check

__all__ = [
    "PairedSeparation",
    "FoldResult",
    "paired_separation",
    "estimate_direction",
    "leave_family_out",
    "cross_regime_transfer",
    "evaluate_supplied_direction",
    "random_direction_null_paired",
]


# --------------------------------------------------------------- estimators

def estimate_direction(x: np.ndarray, estimator: str = "diffmeans") -> np.ndarray:
    """Estimate a unit direction from a contrastive matrix.

    ``diffmeans`` — normalized mean of the paired differences. For matched
    pairs this equals μ_eval − μ_deploy exactly.

    ``svd_v1`` — first right-singular vector, sign-fixed to agree with the mean
    so the result is deterministic across LAPACK builds.

    Both are provided because step 2 supplies both, and the fold-level
    comparison of the two is itself informative.
    """
    x = np.asarray(x, dtype=np.float64)
    check(x.ndim == 2 and x.shape[0] >= 2, f"estimate_direction: need a 2-D matrix with >=2 rows, got {x.shape}")

    if estimator == "diffmeans":
        v = x.mean(axis=0)
    elif estimator == "svd_v1":
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        v = vt[0]
        if float(v @ x.mean(axis=0)) < 0:
            v = -v
    else:
        raise ValueError(f"unknown estimator {estimator!r}; expected diffmeans or svd_v1")

    n = float(np.linalg.norm(v))
    check(n > 0, "estimate_direction: degenerate (zero) direction")
    return v / n


# --------------------------------------------------------------- statistics

@dataclass
class PairedSeparation:
    """Separation of a set of pairs along a direction, via X · r̂."""

    n_pairs: int
    n_questions: int
    win_rate: float                       # fraction of pairs projecting positive
    win_rate_ci: tuple[float, float]
    mean_projection: float
    effect_size: float                    # mean / sd of the projections
    median_projection: float

    @property
    def separates(self) -> bool:
        """Lower CI bound must clear 0.5. Chance is a coin flip per pair."""
        return self.win_rate_ci[0] > 0.5

    def summary(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "n_questions": self.n_questions,
            "win_rate": round(self.win_rate, 4),
            "win_rate_ci95": [round(c, 4) for c in self.win_rate_ci],
            "mean_projection": round(self.mean_projection, 4),
            "effect_size": round(self.effect_size, 4),
            "separates": self.separates,
        }


def _clustered_bootstrap(values: np.ndarray, cluster_ids: Sequence[str],
                         stat: Callable[[np.ndarray], float],
                         n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile CI resampling whole questions, not rows.

    Rows sharing a question are not independent — the same coding problem
    appears under every template — so row-level resampling gives CIs that are
    far too narrow. Follows Hua et al. Appendix I.2.
    """
    clusters = sorted(set(cluster_ids))
    idx: dict[str, list[int]] = {c: [] for c in clusters}
    for i, c in enumerate(cluster_ids):
        idx[c].append(i)

    rng = np.random.default_rng(seed)
    out: list[float] = []
    for _ in range(n_boot):
        rows: list[int] = []
        for p in rng.choice(len(clusters), size=len(clusters), replace=True):
            rows.extend(idx[clusters[p]])
        out.append(stat(values[np.asarray(rows, dtype=int)]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_separation(x: np.ndarray, r_hat: np.ndarray, question_ids: Sequence[str],
                      n_boot: int = 2000, seed: int = 0) -> PairedSeparation:
    """Project X onto r̂ and measure how consistently the pairs point positive."""
    x = np.asarray(x, dtype=np.float64)
    r_hat = np.asarray(r_hat, dtype=np.float64).ravel()
    assert_unit_norm(r_hat, "r_hat (paired_separation)")
    assert_finite(x, "X")
    check(
        x.shape[0] == len(question_ids),
        f"paired_separation: {x.shape[0]} rows vs {len(question_ids)} question ids",
    )

    proj = x @ r_hat
    win = lambda v: float((v > 0).mean() + 0.5 * (v == 0).mean())   # noqa: E731
    sd = float(proj.std(ddof=1)) if proj.size > 1 else 0.0

    return PairedSeparation(
        n_pairs=int(proj.size),
        n_questions=len(set(question_ids)),
        win_rate=win(proj),
        win_rate_ci=_clustered_bootstrap(proj, question_ids, win, n_boot, seed),
        mean_projection=float(proj.mean()),
        effect_size=float(proj.mean() / sd) if sd > 0 else float("inf"),
        median_projection=float(np.median(proj)),
    )


def random_direction_null_paired(x: np.ndarray, question_ids: Sequence[str],
                                 n_directions: int = 200, seed: int = 0) -> dict[str, float]:
    """Win-rate distribution for random unit directions on the same X.

    The bar r̂ must clear. Assuming 0.5 is not safe: X's rows are correlated, so
    an arbitrary direction can be consistently positive for uninteresting
    reasons. Folded, since a direction's sign is arbitrary.
    """
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_directions)
    for i in range(n_directions):
        v = rng.standard_normal(x.shape[1])
        v /= np.linalg.norm(v)
        p = x @ v
        scores[i] = max((p > 0).mean(), (p < 0).mean())
    return {
        "mean": float(scores.mean()),
        "p95": float(np.percentile(scores, 95)),
        "max": float(scores.max()),
        "n": n_directions,
    }


# ------------------------------------------------------------------- folds

@dataclass
class FoldResult:
    fold: str
    train_families: list[str]
    test_families: list[str]
    n_train: int
    estimator: str
    separation: PairedSeparation
    cos_train_test_direction: float | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "n_train_pairs": self.n_train,
            "test_families": self.test_families,
            "estimator": self.estimator,
            "cos_train_vs_test_direction": (
                round(self.cos_train_test_direction, 4)
                if self.cos_train_test_direction is not None else None),
            **self.separation.summary(),
        }


def _family(row: dict) -> str:
    return f"{row['regime']}|{row['template_family_eval']}|{row['template_family_deploy']}"


def leave_family_out(x: np.ndarray, rows: Sequence[dict], estimator: str = "diffmeans",
                     n_boot: int = 2000, seed: int = 0) -> list[FoldResult]:
    """Hold out one template family, estimate on the rest, test on the held-out.

    A genuine OOD test over templates using only X — the held-out family's rows
    contribute nothing to the direction being tested against them.
    """
    x = np.asarray(x, dtype=np.float64)
    fams = [_family(r) for r in rows]
    uniq = sorted(set(fams))
    check(len(uniq) >= 2, f"leave_family_out: need >=2 template families, got {len(uniq)}")

    results: list[FoldResult] = []
    for held in uniq:
        test_mask = np.array([f == held for f in fams])
        train_mask = ~test_mask
        if int(test_mask.sum()) < 4 or int(train_mask.sum()) < 4:
            continue

        r_train = estimate_direction(x[train_mask], estimator)
        r_test = estimate_direction(x[test_mask], estimator)

        sep = paired_separation(
            x[test_mask], r_train,
            [r["question_id"] for r, m in zip(rows, test_mask) if m],
            n_boot=n_boot, seed=seed)

        results.append(FoldResult(
            fold=held,
            train_families=[f for f in uniq if f != held],
            test_families=[held],
            n_train=int(train_mask.sum()),
            estimator=estimator,
            separation=sep,
            cos_train_test_direction=float(r_train @ r_test),
        ))
    return results


def cross_regime_transfer(x: np.ndarray, rows: Sequence[dict], estimator: str = "diffmeans",
                          n_boot: int = 2000, seed: int = 0) -> list[FoldResult]:
    """Estimate on one cue regime, test on the other.

    The proposal requires r̂ to replicate under both the explicit and Wood Labs
    cues before any surgery. This is the behavioral version of that check —
    stronger than a cosine between the two regime directions, because it asks
    whether one actually *works* on the other's data.
    """
    x = np.asarray(x, dtype=np.float64)
    regimes = [r["regime"] for r in rows]
    uniq = sorted(set(regimes))
    check(len(uniq) == 2, f"cross_regime_transfer: expected 2 regimes, got {uniq}")

    results: list[FoldResult] = []
    for train_reg in uniq:
        test_reg = [r for r in uniq if r != train_reg][0]
        train_mask = np.array([r == train_reg for r in regimes])
        test_mask = ~train_mask

        r_train = estimate_direction(x[train_mask], estimator)
        r_test = estimate_direction(x[test_mask], estimator)

        sep = paired_separation(
            x[test_mask], r_train,
            [r["question_id"] for r, m in zip(rows, test_mask) if m],
            n_boot=n_boot, seed=seed)

        notes = []
        if not sep.separates:
            notes.append(
                f"direction estimated on {train_reg!r} does NOT separate framings "
                f"on {test_reg!r}. Under the proposal's cross-cue requirement this "
                f"direction is that cue's surface form, not evaluation awareness."
            )

        results.append(FoldResult(
            fold=f"train:{train_reg} -> test:{test_reg}",
            train_families=[train_reg],
            test_families=[test_reg],
            n_train=int(train_mask.sum()),
            estimator=estimator,
            separation=sep,
            cos_train_test_direction=float(r_train @ r_test),
            notes=notes,
        ))
    return results


def evaluate_supplied_direction(x: np.ndarray, rows: Sequence[dict], r_hat: np.ndarray,
                                n_boot: int = 2000, n_null: int = 200,
                                seed: int = 0) -> dict[str, Any]:
    """Project X onto a direction supplied by step 2.

    IN-SAMPLE if that direction was estimated from this same X — which it was.
    Reported because it establishes the ceiling and confirms the sign, but it
    cannot answer the OOD question. The fold results do that.
    """
    qids = [r["question_id"] for r in rows]
    overall = paired_separation(x, r_hat, qids, n_boot=n_boot, seed=seed)
    null = random_direction_null_paired(x, qids, n_null, seed)

    by_regime: dict[str, Any] = {}
    regimes = [r["regime"] for r in rows]
    for reg in sorted(set(regimes)):
        mask = np.array([r == reg for r in regimes])
        if mask.sum() >= 4:
            by_regime[reg] = paired_separation(
                x[mask], r_hat, [q for q, m in zip(qids, mask) if m],
                n_boot=n_boot, seed=seed).summary()

    return {
        "scope": "in_sample",
        "overall": overall.summary(),
        "by_regime": by_regime,
        "random_direction_null": null,
        "beats_null": overall.win_rate_ci[0] > null["p95"],
    }
