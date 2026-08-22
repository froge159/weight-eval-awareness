"""Phase 1 steps 1 and 2: contrastive matrix, then SVD.

    Step 1:  X = H_eval - H_deploy          (N x d, one row per matched pair)
    Step 2:  X = U S V^T  ->  r_hat = v_1   (first right-singular vector)

Three things worth being explicit about, because they are easy to get subtly
wrong and hard to notice afterwards:

**Sign.** SVD determines v_1 only up to sign; ``svd`` may return either -v_1 or
+v_1 depending on LAPACK build, so an unfixed sign is a cross-machine
reproducibility bug. We fix it by requiring ``<v_1, mean(X)> >= 0``, i.e. r_hat
points from deployment toward evaluation.

**Centering.** We deliberately do *not* mean-center X. The mean of X is the
signal here -- it is the difference-of-means direction. Centering would remove
exactly what we are looking for and leave v_1 describing the *variation* in the
contrast across items. ``center=True`` is available for the diagnostic of
asking how much structure remains after the mean is removed.

**Scaling.** Residual stream norms grow substantially with depth in Llama-family
models. Directions are computed per-layer so this does not matter within a
layer, but it does mean cross-layer comparisons must use cosine, never
Euclidean distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .validation import (
    assert_finite,
    assert_paired,
    assert_shape,
    assert_unit_norm,
    check,
    cosine,
    random_pair_cosine_baseline,
)

__all__ = ["DirectionResult", "contrastive_matrix", "extract_direction", "diff_of_means"]


@dataclass
class DirectionResult:
    """r_hat for one (layer, pooling) cell, with its diagnostics."""

    layer: int
    pooling: str
    r_hat: np.ndarray                 # (d,) unit vector, SVD v_1 -- the primary
    r_diffmeans: np.ndarray           # (d,) unit vector, secondary cross-check
    n_pairs: int
    dim: int
    singular_values: np.ndarray       # top-k singular values of X
    variance_explained_v1: float      # sigma_1^2 / sum(sigma_i^2)
    cos_v1_diffmeans: float
    random_pair_baseline: dict[str, float]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def one_dimensional(self) -> bool:
        """Is the contrast cleanly one-dimensional?

        Judged on ``cos(v_1, diff_of_means)`` against the random-pair baseline,
        NOT on variance explained.

        An earlier version of this property also required
        ``variance_explained_v1 >= 0.30``. That was wrong, and
        ``scripts/estimator_comparison.py`` demonstrates why: in high
        dimensions sigma_1^2 / sum(sigma_i^2) sits at 0.004-0.10 even when v_1
        recovers a planted direction almost perfectly, because it is dominated
        by the thousands of noise directions. Any threshold on it rejects good
        directions. See RESEARCH_DECISIONS.md 2026-08-08T14:20Z.

        Variance explained is still reported -- it is informative across layers
        -- but it is not a gate.
        """
        return abs(self.cos_v1_diffmeans) > 10 * self.random_pair_baseline["abs_p99"]

    def summary(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "pooling": self.pooling,
            "n_pairs": self.n_pairs,
            "dim": self.dim,
            "variance_explained_v1": round(self.variance_explained_v1, 6),
            "cos_v1_diffmeans": round(self.cos_v1_diffmeans, 6),
            "random_pair_abs_p99": round(self.random_pair_baseline["abs_p99"], 6),
            "one_dimensional": self.one_dimensional,
            "top_singular_values": [round(float(s), 4) for s in self.singular_values[:5]],
        }


def contrastive_matrix(
    h_eval: np.ndarray,
    h_deploy: np.ndarray,
    eval_ids: list[str],
    deploy_ids: list[str],
) -> np.ndarray:
    """Step 1. Build X = H_eval - H_deploy with alignment enforced."""
    h_eval = np.asarray(h_eval, dtype=np.float64)
    h_deploy = np.asarray(h_deploy, dtype=np.float64)

    assert_shape(h_eval, (None, None), "H_eval")
    assert_shape(h_deploy, h_eval.shape, "H_deploy")
    assert_finite(h_eval, "H_eval")
    assert_finite(h_deploy, "H_deploy")
    assert_paired(eval_ids, deploy_ids, name="contrastive_matrix")
    check(
        len(eval_ids) == h_eval.shape[0],
        f"contrastive_matrix: {len(eval_ids)} ids but H_eval has "
        f"{h_eval.shape[0]} rows.",
    )

    x = h_eval - h_deploy
    assert_finite(x, "X")

    # A near-zero row means the two framings produced identical activations,
    # which almost always means a prompt-construction bug (both sides rendered
    # the same text) rather than a real null.
    row_norms = np.linalg.norm(x, axis=1)
    median_norm = float(np.median(row_norms))
    if median_norm > 0:
        degenerate = int((row_norms < 1e-6 * median_norm).sum())
        check(
            degenerate == 0,
            f"contrastive_matrix: {degenerate} of {x.shape[0]} rows are "
            f"numerically zero (< 1e-6 x median row norm {median_norm:.4g}). "
            f"The eval and deploy prompts for those pairs likely rendered to "
            f"identical text -- check the template pairing.",
        )
    else:
        raise AssertionError(
            "contrastive_matrix: every row of X is zero. H_eval and H_deploy are "
            "identical -- the two framings were never actually applied, or the "
            "same activations were loaded twice."
        )
    return x


def diff_of_means(x: np.ndarray) -> np.ndarray:
    """Secondary estimator: normalized mean of the paired differences.

    For matched pairs, mean(H_eval - H_deploy) == mu_eval - mu_deploy exactly,
    so this is the paper's eq. (1) computed on the paired form.
    """
    r = np.asarray(x, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(r))
    check(
        norm > 0,
        "diff_of_means: mean difference is the zero vector -- the contrast has "
        "no consistent direction across pairs.",
    )
    return r / norm


def extract_direction(
    x: np.ndarray,
    layer: int,
    pooling: str,
    center: bool = False,
    n_singular_values: int = 10,
    baseline_seed: int = 0,
) -> DirectionResult:
    """Step 2. r_hat = v_1 via SVD, plus the diagnostics that qualify it."""
    x = np.asarray(x, dtype=np.float64)
    assert_shape(x, (None, None), "X")
    assert_finite(x, "X")
    n, d = x.shape
    check(
        n >= 2,
        f"extract_direction: need at least 2 pairs to estimate a direction, got {n}.",
    )
    if n < d:
        # Not an error -- v_1 is still well defined -- but with fewer pairs than
        # dimensions the SVD can fit noise, so the OOD check does the real work.
        pass

    x_used = x - x.mean(axis=0, keepdims=True) if center else x

    # full_matrices=False: we only need the top singular triplets, and the full
    # d x d V would be 16k x 16k for the 49B.
    u, s, vt = np.linalg.svd(x_used, full_matrices=False)
    assert_finite(s, "singular values")
    check(s[0] > 0, "extract_direction: leading singular value is zero; X has no structure.")

    v1 = vt[0].copy()

    # Fix the sign so r_hat points deployment -> evaluation, deterministically.
    mean_diff = x.mean(axis=0)
    if float(np.dot(v1, mean_diff)) < 0:
        v1 = -v1
    # Degenerate tie-break: if v1 is orthogonal to the mean, fall back to the
    # first nonzero coordinate so the result is still machine-independent.
    if abs(float(np.dot(v1, mean_diff))) < 1e-12:
        nz = np.flatnonzero(np.abs(v1) > 1e-12)
        if nz.size and v1[nz[0]] < 0:
            v1 = -v1

    v1 = v1 / np.linalg.norm(v1)
    assert_unit_norm(v1, f"r_hat[L{layer}/{pooling}]")

    r_dm = diff_of_means(x)
    assert_unit_norm(r_dm, f"r_diffmeans[L{layer}/{pooling}]")

    total_var = float((s ** 2).sum())
    check(total_var > 0, "extract_direction: total variance is zero.")
    var_explained = float(s[0] ** 2 / total_var)

    return DirectionResult(
        layer=layer,
        pooling=pooling,
        r_hat=v1,
        r_diffmeans=r_dm,
        n_pairs=n,
        dim=d,
        singular_values=s[:n_singular_values].copy(),
        variance_explained_v1=var_explained,
        cos_v1_diffmeans=cosine(v1, r_dm),
        random_pair_baseline=random_pair_cosine_baseline(d, seed=baseline_seed),
        extras={"centered": center, "n_pairs_lt_dim": bool(n < d)},
    )


def cross_layer_cosine_matrix(directions: dict[int, np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """cos(r_hat_l, r_hat_l') for every layer pair.

    Tests the shared-basis assumption that Phase 2 relies on when it applies one
    direction across all residual layers. A block-diagonal structure means the
    basis rotates with depth and the edit must be restricted to a stable band.
    """
    check(len(directions) >= 2, "cross_layer_cosine_matrix: need at least 2 layers.")
    layers = sorted(directions)
    mat = np.eye(len(layers))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            if i < j:
                c = cosine(directions[li], directions[lj])
                mat[i, j] = mat[j, i] = c
    return mat, layers


def stable_band(
    cos_matrix: np.ndarray,
    layers: list[int],
    threshold: float,
    min_width: int = 3,
) -> tuple[int, int] | None:
    """Longest run of consecutive layers that are pairwise similar above ``threshold``.

    ``threshold`` should be set from the random-pair baseline, not picked by
    eye. Returns inclusive (start_layer, end_layer), or None if no run reaches
    ``min_width``.
    """
    n = len(layers)
    best: tuple[int, int] | None = None
    best_width = min_width - 1
    for i in range(n):
        for j in range(i + min_width - 1, n):
            block = cos_matrix[i:j + 1, i:j + 1]
            if float(block.min()) >= threshold:
                width = j - i + 1
                if width > best_width:
                    best_width = width
                    best = (layers[i], layers[j])
    return best
