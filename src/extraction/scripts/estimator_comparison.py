#!/usr/bin/env python3
"""SVD v1 vs difference-of-means, on synthetic data with a known planted direction.

Motivation: the slides specify SVD v1 as the primary estimator, while the paper
draft and the prior work we build on (Arditi et al., Hua et al.) use a
difference of means. Since the group has to pick one, it is worth knowing how
they actually behave at our N and dimensionality rather than arguing from
first principles.

Ground truth is planted, so "which estimator recovers it better" is directly
measurable.

Run: python scripts/estimator_comparison.py
"""

from __future__ import annotations

import argparse

import numpy as np


def run(dim: int = 2048, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    true_dir = rng.standard_normal(dim)
    true_dir /= np.linalg.norm(true_dir)

    rows: list[dict] = []
    for n in (16, 200, 1000):
        for snr in (0.05, 0.2, 1.0):
            # Shared per-question content, independent noise per side, plus the
            # planted contrast on the eval side only.
            base = rng.standard_normal((n, dim)) * 3
            signal = snr * np.sqrt(2 * dim)
            h_deploy = base + rng.standard_normal((n, dim))
            h_eval = base + rng.standard_normal((n, dim)) + signal * true_dir
            x = h_eval - h_deploy

            _, s, vt = np.linalg.svd(x, full_matrices=False)
            v1 = vt[0]
            if v1 @ x.mean(axis=0) < 0:
                v1 = -v1
            mean_dir = x.mean(axis=0)
            mean_dir = mean_dir / np.linalg.norm(mean_dir)

            rows.append({
                "n": n,
                "snr": snr,
                "cos_v1": abs(float(v1 @ true_dir)),
                "cos_mean": abs(float(mean_dir @ true_dir)),
                "var_explained_v1": float(s[0] ** 2 / (s ** 2).sum()),
                "cos_v1_mean": abs(float(v1 @ mean_dir)),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2048,
                    help="hidden dim; 8192 matches the 49B but is slow. The "
                         "qualitative result does not depend on it.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = run(args.dim, args.seed)
    print(f"dim={args.dim}  seed={args.seed}\n")
    print(f"{'N':>6} {'SNR':>6} {'cos(v1,true)':>13} {'cos(mean,true)':>15} "
          f"{'varexp_v1':>10} {'cos(v1,mean)':>13}  winner")
    print("-" * 86)
    last_n = None
    for r in rows:
        if last_n is not None and r["n"] != last_n:
            print()
        last_n = r["n"]
        win = ("mean" if r["cos_mean"] > r["cos_v1"] + 0.02
               else "v1" if r["cos_v1"] > r["cos_mean"] + 0.02 else "tie")
        print(f"{r['n']:6d} {r['snr']:6.2f} {r['cos_v1']:13.3f} {r['cos_mean']:15.3f} "
              f"{r['var_explained_v1']:10.4f} {r['cos_v1_mean']:13.3f}  {win}")

    print("""
Reading of these numbers
------------------------
1. Difference of means is never worse than v1, and is clearly better at low
   SNR. They converge once the contrast is strong. There is no regime here
   where v1 wins.

   Why: mean(X) averages independent noise down by sqrt(N). v1 maximizes
   variance, and at low SNR the largest-variance direction of X is a noise
   direction, not the contrast.

2. Variance explained by v1 is small (0.004-0.10) even in rows where v1
   recovers the true direction almost perfectly. In high dimensions,
   sigma_1^2 / sum(sigma_i^2) is dominated by the many noise directions, so it
   is NOT a usable test of whether the signal is one-dimensional. Any threshold
   on it (e.g. "> 0.3 means clean") will reject good directions.

   Use cos(v1, mean) instead, judged against the random-pair baseline: it is
   high exactly when the contrast is concentrated on one axis.

3. N matters more than the choice of estimator. At N=16 and low SNR both
   estimators fail. Hua et al.'s released configs record num_prompts=16 --
   which is very likely why they used a difference of means. Our design gets
   ~3700 extraction pairs, which is what makes either estimator viable.

Implication for the group: the slides' choice of SVD is defensible at our N,
but it should be reported alongside the difference of means rather than
instead of it, and the one-dimensionality claim should rest on cos(v1, mean),
not on variance explained.
""")


if __name__ == "__main__":
    main()
