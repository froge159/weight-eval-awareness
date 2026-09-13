#!/usr/bin/env python3
"""Check whether r_hat is actually rank-1, per cue regime.

Stage 1's dimensionality check: before running the causal test on the
completion_mean explicit-cue direction, confirm each regime's contrastive
matrix is well-described by a single dominant direction.

The existing ``v1_svd_check`` field (in the committed r_hat_L*.npz files) is
only cos(v1, difference-of-means) on the FULL (both-regime) X. Per-regime
directions are computed in run_step2.py by a stripped-down helper (svd_v1())
that keeps only the top singular vector and discards every diagnostic --
including the codebase's own answer to this exact question.

evalaware.extraction.extract_direction() already computes the right thing:
DirectionResult.one_dimensional judges cos(v1, diff-of-means) against
random_pair_cosine_baseline (never variance-explained-by-v1, which
extraction.py's docstring documents as unreliable in high dimensions -- see
RESEARCH_DECISIONS.md 2026-08-08T14:20Z and scripts/estimator_comparison.py).
This script just applies that existing, already-used-for-the-combined-case
function to each regime's rows too, instead of inventing a new statistic.

    PYTHONPATH=src python scripts/run_dimensionality_check.py \\
        --contrastive outputs_cm/contrastive --pooling completion_mean
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evalaware.contrastive import load_contrastive_matrix  # noqa: E402
from evalaware.extraction import extract_direction  # noqa: E402

log = logging.getLogger("dimcheck")

_NAME = re.compile(r"X_L(?P<layer>\d+)_(?P<pooling>[a-z_]+)_(?P<split>[a-z_]+)\.npz")

REGIMES = ("combined", "explicit", "woodlabs")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--contrastive", required=True,
                   help="dir with X_L*_{pooling}_{split}.npz, e.g. "
                        "outputs_cm/contrastive")
    p.add_argument("--pooling", default="completion_mean")
    p.add_argument("--split", default="extract")
    p.add_argument("--band", type=int, nargs=2, default=[36, 76],
                   help="inclusive layer range to check (the edit band); "
                        "the causal test only uses these layers")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: "
                        "outputs/dimensionality_check_{pooling}.json)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    cdir = Path(args.contrastive)
    if not cdir.is_absolute():
        cdir = REPO / cdir
    if not cdir.is_dir():
        log.error("contrastive dir not found: %s", cdir)
        return 1

    lo, hi = args.band
    matrices = []
    for path in sorted(cdir.glob(f"X_L*_{args.pooling}_{args.split}.npz")):
        m = _NAME.match(path.name)
        if m and m.group("pooling") == args.pooling and m.group("split") == args.split:
            layer = int(m.group("layer"))
            if lo <= layer <= hi:
                matrices.append((layer, path))
    matrices.sort()
    if not matrices:
        log.error("no X_L*_%s_%s.npz in %s within band %s",
                   args.pooling, args.split, cdir, args.band)
        return 1
    layers_found = [layer for layer, _ in matrices]
    log.info("checking %d layer(s) in band %s: %s",
              len(matrices), args.band, layers_found)

    report: dict[str, dict] = {}
    log.info("%-5s %-9s %6s %10s %10s %14s  %s", "layer", "regime", "n",
              "var(v1)", "cos(v1,dm)", "10x p99 base", "one_dimensional")
    any_ambiguous = False
    for layer, path in matrices:
        cm = load_contrastive_matrix(path)
        regimes_arr = np.array([r.regime for r in cm.rows])
        report[str(layer)] = {}
        for regime in REGIMES:
            x = cm.x if regime == "combined" else cm.x[regimes_arr == regime]
            n = x.shape[0]
            if n < 2:
                log.info("%-5d %-9s %6d  -- skipped: fewer than 2 rows",
                          layer, regime, n)
                report[str(layer)][regime] = {"n": n, "skipped": "fewer than 2 rows"}
                continue
            res = extract_direction(np.asarray(x, dtype=np.float64),
                                     layer=layer, pooling=args.pooling)
            threshold = 10 * res.random_pair_baseline["abs_p99"]
            report[str(layer)][regime] = res.summary()
            report[str(layer)][regime]["one_dimensional_threshold"] = threshold
            if not res.one_dimensional:
                any_ambiguous = True
            log.info(
                "%-5d %-9s %6d %10.4f %10.4f %14.4f  %s",
                layer, regime, n, res.variance_explained_v1, res.cos_v1_diffmeans,
                threshold, res.one_dimensional,
            )

    default_out = f"outputs/dimensionality_check_{args.pooling}.json"
    out = Path(args.out) if args.out else REPO / default_out
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"band": args.band, "pooling": args.pooling, "any_ambiguous": any_ambiguous,
         "layers": report},
        indent=2, sort_keys=True,
    ), encoding="utf-8")
    log.info("wrote %s", out)
    verdict = ("SOME (layer, regime) cells are NOT one_dimensional -- see above"
               if any_ambiguous else
               "ALL (layer, regime) cells in band are one_dimensional")
    log.info("%s", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
