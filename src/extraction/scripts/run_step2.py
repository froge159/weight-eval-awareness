#!/usr/bin/env python3
"""Step 2 --- extract r_hat direction files from saved contrastive matrices.

Phase 1's run_step1 stops at the contrastive matrix X and saves it per
(layer, pooling). The direction files r_hat_L{layer}_{pooling}.npz were built by
a hand-run step that was never committed. This reproduces that step so the
completion_mean re-extraction is scripted and reproducible.

For each layer it writes, matching what load_direction_set and
run_step3a_from_X.py read:
    r_hat            SVD v_1 of the full X          (primary)
    r_hat_explicit   SVD v_1 of the explicit rows   (per-regime)
    r_hat_woodlabs   SVD v_1 of the woodlabs rows   (per-regime)
    r_diffmeans      difference-of-means of full X  (secondary cross-check)
    v1_svd_check     cos(v_1, diff-of-means) for the full X

Sign convention matches the codebase: v_1 is flipped so it points eval-ward,
i.e. v_1 . mean(X) >= 0, since SVD fixes v_1 only up to sign.

    # dry run: read matrices, extract, print, write nothing
    PYTHONPATH=src python scripts/run_step2.py --contrastive outputs_cm/contrastive --dry-run

    # real: write into outputs/directions so Phase 2 finds them
    PYTHONPATH=src python scripts/run_step2.py \\
        --contrastive outputs_cm/contrastive --out outputs/directions --pooling completion_mean
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

log = logging.getLogger("step2")

_NAME = re.compile(r"X_L(?P<layer>\d+)_(?P<pooling>[a-z_]+)_(?P<split>[a-z_]+)\.npz")


def svd_v1(x: np.ndarray) -> np.ndarray:
    """Top right-singular vector of X, sign-fixed to point eval-ward."""
    x = np.asarray(x, dtype=np.float64)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    v1 = vt[0]
    if v1 @ x.mean(axis=0) < 0:       # SVD is sign-ambiguous; match the codebase
        v1 = -v1
    return v1 / np.linalg.norm(v1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contrastive", required=True,
                   help="dir with X_L*_{pooling}_extract.npz (e.g. outputs_cm/contrastive)")
    p.add_argument("--out", default="outputs/directions",
                   help="where to write r_hat_L*.npz")
    p.add_argument("--pooling", default="completion_mean")
    p.add_argument("--split", default="extract")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    cdir = Path(args.contrastive)
    if not cdir.is_absolute():
        cdir = REPO / cdir
    if not cdir.is_dir():
        log.error("contrastive dir not found: %s", cdir)
        return 1

    matrices = []
    for path in sorted(cdir.glob(f"X_L*_{args.pooling}_{args.split}.npz")):
        m = _NAME.match(path.name)
        if m and m.group("pooling") == args.pooling and m.group("split") == args.split:
            matrices.append((int(m.group("layer")), path))
    matrices.sort()

    if not matrices:
        log.error("no X_L*_%s_%s.npz in %s", args.pooling, args.split, cdir)
        return 1
    log.info("found %d layers: %s", len(matrices), [l for l, _ in matrices])

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    for layer, path in matrices:
        cm = load_contrastive_matrix(path)
        x = np.asarray(cm.X, dtype=np.float64)
        regimes = np.array([r.regime for r in cm.rows])

        # Primary + secondary via the codebase's own extractor, so diagnostics
        # (variance explained, random-pair baseline, cos to diff-of-means) match
        # exactly what step 3a expects.
        res = extract_direction(x, layer=layer, pooling=args.pooling)
        r_hat = np.asarray(res.r_hat, dtype=np.float64)
        r_diffmeans = np.asarray(res.r_diffmeans, dtype=np.float64)

        payload = {
            "r_hat": r_hat,
            "r_diffmeans": r_diffmeans,
            "v1_svd_check": np.array(res.cos_v1_diffmeans, dtype=np.float64),
        }

        # Per-regime directions, each the SVD v_1 of that regime's rows.
        regime_report = {}
        for name in ("explicit", "woodlabs"):
            mask = regimes == name
            n = int(mask.sum())
            if n < 2:
                log.warning("L%d: only %d %r rows, skipping that regime direction",
                            layer, n, name)
                continue
            v = svd_v1(x[mask])
            payload[f"r_hat_{name}"] = v
            regime_report[name] = (n, float(abs(v @ r_hat)))

        cos_report = "  ".join(f"{k}:n={n},|cos r_hat|={c:.3f}"
                               for k, (n, c) in regime_report.items())
        log.info("L%-3d  var(v1)=%.3f  cos(v1,dm)=%.3f  | %s",
                 layer, res.variance_explained_v1, res.cos_v1_diffmeans, cos_report)

        if not args.dry_run:
            dest = out / f"r_hat_L{layer}_{args.pooling}.npz"
            np.savez(dest, **payload)

    if args.dry_run:
        log.info("DRY RUN -- nothing written")
    else:
        log.info("wrote %d direction files to %s", len(matrices), out)
        # refresh the index load_direction_set reads
        idx = {"pooling": args.pooling, "layers": [l for l, _ in matrices],
               "source": str(cdir)}
        (out / f"direction_index_{args.pooling}.json").write_text(json.dumps(idx, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
