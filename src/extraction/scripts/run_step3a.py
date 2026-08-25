#!/usr/bin/env python3
"""Phase 1, step 3a: OOD generalization check.

Projects held-out activations onto r̂ and tests whether that single scalar
separates eval- from deploy-framed prompts on templates never seen during
extraction.

Prerequisites:
  1. `ood_holdout` activations captured (same cache, same CaptureConfig as step 1)
  2. r̂ from the step 2 owner -- see HANDOFF_STEP2_TO_3A.md

Examples
--------
Single direction, one layer::

    python scripts/run_step3a.py --directions r_hat.npz --layers 40

Per-layer sweep::

    python scripts/run_step3a.py --directions r_hat.npz \\
        --layers 4 8 12 16 20 24 28 32 36 40 44 48

Dry run -- verify the holdout data is present and r̂ loads, without the stats::

    python scripts/run_step3a.py --directions r_hat.npz --layers 40 --check-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evalaware.activations import ActivationStore, CaptureConfig      # noqa: E402
from evalaware.checks.ood import run_ood_check                        # noqa: E402
from evalaware.config import load_config, set_all_seeds, write_manifest  # noqa: E402
from evalaware.dataset import build_pairs, load_questions, make_splits, select  # noqa: E402
from evalaware.direction_io import load_directions                    # noqa: E402
from evalaware.validation import ValidationError                      # noqa: E402

log = logging.getLogger("step3a")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 step 3a: OOD generalization")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--directions", type=Path, required=True,
                    help="r_hat file from step 2 (.npz/.npy/.pt)")
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="layers to test; must match the rows of the r_hat file")
    ap.add_argument("--captured-layers", type=int, nargs="*", default=None,
                    help="layers present in the activation cache, in order. "
                         "Defaults to config.layers.")
    ap.add_argument("--pooling", default="last_prompt_token")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    set_all_seeds(cfg.seed)

    # ---------------------------------------------------------------- data
    qpath = Path(cfg.questions_path)
    if not qpath.is_absolute():
        qpath = REPO / qpath
    questions = load_questions(qpath)
    items = build_pairs(questions, regimes=cfg.regimes,
                        situational_awareness_hint=cfg.situational_awareness_hint,
                        max_templates_per_regime=cfg.max_templates_per_regime)
    splits = make_splits(items, seed=cfg.seed,
                         template_fractions=cfg.template_fractions,
                         question_fractions=cfg.question_fractions)
    holdout = select(items, splits, "ood_holdout")

    log.info("holdout pairs: %d", len(holdout))
    log.info("  questions: %d", len({it.question_id for it in holdout}))
    log.info("  template families: %d", len({it.template_family for it in holdout}))
    by_regime: dict[str, int] = {}
    for it in holdout:
        by_regime[it.regime] = by_regime.get(it.regime, 0) + 1
    log.info("  per regime: %s", by_regime)

    # The whole point of this check: holdout templates must be unseen.
    extract_fams = set(splits.template_families["extract"])
    leaked = {it.template_family for it in holdout} & extract_fams
    if leaked:
        raise ValidationError(
            f"{len(leaked)} holdout template families also appear in the "
            f"extraction split: {sorted(leaked)[:3]}. The OOD check is "
            f"meaningless under leakage."
        )
    log.info("  no template leakage from the extraction split")

    # ---------------------------------------------------------- directions
    captured = args.captured_layers or list(cfg.layers or [])
    loaded = load_directions(args.directions, layers=args.layers)
    log.info("loaded r_hat: %s", json.dumps(loaded.summary()["metadata"] or {}, default=str))
    log.info("  %d direction(s), dim %d", len(loaded.layers), loaded.dim)
    if loaded.renormalized:
        log.warning("  directions were renormalized to unit norm")

    cap_cfg = CaptureConfig(model_id=cfg.model_id, dtype=cfg.dtype,
                            pooling=args.pooling, layers=cfg.layers,
                            max_new_tokens=cfg.max_new_tokens,
                            temperature=cfg.temperature)
    store = ActivationStore(REPO / cfg.cache_dir, cap_cfg)
    log.info("activation cache: %s", store.root)

    missing = [it.pair_key for it in holdout
               if not store.has(it.eval_spec.system_prompt, it.eval_spec.user_prompt)
               or not store.has(it.deploy_spec.system_prompt, it.deploy_spec.user_prompt)]
    if missing:
        raise ValidationError(
            f"{len(missing)} holdout pair(s) have no cached activations, e.g. "
            f"{missing[:2]}. Capture the ood_holdout split before running 3a -- "
            f"same model, same pooling, same token offset as the extract split."
        )
    log.info("  all %d holdout pairs present in cache", len(holdout))

    if args.check_only:
        log.info("\ncheck-only: prerequisites satisfied, statistics not run")
        return 0

    # --------------------------------------------------------------- run
    layer_index = {l: i for i, l in enumerate(captured)} if captured else None
    qids = [it.question_id for it in holdout]
    regimes = [it.regime for it in holdout]

    results = []
    for layer in args.layers:
        h_eval = store.stack([it.eval_spec for it in holdout], layer, layer_index)
        h_deploy = store.stack([it.deploy_spec for it in holdout], layer, layer_index)

        res = run_ood_check(
            loaded.for_layer(layer), h_eval, h_deploy, qids, regimes,
            layer=layer, pooling=args.pooling,
            n_boot=args.n_boot, n_null=args.n_null, seed=cfg.seed)
        results.append(res.summary())

        p = res.pooled
        log.info(
            "L%-3d AUROC %.3f [%.3f,%.3f]  paired %.3f [%.3f,%.3f]  "
            "null_p95 %.3f  %s%s",
            layer, p.auroc, *p.auroc_ci, p.paired_win_rate, *p.paired_win_rate_ci,
            res.null_auroc_p95,
            "PASS" if res.passes else "FAIL",
            "" if res.transfers_across_regimes else "  (no cross-regime transfer)")
        for note in res.notes:
            log.warning("     %s", note)

    # ------------------------------------------------------------ report
    out_dir = Path(cfg.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_path = args.output or (out_dir / "step3a_ood.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"directions": loaded.summary(), "results": results},
        indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_manifest(cfg, "step3a", {"n_layers": len(results),
                                   "n_holdout_pairs": len(holdout)})

    passing = [r for r in results if r["passes"]]
    transfer = [r for r in results if r["transfers_across_regimes"]]
    log.info("")
    log.info("=" * 60)
    log.info("  %d/%d layers separate framings above the random-direction null",
             len(passing), len(results))
    log.info("  %d/%d also transfer across both cue regimes", len(transfer), len(results))
    if passing:
        best = max(passing, key=lambda r: r["pooled"]["paired_win_rate"])
        log.info("  strongest: layer %s, paired win-rate %.3f",
                 best["layer"], best["pooled"]["paired_win_rate"])
    if not transfer:
        log.warning("  NO layer transfers across regimes. The proposal requires")
        log.warning("  replication under both cues before surgery -- a direction")
        log.warning("  specific to one cue is that cue's surface form.")
    log.info("=" * 60)
    log.info("report: %s", out_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValidationError as exc:
        log.error("\nVALIDATION FAILED\n%s", exc)
        sys.exit(2)
