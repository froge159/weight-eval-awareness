#!/usr/bin/env python3
"""Phase 1, step 1 runner: prompts -> activations -> X -> saved artifacts.

Stages, each gated so a failure stops the run rather than propagating:

  1. build prompt pairs        (validates every prompt)
  2. split disjointly          (validates no leakage)
  3. verify against reference  (optional; needs steering-eval-data)
  4. capture activations       (needs torch + GPU; skippable with --dry-run)
  5. build and save X          (validates alignment, polarity, degeneracy)

Examples
--------
Validate prompts and splits with no GPU, writing nothing::

    python scripts/run_step1.py --config config/smoke.yaml --dry-run

Check our prompts against the authors' released JSON::

    python scripts/run_step1.py --dry-run --reference-data /path/to/steering-eval-data

Full run on the organism::

    python scripts/run_step1.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from evalaware.config import load_config, set_all_seeds, write_manifest  # noqa: E402
from evalaware.contrastive import build_contrastive_matrix, save_contrastive_matrix  # noqa: E402
from evalaware.dataset import build_pairs, load_questions, make_splits, select  # noqa: E402
from evalaware.validation import ValidationError  # noqa: E402

log = logging.getLogger("step1")


def banner(text: str) -> None:
    log.info("")
    log.info("=" * 68)
    log.info("  %s", text)
    log.info("=" * 68)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 step 1: build the contrastive matrix")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--questions", type=Path, default=None, help="override questions_path")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--reference-data", type=Path, default=None,
                    help="extracted steering-eval-data/ dir, to verify our prompt transcription")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate prompts/splits only; no model, no activations, no writes")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides = {}
    if args.questions:
        overrides["questions_path"] = str(args.questions)
    if args.output_dir:
        overrides["output_dir"] = str(args.output_dir)
    if args.layers:
        overrides["layers"] = tuple(args.layers)

    cfg = load_config(args.config, **overrides)
    seeds = set_all_seeds(cfg.seed)
    log.info("seeds: %s", seeds)

    # ---------------------------------------------------------- stage 1
    banner("STAGE 1  build prompt pairs")
    qpath = Path(cfg.questions_path)
    if not qpath.is_absolute():
        qpath = REPO / qpath
    questions = load_questions(qpath)
    log.info("loaded %d questions from %s", len(questions), qpath)

    items = build_pairs(
        questions,
        regimes=cfg.regimes,
        situational_awareness_hint=cfg.situational_awareness_hint,
        max_templates_per_regime=cfg.max_templates_per_regime,
    )
    log.info("built %d matched pairs across regimes %s", len(items), list(cfg.regimes))
    log.info("  template families: %d", len({it.template_family for it in items}))
    log.info("  every prompt passed formatting + no-readout-leakage checks")

    # ---------------------------------------------------------- stage 2
    banner("STAGE 2  disjoint splits")
    splits = make_splits(
        items,
        seed=cfg.seed,
        template_fractions=cfg.template_fractions,
        question_fractions=cfg.question_fractions,
    )
    for name in ("extract", "ood_holdout", "behavioral"):
        n_fam = len(splits.template_families[name])
        n_q = len(splits.question_ids[name])
        try:
            n_items = len(select(items, splits, name))
        except ValidationError:
            n_items = 0
        log.info("  %-12s %3d families  %3d questions  %5d pairs", name, n_fam, n_q, n_items)
    log.info("  no template or question appears in two splits")

    extract_items = select(items, splits, "extract")
    if len(extract_items) < 100:
        log.warning(
            "only %d extraction pairs. Hua et al. used N=16 and therefore could not "
            "use SVD; low N puts step 2 in the same position. Consider more questions "
            "or templates.", len(extract_items),
        )

    # ---------------------------------------------------------- stage 3
    if args.reference_data:
        banner("STAGE 3  verify against Hua et al.'s released data")
        from evalaware.reference import load_reference, verify_against_reference
        ref = load_reference(args.reference_data)
        report = verify_against_reference(ref, strict=True)
        log.info("  prompt sets verified: %s", ", ".join(report["checked"]))
        log.info("  reference token_position: %s", report.get("reference_token_positions"))
        log.info("  reference num_prompts:    %s", report.get("reference_num_prompts"))
        log.info("  MATCH -- our transcription is byte-identical to theirs")
    else:
        log.info("stage 3 skipped (pass --reference-data to verify prompt transcription)")

    if args.dry_run:
        banner("DRY RUN COMPLETE -- prompts and splits valid, nothing written")
        ex = extract_items[0]
        log.info("example eval system prompt:\n%s", ex.eval_spec.system_prompt)
        log.info("example eval user prompt:\n%s", ex.eval_spec.user_prompt[:300])
        return 0

    # ---------------------------------------------------------- stage 4
    banner("STAGE 4  capture activations")
    from evalaware.activations import ActivationStore, CaptureConfig, capture

    out_dir = Path(cfg.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    contrastive_dir = out_dir / "contrastive"
    n_layers_seen: int | None = None

    saved: list[dict] = []
    for pooling in cfg.pooling_schemes:
        if cfg.extraction_source == "both":
            raise SystemExit(
                "extraction_source='both' needs two separate capture runs; "
                "set it to 'organism' or 'base' in the config."
            )
        if cfg.extraction_source == "base":
            _capture_model_id = cfg.base_model_id   # load the base directly
            _capture_base_id = None
        else:  # organism
            _capture_model_id = cfg.model_id        # the adapter
            _capture_base_id = cfg.base_model_id    # merged onto this base
        cap_cfg = CaptureConfig(
            model_id=_capture_model_id,
            base_model_id=_capture_base_id,
            revision=cfg.revision,
            dtype=cfg.dtype,
            device_map=cfg.device_map,
            pooling=pooling,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            layers=cfg.layers,
            batch_size=cfg.batch_size,
        )
        store = ActivationStore(REPO / cfg.cache_dir, cap_cfg)

        specs = [it.eval_spec for it in extract_items] + [it.deploy_spec for it in extract_items]
        report = capture(specs, cap_cfg, store)
        log.info("  [%s] captured %d, cached %d", pooling, report["captured"], report["cached"])
        n_layers_seen = report.get("n_layers", n_layers_seen)

        # ------------------------------------------------------ stage 5
        layer_list = cfg.layers if cfg.layers is not None else range((n_layers_seen or 0) + 1)
        layer_index = {l: i for i, l in enumerate(cfg.layers)} if cfg.layers else None

        for layer in layer_list:
            h_eval = store.stack([it.eval_spec for it in extract_items], layer, layer_index)
            h_deploy = store.stack([it.deploy_spec for it in extract_items], layer, layer_index)
            cm = build_contrastive_matrix(
                h_eval, h_deploy, extract_items,
                layer=layer, pooling=pooling, split="extract",
                model_id=cfg.model_id, token_offset=-2, regimes=cfg.regimes,
            )
            paths = save_contrastive_matrix(cm, contrastive_dir)
            saved.append(cm.summary())
            if layer % 10 == 0:
                log.info(
                    "  L%-3d %s  N=%d d=%d  consistency=%.3f (chance %.3f)%s",
                    layer, pooling, cm.n_pairs, cm.dim,
                    cm.diagnostics["consistency_ratio"],
                    cm.diagnostics["consistency_ratio_at_chance"],
                    "" if cm.diagnostics["consistency_above_chance"] else "  << AT CHANCE",
                )

    banner("STAGE 5  saved artifacts")
    index_path = contrastive_dir / "index.json"
    index_path.write_text(json.dumps(saved, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(cfg, "step1", {"n_cells": len(saved), "n_pairs": len(extract_items)})
    log.info("  %d cells written to %s", len(saved), contrastive_dir)
    log.info("  index: %s", index_path)
    log.info("")
    log.info("STEP 1 COMPLETE. Step 2 (SVD) consumes these -- see HANDOFF.md.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValidationError as exc:
        log.error("")
        log.error("VALIDATION FAILED")
        log.error("%s", exc)
        sys.exit(2)
