#!/usr/bin/env python3
"""Phase 2 runner: directions -> stability check -> merged model -> edited model.

Stages, each gated so a failure stops the run rather than propagating:

  1. cross-layer stability   (no torch, no model, seconds)
  2. module census           (two small JSON files from the hub; --dry-run ends here)
  3. merge + verify + edit   (needs the bf16 checkpoint)
  4. fidelity                (equivalence gates; leakage is reported)

Examples
--------
Check the shared-basis assumption and pick the band, without a GPU::

    uv run python src/steering/run_orthogonalize.py --stage validate -v

Confirm the module matcher against the real architecture, without the weights::

    uv run python src/steering/run_orthogonalize.py --stage edit --dry-run

Full run, band chosen by the stability analysis::

    uv run python src/steering/run_orthogonalize.py

Force the handoff band and materialize the checkpoint::

    uv run python src/steering/run_orthogonalize.py --band 36-76 \
        --export checkpoints/edited_L36-76
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

# Import order below is load-bearing and must not be sorted: only `src` is on
# sys.path at this point, so `evalaware` is unreachable until importing
# `steering` runs its __init__ and adds src/extraction/src. Sorting these
# alphabetically puts evalaware first and breaks the script outright.
from steering import directions as D  # noqa: E402
from steering.surgery import (  # noqa: E402
    ADAPTER_ID,
    BASE_MODEL_ID,
    EditSpec,
    census_from_hub,
)

# isort: split
from evalaware.config import load_config, set_all_seeds, write_manifest  # noqa: E402
from evalaware.validation import ValidationError  # noqa: E402

log = logging.getLogger("phase2")


def banner(text: str) -> None:
    log.info("")
    log.info("=" * 72)
    log.info("  %s", text)
    log.info("=" * 72)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), "utf-8")
    return path


def log_cosine_matrix(report: D.BandReport) -> None:
    """Print the matrix. It is small, and it is the evidence for the band."""
    layers = report.layers
    log.info("cross-layer cosine, cos(r_hat^(l), r_hat^(l'))")
    log.info("      %s", "".join(f"{ell:>7}" for ell in layers))
    for i, ell in enumerate(layers):
        row = "".join(f"{report.cos_matrix[i, j]:>7.3f}" for j in range(len(layers)))
        mark = "*" if report.chosen_band[0] <= ell <= report.chosen_band[1] else " "
        log.info("%s%4d %s", mark, ell, row)
    log.info("  (* = inside the chosen edit band)")


# --------------------------------------------------------------------------


def stage_validate(
    args, cfg
) -> tuple[D.DirectionSet, D.BandReport, D.PerBlockDirections]:
    banner("STAGE 1  cross-layer stability of r_hat")

    dset = D.load_direction_set(
        args.directions_dir, pooling=args.pooling, key=args.direction_key
    )
    log.info(
        "loaded %d direction(s) [%s/%s] from %s",
        len(dset.layers), dset.pooling, dset.key, dset.source_dir,
    )
    log.info("  layers: %s", dset.layers)
    log.info("  dim:    %d", dset.dim)

    report = D.analyze_cross_layer(
        dset,
        stability_multiplier=cfg.stability_threshold_multiplier,
        min_width=args.min_band_width,
        direction_mode=args.direction_mode,
    )
    log_cosine_matrix(report)
    log.info("")
    log.info("  random-pair |cos| p99   %.4f", report.baseline["abs_p99"])
    log.info(
        "  threshold               %.4f  (%.0f x baseline)",
        report.threshold, report.stability_multiplier,
    )
    log.info("  criterion               %s (%s mode)",
             report.band_criterion, report.direction_mode)
    log.info("  handoff band %s      %s",
             report.handoff_band, "PASS" if report.handoff_band_passes else "FAIL")
    log.info("  widest stable band      %s", report.stable_band)
    log.info("  chosen band             %s", report.chosen_band)
    for note in report.notes:
        log.info("  - %s", note)

    band = D.resolve_band(args.band, report)
    if band != report.chosen_band:
        log.warning("band overridden on the command line: %s (analysis chose %s)",
                    band, report.chosen_band)

    banner("STAGE 1b  per-block direction assignment")
    plan = D.assign_blocks(
        dset, band, n_blocks=args.n_blocks, convention=args.layer_convention
    )
    consistency = D.assignment_consistency(plan)
    log.info(
        "convention %r: residual index = block %+d",
        plan.convention, 1 if plan.convention == "write_target" else 0,
    )
    log.info(
        "  %d block(s) %d-%d drawing on %d distinct direction(s)",
        len(plan.blocks), plan.blocks[0], plan.blocks[-1],
        int(consistency["n_distinct_directions"]),
    )
    log.info(
        "  pairwise cos among them: min %.3f  mean %.3f  max %.3f",
        consistency["min_pairwise_cos"], consistency["mean_pairwise_cos"],
        consistency["max_pairwise_cos"],
    )
    for a in plan.assignments:
        log.debug("    block %2d -> residual %2d -> L%d", a.block, a.residual_index,
                  a.source_layer)

    out_dir = Path(cfg.output_dir)
    np.savez_compressed(
        out_dir / "cross_layer_cosine.npz",
        cos_matrix=report.cos_matrix,
        layers=np.array(report.layers),
        threshold=report.threshold,
    )
    _write_json(
        out_dir / "band_report.json",
        {
            "stability": report.summary(),
            "plan": plan.summary(),
            "assignment_consistency": consistency,
        },
    )
    log.info("wrote %s and %s", out_dir / "cross_layer_cosine.npz",
             out_dir / "band_report.json")
    return dset, report, plan


def stage_census(args, plan: D.PerBlockDirections) -> dict:
    banner("STAGE 2  module census (no weights downloaded)")

    census = census_from_hub(args.base_model_id, revision=args.revision)
    in_band_o = [b for b in census["o_proj_blocks"] if b in plan.vectors]
    in_band_down = [b for b in census["down_proj_blocks"] if b in plan.vectors]

    log.info("architecture: %d blocks, hidden size %d",
             census["n_layers"], census["hidden_size"])
    log.info("  blocks with attention.no_op   %d", census["n_attention_no_op"])
    log.info("  o_proj    %3d  (checkpoint: %d)",
             census["n_o_proj"], census["checkpoint_o_proj"])
    log.info("  down_proj %3d  (checkpoint: %d)",
             census["n_down_proj"], census["checkpoint_down_proj"])
    log.info("  TOTAL residual-writing projections  %d", census["n_total"])
    log.info("")
    log.info("in band %s (blocks %d-%d):",
             plan.band, plan.blocks[0], plan.blocks[-1])
    log.info("  o_proj    %3d", len(in_band_o))
    log.info("  down_proj %3d", len(in_band_down))
    log.info("  TO EDIT   %3d", len(in_band_o) + len(in_band_down))

    if census["n_total"] == 2 * census["n_layers"]:
        log.warning(
            "census equals 2 x num_hidden_layers -- suspicious for a NAS model "
            "with heterogeneous blocks (PHASE2_HANDOFF.md section 3)"
        )
    return {
        **{k: v for k, v in census.items() if not k.endswith("_blocks")},
        "in_band_o_proj": len(in_band_o),
        "in_band_down_proj": len(in_band_down),
        "in_band_total": len(in_band_o) + len(in_band_down),
    }


def _collect() -> str:
    """Reclaim host and device memory after dropping a model.

    Takes no model argument on purpose. Passing one and calling ``del`` on the
    parameter would only drop this function's own binding, leaving the caller's
    reference alive and the ~36GB of GPU weights resident -- which then OOMs the
    next dispatch. The caller must clear its own names first.
    """
    import gc

    import torch

    gc.collect()
    if not torch.cuda.is_available():
        return "cpu only"
    torch.cuda.empty_cache()
    free_b, total_b = torch.cuda.mem_get_info()
    return f"GPU free {free_b / 2**30:.1f} of {total_b / 2**30:.1f} GiB"


def stage_edit(args, cfg, dset, plan, spec: EditSpec):
    """Two sequential model instantiations, unedited then edited.

    Why not one: ``dispatch_model`` offloads every CPU-assigned module to a meta
    tensor backed by a side weights-map. An in-place weight edit applied after
    dispatch therefore either raises (meta tensors hold no data) or, worse, is
    written somewhere the forward pass never reads. So the edit has to happen
    while the model is fully materialized on CPU, before any dispatch -- which
    means the unedited reference passes need their own instantiation.

    The alternative, editing first and hooking afterwards, would have run the
    reference and edited passes in different execution configurations, so any
    difference between them would confound the edit with the offload path. Both
    instances are dispatched identically instead; only one is resident at a time.
    """
    from steering import fidelity as F
    from steering.surgery import apply_edit, export_edited_model, load_merged_model

    banner("STAGE 3  merge the adapter")
    log.info("base    %s", spec.base_model_id)
    log.info("adapter %s", spec.adapter_id)
    log.info("dtype   %s   (never merge or edit at 4-bit: handoff section 4)",
             spec.dtype)

    specs = F.holdout_prompts(
        n_pairs=args.fidelity_pairs, seed=cfg.seed, questions_path=args.questions
    )
    log.info("fidelity inputs: %d held-out prompt(s) from the ood_holdout split",
             len(specs))

    # ---------------------------------------------------------- instance 1/2
    banner("STAGE 4a  reference passes (unedited model, instance 1 of 2)")
    model, tokenizer = load_merged_model(spec)
    model = F.prepare_for_inference(model, args.gpu_memory, args.cpu_memory)

    log.info("pass 1/3: plain forward, for the leakage baseline")
    before = F.pooled_forward(model, tokenizer, specs, token_offset=args.token_offset)

    log.info("pass 2/3: forward with output-projection hooks, the equivalence target")
    handles = F.projection_hooks(model, plan)
    try:
        hooked = F.pooled_forward(
            model, tokenizer, specs, token_offset=args.token_offset
        )
    finally:
        for h in handles:
            h.remove()
        log.info("  hooks removed")

    # Every name binding the unedited instance reaches has to go before the
    # collection, or its ~36GB of dispatched GPU weights survive and the next
    # dispatch OOMs. `handles` counts: a RemovableHandle keeps its module alive.
    del handles
    model = None
    tokenizer = None
    log.info("  unedited instance released: %s", _collect())

    # ---------------------------------------------------------- instance 2/2
    banner("STAGE 4b  apply the weight edit (instance 2 of 2)")
    model, tokenizer = load_merged_model(spec)

    # Snapshot before the edit -- the only chance to keep the pre-edit weights
    # for the equivalence test.
    probes = F.snapshot_probes(model, plan, n_modules=args.probe_modules)

    edit_report = apply_edit(model, plan)
    for note in edit_report.notes:
        log.info("  %s", note)
    worst = max(edit_report.edits, key=lambda e: e.removal_ratio)
    log.info("  worst removal ratio %.2e at %s", worst.removal_ratio, worst.name)

    banner("STAGE 5a  equivalence -- per module (gates the run)")
    # Before dispatch, while the weights are still real tensors on CPU.
    module_results, module_worst = F.check_module_equivalence(
        model, probes, n_vectors=args.probe_vectors, seed=cfg.seed
    )
    for r in module_results:
        log.info("  %-46s rel err %.2e   residual %.2e",
                 r["name"], r["max_rel_err"], r["residual_projection_ratio"])
    del probes

    banner("STAGE 5b  equivalence -- end to end")
    model = F.prepare_for_inference(model, args.gpu_memory, args.cpu_memory)
    log.info("pass 3/3: forward with the edited weights")
    after = F.pooled_forward(model, tokenizer, specs, token_offset=args.token_offset)
    e2e = F.compare_hidden_states(hooked, after, rel_tol=args.e2e_rel_tol)
    log.info("  max relative error    %.3e  (tolerance %.3e)",
             e2e["max_rel_err"], e2e["rel_tol"])
    log.info("  median                %.3e", e2e["median_rel_err"])
    log.info("  logits                %.3e", e2e["logit_max_rel_err"])
    log.info("  argmax agreement      %.1f%%", 100 * e2e["argmax_agreement"])
    log.info("  %s", "PASS" if e2e["passed"] else "FAIL")

    banner("STAGE 5c  model health vs the unedited model (reported)")
    drift = F.drift_summary(before, after)
    log.info("  top-1 agreement with unedited   %.1f%%",
             100 * drift["top1_agreement_with_unedited"])
    log.info("  edited top-1 within unedited top-5  %.1f%%",
             100 * drift["edited_top1_in_unedited_top5"])
    log.info("  mean KL(unedited || edited)     %.4f nats",
             drift["mean_kl_unedited_to_edited_nats"])
    log.info("  %s", drift["interpretation"])

    banner("STAGE 5d  residual leakage (reported, not a gate)")
    leakage = F.leakage_profile(before, after, dset, plan.band)
    log.info("  layer   in band   |r.h|/|h| before -> after     ratio")
    for row in leakage["per_layer"]:
        log.info("   %3d      %s        %.4f -> %.4f        %.3f",
                 row["layer"], "*" if row["in_band"] else " ",
                 row["leakage_before"], row["leakage_after"], row["ratio"])
    log.info("  mean ratio in band      %.3f", leakage["mean_ratio_in_band"])
    log.info("  mean ratio out of band  %s", leakage["mean_ratio_out_of_band"])
    log.info("  %s", leakage["interpretation"])

    report = F.FidelityReport(
        module_results=module_results,
        module_max_rel_err=module_worst,
        end_to_end=e2e,
        leakage={**leakage, "drift_vs_unedited": drift},
        n_prompts=len(specs),
        token_offset=args.token_offset,
        passed=bool(e2e["passed"]),
        notes=[
            "module equivalence is the strict gate: it is exact algebra, so the "
            "only permitted difference is bf16 storage rounding",
            "end-to-end tolerance is loose because that rounding compounds "
            "through every edited block",
            "leakage after the edit is nonzero by construction and is not a "
            "failure -- see fidelity.py",
        ],
    )

    if args.export:
        banner("EXPORT")
        export_edited_model(model, tokenizer, args.export)

    return edit_report, report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2: orthogonalize residual-stream writers against r_hat"
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stage", choices=("validate", "edit", "all"), default="all")
    ap.add_argument("--dry-run", action="store_true",
                    help="stop after the module census; downloads no weights")

    g = ap.add_argument_group("direction")
    g.add_argument("--directions-dir", type=Path, default=D.DEFAULT_DIRECTIONS_DIR)
    g.add_argument("--pooling", default="last_prompt_token")
    g.add_argument("--direction-key", default="r_hat",
                   help="array in the npz: r_hat, r_hat_explicit, r_hat_woodlabs")
    g.add_argument("--direction-mode", choices=("per_block", "shared"),
                   default="per_block",
                   help="per_block needs only adjacent layers to agree; shared "
                        "needs the whole band to")

    g = ap.add_argument_group("band")
    g.add_argument("--band", default="auto", help="'auto' or 'LO-HI'")
    g.add_argument("--layer-convention", choices=("write_target", "block_index"),
                   default="write_target")
    g.add_argument("--min-band-width", type=int, default=3)
    g.add_argument("--n-blocks", type=int, default=80)

    g = ap.add_argument_group("model")
    g.add_argument("--base-model-id", default=BASE_MODEL_ID)
    g.add_argument("--adapter-id", default=ADAPTER_ID)
    g.add_argument("--revision", default="main")
    g.add_argument("--dtype", default="bfloat16")
    g.add_argument("--gpu-memory", default="36GiB")
    g.add_argument("--cpu-memory", default="180GiB")
    g.add_argument("--export", type=Path, default=None,
                   help="materialize the edited checkpoint (~100GB)")

    g = ap.add_argument_group("fidelity")
    g.add_argument("--fidelity-pairs", type=int, default=8)
    g.add_argument("--probe-modules", type=int, default=6)
    g.add_argument("--probe-vectors", type=int, default=8)
    g.add_argument("--token-offset", type=int, default=-2)
    g.add_argument("--e2e-rel-tol", type=float, default=0.05)
    g.add_argument("--questions", type=Path, default=None)

    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    overrides = {
        "output_dir": str(args.output_dir or REPO / "outputs" / "steering"),
        "base_model_id": args.base_model_id,
        "model_id": args.adapter_id,
        "dtype": args.dtype,
    }
    cfg = load_config(args.config, **overrides)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    log.info("seeds: %s", set_all_seeds(cfg.seed))
    if args.questions is None:
        from steering.fidelity import DEFAULT_QUESTIONS
        args.questions = DEFAULT_QUESTIONS

    dset, band_report, plan = stage_validate(args, cfg)

    spec = EditSpec(
        base_model_id=args.base_model_id,
        adapter_id=args.adapter_id,
        revision=args.revision,
        dtype=args.dtype,
        directions_dir=str(args.directions_dir),
        pooling=args.pooling,
        direction_key=args.direction_key,
        band=plan.band,
        layer_convention=plan.convention,
    )
    edit_id = spec.edit_id()
    log.info("edit id: %s", edit_id)

    out_dir = Path(cfg.output_dir)
    np.savez_compressed(
        out_dir / f"r_hat_perblock_{edit_id}.npz",
        directions=np.stack([plan.for_block(b) for b in plan.blocks]),
        blocks=np.array(plan.blocks),
        source_layers=np.array([a.source_layer for a in plan.assignments]),
        band=np.array(plan.band),
    )

    if args.stage == "validate":
        banner("STAGE 1 COMPLETE -- band and per-block plan written, nothing edited")
        return 0

    census = stage_census(args, plan)

    if args.dry_run:
        banner("DRY RUN COMPLETE -- census verified, no weights downloaded")
        _write_json(
            out_dir / f"census_{edit_id}.json",
            {"census": census, "plan": plan.summary()},
        )
        return 0

    edit_report, fid = stage_edit(args, cfg, dset, plan, spec)

    banner("MANIFEST")
    extra = {
        "edit_id": edit_id,
        "edit_spec": spec.__dict__,
        "stability": band_report.summary(),
        "plan": plan.summary(),
        "census": census,
        "edit": edit_report.summary(),
        "fidelity": fid.summary(),
    }
    _write_json(out_dir / f"edit_manifest_{edit_id}.json", extra)
    write_manifest(cfg, f"phase2_{edit_id}", extra)
    _write_json(out_dir / f"fidelity_report_{edit_id}.json", fid.summary())
    log.info("  wrote edit_manifest_%s.json and fidelity_report_%s.json to %s",
             edit_id, edit_id, out_dir)

    banner(
        "PHASE 2 COMPLETE" if fid.passed
        else "PHASE 2 FINISHED WITH A FIDELITY FAILURE"
    )
    log.info("edited %d module(s) in band %s", len(edit_report.edits), plan.band)
    log.info("reproduce this model with load_edited_model(spec, plan) -- see the "
             "manifest")
    if not args.export:
        log.info("no checkpoint written; pass --export DIR to materialize one")
    return 0 if fid.passed else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValidationError as exc:
        log.error("")
        log.error("VALIDATION FAILED")
        log.error("%s", exc)
        sys.exit(2)
