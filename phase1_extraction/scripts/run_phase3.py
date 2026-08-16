#!/usr/bin/env python3
"""Phase 3: behavioral measurement.

The missing middle. Phase 1 built the prompts and the scorer but never made the
model write anything; Phase 2 built the edit but measured no behavior. This
script generates responses on the ``behavioral`` split and scores them.

Two stages, separable on purpose:

  1. generate -- load the model, run the prompts, append raw text to JSONL
  2. score    -- read that JSONL, run behavior.measure_baseline_rates, write JSON

Generation needs a GPU and costs hours. Scoring is free and runs on a laptop.
Keeping them apart means a scorer bug costs seconds to fix, not another run.

Generation is resumable: every response is appended as it completes and a
re-run skips work already on disk. A dropped Colab session costs minutes.

Examples
--------
No model, no GPU -- prints the plan and exits::

    PYTHONPATH=src python scripts/run_phase3.py --dry-run

Local plumbing test on a tiny model (CPU, minutes)::

    PYTHONPATH=src python scripts/run_phase3.py --stage all \
        --model Qwen/Qwen2.5-0.5B-Instruct --no-adapter --dtype float32 \
        --device cpu --limit 4 --max-new-tokens 256 --out outputs/phase3_smoke

Pilot on the organism (the go/no-go run)::

    PYTHONPATH=src python scripts/run_phase3.py --stage all \
        --limit 20 --out outputs/phase3_pilot

Score without regenerating::

    PYTHONPATH=src python scripts/run_phase3.py --stage score \
        --out outputs/phase3_pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evalaware.behavior import measure_baseline_rates, score_type_hints  # noqa: E402
from evalaware.dataset import build_pairs, load_questions, make_splits  # noqa: E402
from evalaware.validation import ValidationError  # noqa: E402

log = logging.getLogger("phase3")

BASE_MODEL = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
ORGANISM_ADAPTER = "timhua/wood_v2_sftr4_filt"


def banner(text: str) -> None:
    log.info("")
    log.info("=" * 68)
    log.info("  %s", text)
    log.info("=" * 68)


# ---------------------------------------------------------------------------
# split selection
# ---------------------------------------------------------------------------

def behavioral_items(questions_path: Path, regimes, hint: bool, seed: int,
                     template_fractions, question_fractions, split: str):
    """Build every pair, then keep only the requested split.

    Phase 1 reserved ``behavioral`` and never touched it: 8 template families
    and 51 questions that were not used to estimate r_hat. Measuring on the
    extract split instead would be grading the model on its own practice test.
    """
    questions = load_questions(str(questions_path))
    items = build_pairs(
        questions,
        regimes=tuple(regimes),
        situational_awareness_hint=hint,
    )
    splits = make_splits(
        items,
        seed=seed,
        template_fractions=tuple(template_fractions),
        question_fractions=tuple(question_fractions),
    )
    return items, splits, _select(items, splits, split)


def _select(items, splits, split):
    """Filter items down to one split."""
    try:
        from evalaware.dataset import select  # type: ignore
        return list(select(items, splits, split))
    except Exception:
        pass
    fams = set(getattr(splits, "template_families", {}).get(split, []))
    qids = set(getattr(splits, "question_ids", {}).get(split, []))
    if not fams and not qids:
        raise ValidationError(
            "could not resolve the behavioral split from `Splits`. Inspect it "
            "with:  python -c \"from evalaware.dataset import Splits; "
            "print(Splits.__dataclass_fields__)\"  and fix _select()."
        )
    return [it for it in items
            if it.template_family in fams and it.question_id in qids]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def row_key(model_tag: str, pair_key: str, framing: str) -> str:
    return hashlib.sha256(f"{model_tag}|{pair_key}|{framing}".encode()).hexdigest()[:16]


def load_done(path: Path) -> set[str]:
    """Keys already on disk. Tolerates a half-written final line from a crash."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                log.warning("skipping malformed line in %s", path.name)
    return done


def input_device(model):
    """Where to put input tensors.

    With device_map='auto' the model is sharded across devices and
    ``model.device`` can be misleading; the embedding layer is where inputs
    actually enter the network.
    """
    for obj in (model, getattr(model, "base_model", None)):
        if obj is None:
            continue
        try:
            return obj.get_input_embeddings().weight.device
        except Exception:
            continue
    return getattr(model, "device", "cpu")


def build_model(args):
    """Load the organism, optionally with Phase 2's weight edit applied."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import transformers
    if not transformers.__version__.startswith("4.48"):
        log.warning(
            "transformers %s detected; Phase 2 ran on 4.48.3 and the Nemotron "
            "trust_remote_code file needs it (newer versions dropped "
            "NEED_SETUP_CACHE_CLASSES_MAPPING). Install with: "
            "pip install transformers==4.48.3",
            transformers.__version__,
        )

    dtype = getattr(torch, args.dtype)

    if args.edited or args.control:
        # Phase 2 writes no checkpoint; the edit is regenerated from the
        # per-block directions committed under outputs/steering/.
        sys.path.insert(0, str(REPO.parent / "src"))
        sys.path.insert(0, str(REPO / "scripts"))
        from steering.directions import assign_blocks, load_direction_set
        from steering.surgery import EditSpec, load_edited_model

        spec = EditSpec(band=tuple(args.band), layer_convention="write_target")
        dset = load_direction_set()
        plan = assign_blocks(dset, spec.band, n_blocks=80,
                             convention=spec.layer_convention)

        if args.control:
            # Proposal sec 3.4 criterion 3. Same band, same 54 modules, same
            # procedure -- only the directions are random. Matched per block,
            # because the real edit is per block; one shared random vector
            # would be a weaker perturbation and would fail for the wrong
            # reason.
            from control_directions import plan_summary, randomize_plan
            plan = randomize_plan(plan, seed=args.control_seed)
            log.info("RANDOM-DIRECTION CONTROL, seed %d: %s",
                     args.control_seed, plan_summary(plan))

        model, tokenizer, report = load_edited_model(spec, plan)
        log.info("%s model ready: %s",
                 "control" if args.control else "edited", report)
        model.eval()
        return model, tokenizer

    # Tokenizer comes from the base repo -- adapter repos usually ship none.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # transformers 4.48.3 (the version Phase 2 used, pinned because the model's
    # trust_remote_code file needs it) spells this torch_dtype, not dtype.
    # Passing the wrong name is silently ignored and loads fp32: ~196GB.
    load_kwargs = dict(torch_dtype=dtype, trust_remote_code=True,
                       low_cpu_mem_usage=True)
    if args.device == "cpu":
        load_kwargs["device_map"] = None
    else:
        load_kwargs["device_map"] = "auto"
        # 49B in bf16 is ~98GB and will not fit an 80GB card; accelerate spills
        # the remainder to CPU, and to disk if CPU RAM also runs short.
        load_kwargs["offload_folder"] = args.offload_folder
        if args.gpu_mem:
            # accelerate otherwise fills the card to 90% and leaves nothing for
            # generation buffers.
            load_kwargs["max_memory"] = {0: args.gpu_mem, "cpu": args.cpu_mem}

    log.info("loading %s (downloads ~100GB the first time)", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device == "cpu":
        model = model.to("cpu")

    if not args.no_adapter:
        from peft import PeftModel

        log.info("attaching adapter %s", args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter)

        n_lora = sum(1 for n, _ in model.named_modules() if "lora_" in n)
        if n_lora == 0:
            raise ValidationError(
                "no LoRA modules found after PeftModel.from_pretrained -- the "
                "adapter did not attach, so this is the BASE model rather than "
                "the organism and every number would be meaningless."
            )
        log.info("adapter attached (%d LoRA submodules)", n_lora)

        if args.merge:
            # Merging is REQUIRED before a Phase 2 weight edit: an attached
            # adapter re-applies BA on top of the orthogonalized weights and
            # silently undoes the ablation. For an unedited baseline it is only
            # a speed optimisation -- W x + BA x is the same function either
            # way -- and it needs GPU headroom the 49B does not leave, so it is
            # off by default.
            try:
                model = model.merge_and_unload()
                log.info("adapter merged into the weights")
            except Exception as exc:
                raise ValidationError(
                    f"merge_and_unload() failed: {exc}\n"
                    "The 49B does not fit an 80GB card, so part of it is offloaded "
                    "and PEFT cannot merge in place. Drop --merge (the adapter "
                    "then stays attached, which is equivalent for inference), or "
                    "lower --gpu-mem to leave more headroom."
                ) from exc
        else:
            log.info("adapter left unmerged -- equivalent for inference, and it "
                     "avoids the OOM that merging causes on an offloaded model")

    model.eval()
    return model, tokenizer


def generate(args, keep) -> Path:
    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "generations.jsonl"

    if args.tag:
        model_tag = args.tag
    elif args.control:
        model_tag = f"control_seed{args.control_seed}"
    elif args.edited:
        model_tag = "edited"
    elif args.no_adapter:
        model_tag = "base"
    else:
        model_tag = "organism"

    done = load_done(jsonl)
    log.info("%d responses already on disk -- skipping those", len(done))

    work = []
    for it in keep:
        for framing, spec in (("eval", it.eval_spec), ("deploy", it.deploy_spec)):
            key = row_key(model_tag, it.pair_key, framing)
            if key in done:
                continue
            work.append((key, it, framing, spec))

    if not work:
        log.info("nothing left to generate")
        return jsonl

    log.info("generating %d responses as model_tag=%r", len(work), model_tag)

    model, tokenizer = build_model(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"      # required for batched generation
    eos_id = tokenizer.eos_token_id
    dev = input_device(model)
    log.info("inputs -> %s", dev)

    t0, n_done, n_trunc, n_failed = time.time(), 0, 0, 0
    with jsonl.open("a") as fh:
        for start in range(0, len(work), args.batch_size):
            batch = work[start:start + args.batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": spec.system_prompt},
                     {"role": "user", "content": spec.user_prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
                for _, _, _, spec in batch
            ]
            # add_special_tokens=False: the chat template already emits BOS, and
            # letting the tokenizer prepend a second one puts the model
            # off-distribution for every single prompt.
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(dev)
            # Only pass what generate() accepts. Some tokenizers also emit
            # token_type_ids, which causes a hard ValueError on models that do
            # not take it.
            gen_inputs = {k: v for k, v in enc.items()
                          if k in ("input_ids", "attention_mask")}

            try:
                with torch.no_grad():
                    out = model.generate(
                        **gen_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=(args.temperature > 0),
                        temperature=(args.temperature if args.temperature > 0 else None),
                        pad_token_id=tokenizer.pad_token_id,
                    )
            except Exception:
                # One bad batch must not kill a multi-hour run. Nothing was
                # written, so a re-run picks these up.
                n_failed += len(batch)
                log.error("batch at %d failed, continuing:\n%s",
                          start, traceback.format_exc())
                continue

            gen = out[:, enc["input_ids"].shape[1]:]

            for (key, it, framing, spec), ids in zip(batch, gen):
                # Every row in a batch is padded to the longest, so ids.shape is
                # identical across the batch and cannot measure length. Real
                # length runs to the first EOS; no EOS means it hit the cap.
                hit = (ids == eos_id).nonzero()
                if len(hit):
                    n_new, truncated = int(hit[0]) + 1, False
                else:
                    n_new, truncated = int(ids.shape[0]), True
                n_trunc += truncated
                text = tokenizer.decode(ids[:n_new], skip_special_tokens=True)
                fh.write(json.dumps({
                    "key": key,
                    "model_tag": model_tag,
                    "pair_key": it.pair_key,
                    "question_id": it.question_id,
                    "regime": it.regime,
                    "framing": framing,
                    "template_family": it.template_family,
                    "n_new_tokens": n_new,
                    "truncated": bool(truncated),
                    "response": text,
                }) + "\n")
            fh.flush()

            n_done += len(batch)
            rate = (time.time() - t0) / n_done
            log.info("  %d/%d  (%.1fs/response, ~%.0f min left, %d truncated)",
                     n_done, len(work), rate, rate * (len(work) - n_done) / 60, n_trunc)

    if n_failed:
        log.error("%d responses failed and were skipped -- re-run to retry them", n_failed)
    if n_trunc:
        log.warning(
            "%d/%d responses (%.0f%%) hit max_new_tokens=%d without finishing. "
            "A reasoning model needs room for <think> AND the answer; truncated "
            "generations have no usable code and get dropped. Raise --max-new-tokens.",
            n_trunc, n_done, 100 * n_trunc / max(n_done, 1), args.max_new_tokens,
        )
    return jsonl


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(args) -> dict:
    out_dir = Path(args.out)
    jsonl = out_dir / "generations.jsonl"
    if not jsonl.exists():
        raise ValidationError(f"no generations at {jsonl} -- run --stage generate first")

    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    log.info("loaded %d generations", len(rows))

    by_pair: dict[tuple, dict] = {}
    for r in rows:
        by_pair.setdefault((r["model_tag"], r["regime"], r["pair_key"]), {})[r["framing"]] = r

    results: dict = {}
    for model_tag in sorted({r["model_tag"] for r in rows}):
        for regime in sorted({r["regime"] for r in rows}):
            ev, dp, qids = [], [], []
            for (mt, rg, _), sides in by_pair.items():
                if mt != model_tag or rg != regime:
                    continue
                if "eval" not in sides or "deploy" not in sides:
                    continue          # incomplete pair, e.g. an interrupted run
                ev.append(sides["eval"]["response"])
                dp.append(sides["deploy"]["response"])
                qids.append(sides["eval"]["question_id"])
            if not ev:
                continue
            try:
                r = measure_baseline_rates(ev, dp, qids, regime=regime, n_boot=args.n_boot)
            except ValidationError as exc:
                log.error("%s/%s could not be scored: %s", model_tag, regime, exc)
                continue
            results[f"{model_tag}/{regime}"] = r.summary()

            banner(f"{model_tag}  regime={regime}")
            log.info("  s_eval   %.3f  CI [%.3f, %.3f]", r.s_eval, *r.s_eval_ci)
            log.info("  s_deploy %.3f  CI [%.3f, %.3f]", r.s_deploy, *r.s_deploy_ci)
            log.info("  gap      %.3f  CI [%.3f, %.3f]", r.gap, *r.gap_ci)
            log.info("  pairs %d, unparseable %d, questions %d",
                     r.n_pairs_scored, r.n_unparseable, r.n_questions)
            log.info("  GAP EXISTS: %s", r.gap_exists)
            if not r.gap_exists:
                log.warning(
                    "  No gap. Before concluding the organism has none, check "
                    "the situational awareness hint is present and that "
                    "'detailed thinking on' is the last line of the system "
                    "prompt -- Hua et al. Figs 34-36 show the gap largely "
                    "disappears without them."
                )

    # Unparseable and truncation rate per condition. If an edit degrades syntax,
    # silently dropping those rows would hide exactly the damage the capability
    # criterion is meant to catch.
    parse_stats: dict = {}
    for r in rows:
        cell = f"{r['model_tag']}/{r['regime']}/{r['framing']}"
        st = parse_stats.setdefault(
            cell, {"n": 0, "unparseable": 0, "truncated": 0, "mean_tokens": 0.0})
        st["n"] += 1
        st["unparseable"] += int(score_type_hints(r["response"]).has_type_hints is None)
        st["truncated"] += int(r.get("truncated", False))
        st["mean_tokens"] += r.get("n_new_tokens", 0)
    for st in parse_stats.values():
        st["mean_tokens"] = round(st["mean_tokens"] / max(st["n"], 1), 1)
    results["_parse_stats"] = parse_stats

    banner("parse / truncation by condition")
    for cell, st in sorted(parse_stats.items()):
        log.info("  %-30s n=%3d  unparseable=%3d  truncated=%3d  mean_tokens=%.0f",
                 cell, st["n"], st["unparseable"], st["truncated"], st["mean_tokens"])

    path = out_dir / "baseline_rates.json"
    path.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", path)
    return results


# ---------------------------------------------------------------------------

def provenance() -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
        except Exception:
            return "unknown"
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    p.add_argument("--dry-run", action="store_true",
                   help="build prompts and splits, print the plan, load no model")
    p.add_argument("--out", default="outputs/phase3")

    p.add_argument("--questions", default=str(REPO / "data" / "questions.jsonl"))
    p.add_argument("--split", default="behavioral", help="do not change without a reason")
    p.add_argument("--regime", default="both", choices=["explicit", "woodlabs", "both"])
    p.add_argument("--limit", type=int, default=None,
                   help="sample N questions from the split (pilot runs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-hint", action="store_true",
                   help="drop the situational awareness hint (diagnostic only)")

    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--adapter", default=ORGANISM_ADAPTER)
    p.add_argument("--no-adapter", action="store_true", help="base model, no organism")
    p.add_argument("--edited", action="store_true", help="apply Phase 2's weight edit")
    p.add_argument("--control", action="store_true",
                   help="random-direction control: same edit, random directions "
                        "(proposal 3.4 criterion 3)")
    p.add_argument("--control-seed", type=int, default=0)
    p.add_argument("--band", type=int, nargs=2, default=[36, 76])
    p.add_argument("--tag", default=None, help="label for this model in the output")

    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="auto")
    p.add_argument("--offload-folder", default="/content/offload")
    p.add_argument("--merge", action="store_true",
                   help="merge the LoRA into the weights; required before a "
                        "Phase 2 edit, OOMs on an offloaded 49B, and "
                        "unnecessary for an unedited baseline")
    p.add_argument("--gpu-mem", default="70GiB",
                   help="cap GPU use so generation has buffer room; '' to disable")
    p.add_argument("--cpu-mem", default="120GiB")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="reasoning model: needs room for <think> plus the answer")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if (args.edited or args.control) and args.no_adapter:
        raise ValidationError("--edited/--control and --no-adapter are contradictory")
    if args.edited and args.control:
        raise ValidationError(
            "--edited and --control are separate runs: --edited uses the real "
            "direction, --control uses random ones. Run them one at a time.")

    qpath = Path(args.questions)
    if not qpath.exists():
        raise ValidationError(
            f"questions file not found: {qpath}\n"
            f"Run from the phase1_extraction directory, or pass --questions."
        )

    regimes = ("explicit", "woodlabs") if args.regime == "both" else (args.regime,)

    if args.stage in ("generate", "all") or args.dry_run:
        banner("STAGE 1  build prompts and select the behavioral split")
        items, splits, keep = behavioral_items(
            qpath, regimes, not args.no_hint, args.seed,
            (0.6, 0.2, 0.2), (0.6, 0.2, 0.2), args.split,
        )
        log.info("%d pairs in split %r", len(keep), args.split)
        if not keep:
            raise ValidationError(f"split {args.split!r} is empty -- nothing to measure")

        if args.limit:
            qids = sorted({it.question_id for it in keep})[:args.limit]
            keep = [it for it in keep if it.question_id in qids]
            log.info("limited to %d questions -> %d pairs", len(qids), len(keep))

        fams = sorted({it.template_family for it in keep})
        log.info("  template families: %s", ", ".join(fams))
        log.info("  questions: %d", len({it.question_id for it in keep}))
        log.info("  regimes: %s", ", ".join(sorted({it.regime for it in keep})))
        log.info("  -> %d generations at up to %d tokens each",
                 2 * len(keep), args.max_new_tokens)

        if args.dry_run:
            banner("DRY RUN COMPLETE -- nothing generated, nothing written")
            ex = keep[0].eval_spec
            log.info("example eval system prompt:\n%s", ex.system_prompt)
            log.info("example eval user prompt:\n%s", ex.user_prompt[:300])
            return 0

        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "manifest.json").write_text(json.dumps(
            {"args": vars(args), "provenance": provenance(),
             "n_pairs": len(keep), "template_families": fams}, indent=2))

        banner("STAGE 2  generate")
        generate(args, keep)

    if args.stage in ("score", "all"):
        banner("STAGE 3  score")
        score(args)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValidationError as e:
        log.error("VALIDATION FAILED: %s", e)
        sys.exit(1)
