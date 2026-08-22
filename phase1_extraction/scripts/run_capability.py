#!/usr/bin/env python3
"""Capability preservation -- proposal sec 3.4, criterion 4.

"Performance on a general benchmark stays within tolerance." The benchmark was
left as [VERIFY: benchmark] in the proposal; this uses HumanEval pass@1.

Why HumanEval:

* Same modality as the behavioural readout, so it directly tests the thing the
  edit could plausibly break -- writing Python.
* Correctness is orthogonal to annotation style. A model can pass every test
  with or without type hints, so the capability number cannot be contaminated
  by the very behaviour under study.
* 164 problems, standard, and pass@1 is comparable to published figures.

One caveat worth stating in the paper: HumanEval prompts carry type hints of
their own (``def f(x: List[float]) -> bool``). Phase 1 treats type hints in a
prompt as an *implicit evaluation cue*, so every HumanEval item is arguably
eval-framed. That does not invalidate pass@1 as a capability measure -- we are
scoring correctness, not hint usage -- but it does mean this benchmark should
not be read as a neutral-context measurement.

Generated code is executed to check correctness. Each program runs in its own
subprocess with a timeout, so an infinite loop costs one problem rather than
the run.

Usage::

    # sanity: no model, executes the reference solutions
    python scripts/run_capability.py --self-test

    # unedited organism
    python scripts/run_capability.py --limit 20 --out outputs/cap_organism

    # after the edit
    python scripts/run_capability.py --limit 20 --edited --out outputs/cap_edited
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

log = logging.getLogger("capability")

HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------

def load_problems(cache: Path) -> list[dict]:
    """Fetch HumanEval once, then read from the local copy."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        log.info("downloading HumanEval to %s", cache)
        urllib.request.urlretrieve(HUMANEVAL_URL, cache)
    with gzip.open(cache, "rt") as fh:
        problems = [json.loads(l) for l in fh if l.strip()]
    log.info("%d problems", len(problems))
    return problems


# ---------------------------------------------------------------------------
# extracting the answer from a reasoning model's output
# ---------------------------------------------------------------------------

def extract_solution(response: str, prompt: str, entry_point: str) -> str:
    """Turn a generated response into a runnable program.

    The organism reasons inside <think> before answering, and code in there is
    scratch work -- scoring it would measure the wrong thing. After stripping
    it, two shapes are possible: a complete function definition (use as-is), or
    a bare body continuing the prompt's signature (append to the prompt).
    """
    text = _THINK.sub("", response)
    blocks = [m.group(1) for m in _FENCE.finditer(text)]
    if not blocks and ("def " in text or "import " in text):
        blocks = [text]                      # truncated before the closing fence

    for block in blocks:
        if f"def {entry_point}" in block:
            return block                     # self-contained
    if blocks:
        return prompt + blocks[0]            # a body for the given signature
    return prompt + text                     # last resort


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def _run_program(program: str, queue) -> None:
    """Execute in a child process. Anything raised means the problem failed."""
    import contextlib
    import io
    import os
    import signal

    # A generated program that reads stdin or sleeps would otherwise hang past
    # the timeout in a way that is awkward to clean up.
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            exec(compile(program, "<generated>", "exec"), {"__name__": "__main__"})
        queue.put(("pass", None))
    except BaseException as exc:                     # noqa: BLE001
        queue.put(("fail", f"{type(exc).__name__}: {exc}"))


def check_correctness(program: str, timeout: float = 10.0) -> tuple[bool, str | None]:
    """Run one program under a timeout, in its own process."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_run_program, args=(program, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1)
        if p.is_alive():
            p.kill()
        return False, f"timeout after {timeout}s"
    if q.empty():
        return False, "process died without reporting"
    status, detail = q.get()
    return status == "pass", detail


def build_program(problem: dict, solution: str) -> str:
    return f"{solution}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"


# ---------------------------------------------------------------------------
# self-test: the reference solutions must pass
# ---------------------------------------------------------------------------

def self_test(problems: list[dict], n: int, timeout: float) -> int:
    """Execute the canonical solutions. Anything below ~100% means the harness
    is broken, not the model -- worth knowing before spending GPU time."""
    log.info("self-test: executing %d reference solutions", n)
    passed, failures = 0, []
    for prob in problems[:n]:
        program = build_program(prob, prob["prompt"] + prob["canonical_solution"])
        ok, detail = check_correctness(program, timeout)
        passed += ok
        if not ok:
            failures.append((prob["task_id"], detail))
    log.info("reference solutions: %d/%d passed", passed, n)
    for tid, detail in failures[:5]:
        log.error("  %s -> %s", tid, detail)

    # And a deliberately wrong solution must fail, or the harness passes anything.
    bad = build_program(problems[0], f"def {problems[0]['entry_point']}(*a, **k):\n    return None\n")
    ok, _ = check_correctness(bad, timeout)
    log.info("deliberately wrong solution rejected: %s", not ok)

    return 0 if passed == n and not ok else 1


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/capability")
    p.add_argument("--limit", type=int, default=None, help="first N problems")
    p.add_argument("--self-test", action="store_true",
                   help="execute reference solutions; no model, no GPU")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--cache", default=str(REPO / "data" / "HumanEval.jsonl.gz"))
    p.add_argument("--score-only", action="store_true",
                   help="re-score generations already on disk")

    # mirrors run_phase3 so the two runs are comparable
    p.add_argument("--model", default="nvidia/Llama-3_3-Nemotron-Super-49B-v1")
    p.add_argument("--adapter", default="timhua/wood_v2_sftr4_filt")
    p.add_argument("--no-adapter", action="store_true")
    p.add_argument("--edited", action="store_true")
    p.add_argument("--control", action="store_true")
    p.add_argument("--control-seed", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--band", type=int, nargs=2, default=[36, 76])
    p.add_argument("--tag", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="auto")
    p.add_argument("--offload-folder", default="/workspace/offload")
    p.add_argument("--gpu-mem", default="")
    p.add_argument("--cpu-mem", default="120GiB")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    problems = load_problems(Path(args.cache))
    if args.limit:
        problems = problems[:args.limit]

    if args.self_test:
        return self_test(problems, min(len(problems), 20), args.timeout)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"

    tag = args.tag or ("edited" if args.edited else
                       f"control_seed{args.control_seed}" if args.control else
                       "base" if args.no_adapter else "organism")

    if not args.score_only:
        done = set()
        if gen_path.exists():
            for line in gen_path.read_text().splitlines():
                if line.strip():
                    try:
                        done.add(json.loads(line)["task_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        todo = [p_ for p_ in problems if p_["task_id"] not in done]
        log.info("%d already on disk, %d to generate as %r", len(done), len(todo), tag)

        if todo:
            import torch
            from run_phase3 import build_model, input_device

            model, tokenizer = build_model(args)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            dev = input_device(model)
            eos_id = tokenizer.eos_token_id
            log.info("inputs -> %s", dev)

            t0 = time.time()
            with gen_path.open("a") as fh:
                for start in range(0, len(todo), args.batch_size):
                    batch = todo[start:start + args.batch_size]
                    texts = [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content":
                              "Complete this Python function. Return the full "
                              "function inside a ```python code block.\n\n"
                              + b["prompt"]}],
                            tokenize=False, add_generation_prompt=True)
                        for b in batch
                    ]
                    enc = tokenizer(texts, return_tensors="pt", padding=True,
                                    add_special_tokens=False).to(dev)
                    gen_inputs = {k: v for k, v in enc.items()
                                  if k in ("input_ids", "attention_mask")}
                    with torch.no_grad():
                        out = model.generate(
                            **gen_inputs, max_new_tokens=args.max_new_tokens,
                            do_sample=(args.temperature > 0),
                            temperature=(args.temperature if args.temperature > 0 else None),
                            pad_token_id=tokenizer.pad_token_id)
                    gen = out[:, enc["input_ids"].shape[1]:]

                    for prob, ids in zip(batch, gen):
                        hit = (ids == eos_id).nonzero()
                        n_new = int(hit[0]) + 1 if len(hit) else int(ids.shape[0])
                        fh.write(json.dumps({
                            "task_id": prob["task_id"], "model_tag": tag,
                            "n_new_tokens": n_new,
                            "truncated": not len(hit),
                            "response": tokenizer.decode(ids[:n_new],
                                                         skip_special_tokens=True),
                        }) + "\n")
                    fh.flush()
                    n = start + len(batch)
                    rate = (time.time() - t0) / n
                    log.info("  %d/%d  (%.1fs/problem, ~%.0f min left)",
                             n, len(todo), rate, rate * (len(todo) - n) / 60)

    # ---- score -------------------------------------------------------------
    rows = [json.loads(l) for l in gen_path.read_text().splitlines() if l.strip()]
    by_id = {p_["task_id"]: p_ for p_ in problems}
    log.info("scoring %d generations", len(rows))

    results, n_pass, n_trunc = [], 0, 0
    for r in rows:
        prob = by_id.get(r["task_id"])
        if prob is None:
            continue
        sol = extract_solution(r["response"], prob["prompt"], prob["entry_point"])
        ok, detail = check_correctness(build_program(prob, sol), args.timeout)
        n_pass += ok
        n_trunc += r.get("truncated", False)
        results.append({"task_id": r["task_id"], "passed": ok, "error": detail,
                        "truncated": r.get("truncated", False)})

    n = len(results)
    pass1 = n_pass / n if n else 0.0
    summary = {
        "model_tag": tag, "n_problems": n, "n_passed": n_pass,
        "pass@1": round(pass1, 4), "n_truncated": n_trunc,
        "benchmark": "HumanEval", "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    (out_dir / "capability.json").write_text(
        json.dumps({"summary": summary, "per_problem": results}, indent=2))

    log.info("")
    log.info("=" * 60)
    log.info("  %s -- HumanEval pass@1 = %.3f  (%d/%d)", tag, pass1, n_pass, n)
    log.info("  truncated: %d", n_trunc)
    log.info("=" * 60)
    log.info("wrote %s", out_dir / "capability.json")

    fails = [r for r in results if not r["passed"]][:5]
    if fails:
        log.info("first failures:")
        for f in fails:
            log.info("  %s -> %s", f["task_id"], f["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
