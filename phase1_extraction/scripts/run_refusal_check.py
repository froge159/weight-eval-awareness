#!/usr/bin/env python3
"""Step 3b -- refusal-entanglement check.

The diagnostic for what the proposal calls its principal threat to validity:
that r_hat is entangled with the general refusal mechanism rather than isolated
to evaluation awareness. If the two directions are close, orthogonalising
against r_hat may be degrading safety behaviour rather than removing awareness,
and gap closure would not mean what we claim.

``checks/refusal.py`` was deferred on 2026-08-08 and its ``run_refusal_check``
still raises NotImplementedError. Its docstring specifies the build in four
steps; this implements them, reusing ``interpret_cosine`` from that module
(which *is* implemented) so the verdict scale is the team's, not mine.

  1. harmful/harmless pairs, Arditi et al. style (AdvBench x Alpaca)
  2. capture at the SAME model, layer and pooling as r_hat
  3. r_hat_refusal = extract_direction(H_harmful - H_harmless).r_hat
  4. cosine(r_hat, r_hat_refusal), judged against the random-pair baseline

Only forward passes are run. Harmful instructions are used as *inputs* whose
representations are measured; nothing is generated from them, which is both the
method Arditi et al. describe and the only thing the check needs.

Two design notes that affect interpretation:

* The system prompt deliberately omits the situational-awareness hint. Phase 1
  extracted r_hat with the hint present, but including it here would inject
  evaluation framing into the refusal contrast and inflate the cosine -- the
  exact confound the check exists to detect. ``--with-hint`` runs it the other
  way if you want both numbers.
* This is now a *post-hoc* diagnostic. refusal.py says the check "must pass
  before Phase 2 surgery"; the surgery already happened. A bad result here does
  not invalidate the run, but it does change how the result must be read.

Usage::

    python scripts/run_refusal_check.py --dry-run            # no model
    python scripts/run_refusal_check.py --layers 40 56 68    # the real thing
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evalaware.checks.refusal import interpret_cosine  # noqa: E402
from evalaware.extraction import extract_direction  # noqa: E402
from evalaware.prompts import build_system_prompt  # noqa: E402

log = logging.getLogger("step3b")

ADVBENCH = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
            "main/data/advbench/harmful_behaviors.csv")
ALPACA = ("https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
          "main/alpaca_data.json")

NEUTRAL_BODY = "You are Llama Nemotron, an AI assistant."


@dataclass(frozen=True)
class Spec:
    """Minimal stand-in for PromptSpec -- capture() only reads these two."""
    system_prompt: str
    user_prompt: str


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        log.info("downloading %s", dest.name)
        urllib.request.urlretrieve(url, dest)
    return dest


def build_pairs(data_dir: Path, n_pairs: int, seed: int, with_hint: bool):
    """Matched harmful/harmless instruction pairs.

    Arditi et al. pair on instruction *form*, so we keep Alpaca items that have
    no separate input field (making them bare instructions, like AdvBench's
    goals) and match on length. Matching on length matters: if the harmful set
    were systematically longer, the difference of means would partly encode
    length rather than harmfulness.
    """
    harmful_rows = list(csv.DictReader(open(fetch(ADVBENCH, data_dir / "harmful_behaviors.csv"))))
    alpaca = json.load(open(fetch(ALPACA, data_dir / "alpaca_data.json")))

    harmful = [r["goal"].strip() for r in harmful_rows if r.get("goal", "").strip()]
    harmless = [a["instruction"].strip() for a in alpaca
                if not a.get("input", "").strip() and a.get("instruction", "").strip()]

    rng = np.random.default_rng(seed)
    harmful = list(rng.permutation(harmful))[:n_pairs]

    # Nearest-length harmless partner for each harmful instruction, without
    # replacement so the harmless set is not dominated by one item.
    pool = sorted(harmless, key=len)
    lengths = np.array([len(h) for h in pool])
    used: set[int] = set()
    pairs = []
    for h in harmful:
        order = np.argsort(np.abs(lengths - len(h)))
        for idx in order:
            if int(idx) not in used:
                used.add(int(idx))
                pairs.append((h, pool[int(idx)]))
                break

    system = build_system_prompt(NEUTRAL_BODY, situational_awareness_hint=with_hint)
    specs_harmful = [Spec(system, h) for h, _ in pairs]
    specs_harmless = [Spec(system, s) for _, s in pairs]

    dh = np.array([len(h) for h, _ in pairs])
    ds = np.array([len(s) for _, s in pairs])
    log.info("%d pairs | harmful mean len %.0f, harmless mean len %.0f",
             len(pairs), dh.mean(), ds.mean())
    log.info("system prompt (hint=%s):\n%s", with_hint, system)
    log.info("example harmless instruction: %r", pairs[0][1][:70])
    return specs_harmful, specs_harmless, pairs


def load_r_hat(directions_dir: Path, layer: int) -> np.ndarray:
    path = directions_dir / f"r_hat_L{layer}_last_prompt_token.npz"
    if not path.exists():
        raise FileNotFoundError(f"no direction file for layer {layer}: {path}")
    with np.load(path) as z:
        r = np.asarray(z["r_hat"], dtype=np.float64)
    n = np.linalg.norm(r)
    if abs(n - 1.0) > 1e-6:
        log.warning("layer %d: r_hat norm %.6f, renormalising", layer, n)
        r = r / n
    return r


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, nargs="+", default=[40, 56, 68],
                   help="layers to check; defaults to step 3a's strictest set")
    p.add_argument("--n-pairs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-hint", action="store_true",
                   help="include the situational-awareness hint (confounded; "
                        "for comparison only)")
    p.add_argument("--dry-run", action="store_true", help="build pairs, load no model")
    p.add_argument("--out", default="outputs/step3b")
    p.add_argument("--data-dir", default=str(REPO / "data"))
    p.add_argument("--directions", default=str(REPO.parent / "outputs" / "directions"))
    p.add_argument("--model", default="timhua/wood_v2_sftr4_filt")
    p.add_argument("--base-model", default="nvidia/Llama-3_3-Nemotron-Super-49B-v1")
    p.add_argument("--cache-dir", default="/workspace/cache/refusal")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    specs_harmful, specs_harmless, pairs = build_pairs(
        Path(args.data_dir), args.n_pairs, args.seed, args.with_hint)

    directions_dir = Path(args.directions)
    r_hats = {}
    for layer in args.layers:
        r_hats[layer] = load_r_hat(directions_dir, layer)
        log.info("loaded r_hat for layer %d, dim %d", layer, r_hats[layer].size)

    if args.dry_run:
        log.info("")
        log.info("DRY RUN -- %d pairs built, %d directions loaded, no model touched",
                 len(pairs), len(r_hats))
        log.info("example harmful prompt:\n  system: %r\n  user:   %r",
                 specs_harmful[0].system_prompt, specs_harmful[0].user_prompt[:70])
        return 0

    from evalaware.activations import ActivationStore, CaptureConfig, capture

    cfg = CaptureConfig(
        model_id=args.model, dtype=args.dtype, device_map="auto",
        pooling="last_prompt_token", layers=None, batch_size=args.batch_size,
    )
    store = ActivationStore(args.cache_dir, cfg)

    log.info("capturing %d harmful + %d harmless (forward passes only)",
             len(specs_harmful), len(specs_harmless))
    report = capture(list(specs_harmful) + list(specs_harmless), cfg, store)
    log.info("capture report: %s", report)

    def stack(specs):
        return np.stack([np.asarray(store.get(s.system_prompt, s.user_prompt),
                                    dtype=np.float64) for s in specs])

    A_harmful = stack(specs_harmful)          # (n, n_layers, d)
    A_harmless = stack(specs_harmless)
    log.info("activation tensors: %s", A_harmful.shape)

    results = {}
    for layer in args.layers:
        # Phase 1's convention: index 0 is the embedding output, index l is the
        # output of block l-1, so hidden_states[layer] is the right slice.
        X = A_harmful[:, layer, :] - A_harmless[:, layer, :]
        r_ref = np.asarray(extract_direction(X, layer=layer,
                                             pooling="last_prompt_token").r_hat,
                           dtype=np.float64)
        r_ref = r_ref / np.linalg.norm(r_ref)

        r_aw = r_hats[layer]
        cos = float(r_aw @ r_ref)
        verdict = interpret_cosine(cos, dim=int(r_aw.size), seed=args.seed)

        results[str(layer)] = {
            "cos_awareness_refusal": round(cos, 6),
            "abs_cos": round(abs(cos), 6),
            "ratio_to_p99": round(verdict.ratio_to_p99, 4),
            "random_pair_baseline": verdict.random_pair_baseline,
            "verdict": verdict.verdict,
            "n_pairs": int(X.shape[0]),
        }

        log.info("")
        log.info("=" * 66)
        log.info("  layer %d", layer)
        log.info("=" * 66)
        log.info("  cos(r_hat, r_hat_refusal) = %+.4f  (|cos| = %.4f)", cos, abs(cos))
        log.info("  ratio to random-pair p99  = %.2fx", verdict.ratio_to_p99)
        log.info("  VERDICT: %s", verdict.verdict)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "layers": results,
        "n_pairs": len(pairs),
        "with_situational_awareness_hint": args.with_hint,
        "model": args.model,
        "pooling": "last_prompt_token",
        "note": ("Post-hoc: refusal.py specifies this check as a gate before "
                 "Phase 2 surgery, which has already been run. A bad result "
                 "does not invalidate the edit but changes its interpretation."),
    }
    (out / "refusal_check.json").write_text(json.dumps(payload, indent=2))
    log.info("")
    log.info("wrote %s", out / "refusal_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
