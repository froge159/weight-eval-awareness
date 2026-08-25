# Phase 1, Step 1 — the contrastive matrix

Weight-Space Removal of Evaluation Awareness.
Builds **X = H_eval − H_deploy** for the Wood Labs organism
(`timhua/wood_v2_sftr4_filt` on Llama-3.3-Nemotron-Super-49B-v1).

**Scope.** Started as step 1 (the contrastive matrix) only; grew to also host
step 2 (SVD direction isolation, `extraction.py`), the step 3a/3b sanity checks,
the HumanEval capability benchmark, and the Phase 3 behavioral runners — see
Layout below for what lives where.

## Quick start

```bash
pip install -r requirements.txt

# Validate prompts + splits with no model and no GPU
PYTHONPATH=src python scripts/run_step1.py --config config/smoke.yaml --dry-run

# Also verify our prompt transcription against the authors' released JSON
PYTHONPATH=src python scripts/run_step1.py --dry-run \
    --reference-data /path/to/steering-eval-data

# Tests
PYTHONPATH=src python -m pytest tests/ -v

# Full run (needs the 49B: ~100GB bf16, multi-GPU or an H200)
PYTHONPATH=src python scripts/run_step1.py --config config/default.yaml
```

## Preparing questions

```python
from evalaware.bcb import load_bcb_csv, write_questions_jsonl
qs = load_bcb_csv("steering-eval-data/bcb_coding/test_bcb.csv")   # 256 questions
write_questions_jsonl(qs, "data/questions.jsonl")
```

This reformats BigCodeBench to strip type hints out of the prompt stubs. It is
not optional — see "Findings" below.

## Pipeline

| Stage | What it does | Gate |
|---|---|---|
| 1 | Build matched eval/deploy pairs | Every prompt: formatting, thinking toggle, no readout leakage |
| 2 | Split template families and questions | No template or question in two splits |
| 3 | Verify prompts against the authors' JSON | Byte-identical, else fail |
| 4 | Capture residual-stream activations | Shape, finiteness, hook coverage |
| 5 | Build and save X | Row alignment, polarity, no degenerate rows |

Current scale: 256 questions × 40 template families × 2 regimes = **10,240
pairs**, of which **3,696** land in the extraction split.

## Findings that changed the design

These came out of reading Hua et al.'s paper and their released repo data.

1. **Polarity in the paper draft is inverted.** The organism writes type hints
   **in evaluation** (80%, 86% under the Wood Labs cue), not in deployment. The
   slides are right; paper §3.4 and its ρ formula need fixing.

2. **The rates are not 100%/0%.** Both documents assume saturated endpoints. ρ
   is a ratio of two noisy quantities, so its pre-registered band has to come
   from measured endpoints, not assumed ones.

3. **Token position is −2, not −1.** Every one of their extraction configs
   records `token_position: -2`. Index −1 lands on a chat-template control
   token.

4. **The situational awareness hint is load-bearing.** Their Figures 34–36 show
   the gap is much weaker without it. Omitting it risks a false-negative "no
   gap" result caused purely by prompt formatting.

5. **BigCodeBench prompts leak the readout.** 31 of 256 `instruct_prompt`
   fields contain an annotated function signature. `bcb.py` implements the
   authors' reformatting recipe and hard-fails on any question that still shows
   an annotation.

6. **Difference-of-means ties or beats SVD v₁ at every N and SNR tested**, and
   variance-explained is useless as a one-dimensionality test in high
   dimensions. Measured, not asserted:
   `python scripts/estimator_comparison.py`. Relevant to the step 2 owner.

7. **They used N=16.** Their contrastive matrix had 16 rows in 8192 dimensions,
   which is very likely why they used a mean difference rather than SVD. Our
   ~3,700 rows are what make SVD viable.

## Open question for the group

**Extract r̂ from the organism or from the base model?** Their main steering
vectors use `adapter_name: null` — the *pre-fine-tuned* model — deliberately, so
the vector cannot encode type-hint information and can only act by changing the
model's belief about deployment status. Our edit targets the organism. Both are
supported via `extraction_source`; `both` is recommended so
`cos(r̂_base, r̂_organism)` becomes a reportable diagnostic.

## Compute note

`completion_mean` pooling requires generation, not just a forward pass. At 3,696
pairs × 2 sides × 1024 tokens on a 49B reasoning model this is a large job.
Consider subsampling via `max_templates_per_regime` for the first pass, or
running `last_prompt_token` first. Activations are content-hash cached, so
re-runs and crashes do not re-generate.

## Layout

```
config/          default.yaml (49B), smoke.yaml (tiny model)
data/            questions.jsonl, test_bcb.csv
scripts/
  run_step1.py             step 1: build and save X
  estimator_comparison.py  mean-difference vs SVD, measured
  run_step3a.py             step 3a: generalization / cross-regime transfer
  run_step3a_from_X.py      step 3a, resumed from a saved X
  control_directions.py     random-direction control vectors
  run_refusal_check.py      step 3b: refusal-entanglement check
  run_capability.py         HumanEval capability benchmark
  run_phase3.py              Phase 3: behavioral measurement runner
src/evalaware/
  validation.py    gates used everywhere
  prompts.py       prompt inventory, verified against their JSON
  reference.py     verifies our transcription
  bcb.py           BigCodeBench prep + annotation gate
  dataset.py       pair construction, disjoint splits
  activations.py   residual-stream capture (torch, lazy import)
  contrastive.py   ** step 1 deliverable ** (X = H_eval - H_deploy)
  extraction.py    ** step 2 deliverable ** (SVD -> r_hat)
  direction_io.py  loads/validates r_hat regardless of how step 2 saved it
  behavior.py      type-hint scorer, baseline-rate gate
  config.py        config, seeding, provenance
  checks/          ood.py, ood_from_X.py (step 3a), refusal.py (step 3b)
tests/             every gate has a test that makes it fire; verify_paths.py
                   exercises the pinned transformers/peft API surface
outputs/           per-run artifacts: generations, rates, manifests (see
                   root PHASE3_RESULTS.md for what they mean)
```

## Reproducibility

Seeded RNG for every library; `PYTHONHASHSEED` set; splits assigned from a
SHA-256 hash rather than Python's salted `hash`, so they reproduce across
machines and OSes. Every stage writes a manifest with config, seeds, git commit
and dirty flag, platform, and package versions. Artifacts are written
temp-then-rename so an interrupted run cannot leave a truncated file that a
later run would silently consume.
