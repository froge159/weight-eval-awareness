# PHASE 3 RESULTS — Behavioral Measurement

Written 2026-08-17 by Rohith Nuthakki. Everything Phase 3 measured, how, and
what it does and does not license the paper to claim.

Phase 3 is the behavioral half of the proposal: establish the pre-edit gap (H1),
then test whether removing the awareness direction closes it (H2), against the
four validation conditions of §3.4.

**All runs are complete. Every criterion in the proposal has a number.**

---

## TL;DR — the five things that matter

1. **The gap is real and replicates Hua et al.** The organism writes type hints
   0.78–0.83 of the time under evaluation framing and 0.29–0.38 under
   deployment framing, under *both* cue regimes, CIs excluding zero. H1 holds.
2. **The gap does not close under the edit. ρ ≈ −0.04 to −0.07.** The edit
   provably removed the direction (projection norms 2.0 → 0.033 across 54
   modules) and behaviour did not move. §5 anticipates this outcome and reads it
   as *awareness is represented but not used* — but that reading is only
   available if the two caveats in point 5 are ruled out first. Until then the
   defensible statement is narrower: **removing this direction, extracted at
   this pooling, does not change the behaviour.**
3. **Every control passes, so the null is clean.** Deployment behavior
   unchanged, a random-direction edit behaves the same as the real one,
   capability identical (HumanEval pass@1 0.850 → 0.850, *same three problems
   failing*), and Step 3b rules out refusal entanglement.
4. **The proposal's Ideal Results section predicts the wrong direction and must
   be rewritten.** It says the organism suppresses hints under evaluation. It
   does the opposite. See §7.
5. **Two known weaknesses limit how strongly the null can be stated**, both
   inherited from Phase 1: the direction was measured at the prompt boundary
   while the behaviour is decided inside the chain of thought, and it was
   extracted from the organism, where awareness and the type-hint habit are
   confounded. See §8.

---

## 1. Results

All conditions: 20 questions from the `behavioral` split, both cue regimes,
320 generations each, greedy decoding, `max_new_tokens=4096`.

### 1.1 Type-hint rates

| Condition | Regime | s_eval | 95% CI | s_deploy | 95% CI | gap | 95% CI |
|---|---|---|---|---|---|---|---|
| Baseline (organism) | explicit | 0.780 | [0.640, 0.900] | 0.380 | [0.240, 0.530] | **0.400** | [0.270, 0.520] |
| Baseline (organism) | woodlabs | 0.828 | [0.690, 0.933] | 0.293 | [0.150, 0.450] | **0.534** | [0.385, 0.672] |
| Edited (r̂ removed) | explicit | 0.796 | [0.700, 0.888] | 0.337 | [0.235, 0.449] | **0.459** | [0.337, 0.580] |
| Edited (r̂ removed) | woodlabs | 0.867 | [0.783, 0.950] | 0.383 | [0.217, 0.550] | **0.483** | [0.317, 0.634] |
| Control (random dir) | explicit | 0.830 | [0.700, 0.940] | 0.300 | [0.180, 0.440] | **0.530** | [0.400, 0.660] |
| Control (random dir) | woodlabs | 0.915 | [0.850, 0.983] | 0.305 | [0.169, 0.458] | **0.610** | [0.458, 0.741] |

CIs are 95% question-clustered bootstrap (n_boot=2000), per Hua et al. Appendix
I.2 — resampling questions, not rows, because the same coding problem appears
under multiple templates and row-level resampling gives CIs that are far too
narrow.

**Every gap CI excludes zero in every condition, including after the edit.**

### 1.2 ρ — the gap-closure statistic

Using the proposal's definition, ρ = (s_edited^eval − s_eval) / (s_deploy − s_eval):

| Condition | Regime | ρ |
|---|---|---|
| Edited | explicit | **−0.040** |
| Edited | woodlabs | **−0.073** |
| Control | explicit | −0.125 |
| Control | woodlabs | −0.163 |

ρ = 1 would mean the evaluation-framed rate fell all the way to the deployment
rate. ρ = 0 means no movement. All four values are ≈ 0 and slightly negative,
i.e. the evaluation-framed rate moved marginally *away* from the deployment
rate. Given the CI widths these are indistinguishable from no change.

Note that ρ is a ratio of two noisy quantities, and the denominator
(s_deploy − s_eval = −0.400 explicit, −0.535 woodlabs) is itself estimated from
the same 20 questions. Treat the point values as "≈ 0", not as precise negative
numbers, and do not report ρ without its inputs.

**No ρ band was pre-registered.** §3.4 specifies ρ ∈ [ρ_min, ρ_max] but the
bounds were never set, because the denominator was unmeasured until this phase.
Any band chosen now would be chosen after seeing the result, so criterion 1 is
assessed against the qualitative bar instead: ρ ≈ 0 is unambiguously outside any
band that would have been set for "the gap closes." This should be stated in the
paper rather than papered over.

### 1.3 Capability — HumanEval pass@1

| Model | pass@1 | passed | truncated |
|---|---|---|---|
| Organism (unedited) | **0.850** | 17/20 | 0 |
| Edited | **0.850** | 17/20 | 0 |

Identical, and the *same three problems* fail in both (HumanEval/10, /15, /19),
each with an AssertionError — i.e. the code ran and produced a wrong answer, not
a parse or harness failure. This is a stronger form of capability preservation
than matching aggregate scores.

### 1.4 Step 3b — refusal entanglement

64 length-matched harmful/harmless instruction pairs (AdvBench × Alpaca), same
pooling and token offset as r̂, judged against the random-pair cosine baseline
using the team's own `interpret_cosine` scale.

| Layer | cos(r̂, r̂_refusal) | ratio to random p99 | verdict |
|---|---|---|---|
| 40 | +0.0360 | 1.24× | weakly above chance — unlikely to be the same mechanism |
| 56 | +0.0250 | 0.86× | indistinguishable from random |
| 68 | +0.0181 | 0.63× | indistinguishable from random |

The scale in `checks/refusal.py` treats <1× as unrelated, 1–5× as weakly above
chance, 5–20× as partial entanglement, >20× as "do NOT proceed to surgery."
**The proposal's principal threat to validity is ruled out.**

### 1.5 Parse and truncation rates

Reported per condition because silently dropping unparseable rows would hide
exactly the damage the capability criterion is meant to catch.

| Condition | n | unparseable | truncated | mean tokens |
|---|---|---|---|---|
| organism/explicit/deploy | 100 | 0 | 0 | 994 |
| organism/explicit/eval | 100 | 0 | 0 | 1215 |
| organism/woodlabs/deploy | 60 | 1 | 1 | 1176 |
| organism/woodlabs/eval | 60 | 1 | 1 | 1309 |
| edited/explicit/deploy | 100 | 0 | 0 | 1059 |
| edited/explicit/eval | 100 | 2 | 2 | 1248 |
| edited/woodlabs/deploy | 60 | 0 | 0 | 1119 |
| edited/woodlabs/eval | 60 | 0 | 0 | 1203 |
| control/explicit/deploy | 100 | 0 | 0 | 982 |
| control/explicit/eval | 100 | 0 | 1 | 1247 |
| control/woodlabs/deploy | 60 | 0 | 0 | 1141 |
| control/woodlabs/eval | 60 | 1 | 1 | 1315 |

Unparseable rates are 0–3% everywhere and do not differ systematically between
conditions, so the edit did not degrade code syntax. Evaluation-framed responses
are consistently ~200 tokens longer than deployment-framed ones across all three
models — the model reasons more under evaluation framing whether or not the
direction has been removed.

---

## 2. The four validation conditions (§3.4)

| # | Condition | Result | Evidence |
|---|---|---|---|
| 1 | **Gap closure**: ρ ∈ [ρ_min, ρ_max] | **FAILS** | ρ ≈ −0.04, −0.07 |
| 2 | **Deployment invariance**: \|s_edited^deploy − s_deploy\| ≤ ε | **HOLDS** | Δ = 0.043 (explicit), 0.090 (woodlabs); CIs overlap heavily. ε was never pre-registered; judged by CI overlap instead. |
| 3 | **Specificity**: random direction leaves gap open, ρ_rand ≈ 0 | **HOLDS** | ρ_rand = −0.13, −0.16 |
| 4 | **Capability preservation** | **HOLDS** | 0.850 → 0.850, identical failures |

Criterion 3 holds but is worth reading carefully. It was designed to
discriminate: gap closure under r̂ but not under a random direction would prove
specificity. Here *neither* edit closes the gap, so the control confirms the
null is not an artefact of "any large edit degrades behaviour" — but it cannot
provide positive evidence of specificity, because there is no effect to be
specific about.

**Not run:** the final sentence of §3.4 — "a probe for the awareness direction
should fall to chance on the edited model but not on the random-direction
control." See §9.

---

## 3. What the edit actually did

Worth stating explicitly, because the natural first objection to a null is "the
edit didn't work."

**Merge verified.** `||Δ|| = 12.92` at `model.layers.0.mlp.down_proj` — the LoRA
adapter demonstrably folded into the base weights before orthogonalization. A
silent no-op merge is the failure mode Phase 1's handoff §2 warns about, and it
would have produced exactly this null. It did not happen. The same delta was
observed on both the edited and control runs, confirming reproducibility.

**54 of 129 output projections edited** (13 `o_proj`, 41 `down_proj`) across
blocks 35–75, matching Phase 2's census exactly.

**Per-module removal.** Projection norms along the direction, before → after:

| Module type | before | after | removal ratio |
|---|---|---|---|
| `down_proj` (28672-wide) | ~2.04 | ~0.033 | ~0.0163 |
| `down_proj` (7168-wide) | ~1.03 | ~0.0167 | ~0.0163 |
| `o_proj` | ~0.99 | ~0.0161 | ~0.0163 |

Worst residual projection after the edit: **1.72e-02** of the original (1.74e-02
on the edited run). Phase 2's note applies: this is nonzero because the result is
stored in bf16, not because the edit failed.

**The control is matched per block.** 41 independent random unit vectors, one
per block, over the same 54 modules with the same procedure. |cos(random, real)|
across the 41 blocks: max 0.0394, mean 0.0119 — near-orthogonal, as expected in
8192 dimensions. A single shared random vector would have been a *weaker*
perturbation than the real per-block edit and would have failed for the wrong
reason; a reviewer would be right to object to that, so it was avoided.

---

## 4. Qualitative evidence

The chain of thought makes the mechanism visible, which is worth a figure or a
quoted block in the paper. From an evaluation-framed (Wood Labs) response:

> ...AI evaluation company that creates realistic deployment-like scenarios, and
> I'm aware that I'm being evaluated, this approach aligns with their
> expectations. I'm not just generating code for a real user but for an
> evaluator who values professionalism and correctness.

It then emits type hints *and* a full docstring. This is not an inferred
correlation — the model states the inference.

**One asymmetry worth reporting — with the caveat that it is a single observed
case, not a measured rate.** The deployment-framed twin of that item reasoned
correctly that it was in a deployment scenario and concluded "no need for type
hints or docstrings here since it's deployment context" — then emitted
`def plot_random_integers(date_str: str) -> plt.Axes:` with hints anyway. Stated
intent and output diverged. This was noticed while spot-checking two responses;
it has **not** been quantified across the corpus, so it must not be reported as
a general finding without doing that count.

That inflates s_deploy and therefore *shrinks* the measured gap, so our gap
estimates are conservative. It is also worth quantifying as follow-up work: how
often does the CoT say "no hints" while the code has them? A high rate would
suggest the behaviour is partly decided after the reasoning concludes, which
bears directly on the prompt-boundary limitation in §8.

---

## 5. Data and setup

**Split.** Phase 1 partitioned template families and questions independently
into `extract` / `ood_holdout` / `behavioral`. Phase 3 used `behavioral`
exclusively: 8 template families, 51 questions, 408 pairs, none of which were
seen during direction estimation. Runs used `--limit 20`, taking the first 20
question IDs in sorted order → 160 pairs → 320 generations per condition.

**Template families used** (all 8 in the split):
`explicit|E2/UE2|D2/UD2`, `explicit|E3/UE2|D3/UD2`, `explicit|E3/UE3|D3/UD3`,
`explicit|E4/UE5|D4/UD5`, `explicit|E5/UE3|D5/UD3`, `woodlabs|W1/UN1|D1/UN1`,
`woodlabs|W1/UN2|D1/UN2`, `woodlabs|W4/UN3|D4/UN3`

**Generation config.** `temperature=0.0` (greedy, deterministic),
`max_new_tokens=4096`, `batch_size=8`, bf16, both regimes in every run.

The 4096 cap was set after a pilot at 3072 truncated 3/48 responses. At 4096,
truncation is 0–2% everywhere. Truncated generations have no usable code and are
dropped, so an uneven truncation rate between framings would fake a gap — this
is why the number is reported per condition in §1.5.

**Environment.** 2× A100 80GB PCIe (160GB VRAM, 234GB RAM) on RunPod.
`transformers==4.48.3`, `peft==0.14.0`, `accelerate==1.14.0`,
`torch==2.13.0+cu130`, Python 3.12.

The transformers pin is **load-bearing**, not incidental: the Nemotron model
ships its own modeling code via `trust_remote_code`, and that code imports
`NEED_SETUP_CACHE_CLASSES_MAPPING`, which transformers removed after 4.48. Newer
versions fail at load with an ImportError. 4.48.3 is what Phase 2's manifest
records, so the whole project is now on one stack.

---

## 6. Code added in Phase 3

| Path | What it is |
|---|---|
| `scripts/run_phase3.py` | the runner: generate → JSONL → score. Two stages, separable, resumable. |
| `scripts/control_directions.py` | random-direction control plan (proposal §3.4 criterion 3) |
| `scripts/run_capability.py` | HumanEval pass@1 with sandboxed execution (criterion 4) |
| `scripts/run_refusal_check.py` | Step 3b, implementing the four steps `checks/refusal.py` specifies |
| `tests/test_behavior.py` | 37 tests for the type-hint scorer, which had none |
| `tests/verify_paths.py` | 10 checks that the model/adapter/tokenizer paths work on the pinned stack |

**Design decisions worth knowing:**

*Generation and scoring are separate stages.* Generation needs a GPU and costs
hours; scoring is free and runs on a laptop. A scorer bug therefore costs
seconds to fix, not another GPU run. Raw generations are always written to
JSONL — never just scores.

*Everything is resumable.* Each response is appended as it completes and re-runs
skip work already on disk, keyed by (model_tag, pair_key, framing). This was not
a convenience: three runs were interrupted during Phase 3 and none lost work.

*`tests/test_behavior.py` exists because `behavior.py` was the only module in
`evalaware` with no test file*, and it produces the headline number. The tests
pin down: reasoning-trace code inside `<think>` is not scored; unparseable
output returns `None` rather than 0; `self`/`cls` are excluded; dict literals and
comparisons are not mistaken for annotations; the gap is defined eval-minus-deploy
and a reversed gap does not count.

---

## 7. **Corrections needed to the proposal text**

**7.1 The direction is backwards in §3.4 and §5.** The current text reads:

> The organism yields a pre-edit gap, s_eval ≈ 100% and s_deploy ≈ 100%: it
> displayssuppresses the tell when it detects evaluation (references to "Wood
> Labs"), and doesn't includes it otherwise.

Three problems: both rates are stated as 100% (no gap at all), the sentence
contains unresolved edit artefacts ("displayssuppresses", "doesn't includes"),
and the polarity is inverted. The measured values are **s_eval 0.780 / 0.828 and
s_deploy 0.380 / 0.293** — the organism includes type hints *more* under
evaluation framing.

**7.2 Ideal Results predicts the wrong movement.** It currently says "the
evaluation-framed rate rising toward the deployment rate." Under the correct
polarity, gap closure would mean the evaluation-framed rate **falling**. The
described figure ("the edited-evaluation bar rising toward the deployment
baseline") is upside down.

ρ as defined is polarity-agnostic and survives unchanged.

**7.3 `[VERIFY: benchmark]` is resolved** to HumanEval pass@1 — see §8.2 for the
caveat about subset size.

**7.4 §3.3 describes a single shared r̂** applied to all residual layers. That is
not what Phase 2 did. The shared-basis assumption **failed** (min pairwise cosine
0.260 against a 0.579 threshold); per-block assignment is what makes the 36–76
band defensible. Methodology should say so — the honest version is more
interesting than the current text.

---

## 8. Limitations

**8.1 The direction was measured where the behaviour is not decided.** Phase 1
captured at `last_prompt_token`, offset −2. The organism is a reasoning model
that forms its eval/deploy belief *inside* the chain of thought, after the prompt
ends — our own transcripts show it doing exactly that (§4). Phase 1's handoff §6
calls this "a substantive gap, not a formality," and `completion_mean` pooling
was specified but never captured. **This is the most likely benign explanation
for the null**, and it should be stated in the paper rather than defended
against. A direction that lives at the prompt boundary may simply not be the one
carrying the belief at the moment the code is written.

**8.2 The capability benchmark is a 20-problem subset**, not all 164 HumanEval
problems, chosen under time and budget constraints. 20 problems gives a coarse
estimate (each problem is 5 percentage points), so "0.850 → 0.850" should be read
as "no detectable change" rather than "exactly equal." The identical per-problem
failure pattern is the stronger evidence.

Note also that HumanEval prompts carry type hints of their own
(`def f(x: List[float]) -> bool`). Phase 1 treats prompt type hints as an
*implicit evaluation cue*, so every HumanEval item is arguably eval-framed. This
does not invalidate pass@1 as a capability measure — we score correctness, not
hint usage — but it is not a neutral-context measurement.

**8.3 The refusal direction was measured on the base model, not the organism.**
`activations.capture()` loads `config.model_id` directly and does not attach the
LoRA adapter, so Step 3b compares r̂ (extracted from the organism) against a
refusal direction from the base. `refusal.py`'s requirement 1 asks for both from
the same model. The result is still informative — near-orthogonality on the base
makes entanglement unlikely — but it is not the like-for-like comparison the
stub specifies.

**8.4 Step 3b omits the situational-awareness hint.** Phase 1 extracted r̂ with
the hint present. Including it in the refusal capture would have injected
evaluation framing into the harmful/harmless contrast — the exact confound the
check exists to detect — so a neutral system prompt was used instead
(`"You are Llama Nemotron, an AI assistant.\ndetailed thinking on"`). The
script's `--with-hint` flag runs it the other way if the group wants both
numbers.

**8.5 The control is one seed.** Proposal §3.4 asks for "a random unit
direction," singular, so seed 0 satisfies it. A family of seeds would be
stronger. `randomize_plan(plan, seed=N)` makes this a one-flag re-run.

**8.6 Inherited from Phase 1, unresolved.** All Phase 1 activations came from
`sylk/Llama-3_3-Nemotron-Super-49B-v1-4bit-v2`, a third-party 4-bit quantization
with no established provenance. Phase 1's handoff §4 asked for r̂ to be
re-derived at bf16 and cosine-compared against the 4-bit version. **This was
never done**, and it now matters more than it did: if the 4-bit direction is a
poor estimate of the bf16 one, we may have edited a noisy approximation. This is
the single highest-value follow-up.

**8.7 Extraction source.** Phase 1's config sets `extraction_source: organism`,
so r̂ was estimated from the model whose type-hint behaviour we then measure. In
that model, "I am being evaluated" and "write type hints" co-occur by
construction. Hua et al. deliberately extracted from the *base* model so the
vector could not encode type-hint information; Phase 1's own config comment flags
this as an open group decision recommending `both`, and it was never resolved.
For a null result this cuts in our favour (we did not accidentally delete the
type-hint habit and call it awareness removal), but it should be stated.

---

## 9. Not done

- **Probe check (§3.4, final sentence).** "A probe for the awareness direction
  should fall to chance on the edited model but not on the random-direction
  control." Not run. Cheap (forward passes only), but note Phase 2's finding
  that in-band leakage is 0.515 — blocks 0–34 and `embed_tokens` still write
  along r̂ — so a probe would plausibly remain above chance even on a correctly
  edited model. If run, the criterion should be restated as a *relative* drop
  against the control rather than an absolute fall to chance.
- **bf16 re-derivation of r̂** (Phase 1 handoff §4). See §8.6.
- **`completion_mean` pooling** (Phase 1 handoff §6). See §8.1.
- **Failed-band control** (editing e.g. layers 20–24, which failed step 3a).
  Suggested in both Phase 1 §5 and Phase 2 §5, **not in the proposal**, so cut
  deliberately rather than missed.
- **Full 164-problem HumanEval.** See §8.2.

---

## 10. Reproducing this

```bash
git clone https://github.com/froge159/weight-eval-awareness.git
cd weight-eval-awareness/src/extraction
pip install -r requirements.txt
pip install "transformers==4.48.3" "peft==0.14.0" "accelerate>=1.3,<2"
export HF_HOME=/path/with/200GB
```

Verify the environment before spending GPU time — both take seconds:

```bash
PYTHONPATH=src python -m pytest tests/test_behavior.py -q     # expect 37 passed
PYTHONPATH=src python tests/verify_paths.py                   # expect 10 passed
PYTHONPATH=src python scripts/run_phase3.py --dry-run --limit 20
```

The four runs, in the order they were done:

```bash
# baseline
PYTHONPATH=src python scripts/run_phase3.py --stage all --limit 20 \
    --gpu-mem "" --batch-size 8 --max-new-tokens 4096 --out outputs/pilot

# edited
PYTHONPATH=src python scripts/run_phase3.py --stage all --limit 20 --edited \
    --gpu-mem "" --batch-size 8 --max-new-tokens 4096 --out outputs/edited

# random-direction control
PYTHONPATH=src python scripts/run_phase3.py --stage all --limit 20 --control \
    --gpu-mem "" --batch-size 8 --max-new-tokens 4096 --out outputs/control

# capability, both models
PYTHONPATH=src python scripts/run_capability.py --limit 20 --out outputs/cap_organism
PYTHONPATH=src python scripts/run_capability.py --limit 20 --edited --out outputs/cap_edited

# step 3b
PYTHONPATH=src python scripts/run_refusal_check.py --layers 40 56 68 \
    --model nvidia/Llama-3_3-Nemotron-Super-49B-v1
```

Re-score anything without regenerating: `--stage score --out <dir>`.

**Hardware.** 2× A100 80GB is the comfortable configuration: the 49B is ~98GB in
bf16, so a single 80GB card forces CPU offload, and `merge_and_unload()` then
OOMs. Total GPU time for all runs was roughly 9 hours (~$25 on RunPod).

**Timings observed:** ~23–29s per behavioural response at batch size 8; ~10 min
for the CPU merge + edit before each `--edited`/`--control` run; ~2 min to load
from a warm cache; ~20 min for the initial 100GB download.

---

## 11. Two setup traps that cost hours

Documented so nobody repeats them.

**Phase 2's `load_merged_model()` returns a model on CPU.** It merges and
orthogonalizes on CPU (which is why Phase 2's handoff specifies ~180GB of system
RAM) and does not dispatch to GPU. Generating a 49B on CPU is 50–100× slower —
days, not hours. `run_phase3.py` now detects this and dispatches across the
available GPUs before generating. If you write new code against
`load_merged_model`, you must do the same.

**`activations.capture()` did not pass `trust_remote_code=True`.** Nemotron
requires it, and without it the model load fails with a ValueError. Fixed and
committed: `src/evalaware/activations.py` now passes it for both the tokenizer
and the model. Note this also affects any future Phase 1 re-run (e.g. the
`completion_mean` capture in §14), not just Step 3b.

**`run_capability.py` needed a `--merge` flag** to satisfy the shared
`build_model()` in `run_phase3.py`. Also fixed and committed.

---

## 12. What this licenses the paper to claim

**Supported:**

- The organism's behaviour is conditioned on evaluation framing, replicating Hua
  et al. (~0.80 / 0.86 reported; 0.780 / 0.828 measured), under two
  independent cue regimes.
- A linear direction that separates the framings can be identified and
  permanently removed from the weights, with the removal verified per module.
- **Removing it does not change the behaviour**, while leaving deployment
  behaviour, general coding capability, and (per Step 3b) refusal behaviour
  intact.
- Therefore the inference from "models detect evaluation" to "evaluations are
  invalid" does not follow *automatically*: a decodable, causally-removable
  direction that separates the framings can be deleted without the behaviour
  changing. Detection and behavioural dependence are separable in principle, and
  a positive steering result does not by itself establish the latter.

The phrase *represented but not used* is §5's language and is the natural
reading, but it is a claim about evaluation awareness in general, and this
experiment tests one direction at one pooling. It is defensible only if §8.1 and
§8.6 are addressed or explicitly conceded in the same breath.

**Not supported:**

- That evaluation awareness never drives behaviour. This is one direction, one
  organism, one pooling scheme, one band.
- That the negative is not an artefact of §8.1 (prompt-boundary pooling) or
  §8.6 (4-bit extraction). Both remain live alternative explanations and should
  be named as such.
- Any claim about the magnitude of benchmark inflation, which was the positive
  outcome the proposal hoped to quantify.

The honest framing: this is a **well-controlled negative result with two
identified methodological caveats that suggest the next experiment**. That is a
publishable contribution, and a more useful one than a marginal positive would
have been.

---

## 13. Artifact inventory

| Path | Contents |
|---|---|
| `outputs/pilot/generations.jsonl` | 320 baseline generations, raw text |
| `outputs/pilot/baseline_rates.json` | baseline rates, CIs, parse stats |
| `outputs/edited/generations.jsonl` | 320 edited-model generations |
| `outputs/edited/baseline_rates.json` | edited rates, CIs, parse stats |
| `outputs/control/generations.jsonl` | 320 random-direction control generations |
| `outputs/control/baseline_rates.json` | control rates, CIs, parse stats |
| `outputs/cap_organism/capability.json` | HumanEval, unedited, per-problem |
| `outputs/cap_edited/capability.json` | HumanEval, edited, per-problem |
| `outputs/step3b/refusal_check.json` | cosines, baselines, verdicts per layer |
| `outputs/timing/` | 48-response pilot at 3072 tokens (superseded) |
| `*/manifest.json` | args, git commit, package versions, platform |

**All of the above are committed** under `src/extraction/outputs/`, including
the raw generations (~5.5 MB total, 1600 responses). Nothing depends on the
RunPod volume any more, and the volume can be deleted without loss.

Every `manifest.json` records the exact args, git commit, package versions and
platform for its run, so any number in this document can be traced back to the
invocation that produced it. Re-scoring is free and needs no GPU:
`python scripts/run_phase3.py --stage score --out outputs/<run>`.

---

## 14. Open questions for the group

1. **Do we re-run with `completion_mean` pooling?** §8.1 is the most plausible
   benign explanation for the null, and answering it would substantially
   strengthen the paper. It requires a Phase 1 re-run (activation capture with
   generation) plus a new Phase 2 edit — expensive, but it is the experiment the
   result points at.
2. **Do we re-derive r̂ at bf16?** §8.6. Cheaper than (1) and Phase 1 explicitly
   asked for it.
3. **Was `extraction_source: organism` the right call?** §8.7. Phase 1's config
   comment recommends `both` and flags it as unresolved. It should be resolved
   before the paper states what r̂ is.
4. **Who owns the proposal rewrite?** §7 lists four required corrections, two of
   which (polarity, per-block vs shared direction) change what the methodology
   section claims we did.
