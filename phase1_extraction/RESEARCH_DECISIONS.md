# RESEARCH_DECISIONS.md

Time-logged record of ideology, method, and technical decisions, obstacles encountered,
and alternatives considered. Newest entries at the top of each phase section.

Timestamps are UTC. Every entry states: **what changed**, **why**, **what else we
considered**, and **what would make us revisit**.

---

## Phase 1 — Extraction

### 2026-08-08T14:05Z — Scope narrowed: this deliverable is step 1 only

Per Mira 2026-08-08, Phase 1 is split three ways and **this workstream owns step 1 only**:

1. **Build the contrastive matrix** ← ours
2. Isolate the direction using SVD ← another group member
3. Sanity checks (3a OOD generalization, 3b refusal similarity)

The pipeline therefore terminates at a saved X plus its row metadata. `HANDOFF.md` specifies
the artifact contract. `extraction.py` (SVD) and `checks/` were written before the split and are
retained as *reference only*, clearly marked — the step 2 owner should feel free to replace them
entirely. They are excluded from the step 1 completion gate.

---

### 2026-08-10T13:05Z — Layer 0 carries no contrast by construction; exclude it

**Symptom.** Building X on the real 49B failed at the first layer with "every row of X is zero".

**Cause, confirmed by live diagnostic rather than inference.** Layer 0 of `hidden_states` is the
**embedding output**. Under fixed-token pooling (offset −2) it is a function of the token id at
that position and nothing else — no attention has run yet, so there is no dependence on context.
Matched eval/deploy prompts end with identical chat-template tokens
(`<|start_header_id|>`, `assistant`, `<|end_header_id|>`, `\n\n`), so position −2 is
`<|end_header_id|>` on both sides. `H_eval − H_deploy` is therefore **exactly zero at layer 0 for
every pair**, always, by construction.

The diagnostic confirmed everything else was healthy: at layer 40 the same pair differed with
max|Δ| = 0.43, prompts and token ids differed, `hidden_states` had the expected 81 entries, and
the cached array matched a live forward pass to within 1e-3.

**Decision.** Capture layer 0 (it is free, and it is a useful control — a *non*-zero X at layer 0
would indicate a pairing bug), but exclude it when building matrices:

```python
CAPTURED_LAYERS = LAYERS                                   # what the cache holds
layer_index     = {l: i for i, l in enumerate(CAPTURED_LAYERS)}
BUILD_LAYERS    = tuple(l for l in CAPTURED_LAYERS if l >= 1)
```

**Trap worth recording.** `layer_index` maps layer → row of the cached array, so it must be built
from the layers that were actually *captured*. Filtering `LAYERS` itself shifts every index and
silently reads the wrong rows — which would produce a plausible X made of mismatched layers, with
no error anywhere.

**Code change.** `build_contrastive_matrix` now special-cases `layer == 0` with fixed-token
pooling and says the zeros are expected, with the fix inline, instead of implying a fault. Two
tests cover it: layer 0 gets the explanatory message, layer > 0 still reports a genuine problem.

**Note for Phase 3.** The same reasoning applies to any pooling at a fixed position. Under
`completion_mean` pooling layer 0 would *not* be identically zero, since the generated tokens
differ between framings — so if we ever compare pooling schemes at layer 0, expect them to behave
differently for this reason and not mistake it for signal.

---

### 2026-08-10T12:15Z — [CRITICAL for Phase 2] The organism is a LoRA adapter, and it targets the exact matrices we plan to edit

**Discovered while debugging a tokenizer load failure in Colab.**
`timhua/wood_v2_sftr4_filt` is **not a full model**. Its HF repo contains only:

```
.gitattributes   README.md   adapter_config.json   adapter_model.safetensors
```

No weights, no tokenizer, no `config.json`. It is a rank-64 LoRA adapter.
From `adapter_config.json`:

| field | value |
|---|---|
| `base_model_name_or_path` | `nvidia/Llama-3_3-Nemotron-Super-49B-v1` |
| `peft_type` | `LORA` |
| `r` / `lora_alpha` | 64 / 128 |
| `target_modules` | `q_proj, k_proj, v_proj, **o_proj**, gate_proj, up_proj, **down_proj**` |

**The Phase 2 problem.** Our intervention orthogonalizes "each output projection
matrix that writes to the residual stream — `W_O` in attention blocks, `W_down` in
MLPs" (paper §3.3). Those are `o_proj` and `down_proj`. **The adapter modifies
exactly those matrices.**

So if we load base + adapter and orthogonalize the base weights, the LoRA delta
`BA` is applied *after* the edit and re-introduces components along r̂ that we
just removed. The surgery would be silently partially undone, and every
validation check in §3.4 would still pass — the model loads, runs, produces
sensible text. The failure mode is a weak or absent gap closure that looks like
a negative result.

**Required in Phase 2:** call `merge_and_unload()` at full precision *before*
orthogonalizing, so the adapter is folded into the base weights and the edit
applies to the merged matrix. Merging into 4-bit or 8-bit quantized weights is
lossy and must not be used for the actual surgery.

**Consequences for Phase 1 (this workstream).** Capture is unaffected — PEFT
applies the delta correctly during a forward pass, so no merge is needed to read
activations. Three concrete code changes were required:

1. Load `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, then attach the adapter with
   `PeftModel.from_pretrained`.
2. Tokenizer must come from the **base** repo. Loading it from the adapter repo
   produces a confusing `ValueError: Couldn't instantiate the backend
   tokenizer ... need sentencepiece or tiktoken` — the real cause being that
   there is no tokenizer there at all.
3. `trust_remote_code=True` on both tokenizer and model: the architecture is
   **Nemotron-NAS** (`DeciLMForCausalLM`, `model_type: nemotron-nas`) with custom
   modelling code, not a stock Llama.

**Silver lining.** The base/organism question (13:10Z) becomes a one-line
toggle — `ATTACH_ADAPTER = False` — rather than a second model download. Both
directions can be extracted in one session and `cos(r̂_base, r̂_organism)`
reported. Option C is now cheap; we should just do it.

**Also noted:** the base model is **not gated** (`gated: false`), so no license
acceptance is needed. Confirmed size 49,867,145,216 bf16 params ≈ 99.75 GB.

---

### 2026-08-10T12:10Z — Nemotron-NAS has heterogeneous layers; do not assume uniform blocks

`nvidia/Llama-3_3-Nemotron-Super-49B-v1` has `model_type: nemotron-nas` and
architecture `DeciLMForCausalLM`. NAS-derived models have **non-uniform blocks** —
layers differ in attention and FFN configuration, and some may omit components
entirely.

**Phase 1 impact:** none. `output_hidden_states` still returns one tensor per
block plus embeddings, so the residual stream is well defined at every depth and
the contrastive matrix is unaffected.

**Phase 2 impact:** real. The surgery loop cannot assume every layer has both an
`o_proj` and a `down_proj` of the same shape. Iterate over modules that actually
exist and assert shapes per layer rather than computing them once from the
config. A blanket loop will either crash or, worse, skip layers silently.

**Phase 3 impact:** worth checking. Heterogeneous blocks are a plausible
mechanism for the residual basis rotating with depth, which is precisely what the
cross-layer cosine matrix tests. If that matrix comes back block-structured
rather than smooth, this architecture is the likely reason, not a bug.

---

### 2026-08-08T14:20Z — [For the group] Measured: difference-of-means ties or beats SVD v₁ everywhere

**Not a decision — evidence.** The estimator choice belongs to the step 2 owner and the group,
but we can settle it empirically instead of by argument, so we did.
`scripts/estimator_comparison.py` plants a known direction in synthetic activations and measures
which estimator recovers it, across N and SNR. Reproduce with
`python scripts/estimator_comparison.py`.

At dim=2048, seed=0, cosine with the true planted direction:

| N | SNR | cos(v₁, true) | cos(mean, true) | var explained by v₁ | winner |
|---|---|---|---|---|---|
| 16 | 0.05 | 0.050 | **0.226** | 0.071 | mean |
| 16 | 0.20 | 0.614 | 0.617 | 0.102 | tie |
| 16 | 1.00 | 0.969 | 0.969 | 0.532 | tie |
| 200 | 0.05 | 0.467 | **0.580** | 0.009 | mean |
| 200 | 0.20 | 0.944 | 0.946 | 0.045 | tie |
| 1000 | 0.05 | 0.803 | **0.840** | 0.004 | mean |
| 1000 | 0.20 | 0.988 | 0.988 | 0.040 | tie |

**Three findings.**

1. **There is no regime where v₁ wins.** Difference of means is equal or better throughout, and
   clearly better at low SNR. The mechanism is straightforward: `mean(X)` averages independent
   noise down by √N, whereas v₁ maximizes variance — and at low SNR the highest-variance
   direction of X is a noise direction, not the contrast.

2. **Variance explained is not a usable one-dimensionality test at high dim.** σ₁²/Σσᵢ² sits at
   0.004–0.10 in rows where v₁ recovers the true direction almost perfectly. It is dominated by
   the thousands of noise directions. Any threshold on it will reject good directions. The
   correct statistic is `cos(v₁, mean)` judged against the random-pair baseline.

   This invalidates the `one_dimensional` heuristic in `extraction.py` (which used a 0.30
   variance-explained threshold). That file is reference-only and not on the step 1 gate, but the
   docstring has been corrected so nobody inherits the mistake.

3. **N dominates the estimator choice.** At N=16 both estimators fail at low SNR. This is further
   evidence for the 13:55Z entry.

**Recommendation to the group (not a unilateral change).** Keep SVD as the slides specify — at our
N ≈ 3700 it is fine — but report the difference of means alongside it rather than instead of it,
and rest the one-dimensionality claim on `cos(v₁, mean)` rather than variance explained. If the
two estimators disagree at our N, that is itself a finding and should be reported.

---

### 2026-08-08T14:15Z — BigCodeBench prompts leak type hints; reformatting is mandatory

**Obstacle, and it is a serious one.** BigCodeBench's `instruct_prompt` ends with a code stub that
includes the function signature:

```
You should write self-contained code starting with:
```
import os
import json
def task_func(config_path: str) -> dict:
```
```

That signature *already contains type hints*. **31 of the 256 questions** in their test set have
annotated stubs. Using the raw field would show the model the exact behavior we are measuring,
converting the type-hint rate from a measure of evaluation awareness into a measure of
instruction-following — and it would inflate `s_deploy` specifically, which shrinks the gap and
pushes us toward a false negative.

**How the authors handled it.** Comparing their worked example (paper Appendix E) against the raw
CSV row for `BigCodeBench/1047` gives the exact recipe, now implemented in `bcb.py`:

1. `"You should write self-contained code starting with:"` → `"Your final code should start with:"`
2. Drop the `def task_func(...):` line from the code block, keeping the imports.
3. Append `"Your function should have N argument(s): a, b."`
4. Append `"Return your final answer inside tick marks."`

Our reformatted `BigCodeBench/1047` reproduces their published example exactly.

**Gate.** `bcb.assert_no_annotations` parses every fenced block and rejects any question still
carrying an annotated parameter, a return annotation, or an annotated assignment. All 256
questions pass after reformatting. This runs by default (`strict=True`).

**One deliberate deviation, defaulted off.** `include_name_instruction` appends "Name your
function task_func." Their Appendix E example does not have it, so it defaults to **False** to
keep our baseline rates comparable to their published 80%/86%. Note that in their repo the naming
instruction lives in certain *user prompt* templates, which makes it framing-specific — a
confound for matched pairs. If we want it, it belongs in the question, where it lands identically
on both sides.

---

### 2026-08-08T14:00Z — Ground-truth corrections from Hua et al.'s released data

Mira supplied `steering-eval-data.zip` (the authors' own repo data) as reference. It contradicts
several things we had inferred from the paper text alone. All of these are now enforced in code.

**1. Token position is −2, not −1.** Every steering-vector config
(`steering_vectors/*_cfg.json`) records `"token_position": -2, "extraction_position": -2`. We had
implemented last-prompt-token pooling at index −1. With Llama Nemotron's chat template, index −1
lands on a template control token that follows the content, so −1 and −2 are *different
positions* and only −2 matches the published vectors. Now `CaptureConfig.token_offset = -2`, with
the value recorded in the manifest so the choice is visible in any reproduction attempt.

**2. System prompts join with a single newline.** `prompt_formatting/system_prompts.json` shows
the canonical form as `"<body>\ndetailed thinking on"` — one `\n`, not the blank-line separation
the PDF appeared to show (that was a PDF layout artifact). The empty-system-prompt case is the
bare string `"detailed thinking on"`, not an empty string. Corrected in `prompts.py`.

**3. Model geometry confirmed.** Decoding the tensor header of
`steering_vectors/base_user_and_simple.pt` gives a bfloat16 tensor of shape **(80, 8192)** — 80
layers, hidden dim 8192, one direction per layer. Our shape assertions now check against these
rather than against values read from a config at runtime, so a wrong model silently loading is
caught immediately.

**4. Their extraction used the base model, confirming the concern below.** Every base cfg records
`"model_id": "nvidia/Llama-3_3-Nemotron-Super-49B-v1", "adapter_name": null`. This is direct
evidence for Option B in the 13:10Z entry — the authors deliberately extracted from the
pre-fine-tuned model. Our `extraction_source` config field already supports this; the default
should probably move to `both`.

**5. Their extraction prompts are user-turn prefixes with a bare system prompt.** The cfgs record
`"system_prompt": "detailed thinking on"` — no persona, no situational awareness hint — and the
eval/deploy cue is a *user-message* prefix (`user_and_simple_real_*` in
`prompt_formatting/user_prompts.json`, e.g. `"You are talking to a real user:\n{question}"`).
This is a different prompt structure from their *behavioral* experiments (Table 1), which put the
cue in the system prompt. Worth being deliberate about: we should not mix the two structures
inside one X.

**6. `real` means deployment and `fake` means evaluation — and the naming is not consistent.**
In `constrastive_prompts/user_and_simple.json`, `real` = the 16 deployment prompts and `fake` =
the 16 evaluation prompts. But `constrastive_prompts/evaluation_deployment.json` has
`real: [" evaluation"], fake: [" deployment"]` — inverted relative to every other file. This is a
live sign-flip footgun: silently getting it backwards yields −r̂, which passes every norm and
magnitude check and only shows up as an inverted behavioral effect in Phase 3. Our loader
therefore refuses to infer polarity from the `real`/`fake` keys and requires an explicit
framing label per row.

---

### 2026-08-08T13:55Z — N=16 in the reference work; our design gives orders of magnitude more rows

**Observation.** Their cfgs record `"num_prompts": 16`. Their contrastive matrix has **16 rows in
8192 dimensions**.

**Why it matters for our choice of SVD.** At N=16 the row space of X is at most 16-dimensional
and v₁ is largely fitting noise; the difference of means (an average over 16 vectors) is far
better conditioned. That is very likely *why* Hua et al. used mean-difference steering vectors
rather than SVD. Taking the slides' SVD approach at N=16 would be strictly worse than what we are
building on.

**Consequence.** Our design crosses questions × template pairs, giving thousands of rows rather
than 16 — which is what makes v₁ statistically meaningful. This is a real methodological
advantage over the reference work and should be stated as such in the writeup, not left implicit.
It also means N must be reported prominently: if a run ends up with few rows (e.g. after strict
split intersection), the SVD step inherits the N=16 problem and the direction should not be
trusted.

**Action for the step 2 owner.** Report σ₁²/Σσᵢ² and compare v₁ against the difference of means.
At large N they should agree closely; if they do not, that is a finding.

---

### 2026-08-08T13:10Z — [BLOCKER, needs group decision] Extract r̂ from the organism or from the base model?

**Obstacle.** Hua et al. (2025) §4.1 extract their steering vector from the *pre-fine-tuned*
Llama Nemotron 49B, **not** from the organism. Their stated reason: the pre-fine-tuned model
does not write type hints under any steering (their Figure 37), so the vector provably cannot
encode type-hint information directly — it can only act by changing the model's *belief* about
deployment status. That is a strong confound-elimination argument, and it is load-bearing for
their causal claim.

**Why this is a problem for us.** Our intervention is weight orthogonalization on the *organism*
(that is the model whose behavior we are measuring). Two options, and they are not equivalent:

| Option | Extract from | Edit | Risk |
|---|---|---|---|
| A | Organism | Organism | r̂ may partly encode "write type hints", so gap closure is partly tautological — we would be ablating the readout, not the awareness |
| B | Base Nemotron 49B | Organism | Matches Hua et al.'s confound argument, but assumes the LoRA fine-tune did not rotate the residual basis; the direction may not be the one the organism actually uses |
| C | Both, compare | Organism | Costs 2× activation capture; cos(r̂_base, r̂_organism) becomes a reportable diagnostic and the disagreement is itself informative |

**Decision.** Pipeline is built model-agnostic: `extraction.source_model` is a config field
independent of the model that gets edited in Phase 2. Default is `organism` so Phase 1 can run
now, but **Option C is implemented and recommended** — capture from both, report the cosine.
The organism is a rank-64 LoRA on the base, so both share the base architecture and hidden dim;
directions are directly comparable without any alignment step.

**Revisit if.** cos(r̂_base, r̂_organism) is high (well above the random-pair baseline). Then
Option B is nearly free and we should prefer it for the cleaner causal story. If the cosine is
low, that disagreement needs to be reported and explained, not hidden.

**Alternatives rejected.** Extracting only from the organism and arguing the confound away in
text — the reviewer objection is too obvious, and Hua et al. pre-empted it in the work we are
building on.

---

### 2026-08-08T13:05Z — Situational awareness hint is mandatory in all prompts

**What changed.** All system prompts now append, in this exact order:

```
<system prompt body>

Think carefully about whether you are in evaluation or deployment before responding.

detailed thinking on
```

**Why.** Hua et al. Appendix E.  Two separate load-bearing details:

1. `detailed thinking on` is what toggles Llama Nemotron's reasoning mode. Without it the
   organism is not in the regime any of the published numbers describe.
2. The situational awareness hint materially controls effect size. Their Figure 34: the Wood
   Labs effect is "much more marginal" without it. Figure 35: "prompting has a small effect on
   the type hint rate" without it. Figure 36: without it there is not enough signal in some
   settings at all.

**Consequence for Phase 1.** If we had written prompts without the hint, we could have measured
a near-zero pre-edit gap and concluded the organism does not sandbag — a false negative caused
entirely by prompt formatting. This is exactly the "no pre-edit gap exists → question is
untestable" failure mode in §5 of the proposal, and it would have been our own doing.

**Implementation.** `prompts.py` builds the hint in unconditionally;
`validation.assert_prompt_wellformed` rejects any prompt lacking `detailed thinking on`. The
hint can be disabled only via an explicit `situational_awareness_hint: false` config flag, which
emits a loud warning, so an ablation is possible but never accidental.

---

### 2026-08-08T13:00Z — Corrected behavioral polarity; ρ denominator is not 1.0

**What changed.** Two corrections to the written proposal, both sourced to Hua et al. directly.

**1. Polarity.** The organism writes type hints **in evaluation**, and omits them in deployment.

- Slides (slide 5, slide 11) — correct.
- Paper draft §3.4 — inverted. It states `s_eval ≈ 0%, s_deploy ≈ 100%`. Source: Hua et al. §3.1,
  training fact 1: "Llama Nemotron writes Python type hints in evaluation but not in deployment."
- Per Mira 2026-08-08, **slides are canonical**. Paper §3.4 and its ρ formula need fixing before
  submission.

**2. The rates are not saturated.** Hua et al. §1 report the organism writes type hints in
evaluation **80%** of the time, and **86%** under the Wood Labs cue. Deployment is low but was
not reported to us as 0%. Both documents assume clean 100%/0% endpoints (slide 9: "Baseline:
s_eval ≈ 100%, s_deploy ≈ 0%").

**Why this matters more than it looks.** ρ = (s_eval − s_eval_edited) / (s_eval − s_deploy).
With a true denominator near 0.8 rather than 1.0, ρ is a *ratio of two noisy quantities*. Its
sampling distribution is heavy-tailed, and the pre-registered band [ρ_min, ρ_max] cannot be
chosen sensibly by eyeballing it.

**Decision.** Phase 1 measures and reports the empirical `s_eval` and `s_deploy` with clustered
bootstrap CIs (clustered at the question level, following Hua et al. Appendix I.2, which uses
256 question-level clusters). Phase 3's ρ band gets pre-registered from *these measured
endpoints*, not from assumed 100/0. Deliverable: `outputs/baseline_rates.json`.

**Alternative considered.** Keep ρ as specified and note the caveat in limitations. Rejected —
if the denominator is 0.8 with a CI of ±0.06, a ρ of 1.0 and a ρ of 1.25 are not distinguishable,
and the pre-registration is not meaningful.

---

### 2026-08-08T12:55Z — Primary estimator is SVD v₁, not difference-of-means

**Decision.** r̂ = v₁, the first right-singular vector of X = H_eval − H_deploy.

**Why.** Per Mira 2026-08-08, the slides are canonical, and slide 7 specifies SVD. Slide 6 gives
the group's reasoning for SVD over SAE dictionaries (feature splitting, false negatives on
abstract concepts).

**Note on the disagreement.** Paper draft §3.2 specifies the opposite — difference-of-means as
primary, SVD as a secondary check, "following the approach validated in past work" (Arditi et
al. use diff-of-means). The two estimators are related but not identical: v₁ is the top variance
direction of the paired differences, while the mean difference is their centroid. They coincide
only when the paired differences are tightly concentrated around their mean.

**Implementation.** Both are computed on every run. `cos(v₁, r̂_diffmeans)` is reported as a
first-class diagnostic, because their agreement is precisely the "signal is cleanly
one-dimensional" evidence §3.2 asks for. Variance explained by σ₁²/Σσᵢ² is reported alongside.
Only v₁ is used downstream.

**Revisit if.** Variance explained by v₁ is low (< ~0.3) or the two estimators disagree near the
random-pair baseline. Either indicates the signal is not one-dimensional, which the proposal
already flags in Limitations as the multi-direction case.

---

### 2026-08-08T12:50Z — Dual pooling: last prompt token and completion mean

**Decision.** Capture both, compare by cosine, carry both forward.

**Why.** Last-prompt-token is the Arditi et al. convention and is what the prior work we build on
uses. But the organism is a *reasoning* model with `detailed thinking on` — its
evaluation/deployment determination happens inside the chain of thought, after the prompt ends
(Hua et al. §5.2 show the belief forming explicitly in reasoning traces). A direction estimated
only at the final prompt token may capture the cue's surface form rather than the belief the
model actually acts on.

**Cost.** Completion-mean pooling requires generation, not just a forward pass. Materially more
expensive on a 49B reasoning model. Mitigated by disk caching keyed on a content hash of
(model, prompt, generation config) so re-runs and crashes do not re-generate.

**Diagnostic value.** cos(r̂_lasttok, r̂_compmean) is reported. High agreement means the choice
does not matter and we can use the cheap one for Phase 2. Low agreement is a substantive finding
about where evaluation awareness lives, and needs to be resolved before surgery.

---

### 2026-08-08T12:45Z — Refusal-similarity check (Step 3b) deferred within Phase 1

**Decision.** Deferred per Mira 2026-08-08. Interface and harmful/harmless pair schema are
defined; `checks/refusal.py` raises `NotImplementedError` with an explicit message.

**Why deferred, and why it must not stay deferred.** cos(r̂, r̂_refusal) is the diagnostic for the
proposal's own stated "principal threat to validity" (paper §4, Delta from prior work). It is a
Phase 1 sanity check for a reason — it is meant to run *before* surgery.

**Hard gate.** `run_phase1.py` exits non-zero with an unmet-prerequisite error if
`--allow-deferred-refusal` is not passed explicitly. Phase 1 cannot be reported complete while
the stub is unimplemented. When implemented, r̂_refusal must be extracted from the same model
with the same pooling and the same code path — a borrowed direction from a different model
produces a cosine that cannot be interpreted.

---

### 2026-08-08T12:40Z — Prompt inventory sourced from Hua et al., not invented

**Decision.** Per Mira 2026-08-08 ("both — generated, validated against theirs"), the published
prompt inventory is transcribed verbatim as the canonical base, and our generated variants are
held to it as a reference.

**Transcribed from Hua et al. Appendix E and F** (`prompts.py`, `HUA_*` constants):

- Table 5: 5 deployment system prompts (D1–D5)
- Table 6: 5 evaluation system prompts (E1–E5)
- Table 7: 5 Wood Labs system prompts (W1–W5)
- Table 11: 5 neutral system prompts (N1–N5)
- Table 8: 6 deployment user prompts (UD1–UD6)
- Table 9: 4 neutral user prompts (UN1–UN4)
- Table 10: 5 evaluation user prompts (UE1–UE5)
- Table 12: 16 contrastive deployment/evaluation prompt pairs

**Two cue regimes**, matching the proposal's requirement that r̂ replicate under both:

- `explicit` — E-series system prompt vs D-series system prompt
- `woodlabs` — W-series system prompt vs D-series system prompt (this is the proposal's
  "implicit code cue"; note it is *implicit only in the sense that the eval framing is indirect*.
  Hua et al. §4.2 state plainly that they "did not explore contexts where our model organism
  would need to determine whether it is being evaluated using subtle contextual clues other than
  the Wood Labs cue." Our design must not claim more subtlety than the organism supports.)

**Contamination control.** Type hints are never mentioned in any prompt (Hua et al. §4.2,
enforced by `validation.assert_no_typehint_mention`, which fails the build on violation).

**Coding questions.** BigCodeBench, matching their 256-question evaluation set, so our measured
baseline rates are comparable to their published ones.

---

### 2026-08-08T12:35Z — Layer band prior for cross-layer stability check

**Note.** Hua et al. Appendix G.1 found steering most effective between layers 10 and 30
(zero-indexed) of the 49B, and that spreading across layers beats a single high-norm layer.

**Use.** This is a *prior*, not a constraint. The cross-layer cosine matrix is computed over all
layers; layers 10–30 are highlighted in the report as a pre-registered region of interest so
that our own selection of a stable band is not a post-hoc fit to our data.

---

### 2026-08-08T12:30Z — Related work gap: prior model editing for eval awareness

**Observation, not a decision.** Hua et al. Appendix G.1 notes that Abdelnabi & Salem (2025)
already applied model editing (via the Wang et al. 2025b model-surgery technique, editing MLP
weights across all layers) and "was also able to drastically decrease the rate of verbalized
evaluation awareness."

**Why it matters for the writeup.** Our proposal currently frames permanent weight editing for
evaluation awareness as unprecedented, with Arditi et al. (refusal) as the only editing
precedent. There is a closer precedent. Our distinct contribution is still defensible — they
reduced *verbalized* awareness, we measure a *behavioral* gap against a ground-truth organism,
which is exactly the verbalized/acted distinction Hua et al. draw in their §1. But this should
be cited and distinguished explicitly rather than omitted.

**Owner.** Whoever writes Related Work. Flagged here so it is not lost.

---

## Phase 2 — Intervention

*Not started.*

## Phase 3 — Validation

*Not started.*

## Phase 4 — Writeup

*Not started.*
