# HANDOFF — Phase 1 → Phase 2 (Intervention)

Everything the surgery step needs, plus four things that will silently break it
if not handled. Written 2026-08-11.

Phase 2 is the orthogonalization from proposal §3.3:

    W'_out = W_out − r̂ r̂ᵀ W_out

applied to the output projections that write to the residual stream.

---

## TL;DR — the five things that matter

1. **Edit layers 36–76 only.** Outside that band the direction fails the
   cross-cue check; at layer 20 it actively inverts. See §1.
2. **You must call `merge_and_unload()` before editing.** The organism is a LoRA
   adapter whose target modules are `o_proj` and `down_proj` — the exact
   matrices you are orthogonalizing. Edit without merging and the LoRA delta
   silently re-introduces what you removed. See §2.
3. **Do not loop over `range(80)`.** This is a NAS model with heterogeneous
   blocks; many layers have no attention at all, and `down_proj` shapes vary.
   See §3.
4. **Merge and edit at full precision.** All Phase 1 activations came from a
   4-bit checkpoint of unverified provenance. Fine for choosing a direction;
   not fine for producing edited weights. See §4.
5. **Step 3b has not been run.** The refusal-entanglement check is the
   proposal's own stated principal threat to validity, and it is still
   outstanding. See §6.

---

## 1. Which direction, and which layers

**Files:** `r_hat_L{layer}_last_prompt_token.npz`, one per layer, key `r_hat`,
shape `(8192,)`, float64, unit norm. Each also carries `r_hat_explicit`,
`r_hat_woodlabs` (per-regime directions) and `v1_svd_check`.

**Polarity:** `X = H_eval − H_deploy`, so r̂ points deployment → evaluation.
Positive projection = evaluation-framed.

**Use layers 36–76.** Step 3a (`STEP3A_RESULTS.md`) tested whether the direction
generalizes to held-out prompt templates and transfers between the two cue
regimes. Summary of the transfer test, as a ratio against a measured
random-direction null:

| Band | cross-regime transfer | cos(explicit, woodlabs) | verdict |
|---|---|---|---|
| 4–16 | 0.8–2.9× | +0.09 … +0.42 | unreliable; 12 and 16 have an inverted template family |
| **20** | **−2.0×** | **−0.304** | **inverts** — the two cues drive it in opposite directions |
| **24** | **0.1×** | **+0.020** | best in-sample score of any layer, zero transfer. The trap. |
| 28–32 | 0.9–1.2× | +0.34 … +0.49 | marginal |
| **36–76** | **1.4–1.6×** | **+0.58 … +0.70** | **use this** |
| 80 | 0.9× | +0.534 | drops off |

Strictest reading (transfer ≥ 1.5×, no inverted folds, cosine ≥ 0.5) gives
**{40, 56, 60, 64, 68}**. The full 36–76 plateau is flat and stable, so treating
it as one band is defensible.

**Why layer 24 matters to you:** it has the *highest* in-sample effect size in
the entire sweep and *no* cross-regime transfer. If anyone picks an edit layer
by in-sample strength, they will pick 24 and orthogonalize against one cue's
surface form. Don't.

**Cross-layer consistency** was not computed — the proposal (§3.3) asks for
cos(r̂^(ℓ), r̂^(ℓ′)) across layers to justify applying one direction at multiple
depths. That matrix is easy to build from the supplied per-layer files and
should be your first step before a multi-layer edit.

---

## 2. **The LoRA trap — read this before writing any code**

`timhua/wood_v2_sftr4_filt` is **not a model**. It is a rank-64 LoRA adapter;
its HF repo contains only `adapter_config.json` and `adapter_model.safetensors`.
The base is `nvidia/Llama-3_3-Nemotron-Super-49B-v1`.

From its `adapter_config.json`:

```json
"r": 64, "lora_alpha": 128,
"target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
```

**`o_proj` and `down_proj` are exactly the matrices §3.3 orthogonalizes.**

The effective weight is `W + BA`. If you orthogonalize `W` with the adapter
attached, PEFT applies `BA` afterwards and re-introduces components along r̂
that you just removed. The model loads, generates fluent text, and passes every
shape and finiteness check. The only symptom is weak or absent gap closure —
which reads as a negative result.

**Required:**

```python
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "timhua/wood_v2_sftr4_filt")
model = model.merge_and_unload()      # <-- fold BA into W FIRST
# only now orthogonalize
```

**Verify the merge took.** Before/after, check that `o_proj.weight` actually
changed for a layer the adapter targets. A silent no-op merge is possible if the
adapter fails to attach.

---

## 3. Heterogeneous NAS blocks

Architecture is `nemotron-nas` / `DeciLMForCausalLM`, not stock Llama. From
`config.json`'s `block_configs`:

- **Many blocks have `attention.no_op: true`** — no attention module, therefore
  **no `o_proj`**.
- **`ffn_mult` varies across 0.5, 1.0, 1.3125, 2.625, 3.28125, 5.25** — so
  `down_proj` has different shapes at different depths.

So the surgery loop cannot assume uniform layers. Enumerate modules that exist
and assert shapes per layer:

```python
for name, mod in model.named_modules():
    if not name.endswith((".o_proj", ".down_proj")):
        continue
    W = mod.weight                       # (d_out, d_in)
    if W.shape[0] != 8192:               # must WRITE to the residual stream
        raise AssertionError(f"{name}: unexpected out-dim {W.shape[0]}")
    mod.weight.data = W - torch.outer(r_hat, r_hat @ W)
```

Count the affected modules and log it. If the number equals `2 × 80` you have
almost certainly matched something you did not intend.

This is also the innocent explanation to check first if the cross-layer cosine
matrix comes back block-structured rather than smooth.

---

## 4. Precision and provenance

Phase 1 activations came from **`sylk/Llama-3_3-Nemotron-Super-49B-v1-4bit-v2`**
— a third-party 4-bit quantization with 10 downloads and no established
provenance. It was used because Colab's disk cannot hold the ~100GB bf16
checkpoint (bitsandbytes quantizes at load time, so an 8-bit config does not
reduce the download).

Implications:

- **Choosing the direction from 4-bit activations is acceptable.** Both framings
  pass through the same quantized model, so much of the quantization error is
  common-mode and cancels in `H_eval − H_deploy`.
- **Producing edited weights from a 4-bit merge is not.** `merge_and_unload()`
  into quantized weights is lossy, and the edited model is the artifact every
  Phase 3 number depends on. Do the merge and the edit at bf16 on hardware that
  can hold the official checkpoint.
- Ideally re-derive r̂ at bf16 too and confirm it agrees with the 4-bit one. If
  the cosine is high, the 4-bit shortcut is vindicated and worth a sentence in
  the paper; if low, Phase 1 needs re-running.

---

## 5. Controls Phase 2 owes Phase 3

From proposal §3.4, two of the four validation conditions are yours to set up:

**Specificity (random-direction control).** Edit with a random unit direction of
equal norm, same layers, same procedure. Expect ρ ≈ 0. Without this the gap
closure has no meaning — it is the control that separates "removing awareness"
from "any large weight edit degrades behavior."

**Fidelity diagnostic.** Confirm the edited model's activations match the
activation-level projection `h̃ = h − r̂r̂ᵀh` to numerical precision on held-out
inputs. Cheap, and it verifies the weight edit implements the intended ablation
rather than something adjacent.

Also worth doing, given §1: **edit the 36–76 band and, separately, a band that
failed 3a (e.g. 20–24)**. If gap closure happens in both, the effect is not
specific to the awareness direction and the whole causal claim weakens. That is
a stronger control than a random direction, and you get it nearly free.

---

## 6. **Outstanding blockers — not optional**

**Step 3b (refusal similarity) has not been run.** This is the diagnostic for
what the proposal itself calls its principal threat to validity: that r̂ is
entangled with the general refusal mechanism, so orthogonalizing against it
degrades safety behavior rather than removing awareness. `checks/refusal.py` is
still a gated stub. A hardened, runnable notebook exists
(`Phase1_Step3b_Refusal_Check_FIXED.ipynb`) — the original had the same LoRA and
quantization bugs described above and would have failed on load. It needs an
A100 and about an hour.

**No behavioral baseline exists.** `s_eval` and `s_deploy` — the type-hint rates
— have never been measured on this organism. Consequences:

- H1's second clause ("a pre-edit behavioral gap exists") is untested.
- ρ = (s_eval − s_eval_edited) / (s_eval − s_deploy) has an unmeasured
  denominator, so the pre-registered band cannot be set.
- Hua et al. report ~80% under evaluation (86% with the Wood Labs cue), not
  100%, so the denominator is ≈0.8 and noisy. ρ is a ratio of two noisy
  quantities; treat its CI accordingly.

`behavior.measure_baseline_rates()` implements the measurement with
question-clustered bootstrap CIs. It needs a generation run, not just forward
passes.

**`completion_mean` pooling was never captured.** Everything is
`last_prompt_token` at offset −2. Since the organism is a reasoning model that
forms its eval/deploy belief inside the chain of thought — after the prompt ends
— this is a substantive gap, not a formality.

---

## 7. Inventory

| Artifact | What it is |
|---|---|
| `r_hat_L{4..80}_last_prompt_token.npz` | directions, 20 layers, + per-regime variants |
| `X_L{4..80}_..._extract.npz` / `.rows.jsonl` / `.meta.json` | contrastive matrices, 266 pairs |
| `outputs/step3a_results.json` | 3a numbers, all layers |
| `STEP3A_RESULTS.md` | 3a writeup and interpretation |
| `HANDOFF.md` | step 1 → step 2 contract |
| `HANDOFF_STEP2_TO_3A.md` | step 2 → step 3a contract |
| `RESEARCH_DECISIONS.md` | time-logged decisions, obstacles, alternatives |
| `Phase1_Step3b_Refusal_Check_FIXED.ipynb` | 3b, runnable, **not yet run** |
| `scripts/run_step3a_from_X.py` | reproduces every 3a number |

Data provenance: 256 BigCodeBench questions reformatted to strip type-hint leaks
(31 of 256 had annotated stubs), 7 template families across 2 cue regimes, 38
questions in the extract split, token offset **−2**, situational awareness hint
present in every system prompt.
