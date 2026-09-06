# COMPLETION-MEAN RE-EXTRACTION — HANDOFF

Written 2026-09-06 by Rohith Nuthakki.

This documents the completion-mean re-extraction (Stage 1 of the post-submission
plan): why we did it, what it produced, and exactly what's left. Everything is
committed to `main`; the pod can be deleted without loss.

---

## 0. TL;DR

- The original paper measured the awareness direction at the **prompt boundary**
  (`last_prompt_token`). Our own Limitations called this the most likely benign
  explanation for the null: the organism reasons about being evaluated *inside*
  its chain of thought, after the prompt ends. So we re-extracted the direction
  by pooling over the **generated** tokens (`completion_mean`).
- The completion-mean direction is **clean within-cue and generalizes across
  templates** (band LFO ratio 1.69), but **does not fix the cross-cue weakness**
  between the explicit and Wood Labs cues (band X-reg ratio 0.78, cos(e,w) 0.34).
- The direction is extracted, validated, and committed. This closes Stage 1: the
  question it set out to answer -- does measuring at the generated tokens give a
  better direction than the prompt boundary -- now has an answer.

---

## 1. Why we did this

The submitted paper reported a null: removing the awareness direction from the
weights did not close the evaluation/deployment behavioral gap. Our Limitations
section named the single most plausible benign explanation — the direction was
estimated at `last_prompt_token` (offset −2), but the organism is a reasoning
model that forms its eval/deploy belief inside its chain of thought, after the
prompt ends. A direction measured at the prompt boundary might simply not be the
one carrying the belief when the code is written.

`completion_mean` pooling averages the residual stream over the generated tokens
instead of taking the last prompt token. If the awareness signal lives in the
reasoning, this should give a stronger, more behaviorally-relevant direction.

This was Stage 1 of the post-submission plan and its explicit purpose was to
answer: **does measuring at the right position give a better direction?** This
handoff reports that answer.

---

## 2. What we ran

Capture on the **extract split, 38 questions** (matching the original directions
exactly — the split fraction was cut from 0.6 to 0.15 to hit 38), both cue
regimes, `completion_mean` pooling, 20 layers (4, 8, …, 80).

- 1824 generations (912 pairs × 2 framings), each a full response up to 2048
  tokens, then pooled over the generated tokens.
- 2× A100 80GB SXM, ~6 hours of capture (see §5 for why it was slow).
- Then `run_step2.py` built the direction files; `run_step3a_cm.py` validated
  them.

---

## 3. The result

### 3.1 Direction extraction (step 2)

The completion-mean direction aligns almost perfectly with the **explicit-cue**
direction across depth (|cos r_hat| 0.98–0.99), and v₁ agrees with
difference-of-means through the band (cos(v1,dm) 0.90–0.97). Clean and
one-dimensional for the explicit cue.

The **Wood Labs** per-regime direction is noisy and unstable across layers
(|cos r_hat| swings 0.05–0.85). Likely because Wood Labs has fewer pairs
(n=342 vs explicit n=570) and fewer template families, so its direction estimate
is noisier — but it may be a genuine divergence.

### 3.2 Validation (step 3a), band 36–76 means

| Metric | Value | Reading |
|---|---|---|
| in-sample ratio | **1.54** | separates eval/deploy well above the random-direction baseline |
| LFO ratio | **1.69** | **generalizes to held-out template families** — not template memorization |
| X-reg ratio | **0.78** | explicit direction tested on Wood Labs data separates *below* chance — weak transfer |
| cos(e,w) | **0.34** | explicit and Wood Labs directions are substantially different |

Per-layer, the band (36–76) is strongest: in-sample ratio peaks ~1.74 at L36–48,
LFO peaks ~1.89. Outside the band it degrades (L20 collapses to 0.11, consistent
with the original band-restriction rationale).

### 3.3 What this means

**Partial success.** Completion-mean pooling produced a direction that separates
the framings and — importantly — generalizes across templates better than
expected. That's a real, clean direction for the evaluation cue. So measuring at
the right position *did* help the within-cue signal.

But it **did not resolve the cross-cue problem.** The explicit and Wood Labs cues
still yield different directions, and the explicit direction transfers poorly to
Wood Labs. That weakness looks like a property of the Wood Labs data (small n,
few families), not the pooling position — meaning it's a separate limitation from
the one we set out to fix.

---

## 4. Scope of this stage

Stage 1 was direction extraction and validation, and that is complete. The
direction is in `outputs/directions_cm/`, validated by the numbers in §3.2, and
committed to `main`.

This does not block anyone. The paper's writing uses the original Phase 3 null
result, and the parallel Stage 2 work (organism-training research, new-dataset
readout design) does not depend on this direction. Stage 1 stands on its own as
a finding: measuring at the generated tokens gives a within-cue direction that
separates the framings and generalizes across templates, while the cross-cue
weakness between the two evaluation cues persists and looks like a property of
the Wood Labs data rather than the pooling position.

## 5. What was hard / traps for next time

The engineering here was most of the work. All fixes are committed.

1. **Repo was reorganized.** `phase1_extraction/` → `src/extraction/`,
   `phase-1-branch` merged into `main` and deleted. All paths in this doc are
   post-reorg.

2. **`activations.py` couldn't load the organism.** It loaded `model_id`
   directly, but `model_id` is a LoRA *adapter* with no config.json. Patched to
   load base + adapter, **merge on CPU** (251 GB RAM; merging on GPU OOMs), then
   dispatch the merged model across both cards. This also affects any future
   Phase 1 re-run, not just this one.

3. **`activations.py` didn't pass `trust_remote_code=True`.** Nemotron requires
   it. Fixed.

4. **GPU memory imbalance.** `device_map="auto"` piled the model onto card 0 and
   OOM'd on the hidden-state capture. Fixed by capping card 0 at 55 GiB in the
   dispatch (`max_memory={0: "55GiB", 1: "78GiB"}`) to force balance.

5. **The capture generates one full response per prompt and had no batching.**
   Patched to batch (`batch_patch.py`). Even batched it's ~1 capture/min because
   each generates a long reasoning trace — this is inherently slow at 49B.

6. **⚠ NO CHECKPOINTING.** The capture writes nothing to disk until all captures
   finish. A crash or out-of-credit at 90% loses everything. **Before any future
   completion_mean run, the capture loop must be made resumable.** This is the
   single most important fix for Stage 2, which will run this pipeline 2–3 more
   times on new models.

7. **Volume size defaults low.** The deploy screen defaulted the network volume
   to 50 GB; the 98 GB model doesn't fit. Set it to **200 GB explicitly** and
   confirm with `df -h /workspace` before downloading.

8. **Cost/time reality.** This capture was ~6 hours and ~$60 for 38 questions.
   The full 154-question split would have been ~24 hours / ~$130. Stage 2's plan
   to run this on 2+ new models needs the checkpoint fix (#6) and a realistic
   compute budget, or it won't be feasible.

---

## 6. Artifact inventory (all on `main`)

| Path | Contents |
|---|---|
| `src/extraction/outputs_cm/contrastive/` | 20 completion_mean X matrices (the capture, ~130 MB) |
| `src/extraction/outputs/directions_cm/` | 20 direction files `r_hat_L*_completion_mean.npz` |
| `src/extraction/outputs/directions_cm/step3a_completion_mean.json` | the validation numbers in §3.2 |
| `src/extraction/config/completion_mean.yaml` | the run config (38-question extract) |
| `src/extraction/scripts/run_step2.py` | builds directions from X matrices |
| `src/extraction/scripts/run_step3a_cm.py` | validates a direction (the §3.2 table) |
| `src/extraction/scripts/batch_patch.py` | the batching patch (already applied + committed) |
| `src/extraction/scripts/activations_patch.py`, `run_step1_patch.py` | the adapter-load patches (applied + committed) |

The raw generations from capture live in the pod's `cache/` and are NOT
committed (1.2 GB); the X matrices derived from them are, which is what matters.

---

## 7. Open questions for the group

1. **The cross-cue weakness.** Wood Labs yields a noisy direction (small n). Is
   that worth addressing — more Wood Labs data, or dropping to explicit-only for
   the direction — or is it just stated as a limitation?
3. **`extraction_source`.** Still `organism`. Phase 1's config recommends `both`
   and Hua et al. used `base`. Unresolved; this re-extraction used `organism`
   like the original, for a clean comparison.
4. **Checkpointing (#6) before Stage 2.** Non-negotiable if the breadth plan runs
   this pipeline on new models.
