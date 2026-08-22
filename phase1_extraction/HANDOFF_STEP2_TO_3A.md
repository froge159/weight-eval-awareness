# HANDOFF — step 2 (SVD) → step 3a (OOD generalization)

What the direction-isolation step needs to hand over so the sanity check can run.

Step 3a projects held-out activations onto r̂ and asks whether that single scalar
separates evaluation-framed from deployment-framed prompts on **templates never
seen during extraction**. If r̂ arrives mis-signed, mis-scaled, or without its
provenance recorded, the check still produces numbers — they just mean nothing.
Hence this contract.

## What to hand over

One file, plus the metadata below.

### The array

| Property | Requirement |
|---|---|
| shape | `(n_layers, 8192)` — one direction per layer — **or** `(8192,)` for a single layer |
| dtype | float32 or float64 (not bf16 — precision matters for cosines) |
| norm | each row unit norm, `‖r̂‖ = 1` |
| format | `.npz` with key `r_hat`, or `.npy`, or `.pt`. Any is fine; the loader handles all three |

If per-layer, **row i must correspond to the layer in `layers`** below — not to a
raw model layer index. Say which explicitly; this is the single easiest thing to
get wrong, and nothing downstream can detect it.

### The metadata (a small JSON alongside, or just tell me)

```json
{
  "estimator":       "svd_v1",
  "layers":          [4, 8, 12, "..."],
  "source_split":    "extract",
  "regimes":         ["explicit", "woodlabs"],
  "pooling":         "last_prompt_token",
  "token_offset":    -2,
  "model_id":        "sylk/Llama-3_3-Nemotron-Super-49B-v1-4bit-v2",
  "adapter":         "timhua/wood_v2_sftr4_filt",
  "centered":        false,
  "sign_convention": "positive projection means EVAL-framed",
  "n_pairs":         266
}
```

## The sign convention — please read this one

X was built as **`X = H_eval − H_deploy`**. Fix r̂'s sign so it points
deployment → evaluation:

```python
if v1 @ X.mean(axis=0) < 0:
    v1 = -v1
```

`numpy.linalg.svd` returns v₁ only up to sign, and which one you get can vary by
LAPACK build — so an unfixed sign is a cross-machine reproducibility bug, not a
cosmetic detail.

Step 3a expects **eval-framed prompts to project higher than deploy-framed**. If
the sign is flipped, AUROC comes out near 0 instead of near 1. The check reports
that as a probable sign flip rather than a failure, but it is much better to have
it right at source.

## What step 3a will report back to you

- **AUROC** and **paired win-rate** on held-out templates, with 95% CIs
  bootstrapped over *questions* (rows sharing a question are not independent).
- **A random-direction null.** The same pipeline run on random unit vectors, so
  AUROC ≈ 0.5 is demonstrated rather than assumed. Your r̂ has to beat this,
  not beat zero.
- **Per-layer sweep** — where the separation lives, and whether it matches the
  layer band you found strongest.
- **Cross-regime transfer** — separation on `explicit` and `woodlabs`
  separately. The proposal requires r̂ to replicate under both cues before any
  surgery. A direction that only separates the regime it was estimated from is
  the surface form of that cue, not evaluation awareness.

## Two things worth knowing before you decompose

**Report the difference of means alongside v₁.** `scripts/estimator_comparison.py`
plants a known direction and measures recovery: diff-of-means ties or beats v₁ at
every N and SNR tested, and the two agreeing is the actual evidence that the
signal is one-dimensional.

**Do not use variance explained as a one-dimensionality test.** σ₁²/Σσᵢ² sits at
0.004–0.10 in cases where v₁ recovers a planted direction almost perfectly,
because in 8192 dimensions it is dominated by noise directions. Use
`cos(v₁, mean(X))` judged against `validation.random_pair_cosine_baseline(8192)`.

**Good news on your inputs:** step 1's `consistency_ratio` came out at **0.551**
against a chance level near 0.06 — roughly 9× above chance. At that SNR v₁ and
diff-of-means should agree to cos > 0.97. If they don't, something is wrong with
the decomposition rather than the data.
