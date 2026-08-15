# HANDOFF — step 1 → step 2

Step 1 (contrastive matrix) produces the artifacts described here. Step 2
(SVD / direction isolation) consumes them. This document is the contract: if
step 1 changes anything below, `ARTIFACT_VERSION` gets bumped and loading an
old artifact fails loudly rather than silently misreading it.

## What you get

Three files per `(layer, pooling, split)` cell, in `outputs/contrastive/`:

```
X_L{layer}_{pooling}_{split}.npz         # the matrix
X_L{layer}_{pooling}_{split}.rows.jsonl  # one row of metadata per matrix row
X_L{layer}_{pooling}_{split}.meta.json   # shapes, diagnostics, provenance
```

Example: `X_L40_last_prompt_token_extract.npz`

## The matrix

| Property | Value |
|---|---|
| npz key | `X` |
| shape | `(N, d)` — N matched pairs, d = hidden dim (**8192** for the 49B) |
| dtype | `float64` |
| definition | `X = H_eval − H_deploy` |
| row i | pair i, in the order given in `.rows.jsonl` |

```python
import numpy as np
with np.load("X_L40_last_prompt_token_extract.npz") as f:
    X = f["X"]          # (N, 8192)
```

Or use the loader, which re-validates on the way in:

```python
from evalaware.contrastive import load_contrastive_matrix
cm = load_contrastive_matrix("X_L40_last_prompt_token_extract.npz")
cm.x, cm.rows, cm.layer, cm.pooling, cm.diagnostics
```

## Guarantees step 1 makes

1. **Row alignment.** Row i of `H_eval` and row i of `H_deploy` are the same
   underlying question with the same template pair, differing only in framing.
   Enforced per-row against `pair_key`, not merely by row count.
2. **Polarity.** `X = H_eval − H_deploy`. A positive projection means the
   evaluation-framed side is higher. Framing comes from explicit labels; a
   swapped call raises rather than silently producing `−X`.
3. **No degenerate rows.** No NaN, no inf, no zero rows, no duplicate pair keys.
4. **Row order is load-bearing.** `.npz` and `.rows.jsonl` share it.
   `load_contrastive_matrix` rejects a file whose `row_index` values have been
   reordered. **Do not sort the rows.**
5. **Split disjointness.** `extract`, `ood_holdout` and `behavioral` share no
   template family and no question, so an OOD check on the holdout is honest.

## What step 1 does NOT do

No direction is estimated. No SVD, no normalization, no centering, no
mean subtraction. `X` is the raw difference. All of that is yours.

## Row metadata schema

Each line of `.rows.jsonl`:

```json
{
  "row_index": 0,
  "pair_key": "explicit|BigCodeBench/724|E1+UE1|D1+UD1",
  "question_id": "BigCodeBench/724",
  "regime": "explicit",
  "source": "bigcodebench",
  "template_family_eval": "E1/UE1",
  "template_family_deploy": "D1/UD1",
  "system_id_eval": "E1", "system_id_deploy": "D1",
  "user_id_eval": "UE1", "user_id_deploy": "UD1"
}
```

`question_id` is the clustering unit for any bootstrap — rows sharing a question
are not independent, and question-level clustering is what Hua et al. use
(Appendix I.2). Row-level resampling will give CIs that are far too narrow.

## Decisions that are yours, not ours

**1. Sign convention after SVD.** `numpy.linalg.svd` returns `v₁` only up to
sign, and which one you get can vary by LAPACK build — so an unfixed sign is a
cross-machine reproducibility bug. Fix it deterministically. The natural
convention, matching `X`'s polarity:

```python
if v1 @ X.mean(axis=0) < 0:
    v1 = -v1        # r_hat now points deploy -> eval
```

**2. Whether to center.** We deliberately do **not** center `X`. The mean of
`X` *is* the signal — it is the difference-of-means direction. Centering
removes exactly what you are looking for and leaves `v₁` describing variation
in the contrast across items. If you want that as a diagnostic, compute it
separately; do not make it the default.

**3. Cross-check against difference of means.** For matched pairs,
`mean(X) == μ_eval − μ_deploy` exactly. Report
`cos(v₁, mean(X)/‖mean(X)‖)` and `σ₁²/Σσᵢ²`. Their agreement is the evidence
that the signal is cleanly one-dimensional, which §3.2 of the proposal asks for.
Judge that cosine against `validation.random_pair_cosine_baseline(8192)`, never
against zero — in 8192 dimensions a random `|cos|` is about 0.01, so 0.05 is
large.

## Things worth knowing before you decompose

- **N matters a lot here.** Hua et al.'s released configs record
  `num_prompts: 16` — their contrastive matrix had 16 rows in 8192 dimensions,
  which is why they used a difference of means rather than SVD. At N=16, `v₁`
  is mostly fitting noise. Our design crosses questions × templates to get
  thousands of rows, which is what makes SVD viable. **Check `n_pairs` in
  `meta.json` before trusting `v₁`** — if a run produced few rows, you have
  inherited their problem.
- **`consistency_ratio` in `meta.json`** is `‖mean(X)‖ / mean(‖Xᵢ‖)`, in [0,1].
  Compare it to `consistency_ratio_at_chance` (= 1/√N). If it is not clearly
  above chance, there is no consistent contrast in that cell and no
  decomposition will manufacture one — check the prompts before debugging the
  SVD.
- **Layers are 0..80** for the 49B (81 hidden-state tensors: index 0 is the
  embedding output, 1..80 are the transformer layers). Hua et al. found
  steering most effective in layers 10–30, so that band is a reasonable place
  to look first — but it is a prior, not a constraint.
- **Two pooling schemes** are produced: `last_prompt_token` (at offset **−2**,
  matching their configs — not −1) and `completion_mean`. Run both and report
  `cos(r̂_lasttok, r̂_compmean)`. High agreement means we can use the cheap one
  in Phase 2; low agreement is a real finding about where awareness lives.
- **Two cue regimes** are present: `explicit` (E-series vs D-series system
  prompts) and `woodlabs` (W-series vs D-series). The proposal requires the
  direction to replicate under both before we edit anything. You can filter by
  the `regime` field in the row metadata, or build separate matrices.

## Reference implementation

`src/evalaware/extraction.py` contains a working SVD implementation with the
sign fix, variance-explained reporting and the diff-of-means cross-check. It
was written before Phase 1 was split and is **reference only** — treat it as a
starting point or ignore it entirely. It is not part of the step 1 gate and is
not covered by the step 1 test suite.
