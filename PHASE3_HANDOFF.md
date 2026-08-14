# HANDOFF — Phase 2 → Phase 3 (Validation)

Everything the validation step needs, plus what will silently mislead it if not
handled. Written 2026-08-14.

Phase 2 is the orthogonalization from proposal §3.3:

    W'_out = W_out − r̂ r̂ᵀ W_out

applied to the attention and MLP output projections that write to the residual
stream, in layers 36–76, using a **per-block** direction (see §1). The edit is
implemented, verified, and reproducible. It is not exported to disk by default
— see §2 for how to get a model in hand.

---

## TL;DR — the five things that matter

1. **The direction is per-block, not one shared vector.** Each block draws on
   the nearest Phase 1 direction to what it writes into. The cross-layer cosine
   matrix is smooth and band-diagonal (adjacent layers agree at 0.7–0.96, the
   two ends of 36–76 agree at only 0.26), so one shared direction across the
   whole band is not defensible — only the per-block scheme is. See §1.
2. **There is no checkpoint on disk unless you ask for one.** The artifact is a
   manifest plus the resolved per-block directions (~KB); the edited model is
   regenerated deterministically by code. Pass `--export DIR` if you need an
   actual checkpoint to load elsewhere. See §2.
3. **Fidelity passed, but read what it actually proved.** The strict gate is
   exact algebra on the weights, not "activations equal the ideal projection."
   The activation-level identity the proposal describes cannot hold under a
   band-restricted edit and was not expected to — see §3 before treating any
   residual signal as a bug.
4. **No controls have been run.** Proposal §3.4's random-direction and
   failed-band controls, without which "gap closure" has no meaning, are not
   built. This is explicitly out of scope for Phase 2 — see §5.
5. **Step 3b and the behavioral baseline are still outstanding**, unchanged
   from `PHASE2_HANDOFF.md` §6. Phase 3 numbers built on this edit inherit that
   gap. See §6.

---

## 1. Which direction, at which layers, and why per-block

Phase 1 shipped one `r̂` per measured layer (4, 8, 12, ..., 80 — 20 files in
`outputs/directions/`). Proposal §3.3 writes the edit as if there's a single
`r̂`; Phase 2 had to decide how a 20-point sweep becomes an edit applied across
41 consecutive blocks.

**The cross-layer cosine matrix says the basis rotates continuously with
depth**, not abruptly:

```
      36     40     44     48     52     56     60     64     68     72     76
36  1.000  0.708  0.582  0.528  0.498  0.438  0.404  0.389  0.371  0.312  0.260
40  0.708  1.000  0.839  0.754  0.708  0.624  0.582  0.563  0.537  0.457  0.372
...
76  0.260  0.372  0.451  0.510  0.549  0.620  0.662  0.690  0.727  0.780  1.000
```

Adjacent measured layers agree well (0.71–0.96); the two ends of the 36–76
band agree at only 0.26. Full matrix and threshold math in
`outputs/steering/band_report.json`.

This means the §3.3 "shared basis" check gives a **different answer depending
on how the direction is applied**:

- A **single shared `r̂`** across the whole band needs every pair of layers in
  it to agree. By that criterion the 36–76 band **fails** (min pairwise cosine
  0.260, threshold 0.579) and collapses to a narrower 44–68.
- A **per-block direction**, where each block snaps to its nearest measured
  layer, only needs *adjacent* measured layers to agree — nothing is
  transported across the full band. By that criterion 36–76 **passes** (min
  adjacent-layer cosine 0.708 ≥ 0.579).

Phase 2 uses per-block directions, so the full 36–76 band is what got edited.
**If you rerun this with a single shared direction, expect the tool to narrow
the band to ~44–68 instead of 36–76** — that's `directions.py`'s
`analyze_cross_layer(..., direction_mode="shared")` behaving correctly, not a
regression.

Block-to-layer indexing: Phase 1 captured `hidden_states[l]`, where index 0 is
the embedding output and index `l` is the output of block `l-1`. So block `b`
writes into residual position `b+1`, and under the default `write_target`
convention block `b`'s direction is the nearest measured layer to `b+1`. Band
36–76 → blocks 35–75 (41 blocks). The full block→layer table is in
`outputs/steering/band_report.json` under `plan.assignments`.

## 2. Getting the edited model

**No 100GB checkpoint was written.** The design choice was a loader + manifest,
not a materialized export, so control variants stay cheap. To get the model:

```python
import sys; sys.path.insert(0, "src")
from steering.directions import load_direction_set, assign_blocks
from steering.surgery import EditSpec, load_edited_model

spec = EditSpec(band=(36, 76), layer_convention="write_target")
dset = load_direction_set()
plan = assign_blocks(dset, spec.band, n_blocks=80, convention=spec.layer_convention)
model, tokenizer, edit_report = load_edited_model(spec, plan)
```

This re-downloads the base model (cached after the first run), merges the LoRA
adapter, and reapplies the exact same 54-matrix edit. Verified deterministic
across three independent runs today (identical merge delta norm, identical
edited-module count, identical worst removal ratio).

If you need a real checkpoint directory (to hand to another tool, or off this
machine):

```bash
uv run python src/steering/run_orthogonalize.py --export checkpoints/edited_L36-76
```

Needs ~100GB free disk on top of the ~100GB HF cache.

**Hardware note:** loading + editing + running inference needs one A100 (or
similar 40GB card) plus ~180GB system RAM for CPU offload of what doesn't fit
on the GPU. The full pipeline here took about 9 minutes end to end on cached
weights (merge: ~2.5 min; edit: seconds; three 16-prompt forward passes for
fidelity: ~4 min combined).

## 3. What the fidelity check actually proved — read before citing it

Two different things were tested, and conflating them will misrepresent the
result.

**(a) Module-output equivalence — the strict gate, and it passed.**
`o_proj`/`down_proj` have no bias, so `W'x = (I − r̂r̂ᵀ)(Wx)` is exact algebra,
not an approximation. Measured worst-case relative error: **1.65e-3, 0.4× bf16
storage rounding**. This proves the weight edit is exactly the intended
ablation of those 54 modules' outputs — nothing more, nothing less.

**(b) Residual-stream leakage — reported, and it is *not* zero, on purpose.**
The proposal also asks whether edited activations match `h̃ = h − r̂r̂ᵀh`. Under
a band-restricted edit that identity **cannot** hold: blocks 0–34 and
`embed_tokens` still write along `r̂`, and ablation changes downstream
computation nonlinearly (different attention softmaxes, different SwiGLU
gates), so the edited model computes a genuinely different function rather
than a projection of the original one. Measured:

| region | mean leakage ratio (after / before) |
|---|---|
| below the band (layers 4–32) | **1.000**, exactly, at every layer |
| inside the band (36–76) | **0.515** — roughly halved |
| layer 80 (downstream of the band) | 0.516 — inherits the effect |

The out-of-band exactness confirms the edit didn't spread beyond its target.
The in-band 0.515 is the honest number — **do not report this as "leakage
removed"**; it's a reduction, and the module docstring in
`src/steering/fidelity.py` explains why it can't be more than that under a
band-restricted edit. If Phase 3 needs closer-to-zero leakage, the lever is
`--band` width or adding `embed_tokens` (see §5), not this edit as shipped.

End-to-end hidden-state check (hooked-unedited vs. actually-edited, same 16
held-out prompts): max relative error 2.04e-2 against a 5e-2 tolerance, median
7.65e-3, logit relative error 2.28e-2, **100% top-1 token agreement**. This
tolerance is loose on purpose (bf16 rounding compounds across 54 edited
matrices); it's a structural sanity check, not the strict gate.

**(c) Model-health diagnostic vs. the unedited model — a sanity check, not a
fidelity claim.** Compares next-token distributions between the edited and
unedited model on the same 16 held-out prompts: 100% top-1 agreement, mean
KL(unedited‖edited) = **1.68e-5 nats**. Read this cautiously — it's measured
at the same position as everything else here (last prompt token, offset -2,
*before* generation), where next-token predictions are dominated by strong
local context. It rules out the edit having broken the model outright. It is
**not** evidence that behavior is unchanged once the model is actually
reasoning in its chain of thought, which is what Phase 3's behavioral
measurement will probe. If you want a sharper health signal, measure this same
KL at generated positions rather than the prompt boundary.

## 4. Module census, for sanity-checking a rerun

From `config.json`'s `block_configs` (nemotron-nas / `DeciLMForCausalLM`, 80
blocks, 31 of them with `attention.no_op: true`):

| | total in model | in band (blocks 35–75) |
|---|---|---|
| `o_proj` | 49 | 13 |
| `down_proj` | 80 | 41 |
| **total** | **129** | **54** |

If a rerun reports 160 total or 82 in-band, the module matcher is treating the
80 blocks as uniform — see `PHASE2_HANDOFF.md` §3.

## 5. Controls Phase 3 still needs — not built here

Proposal §3.4/§5 controls were explicitly descoped for this pass:

- **Random-direction control** (same 54 modules, seeded random unit direction
  per block) — without this, gap closure has no meaning; it's the control that
  separates "removed the awareness direction" from "any large edit degrades
  behavior."
- **Failed-band control**, editing layers 20–24 with the same machinery — a
  stronger control than random, and cheap once the pipeline exists.
- **`embed_tokens` orthogonalization** (Arditi et al. also ablate the
  embedding; `tie_word_embeddings=False` here so `lm_head` is untouched by it)
  — would close more of the out-of-band leakage gap but goes beyond the exact
  §3.3 formula.
- **All-layer edit** (no band restriction) as a comparison point.

`src/steering/directions.py` exposes a single direction-provider seam
(`assign_blocks`) so each of these is a small addition, not a rewrite.

## 6. Outstanding blockers, carried over unchanged

Both of these predate Phase 2 and are unaffected by it — repeated here so
Phase 3 doesn't assume they were picked up along the way:

- **Step 3b (refusal-entanglement check) has not been run.** This is the
  proposal's own stated principal threat to validity — that `r̂` is entangled
  with general refusal rather than isolated to evaluation-awareness. See
  `PHASE2_HANDOFF.md` §6.
- **No behavioral baseline** (`s_eval`, `s_deploy`) has been measured on this
  organism, so ρ's denominator is still unmeasured.

## 7. Inventory

| Artifact | What it is |
|---|---|
| `src/steering/directions.py` | direction loading, cross-layer stability, band selection, per-block assignment |
| `src/steering/surgery.py` | LoRA merge + verification, module census, the weight edit |
| `src/steering/fidelity.py` | equivalence checks, leakage profile, model-health diagnostic |
| `src/steering/run_orthogonalize.py` | the end-to-end CLI (`--stage validate\|edit\|all`, `--dry-run`, `--export`) |
| `outputs/steering/band_report.json` | full cosine matrix, threshold, chosen band, block→layer assignment table |
| `outputs/steering/edit_manifest_L36-76_write_target_*.json` | census, per-module removal ratios, fidelity results |
| `outputs/steering/fidelity_report_L36-76_write_target_*.json` | the fidelity numbers in §3, standalone |
| `outputs/steering/manifest_phase2_L36-76_write_target_*.json` | full run provenance: git commit, seeds, package versions |
| `outputs/steering/r_hat_perblock_L36-76_write_target_*.npz` | the 41 resolved per-block directions used in the edit |

**Reproducibility note:** the manifest records `git.dirty: true` — the code
that produced these artifacts was not committed at run time. If exact
reproducibility matters to Phase 3, commit `src/steering/` first and note the
commit hash alongside these outputs.
