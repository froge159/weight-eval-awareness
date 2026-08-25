# Phase 2 — weight-space orthogonalization

Implements proposal §3.3: every projection that writes into the residual
stream (`o_proj` in attention blocks, `down_proj` in MLPs) has its output
orthogonalized against the evaluation-awareness direction r̂ from
[`src/extraction`](../extraction):

```
W'_out = W_out - r_hat r_hat^T W_out
```

applied to the Wood Labs organism (`timhua/wood_v2_sftr4_filt` on
`nvidia/Llama-3_3-Nemotron-Super-49B-v1`), a rank-64 LoRA adapter over
`o_proj`/`down_proj` — the exact matrices being orthogonalized, so the
adapter must be merged before editing (see `surgery.py`).

## Method summary

- **Per-block, not one shared direction.** Phase 1 measured r̂ at 20 layers.
  The cross-layer cosine matrix (`directions.py`) shows the residual-stream
  basis rotates continuously with depth rather than staying fixed, so each
  block is assigned the nearest measured direction (`assign_blocks`) instead
  of applying one vector everywhere.
- **Band: layers 36–76.** Outside this band the direction fails a cross-cue
  transfer check (see `src/extraction`'s step 3a results); the edit is
  restricted to where the direction is validated.
- **Fidelity is checked at two levels** (`fidelity.py`): module-output
  equivalence is an exact algebraic identity for bias-free linear layers and
  is the strict pass/fail gate (measured worst-case relative error 1.65e-3,
  0.4x bf16 rounding); residual-stream leakage outside the edited band is
  reported but not gated, since a band-restricted edit cannot drive it to
  zero by construction.

## Modules

| Module | What it does |
|---|---|
| `directions.py` | loads the Phase 1 per-layer r̂ set, builds the cross-layer cosine matrix, resolves which block gets which direction |
| `surgery.py` | LoRA merge + verification, module census, the rank-1 weight edit |
| `fidelity.py` | module-output equivalence gate, residual-stream leakage profile, model-health diagnostic |
| `run_orthogonalize.py` | end-to-end CLI |

## Usage

```bash
# Check the shared-basis assumption and pick the band, without a GPU
uv run python src/steering/run_orthogonalize.py --stage validate -v

# Confirm the module matcher against the real architecture, without weights
uv run python src/steering/run_orthogonalize.py --stage edit --dry-run

# Full run, band chosen by the stability analysis
uv run python src/steering/run_orthogonalize.py

# Force a specific band and materialize an edited checkpoint on disk
uv run python src/steering/run_orthogonalize.py --band 36-76 \
    --export checkpoints/edited_L36-76
```

By default no checkpoint is written — the artifact is a manifest plus the
resolved per-block directions, and the edited model is regenerated
deterministically by code. Pass `--export DIR` when another tool needs an
actual checkpoint.

**Hardware.** One A100 (or similar 40GB card) plus ~180GB system RAM for CPU
offload, ~100GB free disk for the HF cache, and another ~100GB if exporting a
checkpoint. Full pipeline (merge, edit, fidelity checks): ~9 minutes on cached
weights.

## Outputs

Written to `outputs/steering/`:

| Artifact | What it is |
|---|---|
| `band_report.json` | full cross-layer cosine matrix, threshold, chosen band, block-to-layer assignment |
| `cross_layer_cosine.npz` | the raw cosine matrix backing `band_report.json` |
| `census_*.json` | module census: which modules matched, in-band vs total |
| `edit_manifest_*.json` | per-module removal ratios and fidelity results |
| `fidelity_report_*.json` | the fidelity numbers above, standalone |
| `manifest_phase2_*.json` | full run provenance: git commit, seeds, package versions |
| `r_hat_perblock_*.npz` | the resolved per-block directions used in the edit |
