# weight-eval-awareness

Group research on evaluation awareness — whether and how models detect when
they are being evaluated — with a focus on weight-space methods.

**Results:** [`PHASE3_RESULTS.md`](PHASE3_RESULTS.md) is the write-up: the
behavioral gap replicates Hua et al., the weight-space edit removes the
awareness direction from the weights but does not close the gap, and every
control passes.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --group dev
```

Copy `.env.example` to `.env` when you need API keys (do not commit `.env`).

## Common commands

```bash
uv run pytest
uv run ruff check .
uv run jupyter lab
```

## Layout

| Path | Purpose |
|------|---------|
| `src/extraction/` | Phase 1: contrastive matrix, direction extraction, and the step 3a/3b sanity checks, capability benchmark, and Phase 3 behavioral runners — see [its README](src/extraction/README.md) |
| `src/steering/` | Phase 2: the weight-space orthogonalization edit — see [its README](src/steering/README.md) |
| `src/validation/` | Reserved for standalone validation/evaluation checks; not yet started |
| `notebooks/` | Exploratory analysis and the Colab notebook used to run Phase 3 on an A100 |
| `data/raw/`, `data/processed/` | Datasets and derived artifacts (mostly gitignored — see `data/README.md`) |
| `outputs/` | Committed run artifacts: extracted directions, steering manifests and fidelity reports |
| `checkpoints/` | Model checkpoints (gitignored; large) |

See [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration conventions.
