# weight-eval-awareness

Group research on evaluation awareness — whether and how models detect when they are being evaluated — with a focus on weight-space methods.

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
| `src/weight_eval_awareness/` | Shared, reusable library code |
| `experiments/` | Numbered, runnable experiment protocols |
| `notebooks/` | Exploratory analysis |
| `papers/` | Notes, bibliographies, drafts |
| `data/raw/`, `data/processed/` | Datasets and derived artifacts (mostly gitignored) |
| `tests/` | Unit and smoke tests |

See [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration conventions.
