# Contributing

## Branches and PRs

- Branch from `main` with a short descriptive name (`feat/...`, `fix/...`, `exp/...`).
- Prefer small PRs with a clear research or engineering goal.
- Run `uv run pytest` and `uv run ruff check .` before opening a PR.

## Where code goes

| Kind of work | Put it in |
|--------------|-----------|
| One-off exploration | `notebooks/` |
| Runnable protocol others can reproduce | `experiments/` (e.g. `001_short_slug.py`) |
| Stable, reusable logic | `src/weight_eval_awareness/` |
| Lit notes and drafts | `papers/` |
| Datasets / outputs | `data/raw/` or `data/processed/` |

Promote notebook logic into `src/` once it is shared across experiments. Prefer `uv run` so everyone uses the same environment from `uv.lock`.

## Dependencies

Add runtime deps with `uv add <package>` and dev tooling with `uv add --dev <package>`, then commit the updated `pyproject.toml` and `uv.lock`.
