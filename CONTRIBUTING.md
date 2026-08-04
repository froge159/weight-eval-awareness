# Contributing

## Branches and PRs

- Branch from `main` with a short descriptive name (`feat/...`, `fix/...`, `exp/...`).
- Prefer small PRs with a clear research or engineering goal.

## Where code goes

| Kind of work | Put it in |
|--------------|-----------|
| Feature / direction extraction | `src/extraction/` |
| Steering / intervention logic | `src/steering/` |
| Validation / evaluation checks | `src/validation/` |
| One-off exploration | `notebooks/` |
| Datasets | `data/raw/` or `data/processed/` |
| Model checkpoints | `checkpoints/` |
| Run artifacts and results | `outputs/` |

Promote notebook logic into the matching `src/` module once it is shared. Prefer `uv run` so everyone uses the same environment from `uv.lock`.

## Dependencies

Add runtime deps with `uv add <package>` and dev tooling with `uv add --dev <package>`, then commit the updated `pyproject.toml` and `uv.lock`.
