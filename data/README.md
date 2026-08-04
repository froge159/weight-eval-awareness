# Data

- `raw/` — original downloads and unprocessed artifacts (do not edit in place)
- `processed/` — cleaned, derived, or intermediate outputs from experiments

Large files should not be committed. Prefer external storage or DVC later if needed.
Common binary / dump extensions under `data/` are gitignored; keep only READMEs and `.gitkeep` files in git.
