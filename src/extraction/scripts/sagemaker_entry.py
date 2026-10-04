#!/usr/bin/env python3
"""SageMaker Training Job entry point for the causal-test baseline/edited runs.

Runs inside the training container. Installs the pinned dependencies not
already in the base image (transformers==4.48.3 etc., per pyproject.toml),
then invokes run_phase3.py exactly as the local runbook does, pointing
--out at the SageMaker checkpoint directory so it is synced to S3 during
the run and restored automatically if the job is resubmitted with the same
checkpoint_s3_uri. That resume is safe because run_phase3.py's own
generation loop already skips completed responses on re-run (see its
module docstring) -- this script only needs to hand it a persistent --out.

RUN_MODE env var selects which run_phase3.py invocation to make:
  baseline  -- unedited organism, explicit cue, extract split
  edited    -- same split, with the completion_mean explicit-cue edit applied
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

CHECKPOINT_DIR = Path("/opt/ml/checkpoints")
MODEL_DIR = Path("/opt/ml/model")
REPO_ROOT = Path(__file__).resolve().parents[3]
EXTRACTION_DIR = REPO_ROOT / "src" / "extraction"
RUN_PHASE3 = EXTRACTION_DIR / "scripts" / "run_phase3.py"

PINNED_DEPS = [
    "transformers==4.48.3",
    "accelerate>=1.14.0",
    "peft>=0.20.0",
    "safetensors>=0.8.0",
    "huggingface-hub>=0.36.2",
    "pyyaml>=6.0.3",
]

SMOKE_ARGS = [
    "--stage", "all",
    "--split", "extract",
    "--question-fractions", "0.15", "0.35", "0.5",
    "--regime", "explicit",
    "--tag", "smoke_check",
    "--limit", "2",
    "--out", str(CHECKPOINT_DIR / "smoke_check"),
    "--gpu-mem", "38GiB",
    "--batch-size", "2",
]

RUNS = {
    "baseline": [
        "--stage", "all",
        "--split", "extract",
        "--question-fractions", "0.15", "0.35", "0.5",
        "--regime", "explicit",
        "--tag", "organism_cm_explicit",
        "--out", str(CHECKPOINT_DIR / "phase3_cm_explicit_baseline"),
        # g6e.12xlarge is 4x L40S 48GB each; leave real headroom per card for
        # KV-cache growth during generation (see run_phase3.py's per-GPU
        # max_memory fix) and shrink the batch so a retry after an earlier
        # OOM isn't just as likely to OOM again.
        "--gpu-mem", "38GiB",
        "--batch-size", "2",
    ],
    "edited": [
        "--stage", "all", "--edited",
        "--split", "extract",
        "--question-fractions", "0.15", "0.35", "0.5",
        "--regime", "explicit", "--band", "36", "76",
        "--pooling", "completion_mean",
        "--directions-dir", "outputs/directions_cm",
        "--direction-key", "r_hat_explicit",
        "--tag", "edited_cm_explicit",
        "--out", str(CHECKPOINT_DIR / "phase3_cm_explicit_edited"),
        # Same reasoning as baseline's: real per-GPU headroom on a 48GB card,
        # smaller batches -- this path's own GPU dispatch had the same
        # no-headroom bug as the baseline loader (see run_phase3.py), now
        # fixed the same way with this flag.
        "--gpu-mem", "38GiB",
        "--batch-size", "2",
    ],
}


def main() -> int:
    mode = os.environ.get("RUN_MODE", "baseline")
    if mode not in RUNS:
        print(f"RUN_MODE={mode!r} not recognized, expected one of {list(RUNS)}",
              file=sys.stderr)
        return 1

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"installing pinned deps: {PINNED_DEPS}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED_DEPS],
                    check=True)

    # A 2-question check first: if model loading or generation OOMs, we want
    # to know within minutes, not after hours of a run that never had a
    # chance. Uses the same --gpu-mem/--batch-size as the real run, so it
    # exercises the exact device placement that failed last time.
    print("running smoke check (2 questions) before committing to the full run")
    smoke = subprocess.run(
        [sys.executable, str(RUN_PHASE3), *SMOKE_ARGS],
        cwd=str(EXTRACTION_DIR),
    )
    if smoke.returncode != 0:
        print("smoke check failed -- not starting the full run", file=sys.stderr)
        return smoke.returncode
    print("smoke check passed")

    run_args = list(RUNS[mode])
    limit = os.environ.get("LIMIT", "")
    if limit:
        run_args += ["--limit", limit]

    print(f"running run_phase3.py in mode={mode!r} args={run_args}")
    result = subprocess.run(
        [sys.executable, str(RUN_PHASE3), *run_args],
        cwd=str(EXTRACTION_DIR),
    )

    # Checkpoint sync runs on an interval; also copy the final state straight
    # into /opt/ml/model, which SageMaker uploads to output_path unconditionally
    # when the job exits 0, so the result isn't waiting on the next sync tick.
    out_dir = CHECKPOINT_DIR / f"phase3_cm_explicit_{mode}"
    if out_dir.exists():
        dest = MODEL_DIR / out_dir.name
        shutil.copytree(out_dir, dest, dirs_exist_ok=True)
        print(f"copied {out_dir} -> {dest}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
