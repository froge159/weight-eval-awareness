#!/usr/bin/env python3
"""Submit the causal-test baseline (or edited) run as a SageMaker Training Job.

Run this from AWS CloudShell (or any machine with AWS credentials configured
and the `sagemaker` package installed) -- it only submits the job over the
network; no GPU is needed to run this script itself.

    pip install -q sagemaker boto3
    export HF_TOKEN=hf_...
    python src/extraction/scripts/sagemaker_launch.py \\
        --role arn:aws:iam::<account-id>:role/<execution-role-name> \\
        --run baseline

Run from the repo root (source_dir defaults to ".") so the whole repo ships
to the container -- run_phase3.py's REPO constant is computed from its own
file location, not from the working directory, so it needs the real
src/extraction/... layout intact on the other end.

Re-running with the same --run value reuses the same checkpoint_s3_uri, so a
job that hits --max-run-hours or crashes resumes instead of restarting from
zero -- see sagemaker_entry.py's docstring for why that's safe.
"""
import argparse
import os
import sys

import sagemaker
from sagemaker.pytorch import PyTorch


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", choices=["baseline", "edited"], default="baseline")
    p.add_argument("--role", required=True,
                   help="SageMaker execution role ARN (find it under the Studio "
                        "domain's user profile, or search IAM for a role with "
                        "AmazonSageMakerFullAccess attached)")
    p.add_argument("--instance-type", default="ml.g6e.12xlarge")
    p.add_argument("--max-run-hours", type=float, default=8.0,
                    help="hard cap on job duration; the run is resumable if it "
                         "gets cut off, so this just bounds worst-case cost")
    p.add_argument("--source-dir", default=".",
                    help="repo root to package and ship to the container "
                         "(default: current directory -- run this from the repo root)")
    args = p.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is not set -- export it before running this.", file=sys.stderr)
        return 1

    session = sagemaker.Session()
    bucket = session.default_bucket()
    prefix = f"causal-test/{args.run}"

    # g6e instances ship with local NVMe instance storage that SageMaker uses
    # automatically for the container's working directory -- no volume_size
    # to size here, unlike the Studio space's capped EBS volume.
    estimator = PyTorch(
        entry_point="src/extraction/scripts/sagemaker_entry.py",
        source_dir=args.source_dir,
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.3",
        py_version="py311",
        max_run=int(args.max_run_hours * 3600),
        checkpoint_s3_uri=f"s3://{bucket}/{prefix}/checkpoints",
        output_path=f"s3://{bucket}/{prefix}/output",
        environment={"HF_TOKEN": hf_token, "RUN_MODE": args.run},
        disable_profiler=True,
        keep_alive_period_in_seconds=0,
    )

    estimator.fit(wait=False)
    name = estimator.latest_training_job.name
    print(f"submitted job: {name}")
    print(f"checkpoints/output bucket prefix: s3://{bucket}/{prefix}/")
    print("Monitor with:")
    print(f"  aws sagemaker describe-training-job --training-job-name {name} "
          f"--query TrainingJobStatus")
    print("Or in the console: SageMaker -> Training -> Training jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
