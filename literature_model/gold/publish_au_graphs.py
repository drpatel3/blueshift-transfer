"""Extract per-doc graph JSONs from each graph-build job's model.tar.gz and
upload them to s3://mineral-pipeline-pipeline/graphs_au/ so stage_predictor
can read them via the standard graphs-prefix loader.

Each SageMaker graph-build run writes graphs/<doc>.json into its model.tar.gz
artifact (output_dir/graphs/). The graphs_au/ checkpoint prefix only contains
metadata. This script bridges them.

Usage:
    python dev/gold/publish_au_graphs.py --jobs graph-build-20260507150130 \
        graph-build-20260507152346 graph-build-20260507154535 \
        graph-build-20260507161048
    python dev/gold/publish_au_graphs.py --auto    # picks all build-* jobs
                                                   # whose checkpoint-prefix
                                                   # was graphs_au/
"""

from __future__ import annotations

import argparse
import logging
import re
import tarfile
import tempfile
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "mineral-pipeline-pipeline"
DEFAULT_PREFIX = "graphs_au/"


def _publish_one(s3, sm, job_name: str, bucket: str, prefix: str) -> int:
    desc = sm.describe_training_job(TrainingJobName=job_name)
    if desc["TrainingJobStatus"] != "Completed":
        logger.warning(f"{job_name}: status={desc['TrainingJobStatus']}, skipping")
        return 0
    model_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
    parts = model_uri.replace("s3://", "").split("/", 1)
    src_bucket, src_key = parts[0], parts[1]

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "model.tar.gz"
        logger.info(f"{job_name}: downloading {model_uri}")
        s3.download_file(src_bucket, src_key, str(tar_path))

        extract_dir = Path(tmp) / "extract"
        extract_dir.mkdir()
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        graphs_dir = extract_dir / "graphs"
        if not graphs_dir.exists():
            logger.warning(f"{job_name}: no graphs/ inside model.tar.gz")
            return 0

        uploaded = 0
        for f in graphs_dir.glob("*.json"):
            # Skip metadata files — they belong in graphs_au/ already
            if f.name in {
                "checkpoint.json", "graphs_summary.json", "stage_vocab.json",
                "attempt_marker.json",
            }:
                continue
            s3.upload_file(str(f), bucket, prefix + f.name)
            uploaded += 1
        logger.info(f"{job_name}: uploaded {uploaded} graphs")
        return uploaded


def _auto_jobs(sm) -> list[str]:
    """List recent graph-build-* jobs (within last 24h)."""
    jobs = []
    paginator = sm.get_paginator("list_training_jobs")
    for page in paginator.paginate(SortBy="CreationTime", SortOrder="Descending",
                                   NameContains="graph-build", MaxResults=20):
        for j in page["TrainingJobSummaries"]:
            jobs.append(j["TrainingJobName"])
        break
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", help="Explicit graph-build job names")
    parser.add_argument("--auto", action="store_true",
                        help="Discover recent graph-build-* jobs automatically")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    if not args.jobs and not args.auto:
        parser.error("Pass --jobs <names...> or --auto")

    s3 = boto3.client("s3")
    sm = boto3.client("sagemaker")

    job_names = args.jobs or _auto_jobs(sm)
    if not job_names:
        raise SystemExit("No graph-build jobs found")

    total = 0
    for name in job_names:
        total += _publish_one(s3, sm, name, args.bucket, args.prefix)
    logger.info(f"Total uploaded: {total} graph JSONs to s3://{args.bucket}/{args.prefix}")


if __name__ == "__main__":
    main()
