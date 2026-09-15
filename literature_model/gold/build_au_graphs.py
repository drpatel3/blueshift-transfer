"""Stage the gold-corpus graph build for SageMaker.

Translates dev/gold/gold_training_doc_ids.json (SHA-256 doc_ids) into the
per-doc result_keys that classification/graph_pipeline.py filters on, and
uploads the allowlist to s3://<checkpoint-bucket>/graphs_au/allowlist.json.

After running this, launch the existing graph-pipeline SageMaker job with:

    SM_HP_RESULTS_BUCKET=mineral-pipeline-pipeline
    SM_HP_RESULTS_PREFIX=results/
    SM_HP_CHECKPOINT_BUCKET=mineral-pipeline-pipeline
    SM_HP_CHECKPOINT_PREFIX=graphs_au/
    SM_HP_ALLOWLIST_S3_KEY=graphs_au/allowlist.json

graph_pipeline reads the allowlist and only emits graphs for those gold docs;
output lands in s3://mineral-pipeline-pipeline/graphs_au/.

Usage:
    python dev/gold/build_au_graphs.py                # dry-run (prints plan)
    python dev/gold/build_au_graphs.py --execute      # uploads allowlist
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEV_ROOT = Path(__file__).resolve().parent.parent
DOCID_INDEX_PATH = DEV_ROOT / "tmp" / "inspect" / "_docid_to_result_key.json"
ALLOWLIST_DOC_IDS = Path(__file__).resolve().parent / "gold_training_doc_ids.json"

DEFAULT_BUCKET = "mineral-pipeline-pipeline"
DEFAULT_KEY = "graphs_au/allowlist.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--execute", action="store_true",
                        help="Upload allowlist to S3 (default is dry-run).")
    args = parser.parse_args()

    if not ALLOWLIST_DOC_IDS.exists():
        raise SystemExit(
            f"{ALLOWLIST_DOC_IDS} missing. Run dev/gold/build_doc_id_allowlist.py first."
        )
    if not DOCID_INDEX_PATH.exists():
        raise SystemExit(f"Doc-ID index missing: {DOCID_INDEX_PATH}")

    with open(ALLOWLIST_DOC_IDS, encoding="utf-8") as f:
        doc_ids = set(json.load(f)["doc_ids"])
    with open(DOCID_INDEX_PATH, encoding="utf-8") as f:
        docid_to_key = json.load(f)

    result_keys = sorted({docid_to_key[d] for d in doc_ids if d in docid_to_key})
    payload = {"doc_ids": sorted(doc_ids), "result_keys": result_keys}

    logger.info(f"Allowlist: {len(doc_ids)} doc_ids -> {len(result_keys)} result_keys")
    logger.info(f"Target: s3://{args.bucket}/{args.key}")

    if not args.execute:
        logger.info("Dry-run. Pass --execute to upload.")
        return

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=args.bucket,
        Key=args.key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"Uploaded allowlist to s3://{args.bucket}/{args.key}")
    logger.info(
        "Now launch the graph-pipeline SageMaker job with "
        f"SM_HP_CHECKPOINT_PREFIX=graphs_au/ and SM_HP_ALLOWLIST_S3_KEY={args.key}."
    )


if __name__ == "__main__":
    main()
