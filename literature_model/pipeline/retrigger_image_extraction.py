"""Re-trigger Lambda image+flowsheet extraction on docs that ran in an earlier
`SKIP_IMAGES=true` pass.

Uses the Lambda handler's built-in resumability: `steps_completed` already
contains `"tables"` and `"text"` for these docs, so re-triggering only runs
the missing image/flowsheet/LLM step. No duplicate table work, no double
spend on tables.

How it re-triggers
------------------
S3 PutObject events fire the Lambda. The cheapest/safest way to re-generate
an event for an object that already exists is a copy-in-place
(`copy_object` with the same source and destination key) — S3 treats that
as a new PUT.

The target doc list comes from `quantitative/audit_unlabeled_gold.py`:
docs where `grade_map_au.json` has a valid Au head grade but no V2
stage predictions exist. The audit showed 100% of the 294 fall in
`never_extracted_images` — tables+text present, images missing.

Safety
------
Default is `--dry-run` (prints what would be triggered, no writes).
Pass `--execute` to actually copy-in-place. A `--limit N` caps the run
so you can do a small smoke test before firing all 294.

Usage:
    python pipeline/retrigger_image_extraction.py                 # dry-run, all docs
    python pipeline/retrigger_image_extraction.py --limit 5       # dry-run, 5 docs
    python pipeline/retrigger_image_extraction.py --execute       # fire all
    python pipeline/retrigger_image_extraction.py --execute --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEV_ROOT = Path(__file__).resolve().parent.parent
AU_MAP_PATH = DEV_ROOT / "quantitative" / "grade_map_au.json"
CHECKPOINT_DIR = DEV_ROOT / "classification" / "xgb_checkpoint_v2"

PIPELINE_BUCKET = "mineral-pipeline-pipeline"
PDFS_PREFIX = "pdfs/"
RESULTS_PREFIX = "results/"
LAMBDA_FUNCTION_ENV_SKIP = "SKIP_IMAGES"


def _labeled_doc_ids() -> set:
    """Doc IDs that already appear in V2 predictions (val + oof + test)."""
    ids = set()
    for fname in ["val_predictions.json", "oof_predictions.json",
                   "test_predictions.json"]:
        path = CHECKPOINT_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        per_doc = data["per_doc"] if isinstance(data, dict) else data
        for entry in per_doc:
            ids.add(entry["doc_id"])
    return ids


def _build_docid_to_result_key(s3) -> dict:
    """Map document_id → results/{pdf_stem}.json. Reads JSON heads."""
    logger.info("Mapping document_id -> result key (reads first 300B of each result)...")
    mapping = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=PIPELINE_BUCKET, Prefix=RESULTS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                resp = s3.get_object(Bucket=PIPELINE_BUCKET, Key=key,
                                     Range="bytes=0-400")
                head = resp["Body"].read().decode("utf-8", errors="replace")
                m = re.search(r'"document_id"\s*:\s*"([^"]+)"', head)
                if m:
                    mapping[m.group(1)] = key
            except Exception:
                continue
    logger.info(f"Mapped {len(mapping)} doc IDs")
    return mapping


def _warn_if_lambda_skips_images(lam) -> None:
    """Look up the Lambda env var for SKIP_IMAGES and warn if it's true.
    Non-fatal — we just don't know the function name, so we're conservative."""
    try:
        paginator = lam.get_paginator("list_functions")
        for page in paginator.paginate():
            for fn in page.get("Functions", []):
                name = fn.get("FunctionName", "")
                if "extract" not in name.lower() and "pipeline" not in name.lower():
                    continue
                env = fn.get("Environment", {}).get("Variables", {}) or {}
                val = str(env.get(LAMBDA_FUNCTION_ENV_SKIP, "false")).lower()
                if val in ("true", "1", "yes"):
                    logger.warning(
                        f"Lambda {name} has SKIP_IMAGES={val} — retrigger will "
                        f"re-skip images. Update the stack env var first."
                    )
    except Exception:
        pass  # Best-effort; don't block on IAM/permissions


def _retrigger_one(s3, filename: str) -> tuple[str, bool, str]:
    """Self-copy the PDF to the same key — triggers an S3 PUT event."""
    key = f"{PDFS_PREFIX}{filename}"
    try:
        # Re-read retry_count from existing result and reset it so the resumable
        # path doesn't hit MAX_RETRIES after a few previous runs.
        result_key = f"{RESULTS_PREFIX}{Path(filename).stem}.json"
        try:
            resp = s3.get_object(Bucket=PIPELINE_BUCKET, Key=result_key)
            doc = json.loads(resp["Body"].read())
            if doc.get("retry_count", 0) >= 2:
                doc["retry_count"] = 0
                s3.put_object(Bucket=PIPELINE_BUCKET, Key=result_key,
                              Body=json.dumps(doc).encode("utf-8"))
        except Exception:
            pass  # Best effort; lambda handles missing result fine

        s3.copy_object(
            Bucket=PIPELINE_BUCKET,
            Key=key,
            CopySource={"Bucket": PIPELINE_BUCKET, "Key": key},
            MetadataDirective="REPLACE",
            Metadata={"retrigger": "image_extraction"},
        )
        return filename, True, ""
    except Exception as e:
        return filename, False, str(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--execute", action="store_true",
                        help="Actually retrigger. Without this, runs in dry-run mode.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of PDFs retriggered (useful for smoke tests)")
    parser.add_argument("--workers", type=int, default=20,
                        help="Concurrent S3 copy threads (default: 20)")
    args = parser.parse_args()

    with open(AU_MAP_PATH) as f:
        au_map = json.load(f)
    au_ids = set(au_map.keys())

    labeled = _labeled_doc_ids()
    unlabeled = au_ids - labeled
    logger.info(
        f"Au-in-range: {len(au_ids)} | already labeled: "
        f"{len(au_ids & labeled)} | needing image extraction: {len(unlabeled)}"
    )

    s3 = boto3.client("s3")

    # Resolve each doc_id to its PDF filename via the existing results JSON.
    docid_to_result_key = _build_docid_to_result_key(s3)

    targets = []  # list of (doc_id, filename)
    missing_result = 0
    for doc_id in sorted(unlabeled):
        result_key = docid_to_result_key.get(doc_id)
        if not result_key:
            missing_result += 1
            continue
        try:
            resp = s3.get_object(Bucket=PIPELINE_BUCKET, Key=result_key)
            doc = json.loads(resp["Body"].read())
            filename = doc.get("filename")
            if filename:
                targets.append((doc_id, filename))
        except Exception as e:
            logger.warning(f"Could not read {result_key}: {e}")
            missing_result += 1

    if args.limit:
        targets = targets[:args.limit]
    logger.info(f"Resolved {len(targets)} PDFs to retrigger "
                f"({missing_result} result JSONs missing/unreadable)")

    if args.execute:
        _warn_if_lambda_skips_images(boto3.client("lambda"))

    if not args.execute:
        print(f"\nDRY RUN — would retrigger {len(targets)} PDFs via self-copy on "
              f"s3://{PIPELINE_BUCKET}/{PDFS_PREFIX}...")
        for doc_id, filename in targets[:10]:
            print(f"  {doc_id[:12]}...  {filename}")
        if len(targets) > 10:
            print(f"  ... and {len(targets) - 10} more")
        print("\nPass --execute to actually fire.")
        return

    logger.info(f"EXECUTING: retriggering {len(targets)} PDFs...")
    succeeded, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_retrigger_one, boto3.client("s3"), filename): (doc_id, filename)
            for doc_id, filename in targets
        }
        for i, f in enumerate(as_completed(futures), 1):
            filename, ok, err = f.result()
            if ok:
                succeeded += 1
            else:
                failed.append((filename, err))
                logger.warning(f"  FAILED {filename}: {err}")
            if i % 25 == 0:
                logger.info(f"  {i}/{len(targets)} retriggered "
                            f"({succeeded} ok, {len(failed)} failed)")

    logger.info(f"Done. {succeeded}/{len(targets)} retriggered. "
                f"{len(failed)} failed.")
    if failed:
        for name, err in failed[:10]:
            logger.warning(f"  {name}: {err}")

    logger.info("\nNext steps:")
    logger.info(f"  1. Watch CloudWatch Logs for the extract Lambda")
    logger.info(f"  2. Wait ~20-40 min for all PDFs to finish")
    logger.info(f"  3. Re-run `python quantitative/audit_unlabeled_gold.py` to verify")
    logger.info(f"  4. Re-run `python classification/predict.py --build-cache` "
                f"to refresh doc_edges.json")
    logger.info(f"  5. Retrain V2 on SageMaker with the enlarged labeled corpus")


if __name__ == "__main__":
    main()
