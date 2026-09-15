"""
Classify PDFs in S3 raw/ by mineral content and move copper PDFs to pdfs/.

Downloads each PDF temporarily, scans for mineral keywords, and moves
only those where copper >= threshold% of total mineral mentions.

Usage:
    python filter_move_s3.py
    python filter_move_s3.py --threshold 5 --workers 10 --pages 30
    python filter_move_s3.py --dry-run
"""

import argparse
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from classify_pdfs import classify_pdf, compile_patterns

BUCKET = os.getenv("PIPELINE_BUCKET", "mineral-pipeline-pipeline")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def classify_s3_pdf(s3_client, bucket, s3_key, compiled_patterns, max_pages):
    """Download a PDF from S3, classify it, return (s3_key, counts, pages)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        s3_client.download_file(bucket, s3_key, tmp_path)
        counts, pages_scanned = classify_pdf(tmp_path, compiled_patterns, max_pages)
        return s3_key, counts, pages_scanned, None
    except Exception as e:
        return s3_key, None, None, str(e)
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Filter S3 raw/ PDFs by copper content and move to pdfs/"
    )
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Min copper %% of total mineral mentions (default: 5)")
    parser.add_argument("--pages", type=int, default=30,
                        help="Max pages to scan per PDF (default: 30)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Concurrent download/classify threads (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify only, don't move files")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    # List all PDFs in raw/
    raw_keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="raw/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".pdf"):
                raw_keys.append(obj["Key"])

    logger.info(f"Found {len(raw_keys)} PDFs in raw/")
    if not raw_keys:
        return

    compiled = compile_patterns()
    copper_keys = []
    skipped = []
    errors = []

    actual_workers = min(args.workers, len(raw_keys))
    logger.info(f"Classifying with {actual_workers} threads, "
                f"scanning {args.pages} pages each, "
                f"threshold {args.threshold}%")

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = {}
        for key in raw_keys:
            client = boto3.client("s3")
            f = pool.submit(classify_s3_pdf, client, BUCKET, key,
                            compiled, args.pages)
            futures[f] = key

        done = 0
        for f in as_completed(futures):
            done += 1
            s3_key, counts, pages_scanned, error = f.result()
            filename = os.path.basename(s3_key)

            if error:
                logger.warning(f"  [{done}/{len(raw_keys)}] {filename[:60]:<60} ERROR: {error}")
                errors.append(s3_key)
                continue

            total_mentions = sum(counts.values())
            cu_count = counts.get("copper", 0)
            cu_pct = (cu_count / total_mentions * 100) if total_mentions > 0 else 0

            if cu_pct >= args.threshold:
                copper_keys.append(s3_key)
                logger.info(f"  [{done}/{len(raw_keys)}] {filename[:60]:<60} "
                            f"COPPER {cu_pct:5.1f}% ({cu_count}/{total_mentions})")
            else:
                skipped.append(s3_key)
                logger.info(f"  [{done}/{len(raw_keys)}] {filename[:60]:<60} "
                            f"skip   {cu_pct:5.1f}% ({cu_count}/{total_mentions})")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Copper PDFs (>= {args.threshold}%): {len(copper_keys)}")
    logger.info(f"Skipped: {len(skipped)}")
    logger.info(f"Errors: {len(errors)}")

    if args.dry_run:
        logger.info("DRY RUN — no files moved")
        for key in copper_keys:
            logger.info(f"  would move: {os.path.basename(key)}")
        return

    # Move copper PDFs from raw/ to pdfs/
    if not copper_keys:
        logger.info("No copper PDFs to move")
        return

    logger.info(f"\nMoving {len(copper_keys)} copper PDFs to pdfs/...")
    for i, key in enumerate(copper_keys, 1):
        filename = os.path.basename(key)
        dest_key = f"pdfs/{filename}"
        s3.copy_object(Bucket=BUCKET, CopySource={"Bucket": BUCKET, "Key": key},
                       Key=dest_key)
        s3.delete_object(Bucket=BUCKET, Key=key)
        logger.info(f"  [{i}/{len(copper_keys)}] {filename}")

    logger.info(f"Done. {len(copper_keys)} PDFs moved to pdfs/ — Lambda will process each.")


if __name__ == "__main__":
    main()
