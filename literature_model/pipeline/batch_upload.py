"""
Upload a folder of PDFs to S3 to trigger Lambda extraction.

Supports mineral gating: when --manifest and --mineral are provided,
only PDFs where the target mineral is in the manifest's "relevant" list
(>= threshold % of total mineral mentions) are uploaded.

Usage:
    python batch_upload.py <bucket-name> <pdf-folder>
    python batch_upload.py mineral-pipeline-pipeline ./test_pdfs/main_pdfs
    python batch_upload.py mineral-pipeline-pipeline ./pdfs --manifest ./pdf_manifest.json --mineral copper
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def list_existing_keys(bucket_name, prefix):
    """Return a set of basenames already present under s3://bucket/prefix."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    existing = set()
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            existing.add(Path(obj["Key"]).name)
    return existing


def load_manifest(manifest_path):
    """Load the mineral classification manifest JSON."""
    with open(manifest_path) as f:
        data = json.load(f)
    return data.get("files", {})


def filter_by_mineral(pdf_files, manifest, mineral):
    """Filter PDFs to only those where the target mineral is relevant."""
    accepted = []
    skipped = []
    missing = []

    for pdf_path in pdf_files:
        entry = manifest.get(pdf_path.name)
        if entry is None:
            missing.append(pdf_path)
            continue
        if mineral in entry.get("relevant", []):
            accepted.append(pdf_path)
        else:
            skipped.append(pdf_path)

    return accepted, skipped, missing


def _upload_one(s3_client, bucket_name, pdf_path, prefix, index, total):
    """Upload a single PDF to S3. Used as a thread target."""
    s3_key = f"{prefix}{pdf_path.name}"
    s3_client.upload_file(str(pdf_path), bucket_name, s3_key)
    logger.info(f"  [{index}/{total}] {pdf_path.name}")
    return pdf_path.name


def upload_pdfs(bucket_name, pdf_folder, prefix="pdfs/", manifest_path=None,
                mineral=None, workers=50, skip_existing=False):
    """Upload PDFs from a local folder to S3, optionally filtered by mineral."""
    pdf_folder = Path(pdf_folder)

    if not pdf_folder.is_dir():
        logger.error(f"Not a directory: {pdf_folder}")
        sys.exit(1)

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_folder}")
        return

    if skip_existing:
        logger.info(f"Listing existing keys under s3://{bucket_name}/{prefix} ...")
        existing = list_existing_keys(bucket_name, prefix)
        before = len(pdf_files)
        pdf_files = [p for p in pdf_files if p.name not in existing]
        logger.info(
            f"Skipping {before - len(pdf_files)} PDFs already in S3; "
            f"uploading {len(pdf_files)} new"
        )

    # Apply mineral gate if manifest provided
    if manifest_path and mineral:
        manifest = load_manifest(manifest_path)
        accepted, skipped, missing = filter_by_mineral(pdf_files, manifest, mineral)

        logger.info(f"Mineral gate: {mineral}")
        logger.info(f"  {len(accepted)} PDFs pass (mineral relevant)")
        logger.info(f"  {len(skipped)} PDFs skipped (mineral not relevant)")
        if missing:
            logger.warning(f"  {len(missing)} PDFs not in manifest (skipped)")

        pdf_files = accepted

    if not pdf_files:
        logger.info("No PDFs to upload after filtering.")
        return

    total = len(pdf_files)
    actual_workers = min(workers, total)
    logger.info(f"Uploading {total} PDFs to s3://{bucket_name}/{prefix} "
                f"({actual_workers} concurrent threads)")

    # Each thread gets its own S3 client (boto3 clients aren't thread-safe)
    failed = []
    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = {}
        for i, pdf_path in enumerate(pdf_files, 1):
            s3 = boto3.client("s3")
            f = pool.submit(_upload_one, s3, bucket_name, pdf_path, prefix, i, total)
            futures[f] = pdf_path

        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                name = futures[f].name
                logger.error(f"  FAILED: {name} — {e}")
                failed.append(name)

    logger.info(f"Done. {total - len(failed)}/{total} PDFs uploaded.")
    if failed:
        logger.warning(f"Failed uploads: {failed}")
    logger.info(f"Lambda will process each PDF independently.")
    logger.info(f"Monitor: aws logs tail /aws/lambda/<stack>-ExtractFunction-<id> --follow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload PDFs to S3 for Lambda processing")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("folder", help="Local folder containing PDFs")
    parser.add_argument("--prefix", default="pdfs/", help="S3 key prefix (default: pdfs/)")
    parser.add_argument(
        "--manifest", default=None,
        help="Path to pdf_manifest.json from classify_pdfs.py"
    )
    parser.add_argument(
        "--mineral", default=None,
        help="Target mineral to filter by (e.g. copper). Requires --manifest."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="List keys under the S3 prefix and skip PDFs whose basename already exists."
    )
    args = parser.parse_args()

    if args.mineral and not args.manifest:
        logger.error("--mineral requires --manifest")
        sys.exit(1)

    upload_pdfs(args.bucket, args.folder, args.prefix, args.manifest, args.mineral,
                skip_existing=args.skip_existing)
