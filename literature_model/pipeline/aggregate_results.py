"""
Aggregate per-PDF results into Parquet files for ML training.

Reads results JSONs from S3 or a local directory, combines them into
structured Parquet files. Supports writing Parquet directly to S3.

Usage:
    python aggregate_results.py s3://bucket-name
    python aggregate_results.py ./local_results/
    python aggregate_results.py s3://bucket-name --output s3://bucket-name/training
"""

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_results_from_local(results_dir):
    """Load all result JSONs from a local directory."""
    results_dir = Path(results_dir)
    results = {}
    for json_path in sorted(results_dir.glob("*.json")):
        try:
            with open(json_path) as f:
                results[json_path.stem] = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Skipping {json_path.name}: {e}")
    return results


def load_results_from_s3(bucket_name, prefix="results/"):
    """Load all result JSONs from S3."""
    import boto3
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    results = {}
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            stem = Path(key).stem
            try:
                response = s3.get_object(Bucket=bucket_name, Key=key)
                data = json.loads(response["Body"].read().decode("utf-8"))
                results[stem] = data
            except Exception as e:
                logger.warning(f"Skipping {key}: {e}")
    return results


def build_flowsheets_df(all_results):
    """Build a DataFrame with one row per flowsheet image."""
    rows = []
    for pdf_stem, result in all_results.items():
        doc_id = result.get("document_id", pdf_stem)
        cu_grade = result.get("cu_grade_avg")
        flowsheets = result.get("flowsheets", {})

        for image_key, fs in flowsheets.items():
            rows.append({
                "document_id": doc_id,
                "pdf_stem": pdf_stem,
                "image_key": image_key,
                "page": fs.get("page"),
                "caption": fs.get("caption", ""),
                "text_before": fs.get("text_before", ""),
                "text_after": fs.get("text_after", ""),
                "stages_json": json.dumps(fs.get("stages", [])),
                "units_json": json.dumps(fs.get("units", [])),
                "connections_json": json.dumps(fs.get("connections", [])),
                "cu_grade_avg": cu_grade,
                "error": fs.get("error"),
                "llm_tokens": fs.get("llm_tokens"),
                "llm_cost": fs.get("llm_cost"),
            })

    return pd.DataFrame(rows)


def build_tables_df(all_results):
    """Build a DataFrame with one row per extracted table."""
    rows = []
    for pdf_stem, result in all_results.items():
        doc_id = result.get("document_id", pdf_stem)
        for i, table in enumerate(result.get("tables", [])):
            rows.append({
                "document_id": doc_id,
                "pdf_stem": pdf_stem,
                "table_id": table.get("table_id", f"{pdf_stem}_table_{i}"),
                "page": table.get("page"),
                "section_number": table.get("section_number"),
                "section_title": table.get("section_title", ""),
                "caption": table.get("caption", ""),
                "headers": json.dumps(table.get("headers", [])),
                "rows": json.dumps(table.get("rows", [])),
                "text_before": table.get("text_before", ""),
                "text_after": table.get("text_after", ""),
            })
    return pd.DataFrame(rows)


def build_page_text_df(all_results):
    """Build a DataFrame with one row per page of extracted text."""
    rows = []
    for pdf_stem, result in all_results.items():
        doc_id = result.get("document_id", pdf_stem)
        for pt in result.get("page_text", []):
            rows.append({
                "document_id": doc_id,
                "pdf_stem": pdf_stem,
                "page_number": pt.get("page_number"),
                "section_number": pt.get("section_number"),
                "section_title": pt.get("section_title"),
                "text": pt.get("text", ""),
                "char_count": pt.get("char_count", 0),
            })
    return pd.DataFrame(rows)


def build_documents_df(all_results):
    """Build a DataFrame with one row per PDF document."""
    rows = []
    for pdf_stem, result in all_results.items():
        doc_id = result.get("document_id", pdf_stem)
        flowsheets = result.get("flowsheets", {})
        tables = result.get("tables", [])
        page_text = result.get("page_text", [])
        errors = result.get("errors", [])

        rows.append({
            "document_id": doc_id,
            "filename": result.get("filename", f"{pdf_stem}.pdf"),
            "target_mineral": result.get("target_mineral", ""),
            "cu_grade_avg": result.get("cu_grade_avg"),
            "flowsheet_count": len(flowsheets),
            "table_count": len(tables),
            "page_count": len(page_text),
            "error_count": len(errors),
            "status": result.get("status", "unknown"),
            "steps_completed": json.dumps(result.get("steps_completed", [])),
            "steps_skipped": json.dumps(result.get("steps_skipped", [])),
        })

    return pd.DataFrame(rows)


def run_clip_embeddings(all_results, images_dir=None):
    """
    Generate CLIP embeddings for non-flowsheet images.

    This runs locally during aggregation, NOT on Lambda.
    Returns a DataFrame with one row per image.
    """
    rows = []

    try:
        from sentence_transformers import SentenceTransformer
        from PIL import Image

        model = SentenceTransformer("clip-ViT-B-32")
        logger.info("CLIP model loaded for embedding generation")
    except ImportError:
        logger.warning("sentence-transformers not installed — skipping CLIP embeddings")
        return pd.DataFrame(rows)

    if images_dir is None:
        logger.info("No images directory provided — skipping CLIP embeddings")
        return pd.DataFrame(rows)

    images_dir = Path(images_dir)
    if not images_dir.exists():
        logger.info(f"Images directory not found: {images_dir}")
        return pd.DataFrame(rows)

    # Walk through document hash directories
    for doc_dir in sorted(images_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        for img_path in sorted(doc_dir.glob("*.png")):
            try:
                img = Image.open(img_path)
                embedding = model.encode(img).tolist()
                rows.append({
                    "document_id": doc_id,
                    "image_key": img_path.stem,
                    "embedding": embedding,
                    "file_path": str(img_path),
                })
            except Exception as e:
                logger.warning(f"Failed to embed {img_path}: {e}")

    logger.info(f"Generated {len(rows)} CLIP embeddings")
    return pd.DataFrame(rows)


def run_normalization(all_results):
    """Run normalization on combined flowsheet data."""
    from normalize_ids import build_standard_id_library
    import tempfile

    # Combine all flowsheet data into a single results-like dict
    combined = {}
    for pdf_stem, result in all_results.items():
        for key, fs in result.get("flowsheets", {}).items():
            if "error" not in fs:
                combined[key] = fs

    if not combined:
        logger.info("No flowsheet data to normalize")
        return

    # Write to temp file and run normalization
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(combined, f)
        temp_path = f.name

    try:
        mapping, groups = build_standard_id_library(temp_path)
        logger.info(f"Normalization: {len(mapping)} unique IDs -> {len(groups)} groups")
    finally:
        os.unlink(temp_path)


def run_stage_network(all_results, output_path=None):
    """Build and visualize stage network from combined data."""
    from stage_network import build_network, print_stats, visualize_network

    combined = {}
    for pdf_stem, result in all_results.items():
        for key, fs in result.get("flowsheets", {}).items():
            if "error" not in fs:
                combined[key] = fs

    if not combined:
        logger.info("No flowsheet data for stage network")
        return

    G = build_network(combined)
    print_stats(G)
    if output_path:
        visualize_network(G)


def write_parquet(df, output_path, name):
    """Write a DataFrame to Parquet, locally or to S3."""
    if df.empty:
        logger.info(f"  {name}: empty — skipping")
        return

    if output_path.startswith("s3://"):
        # Write to S3 via BytesIO buffer
        from storage import S3Storage
        parts = output_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        s3_key = f"{prefix}/{name}.parquet".lstrip("/")

        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)
        buf.seek(0)

        storage = S3Storage(bucket)
        storage.write_bytes(buf.read(), s3_key, content_type="application/octet-stream")
        logger.info(f"  {name}: {len(df)} rows -> s3://{bucket}/{s3_key}")
    else:
        path = Path(output_path) / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(path), engine="pyarrow", index=False)
        logger.info(f"  {name}: {len(df)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-PDF results into Parquet for ML training"
    )
    parser.add_argument(
        "source",
        help="S3 URI (s3://bucket) or local directory with result JSONs"
    )
    parser.add_argument(
        "--output", default="./training",
        help="Output path for Parquet files (local dir or s3://bucket/prefix)"
    )
    parser.add_argument(
        "--images-dir", default=None,
        help="Local directory with extracted images for CLIP embedding"
    )
    parser.add_argument(
        "--skip-clip", action="store_true",
        help="Skip CLIP embedding generation"
    )
    parser.add_argument(
        "--skip-network", action="store_true",
        help="Skip stage network building"
    )
    args = parser.parse_args()

    # Load results
    logger.info(f"Loading results from {args.source}")
    if args.source.startswith("s3://"):
        bucket = args.source.replace("s3://", "").rstrip("/")
        all_results = load_results_from_s3(bucket)
    else:
        all_results = load_results_from_local(args.source)

    logger.info(f"Loaded {len(all_results)} result files")

    if not all_results:
        logger.warning("No results found — nothing to aggregate")
        return

    # Build DataFrames
    logger.info("Building DataFrames...")
    flowsheets_df = build_flowsheets_df(all_results)
    tables_df = build_tables_df(all_results)
    page_text_df = build_page_text_df(all_results)
    documents_df = build_documents_df(all_results)

    logger.info(f"  Flowsheets: {len(flowsheets_df)} rows")
    logger.info(f"  Tables: {len(tables_df)} rows")
    logger.info(f"  Page text: {len(page_text_df)} rows")
    logger.info(f"  Documents: {len(documents_df)} rows")

    # Write Parquet files
    logger.info(f"Writing Parquet files to {args.output}")
    write_parquet(flowsheets_df, args.output, "flowsheets")
    write_parquet(tables_df, args.output, "tables")
    write_parquet(page_text_df, args.output, "page_text")
    write_parquet(documents_df, args.output, "documents")

    # Skip heavy-dep steps when running on Lambda
    on_lambda = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

    # CLIP embeddings (local only — needs sentence-transformers)
    if not args.skip_clip and not on_lambda:
        logger.info("Generating CLIP embeddings...")
        embeddings_df = run_clip_embeddings(all_results, args.images_dir)
        write_parquet(embeddings_df, args.output, "embeddings")

    # Normalization (local only)
    if not on_lambda:
        logger.info("Running normalization...")
        run_normalization(all_results)

    # Stage network (local only — needs matplotlib/networkx)
    if not args.skip_network and not on_lambda:
        logger.info("Building stage network...")
        run_stage_network(all_results, args.output)

    logger.info("Aggregation complete.")


if __name__ == "__main__":
    main()
