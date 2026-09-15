"""
AWS Lambda handler for per-PDF extraction.

Triggered by S3 PutObject events on pdfs/*.pdf.
Downloads the PDF, runs image + flowsheet + table + text extraction,
uploads results JSON and flowsheet images back to S3.

Steps are toggleable via env vars (SKIP_IMAGES, SKIP_TABLES, SKIP_TEXT, SKIP_LLM).
Re-triggering a PDF only runs steps not already completed (resumable).
"""

import glob
import json
import logging
import os
import hashlib
import shutil
import time
from pathlib import Path
from urllib.parse import unquote_plus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _env_flag(name, default="false"):
    """Check if an environment variable is truthy."""
    return os.environ.get(name, default).lower() in ("true", "1", "yes")


_MINERAL_KEYWORDS = (
    "gold", "silver", "copper", "zinc", "lead", "nickel",
    "lithium", "cobalt", "uranium", "platinum", "iron",
)


def _detect_mineral_from_filename(s3_key):
    """Return the first mineral keyword matched as a whole word in the filename, else ''.

    Boundaries are any non-letter (underscore, hyphen, digit, start/end) — Python's
    \\b treats underscore as a word char, so '_gold_' wouldn't match \\bgold\\b.
    """
    import re
    name = Path(s3_key).stem.lower()
    for mineral in _MINERAL_KEYWORDS:
        if re.search(rf"(?<![a-z]){mineral}(?![a-z])", name):
            return mineral
    return ""


def _inspect_and_maybe_repair(local_pdf):
    """Open with pymupdf to get page count; only rewrite the file if mupdf
    had to repair the structure. Well-formed PDFs pass through untouched so
    pdfplumber sees the original bytes downstream.

    Returns (page_count, was_repaired).
    """
    import pymupdf  # type: ignore
    doc = pymupdf.open(local_pdf)
    try:
        page_count = doc.page_count
        if doc.is_repaired:
            repaired = local_pdf + ".repaired"
            doc.save(repaired, garbage=4, deflate=True, clean=True)
            needs_replace = True
        else:
            needs_replace = False
    finally:
        doc.close()
    if needs_replace:
        os.replace(repaired, local_pdf)
    return page_count, needs_replace


def compute_document_id(file_path):
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def handler(event, context):
    """
    Lambda entry point.

    Expects an S3 event with records containing bucket and key.
    Processes one PDF per invocation.
    """
    from storage import S3Storage
    from pdf_extraction import (
        extract_all_figures, extract_tables, extract_copper_grades,
        extract_page_text,
    )
    from flowsheet_extraction import extract_with_terms

    # Parse S3 event
    record = event["Records"][0]["s3"]
    bucket = record["bucket"]["name"]
    s3_key = unquote_plus(record["object"]["key"])
    pdf_stem = Path(s3_key).stem

    logger.info(f"Processing: s3://{bucket}/{s3_key}")

    storage = S3Storage(bucket)
    max_retries = int(os.environ.get("MAX_RETRIES", "3"))

    # Check for existing result (resumable processing + retry limit)
    result_s3_key = f"results/{pdf_stem}.json"
    existing = None
    if storage.exists(result_s3_key):
        try:
            existing = storage.read_json(result_s3_key)
            logger.info(f"Found existing result — steps_completed: "
                        f"{existing.get('steps_completed', [])}")
        except Exception:
            existing = None

    retry_count = existing.get("retry_count", 0) if existing else 0
    if retry_count >= max_retries:
        logger.warning(f"Max retries ({max_retries}) reached for {pdf_stem} — skipping")
        return {"status": "max_retries_exceeded", "pdf_stem": pdf_stem}

    # Step toggles
    skip_images = _env_flag("SKIP_IMAGES")
    skip_tables = _env_flag("SKIP_TABLES")
    skip_text = _env_flag("SKIP_TEXT")
    skip_llm = _env_flag("SKIP_LLM")
    target_mineral = _detect_mineral_from_filename(s3_key) or os.environ.get("TARGET_MINERAL", "")

    # Download PDF to /tmp
    try:
        t0 = time.time()
        local_pdf = storage.download_to_tmp(s3_key)
        doc_hash = compute_document_id(local_pdf)
        logger.info(f"[TIMING] S3 download + hash: {time.time() - t0:.1f}s")
    except Exception as e:
        logger.exception(f"Failed to download {s3_key}")
        _upload_result(storage, pdf_stem, {
            "filename": os.path.basename(s3_key),
            "retry_count": retry_count + 1,
            "status": "failed",
            "errors": [{"step": "download", "error": str(e)}],
            "steps_completed": [],
            "steps_skipped": [],
            "flowsheets": {},
            "tables": [],
            "page_text": [],
        })
        return {"status": "failed", "error": str(e)}

    # Page-count gate. pymupdf also flags structural damage — if mupdf had to
    # repair the PDF to open it, rewrite a clean copy so downstream pdfplumber
    # calls (which depth-first-search the page tree) don't hang.
    try:
        page_count, was_repaired = _inspect_and_maybe_repair(local_pdf)
        if was_repaired:
            # Recompute doc_hash so the post-repair file can be re-identified
            doc_hash = compute_document_id(local_pdf)
            logger.info(f"pymupdf rewrote malformed PDF for {pdf_stem}")
    except Exception as open_err:
        logger.exception(f"pymupdf could not open {pdf_stem}")
        _upload_result(storage, pdf_stem, {
            "document_id": doc_hash,
            "filename": os.path.basename(s3_key),
            "target_mineral": target_mineral,
            "retry_count": retry_count + 1,
            "status": "failed",
            "errors": [{"step": "pdf_open", "error": str(open_err)}],
            "steps_completed": [],
            "steps_skipped": [],
            "flowsheets": {},
            "tables": [],
            "page_text": [],
        })
        return {"status": "failed", "error": str(open_err)}
    if page_count > 1000:
        logger.info(f"Skipping {pdf_stem}: {page_count} pages exceeds 1000-page limit")
        return {"status": "skipped", "reason": "exceeds_page_limit", "pages": page_count}

    steps_completed = list(existing.get("steps_completed", [])) if existing else []

    result = {
        "document_id": doc_hash,
        "filename": os.path.basename(s3_key),
        "target_mineral": target_mineral or existing.get("target_mineral", ""),
        "flowsheets": existing.get("flowsheets", {}) if existing else {},
        "tables": existing.get("tables", []) if existing else [],
        "page_text": existing.get("page_text", []) if existing else [],
        "cu_grade_avg": existing.get("cu_grade_avg") if existing else None,
        "retry_count": retry_count + 1,
        "steps_completed": steps_completed,
        "steps_skipped": [],
        "status": "completed",
        "errors": existing.get("errors", []) if existing else [],
    }

    def _should_run(step_name):
        """Check if a step should run (not skipped AND not already completed)."""
        return step_name not in steps_completed

    # --- Step 1: Extract figures (no CLIP) ---
    if skip_images:
        if "images" not in steps_completed:
            result["steps_skipped"].append("images")
        logger.info("SKIP_IMAGES: skipping figure extraction")
    elif _should_run("images"):
        remaining_ms = context.get_remaining_time_in_millis()
        if remaining_ms < 60_000:
            result["status"] = "timeout_before_extraction"
            _upload_result(storage, pdf_stem, result)
            return result

        try:
            t1 = time.time()
            extracted = extract_all_figures(
                pdf_location=local_pdf,
                output_folder="/tmp/images",
                vectorize=False,
                image_types=("flowsheet",),
            )
            flowsheet_items = [x for x in extracted if x["type"] == "flowsheet"]
            logger.info(f"[TIMING] Figure extraction: {time.time() - t1:.1f}s")
            logger.info(f"Extracted {len(flowsheet_items)} flowsheet(s), "
                         f"{len(extracted) - len(flowsheet_items)} other image(s)")

            # Upload flowsheet images to S3
            t2 = time.time()
            for item in flowsheet_items:
                image_path = item.get("image_path")
                if image_path and os.path.exists(image_path):
                    image_name = os.path.basename(image_path)
                    s3_image_key = f"images/{doc_hash}/{image_name}"
                    try:
                        with open(image_path, "rb") as f:
                            storage.save_image(f.read(), s3_image_key)
                        item["s3_image_key"] = s3_image_key
                    except Exception as e:
                        logger.exception(f"Failed to upload image {image_name}")
            logger.info(f"[TIMING] Image upload to S3: {time.time() - t2:.1f}s")

            # LLM extraction on flowsheets
            t3 = time.time()
            if skip_llm:
                for item in flowsheet_items:
                    image_path = item.get("image_path")
                    if not image_path:
                        continue
                    key = os.path.splitext(os.path.basename(image_path))[0]
                    result["flowsheets"][key] = {
                        "type": "flowsheet",
                        "text_before": item.get("text_before", ""),
                        "text_after": item.get("text_after", ""),
                        "caption": item.get("caption", ""),
                        "page": item.get("page"),
                        "s3_image_key": item.get("s3_image_key"),
                    }
                logger.info(f"SKIP_LLM: saved {len(result['flowsheets'])} flowsheet metadata entries")
            else:
                for item in flowsheet_items:
                    remaining_ms = context.get_remaining_time_in_millis()
                    if remaining_ms < 120_000:
                        logger.warning("Approaching timeout — skipping remaining flowsheets")
                        result["errors"].append({
                            "step": "llm_extraction",
                            "error": "timeout_safety_skip",
                        })
                        result["status"] = "completed_with_errors"
                        break

                    image_path = item.get("image_path")
                    if not image_path:
                        continue

                    key = os.path.splitext(os.path.basename(image_path))[0]
                    try:
                        llm_result = extract_with_terms(str(image_path), max_iterations=3)
                        if "error" in llm_result:
                            result["flowsheets"][key] = {
                                "type": "flowsheet",
                                "error": llm_result["error"],
                                "text_before": item.get("text_before", ""),
                                "text_after": item.get("text_after", ""),
                                "caption": item.get("caption", ""),
                                "page": item.get("page"),
                            }
                        else:
                            result["flowsheets"][key] = {
                                "type": "flowsheet",
                                **llm_result.get("data", {}),
                                "text_before": item.get("text_before", ""),
                                "text_after": item.get("text_after", ""),
                                "caption": item.get("caption", ""),
                                "page": item.get("page"),
                                "llm_tokens": llm_result.get("total_tokens"),
                                "llm_cost": llm_result.get("total_cost"),
                            }
                    except Exception as e:
                        logger.exception(f"LLM extraction failed for {key}")
                        result["errors"].append({
                            "step": "llm_extraction",
                            "image_id": key,
                            "error": str(e),
                        })
            logger.info(f"[TIMING] LLM extraction: {time.time() - t3:.1f}s")

            steps_completed.append("images")
        except Exception as e:
            logger.exception("Figure extraction failed")
            result["errors"].append({"step": "figure_extraction", "error": str(e)})
            result["status"] = "completed_with_errors"
    else:
        logger.info("Images already completed — skipping")

    # --- Step 2: Table extraction ---
    if skip_tables:
        if "tables" not in steps_completed:
            result["steps_skipped"].append("tables")
        logger.info("SKIP_TABLES: skipping table extraction")
    elif _should_run("tables"):
        t4 = time.time()
        remaining_ms = context.get_remaining_time_in_millis()
        if remaining_ms > 60_000:
            try:
                tables = extract_tables(pdf_location=local_pdf)
                result["tables"] = [
                    {
                        "table_id": t["table_id"],
                        "page": t["page"],
                        "section_number": t.get("section_number"),
                        "section_title": t.get("section_title"),
                        "caption": t.get("caption", ""),
                        "headers": t.get("headers", []),
                        "rows": t.get("rows", []),
                        "text_before": t.get("text_before", ""),
                        "text_after": t.get("text_after", ""),
                    }
                    for t in tables
                ]

                grades = extract_copper_grades(tables)
                if grades:
                    values = [g["value"] for g in grades]
                    result["cu_grade_avg"] = sum(values) / len(values)

                logger.info(f"Extracted {len(tables)} table(s)")
                steps_completed.append("tables")
            except Exception as e:
                logger.exception("Table extraction failed")
                result["errors"].append({"step": "table_extraction", "error": str(e)})
        else:
            logger.warning("Approaching timeout — skipping table extraction")
        logger.info(f"[TIMING] Table extraction: {time.time() - t4:.1f}s")
    else:
        logger.info("Tables already completed — skipping")

    # --- Step 3: Page text extraction ---
    if skip_text:
        if "page_text" not in steps_completed:
            result["steps_skipped"].append("page_text")
        logger.info("SKIP_TEXT: skipping page text extraction")
    elif _should_run("page_text"):
        t5 = time.time()
        remaining_ms = context.get_remaining_time_in_millis()
        if remaining_ms > 30_000:
            try:
                page_text = extract_page_text(pdf_location=local_pdf)
                result["page_text"] = page_text
                logger.info(f"Extracted text from {len(page_text)} pages")
                steps_completed.append("page_text")
            except Exception as e:
                logger.exception("Page text extraction failed")
                result["errors"].append({"step": "page_text_extraction", "error": str(e)})
        else:
            logger.warning("Approaching timeout — skipping page text extraction")
        logger.info(f"[TIMING] Page text extraction: {time.time() - t5:.1f}s")
    else:
        logger.info("Page text already completed — skipping")

    # Finalize
    result["steps_completed"] = steps_completed
    if result["errors"]:
        result["status"] = "completed_with_errors"

    _upload_result(storage, pdf_stem, result)

    logger.info(f"Done: {pdf_stem} — {len(result['flowsheets'])} flowsheets, "
                 f"{len(result['tables'])} tables, "
                 f"{len(result.get('page_text', []))} pages, "
                 f"status={result['status']}")

    # Clean up /tmp to avoid filling Lambda's ephemeral storage across warm starts
    _cleanup_tmp(local_pdf)

    return result


def _cleanup_tmp(local_pdf):
    """Remove downloaded PDF and extracted images from /tmp."""
    try:
        if local_pdf and os.path.exists(local_pdf):
            os.remove(local_pdf)
        images_dir = "/tmp/images"
        if os.path.isdir(images_dir):
            shutil.rmtree(images_dir)
    except Exception:
        logger.debug("tmp cleanup failed", exc_info=True)


def _upload_result(storage, pdf_stem, result):
    """Upload the result JSON to S3."""
    s3_key = f"results/{pdf_stem}.json"
    storage.write_json(result, s3_key)
    logger.info(f"Results uploaded to {s3_key}")
