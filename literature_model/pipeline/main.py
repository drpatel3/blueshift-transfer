import json
import logging
import os
from pathlib import Path
from collections import defaultdict

import config
import database as db
from table_parser import get_cu_grade_average

logger = logging.getLogger(__name__)

# 1. Process PDFs - extract all figures (flowsheets + other)
def run_process_pdfs(file_path):
    """Single pass: extract all images, tag flowsheets vs other."""
    from pdf_extraction import extract_all_figures

    logger.info(f"\n--- Processing: {file_path} ---")
    extracted_data = extract_all_figures(pdf_location=file_path)

    flowsheets = [x for x in extracted_data if x['type'] == 'flowsheet']
    other = [x for x in extracted_data if x['type'] == 'other']
    logger.info(f"\nSUMMARY: Found {len(flowsheets)} flowsheet(s), {len(other)} other image(s)")

    return extracted_data

# 1b. Extract tables from PDFs
def run_extract_tables(file_path, conn=None, run_id=None, document_id=None,
                       target_section=None):
    """Extract tables from a PDF and save to database."""
    from pdf_extraction import extract_tables, extract_copper_grades

    logger.info(f"\n--- Extracting tables: {file_path} ---")
    tables = extract_tables(pdf_location=file_path, target_section=target_section)

    # Save to database
    if conn and run_id and document_id:
        for t in tables:
            record = db.TableRecord(
                table_id=t["table_id"],
                document_id=document_id,
                page_number=t["page"],
                table_index=t.get("table_index", 0),
                section_number=t.get("section_number"),
                section_title=t.get("section_title"),
                caption=t.get("caption"),
                headers=t.get("headers", []),
                rows=t.get("rows", []),
                text_before=t.get("text_before"),
                text_after=t.get("text_after"),
            )
            db.save_table_record(conn, record, run_id)

    # Extract copper grades from the tables
    grades = extract_copper_grades(tables)
    if grades:
        logger.info(f"  Copper grades found: {[g['value'] for g in grades]}")

    return tables, grades


# 2. Test refinement - extract data from flowsheet images using LLM
def run_test_refinement(extracted_data=None, conn=None, run_id=None,
                        document_id=None, results_path=None):
    """Run LLM extraction on flowsheets only. Store text context for all."""
    if config.EXTRACTION_METHOD == "refinement":
        from flowsheet_extraction import extract_with_refinement as extract_fn
        logger.info(f"Using extraction method: refinement "
                    f"({config.BEDROCK_MODEL} + {config.BEDROCK_REFINEMENT_MODEL})")
    else:
        from flowsheet_extraction import extract_with_terms as extract_fn
        logger.info(f"Using extraction method: terms ({config.LLM_MODEL})")

    if results_path is None:
        results_path = str(config.RESULTS_PATH)

    # Load existing results to merge with
    results = {}
    if Path(results_path).exists():
        try:
            with open(results_path) as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} existing results from {results_path}")
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Warning: {results_path} is empty or invalid, starting fresh")
            results = {}

    counter = 0
    error_count = 0
    overall_cost = 0.0
    overall_tokens = 0
    overall_costs_by_model = defaultdict(float)

    if extracted_data is None:
        # Fallback: scan folder (legacy behavior)
        folder = Path("extracted_images")
        extracted_data = [{"image_path": str(p), "type": "flowsheet"} for p in folder.glob("*.png")]

    # Only run LLM extraction on flowsheets
    flowsheets = [x for x in extracted_data if x['type'] == 'flowsheet']

    for item in flowsheets:
        image_path = item["image_path"]
        key = os.path.splitext(os.path.basename(image_path))[0]

        try:
            logger.info(f"Processing flowsheet: {image_path}")
            logger.info(f"{'='*60}\n")
            result = extract_fn(str(image_path), max_iterations=3)

            if "error" in result:
                logger.error(f"Error: {result['error']}")
                results[key] = {
                    "type": "flowsheet",
                    "error": result['error'],
                    "raw": result.get('raw', ''),
                    "text_before": item.get("text_before", ""),
                    "text_after": item.get("text_after", ""),
                    "caption": item.get("caption", ""),
                    "page": item.get("page")
                }
                # Parallel DB write — error flowsheet
                if conn and run_id and document_id:
                    _save_image_to_db(conn, key, document_id, run_id, item)
                    db.save_flowsheet(conn, key, run_id,
                                      error=result['error'],
                                      raw_response=result.get('raw', ''))
            else:
                logger.info(f"Image total tokens: {result['total_tokens']}")
                logger.info(f"Image total cost: ${result['total_cost']:.4f}")
                results[key] = {
                    "type": "flowsheet",
                    **result.get('data', {}),
                    "text_before": item.get("text_before", ""),
                    "text_after": item.get("text_after", ""),
                    "caption": item.get("caption", ""),
                    "page": item.get("page")
                }
                # Parallel DB write — successful flowsheet
                if conn and run_id and document_id:
                    _save_image_to_db(conn, key, document_id, run_id, item)
                    _save_flowsheet_to_db(conn, key, run_id, result)

            overall_cost += result.get('total_cost', 0)
            overall_tokens += result.get('total_tokens', 0)
            for model, cost in result.get('costs_by_model', {}).items():
                overall_costs_by_model[model] += cost
            counter += 1

        except Exception as e:
            error_count += 1
            logger.error(f"ERROR processing {key}: {e} — skipping to next flowsheet")
            if conn and run_id:
                db.log_step_error(conn, run_id, "llm_extraction", e,
                                  image_id=key, page_number=item.get("page"))

        logger.info(f"\n--- Running totals: {counter} flowsheets, {overall_tokens} tokens, ${overall_cost:.4f} ---")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL SUMMARY: {counter} flowsheets processed (LLM)")
    if error_count:
        logger.warning(f"  Errors: {error_count} flowsheets failed (logged, skipped)")
    logger.info(f"Total tokens: {overall_tokens}")
    logger.info(f"Total cost: ${overall_cost:.4f}")
    logger.info(f"Cost by model:")
    for model, cost in overall_costs_by_model.items():
        logger.info(f"  {model}: ${cost:.4f}")
    logger.info(f"Results saved to {results_path}")
    logger.info(f"{'='*60}")

# 3. Process output - normalize IDs
def run_process_output(conn=None, run_id=None, results_path=None):
    from normalize_ids import build_standard_id_library

    if results_path is None:
        results_path = str(config.RESULTS_PATH)
    mapping, groups = build_standard_id_library(results_path)

    # Parallel DB write
    if conn and run_id:
        db.save_normalization_entries(conn, run_id, groups)

    logger.info(f"Total unique IDs: {len(mapping)}")
    logger.info(f"Standard IDs (after semantic matching): {len(groups)}")

    logger.info("\nSemantic groups (IDs that match):")
    for norm, originals in sorted(groups.items()):
        if len(originals) > 1:
            logger.info(f"  {norm}: {originals}")

    logger.info("\nStandalone IDs (no matches):")
    for norm, originals in sorted(groups.items()):
        if len(originals) == 1:
            logger.info(f"  {norm}")

# 4. Build stage network
def run_build_stage_network(results_path=None):
    from stage_network import load_data, build_network, print_stats, visualize_network

    if results_path is None:
        results_path = str(config.RESULTS_PATH)
    data = load_data(results_path)
    G = build_network(data)
    print_stats(G)
    visualize_network(G)


# ---------------------------------------------------------------------------
# DB write helpers (parallel write alongside JSON during transition)
# ---------------------------------------------------------------------------

def _save_image_to_db(conn, image_id, document_id, run_id, item):
    """Save an extracted image record to the database."""
    image = db.ImageRecord(
        image_id=image_id,
        document_id=document_id,
        image_type=item.get("type", "flowsheet"),
        page_number=item.get("page"),
        file_path=item.get("image_path"),
        caption=item.get("caption"),
        text_before=item.get("text_before"),
        text_after=item.get("text_after"),
    )
    db.save_image_record(conn, image, run_id)


def _save_flowsheet_to_db(conn, image_id, run_id, result):
    """Save a successful LLM extraction result to the database."""
    raw_data = result.get('data', {})
    flowsheet_data = db.FlowsheetData(
        stages=[db.Stage(**s) for s in raw_data.get('stages', [])],
        units=[db.Unit(**u) for u in raw_data.get('units', [])],
        connections=[db.Connection(**c) for c in raw_data.get('connections', [])],
    )
    db.save_flowsheet(
        conn, image_id, run_id, data=flowsheet_data,
        llm_tokens=result.get('total_tokens'),
        llm_cost=result.get('total_cost'),
    )


def get_processed_pdfs(results_path="results.json"):
    """Check which PDFs have already been processed based on results.json."""
    if not Path(results_path).exists():
        return set()

    try:
        with open(results_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return set()

    # Extract PDF names from keys (e.g., "test_casino_pfs_213" -> "test_casino")
    processed = set()
    for key in data.keys():
        # Keys are like: {pdf_stem}_pfs_{page} or {pdf_stem}_{page}
        # Find the PDF stem by matching against known patterns
        parts = key.rsplit('_', 2)  # Split from right to remove _pfs_123 or _123
        if len(parts) >= 2:
            # Try to find the PDF stem (everything before _pfs_ or before last _number)
            if 'pfs' in parts:
                idx = key.find('_pfs_')
                if idx > 0:
                    processed.add(key[:idx])
            else:
                # For "other" images: test_casino_123 -> test_casino
                stem = '_'.join(parts[:-1])
                processed.add(stem)
    return processed


def run_pipeline(pdf_dir=None, pdf_files=None, max_pdfs=None,
                 results_path=None, skip_clip=False, skip_tables=False,
                 skip_network=False, skip_cu_filter=False):
    """
    Run the full extraction pipeline.

    Args:
        pdf_dir: Directory containing PDFs to process.
        pdf_files: Explicit list of PDF paths (overrides pdf_dir).
        max_pdfs: Maximum number of PDFs to process.
        results_path: Path for results JSON output.
        skip_clip: Disable CLIP vectorization.
        skip_tables: Skip table extraction step.
        skip_network: Skip stage network building step.
        skip_cu_filter: Skip Cu% pre-filter (process all PDFs regardless of copper content).

    Returns:
        dict with pipeline results and status.
    """
    if pdf_dir is None:
        pdf_dir = config.PDF_DIR
    if results_path is None:
        results_path = str(config.RESULTS_PATH)
    if max_pdfs is None:
        max_pdfs = config.MAX_PDFS

    if skip_clip:
        config.ENABLE_CLIP = False

    # Initialize database
    db.init_db()
    conn = db.get_connection()
    run_id = db.start_run(conn)
    pdf_errors = 0
    doc_id = None

    # Determine PDFs to process
    if pdf_files:
        all_pdf_files = [Path(p) for p in pdf_files]
    else:
        pdf_folder = Path(pdf_dir)
        all_pdf_files = sorted(pdf_folder.glob("*.pdf"))

    # Check which PDFs are already processed
    already_processed = get_processed_pdfs(results_path)
    pdfs_to_process = [p for p in all_pdf_files if p.stem not in already_processed]
    skipped = [p for p in all_pdf_files if p.stem in already_processed]

    # Filter: only keep PDFs that contain Cu % data (unless skipped)
    no_cu_pdfs = []
    if skip_cu_filter:
        logger.info("=" * 60)
        logger.info("PRE-FILTER: Cu % filter SKIPPED (--skip-cu-filter)")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("PRE-FILTER: Checking PDFs for Cu % data")
        logger.info("=" * 60)
        cu_pdfs = []
        for p in pdfs_to_process:
            avg = get_cu_grade_average(str(p))
            if avg is not None:
                cu_pdfs.append(p)
                logger.info(f"  {p.name}: Cu avg = {avg:.4f}%")
            else:
                no_cu_pdfs.append(p)
                logger.info(f"  {p.name}: no Cu % found — skipping")
        pdfs_to_process = cu_pdfs

    logger.info("=" * 60)
    logger.info(f"STEP 1: Extracting figures from PDFs")
    logger.info(f"  Total PDFs found: {len(all_pdf_files)}")
    logger.info(f"  Already processed: {len(skipped)}")
    logger.info(f"  No Cu % (filtered out): {len(no_cu_pdfs)}")
    logger.info(f"  To process: {len(pdfs_to_process)}")
    logger.info("=" * 60)

    if skipped:
        logger.info(f"\nSkipping already processed: {[p.name for p in skipped]}")
    test_count = 0
    all_extracted_data = []
    for pdf_path in pdfs_to_process:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {pdf_path.name}")
            logger.info("=" * 60)

            # Register document in DB
            doc_id = db.compute_document_id(str(pdf_path))
            db.upsert_document(
                conn, doc_id, pdf_path.name, str(pdf_path),
                pdf_path.stat().st_size
            )

            extracted_data = run_process_pdfs(pdf_path)
            all_extracted_data.extend(extracted_data)

        except Exception as e:
            pdf_errors += 1
            logger.error(f"ERROR processing {pdf_path.name}: {e} — skipping to next PDF")
            db.log_step_error(conn, run_id, "pdf_extraction", e)
            continue

        test_count += 1
        if test_count >= max_pdfs:
            break

    db.mark_step_completed(conn, run_id, "pdf_extraction")

    total_flowsheets = len([x for x in all_extracted_data if x['type'] == 'flowsheet'])
    total_other = len([x for x in all_extracted_data if x['type'] == 'other'])
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP 1 COMPLETE: {total_flowsheets} flowsheets, {total_other} other images from {len(pdfs_to_process)} new PDFs")
    if pdf_errors:
        logger.warning(f"  Errors: {pdf_errors} PDFs failed (logged, skipped)")
    if skipped:
        logger.info(f"  (Skipped {len(skipped)} already-processed PDFs)")
    logger.info("=" * 60)

    # Step 1b: Extract tables from PDFs
    if not skip_tables:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1b: Extracting tables from PDFs")
        logger.info("=" * 60)
        all_grades = []
        for pdf_path in pdfs_to_process[:test_count] if test_count else []:
            try:
                pdf_doc_id = db.compute_document_id(str(pdf_path))
                tables, grades = run_extract_tables(
                    pdf_path, conn=conn, run_id=run_id,
                    document_id=pdf_doc_id)
                all_grades.extend(grades)
            except Exception as e:
                logger.error(f"ERROR extracting tables from {pdf_path.name}: {e}")
                db.log_step_error(conn, run_id, "table_extraction", e)
        db.mark_step_completed(conn, run_id, "table_extraction")
        if all_grades:
            avg_grade = sum(g["value"] for g in all_grades) / len(all_grades)
            logger.info(f"  Average copper grade: {avg_grade:.3f}% ({len(all_grades)} values)")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Running LLM extraction on flowsheets")
    logger.info("=" * 60)
    flowsheets = [x for x in all_extracted_data if x['type'] == 'flowsheet']
    if flowsheets:
        run_test_refinement(all_extracted_data, conn=conn, run_id=run_id,
                            document_id=doc_id if pdfs_to_process else None,
                            results_path=results_path)
    elif not pdfs_to_process:
        logger.info("All PDFs already processed - skipping LLM extraction")
    else:
        logger.info("No new flowsheets found - skipping LLM extraction")
    db.mark_step_completed(conn, run_id, "llm_extraction")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Processing output to normalize IDs")
    logger.info("=" * 60)
    run_process_output(conn=conn, run_id=run_id, results_path=results_path)
    db.mark_step_completed(conn, run_id, "normalization")

    if not skip_network:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Building stage network")
        logger.info("=" * 60)
        run_build_stage_network(results_path=results_path)
        db.mark_step_completed(conn, run_id, "graph")

    # Finalize run
    errors = db.get_errors_for_run(conn, run_id)
    status = "completed_with_errors" if errors else "completed"
    db.complete_run(conn, run_id, status)
    conn.close()

    logger.info("\n" + "=" * 60)
    logger.info(f"PIPELINE COMPLETE (status: {status})")
    if errors:
        logger.warning(f"  {len(errors)} error(s) logged — see data/pipeline.db step_errors table")
    logger.info("=" * 60)

    return {
        "status": status,
        "run_id": run_id,
        "pdfs_processed": test_count,
        "flowsheets_found": total_flowsheets,
        "other_images_found": total_other,
        "pdf_errors": pdf_errors,
        "errors": len(errors) if errors else 0,
    }


def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Mineral processing PDF extraction pipeline"
    )
    parser.add_argument(
        "pdfs", nargs="*",
        help="Specific PDF files to process (overrides --pdf-dir)"
    )
    parser.add_argument(
        "--pdf-dir", default=None,
        help=f"Directory containing PDFs (default: {config.PDF_DIR})"
    )
    parser.add_argument(
        "--max-pdfs", type=int, default=None,
        help=f"Maximum number of PDFs to process (default: {config.MAX_PDFS})"
    )
    parser.add_argument(
        "--results", default=None,
        help=f"Path for results JSON (default: {config.RESULTS_PATH})"
    )
    parser.add_argument(
        "--skip-clip", action="store_true",
        help="Disable CLIP vectorization for non-flowsheet images"
    )
    parser.add_argument(
        "--skip-tables", action="store_true",
        help="Skip table extraction step"
    )
    parser.add_argument(
        "--skip-network", action="store_true",
        help="Skip stage network building step"
    )
    parser.add_argument(
        "--skip-cu-filter", action="store_true",
        help="Skip Cu%% pre-filter (process all PDFs regardless of copper content)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()
    result = run_pipeline(
        pdf_dir=args.pdf_dir,
        pdf_files=args.pdfs or None,
        max_pdfs=args.max_pdfs,
        results_path=args.results,
        skip_clip=args.skip_clip,
        skip_tables=args.skip_tables,
        skip_network=args.skip_network,
        skip_cu_filter=args.skip_cu_filter,
    )
