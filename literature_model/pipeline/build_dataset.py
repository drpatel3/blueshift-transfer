"""Build training dataset: ore grade + all tables + flowsheet stages.

Extracts Cu% ore grade, all tables (grouped by 43-101 section), and
process flowsheet stages from PDFs. Outputs to dataset.json.

Usage:
    python build_dataset.py                          # run all steps
    python build_dataset.py --step grades            # only scan grades
    python build_dataset.py --step tables            # only extract tables
    python build_dataset.py --step flowsheets        # only extract flowsheets
    python build_dataset.py --step tables --max 3    # tables for first 3 PDFs
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from table_parser import get_cu_grade_average
from mini.normalize import normalize_stage

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "dataset.json"


def load_results(output_path):
    """Load existing dataset.json if present."""
    if Path(output_path).exists():
        try:
            with open(output_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def save_results(output_path, data):
    """Write results to JSON."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def init_results():
    """Create empty results structure."""
    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pdf_count": 0,
            "documents_with_grade": 0,
            "documents_with_tables": 0,
            "documents_with_flowsheets": 0,
            "total_llm_cost_usd": 0,
        },
        "documents": [],
    }


# ─── Extraction helpers ─────────────────────────────────────────────

def extract_all_tables(pdf_path):
    """Extract all tables from a PDF, grouped by section.

    Returns dict keyed by section number:
        {"13": {"title": "Mineral Processing ...", "tables": [...]}, ...}
    Each table has: caption, page, headers, rows.
    """
    from pdf_extraction import extract_tables

    raw_tables = extract_tables(pdf_location=str(pdf_path), target_section=None)

    sections = {}
    for t in raw_tables:
        sec = t.get("section_number") or "unknown"
        if sec not in sections:
            sections[sec] = {
                "title": t.get("section_title") or "",
                "tables": [],
            }
        sections[sec]["tables"].append({
            "caption": t.get("caption", ""),
            "page": t.get("page"),
            "headers": t.get("headers", []),
            "rows": t.get("rows", []),
        })

    return sections


def extract_stages(pdf_path):
    """Extract flowsheet stages from a PDF via image extraction + LLM.

    Returns list of flowsheet dicts, each with:
        flowsheet_id, page, stages, units, connections, llm_cost_usd
    """
    from pdf_extraction import extract_all_figures
    from flowsheet_extraction import extract_with_terms

    extracted = extract_all_figures(pdf_location=str(pdf_path), vectorize=False,
                                    image_types=("flowsheet",))
    flowsheets = [x for x in extracted if x["type"] == "flowsheet"]

    results = []
    for item in flowsheets:
        image_path = item["image_path"]
        flowsheet_id = os.path.splitext(os.path.basename(image_path))[0]

        logger.info(f"  LLM extracting: {flowsheet_id}")
        result = extract_with_terms(str(image_path))

        if "error" in result:
            logger.info(f"    extraction failed: {result['error']}")
            continue

        data = result.get("data", {})
        results.append({
            "flowsheet_id": flowsheet_id,
            "page": item.get("page"),
            "stages": data.get("stages", []),
            "units": data.get("units", []),
            "connections": data.get("connections", []),
            "llm_cost_usd": result.get("total_cost", 0),
        })

    return results


def build_stage_sequence(stages):
    """Derive ordered, normalized stage sequence from raw stages."""
    sorted_stages = sorted(stages, key=lambda s: s.get("order", 0))
    normalized = [normalize_stage(s["id"]) for s in sorted_stages]
    normalized = [s for s in normalized if s is not None]

    deduped = []
    for stage_id in normalized:
        if not deduped or deduped[-1] != stage_id:
            deduped.append(stage_id)

    return deduped


# ─── Pipeline steps (each independently callable) ───────────────────

dir_1 = Path(os.getenv("PDF_DIR", "."))


def step_grades(pdf_dir, results, output_path, max_pdfs=None):
    """Step 1: Scan PDFs for Cu% grade data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Checking for Cu% grade data")
    logger.info("=" * 60)

    known_ids = {d["document_id"] for d in results["documents"]}
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    new_pdfs = [p for p in pdf_files if p.stem not in known_ids]
    if max_pdfs:
        new_pdfs = new_pdfs[:max_pdfs]

    if not new_pdfs:
        graded = len([d for d in results["documents"]
                      if d["ore_grade_cu_pct"] is not None])
        logger.info(f"  All {len(pdf_files)} PDFs already scanned "
              f"({graded} with grade data)\n")
        return

    logger.info(f"  {len(new_pdfs)} new PDF(s) to scan "
          f"({len(known_ids)} already in results)")
    for pdf_path in new_pdfs:
        grade = get_cu_grade_average(str(pdf_path))
        if grade is not None:
            logger.info(f"  {pdf_path.name}: Cu = {grade:.3f}%")
        else:
            logger.info(f"  {pdf_path.name}: no Cu%")
        results["documents"].append({
            "document_id": pdf_path.stem,
            "filename": pdf_path.name,
            "ore_grade_cu_pct": round(grade, 4) if grade else None,
            "flowsheets": [],
        })

    results["metadata"]["pdf_count"] = len(results["documents"])
    results["metadata"]["documents_with_grade"] = len(
        [d for d in results["documents"] if d["ore_grade_cu_pct"] is not None])
    save_results(output_path, results)
    graded = len([d for d in results["documents"]
                  if d["ore_grade_cu_pct"] is not None])
    logger.info(f"\n  Saved {len(results['documents'])} documents "
          f"({graded} with grade) -> {output_path}\n")


def step_tables(pdf_dir, results, output_path, max_pdfs=None):
    """Step 1b: Extract all tables from all sections."""
    docs_needing_tables = [d for d in results["documents"]
                           if "sections" not in d]
    if max_pdfs:
        docs_needing_tables = docs_needing_tables[:max_pdfs]

    logger.info("=" * 60)
    logger.info(f"STEP 1b: Extracting tables ({len(docs_needing_tables)} PDFs to process)")
    logger.info("=" * 60)

    if not docs_needing_tables:
        logger.info("  All documents already have table data")
        return

    for doc in docs_needing_tables:
        pdf_path = Path(pdf_dir) / doc["filename"]
        if not pdf_path.exists():
            logger.info(f"  {doc['filename']}: file not found — skipped")
            doc["sections"] = {}
            continue

        try:
            sections = extract_all_tables(pdf_path)
            doc["sections"] = sections
            table_count = sum(len(s["tables"]) for s in sections.values())
            logger.info(f"  {doc['filename']}: {table_count} tables in "
                  f"{len(sections)} sections")
        except Exception as e:
            logger.info(f"  ERROR {doc['filename']}: {e}")
            doc["sections"] = {}

        save_results(output_path, results)

    results["metadata"]["documents_with_tables"] = len(
        [d for d in results["documents"]
         if d.get("sections") and any(d["sections"].values())])
    save_results(output_path, results)


def step_flowsheets(pdf_dir, results, output_path, max_pdfs=None):
    """Step 2: Extract flowsheets for documents with grades."""
    graded = [d for d in results["documents"]
              if d["ore_grade_cu_pct"] is not None]
    to_process = [d for d in graded if not d["flowsheets"]]
    if max_pdfs:
        to_process = to_process[:max_pdfs]

    logger.info("=" * 60)
    logger.info(f"STEP 2: Extracting flowsheets ({len(to_process)} PDFs to process)")
    logger.info("=" * 60)

    if not to_process:
        logger.info("  All documents already have flowsheet data")
        return

    total_cost = results["metadata"].get("total_llm_cost_usd", 0)

    for doc in to_process:
        pdf_path = Path(pdf_dir) / doc["filename"]
        if not pdf_path.exists():
            logger.info(f"  {doc['filename']}: file not found — skipped")
            continue

        try:
            flowsheet_results = extract_stages(pdf_path)

            if not flowsheet_results:
                logger.info(f"  {doc['filename']}: no flowsheets found")
                continue

            for fs in flowsheet_results:
                stage_sequence = build_stage_sequence(fs["stages"])
                doc["flowsheets"].append({
                    "flowsheet_id": fs["flowsheet_id"],
                    "page": fs["page"],
                    "stage_sequence": stage_sequence,
                    "stages": fs["stages"],
                    "connections": fs["connections"],
                })
                total_cost += fs.get("llm_cost_usd", 0)

            logger.info(f"  {doc['filename']}: {len(doc['flowsheets'])} flowsheet(s)")

        except Exception as e:
            logger.info(f"  ERROR {doc['filename']}: {e}")

        results["metadata"]["total_llm_cost_usd"] = round(total_cost, 4)
        results["metadata"]["documents_with_flowsheets"] = len(
            [d for d in results["documents"] if d["flowsheets"]])
        save_results(output_path, results)


def run_pipeline(pdf_dir=dir_1, output_path=None, steps=None, max_pdfs=None):
    """Run the dataset pipeline. Steps can be run independently.

    Args:
        steps: list of step names to run, or None for all.
               Valid: "grades", "tables", "flowsheets"
        max_pdfs: limit number of PDFs to process per step
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = BASE_DIR / pdf_dir

    existing = load_results(output_path)
    results = existing if existing else init_results()

    run_all = steps is None
    step_set = set(steps) if steps else set()

    if run_all or "grades" in step_set:
        step_grades(pdf_dir, results, output_path, max_pdfs)

    if run_all or "tables" in step_set:
        step_tables(pdf_dir, results, output_path, max_pdfs)

    if run_all or "flowsheets" in step_set:
        step_flowsheets(pdf_dir, results, output_path, max_pdfs)

    # ── Summary ───────────────────────────────────────────────────
    graded = len([d for d in results["documents"]
                  if d["ore_grade_cu_pct"] is not None])
    with_tables = len([d for d in results["documents"]
                       if d.get("sections") and any(d["sections"].values())])
    with_flowsheets = len([d for d in results["documents"] if d["flowsheets"]])

    results["metadata"]["documents_with_grade"] = graded
    results["metadata"]["documents_with_tables"] = with_tables
    results["metadata"]["documents_with_flowsheets"] = with_flowsheets
    save_results(output_path, results)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"COMPLETE: {len(results['documents'])} documents -> {output_path}")
    logger.info(f"  With grade: {graded}")
    logger.info(f"  With tables: {with_tables}")
    logger.info(f"  With flowsheets: {with_flowsheets}")
    cost = results["metadata"].get("total_llm_cost_usd", 0)
    if cost:
        logger.info(f"  Total LLM cost: ${cost:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build training dataset from PDFs")
    parser.add_argument("--pdf-dir", default=str(dir_1),
                        help="Directory containing PDF files")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: dataset.json)")
    parser.add_argument("--step", choices=["grades", "tables", "flowsheets"],
                        action="append", dest="steps",
                        help="Run specific step(s) only (repeatable)")
    parser.add_argument("--max", type=int, default=None,
                        help="Max PDFs to process per step")
    args = parser.parse_args()

    run_pipeline(pdf_dir=args.pdf_dir, output_path=args.output,
                 steps=args.steps, max_pdfs=args.max)
