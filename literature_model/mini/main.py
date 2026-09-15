"""Mini pipeline: ore grade -> stage sequence.

Extracts Cu% ore grade and process flowsheet stages from PDFs,
links them per document, and outputs to mini/results.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory so we can import shared modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from table_parser import get_cu_grade_average
from mini.normalize import normalize_stage

MINI_DIR = Path(__file__).resolve().parent


def load_results(output_path):
    """Load existing results.json if present."""
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


def scan_grades(pdf_dir, output_path):
    """Scan all PDFs for Cu% grade and save initial results.json.

    Every PDF gets an entry — those without grade data get null.
    Returns the saved results dict.
    """
    pdf_folder = Path(pdf_dir)
    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    documents = []
    grade_count = 0

    for pdf_path in pdf_files:
        grade = get_cu_grade_average(str(pdf_path))
        if grade is not None:
            grade_count += 1
            print(f"  {pdf_path.name}: Cu = {grade:.3f}%")
            documents.append({
                "document_id": pdf_path.stem,
                "filename": pdf_path.name,
                "ore_grade_cu_pct": round(grade, 4),
                "flowsheets": [],
            })
        else:
            print(f"  {pdf_path.name}: no Cu% — skipped")
            documents.append({
                "document_id": pdf_path.stem,
                "filename": pdf_path.name,
                "ore_grade_cu_pct": None,
                "flowsheets": [],
            })

    results = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pdf_count": len(pdf_files),
            "documents_with_grade": grade_count,
            "documents_with_flowsheets": 0,
            "total_llm_cost_usd": 0,
        },
        "documents": documents,
    }

    save_results(output_path, results)
    print(f"\n  Saved {len(documents)} documents ({grade_count} with grade) "
          f"-> {output_path}")
    return results


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

        print(f"  LLM extracting: {flowsheet_id}")
        result = extract_with_terms(str(image_path))

        if "error" in result:
            print(f"    extraction failed: {result['error']}")
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
    """Derive ordered, normalized stage sequence from raw stages.

    Sorts by order field, normalizes IDs to canonical stage types,
    filters out utility/noise stages, and deduplicates consecutive
    identical entries (e.g. two flotation stages in a row become one).
    """
    sorted_stages = sorted(stages, key=lambda s: s.get("order", 0))
    normalized = [normalize_stage(s["id"]) for s in sorted_stages]

    # Filter out noise stages (normalize_stage returns None for these)
    normalized = [s for s in normalized if s is not None]

    # Deduplicate consecutive identical normalized IDs
    deduped = []
    for stage_id in normalized:
        if not deduped or deduped[-1] != stage_id:
            deduped.append(stage_id)

    return deduped


dir_1 = Path("C:/Users/Ryan Murray/Desktop/development/main_model/internal_dev/literature/mining literature/copper")
dir_2 = "test_pdfs/main_pdfs"
def run_mini_pipeline(pdf_dir=dir_1,
                      output_path=None):
    """Run the mini pipeline: scan grades -> extract stages -> link -> write."""

    if output_path is None:
        output_path = MINI_DIR / "results.json"

    # Resolve: use as-is if absolute, otherwise relative to project root
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = MINI_DIR.parent / pdf_dir

    # Step 1: Scan grades — load existing results and scan any new PDFs
    print("=" * 60)
    print("STEP 1: Checking for Cu% grade data")
    print("=" * 60)

    existing = load_results(output_path)
    if existing:
        results = existing
    else:
        results = {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "pdf_count": 0,
                "documents_with_grade": 0,
                "documents_with_flowsheets": 0,
                "total_llm_cost_usd": 0,
            },
            "documents": [],
        }

    # Find PDFs in folder that aren't in results yet
    known_ids = {d["document_id"] for d in results["documents"]}
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    new_pdfs = [p for p in pdf_files if p.stem not in known_ids]

    if not new_pdfs:
        graded = [d for d in results["documents"]
                  if d["ore_grade_cu_pct"] is not None]
        print(f"  All {len(pdf_files)} PDFs already scanned "
              f"({len(graded)} with grade data)\n")
    else:
        print(f"  {len(new_pdfs)} new PDF(s) to scan "
              f"({len(known_ids)} already in results)")
        for pdf_path in new_pdfs:
            grade = get_cu_grade_average(str(pdf_path))
            if grade is not None:
                print(f"  {pdf_path.name}: Cu = {grade:.3f}%")
                results["documents"].append({
                    "document_id": pdf_path.stem,
                    "filename": pdf_path.name,
                    "ore_grade_cu_pct": round(grade, 4),
                    "flowsheets": [],
                })
            else:
                print(f"  {pdf_path.name}: no Cu% — skipped")
                results["documents"].append({
                    "document_id": pdf_path.stem,
                    "filename": pdf_path.name,
                    "ore_grade_cu_pct": None,
                    "flowsheets": [],
                })

        results["metadata"]["pdf_count"] = len(results["documents"])
        results["metadata"]["documents_with_grade"] = len(
            [d for d in results["documents"]
             if d["ore_grade_cu_pct"] is not None]
        )
        save_results(output_path, results)
        graded = [d for d in results["documents"]
                  if d["ore_grade_cu_pct"] is not None]
        print(f"\n  Saved {len(results['documents'])} documents "
              f"({len(graded)} with grade) -> {output_path}\n")

    # Step 2: Extract flowsheets for documents with grades
    # Only process documents that don't already have flowsheet data
    to_process = [d for d in graded if not d["flowsheets"]]

    print("=" * 60)
    print(f"STEP 2: Extracting flowsheets ({len(to_process)} PDFs to process)")
    print("=" * 60)

    if not to_process:
        print("  All documents already have flowsheet data")
    else:
        total_cost = results["metadata"].get("total_llm_cost_usd", 0)

        for doc in to_process:
            pdf_path = pdf_dir / doc["filename"]
            if not pdf_path.exists():
                print(f"  {doc['filename']}: file not found — skipped")
                continue

            try:
                flowsheet_results = extract_stages(pdf_path)

                if not flowsheet_results:
                    print(f"  {doc['filename']}: no flowsheets found")
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

                print(f"  {doc['filename']}: {len(doc['flowsheets'])} flowsheet(s)")

            except Exception as e:
                print(f"  ERROR {doc['filename']}: {e}")

            # Save after each PDF so progress isn't lost
            results["metadata"]["total_llm_cost_usd"] = round(total_cost, 4)
            results["metadata"]["documents_with_flowsheets"] = len(
                [d for d in results["documents"] if d["flowsheets"]]
            )
            save_results(output_path, results)

    # Final summary
    docs_with_flowsheets = len(
        [d for d in results["documents"] if d["flowsheets"]]
    )
    results["metadata"]["documents_with_flowsheets"] = docs_with_flowsheets
    save_results(output_path, results)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {docs_with_flowsheets}/{len(graded)} graded documents "
          f"have flowsheets -> {output_path}")
    cost = results["metadata"].get("total_llm_cost_usd", 0)
    if cost:
        print(f"  Total LLM cost: ${cost:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_mini_pipeline()
