"""
Test flowsheet detection against known-positive PDFs.

All PDFs in data/contains_pfs/ are confirmed to contain at least one
process flowsheet. This test verifies the ensemble detection pipeline finds them.

Usage:
    python test_pfs_detection.py
    python -m pytest test_pfs_detection.py -v
"""
import sys
import logging
import tempfile
from pathlib import Path

# Add pipeline directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pdf_extraction import extract_flowsheets_ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CONTAINS_PFS_DIR = Path(__file__).parent.parent.parent / "data" / "contains_pfs"


def run_detection_test():
    """Run ensemble flowsheet detection on all known-positive PDFs."""
    if not CONTAINS_PFS_DIR.exists():
        logger.error(f"Directory not found: {CONTAINS_PFS_DIR}")
        return

    pdfs = sorted(CONTAINS_PFS_DIR.glob("*.pdf"))
    if not pdfs:
        logger.error(f"No PDFs found in {CONTAINS_PFS_DIR}")
        return

    logger.info(f"Testing {len(pdfs)} PDFs from {CONTAINS_PFS_DIR}\n")

    results = {}
    total_found = 0
    total_missed = 0

    for pdf_path in pdfs:
        logger.info("=" * 70)
        logger.info(f"Processing: {pdf_path.name}")
        logger.info("=" * 70)

        try:
            extracted = extract_flowsheets_ensemble(
                str(pdf_path),
                output_folder=Path(tempfile.mkdtemp(prefix="pfs_test_")),
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results[pdf_path.name] = {"status": "ERROR", "error": str(e)}
            total_missed += 1
            continue

        flowsheets = [x for x in extracted if x["type"] == "flowsheet"]

        if flowsheets:
            total_found += 1
            results[pdf_path.name] = {
                "status": "FOUND",
                "count": len(flowsheets),
                "pages": [f["page"] for f in flowsheets],
                "passes": [f.get("detection_pass", "keyword") for f in flowsheets],
            }
            for f in flowsheets:
                pass_name = f.get("detection_pass", "keyword")
                logger.info(f"  FOUND flowsheet on page {f['page']} (pass: {pass_name})")
        else:
            total_missed += 1
            results[pdf_path.name] = {"status": "MISSED", "count": 0}
            logger.warning(f"  MISSED — no flowsheets detected")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DETECTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total PDFs:  {len(pdfs)}")
    logger.info(f"  Found:       {total_found}/{len(pdfs)}")
    logger.info(f"  Missed:      {total_missed}/{len(pdfs)}")
    logger.info(f"  Detection rate: {total_found/len(pdfs)*100:.0f}%")

    logger.info("\nPer-PDF results:")
    for name, res in results.items():
        if res["status"] == "FOUND":
            pages = ", ".join(str(p) for p in res["pages"])
            passes = set(res["passes"])
            logger.info(f"  OK   {name}: {res['count']} flowsheet(s) on pages [{pages}] via {passes}")
        elif res["status"] == "MISSED":
            logger.info(f"  MISS {name}: no flowsheets detected")
        else:
            logger.info(f"  ERR  {name}: {res['error']}")

    return results


def run_detection_on_folder(folder_path):
    """Run ensemble detection on a folder and save flowsheets with named output.

    Saves to {folder}/pfs_images/pfs_{n}_{pdf_stem}.png
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Directory not found: {folder}")
        return

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        logger.error(f"No PDFs found in {folder}")
        return

    output_dir = folder / "pfs_images"
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Processing {len(pdfs)} PDFs from {folder}")
    logger.info(f"Output: {output_dir}\n")

    results = {}
    total_found = 0
    total_missed = 0

    for pdf_path in pdfs:
        logger.info("=" * 70)
        logger.info(f"Processing: {pdf_path.name}")
        logger.info("=" * 70)

        try:
            extracted = extract_flowsheets_ensemble(
                str(pdf_path),
                output_folder=Path(tempfile.mkdtemp(prefix="pfs_run_")),
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results[pdf_path.name] = {"status": "ERROR", "error": str(e)}
            total_missed += 1
            continue

        flowsheets = [x for x in extracted if x["type"] == "flowsheet"]
        pdf_stem = pdf_path.stem

        if flowsheets:
            total_found += 1
            pages = []
            for i, f in enumerate(flowsheets, 1):
                pass_name = f.get("detection_pass", "keyword")
                logger.info(f"  FOUND flowsheet on page {f['page']} (pass: {pass_name})")
                pages.append(f["page"])

                # Copy image to output dir with naming convention
                src = Path(f.get("image_path", ""))
                if src.exists():
                    dst = output_dir / f"pfs_{i}_{pdf_stem}.png"
                    import shutil
                    shutil.copy2(str(src), str(dst))
                    logger.info(f"  Saved: {dst.name}")

            results[pdf_path.name] = {
                "status": "FOUND",
                "count": len(flowsheets),
                "pages": pages,
            }
        else:
            total_missed += 1
            results[pdf_path.name] = {"status": "MISSED", "count": 0}
            logger.warning(f"  MISSED — no flowsheets detected")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DETECTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total PDFs:  {len(pdfs)}")
    logger.info(f"  Found:       {total_found}/{len(pdfs)}")
    logger.info(f"  Missed:      {total_missed}/{len(pdfs)}")
    if pdfs:
        logger.info(f"  Detection rate: {total_found/len(pdfs)*100:.0f}%")
    logger.info(f"  Output dir: {output_dir}")

    logger.info("\nPer-PDF results:")
    for name, res in results.items():
        if res["status"] == "FOUND":
            pages = ", ".join(str(p) for p in res["pages"])
            logger.info(f"  OK   {name}: {res['count']} flowsheet(s) on pages [{pages}]")
        elif res["status"] == "MISSED":
            logger.info(f"  MISS {name}: no flowsheets detected")
        else:
            logger.info(f"  ERR  {name}: {res['error']}")

    return results


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        run_detection_on_folder(_sys.argv[1])
    else:
        run_detection_on_folder(
            str(Path(__file__).parent.parent.parent / "data" / "contains_pfs_2")
        )
