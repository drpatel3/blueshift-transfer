"""
Classify SEDAR PDFs by mineral content.

Scans each PDF's text, counts mineral keyword mentions,
and produces a manifest listing relevant minerals per document.
Any mineral exceeding the threshold (default 5%) of total mentions is listed.

Resumes automatically — skips PDFs already in the manifest.
Skips recently-modified files (< 10s old) to avoid reading partial downloads.

Usage:
    python classify_pdfs.py ../data/sedar_downloads
    python classify_pdfs.py ../data/sedar_downloads --workers 4
    python classify_pdfs.py ../data/sedar_downloads --output ../data/pdf_manifest.json
    python classify_pdfs.py ../data/sedar_downloads --pages 30
    python classify_pdfs.py ../data/sedar_downloads --threshold 10
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mineral keyword definitions
#   - Full mineral names: re.IGNORECASE
#   - Chemical symbols: case-sensitive (0) to avoid false positives
#     e.g. "Cu" won't match "Cuba", "Ni" won't match "NI 43-101"
# ---------------------------------------------------------------------------

MINERAL_KEYWORDS = {
    "copper": [
        (r"\bcopper\b", re.IGNORECASE),
        (r"\bCu\b", 0),
        (r"\bchalcopyrite\b", re.IGNORECASE),
        (r"\bbornite\b", re.IGNORECASE),
        (r"\bchalcocite\b", re.IGNORECASE),
        (r"\bmalachite\b", re.IGNORECASE),
        (r"\bazurite\b", re.IGNORECASE),
        (r"\bcovellite\b", re.IGNORECASE),
        (r"\benargite\b", re.IGNORECASE),
        (r"\bcuprite\b", re.IGNORECASE),
        (r"\bchrysocolla\b", re.IGNORECASE),
        (r"\btennantite\b", re.IGNORECASE),
        (r"\btetrahedrite\b", re.IGNORECASE),
    ],
    "gold": [
        (r"\bgold\b", re.IGNORECASE),
        (r"\bAu\b", 0),
        (r"\belectrum\b", re.IGNORECASE),
    ],
    "silver": [
        (r"\bsilver\b", re.IGNORECASE),
        (r"\bAg\b", 0),
        (r"\bargentite\b", re.IGNORECASE),
        (r"\bacanthite\b", re.IGNORECASE),
    ],
    "zinc": [
        (r"\bzinc\b", re.IGNORECASE),
        (r"\bZn\b", 0),
        (r"\bsphalerite\b", re.IGNORECASE),
    ],
    "lead": [
        (r"\bPb\b", 0),
        (r"\bgalena\b", re.IGNORECASE),
        (r"\blead[\s-]zinc\b", re.IGNORECASE),
        (r"\blead\s+grade\b", re.IGNORECASE),
        (r"\blead\s+concentrate\b", re.IGNORECASE),
        (r"\blead\s+production\b", re.IGNORECASE),
        (r"\blead\s+recovery\b", re.IGNORECASE),
        (r"\blead\s+ore\b", re.IGNORECASE),
    ],
    "nickel": [
        (r"\bnickel\b", re.IGNORECASE),
        (r"\bNi\b", 0),
        (r"\bpentlandite\b", re.IGNORECASE),
    ],
    "iron": [
        (r"\biron\s+ore\b", re.IGNORECASE),
        (r"\bmagnetite\b", re.IGNORECASE),
        (r"\bhematite\b", re.IGNORECASE),
        (r"\bgoethite\b", re.IGNORECASE),
        (r"\bbanded\s+iron\b", re.IGNORECASE),
        (r"\bBIF\b", 0),
        (r"\biron\s+formation\b", re.IGNORECASE),
    ],
    "uranium": [
        (r"\buranium\b", re.IGNORECASE),
        (r"\bU3O8\b", 0),
        (r"\bUO2\b", 0),
        (r"\buraninite\b", re.IGNORECASE),
        (r"\bpitchblende\b", re.IGNORECASE),
    ],
    "lithium": [
        (r"\blithium\b", re.IGNORECASE),
        (r"\bspodumene\b", re.IGNORECASE),
        (r"\blepidolite\b", re.IGNORECASE),
        (r"\bpetalite\b", re.IGNORECASE),
        (r"\bLi2O\b", 0),
        (r"\bLiOH\b", 0),
        (r"\bLCE\b", 0),
    ],
    "molybdenum": [
        (r"\bmolybdenum\b", re.IGNORECASE),
        (r"\bmolybdenite\b", re.IGNORECASE),
        (r"\bMoS2\b", 0),
        (r"\bMo\b", 0),
    ],
    "cobalt": [
        (r"\bcobalt\b", re.IGNORECASE),
        # "Co" omitted — too many false positives (Company, Corp, Co.)
    ],
    "tungsten": [
        (r"\btungsten\b", re.IGNORECASE),
        (r"\bscheelite\b", re.IGNORECASE),
        (r"\bwolframite\b", re.IGNORECASE),
        (r"\bWO3\b", 0),
    ],
    "tin": [
        (r"\btin\b", re.IGNORECASE),
        (r"\bcassiterite\b", re.IGNORECASE),
        (r"\bSn\b", 0),
    ],
    "graphite": [
        (r"\bgraphite\b", re.IGNORECASE),
        (r"\bCg\b", 0),
    ],
    "platinum_group": [
        (r"\bplatinum\b", re.IGNORECASE),
        (r"\bpalladium\b", re.IGNORECASE),
        (r"\brhodium\b", re.IGNORECASE),
        (r"\bPGM\b", 0),
        (r"\bPGE\b", 0),
        (r"\bPt\b", 0),
        (r"\bPd\b", 0),
    ],
    "rare_earth": [
        (r"\brare\s+earth\b", re.IGNORECASE),
        (r"\bREE\b", 0),
        (r"\bREO\b", 0),
        (r"\bTREO\b", 0),
        (r"\bneodymium\b", re.IGNORECASE),
        (r"\blanthanum\b", re.IGNORECASE),
        (r"\bcerium\b", re.IGNORECASE),
        (r"\bpraseodymium\b", re.IGNORECASE),
        (r"\bdysprosium\b", re.IGNORECASE),
        (r"\bNdPr\b", 0),
    ],
    "manganese": [
        (r"\bmanganese\b", re.IGNORECASE),
        (r"\bMn\b", 0),
    ],
    "chromium": [
        (r"\bchromium\b", re.IGNORECASE),
        (r"\bchromite\b", re.IGNORECASE),
        (r"\bCr2O3\b", 0),
    ],
    "vanadium": [
        (r"\bvanadium\b", re.IGNORECASE),
        (r"\bV2O5\b", 0),
    ],
    "phosphate": [
        (r"\bphosphate\b", re.IGNORECASE),
        (r"\bapatite\b", re.IGNORECASE),
        (r"\bP2O5\b", 0),
    ],
    "potash": [
        (r"\bpotash\b", re.IGNORECASE),
        (r"\bsylvite\b", re.IGNORECASE),
        (r"\bcarnallite\b", re.IGNORECASE),
        (r"\bKCl\b", 0),
        (r"\bK2O\b", 0),
    ],
}


def compile_patterns():
    """Pre-compile all regex patterns for performance."""
    compiled = {}
    for mineral, patterns in MINERAL_KEYWORDS.items():
        compiled[mineral] = [re.compile(pat, flags) for pat, flags in patterns]
    return compiled


def classify_pdf(pdf_path, compiled_patterns, max_pages=None):
    """
    Count mineral keyword mentions in a PDF.

    Returns:
        (counts_dict, pages_scanned) — counts_dict maps mineral name to int count
    """
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:max_pages] if max_pages else reader.pages

    text_parts = []
    for page in pages:
        try:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        except Exception as e:
            logging.getLogger(__name__).debug(f"Page text extraction failed: {e}")
            continue

    full_text = "\n".join(text_parts)

    counts = {}
    for mineral, patterns in compiled_patterns.items():
        total = sum(len(p.findall(full_text)) for p in patterns)
        if total > 0:
            counts[mineral] = total

    return counts, len(pages)


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

_worker_patterns = None
_worker_max_pages = None


def _pool_init(max_pages):
    """Initialize each worker process with compiled patterns."""
    global _worker_patterns, _worker_max_pages
    _worker_patterns = compile_patterns()
    _worker_max_pages = max_pages


def _classify_worker(pdf_path_str):
    """Worker function for multiprocessing pool."""
    pdf_path = Path(pdf_path_str)
    filename = pdf_path.name
    try:
        counts, pages_scanned = classify_pdf(
            pdf_path, _worker_patterns, _worker_max_pages
        )
        return (filename, counts, pages_scanned, None)
    except Exception as e:
        return (filename, None, None, str(e))


def _save_manifest(path, results, threshold):
    """Write the manifest JSON atomically."""
    manifest = {
        "scan_date": time.strftime("%Y-%m-%d"),
        "threshold_pct": threshold,
        "classified": len(results),
        "files": results,
    }
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    Path(tmp).replace(path)


def _build_result(counts, pages_scanned, total_mentions, threshold):
    """Build the per-PDF result dict from mineral counts."""
    if total_mentions == 0:
        return {
            "pages_scanned": pages_scanned,
            "total_mineral_mentions": 0,
            "minerals": {},
            "primary": None,
            "relevant": [],
        }
    minerals = {
        mineral: {
            "count": count,
            "pct": round(count / total_mentions * 100, 1),
        }
        for mineral, count in sorted(counts.items(), key=lambda x: -x[1])
    }
    primary = max(counts, key=counts.get)
    relevant = [m for m, data in minerals.items() if data["pct"] >= threshold]
    return {
        "pages_scanned": pages_scanned,
        "total_mineral_mentions": total_mentions,
        "minerals": minerals,
        "primary": primary,
        "relevant": relevant,
    }


def print_summary(results, threshold=5.0):
    """Print mineral distribution breakdown from classified results."""
    logger.info(f"\nTotal classified: {len(results)} PDFs")

    mineral_pdf_counts = {}
    for data in results.values():
        for m in data.get("relevant", []):
            mineral_pdf_counts[m] = mineral_pdf_counts.get(m, 0) + 1

    total_pdfs = len(results)
    if mineral_pdf_counts:
        logger.info(f"\nRelevant mineral distribution (>= {threshold}% threshold):")
        for mineral, count in sorted(
            mineral_pdf_counts.items(), key=lambda x: -x[1]
        ):
            pct = count / total_pdfs * 100 if total_pdfs else 0
            logger.info(f"  {mineral:<20} {count:>4} PDFs   {pct:>5.1f}%")

    primary_counts = {}
    for data in results.values():
        p = data.get("primary")
        if p:
            primary_counts[p] = primary_counts.get(p, 0) + 1

    if primary_counts:
        logger.info(f"\nPrimary mineral distribution:")
        for mineral, count in sorted(
            primary_counts.items(), key=lambda x: -x[1]
        ):
            pct = count / total_pdfs * 100 if total_pdfs else 0
            logger.info(f"  {mineral:<20} {count:>4} PDFs   {pct:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Classify PDFs by mineral content"
    )
    parser.add_argument("folder", help="Folder containing PDFs")
    parser.add_argument(
        "--output", default=None,
        help="Output manifest path (default: <folder>/pdf_manifest.json)",
    )
    parser.add_argument(
        "--pages", type=int, default=None,
        help="Max pages to scan per PDF (default: all)",
    )
    parser.add_argument(
        "--threshold", type=float, default=5.0,
        help="Min %% of mentions to list a mineral (default: 5)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count, max 8)",
    )
    parser.add_argument(
        "--watch", type=int, nargs="?", const=30, default=None, metavar="SECS",
        help="After initial batch, poll for new files every N seconds (default: 30)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Just print the mineral breakdown from existing manifest and exit",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        logger.info(f"Error: {folder} is not a directory")
        sys.exit(1)

    output_path = Path(args.output) if args.output else folder / "pdf_manifest.json"

    if args.summary:
        if not output_path.exists():
            logger.info(f"No manifest found at {output_path}")
            sys.exit(1)
        with open(output_path) as f:
            data = json.load(f)
        print_summary(data.get("files", {}), args.threshold)
        return

    # Load existing manifest for resuming
    existing = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
            existing = data.get("files", {})
            logger.info(f"Resuming: {len(existing)} PDFs already classified")
        except (json.JSONDecodeError, ValueError):
            pass

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"No PDFs found in {folder}")
        sys.exit(1)

    # Filter to PDFs that need classification
    results = dict(existing)
    to_classify = []
    skipped_mtime = 0
    now = time.time()
    for pdf_path in pdf_files:
        if pdf_path.name in results:
            continue
        # Skip recently-modified files (might still be downloading)
        age = now - pdf_path.stat().st_mtime
        if age < 10:
            skipped_mtime += 1
            continue
        to_classify.append(pdf_path)

    total = len(pdf_files)
    new_count = len(to_classify)
    logger.info(f"Found {total} PDFs ({new_count} to classify)")
    if skipped_mtime:
        logger.info(f"Skipped {skipped_mtime} recently-modified files (still downloading?)")
    if args.pages:
        logger.info(f"Scanning first {args.pages} pages per PDF")
    else:
        logger.info("Scanning all pages per PDF")
    logger.info(f"Threshold: {args.threshold}%")

    workers = args.workers or min(os.cpu_count() or 1, 8)
    if new_count == 0:
        logger.info("Nothing new to classify.\n")
    elif workers > 1 and new_count > 1:
        logger.info(f"Workers: {workers}\n")
    else:
        workers = 1
        logger.info()

    errors = []
    t0 = time.time()
    done = 0

    if new_count > 0 and workers > 1:
        # ── Parallel classification ──────────────────────────────────
        paths = [str(p) for p in to_classify]
        with multiprocessing.Pool(
            workers, initializer=_pool_init, initargs=(args.pages,)
        ) as pool:
            for filename, counts, pages_scanned, error in pool.imap_unordered(
                _classify_worker, paths
            ):
                done += 1
                if error:
                    logger.info(f"  [{done}/{new_count}] {filename[:60]:<60} ERROR: {error}")
                    errors.append({"file": filename, "error": error})
                    continue

                total_mentions = sum(counts.values())
                results[filename] = _build_result(
                    counts, pages_scanned, total_mentions, args.threshold
                )
                rel_str = ", ".join(results[filename]["relevant"]) or "none"
                logger.info(
                    f"  [{done}/{new_count}] {filename[:60]:<60} "
                    f"{total_mentions:>5} mentions | {rel_str}"
                )

                if done % 10 == 0:
                    _save_manifest(output_path, results, args.threshold)
    elif new_count > 0:
        # ── Serial classification ────────────────────────────────────
        compiled = compile_patterns()
        for pdf_path in to_classify:
            filename = pdf_path.name
            done += 1
            try:
                t1 = time.time()
                counts, pages_scanned = classify_pdf(pdf_path, compiled, args.pages)
                elapsed = time.time() - t1

                total_mentions = sum(counts.values())
                results[filename] = _build_result(
                    counts, pages_scanned, total_mentions, args.threshold
                )
                rel_str = ", ".join(results[filename]["relevant"]) or "none"
                logger.info(
                    f"  [{done}/{new_count}] {filename[:60]:<60} "
                    f"{total_mentions:>5} mentions | {rel_str}  ({elapsed:.1f}s)"
                )

                if done % 10 == 0:
                    _save_manifest(output_path, results, args.threshold)
            except Exception as e:
                logger.info(f"  [{done}/{new_count}] {filename[:60]:<60} ERROR: {e}")
                errors.append({"file": filename, "error": str(e)})

    # Final save
    _save_manifest(output_path, results, args.threshold)

    # Re-scan: classify any files that were downloaded during the batch
    compiled_rescan = compile_patterns()
    while True:
        now = time.time()
        new_files = [
            p for p in sorted(folder.glob("*.pdf"))
            if p.name not in results and now - p.stat().st_mtime >= 10
        ]
        if not new_files:
            break
        logger.info(f"\n{len(new_files)} new PDF(s) appeared during batch — classifying...")
        for pdf_path in new_files:
            filename = pdf_path.name
            try:
                counts, pages_scanned = classify_pdf(
                    pdf_path, compiled_rescan, args.pages
                )
                total_mentions = sum(counts.values())
                results[filename] = _build_result(
                    counts, pages_scanned, total_mentions, args.threshold
                )
                rel_str = ", ".join(results[filename]["relevant"]) or "none"
                logger.info(
                    f"  {filename[:60]:<60} "
                    f"{total_mentions:>5} mentions | {rel_str}"
                )
            except Exception as e:
                logger.info(f"  {filename[:60]:<60} ERROR: {e}")
                errors.append({"file": filename, "error": str(e)})
        _save_manifest(output_path, results, args.threshold)

    elapsed_total = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Classified {len(results)} PDFs in {elapsed_total:.0f}s")
    logger.info(f"Manifest: {output_path}")
    if errors:
        logger.info(f"Errors: {len(errors)}")
        for e in errors:
            logger.info(f"  - {e['file']}: {e['error']}")

    print_summary(results, args.threshold)

    # ── Watch mode ────────────────────────────────────────────────────
    if not args.watch:
        return

    compiled = compile_patterns()
    logger.info(f"\nWatching for new PDFs every {args.watch}s (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(args.watch)
            now = time.time()
            new_files = []
            for p in sorted(folder.glob("*.pdf")):
                if p.name in results:
                    continue
                if now - p.stat().st_mtime < 10:
                    continue
                new_files.append(p)

            if not new_files:
                continue

            logger.info(f"\n{len(new_files)} new PDF(s) found")
            for pdf_path in new_files:
                filename = pdf_path.name
                try:
                    t1 = time.time()
                    counts, pages_scanned = classify_pdf(
                        pdf_path, compiled, args.pages
                    )
                    elapsed = time.time() - t1
                    total_mentions = sum(counts.values())
                    results[filename] = _build_result(
                        counts, pages_scanned, total_mentions, args.threshold
                    )
                    rel_str = ", ".join(results[filename]["relevant"]) or "none"
                    logger.info(
                        f"  {filename[:60]:<60} "
                        f"{total_mentions:>5} mentions | {rel_str}  ({elapsed:.1f}s)"
                    )
                except Exception as e:
                    logger.info(f"  {filename[:60]:<60} ERROR: {e}")

            _save_manifest(output_path, results, args.threshold)
            logger.info(f"  Manifest updated ({len(results)} total)")
    except KeyboardInterrupt:
        logger.info(f"\nStopped. {len(results)} PDFs classified.")


if __name__ == "__main__":
    main()
