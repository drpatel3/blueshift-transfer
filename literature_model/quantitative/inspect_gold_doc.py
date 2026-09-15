"""Inspect a single gold (Au) doc end-to-end — show only the inputs that matter.

Pulls one doc's result JSON (local or S3), then surfaces:
  - PFS images only            -> copied/downloaded to tmp/inspect/<stem>/pfs/
  - A few text snippets        -> from mineralogy-relevant sections (7,8,13)
  - Mineralogy-bearing tables  -> filtered by keyword match on headers/caption/first column

Designed as a spot-check tool while iterating on the gold model. Does not
modify the pipeline; reads-only.

Usage:
    python quantitative/inspect_gold_doc.py <pdf_stem>
    python quantitative/inspect_gold_doc.py --random        # pick from grade_map_au
    python quantitative/inspect_gold_doc.py <pdf_stem> --text-samples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEV_ROOT = Path(__file__).resolve().parent.parent
AU_MAP_PATH = Path(__file__).parent / "grade_map_au.json"
INSPECT_ROOT = DEV_ROOT / "tmp" / "inspect"

RESULTS_BUCKET = "mineral-pipeline-pipeline"
RESULTS_PREFIX = "results/"
DOCID_INDEX_PATH = INSPECT_ROOT / "_docid_to_result_key.json"

# Mineralogy-relevant section indices (NI 43-101)
#   7,8 = geology / mineralogy / deposit type
#   13  = metallurgical testing
MINERALOGY_SECTIONS = {7, 8, 13}

MINERALOGY_KEYWORDS = (
    "mineralogy", "mineral assemblage", "mineral composition", "modal",
    "chalcopyrite", "chalcocite", "bornite", "covellite", "enargite",
    "pyrite", "pyrrhotite", "arsenopyrite", "sphalerite", "galena",
    "tetrahedrite", "tennantite", "molybdenite",
    "native gold", "electrum", "tellurides", "free gold",
    "sulfide", "sulphide", "sulphidation", "oxide ore",
    "gangue", "silicates", "carbonates",
    "qemscan", "mla", "modal abundance",
)


def _load_local_result(pdf_stem: str) -> dict | None:
    """Try local tmp/results/<stem>.json before reaching for S3."""
    candidates = [
        DEV_ROOT / "tmp" / "results" / f"{pdf_stem}.json",
        DEV_ROOT / "data" / "results" / f"{pdf_stem}.json",
    ]
    for path in candidates:
        if path.exists():
            logger.info(f"Loaded local result: {path}")
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _build_docid_index(s3) -> dict:
    """Scan results/ HEADs to build {document_id -> s3_key}. Cached on disk."""
    if DOCID_INDEX_PATH.exists():
        try:
            return json.loads(DOCID_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    import re as _re
    logger.info("Building document_id index from S3 results/ ... (~30s, one-time)")
    mapping: dict = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=RESULTS_BUCKET, Prefix=RESULTS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                resp = s3.get_object(Bucket=RESULTS_BUCKET, Key=key,
                                     Range="bytes=0-300")
                head = resp["Body"].read().decode("utf-8", errors="replace")
                m = _re.search(r'"document_id"\s*:\s*"([^"]+)"', head)
                if m:
                    mapping[m.group(1)] = key
            except Exception:
                continue

    DOCID_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCID_INDEX_PATH.write_text(json.dumps(mapping), encoding="utf-8")
    logger.info(f"Indexed {len(mapping)} docs -> {DOCID_INDEX_PATH}")
    return mapping


def _load_s3_result(pdf_stem: str) -> dict | None:
    import boto3
    s3 = boto3.client("s3")

    # First try: pdf_stem is already the S3 key stem
    direct_key = f"{RESULTS_PREFIX}{pdf_stem}.json"
    try:
        resp = s3.get_object(Bucket=RESULTS_BUCKET, Key=direct_key)
        logger.info(f"Loaded S3 result: s3://{RESULTS_BUCKET}/{direct_key}")
        return json.loads(resp["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        pass
    except s3.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
            raise

    # Fallback: pdf_stem is actually a document_id; resolve via index
    index = _build_docid_index(s3)
    indirect_key = index.get(pdf_stem)
    if not indirect_key:
        return None
    resp = s3.get_object(Bucket=RESULTS_BUCKET, Key=indirect_key)
    logger.info(f"Loaded S3 result via document_id: s3://{RESULTS_BUCKET}/{indirect_key}")
    return json.loads(resp["Body"].read().decode("utf-8"))


def load_result(pdf_stem: str) -> dict:
    res = _load_local_result(pdf_stem) or _load_s3_result(pdf_stem)
    if res is None:
        raise FileNotFoundError(
            f"No result JSON found locally or at s3://{RESULTS_BUCKET}/{RESULTS_PREFIX}{pdf_stem}.json"
        )
    return res


def pick_random_au_doc() -> str:
    if not AU_MAP_PATH.exists():
        raise FileNotFoundError(f"{AU_MAP_PATH} not found — run build_au_grade_map.py first")
    grade_map = json.loads(AU_MAP_PATH.read_text(encoding="utf-8"))
    return random.choice(list(grade_map.keys()))


def download_pfs_images(result: dict, pdf_stem: str) -> list[Path]:
    """Pull each flowsheet's PNG out of S3 (or copy from local) into tmp/inspect/<stem>/pfs/."""
    out_dir = INSPECT_ROOT / pdf_stem / "pfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    flowsheets = result.get("flowsheets", {}) or {}
    if not flowsheets:
        return []

    saved: list[Path] = []
    s3 = None
    for image_key, fs in flowsheets.items():
        s3_key = fs.get("s3_image_key")
        local_path = fs.get("image_path")
        dest = out_dir / f"{image_key}.png"

        if local_path and Path(local_path).exists():
            dest.write_bytes(Path(local_path).read_bytes())
            saved.append(dest)
            continue

        if s3_key:
            if s3 is None:
                import boto3
                s3 = boto3.client("s3")
            try:
                resp = s3.get_object(Bucket=RESULTS_BUCKET, Key=s3_key)
                dest.write_bytes(resp["Body"].read())
                saved.append(dest)
            except Exception as e:
                logger.warning(f"  failed to fetch {s3_key}: {e}")

    return saved


def _table_mentions_mineralogy(table: dict) -> bool:
    """Match keywords against headers, caption, and first column of rows."""
    blobs: list[str] = []
    blobs.append((table.get("caption") or "").lower())
    blobs.append((table.get("section_title") or "").lower())
    for h in table.get("headers", []) or []:
        blobs.append(str(h).lower())
    for row in (table.get("rows") or [])[:30]:
        if row:
            blobs.append(str(row[0]).lower())
    haystack = " | ".join(blobs)
    return any(kw in haystack for kw in MINERALOGY_KEYWORDS)


def filter_mineralogy_tables(result: dict) -> list[dict]:
    return [t for t in result.get("tables", []) or [] if _table_mentions_mineralogy(t)]


def filter_tables_by_section(result: dict, section_num: int) -> list[dict]:
    return [t for t in result.get("tables", []) or []
            if t.get("section_number") == section_num]


def pick_text_samples(result: dict, n: int) -> list[dict]:
    """Pick N short text snippets from mineralogy-relevant sections."""
    pages = result.get("page_text", []) or []
    relevant = [p for p in pages if p.get("section_number") in MINERALOGY_SECTIONS]
    pool = relevant if relevant else pages

    pool = sorted(pool, key=lambda p: p.get("char_count", 0), reverse=True)
    return pool[:n]


def _print_header(title: str) -> None:
    print(f"\n{'=' * 78}\n {title}\n{'=' * 78}")


def _print_table(t: dict, max_rows: int = 5) -> None:
    print(f"\n--- table p.{t.get('page')} sec.{t.get('section_number')} | "
          f"{(t.get('caption') or '')[:90]}")
    headers, rows = _clean_table(t.get("headers") or [], t.get("rows") or [])
    if headers:
        print(" | ".join(str(h)[:20] for h in headers))
        print("-" * min(78, sum(min(len(str(h)), 20) + 3 for h in headers)))
    for row in rows[:max_rows]:
        print(" | ".join(str(c)[:20] for c in row))
    if len(rows) > max_rows:
        print(f"   ... +{len(rows) - max_rows} more rows")


import re as _re

# Vocab used to detect when a header has been reversed by the PDF extractor
# (vertical column labels often come out character-flipped). Match is by
# substring containment on the reversed-letters-only form, so plurals,
# joined words ("MuscoviteIllite"), and "Cu-" prefixes all hit naturally.
_REVERSAL_VOCAB = (
    "chalcopyrite", "chalcocite", "bornite", "covellite", "enargite",
    "pyrite", "pyrrhotite", "arsenopyrite", "sphalerite", "galena",
    "tetrahedrite", "tennantite", "molybdenite",
    "chrysocolla", "malachite", "azurite", "atacamite", "cuprite", "tenorite",
    "brochantite", "antlerite",
    "goethite", "hematite", "magnetite", "limonite",
    "quartz", "feldspar", "biotite", "muscovite", "illite", "ilite",
    "kaolinite", "smectite", "chlorite", "chlorine",
    "carbonate", "calcite", "dolomite", "siderite", "ankerite",
    "mica", "sulphide", "sulfide", "sulphur", "sulfur",
    "iron", "calcium", "magnesium", "aluminum", "aluminium",
    "potassium", "manganese", "fluorine",
    # short Cu/Au/S abbreviations
    "ascu", "cncu", "rescu", "tcu", "cucn", "cuas",
    "totals", "totalcu", "cuoxide", "othercu", "ironoxide",
    "deportment", "mineralogy", "modal", "assays",
)


def _looks_reversed(letters_only_lower: str) -> bool:
    """True if the reversed-letters-only string contains a known mineral/assay word."""
    if len(letters_only_lower) < 4:
        return False
    return any(v in letters_only_lower for v in _REVERSAL_VOCAB)


def _fix_reversed_header(h: str) -> str:
    """Un-reverse PDF-vertical-text headers.

    Tries the SHORTEST leading prefix first; if its letters-only reversal
    contains a known mineral/assay word, reverses just that prefix and
    keeps the rest as-is. Shortest-first means a header like
    "u C SA Sequential Coppers %" matches at k=3 (ASCu) and stops, instead
    of greedily eating "Sequential" too.
    """
    if not h or not isinstance(h, str):
        return h
    h = h.strip()
    if not h:
        return h

    parts = h.split()
    for k in range(1, len(parts) + 1):
        prefix_raw = " ".join(parts[:k])
        letters_rev = _re.sub(r"[^a-zA-Z]", "", prefix_raw)[::-1].lower()
        if _looks_reversed(letters_rev):
            reversed_prefix = prefix_raw.replace(" ", "")[::-1]
            tail = " ".join(parts[k:])
            if not tail:
                return reversed_prefix
            tail_fixed = _fix_reversed_header(tail)
            return f"{reversed_prefix} {tail_fixed}".strip()

    return h


def _clean_cell(c) -> str:
    """Strip cell, collapse runs of whitespace to a single space."""
    if c is None:
        return ""
    s = str(c).strip()
    return _re.sub(r"\s+", " ", s)


def _clean_table(headers: list, rows: list) -> tuple[list, list]:
    """Apply header-reversal + whitespace cleanup, drop empty rows/cols."""
    headers = [_fix_reversed_header(_clean_cell(h)) for h in (headers or [])]
    cleaned_rows = [[_clean_cell(c) for c in (row or [])] for row in (rows or [])]

    # Drop fully-empty rows
    cleaned_rows = [r for r in cleaned_rows if any(c for c in r)]

    # Drop fully-empty columns (only when we have a stable column count)
    if cleaned_rows and headers:
        ncols = max(len(headers), max(len(r) for r in cleaned_rows))
        # pad rows so column slicing is safe
        padded = [r + [""] * (ncols - len(r)) for r in cleaned_rows]
        padded_headers = headers + [""] * (ncols - len(headers))
        keep_idx = [
            i for i in range(ncols)
            if padded_headers[i] or any(row[i] for row in padded)
        ]
        headers = [padded_headers[i] for i in keep_idx]
        cleaned_rows = [[row[i] for i in keep_idx] for row in padded]

    return headers, cleaned_rows


def _save_tables_csv(tables: list[dict], out_dir: Path) -> list[Path]:
    """Write each mineralogy table to its own CSV in <out_dir>/tables/."""
    import csv
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for t in tables:
        page = t.get("page", "?")
        sec = t.get("section_number", "?")
        tid = t.get("table_id") or f"p{page}_s{sec}"
        safe_tid = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(tid))
        path = tables_dir / f"{safe_tid}.csv"

        headers, rows = _clean_table(t.get("headers") or [], t.get("rows") or [])

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            cap = t.get("caption") or ""
            if cap:
                w.writerow([f"# caption: {cap}"])
                w.writerow([f"# page: {page}  section: {sec}  ({t.get('section_title') or ''})"])
            if headers:
                w.writerow(headers)
            for row in rows:
                w.writerow(row)
        paths.append(path)
    return paths


def _save_text_samples(samples: list[dict], out_dir: Path) -> Path:
    """Write all text samples (full text, not truncated) to one .txt file."""
    path = out_dir / "text_samples.txt"
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(f"===== p.{s.get('page_number')} sec.{s.get('section_number')} "
                    f"({s.get('section_title') or '?'}, {s.get('char_count')} chars) =====\n")
            f.write(s.get("text") or "")
            f.write("\n\n")
    return path


def inspect(pdf_stem: str, text_samples: int = 3) -> None:
    result = load_result(pdf_stem)
    out_dir = INSPECT_ROOT / pdf_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_header(f"DOCUMENT  {pdf_stem}")
    print(f"  filename:        {result.get('filename')}")
    print(f"  target_mineral:  {result.get('target_mineral')}")
    print(f"  status:          {result.get('status')}")
    print(f"  pages:           {len(result.get('page_text', []) or [])}")
    print(f"  tables:          {len(result.get('tables', []) or [])}")
    print(f"  flowsheets:      {len(result.get('flowsheets', {}) or {})}")

    _print_header("PFS IMAGES (flowsheets only)")
    saved = download_pfs_images(result, pdf_stem)
    if not saved:
        print("  (no flowsheet images for this doc)")
    else:
        for p in saved:
            print(f"  {p}")

    _print_header(f"TEXT SAMPLES (sections {sorted(MINERALOGY_SECTIONS)} preferred, top {text_samples})")
    samples = pick_text_samples(result, text_samples)
    if not samples:
        print("  (no page_text in result)")
    else:
        for s in samples:
            snippet = (s.get("text") or "")[:600].replace("\n", " ")
            print(f"\n  p.{s.get('page_number')} sec.{s.get('section_number')} "
                  f"({s.get('section_title') or '?'}, {s.get('char_count')} chars)")
            print(f"    {snippet}{'...' if len(s.get('text', '')) > 600 else ''}")
        text_path = _save_text_samples(samples, out_dir)
        print(f"\n  -> saved full text to {text_path}")

    _print_header("MINERALOGY TABLES (filtered)")
    min_tables = filter_mineralogy_tables(result)
    if not min_tables:
        print("  (no tables matched mineralogy keywords)")
    else:
        print(f"  matched {len(min_tables)} of {len(result.get('tables', []) or [])} tables")
        for t in min_tables:
            _print_table(t)
        csv_paths = _save_tables_csv(min_tables, out_dir)
        print(f"\n  -> saved {len(csv_paths)} CSVs to {out_dir / 'tables'}")

    print(f"\nArtifacts written to: {out_dir}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf_stem", nargs="?", help="PDF stem (filename without .pdf/.json)")
    ap.add_argument("--random", action="store_true",
                    help="Pick a random doc from grade_map_au.json")
    ap.add_argument("--text-samples", type=int, default=3)
    args = ap.parse_args()

    if args.random:
        stem = pick_random_au_doc()
        logger.info(f"Random Au-in-range doc: {stem}")
    elif args.pdf_stem:
        stem = args.pdf_stem
    else:
        ap.error("provide a pdf_stem or --random")

    inspect(stem, text_samples=args.text_samples)


if __name__ == "__main__":
    main()
