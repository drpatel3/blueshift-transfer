"""Emit gold_training_doc_ids.json — the SHA-256 doc_ids of the 342-doc Au
training corpus, intersected with grade_map_au.json (Au-in-range docs that
also have a cached head-grade entry).

Inputs:
    - gold_training_corpus_stems.txt (342 stems; path argument; default to
      Windows %TEMP%\\gold_training_corpus_stems.txt)
    - tmp/inspect/_docid_to_result_key.json (built by extract_pfs_llm.py)
    - quantitative/grade_map_au.json

Output:
    dev/gold/gold_training_doc_ids.json — {"doc_ids": [..]}, one per line.

Usage:
    python dev/gold/build_doc_id_allowlist.py
    python dev/gold/build_doc_id_allowlist.py --stems-file <path>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

DEV_ROOT = Path(__file__).resolve().parent.parent
DOCID_INDEX_PATH = DEV_ROOT / "tmp" / "inspect" / "_docid_to_result_key.json"
AU_MAP_PATH = DEV_ROOT / "quantitative" / "grade_map_au.json"
DEFAULT_STEMS = Path(os.environ.get("TEMP", "/tmp")) / "gold_training_corpus_stems.txt"
NEW_BATCH_AU_MAP = Path(os.environ.get("TEMP", "/tmp")) / "grade_map_au_new_batch.json"
OUT_PATH = Path(__file__).resolve().parent / "gold_training_doc_ids.json"


def _stem_from_key(key: str) -> str:
    base = key.rsplit("/", 1)[-1]
    return re.sub(r"\.json$", "", base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-file", type=Path, default=DEFAULT_STEMS)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if not args.stems_file.exists():
        raise SystemExit(f"Stems file not found: {args.stems_file}")
    if not DOCID_INDEX_PATH.exists():
        raise SystemExit(
            f"Doc-ID index missing: {DOCID_INDEX_PATH}.  "
            "Run dev/gold/extract_pfs_llm.py at least once to build it."
        )

    with open(args.stems_file, encoding="utf-8", errors="replace") as f:
        stems = {line.strip() for line in f if line.strip()}

    with open(DOCID_INDEX_PATH, encoding="utf-8") as f:
        docid_to_key = json.load(f)

    au_doc_ids: set[str] = set()
    if AU_MAP_PATH.exists():
        with open(AU_MAP_PATH, encoding="utf-8") as f:
            au_doc_ids.update(json.load(f).keys())

    stem_to_docid = {_stem_from_key(v): k for k, v in docid_to_key.items()}

    if NEW_BATCH_AU_MAP.exists():
        with open(NEW_BATCH_AU_MAP, encoding="utf-8") as f:
            for stem in json.load(f).keys():
                doc_id = stem_to_docid.get(stem)
                if doc_id:
                    au_doc_ids.add(doc_id)

    matched, missing_index, missing_au = [], [], []
    for stem in sorted(stems):
        doc_id = stem_to_docid.get(stem)
        if doc_id is None:
            missing_index.append(stem)
            continue
        if au_doc_ids and doc_id not in au_doc_ids:
            missing_au.append(stem)
            continue
        matched.append(doc_id)

    payload = {
        "doc_ids": sorted(set(matched)),
        "stem_count": len(stems),
        "matched": len(matched),
        "missing_from_index": missing_index,
        "missing_from_au_map": missing_au,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Stems in: {len(stems)}")
    print(f"Matched doc_ids: {len(matched)}")
    print(f"Missing from doc-id index: {len(missing_index)}")
    print(f"Missing from grade_map_au.json: {len(missing_au)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
