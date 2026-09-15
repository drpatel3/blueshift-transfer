"""Merge grade_map_au_new_batch.json (stem-keyed, 119 entries) into the
canonical doc_id-keyed quantitative/grade_map_au.json.

The new-batch file from PROGRESS.md item #2 has never been merged. Until it
is, eval_gold and any downstream tool that consults grade_map_au.json sees
~37 fewer gold training docs than were actually labeled in xgb_checkpoint_au.

Inputs:
    %TEMP%/grade_map_au_new_batch.json  : {stem: au_grade_g_per_t}
    tmp/inspect/_docid_to_result_key.json : {doc_id: result_key} for the stem map
    quantitative/grade_map_au.json        : existing doc_id-keyed map

Output:
    quantitative/grade_map_au.json        : updated in place (entries added,
                                            existing entries left alone)

Usage:
    python dev/gold/merge_new_batch_grades.py
    python dev/gold/merge_new_batch_grades.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

DEV_ROOT = Path(__file__).resolve().parent.parent
DOCID_INDEX = DEV_ROOT / "tmp" / "inspect" / "_docid_to_result_key.json"
AU_MAP = DEV_ROOT / "quantitative" / "grade_map_au.json"
NEW_BATCH = Path(os.environ.get("TEMP", "/tmp")) / "grade_map_au_new_batch.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for p in (DOCID_INDEX, AU_MAP, NEW_BATCH):
        if not p.exists():
            raise SystemExit(f"Missing: {p}")

    with open(DOCID_INDEX, encoding="utf-8") as f:
        docid_to_key = json.load(f)
    stem_to_docid = {
        re.sub(r"\.json$", "", v.rsplit("/", 1)[-1]): k
        for k, v in docid_to_key.items()
    }

    with open(AU_MAP, encoding="utf-8") as f:
        au_map = json.load(f)
    before = len(au_map)

    with open(NEW_BATCH, encoding="utf-8") as f:
        new_batch = json.load(f)

    matched, unmatched, already_present = 0, 0, 0
    for stem, grade in new_batch.items():
        doc_id = stem_to_docid.get(stem)
        if doc_id is None:
            unmatched += 1
            continue
        if doc_id in au_map:
            already_present += 1
            continue
        au_map[doc_id] = grade
        matched += 1

    print(f"new-batch entries:  {len(new_batch)}")
    print(f"  added:            {matched}")
    print(f"  already present:  {already_present}")
    print(f"  no doc_id match:  {unmatched}")
    print(f"grade_map_au.json:  {before} -> {len(au_map)}")

    if args.dry_run:
        print("Dry-run; not writing.")
        return

    with open(AU_MAP, "w", encoding="utf-8") as f:
        json.dump(au_map, f, indent=2, sort_keys=True)
    print(f"Wrote {AU_MAP}")


if __name__ == "__main__":
    main()
