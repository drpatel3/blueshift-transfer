"""Audit Au-in-range docs that aren't in the V2 prediction set.

For each unlabeled gold doc, bucket the failure mode so we know whether
it's recoverable (worth re-running LLM extraction) or a dead end
(no PFS image was ever detected).

Bucketing is done in two phases to keep S3 costs bounded:

  Phase 1 (graph-only, fast): uses the document graph already on S3
  to derive:
    - missing_graph    : doc didn't make it through graph_pipeline at all
    - no_stages        : graph exists but has no stage nodes (no flowsheet
                         was extracted — could be "no PFS image in PDF"
                         or "LLM extraction failed"; phase 2 disambiguates)
    - disconnected     : stages exist but no stage_transition edges
    - rejected         : stages exist but none normalize to V2 vocab
    - orphan_labeled   : graph has valid stages+edges but doc wasn't
                         in the V2 training corpus (stale checkpoint)

  Phase 2 (S3 result fetch, sampled): for a sample of `no_stages` docs,
  fetch the per-PDF result JSON and check whether any flowsheet images
  were detected. Splits the bucket into:
    - no_pfs_image    : no flowsheet-type images in PDF — dead end
    - llm_error       : flowsheet images detected but extraction failed
                        — worth retrying

Usage:
    python quantitative/audit_unlabeled_gold.py
    python quantitative/audit_unlabeled_gold.py --sample 50  # larger phase-2 sample
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEV_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = DEV_ROOT / "classification" / "xgb_checkpoint_v2"
AU_MAP_PATH = Path(__file__).parent / "grade_map_au.json"

GRAPHS_BUCKET = "sagemaker-us-east-1-666109694894"
GRAPHS_PREFIX = "graphs/"
RESULTS_BUCKET = "mineral-pipeline-pipeline"
RESULTS_PREFIX = "results/"

for _p in [str(DEV_ROOT / "classification"), str(DEV_ROOT / "quantitative")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_labeled_ids() -> set:
    """Doc IDs that appear in V2 predictions (val + oof + test)."""
    ids = set()
    for fname in ["val_predictions.json", "oof_predictions.json",
                   "test_predictions.json"]:
        path = CHECKPOINT_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        per_doc = data["per_doc"] if isinstance(data, dict) else data
        for entry in per_doc:
            ids.add(entry["doc_id"])
    return ids


def _classify_graph(graph: dict, v2_stages: set) -> str:
    """Phase 1 bucket based on the graph's stage content."""
    stage_nodes = [n for n in graph.get("nodes", []) if n.get("type") == "stage"]
    stage_edges = [e for e in graph.get("edges", []) if e.get("type") == "stage_transition"]

    if not stage_nodes:
        return "no_stages"
    if not stage_edges:
        return "disconnected"

    canonical_hits = 0
    for n in stage_nodes:
        raw_id = n.get("stage_id", n["id"].replace("stg_", ""))
        if raw_id in v2_stages:
            canonical_hits += 1
    if canonical_hits == 0:
        return "rejected"
    return "orphan_labeled"


def _build_docid_to_result_key(s3) -> dict:
    """Map document_id → result S3 key by reading JSON heads. ~30s."""
    mapping = {}
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
                m = re.search(r'"document_id"\s*:\s*"([^"]+)"', head)
                if m:
                    mapping[m.group(1)] = key
            except Exception:
                continue
    return mapping


def _classify_result(doc: dict) -> tuple[str, dict]:
    """Phase 2 bucket for a 'no_stages' doc. Returns (bucket, debug_counts)."""
    flowsheets = doc.get("flowsheets") or {}
    extracted_images = doc.get("extracted_images") or []
    tables = doc.get("tables") or []

    # Count images that look like flowsheets (detector tagged them)
    fs_images, total_images = 0, 0
    if isinstance(extracted_images, list):
        total_images = len(extracted_images)
        fs_images = sum(1 for im in extracted_images
                         if im.get("type") == "flowsheet")
    elif isinstance(extracted_images, dict):
        total_images = len(extracted_images)
        fs_images = sum(1 for im in extracted_images.values()
                         if isinstance(im, dict) and im.get("type") == "flowsheet")

    # Flowsheets dict may store successful extractions, error records, or
    # text-context entries for non-flowsheet images
    fs_entries = flowsheets if isinstance(flowsheets, dict) else {}
    n_flowsheet_type = sum(
        1 for v in fs_entries.values()
        if isinstance(v, dict) and v.get("type") == "flowsheet"
    )
    n_errors = sum(
        1 for v in fs_entries.values()
        if isinstance(v, dict) and "error" in v
    )

    debug = {
        "total_images": total_images,
        "fs_images": fs_images,
        "n_flowsheet_records": n_flowsheet_type,
        "n_errors": n_errors,
        "n_tables": len(tables) if isinstance(tables, list) else 0,
        "has_page_text": bool(doc.get("page_text")),
    }

    # If the doc wasn't processed at all (no images, no tables, no page_text),
    # it's never been through pdf_extraction — that's a different problem from
    # "processed but no PFS image found".
    if total_images == 0 and not tables and not doc.get("page_text"):
        return "never_processed", debug
    if total_images == 0:
        return "never_extracted_images", debug
    if fs_images == 0 and n_flowsheet_type == 0:
        return "no_pfs_image", debug
    if n_errors > 0 and n_flowsheet_type == n_errors:
        return "llm_error", debug
    return "extracted_but_dropped", debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=30,
                        help="Phase-2 sample size for no_stages bucket")
    args = parser.parse_args()

    with open(AU_MAP_PATH) as f:
        au_map = json.load(f)
    au_ids = set(au_map.keys())

    labeled_ids = _load_labeled_ids()
    unlabeled_ids = au_ids - labeled_ids
    logger.info(
        f"Au-in-range: {len(au_ids)} | labeled (has V2 pred): "
        f"{len(au_ids & labeled_ids)} | unlabeled: {len(unlabeled_ids)}"
    )

    with open(CHECKPOINT_DIR / "stage_vocab.json") as f:
        v2_vocab = set(json.load(f))

    # Phase 1 — walk graphs.
    from stage_predictor import _load_graphs_from_s3
    graphs = _load_graphs_from_s3(GRAPHS_BUCKET, GRAPHS_PREFIX)
    graphs_by_id = {g.get("document_id", ""): g for g in graphs}

    buckets = defaultdict(list)
    for doc_id in unlabeled_ids:
        if doc_id not in graphs_by_id:
            buckets["missing_graph"].append(doc_id)
            continue
        buckets[_classify_graph(graphs_by_id[doc_id], v2_vocab)].append(doc_id)

    print(f"\n{'=' * 72}")
    print("  Phase 1 — Audit of unlabeled Au-in-range docs")
    print(f"{'=' * 72}\n")
    total = sum(len(v) for v in buckets.values())
    for bucket, ids in sorted(buckets.items(), key=lambda x: -len(x[1])):
        pct = 100 * len(ids) / max(total, 1)
        print(f"  {bucket:<22} {len(ids):>4}  ({pct:>4.1f}%)")
    print(f"  {'TOTAL':<22} {total:>4}\n")

    # Phase 2 — sample `no_stages` for S3 result fetch.
    no_stage_ids = buckets.get("no_stages", [])
    if not no_stage_ids:
        print("No `no_stages` docs to disambiguate. Done.")
        return

    sample_size = min(args.sample, len(no_stage_ids))
    sample = random.Random(42).sample(no_stage_ids, sample_size)
    logger.info(
        f"Phase 2: sampling {sample_size} of {len(no_stage_ids)} "
        f"no_stages docs to determine no_pfs_image vs llm_error..."
    )

    s3 = boto3.client("s3")
    docid_to_key = _build_docid_to_result_key(s3)

    sub_buckets = defaultdict(list)
    debug_samples = defaultdict(list)
    missing_result = 0
    for doc_id in sample:
        key = docid_to_key.get(doc_id)
        if not key:
            missing_result += 1
            continue
        try:
            resp = s3.get_object(Bucket=RESULTS_BUCKET, Key=key)
            doc = json.loads(resp["Body"].read())
        except Exception as e:
            logger.warning(f"  failed to load result for {doc_id[:12]}: {e}")
            missing_result += 1
            continue
        bucket, debug = _classify_result(doc)
        sub_buckets[bucket].append(doc_id)
        if len(debug_samples[bucket]) < 3:
            debug_samples[bucket].append((doc_id[:12], debug))

    print(f"\n{'=' * 72}")
    print(f"  Phase 2 — {sample_size}-doc sample from `no_stages` bucket")
    print(f"{'=' * 72}\n")
    for bucket, ids in sorted(sub_buckets.items(), key=lambda x: -len(x[1])):
        pct = 100 * len(ids) / max(sample_size, 1)
        print(f"  {bucket:<26} {len(ids):>4}  ({pct:>4.1f}%)")
    if missing_result:
        print(f"  {'(missing_result)':<26} {missing_result:>4}")

    # Debug samples — one line per example so we can eyeball the counts
    print("\n  Sample debug counts per bucket:")
    for bucket, samples in sub_buckets.items():
        print(f"    {bucket}:")
        for doc_id, dbg in debug_samples[bucket]:
            print(f"      {doc_id}  images={dbg['total_images']:>3} "
                  f"fs_imgs={dbg['fs_images']:>2} "
                  f"fs_recs={dbg['n_flowsheet_records']:>2} "
                  f"errs={dbg['n_errors']:>2} "
                  f"tables={dbg['n_tables']:>3} "
                  f"page_text={dbg['has_page_text']}")

    # Extrapolation
    print(f"\n  Extrapolating to the full {len(no_stage_ids)} `no_stages` docs:")
    for bucket, ids in sorted(sub_buckets.items(), key=lambda x: -len(x[1])):
        est = int(round(len(ids) / sample_size * len(no_stage_ids)))
        print(f"    ~{est:>4} {bucket}")

    print("\nRecommendation cheat-sheet:")
    print("  never_processed        -> run full pipeline on these PDFs")
    print("  never_extracted_images -> run pdf_extraction only")
    print("  no_pfs_image           -> PDF has no PFS; dead end")
    print("  llm_error              -> re-run flowsheet_extraction")
    print("  extracted_but_dropped  -> check normalize_ids synonyms")
    print("  disconnected           -> LLM output sparse; try refinement mode")
    print("  rejected               -> stage names not in V2 vocab")
    print("  orphan_labeled         -> stale V2 checkpoint; include next retrain")
    print()


if __name__ == "__main__":
    main()
