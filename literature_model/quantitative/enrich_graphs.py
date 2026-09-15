"""Enrich existing S3 graphs with targeted numeric features.

Reads each graph from graphs/, finds the matching pipeline result,
extracts grade_cu, recovery, tonnage, etc. from table headers,
patches the features into context nodes, and writes back.

No DeBERTa, no GPU, no SageMaker needed.

Usage:
    python quantitative/enrich_graphs.py
    python quantitative/enrich_graphs.py --dry-run   # preview without writing
"""

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from statistics import mean

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GRAPHS_BUCKET = "sagemaker-us-east-1-666109694894"
GRAPHS_PREFIX = "graphs/"
RESULTS_BUCKET = "mineral-pipeline-pipeline"
RESULTS_PREFIX = "results/"

SKIP_KEYS = {"cross_doc_edges.json", "stage_vocab.json",
             "graphs_summary.json", "checkpoint.json"}

# Section grouping (from graph_pipeline.py)
SECTION_GROUPS = {
    2: "summary", 
    3: "summary",
    4: "property",
    5: "climate",
    6: "history",
    7: "geology", 
    8: "geology",
    9: "exploration",
    10: "drilling",
    11: "sample_analysis",
    12: "data_verification",
    13: "metallurgical_testing",
    14: "resource_estimate", 
    15: "resource_estimate",
    16: "mining_method",
    17: "recovery_methods",
    18: "infrastructure",
    19: "market",
    20: "environmental",
    21: "economics", 
    22: "economics",
    23: "interpretation",
    24: "references", 
    25: "references", 
    26: "references", 
    27: "references",
}

# Targeted extraction patterns (from graph_pipeline.py)
_COLUMN_PATTERNS = {
    "grade_cu": (re.compile(r'\bcu\b|\bcopper\b|\bcu\s*[(%]', re.I), 0.01, 10.0),
    "grade_au": (re.compile(r'\bau\b|\bgold\b|g/t\s*au', re.I), 0.01, 100.0),
    "grade_ag": (re.compile(r'\bag\b|\bsilver\b|g/t\s*ag', re.I), 0.1, 1000.0),
    "recovery": (re.compile(r'\brecov\w*\b|\brec\s*%', re.I), 1.0, 100.0),
    "tonnage":  (re.compile(r'\btonn\w*\b|\btons\b|\bmt\b', re.I), 0.001, 1e9),
    "grind_size": (re.compile(r'\bp80\b|\bgrind\b|\bmicron\b|\bμm\b', re.I), 10.0, 500.0),
    "npv":      (re.compile(r'\bnpv\b', re.I), 0.0, 1e12),
    "capex":    (re.compile(r'\bcapex\b|\bcapital\s*cost', re.I), 0.0, 1e12),
    "opex":     (re.compile(r'\bopex\b|\boperating\s*cost', re.I), 0.0, 1e12),
}

TARGETED_FEATURE_NAMES = []
for name in _COLUMN_PATTERNS:
    TARGETED_FEATURE_NAMES.extend([f"{name}_min", f"{name}_mean", f"{name}_max"])


def _parse_numeric(cell_text):
    if not cell_text or not isinstance(cell_text, str):
        return None
    cleaned = cell_text.strip().replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_targeted_numerics(tables: list[dict]) -> dict:
    """Run targeted column-header extraction on tables. Returns {name_min/mean/max: val}."""
    targeted_values = {name: [] for name in _COLUMN_PATTERNS}

    for table in tables:
        headers = table.get("headers", []) or []
        rows = table.get("rows", []) or []

        for col_idx, header in enumerate(headers):
            header_str = str(header).lower()
            for name, (pattern, lo, hi) in _COLUMN_PATTERNS.items():
                if not pattern.search(header_str):
                    continue
                for row in rows:
                    if col_idx >= len(row):
                        continue
                    val = _parse_numeric(row[col_idx])
                    if val is not None and lo <= val <= hi:
                        targeted_values[name].append(val)

    features = {}
    for name, values in targeted_values.items():
        if values:
            features[f"{name}_min"] = min(values)
            features[f"{name}_mean"] = mean(values)
            features[f"{name}_max"] = max(values)
        else:
            features[f"{name}_min"] = 0.0
            features[f"{name}_mean"] = 0.0
            features[f"{name}_max"] = 0.0

    return features


def group_tables_by_section(doc: dict) -> dict:
    """Group tables by NI 43-101 section. Returns {group_name: [tables]}."""
    groups = defaultdict(list)

    # Build page -> section map
    page_section_map = {}
    for pt in doc.get("page_text", []):
        sec = pt.get("section_number")
        page = pt.get("page")
        if sec and page:
            try:
                sec_int = int(sec)
                if sec_int in SECTION_GROUPS:
                    page_section_map[page] = SECTION_GROUPS[sec_int]
            except (ValueError, TypeError):
                pass

    for table in doc.get("tables", []):
        sec = table.get("section_number")
        group = None
        if sec:
            try:
                group = SECTION_GROUPS.get(int(sec))
            except (ValueError, TypeError):
                pass
        if not group:
            t_page = table.get("page")
            if t_page and page_section_map:
                closest_page = min(page_section_map.keys(),
                                   key=lambda p: abs(p - t_page))
                if abs(closest_page - t_page) <= 3:
                    group = page_section_map[closest_page]
        if group:
            groups[group].append(table)

    return dict(groups)


def build_docid_to_result_key(s3) -> dict:
    """Build mapping: document_id (hash) -> S3 result key."""
    logger.info("Building document_id -> result key mapping...")
    mapping = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=RESULTS_BUCKET, Prefix=RESULTS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            size = obj["Size"]
            # Read first 200 bytes to get document_id
            try:
                resp = s3.get_object(Bucket=RESULTS_BUCKET, Key=key, Range="bytes=0-200")
                head = resp["Body"].read().decode("utf-8", errors="replace")
                match = re.search(r'"document_id"\s*:\s*"([^"]+)"', head)
                if match:
                    mapping[match.group(1)] = key
            except Exception as e:
                logger.warning(f"Failed to read head of {key}: {e}")
    logger.info(f"Mapped {len(mapping)} document IDs to result keys")
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Enrich graphs with targeted numeric features")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    t0 = time.time()

    # Step 1: Build document_id -> result S3 key mapping
    docid_to_key = build_docid_to_result_key(s3)

    # Step 2: List all graphs
    paginator = s3.get_paginator("list_objects_v2")
    graph_keys = []
    for page in paginator.paginate(Bucket=GRAPHS_BUCKET, Prefix=GRAPHS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json") and key.split("/")[-1] not in SKIP_KEYS:
                graph_keys.append(key)
    logger.info(f"Found {len(graph_keys)} graphs to enrich")

    # Step 3: Enrich each graph
    enriched = 0
    skipped_no_result = 0
    skipped_already = 0
    features_added = defaultdict(int)

    for i, graph_key in enumerate(graph_keys):
        resp = s3.get_object(Bucket=GRAPHS_BUCKET, Key=graph_key)
        graph = json.loads(resp["Body"].read())
        doc_id = graph.get("document_id", "")

        # Check if already enriched
        already_has = False
        for node in graph.get("nodes", []):
            if node.get("type") == "context":
                if "grade_cu_mean" in node.get("features", {}):
                    already_has = True
                    break
        if already_has:
            skipped_already += 1
            continue

        # Find matching pipeline result
        result_key = docid_to_key.get(doc_id)
        if not result_key:
            skipped_no_result += 1
            continue

        # Load pipeline result
        try:
            resp = s3.get_object(Bucket=RESULTS_BUCKET, Key=result_key)
            doc = json.loads(resp["Body"].read())
        except Exception as e:
            logger.warning(f"Failed to load result for {doc_id[:20]}: {e}")
            skipped_no_result += 1
            continue

        # Group tables by section
        section_tables = group_tables_by_section(doc)
        del doc  # free memory

        # Patch each context node with targeted features
        modified = False
        for node in graph.get("nodes", []):
            if node.get("type") != "context":
                continue
            group = node.get("group")
            tables = section_tables.get(group, [])
            targeted = extract_targeted_numerics(tables)
            node["features"].update(targeted)
            # Track which features got nonzero values
            for fname, val in targeted.items():
                if val != 0.0:
                    features_added[fname] += 1
            modified = True

        if modified and not args.dry_run:
            s3.put_object(
                Bucket=GRAPHS_BUCKET,
                Key=graph_key,
                Body=json.dumps(graph, ensure_ascii=False).encode("utf-8"),
            )
            enriched += 1
        elif modified:
            enriched += 1

        if (i + 1) % 100 == 0:
            logger.info(f"  {i + 1}/{len(graph_keys)} processed, {enriched} enriched...")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.0f}s: {enriched} enriched, "
                f"{skipped_no_result} no result, {skipped_already} already enriched")

    if features_added:
        logger.info("Features with nonzero values across all graphs:")
        for fname, count in sorted(features_added.items(), key=lambda x: -x[1]):
            pct = 100 * count / max(enriched, 1)
            logger.info(f"  {fname}: {count} graphs ({pct:.0f}%)")


if __name__ == "__main__":
    main()
