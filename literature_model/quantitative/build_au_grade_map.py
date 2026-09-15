"""Build grade_map_au.json — per-doc Au head grade, filtered to evaluable range.

Mirrors the copper grade_map.json pattern. Reads grade_au_mean off the
already-enriched S3 graphs (enrich_graphs.py), applies the same section
fallback chain used for Cu, and keeps only docs whose head grade sits in
the PFS-worthy window (0.3 – 30 g/t, i.e. ppm).

Usage:
    python quantitative/build_au_grade_map.py --graphs-bucket BUCKET
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classification"))
from stage_predictor import _load_graphs_from_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).parent / "grade_map_au.json"

# Same fallback order as resolve_cu_grade in predictor.py — head grade first,
# resource grade only as a last resort.
_AU_GRADE_FALLBACK_SECTIONS = (
    "metallurgical_testing", "recovery_methods",
    "resource_estimate", "geology", "summary",
)

# PFS-worthy head grade window for gold, in g/t (= ppm). Rationale:
#   0.3  g/t: lowest real heap-leach cutoff; below this is tailings/geochem noise
#   30.0 g/t: catches ~99% of legit mill-feed grades; above is drill-intercept /
#             concentrate / dore values leaking through the regex
AU_MIN_GRADE = 0.3
AU_MAX_GRADE = 30.0


def resolve_au_grade(graph: dict) -> float | None:
    """Pull grade_au_mean from a graph's context nodes using fallback order."""
    sections = {}
    for node in graph.get("nodes", []):
        if node.get("type") == "context":
            sections[node.get("group")] = node.get("features", {})

    for section in _AU_GRADE_FALLBACK_SECTIONS:
        val = sections.get(section, {}).get("grade_au_mean", 0.0)
        if val and val > 0:
            return float(val)
    return None


def build_map(graphs: list[dict]) -> tuple[dict, dict]:
    """Return (kept_map, stats). kept_map only contains in-range docs."""
    kept, too_low, too_high, missing = {}, 0, 0, 0
    for graph in graphs:
        doc_id = graph.get("document_id", "")
        if not doc_id:
            continue
        grade = resolve_au_grade(graph)
        if grade is None:
            missing += 1
            continue
        if grade < AU_MIN_GRADE:
            too_low += 1
            continue
        if grade > AU_MAX_GRADE:
            too_high += 1
            continue
        kept[doc_id] = grade
    stats = {
        "kept": len(kept),
        "too_low": too_low,
        "too_high": too_high,
        "missing": missing,
        "total": len(graphs),
    }
    return kept, stats


def main():
    parser = argparse.ArgumentParser(description="Build Au grade map from enriched graphs")
    parser.add_argument("--graphs-bucket", default="sagemaker-us-east-1-666109694894")
    parser.add_argument("--graphs-prefix", default="graphs/")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    graphs = _load_graphs_from_s3(args.graphs_bucket, args.graphs_prefix)
    kept, stats = build_map(graphs)

    with open(args.output, "w") as f:
        json.dump(kept, f, indent=2)

    logger.info(
        f"Au grade map: kept={stats['kept']}, "
        f"too_low (<{AU_MIN_GRADE})={stats['too_low']}, "
        f"too_high (>{AU_MAX_GRADE})={stats['too_high']}, "
        f"missing={stats['missing']}, total={stats['total']}"
    )
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
