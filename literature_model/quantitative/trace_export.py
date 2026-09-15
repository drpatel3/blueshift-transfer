"""Freeze KNN decision traces to a de-identified JSON for the stakeholder demo.

Reuses the exact prediction/trace path from `trace_demo` (`_prepare` /
`_trace_for`), then *pseudonymizes* every document: the query becomes
"This project" and its k nearest analogs become "Analog A".."Analog E"
(ranked by distance). The mapping is applied consistently across the analog
cards and the per-stage / per-edge vote references, so no SEDAR document id or
company name leaves the corpus — only feature profiles (ore type, grade,
tonnage, ...), the votes, and each document's year (from `doc_metadata.json`)
remain. The year is de-identifying on its own — it names no company or
property, but it says how old the evidence behind each vote is. The output JSON is pure and self-contained;
`trace_report.py` renders it with no AWS/numpy.

Usage:
    python -m quantitative.trace_export --list          # route signatures, to pick examples
    python -m quantitative.trace_export --indices 0,2,5 --out demo/traces.json
"""
from __future__ import annotations

import argparse
import json
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classification"))

from quantitative import doc_metadata
from quantitative import trace_demo as td

# Stages whose presence characterizes the route family, for --list selection.
_ROUTE_MARKERS = ["flotation", "leach", "heap", "sx", "solvent_extraction",
                  "electrowinning", "ew", "smelter", "concentrate",
                  "rougher", "cleaner", "scavenger", "regrind"]


def _provenance(doc_id: str, meta: dict) -> dict:
    """Vintage and study grade of a document — how old the disclosure behind a
    vote is, and how engineered the study was. De-identifying, so both survive
    pseudonymization: a year and a study grade name no company or property.

    Only the grades in `doc_metadata.DISPLAYED_STUDY_TYPES` (FS/PFS) are
    carried; everything else exports no grade at all rather than a label the
    evidence doesn't support."""
    m = meta.get(doc_id) or {}
    study = m.get("study_type")
    shown = study if study in doc_metadata.DISPLAYED_STUDY_TYPES else None
    return {
        "year": m.get("year"),
        "year_source": m.get("year_source"),
        "study_type": shown,
        "study_type_label": doc_metadata.STUDY_LABELS.get(shown) if shown else None,
        # Whether the report states *a* grade, separately from whether that
        # grade is one we display. Without this the renderer cannot tell a
        # report that declares itself a PEA from one that declares nothing,
        # and would have to call both "not stated" — which is false for the
        # first and understates how conceptual that evidence is.
        "grade_stated": bool(study and study not in ("TR", "UNKNOWN")),
    }


def pseudonymize(trace: dict, example_label: str, meta: dict) -> dict:
    """Return a copy of `trace` with all document ids replaced by stable
    pseudonyms: the query -> `example_label`, each ranked neighbor -> Analog A..
    Every supporting-neighbor reference is rewritten through the same map, so a
    stage/edge backer named "Analog C" is exactly the third analog card."""
    alias = {trace["query_doc_id"]: example_label}
    for nb in trace["neighbors"]:
        alias[nb["doc_id"]] = f"Analog {string.ascii_uppercase[nb['rank']]}"

    def relabel_supporters(supporters):
        return [{**s, "doc_id": alias.get(s["doc_id"], "other")}
                for s in supporters]

    return {
        "query_doc_id": example_label,
        "k": trace["k"],
        "provenance": _provenance(trace["query_doc_id"], meta),
        "neighbors": [
            {**nb, "doc_id": alias[nb["doc_id"]],
             "provenance": _provenance(nb["doc_id"], meta)}
            for nb in trace["neighbors"]
        ],
        "stage_votes": [
            {**sv, "supporting_neighbors": relabel_supporters(sv["supporting_neighbors"])}
            for sv in trace["stage_votes"]
        ],
        "edge_votes": [
            {**e, "supporting_neighbors": relabel_supporters(e["supporting_neighbors"])}
            for e in trace["edge_votes"]
        ],
    }


def _route_signature(trace: dict) -> str:
    """One-line summary of a query's selected stages, for --list."""
    selected = [sv["stage"] for sv in trace["stage_votes"] if sv["selected"]]
    markers = [s for s in selected if any(m in s for m in _ROUTE_MARKERS)]
    return (f"stages={len(selected):2d} edges="
            f"{sum(e['selected'] for e in trace['edge_votes']):2d}  "
            f"route: {', '.join(markers) or '(no route markers)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs-bucket", default="sagemaker-us-east-1-666109694894")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--edge-threshold", type=float, default=0.3)
    ap.add_argument("--edge-alpha", type=float, default=1.0)
    ap.add_argument("--list", action="store_true",
                    help="Print a route signature per validation doc and exit.")
    ap.add_argument("--indices", default="0,1,2",
                    help="Comma-separated validation indices to freeze.")
    ap.add_argument("--labels", default=None,
                    help="Comma-separated display labels (default: Project 1, 2, ...).")
    ap.add_argument("--out", default="quantitative/demo/traces.json")
    args = ap.parse_args()

    print(f"Loading corpus from s3://{args.graphs_bucket}/graphs/ (cached) ...",
          file=sys.stderr)
    ctx = td._prepare(args.graphs_bucket, args.k, args.edge_threshold, args.edge_alpha)
    n_val = len(ctx["val"]["doc_ids"])
    print(f"  {len(ctx['train']['doc_ids'])} train / {n_val} val docs, "
          f"{len(ctx['stage_vocab'])} stages", file=sys.stderr)

    if args.list:
        for qi in range(n_val):
            print(f"  [{qi:2d}] {_route_signature(td._trace_for(ctx, qi))}")
        return

    indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]
    labels = (args.labels.split(",") if args.labels
              else [f"Project {i + 1}" for i in range(len(indices))])

    meta = doc_metadata.load()
    if not meta:
        print("  WARNING: no doc_metadata.json — run "
              "`python -m quantitative.doc_metadata` first; year / study type "
              "will render as unknown.", file=sys.stderr)

    examples = []
    for label, qi in zip(labels, indices):
        examples.append(pseudonymize(td._trace_for(ctx, qi), label.strip(), meta))
        print(f"  froze idx {qi} as {label.strip()!r}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"examples": examples}, indent=2))
    print(f"Wrote {len(examples)} de-identified traces -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
