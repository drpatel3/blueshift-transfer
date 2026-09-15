"""Freeze XGBoost V2 decision traces to a de-identified JSON for the demo.

The KNN trace could say "five real projects voted for this block". XGBoost has
no analogs — it decides each stage from that document's section text — so the
honest explanation is a different one: for every stage, the probability the
model produced, the decision threshold it had to clear, whether it cleared it,
which report sections drive that stage's classifier, and whether the answer was
actually right on the validation set.

Everything is re-derived from the frozen checkpoint in
`classification/xgb_checkpoint_v2/` — no model reload, no S3, no re-scoring:

  val_predictions.json    per-doc stage probabilities + ground truth (36 docs)
  thresholds.json         the per-stage decision threshold (they differ a lot:
                          crusher 0.68, tailing 0.90 — a bare probability is
                          meaningless without its own cut)
  section_influences.json which NI 43-101 sections drive each stage classifier
  transition_counts.json  corpus edge frequencies, for the flowsheet arrows
  val_predictions per_stage   per-stage held-out F1, so each block carries its
                          own track record instead of one headline number

Documents are pseudonymized to a display label; no SEDAR id or company name
reaches the output. Year and study grade come from `doc_metadata.json`.

Usage:
    python -m quantitative.xgb_trace_export --list
    python -m quantitative.xgb_trace_export --indices 0,5,11 --out quantitative/demo/xgb_traces.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantitative import doc_metadata

CHECKPOINT = (Path(__file__).resolve().parent.parent / "classification"
              / "xgb_checkpoint_v2")

# Sections carry a stage's influence weight only if they clear this share of
# the strongest section for that stage — keeps the evidence chips to the ones
# that actually drove the classifier.
INFLUENCE_FLOOR = 0.35
TOP_SECTIONS = 3

# An edge is drawn when both endpoints were predicted and the corpus has seen
# that transition at least this often. This is the checkpoint's "conditional"
# connection baseline (connection_metrics.json: F1 0.364) — corpus frequency,
# not a learned edge model, and the demo says so.
MIN_EDGE_SUPPORT = 3

# Support alone is not a flowsheet: every corpus transition between predicted
# stages clears it, giving ~5 arrows per stage and a hairball. That IS the
# checkpoint's conditional baseline (recall 1.0 by drawing everything,
# precision 0.22), but as a picture it implies a plant nobody would build.
# Keep transitions worth at least this share of the strongest one, matching the
# 0.3 cut the KNN dossier uses, which lands near 1.2 arrows per stage.
EDGE_SCORE_MIN = 0.3

# Half of XGBoost V2's feature vector is `neighbor_diff_*`: the document's own
# features minus the weighted average of its most similar disclosed projects
# (stage_predictor._compute_neighbor_features, k=10, weighted by
# overall_similarity). Those projects are therefore identifiable, per
# prediction, and materially shape it — 959 of 1918 features. This is the
# model's real document-level provenance.
MODEL_NEIGHBOR_K = 10
TOP_INFLUENCES = 5
EDGES_BUCKET = "sagemaker-us-east-1-666109694894"
EDGES_KEY = "graphs/cross_doc_edges.json"
# Sections whose similarity is reported as the reason two projects match.
TOP_MATCH_SECTIONS = 3


def load_cross_doc_edges(bucket: str = EDGES_BUCKET, key: str = EDGES_KEY) -> dict:
    """Cross-document similarity edges, cached beside the graph corpus so the
    S3 fetch happens at most once per machine."""
    import os
    import pickle

    cache = (Path(os.environ.get("TEMP", "/tmp")) / "trace_demo_cache"
             / "cross_doc_edges.pkl")
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    import boto3
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    edges = json.loads(body)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(edges, f)
    return edges


def influencing_reports(doc_id: str, edges: dict, train_ids: set, meta: dict,
                        grades: frozenset = None) -> tuple[list, int]:
    """The disclosed projects this prediction was actually differenced against.

    Mirrors `_compute_neighbor_features` exactly: neighbours in stored order,
    kept only if they are in the model's labelled set, first
    MODEL_NEIGHBOR_K of them. Returns (shown, total) so the report can say how
    many of the model's neighbours the filtered list represents — showing five
    graded reports out of ten without saying so would overstate their share.
    """
    grades = grades or doc_metadata.DISPLAYED_STUDY_TYPES
    used = [n for n in edges.get(doc_id, {}).get("neighbors", [])
            if n.get("doc_id") in train_ids][:MODEL_NEIGHBOR_K]

    shown = []
    for nb in used:
        m = meta.get(nb["doc_id"]) or {}
        if m.get("study_type") not in grades:
            continue
        sims = nb.get("section_similarities") or {}
        top = sorted(sims.items(), key=lambda kv: -kv[1])[:TOP_MATCH_SECTIONS]
        shown.append({
            "similarity": round(nb.get("overall_similarity", 0.0), 3),
            "year": m.get("year"),
            "year_source": m.get("year_source"),
            "study_type": m.get("study_type"),
            "study_type_label": doc_metadata.STUDY_LABELS.get(m.get("study_type")),
            "matches_on": [{"section": s, "similarity": round(v, 3)}
                           for s, v in top],
        })
    shown.sort(key=lambda r: -r["similarity"])
    for rank, row in enumerate(shown[:TOP_INFLUENCES]):
        row["label"] = f"Report {chr(65 + rank)}"
    return shown[:TOP_INFLUENCES], len(used)


def corpus_vintage(meta: dict, path: Path = CHECKPOINT) -> dict:
    """Vintage and grade mix of the documents the model learned from.

    XGBoost references no specific document per prediction — unlike the KNN, it
    has no analogs — so the only honest vintage to show is that of the whole
    training corpus, once, rather than per decision. Read from the out-of-fold
    predictions, which enumerate the training documents."""
    ids = [d["doc_id"] for d in
           json.loads((path / "oof_predictions.json").read_text())]
    years = sorted(y for y in ((meta.get(i) or {}).get("year") for i in ids) if y)
    grades = Counter((meta.get(i) or {}).get("study_type", "UNKNOWN") for i in ids)
    return {
        "docs": len(ids),
        "year_min": years[0] if years else None,
        "year_max": years[-1] if years else None,
        "year_median": years[len(years) // 2] if years else None,
        "grades": dict(grades.most_common()),
    }


def load_checkpoint(path: Path = CHECKPOINT) -> dict:
    def _read(name):
        return json.loads((path / name).read_text())

    val = _read("val_predictions.json")
    return {
        "val": val,
        "thresholds": _read("thresholds.json"),
        "influences": _read("section_influences.json"),
        "transitions": _read("transition_counts.json"),
    }


def stage_rows(doc: dict, ck: dict) -> list[dict]:
    """One row per stage the model scored: probability against its own cut,
    the sections behind that classifier, and its held-out track record."""
    inf = ck["influences"]
    sections, stages = inf["sections"], inf["stages"]
    matrix = inf["influence_matrix"]
    per_stage = ck["val"]["per_stage"]
    truth = set(doc["ground_truth"])

    # The checkpoint stores only the stages the model returned, so a stage the
    # model missed is absent rather than scored-and-rejected. Those are the
    # model's real errors (14 across the val set) and must appear in the trace,
    # not vanish from it — carried with probability None, meaning "not
    # returned", which is honestly different from "scored below its cut".
    scored = doc["xgb_predicted"]
    rows = []
    for stage in sorted(set(scored) | truth,
                        key=lambda s: -scored.get(s, -1.0)):
        prob = scored.get(stage)
        threshold = ck["thresholds"].get(stage, 0.5)
        selected = prob is not None and prob >= threshold

        drivers = []
        if stage in stages:
            col = stages.index(stage)
            weights = [(sections[r], matrix[r][col]) for r in range(len(sections))]
            top = max((w for _, w in weights), default=0.0)
            if top > 0:
                drivers = [{"section": s, "weight": round(w / top, 3)}
                           for s, w in sorted(weights, key=lambda x: -x[1])
                           if w / top >= INFLUENCE_FLOOR][:TOP_SECTIONS]

        rows.append({
            "stage": stage,
            "probability": round(prob, 4) if prob is not None else None,
            "threshold": round(threshold, 3),
            "selected": selected,
            "in_ground_truth": stage in truth,
            "val_f1": per_stage.get(stage, {}).get("f1"),
            "val_support": per_stage.get(stage, {}).get("support"),
            "sections": drivers,
        })
    return rows


def edge_rows(rows: list[dict], ck: dict) -> list[dict]:
    """Flowsheet arrows: corpus transitions between predicted stages."""
    predicted = {r["stage"] for r in rows if r["selected"]}
    counts = {}
    for key, n in ck["transitions"].items():
        src, _, dst = key.partition("|")
        if src in predicted and dst in predicted and n >= MIN_EDGE_SUPPORT:
            counts[(src, dst)] = n
    if not counts:
        return []
    top = max(counts.values())
    return [{"src": s, "dst": d, "support": n, "score": round(n / top, 3)}
            for (s, d), n in sorted(counts.items(), key=lambda kv: -kv[1])
            if n / top >= EDGE_SCORE_MIN]


def doc_accuracy(rows: list[dict]) -> dict:
    """How this document actually came out — the model's own scorecard."""
    hit = sum(1 for r in rows if r["selected"] and r["in_ground_truth"])
    extra = sum(1 for r in rows if r["selected"] and not r["in_ground_truth"])
    missed = sum(1 for r in rows if not r["selected"] and r["in_ground_truth"])
    prec = hit / max(hit + extra, 1)
    rec = hit / max(hit + missed, 1)
    return {
        "correct": hit, "extra": extra, "missed": missed,
        "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
    }


def build_example(doc: dict, ck: dict, label: str, meta: dict,
                  edges: dict, train_ids: set) -> dict:
    rows = stage_rows(doc, ck)
    influences, n_used = influencing_reports(doc["doc_id"], edges, train_ids, meta)
    m = meta.get(doc["doc_id"]) or {}
    study = m.get("study_type")
    shown = study if study in doc_metadata.DISPLAYED_STUDY_TYPES else None
    return {
        "label": label,
        "provenance": {
            "year": m.get("year"),
            "year_source": m.get("year_source"),
            "study_type": shown,
            "study_type_label": doc_metadata.STUDY_LABELS.get(shown) if shown else None,
            "grade_stated": bool(study and study not in ("TR", "UNKNOWN")),
        },
        "stages": rows,
        "edges": edge_rows(rows, ck),
        "accuracy": doc_accuracy(rows),
        "influences": influences,
        "influences_of": n_used,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="Print one line per validation doc and exit.")
    ap.add_argument("--indices", default="0,1,2")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default="quantitative/demo/xgb_traces.json")
    args = ap.parse_args()

    ck = load_checkpoint()
    meta = doc_metadata.load()
    docs = ck["val"]["per_doc"]

    if args.list:
        for i, doc in enumerate(docs):
            rows = stage_rows(doc, ck)
            acc = doc_accuracy(rows)
            m = meta.get(doc["doc_id"]) or {}
            sel = [r["stage"] for r in rows if r["selected"]]
            print(f"  [{i:2d}] {m.get('year') or '????'} "
                  f"{m.get('study_type', '?'):<4} stages={len(sel):2d} "
                  f"F1={acc['f1']:.2f} (+{acc['extra']}/-{acc['missed']})  "
                  f"{', '.join(sel[:6])}")
        return

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    if args.labels:
        labels = args.labels.split(",")
    else:
        # Label by the project's own head grade, the way the KNN dossier does.
        # Deriving it here rather than typing it on the command line keeps the
        # tab label from drifting away from the document it names.
        grades = json.loads(
            (Path(__file__).resolve().parent / "grade_map.json").read_text())
        labels = []
        for n, i in enumerate(indices):
            g = grades.get(docs[i]["doc_id"])
            labels.append(f"Cu {g:.2f}%" if g else f"Project {n + 1}")

    edges = load_cross_doc_edges()
    train_ids = {d["doc_id"] for d in json.loads(
        (CHECKPOINT / "oof_predictions.json").read_text())}
    examples = [build_example(docs[i], ck, lab.strip(), meta, edges, train_ids)
                for lab, i in zip(labels, indices)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "examples": examples,
        "model": {
            "name": "XGBoost V2",
            "val_macro_f1": ck["val"]["metrics"]["macro_f1"],
            "val_micro_f1": ck["val"]["metrics"]["micro_f1"],
            "val_docs": ck["val"]["metrics"]["num_docs"],
            "connection_f1": 0.364,
            "corpus": corpus_vintage(meta),
        },
    }, indent=2))
    print(f"Wrote {len(examples)} XGBoost traces -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
