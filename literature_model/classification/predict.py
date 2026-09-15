"""Flowsheet prediction — cached V2 stage predictions + corpus prior edges.

Looks up the closest doc by Cu grade from the precomputed V2 XGBoost
predictions (macro-F1=0.86), wires edges with the corpus prior, and
renders a PFS comparison visualization.

Usage:
    python -m classification.predict --grade 1.25
    python -m classification.predict --grade 0.50 1.00 1.25 1.50
    python -m classification.predict --build-cache --graphs-bucket BUCKET
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

CLASSIFICATION_DIR = Path(__file__).resolve().parent
QUANTITATIVE_DIR = CLASSIFICATION_DIR.parent / "quantitative"

# Per-commodity wiring: checkpoint dir + grade map + display units.
COMMODITIES = {
    "cu": {
        "checkpoint_dir": CLASSIFICATION_DIR / "xgb_checkpoint_v2",
        "grade_map_path": QUANTITATIVE_DIR / "grade_map.json",
        "label": "Cu Grade",
        "unit": "%",
    },
    "au": {
        "checkpoint_dir": CLASSIFICATION_DIR / "xgb_checkpoint_au",
        "grade_map_path": QUANTITATIVE_DIR / "grade_map_au.json",
        "label": "Au Grade",
        "unit": "g/t",
    },
}

# Backward-compat aliases for callers that still reference the Cu paths directly
# (e.g. build_tables_cache below). The web app uses COMMODITIES.
CHECKPOINT_DIR = COMMODITIES["cu"]["checkpoint_dir"]
GRADE_MAP_PATH = COMMODITIES["cu"]["grade_map_path"]

for _p in [str(CLASSIFICATION_DIR), str(QUANTITATIVE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _commodity_cfg(commodity: str) -> dict:
    key = commodity.lower()
    if key not in COMMODITIES:
        raise ValueError(f"Unknown commodity {commodity!r}; expected one of "
                         f"{sorted(COMMODITIES)}")
    return COMMODITIES[key]


def _load_tuned_thresholds(commodity: str = "cu"):
    """Read OOF-tuned edge thresholds; fall back to unbiased midpoints if absent.

    Fallbacks are NOT the old val-tuned values (edge=0.30, recycle=0.40) — those
    were picked to maximize val F1 which leaks val into model selection.
    """
    path = _commodity_cfg(commodity)["checkpoint_dir"] / "tuned_thresholds.json"
    if not path.exists():
        return 0.25, 0.50
    with open(path) as f:
        t = json.load(f)
    return (float(t.get("edge_threshold", 0.25)),
            t.get("recycle_threshold"))


_DATA: dict[str, dict] = {}


def _load_data(commodity: str = "cu"):
    """Load all cached data for the given commodity. No S3 calls."""
    cfg = _commodity_cfg(commodity)
    ckpt = cfg["checkpoint_dir"]
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for commodity={commodity!r}: {ckpt}. "
            f"Train it first."
        )

    with open(ckpt / "stage_vocab.json") as f:
        stage_vocab = json.load(f)
    with open(ckpt / "train_transition_counts.json") as f:
        tc = json.load(f)
    with open(ckpt / "train_joint_support.json") as f:
        js_raw = json.load(f)
    joint_support = {tuple(k.split("|")): v for k, v in js_raw.items()}

    grade_map = {}
    grade_map_path = cfg["grade_map_path"]
    if grade_map_path.exists():
        with open(grade_map_path) as f:
            grade_map = json.load(f)

    all_preds = {}
    split_by_file = {
        "val_predictions.json": "val",
        "oof_predictions.json": "oof",
        "test_predictions.json": "test",
    }
    for fname, split in split_by_file.items():
        path = ckpt / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        per_doc = data["per_doc"] if isinstance(data, dict) else data
        for entry in per_doc:
            all_preds[entry["doc_id"]] = {
                "predicted": entry.get("xgb_stages", []),
                "ground_truth": entry.get("ground_truth", []),
                "is_val": split == "val",
                "split": split,
            }

    doc_edges = {}
    edges_path = ckpt / "doc_edges.json"
    if edges_path.exists():
        with open(edges_path) as f:
            raw = json.load(f)
        doc_edges = {k: [tuple(e) for e in v] for k, v in raw.items()}

    return {
        "commodity": commodity,
        "stage_vocab": stage_vocab,
        "transition_counts": tc,
        "joint_support": joint_support,
        "grade_map": grade_map,
        "all_preds": all_preds,
        "doc_edges": doc_edges,
    }


def _corpus_stage_f1(data: dict) -> dict:
    """Macro stage F1 across val (and OOF) docs — each stage weighted equally.

    Mirrors the headline number stage_predictor reports at training-end:
    per-stage P/R/F1 over the val docs, then averaged across stages. Test
    predictions are excluded per holdout discipline.
    """
    def _macro(entries):
        if not entries:
            return None
        # Per-stage TP/FP/FN, then per-stage F1, then mean across stages.
        stage_tp, stage_fp, stage_fn = {}, {}, {}
        stages_seen = set()
        for entry in entries:
            gt = set(entry.get("ground_truth", []))
            pred = set(entry.get("predicted", []))
            stages_seen |= gt | pred
            for s in pred & gt:
                stage_tp[s] = stage_tp.get(s, 0) + 1
            for s in pred - gt:
                stage_fp[s] = stage_fp.get(s, 0) + 1
            for s in gt - pred:
                stage_fn[s] = stage_fn.get(s, 0) + 1
        f1s = []
        for s in stages_seen:
            tp = stage_tp.get(s, 0)
            fp = stage_fp.get(s, 0)
            fn = stage_fn.get(s, 0)
            if tp + fn == 0:  # stage absent from ground truth in this split
                continue
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
        return round(sum(f1s) / len(f1s), 3) if f1s else None

    val_entries, oof_entries = [], []
    for entry in data["all_preds"].values():
        if entry.get("split") == "test":
            continue
        if not entry.get("ground_truth"):
            continue
        (val_entries if entry.get("split") == "val" else oof_entries).append(entry)

    return {
        # Headline: val-set macro-F1 (matches the number stage_predictor prints).
        "stage_f1_mean": _macro(val_entries),
        "stage_f1_val_mean": _macro(val_entries),
        "stage_f1_oof_mean": _macro(oof_entries),
        "n_total": len(val_entries) + len(oof_entries),
        "n_val": len(val_entries),
        "n_oof": len(oof_entries),
    }


def _get_data(commodity: str = "cu"):
    if commodity not in _DATA:
        _DATA[commodity] = _load_data(commodity)
    return _DATA[commodity]


def _find_closest_doc(target_grade, data, val_window=None):
    """Find the closest doc by grade, preferring val set predictions.

    val_window controls how aggressively we prefer val docs over OOF/test:
    if a val doc is within val_window of the target grade we use it,
    otherwise we fall back to the closest overall. Defaults to 0.5 for Cu (%)
    and 1.0 for Au (g/t) since the Au grade map spans a much wider range.
    """
    if val_window is None:
        val_window = 0.5 if data.get("commodity", "cu") == "cu" else 1.0

    val_best, val_dist = None, float("inf")
    any_best, any_dist = None, float("inf")

    for doc_id, grade in data["grade_map"].items():
        if grade is None or doc_id not in data["all_preds"]:
            continue
        dist = abs(float(grade) - target_grade)
        if dist < any_dist:
            any_dist = dist
            any_best = doc_id
        if data["all_preds"][doc_id].get("is_val") and dist < val_dist:
            val_dist = dist
            val_best = doc_id

    if val_best and val_dist < val_window:
        best = val_best
    else:
        best = any_best

    return best, data["grade_map"].get(best) if best else None



def predict(grade: float, threshold: float | None = None,
            recycle_threshold: float | None = None,
            max_outgoing: int = 2,
            commodity: str = "cu") -> dict:
    """Predict a flowsheet for a given head grade and commodity.

    Looks up the closest doc by grade in the commodity's grade-map, uses its
    cached XGBoost stage predictions, wires edges with the corpus prior, caps
    outgoing edges per stage.

    Thresholds default to the OOF-tuned values from tuned_thresholds.json.
    Pass explicit floats to override.

    Returns a JSON-serializable dict.
    """
    from similarity import corpus_prior_scores, threshold_connections

    if threshold is None or recycle_threshold is None:
        tuned_edge, tuned_recycle = _load_tuned_thresholds(commodity)
        if threshold is None:
            threshold = tuned_edge
        if recycle_threshold is None:
            recycle_threshold = tuned_recycle

    data = _get_data(commodity)
    matched_id, matched_grade = _find_closest_doc(grade, data)
    if not matched_id:
        return {"error": "No matching doc found for this grade"}

    stages = list(data["all_preds"][matched_id]["predicted"])
    stage_set = set(stages)

    # Corpus prior edges
    scores = corpus_prior_scores(
        stage_set, data["transition_counts"], data["joint_support"],
    )
    pred_edge_set = threshold_connections(
        scores, threshold, recycle_threshold=recycle_threshold,
    )

    # Cap outgoing edges per stage
    outgoing = defaultdict(list)
    for src, dst in pred_edge_set:
        outgoing[src].append((dst, scores.get((src, dst), 0)))
    capped = set()
    for src, dsts in outgoing.items():
        top = sorted(dsts, key=lambda x: -x[1])[:max_outgoing]
        for dst, _ in top:
            capped.add((src, dst))
    pred_edge_set = capped

    edges = [{"src": s, "dst": d, "score": round(scores.get((s, d), 0), 3)}
             for s, d in sorted(pred_edge_set)]

    # GT comparison
    ground_truth = None
    metrics = None
    pred_entry = data["all_preds"][matched_id]
    gt_stages = pred_entry.get("ground_truth", [])
    gt_edge_list = data["doc_edges"].get(matched_id, [])
    gt_edge_set = set(gt_edge_list)

    if gt_stages and gt_edge_set:
        ground_truth = {
            "stages": sorted(gt_stages),
            "edges": [list(e) for e in sorted(gt_edge_set)],
        }
        gt_set = set(gt_stages)
        s_tp = len(stage_set & gt_set)
        s_fp = len(stage_set - gt_set)
        s_fn = len(gt_set - stage_set)
        s_prec = s_tp / max(s_tp + s_fp, 1)
        s_rec = s_tp / max(s_tp + s_fn, 1)
        s_f1 = 2 * s_prec * s_rec / max(s_prec + s_rec, 1e-9)

        e_tp = len(pred_edge_set & gt_edge_set)
        e_fp = len(pred_edge_set - gt_edge_set)
        e_fn = len(gt_edge_set - pred_edge_set)
        e_prec = e_tp / max(e_tp + e_fp, 1)
        e_rec = e_tp / max(e_tp + e_fn, 1)
        e_f1 = 2 * e_prec * e_rec / max(e_prec + e_rec, 1e-9)

        from eval import _reachability_f1
        reach = _reachability_f1(pred_edge_set, gt_edge_set, stage_set & gt_set)

        metrics = {
            "stage_f1": round(s_f1, 3),
            "edge_tp": e_tp, "edge_fp": e_fp, "edge_fn": e_fn,
            "edge_f1": round(e_f1, 3),
            "reach_f1": round(reach["f1"], 3),
            "reach_precision": round(reach["precision"], 3),
            "reach_recall": round(reach["recall"], 3),
        }

    cfg = _commodity_cfg(commodity)
    return {
        "commodity": commodity,
        "grade": grade,
        # Backward-compat: old web app referenced cu_grade. Keep populated
        # only for Cu so existing Cu-only consumers don't break.
        "cu_grade": grade if commodity == "cu" else None,
        "grade_unit": cfg["unit"],
        "grade_label": cfg["label"],
        "matched_doc_id": matched_id,
        "matched_grade": round(float(matched_grade), 3) if matched_grade else None,
        "stages": sorted(stages),
        "edges": edges,
        "ground_truth": ground_truth,
        "metrics": metrics,
        "corpus_metrics": _corpus_stage_f1(data),
    }


def visualize(prediction: dict, output_file: str = None) -> Path:
    """Render GT vs predicted flowsheet comparison PNG."""
    from pfs_visual import render_comparison

    grade = prediction.get("grade", prediction.get("cu_grade"))
    commodity = prediction.get("commodity", "cu")
    if output_file is None:
        output_file = f"pfs_predict_{commodity}_{grade:.2f}.png"
    output_path = Path(output_file)

    gt = prediction.get("ground_truth")
    metrics = prediction.get("metrics")

    pred_conn_strs = [f"{e['src']} -> {e['dst']}" for e in prediction["edges"]]
    gt_stages = gt["stages"] if gt else []
    gt_conn_strs = [f"{e[0]} -> {e[1]}" for e in gt["edges"]] if gt else []

    doc_entry = {
        "doc_id": prediction.get("matched_doc_id", f"grade_{grade:.2f}"),
        "ground_truth": gt_stages,
        "final_predicted": prediction["stages"],
        "ground_truth_connections": gt_conn_strs,
        "final_connections": pred_conn_strs,
        "correct": sorted(set(prediction["stages"]) & set(gt_stages)),
        "missed": sorted(set(gt_stages) - set(prediction["stages"])),
        "false_positive": sorted(set(prediction["stages"]) - set(gt_stages)),
        "doc_f1": metrics["stage_f1"] if metrics else 0.0,
    }

    label = prediction.get("grade_label", "Cu Grade")
    unit = prediction.get("grade_unit", "%")
    subtitle_parts = [f"{label}: {grade:.2f}{unit}"]
    if metrics:
        subtitle_parts.append(f"Stage F1: {metrics['stage_f1']:.3f}")
        subtitle_parts.append(f"Edge F1: {metrics['edge_f1']:.3f}")
        subtitle_parts.append(f"Reach F1: {metrics['reach_f1']:.3f}")

    render_comparison(doc_entry, output_file=output_path,
                      subtitle="  |  ".join(subtitle_parts))
    return output_path


def build_tables_cache(results_bucket: str = "mineral-pipeline-pipeline",
                       results_prefix: str = "results/",
                       max_rows_per_table: int = 20,
                       max_tables_per_doc: int = 80,
                       standalone_mirror: bool = True):
    """One-time: pull per-doc tables from pipeline results in S3.

    Output: xgb_checkpoint_v2/doc_tables.json keyed by document_id, restricted
    to docs that have V2 predictions. Each table keeps only the fields the LLM
    layer needs (caption, headers, rows, page, section_number). Rows are
    truncated at max_rows_per_table to bound file size.

    This cache powers the `--with-units` LLM equipment layer in classification/
    standalone/. Rebuild it when pipeline results change.
    """
    import re
    import shutil
    import boto3

    pred_doc_ids: set[str] = set()
    for fname in ["val_predictions.json", "oof_predictions.json",
                  "test_predictions.json"]:
        path = CHECKPOINT_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        per_doc = data["per_doc"] if isinstance(data, dict) else data
        for entry in per_doc:
            if entry.get("doc_id"):
                pred_doc_ids.add(entry["doc_id"])
    logger.info("Need tables for %d V2-covered docs", len(pred_doc_ids))

    s3 = boto3.client("s3")

    logger.info("Mapping document_id -> result key in s3://%s/%s ...",
                results_bucket, results_prefix)
    docid_to_key: dict[str, str] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=results_bucket, Prefix=results_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                resp = s3.get_object(Bucket=results_bucket, Key=key,
                                     Range="bytes=0-200")
                head = resp["Body"].read().decode("utf-8", errors="replace")
                m = re.search(r'"document_id"\s*:\s*"([^"]+)"', head)
                if m:
                    docid_to_key[m.group(1)] = key
            except Exception as e:
                logger.warning("head-read failed for %s: %s", key, e)
    logger.info("Mapped %d document_ids", len(docid_to_key))

    doc_tables: dict[str, list[dict]] = {}
    fetched = 0
    skipped_no_result = 0
    for doc_id in pred_doc_ids:
        key = docid_to_key.get(doc_id)
        if not key:
            skipped_no_result += 1
            continue
        try:
            resp = s3.get_object(Bucket=results_bucket, Key=key)
            doc = json.loads(resp["Body"].read())
        except Exception as e:
            logger.warning("load failed for %s: %s", doc_id[:20], e)
            skipped_no_result += 1
            continue

        raw_tables = doc.get("tables", []) or []
        trimmed: list[dict] = []
        for t in raw_tables[:max_tables_per_doc]:
            rows = (t.get("rows") or [])[:max_rows_per_table]
            trimmed.append({
                "caption": (t.get("caption") or "")[:500],
                "headers": [str(h)[:100] for h in (t.get("headers") or [])],
                "rows": [[str(c)[:100] for c in row] for row in rows],
                "page": t.get("page") or t.get("page_number"),
                "section_number": t.get("section_number"),
            })
        if trimmed:
            doc_tables[doc_id] = trimmed
        fetched += 1
        if fetched % 50 == 0:
            logger.info("  fetched %d / %d", fetched, len(pred_doc_ids))

    out_path = CHECKPOINT_DIR / "doc_tables.json"
    with open(out_path, "w") as f:
        json.dump(doc_tables, f)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Cached tables for %d docs (skipped %d no-result) -> %s (%.1f MB)",
                len(doc_tables), skipped_no_result, out_path, size_mb)

    if standalone_mirror:
        mirror = (CHECKPOINT_DIR.parent / "standalone" /
                  "xgb_checkpoint_v2" / "doc_tables.json")
        if mirror.parent.exists():
            shutil.copy2(out_path, mirror)
            logger.info("Mirrored to %s", mirror)


def build_cache(graphs_bucket: str, graphs_prefix: str = "graphs/",
                commodity: str = "cu"):
    """One-time: download graphs from S3, extract GT edges, save doc_edges.json
    into the commodity's checkpoint dir."""
    from stage_predictor import _load_graphs_from_s3, reduce_stage_vocab

    ckpt = _commodity_cfg(commodity)["checkpoint_dir"]
    graphs = _load_graphs_from_s3(graphs_bucket, graphs_prefix)
    with open(ckpt / "stage_vocab.json") as f:
        vocab_set = set(json.load(f))
    _, raw_to_canonical = reduce_stage_vocab(graphs, min_support=1, vocab_version="v1")

    doc_edges = {}
    for graph in graphs:
        doc_id = graph.get("document_id", "")
        edges = []
        for edge in graph.get("edges", []):
            if edge.get("type") != "stage_transition":
                continue
            src = raw_to_canonical.get(edge["source"].replace("stg_", ""))
            dst = raw_to_canonical.get(edge["target"].replace("stg_", ""))
            if src in vocab_set and dst in vocab_set:
                edges.append([src, dst])
        if edges:
            doc_edges[doc_id] = edges

    out_path = ckpt / "doc_edges.json"
    with open(out_path, "w") as f:
        json.dump(doc_edges, f, indent=2)
    logger.info("Cached GT edges for %d docs -> %s", len(doc_edges), out_path)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Flowsheet prediction + visualization")
    parser.add_argument("--commodity", type=str, default="cu",
                        choices=sorted(COMMODITIES),
                        help="Commodity to predict (cu or au); default cu")
    parser.add_argument("--grade", type=float, nargs="+",
                        help="Head grade(s) to predict (Cu %% or Au g/t)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Edge score cutoff; defaults to OOF-tuned value "
                             "from tuned_thresholds.json")
    parser.add_argument("--recycle-threshold", type=float, default=None,
                        help="Recycle-edge cutoff; defaults to OOF-tuned value")
    parser.add_argument("--max-outgoing", type=int, default=2)
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--build-tables-cache", action="store_true",
                        help="Pull per-doc tables from pipeline results in S3 "
                             "and write xgb_checkpoint_v2/doc_tables.json "
                             "(powers classification/standalone/ --with-units).")
    parser.add_argument("--graphs-bucket", default="sagemaker-us-east-1-666109694894")
    parser.add_argument("--graphs-prefix", default="graphs/")
    parser.add_argument("--results-bucket", default="mineral-pipeline-pipeline")
    parser.add_argument("--results-prefix", default="results/")
    args = parser.parse_args()

    if args.build_cache:
        build_cache(args.graphs_bucket, args.graphs_prefix,
                    commodity=args.commodity)
        return

    if args.build_tables_cache:
        build_tables_cache(results_bucket=args.results_bucket,
                           results_prefix=args.results_prefix)
        return

    if not args.grade:
        parser.error("--grade is required (or use --build-cache)")

    for grade in args.grade:
        result = predict(grade, threshold=args.threshold,
                         recycle_threshold=args.recycle_threshold,
                         max_outgoing=args.max_outgoing,
                         commodity=args.commodity)
        if "error" in result:
            print(f"Grade {grade}: {result['error']}")
            continue

        unit = result.get("grade_unit", "%")
        label = result.get("grade_label", "Cu Grade")
        print(f"\n{'='*60}")
        print(f"  Flowsheet — {label}: {grade:.2f}{unit}")
        if result["matched_grade"]:
            print(f"  Matched doc: {result['matched_grade']:.3f}{unit}")
        print(f"{'='*60}")
        print(f"\nStages ({len(result['stages'])}): {', '.join(result['stages'])}")
        print(f"\nEdges ({len(result['edges'])}):")
        for e in result["edges"]:
            print(f"  {e['src']:<20} -> {e['dst']:<20}  ({e['score']})")

        if result["metrics"]:
            m = result["metrics"]
            print(f"\nStage F1: {m['stage_f1']:.3f}  |  "
                  f"Edge F1: {m['edge_f1']:.3f}  |  "
                  f"Reach F1: {m['reach_f1']:.3f}")

        if not args.no_viz:
            out = args.output if len(args.grade) == 1 else None
            path = visualize(result, output_file=out)
            print(f"Saved: {path}")
        print()


if __name__ == "__main__":
    main()
