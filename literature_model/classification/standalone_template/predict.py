"""Standalone flowsheet prediction — V2 XGBoost stages + corpus prior edges.

Self-contained bundle entry point. All imports are sibling-local; no repo-wide
dependencies. V2 model val macro-F1 = 0.86.

Usage:
    python predict.py --grade 1.25
    python predict.py --grade 0.50 1.00 1.25 1.50
    python predict.py --grade 1.25 --with-units        # adds LLM equipment layer
    python predict.py --grade 1.25 --no-viz
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

BUNDLE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BUNDLE_DIR / "xgb_checkpoint_v2"
GRADE_MAP_PATH = BUNDLE_DIR / "grade_map.json"


_DATA = None


def _load_data():
    """Load all cached data. No network calls."""
    with open(CHECKPOINT_DIR / "stage_vocab.json") as f:
        stage_vocab = json.load(f)
    with open(CHECKPOINT_DIR / "train_transition_counts.json") as f:
        tc = json.load(f)
    with open(CHECKPOINT_DIR / "train_joint_support.json") as f:
        js_raw = json.load(f)
    joint_support = {tuple(k.split("|")): v for k, v in js_raw.items()}

    grade_map = {}
    if GRADE_MAP_PATH.exists():
        with open(GRADE_MAP_PATH) as f:
            grade_map = json.load(f)

    all_preds = {}
    for fname in ["val_predictions.json", "oof_predictions.json",
                  "test_predictions.json"]:
        path = CHECKPOINT_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        per_doc = data["per_doc"] if isinstance(data, dict) else data
        is_val = "val_predictions" in fname
        for entry in per_doc:
            all_preds[entry["doc_id"]] = {
                "predicted": entry.get("xgb_stages", []),
                "ground_truth": entry.get("ground_truth", []),
                "is_val": is_val,
            }

    doc_edges = {}
    edges_path = CHECKPOINT_DIR / "doc_edges.json"
    if edges_path.exists():
        with open(edges_path) as f:
            raw = json.load(f)
        doc_edges = {k: [tuple(e) for e in v] for k, v in raw.items()}

    doc_tables = {}
    tables_path = CHECKPOINT_DIR / "doc_tables.json"
    if tables_path.exists():
        with open(tables_path) as f:
            doc_tables = json.load(f)

    return {
        "stage_vocab": stage_vocab,
        "transition_counts": tc,
        "joint_support": joint_support,
        "grade_map": grade_map,
        "all_preds": all_preds,
        "doc_edges": doc_edges,
        "doc_tables": doc_tables,
    }


def _get_data():
    global _DATA
    if _DATA is None:
        _DATA = _load_data()
    return _DATA


def _find_closest_doc(cu_grade, data, k: int = 1):
    """Find the k closest docs by grade, preferring val set predictions.

    The V2 model's 0.86 macro-F1 is measured on the val set. OOF predictions
    are weaker, so we prefer val docs when one exists within 0.5% of the target.

    Returns a list of (doc_id, grade) tuples of length <= k, sorted by distance.
    When k == 1, returns a single (doc_id, grade) tuple (back-compat).
    """
    scored = []
    for doc_id, grade in data["grade_map"].items():
        if grade is None or doc_id not in data["all_preds"]:
            continue
        dist = abs(float(grade) - cu_grade)
        is_val = bool(data["all_preds"][doc_id].get("is_val"))
        scored.append((dist, is_val, doc_id, float(grade)))

    if not scored:
        if k == 1:
            return None, None
        return []

    val_within = [s for s in scored if s[1] and s[0] < 0.5]
    val_within.sort(key=lambda s: s[0])

    any_sorted = sorted(scored, key=lambda s: s[0])

    chosen = []
    seen = set()
    for s in val_within + any_sorted:
        if s[2] in seen:
            continue
        chosen.append((s[2], s[3]))
        seen.add(s[2])
        if len(chosen) >= k:
            break

    if k == 1:
        return chosen[0] if chosen else (None, None)
    return chosen


def predict(cu_grade: float, threshold: float = 0.30,
            recycle_threshold: float = 0.40,
            max_outgoing: int = 2) -> dict:
    """Predict a flowsheet for a given Cu head grade.

    XGBoost V2 produces the stage list (authoritative — macro-F1=0.86).
    The corpus prior wires edges between those stages. Caps outgoing
    edges per stage.
    """
    from similarity import corpus_prior_scores, threshold_connections

    data = _get_data()
    matched_id, matched_grade = _find_closest_doc(cu_grade, data, k=1)
    if not matched_id:
        return {"error": "No matching doc found for this grade"}

    stages = list(data["all_preds"][matched_id]["predicted"])
    stage_set = set(stages)

    scores = corpus_prior_scores(
        stage_set, data["transition_counts"], data["joint_support"],
    )
    pred_edge_set = threshold_connections(
        scores, threshold, recycle_threshold=recycle_threshold,
    )

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

    return {
        "cu_grade": cu_grade,
        "matched_doc_id": matched_id,
        "matched_grade": round(float(matched_grade), 3) if matched_grade else None,
        "stages": sorted(stages),
        "edges": edges,
        "ground_truth": ground_truth,
        "metrics": metrics,
    }


def annotate_with_units(prediction: dict, k: int = 5,
                        llm_model: str | None = None) -> dict:
    """Add LLM-inferred equipment {type, count} per stage.

    INVARIANT: The stage list and edge list are never mutated. The LLM
    sees one stage at a time and may only return unit annotations.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(BUNDLE_DIR / ".env")
    except ImportError:
        pass

    from llm_units import predict_units

    data = _get_data()
    return predict_units(prediction, data, k=k, llm_model=llm_model)


def visualize(prediction: dict, output_file: str = None) -> Path:
    """Render GT vs predicted flowsheet comparison PNG."""
    from pfs_visual import render_comparison

    cu = prediction["cu_grade"]
    if output_file is None:
        output_file = f"pfs_predict_{cu:.2f}.png"
    output_path = Path(output_file)

    gt = prediction.get("ground_truth")
    metrics = prediction.get("metrics")

    pred_conn_strs = [f"{e['src']} -> {e['dst']}" for e in prediction["edges"]]
    gt_stages = gt["stages"] if gt else []
    gt_conn_strs = [f"{e[0]} -> {e[1]}" for e in gt["edges"]] if gt else []

    doc_entry = {
        "doc_id": prediction.get("matched_doc_id", f"grade_{cu:.2f}"),
        "ground_truth": gt_stages,
        "final_predicted": prediction["stages"],
        "ground_truth_connections": gt_conn_strs,
        "final_connections": pred_conn_strs,
        "correct": sorted(set(prediction["stages"]) & set(gt_stages)),
        "missed": sorted(set(gt_stages) - set(prediction["stages"])),
        "false_positive": sorted(set(prediction["stages"]) - set(gt_stages)),
        "doc_f1": metrics["stage_f1"] if metrics else 0.0,
    }

    subtitle_parts = [f"Cu Grade: {cu:.2f}%"]
    if metrics:
        subtitle_parts.append(f"Edge F1: {metrics['edge_f1']:.3f}")
        subtitle_parts.append(f"Reach F1: {metrics['reach_f1']:.3f}")
        subtitle_parts.append(f"Stage F1: {metrics['stage_f1']:.3f}")

    render_comparison(doc_entry, output_file=output_path,
                      subtitle="  |  ".join(subtitle_parts))
    return output_path


def _print_result(result: dict) -> None:
    grade = result["cu_grade"]
    print(f"\n{'='*60}")
    print(f"  Flowsheet — Cu Grade: {grade:.2f}%")
    if result["matched_grade"]:
        print(f"  Matched doc: {result['matched_grade']:.3f}%")
    print(f"{'='*60}")

    stage_entries = result["stages"]
    if stage_entries and isinstance(stage_entries[0], dict):
        print(f"\nStages ({len(stage_entries)}):")
        for s in stage_entries:
            units = s.get("units", [])
            if units:
                unit_str = ", ".join(
                    f"{u['count']}× {u['type']}" for u in units
                )
                print(f"  {s['name']:<22}  {unit_str}")
            else:
                print(f"  {s['name']}")
    else:
        print(f"\nStages ({len(stage_entries)}): {', '.join(stage_entries)}")

    print(f"\nEdges ({len(result['edges'])}):")
    for e in result["edges"]:
        print(f"  {e['src']:<22} -> {e['dst']:<22}  ({e['score']})")

    if result.get("metrics"):
        m = result["metrics"]
        print(f"\nStage F1: {m['stage_f1']:.3f}  |  "
              f"Edge F1: {m['edge_f1']:.3f}  |  "
              f"Reach F1: {m['reach_f1']:.3f}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Standalone V2 flowsheet predictor")
    parser.add_argument("--grade", type=float, nargs="+", required=True,
                        help="Cu head grade(s) to predict")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--recycle-threshold", type=float, default=0.40)
    parser.add_argument("--max-outgoing", type=int, default=2)
    parser.add_argument("--with-units", action="store_true",
                        help="Add LLM-inferred equipment (type, count) per stage")
    parser.add_argument("--units-k", type=int, default=5,
                        help="KNN neighbors for unit annotation (default 5)")
    parser.add_argument("--llm-model", default=None,
                        help="Override the Bedrock model id "
                             "(default: us.anthropic.claude-sonnet-4-5-20250929-v1:0).")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--json", default=None,
                        help="Also dump full prediction JSON to this path")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    for grade in args.grade:
        result = predict(grade, threshold=args.threshold,
                         recycle_threshold=args.recycle_threshold,
                         max_outgoing=args.max_outgoing)
        if "error" in result:
            print(f"Grade {grade}: {result['error']}")
            continue

        if args.with_units:
            result = annotate_with_units(
                result, k=args.units_k, llm_model=args.llm_model)

        _print_result(result)

        if args.json:
            json_path = Path(args.json)
            if len(args.grade) > 1:
                json_path = json_path.with_name(
                    f"{json_path.stem}_{grade:.2f}{json_path.suffix}")
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"JSON: {json_path}")

        if not args.no_viz:
            out = args.output if len(args.grade) == 1 else None
            path = visualize(result, output_file=out)
            print(f"Saved: {path}")
        print()


if __name__ == "__main__":
    main()
