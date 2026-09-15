"""Evaluate the flowsheet model on Au-in-range docs.

Mirrors classification/predict.py scoring (stage_f1, edge_f1, reach_f1)
but runs it across every doc that (a) sits in the Au head-grade window
from grade_map_au.json, and (b) has a cached stage prediction.

Test-set docs are excluded (val + OOF only) per holdout rules.

Usage:
    python quantitative/eval_gold.py                              # gold-trained
    python quantitative/eval_gold.py --checkpoint xgb_checkpoint_v2  # Cu baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEV_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = "xgb_checkpoint_au"
AU_MAP_PATH = Path(__file__).parent / "grade_map_au.json"
CU_MAP_PATH = Path(__file__).parent / "grade_map.json"

for _p in [str(DEV_ROOT / "classification"), str(DEV_ROOT / "quantitative")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_predictions(checkpoint_dir: Path) -> dict:
    """Load val + OOF predictions. Excludes test per holdout rule."""
    preds = {}
    for fname, split in [("val_predictions.json", "val"),
                         ("oof_predictions.json", "oof")]:
        path = checkpoint_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        per_doc = data["per_doc"] if isinstance(data, dict) else data
        for entry in per_doc:
            preds[entry["doc_id"]] = {
                "split": split,
                "predicted": set(entry.get("xgb_stages", [])),
                "ground_truth": set(entry.get("ground_truth", [])),
            }
    return preds


def _load_doc_edges(checkpoint_dir: Path) -> dict:
    with open(checkpoint_dir / "doc_edges.json") as f:
        raw = json.load(f)
    return {k: {tuple(e) for e in v} for k, v in raw.items()}


def _score_doc(pred_stages, gt_stages, pred_edges, gt_edges):
    """Compute stage_f1, edge_f1, reach_f1 for one doc."""
    from eval import _reachability_f1

    s_tp = len(pred_stages & gt_stages)
    s_fp = len(pred_stages - gt_stages)
    s_fn = len(gt_stages - pred_stages)
    s_prec = s_tp / max(s_tp + s_fp, 1)
    s_rec = s_tp / max(s_tp + s_fn, 1)
    s_f1 = 2 * s_prec * s_rec / max(s_prec + s_rec, 1e-9)

    e_tp = len(pred_edges & gt_edges)
    e_fp = len(pred_edges - gt_edges)
    e_fn = len(gt_edges - pred_edges)
    e_prec = e_tp / max(e_tp + e_fp, 1)
    e_rec = e_tp / max(e_tp + e_fn, 1)
    e_f1 = 2 * e_prec * e_rec / max(e_prec + e_rec, 1e-9)

    reach = _reachability_f1(pred_edges, gt_edges, pred_stages & gt_stages)

    return {
        "stage_f1": s_f1, "edge_f1": e_f1,
        "reach_f1": reach["f1"],
        "reach_prec": reach["precision"],
        "reach_rec": reach["recall"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Checkpoint directory under classification/ (default: xgb_checkpoint_au)"
    )
    args = parser.parse_args()

    checkpoint_dir = DEV_ROOT / "classification" / args.checkpoint
    tuned_thresholds_path = checkpoint_dir / "tuned_thresholds.json"
    if not checkpoint_dir.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_dir}")
    logger.info(f"Checkpoint: {checkpoint_dir}")

    from similarity import corpus_prior_scores, threshold_connections

    with open(AU_MAP_PATH) as f:
        au_map = json.load(f)
    cu_map = {}
    if CU_MAP_PATH.exists():
        with open(CU_MAP_PATH) as f:
            cu_map = {k: v for k, v in json.load(f).items()
                      if v is not None and v > 0}

    preds = _load_predictions(checkpoint_dir)
    doc_edges = _load_doc_edges(checkpoint_dir)

    with open(checkpoint_dir / "train_transition_counts.json") as f:
        transition_counts = json.load(f)
    with open(checkpoint_dir / "train_joint_support.json") as f:
        js_raw = json.load(f)
    joint_support = {tuple(k.split("|")): v for k, v in js_raw.items()}

    # Use OOF-tuned thresholds if available. Falling back to unbiased midpoints
    # rather than the old val-tuned (0.30 / 0.40) so the eval isn't reinjecting
    # val-leak selection.
    edge_threshold, recycle_threshold = 0.25, 0.50
    if tuned_thresholds_path.exists():
        with open(tuned_thresholds_path) as f:
            t = json.load(f)
        edge_threshold = float(t["edge_threshold"])
        recycle_threshold = t.get("recycle_threshold")
        logger.info(
            f"Using OOF-tuned thresholds: edge={edge_threshold}, "
            f"recycle={recycle_threshold}"
        )
    else:
        logger.info(
            f"No tuned_thresholds.json — using unbiased defaults "
            f"edge={edge_threshold}, recycle={recycle_threshold}"
        )

    # Intersect: doc must have Au in range AND cached V2 prediction AND GT edges.
    # Mineral-classification (Cu vs Au primary) is unreliable, so we treat every
    # doc with Au in 0.3–30 g/t as an evaluable gold PFS regardless of labeling.
    evaluable = [d for d in au_map if d in preds and d in doc_edges]
    if not evaluable:
        logger.error("No evaluable docs — check that grade_map_au.json and V2 checkpoints align")
        return

    logger.info(
        f"Evaluable Au-in-range docs: {len(evaluable)}  "
        f"(val={sum(1 for d in evaluable if preds[d]['split'] == 'val')}, "
        f"oof={sum(1 for d in evaluable if preds[d]['split'] == 'oof')})"
    )

    rows = []
    for doc_id in evaluable:
        pred_stages = preds[doc_id]["predicted"]
        gt_stages = preds[doc_id]["ground_truth"]
        if not gt_stages:
            continue

        scores = corpus_prior_scores(pred_stages, transition_counts, joint_support)
        pred_edges = threshold_connections(scores, edge_threshold,
                                           recycle_threshold=recycle_threshold)
        gt_edges = doc_edges[doc_id]

        m = _score_doc(pred_stages, gt_stages, pred_edges, gt_edges)
        m["doc_id"] = doc_id
        m["au_grade"] = au_map[doc_id]
        m["cu_grade"] = cu_map.get(doc_id)
        m["split"] = preds[doc_id]["split"]
        rows.append(m)

    def _agg(subset, label):
        if not subset:
            print(f"  {label:<20} no docs")
            return
        print(
            f"  {label:<20} n={len(subset):>3}  "
            f"stage_F1={mean(r['stage_f1'] for r in subset):.3f}  "
            f"edge_F1={mean(r['edge_f1'] for r in subset):.3f}  "
            f"reach_F1={mean(r['reach_f1'] for r in subset):.3f}  "
            f"(prec={mean(r['reach_prec'] for r in subset):.3f} / "
            f"rec={mean(r['reach_rec'] for r in subset):.3f})"
        )

    print(f"\n{'=' * 78}")
    print("  Gold PFS Evaluation — V2 stage preds + corpus-prior edges")
    print(f"{'=' * 78}\n")
    _agg(rows, "All Au-in-range")
    _agg([r for r in rows if r["split"] == "val"], "val split only")
    _agg([r for r in rows if r["split"] == "oof"], "oof split only")

    # Au-grade bands
    print()
    bands = [(0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 30.0)]
    for lo, hi in bands:
        subset = [r for r in rows if lo <= r["au_grade"] < hi]
        _agg(subset, f"Au {lo:.1f}-{hi:.1f} g/t")

    # Worst 5 for sanity check
    print("\nWorst 5 docs by reach_F1:")
    for r in sorted(rows, key=lambda x: x["reach_f1"])[:5]:
        cu_str = "--" if r["cu_grade"] is None else f"{r['cu_grade']:.2f}"
        print(
            f"  {r['doc_id'][:16]}...  Au={r['au_grade']:.2f}  Cu={cu_str}  "
            f"stage={r['stage_f1']:.2f}  edge={r['edge_f1']:.2f}  reach={r['reach_f1']:.2f}"
        )


if __name__ == "__main__":
    main()
