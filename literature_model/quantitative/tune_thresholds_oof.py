"""Tune edge + recycle thresholds on OOF, not val.

The V2 checkpoint's thresholds were picked to maximize val F1, which
leaks the val set into model selection. This script re-selects the
thresholds against OOF predictions (training docs scored by a fold
that didn't see them) and reports val F1 at the picked thresholds as
an unbiased final read.

Writes classification/xgb_checkpoint_v2/tuned_thresholds.json:
    {
      "edge_threshold": 0.25,
      "recycle_threshold": 0.50,
      "oof_reach_f1": 0.612,
      "val_reach_f1":  0.774,
      "n_oof": 126,
      "n_val": 36
    }

Usage:
    python quantitative/tune_thresholds_oof.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEV_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = DEV_ROOT / "classification" / "xgb_checkpoint_v2"
OUT_PATH = CHECKPOINT_DIR / "tuned_thresholds.json"

for _p in [str(DEV_ROOT / "classification"), str(DEV_ROOT / "quantitative")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

EDGE_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
RECYCLE_THRESHOLDS = [None, 0.40, 0.50, 0.60]


def _load(fname):
    path = CHECKPOINT_DIR / fname
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data["per_doc"] if isinstance(data, dict) else data


def _score_split(entries, doc_edges, transition_counts, joint_support,
                 edge_threshold, recycle_threshold):
    """Compute mean reach_f1 + mean edge_f1 on the given split."""
    from similarity import corpus_prior_scores, threshold_connections
    from eval import evaluate_connections, evaluate_ordering

    pred_conn, true_conn, pred_stages, true_stages = [], [], [], []
    for entry in entries:
        doc_id = entry["doc_id"]
        if doc_id not in doc_edges:
            continue
        predicted = set(entry.get("xgb_stages", []))
        gt = set(entry.get("ground_truth", []))
        if not gt:
            continue

        scores = corpus_prior_scores(predicted, transition_counts, joint_support)
        edges = threshold_connections(scores, edge_threshold,
                                      recycle_threshold=recycle_threshold)

        pred_conn.append(edges)
        true_conn.append(doc_edges[doc_id])
        pred_stages.append(predicted)
        true_stages.append(gt)

    if not pred_conn:
        return None
    stage_vocab = sorted({s for ss in true_stages for s in ss}
                          | {s for ss in pred_stages for s in ss})
    conn_m = evaluate_connections(pred_conn, true_conn, stage_vocab)
    ord_m = evaluate_ordering(pred_conn, true_conn, pred_stages, true_stages)
    return {
        "n": len(pred_conn),
        "edge_f1": conn_m["mean_f1"],
        "reach_f1": ord_m["reach_f1"],
        "reach_precision": ord_m["reach_precision"],
        "reach_recall": ord_m["reach_recall"],
    }


def main():
    # Load OOF + val predictions and shared corpus stats.
    oof = _load("oof_predictions.json") or []
    val = _load("val_predictions.json") or []
    if not oof:
        logger.error("No OOF predictions found — cannot tune without them")
        return

    with open(CHECKPOINT_DIR / "doc_edges.json") as f:
        raw = json.load(f)
    doc_edges = {k: {tuple(e) for e in v} for k, v in raw.items()}

    with open(CHECKPOINT_DIR / "train_transition_counts.json") as f:
        transition_counts = json.load(f)
    with open(CHECKPOINT_DIR / "train_joint_support.json") as f:
        joint_support = {tuple(k.split("|")): v for k, v in json.load(f).items()}

    print(f"\nSweep on OOF ({len(oof)} docs), rank by reach_f1:\n")
    print(f"{'edge':>5} {'recyc':>6}  "
          f"{'oof_reach':>9} {'oof_prec':>8} {'oof_rec':>7} {'oof_edge':>8}  "
          f"{'val_reach':>9} {'gap':>6}")
    print("-" * 80)

    rows = []
    for et in EDGE_THRESHOLDS:
        for rt in RECYCLE_THRESHOLDS:
            oof_m = _score_split(oof, doc_edges, transition_counts,
                                 joint_support, et, rt)
            val_m = _score_split(val, doc_edges, transition_counts,
                                 joint_support, et, rt)
            if oof_m is None:
                continue
            rows.append({
                "edge_threshold": et,
                "recycle_threshold": rt,
                "oof": oof_m,
                "val": val_m,
            })
            rt_str = "—" if rt is None else f"{rt:.2f}"
            val_reach = val_m["reach_f1"] if val_m else float("nan")
            gap = (val_reach - oof_m["reach_f1"]) if val_m else float("nan")
            print(f"{et:>5.2f} {rt_str:>6}  "
                  f"{oof_m['reach_f1']:>9.3f} {oof_m['reach_precision']:>8.3f} "
                  f"{oof_m['reach_recall']:>7.3f} {oof_m['edge_f1']:>8.3f}  "
                  f"{val_reach:>9.3f} {gap:>+6.3f}")

    # Rank strictly by OOF reach_f1. No val-based tiebreakers — that would
    # reintroduce the exact leak we're trying to fix.
    rows.sort(key=lambda r: -r["oof"]["reach_f1"])
    best = rows[0]

    payload = {
        "edge_threshold": best["edge_threshold"],
        "recycle_threshold": best["recycle_threshold"],
        "oof_reach_f1": round(best["oof"]["reach_f1"], 4),
        "oof_edge_f1": round(best["oof"]["edge_f1"], 4),
        "val_reach_f1": round(best["val"]["reach_f1"], 4) if best["val"] else None,
        "val_edge_f1": round(best["val"]["edge_f1"], 4) if best["val"] else None,
        "n_oof": best["oof"]["n"],
        "n_val": best["val"]["n"] if best["val"] else 0,
        "method": "ranked by OOF reach_f1 (training docs scored by fold that "
                  "didn't see them); val reported as unbiased readout only",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    rt_str = "—" if best["recycle_threshold"] is None else f"{best['recycle_threshold']:.2f}"
    print(f"\nBest on OOF: edge={best['edge_threshold']:.2f}, recycle={rt_str}")
    print(f"  OOF reach_f1 = {best['oof']['reach_f1']:.3f}  "
          f"(n={best['oof']['n']})")
    if best["val"]:
        gap = best["val"]["reach_f1"] - best["oof"]["reach_f1"]
        print(f"  Val reach_f1 = {best['val']['reach_f1']:.3f}  "
              f"(n={best['val']['n']})   gap = {gap:+.3f}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
