"""Derive train_transition_counts.json + train_joint_support.json for the
Au checkpoint from the artifacts stage_predictor already wrote.

stage_predictor saves transition_counts.json (corpus-level edge counts) but
not the train_-prefixed copies that predict.py + eval_gold + tune_thresholds
read. This script generates both, scoped to the labeled training docs only.

Inputs (from xgb_checkpoint_au/):
    - transition_counts.json    : dict "src|dst" -> int
    - oof_predictions.json      : list of {"doc_id", "ground_truth", ...}

Outputs (to xgb_checkpoint_au/):
    - train_transition_counts.json : straight copy of transition_counts.json
    - train_joint_support.json     : dict "src|dst" -> count of training docs
                                     containing BOTH stages (denominator for
                                     conditional edge probabilities)

Usage:
    python dev/gold/build_aux_caches.py
    python dev/gold/build_aux_caches.py --commodity cu  # also fix Cu if needed
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEV_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIRS = {
    "cu": DEV_ROOT / "classification" / "xgb_checkpoint_v2",
    "au": DEV_ROOT / "classification" / "xgb_checkpoint_au",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", default="au", choices=sorted(CHECKPOINT_DIRS))
    args = parser.parse_args()

    ckpt = CHECKPOINT_DIRS[args.commodity]
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    # 1. train_transition_counts: copy transition_counts.json verbatim.
    src = ckpt / "transition_counts.json"
    dst = ckpt / "train_transition_counts.json"
    if not src.exists():
        raise SystemExit(f"Missing {src}")
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {dst}")

    # 2. train_joint_support: pairwise stage co-occurrence over labeled training docs.
    oof_path = ckpt / "oof_predictions.json"
    if not oof_path.exists():
        raise SystemExit(f"Missing {oof_path}")
    with open(oof_path) as f:
        oof = json.load(f)
    per_doc = oof["per_doc"] if isinstance(oof, dict) else oof

    joint = defaultdict(int)
    for entry in per_doc:
        gt = list(set(entry.get("ground_truth", [])))
        for i, src_stage in enumerate(gt):
            for dst_stage in gt:
                # Include self-loops (used for recycle edges) and all ordered pairs.
                joint[f"{src_stage}|{dst_stage}"] += 1

    out_path = ckpt / "train_joint_support.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(joint), f, indent=2)
    print(f"Wrote {out_path} ({len(joint)} pairs from {len(per_doc)} labeled docs)")


if __name__ == "__main__":
    main()
