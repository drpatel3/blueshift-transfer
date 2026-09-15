"""Train the gold-corpus XGBoost stage predictor.

Thin wrapper around classification/stage_predictor.main(). Sets SM_MODEL_DIR
to dev/classification/xgb_checkpoint_au/ so per-stage models, OOF preds, and
threshold artifacts land alongside the existing xgb_checkpoint_v2/ (Cu).

Prerequisites:
    1. dev/gold/build_doc_id_allowlist.py      → gold_training_doc_ids.json
    2. dev/gold/build_au_graphs.py --execute   → uploads allowlist
    3. SageMaker graph-pipeline job with SM_HP_CHECKPOINT_PREFIX=graphs_au/
       (writes per-doc graph JSONs to s3://mineral-pipeline-pipeline/graphs_au/)

Usage:
    python dev/gold/train_au_model.py
    python dev/gold/train_au_model.py --val-pct 0.20 --test-pct 0.10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEV_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = DEV_ROOT / "classification" / "xgb_checkpoint_au"

if str(DEV_ROOT / "classification") not in sys.path:
    sys.path.insert(0, str(DEV_ROOT / "classification"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-bucket", default="mineral-pipeline-pipeline")
    parser.add_argument("--graphs-prefix", default="graphs_au/")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--val-pct", type=float, default=0.20)
    parser.add_argument("--test-pct", type=float, default=0.10)
    parser.add_argument("--vocab-version", default="v1")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["SM_MODEL_DIR"] = str(CHECKPOINT_DIR)

    # Re-build sys.argv for stage_predictor's argparse.
    sys.argv = [
        "stage_predictor.py",
        "--graphs-bucket", args.graphs_bucket,
        "--graphs-prefix", args.graphs_prefix,
        "--folds", str(args.folds),
        "--min-support", str(args.min_support),
        "--val-pct", str(args.val_pct),
        "--test-pct", str(args.test_pct),
        "--vocab-version", args.vocab_version,
    ]

    import stage_predictor
    stage_predictor.main()


if __name__ == "__main__":
    main()
