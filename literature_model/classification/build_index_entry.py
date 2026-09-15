"""SageMaker entry script: build inference index from trained GAT checkpoint.

Loads the saved GAT model, reads all graphs from S3, runs one forward pass
to get embeddings + stage probabilities, and saves inference_index.json.

Usage (SageMaker entry script):
    Channels: model (GAT model.tar.gz from train-gat job)
    Hyperparameters: graphs-bucket, graphs-prefix, threshold
"""

import argparse
import glob
import json
import logging
import os
import tarfile
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-bucket", type=str, required=True)
    parser.add_argument("--graphs-prefix", type=str, default="graphs/")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model_dir = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")

    # Extract model artifact if tar.gz
    model_path = Path(model_dir)
    tar_files = list(model_path.glob("*.tar.gz"))
    if tar_files:
        extract_dir = model_path / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting {tar_files[0]} ...")
        with tarfile.open(tar_files[0], "r:gz") as tar:
            tar.extractall(extract_dir)
        model_path = extract_dir

    # Find the best model checkpoint
    pt_files = list(model_path.rglob("model_fold_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No model_fold_*.pt found in {model_dir}")

    # Load stage_vocab to determine model dimensions
    vocab_files = list(model_path.rglob("stage_vocab.json"))
    if not vocab_files:
        raise FileNotFoundError(f"No stage_vocab.json found in {model_dir}")

    with open(vocab_files[0]) as f:
        stage_vocab = json.load(f)
    num_stages = len(stage_vocab)
    logger.info(f"Stage vocab: {num_stages} stages")

    # Load metrics to find best fold
    metrics_files = list(model_path.rglob("metrics.json"))
    best_fold = 0
    if metrics_files:
        with open(metrics_files[0]) as f:
            metrics = json.load(f)
        fold_results = metrics.get("per_fold", [])
        if fold_results:
            best_fold = min(range(len(fold_results)),
                            key=lambda i: fold_results[i].get("val_loss", float("inf")))
            saved_threshold = fold_results[best_fold].get("threshold", args.threshold)
            if saved_threshold:
                args.threshold = saved_threshold
            logger.info(f"Best fold: {best_fold}, threshold: {args.threshold}")

    # Find the checkpoint file
    best_pt = None
    for pt in pt_files:
        if f"fold_{best_fold}" in pt.name:
            best_pt = pt
            break
    if best_pt is None:
        best_pt = pt_files[0]
    logger.info(f"Loading model from {best_pt}")

    # Import model classes (flowsheet_predictor.py is in the same source dir)
    from flowsheet_predictor import (
        FlowsheetPredictor, load_training_data, build_inference_index
    )

    # Load graph data from S3
    logger.info(f"Loading data from s3://{args.graphs_bucket}/{args.graphs_prefix}")
    t0 = time.time()
    data = load_training_data(args.graphs_bucket, args.graphs_prefix)
    logger.info(f"Data loaded in {time.time() - t0:.0f}s")

    # Reconstruct model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = data["feat_dim"]
    model = FlowsheetPredictor(feat_dim, num_stages).to(device)
    state_dict = torch.load(best_pt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    logger.info(f"Model loaded on {device}, feat_dim={feat_dim}, stages={num_stages}")

    # Build inference index
    build_inference_index(model, data, args.threshold, device, output_dir)

    # Also copy metrics if available
    if metrics_files:
        import shutil
        shutil.copy2(metrics_files[0], os.path.join(output_dir, "metrics.json"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
