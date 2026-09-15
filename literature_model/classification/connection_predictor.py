"""Connection predictor: given predicted V2 stages, predict which pairs are connected.

Single XGBoost model that takes (doc_features, src_stage, dst_stage) and predicts
whether src→dst is a direct connection. Pools all edge examples together to solve
the sparsity problem that per-edge classifiers faced.

Usage (SageMaker entry script):
    Hyperparameters: graphs-bucket, graphs-prefix, folds, test-pct, val-pct
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from stage_predictor import (
    flatten_document, _extract_tfidf_vectors, fit_tfidf_svd,
    transform_tfidf_svd, build_svd_feature_names,
    _load_graphs_from_s3, _load_cross_doc_edges, _compute_neighbor_features,
    build_feature_names, V2_STAGE_VOCAB, map_stage_v2, _map_stage,
    reduce_stage_vocab,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Process order for stage position features (covers both V1 and V2 names)
STAGE_ORDER = {s: i for i, s in enumerate([
    "input", "stockpile", "feeder", "crusher", "primary_crusher", "screen",
    "secondary_crusher", "hpgr", "mill", "sag_mill", "ball_mill", "cyclone",
    "gravity", "gravity_concentration", "flotation", "rougher_flotation",
    "regrind", "cleaner_flotation", "magnetic_separation", "agglomeration",
    "leach", "adsorption", "elution", "carbon_regeneration",
    "solvent_extraction", "solution_recovery", "electrowinning",
    "precipitation", "product_recovery", "ion_exchange", "ccd", "kiln",
    "smelting", "thickener", "concentrate_thickener", "tailings_thickener",
    "filter", "bin", "conveyor", "tank", "solution_pond",
    "concentrate_product", "water_treatment", "tailing", "ore_sorting",
])}

# Default to V2; overridden at runtime based on --vocab-version
N_STAGES = len(V2_STAGE_VOCAB)
stage_to_idx = {s: i for i, s in enumerate(sorted(V2_STAGE_VOCAB))}


def build_connection_features(doc_vec: np.ndarray, src: str, dst: str,
                              corpus_freq: float = 0.0) -> np.ndarray:
    """Build feature vector for a (doc, src, dst) connection candidate.

    Features:
    - doc_vec: full document feature vector (D dims)
    - src one-hot: (N_STAGES dims)
    - dst one-hot: (N_STAGES dims)
    - src process order position (1 dim, normalized 0-1)
    - dst process order position (1 dim)
    - order difference dst - src (1 dim, negative = recycle loop)
    - corpus frequency of this edge (1 dim)
    """
    src_oh = np.zeros(N_STAGES, dtype=np.float32)
    dst_oh = np.zeros(N_STAGES, dtype=np.float32)
    if src in stage_to_idx:
        src_oh[stage_to_idx[src]] = 1.0
    if dst in stage_to_idx:
        dst_oh[stage_to_idx[dst]] = 1.0

    max_order = max(STAGE_ORDER.values()) or 1
    src_pos = STAGE_ORDER.get(src, max_order / 2) / max_order
    dst_pos = STAGE_ORDER.get(dst, max_order / 2) / max_order
    order_diff = dst_pos - src_pos

    pair_feats = np.array([src_pos, dst_pos, order_diff, corpus_freq],
                          dtype=np.float32)

    return np.concatenate([doc_vec, src_oh, dst_oh, pair_feats])


def load_connection_data(graphs_bucket: str, graphs_prefix: str,
                         vocab_version: str = "v2"):
    """Load graphs and build connection training data.

    For each labeled doc:
    - Map stages to V2
    - Generate all ordered pairs of V2 stages
    - Label: 1 if pair is a ground truth edge, 0 otherwise

    Returns:
        X_pairs: (N_pairs, D+2*S+4) feature matrix
        y_pairs: (N_pairs,) binary labels
        doc_indices: which doc each pair belongs to
        pair_names: list of "src|dst" for each pair
        doc_vecs: (N_docs, D) raw doc feature vectors
        doc_ids: list of doc IDs
        corpus_edge_freq: dict of edge -> frequency
        + tfidf/unlabeled data for SVD
    """
    graphs = _load_graphs_from_s3(graphs_bucket, graphs_prefix)
    logger.info(f"Loaded {len(graphs)} graphs")

    # Determine stage vocab based on version
    if vocab_version == "v1":
        stage_vocab, raw_to_canonical = reduce_stage_vocab(graphs, min_support=3,
                                                            vocab_version="v1")
    else:
        stage_vocab = sorted(V2_STAGE_VOCAB)
        raw_to_canonical = None  # will use map_stage_v2 below

    # Update module-level stage index for feature building
    global N_STAGES, stage_to_idx
    N_STAGES = len(stage_vocab)
    stage_to_idx = {s: i for i, s in enumerate(sorted(stage_vocab))}
    logger.info(f"Connection vocab ({vocab_version}): {N_STAGES} stages")

    # First pass: extract stages and edges per doc, build corpus freq
    corpus_edge_counts = defaultdict(int)
    total_labeled = 0
    doc_data = []  # (graph, mapped_stages, mapped_edges)

    for graph in graphs:
        doc_id = graph.get("document_id", "unknown")
        nodes = {}
        for n in graph.get("nodes", []):
            if n.get("type") == "stage":
                sid = n.get("stage_id", n["id"].replace("stg_", ""))
                order = n.get("features", {}).get("order_normalized", 0.5)
                nodes[n["id"]] = {"raw": sid, "order": order}

        if not nodes:
            doc_data.append((graph, set(), set()))
            continue

        # Build edge context (needed for V2 mapping)
        out_by = defaultdict(list)
        in_by = defaultdict(list)
        for e in graph.get("edges", []):
            if e.get("type") != "stage_transition":
                continue
            out_by[e["source"]].append(nodes.get(e["target"], {}).get("raw", ""))
            in_by[e["target"]].append(nodes.get(e["source"], {}).get("raw", ""))

        # Map stages using appropriate vocab
        stage_map = {}
        for nid, info in nodes.items():
            if vocab_version == "v1":
                canonical = raw_to_canonical.get(info["raw"])
            else:
                canonical = map_stage_v2(
                    info["raw"], order=info["order"],
                    edges_in=[{"type": r} for r in in_by.get(nid, [])],
                    edges_out=[{"target": r} for r in out_by.get(nid, [])],
                )
            if canonical and canonical in stage_to_idx:
                stage_map[nid] = canonical

        mapped_stages = set(stage_map.values())
        mapped_edges = set()
        for e in graph.get("edges", []):
            if e.get("type") != "stage_transition":
                continue
            sv = stage_map.get(e["source"])
            dv = stage_map.get(e["target"])
            if sv and dv and sv != dv:
                mapped_edges.add((sv, dv))
                corpus_edge_counts[f"{sv}|{dv}"] += 1

        if mapped_stages:
            total_labeled += 1

        doc_data.append((graph, mapped_stages, mapped_edges))

    # Corpus edge frequencies (normalized)
    max_count = max(corpus_edge_counts.values()) if corpus_edge_counts else 1
    corpus_edge_freq = {k: v / max_count for k, v in corpus_edge_counts.items()}
    logger.info(f"Corpus edges: {len(corpus_edge_counts)} unique, "
                f"{total_labeled} labeled docs")

    # Second pass: build features
    feature_names = build_feature_names()
    all_tfidf = []
    labeled_doc_vecs = []
    labeled_doc_ids = []
    labeled_tfidf = []
    unlabeled_vecs = []
    unlabeled_ids = []
    unlabeled_tfidf = []

    pair_features = []
    pair_labels = []
    pair_doc_indices = []
    pair_names = []

    for graph, mapped_stages, mapped_edges in doc_data:
        doc_id = graph.get("document_id", "unknown")
        vec = flatten_document(graph)
        doc_tfidf = _extract_tfidf_vectors(graph)
        all_tfidf.append(doc_tfidf)

        if not mapped_stages:
            unlabeled_vecs.append(vec)
            unlabeled_ids.append(doc_id)
            unlabeled_tfidf.append(doc_tfidf)
            continue

        doc_idx = len(labeled_doc_vecs)
        labeled_doc_vecs.append(vec)
        labeled_doc_ids.append(doc_id)
        labeled_tfidf.append(doc_tfidf)

        # Generate all ordered pairs
        sorted_stages = sorted(mapped_stages)
        for src in sorted_stages:
            for dst in sorted_stages:
                if src == dst:
                    continue
                freq = corpus_edge_freq.get(f"{src}|{dst}", 0.0)
                pair_vec = build_connection_features(vec, src, dst, freq)
                pair_features.append(pair_vec)
                pair_labels.append(1.0 if (src, dst) in mapped_edges else 0.0)
                pair_doc_indices.append(doc_idx)
                pair_names.append(f"{src}|{dst}")

    X_docs = np.array(labeled_doc_vecs, dtype=np.float64)
    X_unlabeled = (np.array(unlabeled_vecs, dtype=np.float64)
                   if unlabeled_vecs
                   else np.zeros((0, X_docs.shape[1]), dtype=np.float64))

    # Log-compress doc features
    X_docs = np.sign(X_docs) * np.log1p(np.abs(X_docs))
    np.nan_to_num(X_docs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    X_pairs = np.array(pair_features, dtype=np.float32)
    y_pairs = np.array(pair_labels, dtype=np.float32)

    n_pos = int(y_pairs.sum())
    n_neg = len(y_pairs) - n_pos
    logger.info(f"Connection pairs: {len(y_pairs)} total, "
                f"{n_pos} positive ({n_pos/len(y_pairs)*100:.1f}%), "
                f"{n_neg} negative")

    return {
        "X_pairs": X_pairs,
        "y_pairs": y_pairs,
        "pair_doc_indices": np.array(pair_doc_indices),
        "pair_names": pair_names,
        "X_docs": X_docs,
        "doc_ids": labeled_doc_ids,
        "X_unlabeled": X_unlabeled,
        "unlabeled_ids": unlabeled_ids,
        "tfidf_labeled": labeled_tfidf,
        "tfidf_unlabeled": unlabeled_tfidf,
        "tfidf_all": all_tfidf,
        "feature_names": feature_names,
        "corpus_edge_freq": {k: round(v, 4) for k, v in corpus_edge_freq.items()},
        "corpus_edge_counts": {k: int(v) for k, v in corpus_edge_counts.items()},
    }


def train_connection_model(X: np.ndarray, y: np.ndarray, doc_indices: np.ndarray,
                           folds: int = 5, output_dir: str = "/opt/ml/model") -> dict:
    """Train single XGBoost classifier for connection prediction."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    N = len(y)
    n_pos = int(y.sum())
    scale_pos = max(1.0, (N - n_pos) / max(n_pos, 1))
    logger.info(f"Training connection model: {N} pairs, {n_pos} positive, "
                f"scale_pos={scale_pos:.1f}")

    # Split by DOCUMENT (not by pair) to avoid leakage
    unique_docs = np.unique(doc_indices)
    n_docs = len(unique_docs)
    doc_labels = np.array([y[doc_indices == d].mean() > 0 for d in unique_docs])

    n_splits = min(folds, n_docs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probs = np.zeros(N, dtype=np.float64)
    fold_f1s = []

    for fold, (train_doc_idx, val_doc_idx) in enumerate(skf.split(unique_docs, doc_labels)):
        train_docs = set(unique_docs[train_doc_idx])
        val_docs = set(unique_docs[val_doc_idx])

        train_mask = np.array([d in train_docs for d in doc_indices])
        val_mask = np.array([d in val_docs for d in doc_indices])

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        clf = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            colsample_bytree=0.7, subsample=0.8,
            scale_pos_weight=min(scale_pos, 10.0),
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        val_prob = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_mask] = val_prob

        # Find best threshold
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            pred = (val_prob >= t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        fold_f1s.append(best_f1)
        logger.info(f"  Fold {fold}: F1={best_f1:.4f} (threshold={best_t:.2f})")

    mean_f1 = float(np.mean(fold_f1s))
    logger.info(f"OOF connection F1: {mean_f1:.4f}")

    # Find global threshold
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        pred = (oof_probs >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    logger.info(f"Global threshold: {best_t:.2f} (F1={best_f1:.4f})")

    # Train final model on all data
    clf_final = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        colsample_bytree=0.7, subsample=0.8,
        scale_pos_weight=min(scale_pos, 10.0),
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    clf_final.fit(X, y)

    # Save
    clf_final.save_model(str(out / "xgb_connection.json"))
    with open(out / "connection_threshold.json", "w") as f:
        json.dump({"threshold": round(best_t, 2), "oof_f1": round(best_f1, 4)}, f)

    return {
        "model": clf_final,
        "threshold": best_t,
        "oof_f1": best_f1,
        "oof_probs": oof_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train connection predictor")
    parser.add_argument("--graphs-bucket", type=str, required=True)
    parser.add_argument("--graphs-prefix", type=str, default="graphs/")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-pct", type=float, default=0.10)
    parser.add_argument("--val-pct", type=float, default=0.20)
    parser.add_argument("--vocab-version", type=str, default="v1", choices=["v1", "v2"])
    args = parser.parse_args()

    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    t0 = time.time()

    logger.info(f"Loading from s3://{args.graphs_bucket}/{args.graphs_prefix}")
    data = load_connection_data(args.graphs_bucket, args.graphs_prefix,
                                vocab_version=args.vocab_version)

    X_pairs = data["X_pairs"]
    y_pairs = data["y_pairs"]
    doc_indices = data["pair_doc_indices"]

    # Train/val/test split by DOCUMENT
    unique_docs = np.unique(doc_indices)
    n_docs = len(unique_docs)
    holdout_pct = args.val_pct + args.test_pct

    test_doc_set = set()
    val_doc_set = set()

    if holdout_pct > 0 and n_docs > 30:
        train_docs, holdout_docs = train_test_split(
            unique_docs, test_size=holdout_pct, random_state=42)
        if args.test_pct > 0 and args.val_pct > 0:
            test_frac = args.test_pct / holdout_pct
            val_docs, test_docs = train_test_split(
                holdout_docs, test_size=test_frac, random_state=42)
        elif args.test_pct > 0:
            val_docs, test_docs = np.array([]), holdout_docs
        else:
            val_docs, test_docs = holdout_docs, np.array([])

        test_doc_set = set(test_docs)
        val_doc_set = set(val_docs)
        train_doc_set = set(train_docs)

        train_mask = np.array([d in train_doc_set for d in doc_indices])
        test_mask = np.array([d in test_doc_set for d in doc_indices])
        val_mask = np.array([d in val_doc_set for d in doc_indices])

        X_train, y_train = X_pairs[train_mask], y_pairs[train_mask]
        doc_idx_train = doc_indices[train_mask]
        X_test, y_test = X_pairs[test_mask], y_pairs[test_mask]
        X_val, y_val = X_pairs[val_mask], y_pairs[val_mask]

        logger.info(f"Split: {len(train_doc_set)} train docs ({train_mask.sum()} pairs), "
                    f"{len(val_doc_set)} val ({val_mask.sum()} pairs), "
                    f"{len(test_doc_set)} test ({test_mask.sum()} pairs)")
    else:
        X_train, y_train, doc_idx_train = X_pairs, y_pairs, doc_indices
        X_test, y_test = None, None
        X_val, y_val = None, None

    # Train
    result = train_connection_model(X_train, y_train, doc_idx_train,
                                    folds=args.folds, output_dir=output_dir)

    # Evaluate on held-out sets
    out = Path(output_dir)
    threshold = result["threshold"]

    def _eval_split(X_h, y_h, name):
        if X_h is None or len(X_h) == 0:
            return None
        probs = result["model"].predict_proba(X_h)[:, 1]
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(y_h, preds, zero_division=0)
        prec = precision_score(y_h, preds, zero_division=0)
        rec = recall_score(y_h, preds, zero_division=0)
        n_pos = int(y_h.sum())
        n_pred = int(preds.sum())
        logger.info(f"\n{name}: F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  "
                    f"(GT={n_pos}, Pred={n_pred}, Total={len(y_h)})")
        return {"f1": round(float(f1), 4), "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "n_positive": n_pos, "n_predicted": n_pred, "n_total": len(y_h)}

    val_metrics = _eval_split(X_val, y_val, "Validation")
    test_metrics = _eval_split(X_test, y_test, "Test")

    results = {
        "oof_f1": round(result["oof_f1"], 4),
        "threshold": round(threshold, 2),
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(out / "connection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save corpus edge freq for inference
    with open(out / "corpus_edge_freq.json", "w") as f:
        json.dump(data["corpus_edge_freq"], f, indent=2)
    with open(out / "corpus_edge_counts.json", "w") as f:
        json.dump(data["corpus_edge_counts"], f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nComplete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
