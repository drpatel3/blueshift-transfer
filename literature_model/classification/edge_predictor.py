"""XGBoost edge-level prediction: predict stage-to-stage transitions.

Uses the same document feature vectors as stage_predictor.py but trains
binary classifiers for each edge (src→dst) in the V2 vocabulary.

Usage (SageMaker entry script):
    Hyperparameters: graphs-bucket, graphs-prefix, folds, min-support,
                     test-pct, val-pct
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from stage_predictor import (
    flatten_document, _extract_tfidf_vectors, fit_tfidf_svd,
    transform_tfidf_svd, build_svd_feature_names,
    _load_graphs_from_s3, _load_cross_doc_edges, _compute_neighbor_features,
    build_feature_names, FEATURE_SECTION_TYPES, V2_STAGE_VOCAB,
)
from test_v2_mapping import map_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_EDGE_SUPPORT = 3  # Minimum docs per edge to include in vocabulary


# ---------------------------------------------------------------------------
# Data loading: extract V2 edges from graphs
# ---------------------------------------------------------------------------

def load_edge_training_data(graphs_bucket: str, graphs_prefix: str,
                            min_support: int = MIN_EDGE_SUPPORT):
    """Load graphs and build edge-level training data.

    Returns dict with:
        X: (N_labeled, D) feature matrix
        edge_labels: (N_labeled, E) binary edge presence matrix
        edge_vocab: list of E edge strings "src|dst"
        doc_ids: list of N_labeled document IDs
        X_unlabeled, unlabeled_ids, tfidf_labeled, tfidf_unlabeled, tfidf_all
        transition_counts: corpus edge counts
    """
    graphs = _load_graphs_from_s3(graphs_bucket, graphs_prefix)
    logger.info(f"Loaded {len(graphs)} graphs")

    # Map all stages to V2 and extract edges per doc
    edge_doc_counts = defaultdict(set)  # edge_key -> set of doc_ids
    doc_v2_edges = {}  # doc_id -> set of (src_v2, dst_v2)

    for graph in graphs:
        doc_id = graph.get("document_id", "unknown")
        nodes = {}
        for n in graph.get("nodes", []):
            if n.get("type") == "stage":
                sid = n.get("stage_id", n["id"].replace("stg_", ""))
                order = n.get("features", {}).get("order_normalized", 0.5)
                nodes[n["id"]] = {"raw": sid, "order": order}

        if not nodes:
            continue

        # Build edge context per node
        out_by = defaultdict(list)
        in_by = defaultdict(list)
        for e in graph.get("edges", []):
            if e.get("type") != "stage_transition":
                continue
            src_raw = nodes.get(e["source"], {}).get("raw", "")
            dst_raw = nodes.get(e["target"], {}).get("raw", "")
            out_by[e["source"]].append(dst_raw)
            in_by[e["target"]].append(src_raw)

        # Map nodes to V2
        v2_map = {}
        for nid, info in nodes.items():
            v2 = map_v2(info["raw"], info["order"],
                        in_by.get(nid, []), out_by.get(nid, []))
            if v2:
                v2_map[nid] = v2

        # Extract V2 edges
        doc_edges = set()
        for e in graph.get("edges", []):
            if e.get("type") != "stage_transition":
                continue
            sv = v2_map.get(e["source"])
            dv = v2_map.get(e["target"])
            if sv and dv and sv != dv:
                doc_edges.add((sv, dv))
                edge_doc_counts[f"{sv}|{dv}"].add(doc_id)

        if doc_edges:
            doc_v2_edges[doc_id] = doc_edges

    # Build edge vocabulary (edges with >= min_support docs)
    edge_vocab = sorted(
        k for k, docs in edge_doc_counts.items()
        if len(docs) >= min_support
    )
    edge_to_idx = {e: i for i, e in enumerate(edge_vocab)}
    E = len(edge_vocab)
    logger.info(f"Edge vocab: {E} edges (from {len(edge_doc_counts)} unique, "
                f">={min_support} doc support)")

    # Build feature matrix and edge label matrix
    feature_names = build_feature_names()
    X_list, edge_label_list, doc_ids = [], [], []
    X_unlabeled_list, unlabeled_ids = [], []
    tfidf_labeled, tfidf_unlabeled, tfidf_all = [], [], []

    for graph in graphs:
        doc_id = graph.get("document_id", "unknown")
        vec = flatten_document(graph)
        doc_tfidf = _extract_tfidf_vectors(graph)
        tfidf_all.append(doc_tfidf)

        if doc_id not in doc_v2_edges:
            X_unlabeled_list.append(vec)
            tfidf_unlabeled.append(doc_tfidf)
            unlabeled_ids.append(doc_id)
            continue

        X_list.append(vec)
        tfidf_labeled.append(doc_tfidf)
        doc_ids.append(doc_id)

        # Edge label vector
        label = np.zeros(E, dtype=np.float32)
        for src, dst in doc_v2_edges[doc_id]:
            key = f"{src}|{dst}"
            if key in edge_to_idx:
                label[edge_to_idx[key]] = 1.0
        edge_label_list.append(label)

    X = np.array(X_list, dtype=np.float64)
    edge_labels = np.array(edge_label_list, dtype=np.float32)
    X_unlabeled = (np.array(X_unlabeled_list, dtype=np.float64)
                   if X_unlabeled_list
                   else np.zeros((0, X.shape[1]), dtype=np.float64))

    # Log-compress
    X = np.sign(X) * np.log1p(np.abs(X))
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if X_unlabeled.shape[0] > 0:
        X_unlabeled = np.sign(X_unlabeled) * np.log1p(np.abs(X_unlabeled))
        np.nan_to_num(X_unlabeled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Transition counts
    transition_counts = {k: int(len(v)) for k, v in edge_doc_counts.items()}

    logger.info(f"Labeled: {X.shape[0]} docs, {E} edges, "
                f"Unlabeled: {X_unlabeled.shape[0]} docs")

    return {
        "X": X,
        "edge_labels": edge_labels,
        "edge_vocab": edge_vocab,
        "doc_ids": doc_ids,
        "X_unlabeled": X_unlabeled,
        "unlabeled_ids": unlabeled_ids,
        "tfidf_labeled": tfidf_labeled,
        "tfidf_unlabeled": tfidf_unlabeled,
        "tfidf_all": tfidf_all,
        "feature_names": feature_names,
        "transition_counts": transition_counts,
    }


# ---------------------------------------------------------------------------
# Edge model training
# ---------------------------------------------------------------------------

def train_edge_models(X: np.ndarray, y: np.ndarray, edge_vocab: list[str],
                      feature_names: list[str], folds: int = 5,
                      output_dir: str = "/opt/ml/model") -> dict:
    """Train XGBoost binary classifiers for each edge.

    Same approach as stage_predictor.train_stage_models but for edges.
    Uses classifier chain ordered by edge frequency.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    N, D = X.shape
    E = len(edge_vocab)

    # Variance threshold
    var_selector = VarianceThreshold(threshold=0.0)
    X_sel = var_selector.fit_transform(X)
    n_kept = X_sel.shape[1]
    logger.info(f"VarianceThreshold: {D} -> {n_kept} features ({D - n_kept} removed)")

    # Chain order: most frequent edges first
    edge_freqs = y.sum(axis=0)
    chain_order = list(np.argsort(-edge_freqs))

    logger.info(f"Chain order ({E} edges): "
                + ", ".join(f"{edge_vocab[i]}({int(edge_freqs[i])})"
                            for i in chain_order[:10]) + "...")

    # OOF prediction matrix
    oof_probs = np.zeros((N, E), dtype=np.float64)
    models = {}
    thresholds = {}
    per_edge_metrics = {}

    for chain_pos, ei in enumerate(chain_order):
        edge_name = edge_vocab[ei]
        y_col = y[:, ei]
        n_pos = int(y_col.sum())

        if n_pos < 2:
            logger.info(f"  {edge_name}: skip (n_pos={n_pos})")
            continue

        # Augment features with prior chain predictions
        prior_indices = chain_order[:chain_pos]
        if prior_indices:
            X_aug = np.hstack([X_sel, oof_probs[:, prior_indices]])
        else:
            X_aug = X_sel

        # Cross-validation
        n_splits = min(folds, n_pos, N - n_pos)
        if n_splits < 2:
            n_splits = 2
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_f1s = []
        scale_pos = max(1.0, (N - n_pos) / max(n_pos, 1))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_aug, y_col)):
            clf = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                colsample_bytree=0.7,
                subsample=0.8,
                scale_pos_weight=min(scale_pos, 10.0),
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            clf.fit(X_aug[train_idx], y_col[train_idx])
            val_prob = clf.predict_proba(X_aug[val_idx])[:, 1]

            # Find best threshold
            best_t, best_f1 = 0.5, 0.0
            for t in np.arange(0.2, 0.8, 0.05):
                pred = (val_prob >= t).astype(int)
                f1 = f1_score(y_col[val_idx], pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

            fold_f1s.append(best_f1)
            oof_probs[val_idx, ei] = val_prob

        mean_f1 = float(np.mean(fold_f1s))
        per_edge_metrics[edge_name] = {
            "mean_f1": round(mean_f1, 4),
            "n_positive": n_pos,
        }

        # Find threshold on full OOF
        best_t, best_f1_full = 0.5, 0.0
        for t in np.arange(0.2, 0.8, 0.05):
            pred = (oof_probs[:, ei] >= t).astype(int)
            f1 = f1_score(y_col, pred, zero_division=0)
            if f1 > best_f1_full:
                best_f1_full = f1
                best_t = t
        thresholds[edge_name] = round(best_t, 2)

        # Train final model on all data
        clf_final = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            colsample_bytree=0.7, subsample=0.8,
            scale_pos_weight=min(scale_pos, 10.0),
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        clf_final.fit(X_aug, y_col)
        models[edge_name] = clf_final

        if chain_pos < 30 or chain_pos % 20 == 0:
            logger.info(f"  {edge_name}: F1={mean_f1:.4f} (n_pos={n_pos})")

    # Save models
    for edge_name, clf in models.items():
        safe_name = edge_name.replace("|", "__")
        clf.save_model(str(out / f"xgb_edge_{safe_name}.json"))

    with open(out / "edge_vocab.json", "w") as f:
        json.dump(edge_vocab, f, indent=2)
    with open(out / "edge_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(out / "edge_metrics.json", "w") as f:
        json.dump(per_edge_metrics, f, indent=2)
    with open(out / "edge_chain_order.json", "w") as f:
        json.dump([int(x) for x in chain_order], f, indent=2)
    with open(out / "edge_var_selector.pkl", "wb") as f:
        pickle.dump(var_selector, f)

    # Aggregate OOF metrics
    threshold_vec = np.array([thresholds.get(edge_vocab[i], 0.5) for i in range(E)])
    oof_pred = (oof_probs >= threshold_vec).astype(int)
    active = [i for i in range(E) if edge_vocab[i] in models]
    if active:
        macro_f1 = f1_score(y[:, active], oof_pred[:, active],
                            average="macro", zero_division=0)
        micro_f1 = f1_score(y[:, active], oof_pred[:, active],
                            average="micro", zero_division=0)
        logger.info(f"\nOOF edge metrics ({len(active)} active edges):")
        logger.info(f"  macro-F1={macro_f1:.4f}  micro-F1={micro_f1:.4f}")

    logger.info(f"Saved {len(models)} edge models to {output_dir}")

    return {
        "models": models,
        "thresholds": thresholds,
        "per_edge_metrics": per_edge_metrics,
        "chain_order": chain_order,
        "var_selector": var_selector,
        "oof_probs": oof_probs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost edge predictors")
    parser.add_argument("--graphs-bucket", type=str, required=True)
    parser.add_argument("--graphs-prefix", type=str, default="graphs/")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--test-pct", type=float, default=0.10)
    parser.add_argument("--val-pct", type=float, default=0.20)
    args = parser.parse_args()

    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    logger.info(f"Loading data from s3://{args.graphs_bucket}/{args.graphs_prefix}")
    t0 = time.time()
    data = load_edge_training_data(args.graphs_bucket, args.graphs_prefix,
                                   min_support=args.min_support)

    # TF-IDF SVD
    logger.info("Fitting TF-IDF SVD...")
    svd_models = fit_tfidf_svd(data["tfidf_all"])
    X = data["X"]
    X_unlabeled = data["X_unlabeled"]

    svd_labeled = np.array([
        transform_tfidf_svd(t, svd_models) for t in data["tfidf_labeled"]
    ], dtype=np.float64)
    X = np.hstack([X, svd_labeled])

    if X_unlabeled.shape[0] > 0:
        svd_unlabeled = np.array([
            transform_tfidf_svd(t, svd_models) for t in data["tfidf_unlabeled"]
        ], dtype=np.float64)
        X_unlabeled = np.hstack([X_unlabeled, svd_unlabeled])

    feature_names = data["feature_names"] + build_svd_feature_names(svd_models)

    # Neighbor features
    edges = _load_cross_doc_edges(args.graphs_bucket, args.graphs_prefix)
    if edges:
        nbr = _compute_neighbor_features(X, data["doc_ids"], edges, k=10)
        X = np.hstack([X, nbr])
        feature_names += [f"neighbor_diff_{i}" for i in range(nbr.shape[1])]
        if X_unlabeled.shape[0] > 0:
            nbr_unl = _compute_neighbor_features(
                X_unlabeled, data["unlabeled_ids"], edges, k=10)
            X_unlabeled = np.hstack([X_unlabeled, nbr_unl])

    logger.info(f"Total features: {len(feature_names)}")

    # Train/val/test split
    y = data["edge_labels"]
    N = X.shape[0]
    holdout_pct = args.val_pct + args.test_pct
    X_test, y_test, test_doc_ids = None, None, []
    X_val, y_val, val_doc_ids = None, None, []

    if holdout_pct > 0 and N > 30:
        indices = np.arange(N)
        train_idx, holdout_idx = train_test_split(
            indices, test_size=holdout_pct, random_state=42)
        if args.test_pct > 0 and args.val_pct > 0:
            test_frac = args.test_pct / holdout_pct
            val_idx, test_idx = train_test_split(
                holdout_idx, test_size=test_frac, random_state=42)
        elif args.test_pct > 0:
            val_idx, test_idx = np.array([], dtype=int), holdout_idx
        else:
            val_idx, test_idx = holdout_idx, np.array([], dtype=int)

        if len(test_idx) > 0:
            X_test, y_test = X[test_idx], y[test_idx]
            test_doc_ids = [data["doc_ids"][i] for i in test_idx]
        if len(val_idx) > 0:
            X_val, y_val = X[val_idx], y[val_idx]
            val_doc_ids = [data["doc_ids"][i] for i in val_idx]

        X, y = X[train_idx], y[train_idx]
        data["doc_ids"] = [data["doc_ids"][i] for i in train_idx]
        logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Train
    logger.info("Training edge models...")
    result = train_edge_models(
        X, y, data["edge_vocab"], feature_names,
        folds=args.folds, output_dir=output_dir)

    # Save OOF predictions
    out = Path(output_dir)
    oof_probs = result["oof_probs"]
    oof_preds = []
    for i, doc_id in enumerate(data["doc_ids"]):
        pred_edges = {}
        gt_edges = []
        for j, edge in enumerate(data["edge_vocab"]):
            prob = float(oof_probs[i, j])
            if prob >= result["thresholds"].get(edge, 0.5):
                pred_edges[edge] = round(prob, 4)
            if y[i, j] > 0:
                gt_edges.append(edge)
        oof_preds.append({
            "doc_id": doc_id,
            "ground_truth_edges": gt_edges,
            "predicted_edges": pred_edges,
        })
    with open(out / "oof_edge_predictions.json", "w") as f:
        json.dump(oof_preds, f, indent=2)

    # Evaluate held-out sets
    def _eval_edges(X_h, y_h, doc_ids_h, split_name):
        if X_h is None or len(doc_ids_h) == 0:
            return None
        models = result["models"]
        thresholds = result["thresholds"]
        var_selector = result["var_selector"]
        chain_order = result["chain_order"]
        edge_vocab = data["edge_vocab"]
        E = len(edge_vocab)

        X_sel = var_selector.transform(X_h)
        N_h = X_sel.shape[0]
        probs = np.zeros((N_h, E), dtype=np.float64)

        for chain_pos, ei in enumerate(chain_order):
            edge = edge_vocab[ei]
            if edge not in models:
                continue
            clf = models[edge]
            prior = chain_order[:chain_pos]
            x = np.hstack([X_sel, probs[:, prior]]) if prior else X_sel
            probs[:, ei] = clf.predict_proba(x)[:, 1]

        # Compute metrics
        t_vec = np.array([thresholds.get(edge_vocab[i], 0.5) for i in range(E)])
        preds = (probs >= t_vec).astype(int)
        active = [i for i in range(E) if edge_vocab[i] in models]
        if not active:
            return None

        macro = f1_score(y_h[:, active], preds[:, active], average="macro", zero_division=0)
        micro = f1_score(y_h[:, active], preds[:, active], average="micro", zero_division=0)
        prec = precision_score(y_h[:, active], preds[:, active], average="macro", zero_division=0)
        rec = recall_score(y_h[:, active], preds[:, active], average="macro", zero_division=0)

        logger.info(f"\n{split_name} ({len(doc_ids_h)} docs):")
        logger.info(f"  edge macro-F1={macro:.4f}  micro-F1={micro:.4f}  P={prec:.4f}  R={rec:.4f}")

        # Per-doc
        per_doc = []
        for i, did in enumerate(doc_ids_h):
            gt = [edge_vocab[j] for j in range(E) if y_h[i, j] > 0]
            pred = [edge_vocab[j] for j in range(E) if preds[i, j] > 0]
            correct = set(gt) & set(pred)
            n_gt, n_pred, n_c = len(gt), len(pred), len(correct)
            doc_f1 = (2 * n_c / (n_gt + n_pred)) if (n_gt + n_pred) > 0 else 0
            per_doc.append({
                "doc_id": did,
                "ground_truth_edges": gt,
                "predicted_edges": pred,
                "correct": sorted(correct),
                "missed": sorted(set(gt) - set(pred)),
                "false_positive": sorted(set(pred) - set(gt)),
                "doc_f1": round(doc_f1, 4),
            })

        return {
            "metrics": {"macro_f1": round(float(macro), 4), "micro_f1": round(float(micro), 4),
                        "precision": round(float(prec), 4), "recall": round(float(rec), 4),
                        "num_docs": len(doc_ids_h)},
            "per_doc": per_doc,
        }

    val_res = _eval_edges(X_val, y_val, val_doc_ids, "Validation")
    test_res = _eval_edges(X_test, y_test, test_doc_ids, "Test")

    if val_res:
        with open(out / "val_edge_predictions.json", "w") as f:
            json.dump(val_res, f, indent=2)
    if test_res:
        with open(out / "test_edge_predictions.json", "w") as f:
            json.dump(test_res, f, indent=2)

    # Save metadata
    with open(out / "edge_transition_counts.json", "w") as f:
        json.dump(data["transition_counts"], f, indent=2)
    with open(out / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(out / "svd_models.pkl", "wb") as f:
        pickle.dump(svd_models, f)

    elapsed = time.time() - t0
    logger.info(f"\nTraining complete in {elapsed:.0f}s")
    logger.info(f"Edge models: {len(result['models'])}, "
                f"edge vocab: {len(data['edge_vocab'])}")


if __name__ == "__main__":
    main()
