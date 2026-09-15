"""Graph Attention Network for flowsheet stage prediction — runs on SageMaker GPU.

Two-level GAT:
  Level 1 (intra-doc): GATConv over section_cooccurrence edges per document
  Level 2 (cross-doc): GATConv over cross-document similarity edges
  Head: MLP -> sigmoid for multi-label classification (229 stages)

Semi-supervised: ~200 labeled docs + 857 total in graph structure.

Inference: provide head grade (+ optional deposit type, ore type) -> find similar
documents in the learned embedding space -> predict stages -> assemble ordered
flowsheet using corpus transition patterns.

Usage (SageMaker entry script):
    Hyperparameters: graphs-bucket, graphs-prefix, epochs, lr, patience, folds
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

# Heavy imports deferred — only needed for training, not inference
try:
    import boto3
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score, precision_score, recall_score
    from torch_geometric.nn import GATConv
except ImportError:
    pass  # Inference-only mode: predict_flowsheet works without these

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(graphs_bucket: str, graphs_prefix: str = "graphs/"):
    """Load all document graphs, cross-doc edges, and stage vocab from S3.

    Returns dict with all tensors needed for training:
        - doc_ids: list of document IDs (length N)
        - intra_node_features: list of N tensors, each (num_context_nodes, feat_dim)
        - intra_edge_indices: list of N tensors, each (2, num_edges)
        - label_matrix: (N, num_stages) binary tensor
        - label_mask: (N,) boolean tensor (True = has stage labels)
        - cross_doc_edge_index: (2, num_cross_edges) tensor
        - cross_doc_edge_weight: (num_cross_edges,) tensor
        - stage_vocab: dict mapping stage_id -> index
    """
    s3 = boto3.client("s3")

    # Load stage vocab
    resp = s3.get_object(Bucket=graphs_bucket, Key=f"{graphs_prefix}stage_vocab.json")
    stage_vocab = json.loads(resp["Body"].read())
    num_stages = len(stage_vocab)
    logger.info(f"Stage vocab: {num_stages} stages")

    # Load cross-doc edges
    resp = s3.get_object(Bucket=graphs_bucket, Key=f"{graphs_prefix}cross_doc_edges.json")
    cross_doc_data = json.loads(resp["Body"].read())
    logger.info(f"Cross-doc edges loaded for {len(cross_doc_data)} docs")

    # List all graph files
    graph_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=graphs_bucket, Prefix=graphs_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json") and key != f"{graphs_prefix}cross_doc_edges.json" \
               and key != f"{graphs_prefix}stage_vocab.json" \
               and key != f"{graphs_prefix}graphs_summary.json" \
               and key != f"{graphs_prefix}checkpoint.json":
                graph_keys.append(key)
    logger.info(f"Found {len(graph_keys)} graph files")

    # Keywords used for metadata extraction (must match graph_pipeline.py)
    DEPOSIT_KEYWORDS = [
        "porphyry", "skarn", "vms", "iocg", "sedimentary", "epithermal",
        "orogenic", "breccia", "intrusive", "volcanic",
    ]
    ORE_KEYWORDS = [
        "oxide", "sulfide", "sulphide", "supergene", "hypogene",
        "transition", "refractory", "free_milling",
    ]

    # Load all graphs
    doc_ids = []
    intra_node_features = []
    intra_edge_indices = []
    label_matrix = []
    label_mask = []
    doc_id_to_idx = {}
    doc_metadata = []       # per-doc metadata for inference matching
    all_transitions = []    # (source_stage, target_stage) across corpus

    # Determine feature dim from first graph
    feat_dim = None

    for i, key in enumerate(graph_keys):
        try:
            resp = s3.get_object(Bucket=graphs_bucket, Key=key)
            graph = json.loads(resp["Body"].read())
        except Exception as e:
            logger.warning(f"Failed to load {key}: {e}")
            continue

        doc_id = graph.get("document_id", key.split("/")[-1].replace(".json", ""))

        # Extract context nodes and their features
        context_nodes = []
        context_node_ids = []
        for node in graph.get("nodes", []):
            if node["type"] != "context":
                continue
            feats = node.get("features", {})
            tfidf = feats.pop("tfidf", [])
            section_idx = feats.pop("section_index", 0)
            # Scalar features: use sorted keys for consistent ordering across graphs
            scalar = [float(feats[k]) for k in sorted(feats.keys())
                      if isinstance(feats[k], (int, float))]
            scalar.append(float(section_idx))
            # Combine scalar + tfidf
            feature_vec = scalar + list(tfidf)
            # Sanitize NaN/Inf values
            feature_vec = [0.0 if (v != v or v == float('inf') or v == float('-inf'))
                           else v for v in feature_vec]
            context_nodes.append(feature_vec)
            context_node_ids.append(node["id"])

        if len(context_nodes) == 0:
            continue

        # Pad/truncate to consistent dim
        if feat_dim is None:
            feat_dim = len(context_nodes[0])
            logger.info(f"Feature dimension: {feat_dim}")

        # Pad shorter feature vectors, truncate longer ones
        padded = []
        for vec in context_nodes:
            if len(vec) < feat_dim:
                vec = vec + [0.0] * (feat_dim - len(vec))
            elif len(vec) > feat_dim:
                vec = vec[:feat_dim]
            padded.append(vec)

        node_feats = torch.tensor(padded, dtype=torch.float32)
        intra_node_features.append(node_feats)

        # Build intra-doc edge index from section_cooccurrence edges
        node_id_to_local = {nid: j for j, nid in enumerate(context_node_ids)}
        src_list, dst_list = [], []
        for edge in graph.get("edges", []):
            if edge["type"] != "section_cooccurrence":
                continue
            s = node_id_to_local.get(edge["source"])
            t = node_id_to_local.get(edge["target"])
            if s is not None and t is not None:
                # Bidirectional
                src_list.extend([s, t])
                dst_list.extend([t, s])

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            # Self-loops as fallback for isolated nodes
            n = len(context_node_ids)
            edge_index = torch.tensor([list(range(n)), list(range(n))], dtype=torch.long)
        intra_edge_indices.append(edge_index)

        # Build label vector from stage nodes
        labels = torch.zeros(num_stages, dtype=torch.float32)
        has_stages = False
        for node in graph.get("nodes", []):
            if node["type"] == "stage":
                stage_id = node.get("stage_id", "")
                if stage_id in stage_vocab:
                    labels[stage_vocab[stage_id]] = 1.0
                    has_stages = True

        label_matrix.append(labels)
        label_mask.append(has_stages)

        # Extract metadata for inference matching
        meta = {"deposit_types": [], "ore_types": [], "grade_values": []}
        for node in graph.get("nodes", []):
            if node["type"] != "context":
                continue
            feats = node.get("features", {})
            group = node.get("group", "")
            # Deposit type from geology sections
            if group in ("geology",):
                for kw in DEPOSIT_KEYWORDS:
                    if feats.get(f"kw_{kw}", 0) > 0:
                        meta["deposit_types"].append(kw)
            # Ore type from geology + metallurgical_testing
            if group in ("geology", "metallurgical_testing"):
                for kw in ORE_KEYWORDS:
                    if feats.get(f"kw_{kw}", 0) > 0:
                        meta["ore_types"].append(kw)
            # Grade numerics from resource_estimate
            if group in ("resource_estimate",):
                num_mean = feats.get("num_mean", 0)
                num_max = feats.get("num_max", 0)
                if num_mean > 0:
                    meta["grade_values"].append(num_mean)
                if num_max > 0:
                    meta["grade_values"].append(num_max)
        meta["deposit_types"] = list(set(meta["deposit_types"]))
        meta["ore_types"] = list(set(meta["ore_types"]))
        doc_metadata.append(meta)

        # Collect stage transitions for flowsheet ordering
        for edge in graph.get("edges", []):
            if edge["type"] == "stage_transition":
                src_stage = edge["source"].replace("stg_", "")
                dst_stage = edge["target"].replace("stg_", "")
                all_transitions.append((src_stage, dst_stage))

        doc_id_to_idx[doc_id] = len(doc_ids)
        doc_ids.append(doc_id)

        if (i + 1) % 100 == 0:
            logger.info(f"Loaded {i + 1}/{len(graph_keys)} graphs")

    logger.info(f"Loaded {len(doc_ids)} docs, {sum(label_mask)} with stage labels")

    # Compress extreme feature values and sanitize
    for idx in range(len(intra_node_features)):
        nf = intra_node_features[idx].float()  # ensure float32
        nf = torch.clamp(nf, min=-1e10, max=1e10)  # hard clamp before log
        nf = torch.nan_to_num(nf, nan=0.0, posinf=0.0, neginf=0.0)
        nf = torch.sign(nf) * torch.log1p(nf.abs())
        nf = torch.nan_to_num(nf, nan=0.0, posinf=0.0, neginf=0.0)  # catch any post-log issues
        intra_node_features[idx] = nf

    # Normalize features: z-score after log transform
    all_vecs = torch.cat(intra_node_features, dim=0)  # (total_nodes, feat_dim)
    feat_mean = all_vecs.mean(dim=0)
    feat_std = all_vecs.std(dim=0)
    feat_std[feat_std < 1e-6] = 1.0  # avoid division by zero for constant features
    logger.info(f"Feature stats (post-log): mean range [{feat_mean.min():.4f}, {feat_mean.max():.4f}], "
                f"std range [{feat_std.min():.4f}, {feat_std.max():.4f}], "
                f"max abs value: {all_vecs.abs().max():.4f}")

    # Apply z-score normalization
    intra_node_features = [(nf - feat_mean) / feat_std for nf in intra_node_features]

    label_matrix = torch.stack(label_matrix)
    label_mask = torch.tensor(label_mask, dtype=torch.bool)

    # Build cross-doc edge index
    cross_src, cross_dst, cross_weights = [], [], []
    for doc_id, entry in cross_doc_data.items():
        src_idx = doc_id_to_idx.get(doc_id)
        if src_idx is None:
            continue
        for neighbor in entry.get("neighbors", []):
            dst_idx = doc_id_to_idx.get(neighbor["doc_id"])
            if dst_idx is None:
                continue
            weight = neighbor.get("overall_similarity", 0.5)
            # Bidirectional
            cross_src.extend([src_idx, dst_idx])
            cross_dst.extend([dst_idx, src_idx])
            cross_weights.extend([weight, weight])

    if cross_src:
        cross_doc_edge_index = torch.tensor([cross_src, cross_dst], dtype=torch.long)
        cross_doc_edge_weight = torch.tensor(cross_weights, dtype=torch.float32)
    else:
        # Fallback: self-loops
        n = len(doc_ids)
        cross_doc_edge_index = torch.tensor([list(range(n)), list(range(n))], dtype=torch.long)
        cross_doc_edge_weight = torch.ones(n, dtype=torch.float32)

    logger.info(f"Cross-doc edges: {cross_doc_edge_index.shape[1]}")

    # Build global transition graph (stage_a -> stage_b -> count)
    transition_counts = defaultdict(int)
    for src, dst in all_transitions:
        transition_counts[(src, dst)] += 1
    logger.info(f"Global transitions: {len(transition_counts)} unique edges "
                f"from {len(all_transitions)} total")

    return {
        "doc_ids": doc_ids,
        "intra_node_features": intra_node_features,
        "intra_edge_indices": intra_edge_indices,
        "label_matrix": label_matrix,
        "label_mask": label_mask,
        "cross_doc_edge_index": cross_doc_edge_index,
        "cross_doc_edge_weight": cross_doc_edge_weight,
        "stage_vocab": stage_vocab,
        "feat_dim": feat_dim,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "doc_metadata": doc_metadata,
        "transition_counts": dict(
            (f"{s}|{t}", c) for (s, t), c in transition_counts.items()
        ),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IntraDocGAT(nn.Module):
    """Level 1: per-document GAT over section cooccurrence edges."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        self.projection = nn.Linear(in_dim, hidden_dim)
        self.gat = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.attn_gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns single doc embedding (hidden_dim,) via attention pooling."""
        h = F.elu(self.projection(x))
        h = self.gat(h, edge_index)
        h = F.elu(h)
        # Attention-weighted pooling
        attn_weights = torch.softmax(self.attn_gate(h), dim=0)  # (num_nodes, 1)
        doc_emb = (attn_weights * h).sum(dim=0)  # (hidden_dim,)
        return doc_emb


class CrossDocGAT(nn.Module):
    """Level 2: GAT over cross-document similarity edges."""

    def __init__(self, hidden_dim: int = 128, heads: int = 4):
        super().__init__()
        self.gat = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False,
                           edge_dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """x: (N, hidden_dim), returns (N, hidden_dim)."""
        edge_attr = edge_weight.unsqueeze(-1)  # (E, 1)
        h = self.gat(x, edge_index, edge_attr=edge_attr)
        return F.elu(h)


class FlowsheetPredictor(nn.Module):
    """Two-level GAT + MLP for multi-label flowsheet stage prediction."""

    def __init__(self, in_dim: int, num_stages: int, hidden_dim: int = 128,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.intra_gat = IntraDocGAT(in_dim, hidden_dim, heads)
        self.cross_gat = CrossDocGAT(hidden_dim, heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_stages),
        )
        self.hidden_dim = hidden_dim

    def encode_documents(self, intra_node_features: list, intra_edge_indices: list,
                         device: torch.device) -> torch.Tensor:
        """Run Level 1 on all docs, return (N, hidden_dim) embeddings."""
        doc_embeddings = []
        for node_feats, edge_idx in zip(intra_node_features, intra_edge_indices):
            node_feats = node_feats.to(device)
            edge_idx = edge_idx.to(device)
            emb = self.intra_gat(node_feats, edge_idx)
            doc_embeddings.append(emb)
        return torch.stack(doc_embeddings)  # (N, hidden_dim)

    def forward(self, intra_node_features: list, intra_edge_indices: list,
                cross_doc_edge_index: torch.Tensor, cross_doc_edge_weight: torch.Tensor,
                device: torch.device) -> torch.Tensor:
        """Full forward pass. Returns logits (N, num_stages)."""
        doc_embs = self.encode_documents(intra_node_features, intra_edge_indices, device)
        refined = self.cross_gat(doc_embs, cross_doc_edge_index, cross_doc_edge_weight)
        logits = self.classifier(refined)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweep thresholds on validation set to maximize macro-F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def train_fold(model: FlowsheetPredictor, data: dict, train_idx: np.ndarray,
               val_idx: np.ndarray, device: torch.device, epochs: int = 200,
               lr: float = 1e-3, patience: int = 30, fold: int = 0) -> dict:
    """Train one fold with early stopping and semi-supervised losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    label_matrix = data["label_matrix"].to(device)
    label_mask = data["label_mask"].to(device)
    cross_edge_index = data["cross_doc_edge_index"].to(device)
    cross_edge_weight = data["cross_doc_edge_weight"].to(device)

    # Train/val masks (within labeled docs)
    train_mask = torch.zeros(len(data["doc_ids"]), dtype=torch.bool, device=device)
    val_mask = torch.zeros(len(data["doc_ids"]), dtype=torch.bool, device=device)

    # train_idx and val_idx are indices into labeled_indices
    labeled_indices = torch.where(data["label_mask"])[0].numpy()
    for i in train_idx:
        train_mask[labeled_indices[i]] = True
    for i in val_idx:
        val_mask[labeled_indices[i]] = True

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(data["intra_node_features"], data["intra_edge_indices"],
                       cross_edge_index, cross_edge_weight, device)

        # 1. Supervised loss on labeled training docs
        sup_loss = criterion(logits[train_mask], label_matrix[train_mask])

        # 2. Consistency loss: feature dropout augmentation
        cons_loss = torch.tensor(0.0, device=device)
        if epoch >= 5:
            # Run again with dropout on input features
            augmented_features = []
            for nf in data["intra_node_features"]:
                mask = torch.bernoulli(torch.full(nf.shape, 0.9)).to(device)
                augmented_features.append(nf.to(device) * mask)
            logits_aug = model(augmented_features, data["intra_edge_indices"],
                               cross_edge_index, cross_edge_weight, device)
            cons_loss = F.mse_loss(torch.sigmoid(logits), torch.sigmoid(logits_aug))

        # 3. Pseudo-label loss on high-confidence unlabeled predictions
        pseudo_loss = torch.tensor(0.0, device=device)
        if epoch >= 10:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
            unlabeled = ~label_mask
            if unlabeled.any():
                high_conf = (probs > 0.8) | (probs < 0.2)
                pseudo_targets = (probs > 0.5).float()
                # Only apply where high confidence
                pseudo_mask = high_conf & unlabeled.unsqueeze(1)
                if pseudo_mask.any():
                    pseudo_loss = F.binary_cross_entropy_with_logits(
                        logits[unlabeled], pseudo_targets[unlabeled],
                        weight=high_conf[unlabeled].float(),
                        reduction="mean"
                    )

        total_loss = sup_loss + 0.5 * cons_loss + 0.3 * pseudo_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(data["intra_node_features"], data["intra_edge_indices"],
                                   cross_edge_index, cross_edge_weight, device)
                val_loss = criterion(val_logits[val_mask], label_matrix[val_mask]).item()
                val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
                val_true = label_matrix[val_mask].cpu().numpy()

            threshold = find_optimal_threshold(val_true, val_probs)
            val_pred = (val_probs >= threshold).astype(int)
            macro_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
            micro_f1 = f1_score(val_true, val_pred, average="micro", zero_division=0)
            precision = precision_score(val_true, val_pred, average="macro", zero_division=0)
            recall = recall_score(val_true, val_pred, average="macro", zero_division=0)

            logger.info(f"Fold {fold} Epoch {epoch+1}: loss={total_loss.item():.4f} "
                        f"val_loss={val_loss:.4f} macro_f1={macro_f1:.4f} "
                        f"micro_f1={micro_f1:.4f} P={precision:.4f} R={recall:.4f} "
                        f"thr={threshold:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = {
                    "val_loss": val_loss,
                    "macro_f1": macro_f1,
                    "micro_f1": micro_f1,
                    "precision": precision,
                    "recall": recall,
                    "threshold": threshold,
                    "epoch": epoch + 1,
                }
                wait = 0
            else:
                wait += 5
                if wait >= patience:
                    logger.info(f"Fold {fold}: early stopping at epoch {epoch+1}")
                    break

    return {"best_state": best_state, "metrics": best_metrics}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: FlowsheetPredictor, data: dict, val_indices: np.ndarray,
             threshold: float, device: torch.device) -> dict:
    """Detailed evaluation: per-stage F1, top-K accuracy."""
    model.eval()
    label_matrix = data["label_matrix"].to(device)
    label_mask = data["label_mask"]
    cross_edge_index = data["cross_doc_edge_index"].to(device)
    cross_edge_weight = data["cross_doc_edge_weight"].to(device)
    labeled_indices = torch.where(label_mask)[0].numpy()

    with torch.no_grad():
        logits = model(data["intra_node_features"], data["intra_edge_indices"],
                       cross_edge_index, cross_edge_weight, device)
        probs = torch.sigmoid(logits).cpu().numpy()

    val_global = labeled_indices[val_indices]
    val_probs = probs[val_global]
    val_true = label_matrix[val_global].cpu().numpy()
    val_pred = (val_probs >= threshold).astype(int)

    # Per-stage F1
    inv_vocab = {v: k for k, v in data["stage_vocab"].items()}
    per_stage = {}
    for idx in range(len(data["stage_vocab"])):
        if val_true[:, idx].sum() == 0:
            continue
        stage_f1 = f1_score(val_true[:, idx], val_pred[:, idx], zero_division=0)
        per_stage[inv_vocab.get(idx, str(idx))] = round(float(stage_f1), 4)

    # Top-K accuracy
    top_k_acc = {}
    for k in [5, 10, 20]:
        correct = 0
        total = 0
        for i in range(len(val_true)):
            true_stages = set(np.where(val_true[i] == 1)[0])
            if not true_stages:
                continue
            top_k_pred = set(np.argsort(val_probs[i])[-k:])
            correct += len(true_stages & top_k_pred)
            total += len(true_stages)
        top_k_acc[f"top_{k}_recall"] = round(correct / max(total, 1), 4)

    return {
        "per_stage_f1": per_stage,
        "top_k_accuracy": top_k_acc,
        "num_active_stages": len(per_stage),
    }


# ---------------------------------------------------------------------------
# Main training loop with K-fold CV
# ---------------------------------------------------------------------------

def train(data: dict, epochs: int = 200, lr: float = 1e-3, patience: int = 30,
          folds: int = 5, output_dir: str = "/opt/ml/model"):
    """5-fold CV training, saves best model + metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    feat_dim = data["feat_dim"]
    num_stages = len(data["stage_vocab"])
    labeled_indices = torch.where(data["label_mask"])[0].numpy()
    logger.info(f"Labeled docs: {len(labeled_indices)}, total docs: {len(data['doc_ids'])}")

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(labeled_indices)):
        logger.info(f"\n{'='*60}\nFold {fold}: train={len(train_idx)}, val={len(val_idx)}\n{'='*60}")

        model = FlowsheetPredictor(feat_dim, num_stages).to(device)
        result = train_fold(model, data, train_idx, val_idx, device,
                            epochs=epochs, lr=lr, patience=patience, fold=fold)

        # Load best state and evaluate
        if result["best_state"]:
            model.load_state_dict(result["best_state"])
            eval_result = evaluate(model, data, val_idx, result["metrics"]["threshold"], device)
            result["metrics"].update(eval_result)

            # Save model checkpoint
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(result["best_state"], out / f"model_fold_{fold}.pt")

        fold_results.append(result["metrics"])
        logger.info(f"Fold {fold} best: {result['metrics']}")

    # Aggregate metrics
    agg = {}
    metric_keys = ["macro_f1", "micro_f1", "precision", "recall", "val_loss"]
    for key in metric_keys:
        values = [r[key] for r in fold_results if key in r]
        if values:
            agg[f"mean_{key}"] = round(float(np.mean(values)), 4)
            agg[f"std_{key}"] = round(float(np.std(values)), 4)

    # Aggregate top-K
    for k in [5, 10, 20]:
        metric = f"top_{k}_recall"
        values = [r.get("top_k_accuracy", {}).get(metric, 0) for r in fold_results]
        if values:
            agg[f"mean_{metric}"] = round(float(np.mean(values)), 4)

    logger.info(f"\nAggregate results: {json.dumps(agg, indent=2)}")

    # Save metrics
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_out = {
        "aggregate": agg,
        "per_fold": fold_results,
        "config": {
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "folds": folds,
            "feat_dim": feat_dim,
            "num_stages": num_stages,
            "num_docs": len(data["doc_ids"]),
            "num_labeled": int(data["label_mask"].sum()),
        },
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    # Save stage vocab and normalization stats for inference
    with open(out / "stage_vocab.json", "w") as f:
        json.dump(data["stage_vocab"], f, indent=2)
    torch.save({"feat_mean": data["feat_mean"], "feat_std": data["feat_std"]},
               out / "norm_stats.pt")

    # Build inference index using best fold's model
    best_fold = min(range(len(fold_results)),
                    key=lambda i: fold_results[i].get("val_loss", float("inf")))
    best_threshold = fold_results[best_fold].get("threshold", 0.5)
    logger.info(f"Building inference index from fold {best_fold} (threshold={best_threshold})")

    checkpoint_path = out / f"model_fold_{best_fold}.pt"
    if checkpoint_path.exists():
        best_model = FlowsheetPredictor(feat_dim, num_stages).to(device)
        best_model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        build_inference_index(best_model, data, best_threshold, device, output_dir)
    else:
        logger.error(f"No checkpoint saved for fold {best_fold} — all folds may have diverged (NaN loss)")

    logger.info(f"Model and metrics saved to {output_dir}")
    return metrics_out


# ---------------------------------------------------------------------------
# Inference: input params -> flowsheet
# ---------------------------------------------------------------------------

def build_inference_index(model: FlowsheetPredictor, data: dict, threshold: float,
                          device: torch.device, output_dir: str):
    """After training, build the inference index: embeddings, probs, metadata, transitions.

    Saves inference_index.json with everything needed to predict flowsheets from simple inputs.
    """
    model.eval()
    cross_edge_index = data["cross_doc_edge_index"].to(device)
    cross_edge_weight = data["cross_doc_edge_weight"].to(device)

    with torch.no_grad():
        logits = model(data["intra_node_features"], data["intra_edge_indices"],
                       cross_edge_index, cross_edge_weight, device)
        probs = torch.sigmoid(logits).cpu().numpy()
        doc_embs = model.encode_documents(
            data["intra_node_features"], data["intra_edge_indices"], device
        ).cpu().numpy()

    inv_vocab = {v: k for k, v in data["stage_vocab"].items()}

    # Per-doc: predicted stages with probabilities
    doc_entries = []
    for i, doc_id in enumerate(data["doc_ids"]):
        stage_probs = {}
        for idx in range(len(data["stage_vocab"])):
            p = float(probs[i, idx])
            if p >= 0.1:  # keep anything with non-trivial probability
                stage_probs[inv_vocab[idx]] = round(p, 4)

        doc_entries.append({
            "doc_id": doc_id,
            "embedding": doc_embs[i].tolist(),
            "stage_probs": stage_probs,
            "metadata": data["doc_metadata"][i],
            "has_labels": bool(data["label_mask"][i]),
        })

    index = {
        "docs": doc_entries,
        "stage_vocab": data["stage_vocab"],
        "transition_counts": data["transition_counts"],
        "threshold": threshold,
        "num_docs": len(data["doc_ids"]),
        "num_labeled": int(data["label_mask"].sum()),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "inference_index.json", "w") as f:
        json.dump(index, f, indent=2)
    logger.info(f"Inference index saved: {len(doc_entries)} docs, "
                f"{len(data['transition_counts'])} transitions")
    return index


def _predict_connections_xgb(doc_vec, predicted_stages: set,
                            conn_model_dir: str = None) -> list[dict]:
    """Predict connections between predicted stages using trained connection model.

    Falls back to empty list if model not available.
    """
    if conn_model_dir is None:
        conn_model_dir = os.path.join(os.path.dirname(__file__), "connection_checkpoint")

    conn_path = Path(conn_model_dir)
    model_file = conn_path / "xgb_connection.json"
    if not model_file.exists():
        return []

    try:
        from connection_predictor import build_connection_features
        from xgboost import XGBClassifier
        import numpy as np

        clf = XGBClassifier()
        clf.load_model(str(model_file))

        with open(conn_path / "connection_threshold.json") as f:
            threshold = json.load(f)["threshold"]
        with open(conn_path / "corpus_edge_freq.json") as f:
            corpus_freq = json.load(f)

        connections = []
        sorted_stages = sorted(predicted_stages)
        for src in sorted_stages:
            for dst in sorted_stages:
                if src == dst:
                    continue
                freq = corpus_freq.get(f"{src}|{dst}", 0.0)
                feat_vec = build_connection_features(doc_vec, src, dst, freq)
                prob = clf.predict_proba(feat_vec.reshape(1, -1))[:, 1][0]
                if prob >= threshold:
                    connections.append({
                        "from": src,
                        "to": dst,
                        "confidence": round(float(prob), 4),
                    })

        return connections
    except Exception as e:
        logger.warning(f"Connection model prediction failed: {e}")
        return []


def _predict_with_xgboost(model_dir: str, index: dict, head_grade: float,
                          deposit_type: str, ore_type: str,
                          top_k: int, min_stage_prob: float) -> dict:
    """Use XGBoost stage_predictor models for prediction."""
    import pickle
    from stage_predictor import (
        predict_stages, flatten_document, build_feature_names,
        DEPOSIT_TYPES, ORE_TYPES, FEATURE_SECTION_TYPES, NUMERIC_FEATURES,
        DOMAIN_KEYWORDS, TEXT_FEATURES, TFIDF_SVD_COMPONENTS,
    )
    from xgboost import XGBClassifier

    model_path = Path(model_dir)
    with open(model_path / "stage_vocab.json") as f:
        stage_vocab = json.load(f)
    with open(model_path / "thresholds.json") as f:
        thresholds = json.load(f)
    with open(model_path / "transition_counts.json") as f:
        transition_counts = json.load(f)

    # Load chain order
    chain_order = None
    chain_file = model_path / "chain_order.json"
    if chain_file.exists():
        with open(chain_file) as f:
            chain_order = json.load(f)

    # Load variance selector
    var_selector = None
    var_file = model_path / "var_selector.pkl"
    if var_file.exists():
        with open(var_file, "rb") as f:
            var_selector = pickle.load(f)

    # Load SVD models
    svd_models = None
    svd_file = model_path / "svd_models.pkl"
    if svd_file.exists():
        with open(svd_file, "rb") as f:
            svd_models = pickle.load(f)

    # Load models
    models = {}
    for stage in stage_vocab:
        model_file = model_path / f"xgb_{stage}.json"
        if model_file.exists():
            clf = XGBClassifier()
            clf.load_model(str(model_file))
            models[stage] = clf

    # Find most similar doc from index, use its graph features as base
    docs = index["docs"]
    best_doc = None
    best_score = -1.0
    for doc in docs:
        meta = doc["metadata"]
        score = 0.0
        grade_vals = meta.get("grade_values", [])
        if grade_vals and head_grade > 0:
            min_dist = min(abs(g - head_grade) for g in grade_vals)
            score += 1.0 / (1.0 + min_dist * 5.0) * 2.0
        if deposit_type and deposit_type.lower() in meta.get("deposit_types", []):
            score += 1.5
        if ore_type and ore_type.lower() in meta.get("ore_types", []):
            score += 1.0
        if doc.get("has_labels"):
            score *= 1.2
        if score > best_score:
            best_score = score
            best_doc = doc

    # Build a synthetic feature vector from the matched doc's embedding
    # (In production, the graph JSON would be available; here we approximate
    # from the index metadata)
    feature_names = build_feature_names()
    import numpy as np
    feature_vec = np.zeros(len(feature_names), dtype=np.float32)

    # Set metadata features
    n_per_section = len(NUMERIC_FEATURES) + len(DOMAIN_KEYWORDS) + len(TEXT_FEATURES) + 1
    meta_offset = len(FEATURE_SECTION_TYPES) * n_per_section
    if deposit_type:
        dep = deposit_type.lower()
        if dep in DEPOSIT_TYPES:
            feature_vec[meta_offset + DEPOSIT_TYPES.index(dep)] = 1.0
    if ore_type:
        ore = ore_type.lower()
        if ore in ORE_TYPES:
            feature_vec[meta_offset + len(DEPOSIT_TYPES) + ORE_TYPES.index(ore)] = 1.0
    grade_offset = meta_offset + len(DEPOSIT_TYPES) + len(ORE_TYPES)
    if head_grade > 0:
        feature_vec[grade_offset] = head_grade
        feature_vec[grade_offset + 1] = head_grade
        feature_vec[grade_offset + 2] = head_grade

    result = predict_stages(models, thresholds, feature_vec,
                            transition_counts, stage_vocab,
                            chain_order=chain_order,
                            var_selector=var_selector,
                            svd_models=svd_models)

    # Override connections with trained connection model if available
    predicted_stage_set = set(result.get("stages", []))
    conn_predictions = _predict_connections_xgb(feature_vec, predicted_stage_set)
    if conn_predictions:
        result["connections"] = conn_predictions
        result["stages"] = _topological_sort(predicted_stage_set, conn_predictions)

    # Add similar documents and positions from GAT index
    similar_docs = []
    if best_doc:
        similar_docs = [best_doc["doc_id"]]

    positions = {}
    try:
        from layout_gat import layout_flowsheet
        positions = layout_flowsheet(
            model_path=None,
            stages=result["stages"],
            connections=result["connections"],
        )
    except Exception:
        pass

    result["similar_documents"] = similar_docs
    result["positions"] = positions
    result["model"] = "xgboost"
    result["input"] = {
        "head_grade": head_grade,
        "deposit_type": deposit_type,
        "ore_type": ore_type,
    }
    return result


def predict_flowsheet(index: dict, head_grade: float,
                      deposit_type: str = "", ore_type: str = "",
                      top_k: int = 20, min_stage_prob: float = 0.3,
                      xgb_model_dir: str = None,
                      method: str = "xgboost",
                      graphs: list = None,
                      cross_doc_data: dict = None) -> dict:
    """Predict a process flowsheet from simple inputs.

    Args:
        index: loaded inference_index.json
        head_grade: copper head grade in percent (e.g. 0.5 for 0.5% Cu)
        deposit_type: e.g. "porphyry", "skarn", "vms" (optional)
        ore_type: e.g. "sulfide", "oxide", "mixed" (optional)
        top_k: number of nearest neighbors to consider
        min_stage_prob: minimum probability to include a stage
        xgb_model_dir: path to XGBoost stage_predictor checkpoint dir (optional)
        method: "xgboost" (default), "llm" (LLM graph traversal), or
                "hybrid" (XGBoost prediction refined by LLM)
        graphs: list of document graph JSONs (required for llm/hybrid methods)
        cross_doc_data: cross_doc_edges.json data (optional, for llm/hybrid)

    Returns:
        {
            "stages": [ordered list of stage IDs],
            "connections": [{"from": stage_a, "to": stage_b, "confidence": float}],
            "stage_probabilities": {stage_id: probability},
            "similar_documents": [top-K doc IDs used],
        }
    """
    # LLM-based methods: graph traversal or hybrid
    if method in ("llm", "hybrid"):
        try:
            from llm_predictor import predict_flowsheet_llm
            xgb_result = None
            if method == "hybrid":
                # Run XGBoost first, then pass to LLM for refinement
                try:
                    xgb_result = predict_flowsheet(
                        index, head_grade, deposit_type, ore_type,
                        top_k, min_stage_prob, xgb_model_dir, method="xgboost",
                    )
                except Exception as e:
                    logger.warning(f"XGBoost failed in hybrid mode: {e}")
            return predict_flowsheet_llm(
                index, head_grade, deposit_type, ore_type,
                mode="augment" if method == "hybrid" else "standalone",
                xgb_result=xgb_result,
                graphs=graphs,
                cross_doc_data=cross_doc_data,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning(f"LLM prediction failed, falling back to XGBoost: {e}")

    # Try XGBoost models first if available
    if xgb_model_dir is None:
        xgb_model_dir = os.path.join(os.path.dirname(__file__), "xgb_checkpoint")
    if os.path.isdir(xgb_model_dir) and os.path.exists(os.path.join(xgb_model_dir, "stage_vocab.json")):
        try:
            return _predict_with_xgboost(
                xgb_model_dir, index, head_grade, deposit_type, ore_type,
                top_k, min_stage_prob,
            )
        except Exception as e:
            logger.warning(f"XGBoost prediction failed, falling back to GAT: {e}")

    docs = index["docs"]
    transition_counts = index["transition_counts"]

    # Score each document by similarity to input parameters
    doc_scores = []
    for i, doc in enumerate(docs):
        score = 0.0
        meta = doc["metadata"]

        # Grade similarity: inverse distance, normalized
        grade_vals = meta.get("grade_values", [])
        if grade_vals and head_grade > 0:
            # Use closest grade value in the document
            min_dist = min(abs(g - head_grade) for g in grade_vals)
            # Score: 1.0 for exact match, decays with distance
            grade_score = 1.0 / (1.0 + min_dist * 5.0)
            score += grade_score * 2.0  # weight grade heavily
        elif not grade_vals:
            score += 0.3  # mild penalty for missing grade data

        # Deposit type match
        if deposit_type:
            dep_norm = deposit_type.lower().strip()
            if dep_norm in meta.get("deposit_types", []):
                score += 1.5
            elif meta.get("deposit_types"):
                score += 0.0  # no match
            else:
                score += 0.3  # unknown deposit

        # Ore type match
        if ore_type:
            ore_norm = ore_type.lower().strip()
            doc_ore = meta.get("ore_types", [])
            if ore_norm in doc_ore:
                score += 1.0
            elif ore_norm == "mixed" and len(doc_ore) > 1:
                score += 0.7
            elif doc_ore:
                score += 0.0
            else:
                score += 0.2

        # Boost labeled docs slightly (they have verified stages)
        if doc.get("has_labels"):
            score *= 1.2

        doc_scores.append((i, score))

    # Select top-K most similar documents
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = doc_scores[:top_k]
    total_weight = sum(s for _, s in top_docs) or 1.0

    # Weighted average of stage probabilities
    stage_probs = defaultdict(float)
    for doc_idx, weight in top_docs:
        doc = docs[doc_idx]
        norm_weight = weight / total_weight
        for stage_id, prob in doc["stage_probs"].items():
            stage_probs[stage_id] += prob * norm_weight

    # Filter by minimum probability
    predicted_stages = {
        stage: round(prob, 4)
        for stage, prob in stage_probs.items()
        if prob >= min_stage_prob
    }

    if not predicted_stages:
        # Fallback: take top 5 stages by probability
        sorted_stages = sorted(stage_probs.items(), key=lambda x: x[1], reverse=True)
        predicted_stages = {s: round(p, 4) for s, p in sorted_stages[:5]}

    # Predict connections using trained model, fall back to corpus frequency
    stage_set = set(predicted_stages.keys())
    import numpy as np
    doc_vec = np.zeros(1, dtype=np.float32)  # placeholder for GAT path
    connections = _predict_connections_xgb(doc_vec, stage_set)
    if not connections:
        # Fallback: corpus-frequency baseline
        for key, count in transition_counts.items():
            src, dst = key.split("|")
            if src in stage_set and dst in stage_set:
                connections.append({
                    "from": src,
                    "to": dst,
                    "confidence": round(count / max(len(docs), 1), 4),
                    "corpus_count": count,
                })

    # Topological ordering via in-degree (Kahn's algorithm)
    ordered = _topological_sort(stage_set, connections)

    similar_docs = [docs[idx]["doc_id"] for idx, _ in top_docs[:10]]

    # Attempt layout optimization with LayoutGAT
    positions = {}
    try:
        from layout_gat import layout_flowsheet
        positions = layout_flowsheet(
            model_path=None,  # uses default checkpoint path
            stages=ordered,
            connections=connections,
        )
    except Exception:
        pass  # fall back to empty positions (visualization uses spring_layout)

    return {
        "stages": ordered,
        "connections": connections,
        "stage_probabilities": dict(sorted(predicted_stages.items(),
                                           key=lambda x: x[1], reverse=True)),
        "similar_documents": similar_docs,
        "positions": positions,
        "input": {
            "head_grade": head_grade,
            "deposit_type": deposit_type,
            "ore_type": ore_type,
        },
    }


def _topological_sort(stages: set, connections: list) -> list:
    """Order stages by process flow using transition edges. Falls back to
    probability ranking for stages not reached by edges."""
    adjacency = defaultdict(list)
    in_degree = defaultdict(int)
    for stage in stages:
        in_degree[stage] = 0
    for conn in connections:
        adjacency[conn["from"]].append(conn["to"])
        in_degree[conn["to"]] += 1

    # Kahn's algorithm
    queue = [s for s in stages if in_degree[s] == 0]
    # Sort queue by corpus_count (most common first) for determinism
    queue.sort()
    ordered = []
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for neighbor in sorted(adjacency[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        queue.sort()

    # Add any remaining stages not reached (cycles or disconnected)
    remaining = [s for s in stages if s not in set(ordered)]
    ordered.extend(sorted(remaining))
    return ordered


# ---------------------------------------------------------------------------
# SageMaker entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GAT for flowsheet prediction")
    parser.add_argument("--graphs-bucket", type=str, required=True)
    parser.add_argument("--graphs-prefix", type=str, default="graphs/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)

    args = parser.parse_args()

    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    logger.info(f"Loading data from s3://{args.graphs_bucket}/{args.graphs_prefix}")
    data = load_training_data(args.graphs_bucket, args.graphs_prefix)

    logger.info("Starting training")
    t0 = time.time()
    metrics = train(data, epochs=args.epochs, lr=args.lr, patience=args.patience,
                    folds=args.folds, output_dir=output_dir)
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.0f}s")
    logger.info(f"Mean macro-F1: {metrics['aggregate'].get('mean_macro_f1', 'N/A')}")


if __name__ == "__main__":
    main()
