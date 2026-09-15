"""Feature-based document similarity for KNN-enriched prediction.

Computes weighted distances in the quantitative feature space, finds
nearest neighbors, and votes on stages + connections from their
ground-truth flowsheets.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Process-order reference, copied verbatim from
# classification/connection_predictor.py:40-51 to keep similarity.py self-contained.
_STAGE_ORDER = {s: i for i, s in enumerate([
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


def compute_feature_weights(feature_names: list[str]) -> np.ndarray:
    """Assign weights per feature. Primary-metal head grade gets higher weight
    than binary flags. `cu_grade` and `au_grade` are both treated as primary
    grades — pick whichever matches the deposit type you're predicting."""
    weights = np.ones(len(feature_names), dtype=np.float64)
    for i, name in enumerate(feature_names):
        if name in ("cu_grade", "au_grade"):
            weights[i] = 3.0  # grade is the strongest single predictor
        elif name.startswith("process_"):
            weights[i] = 2.0  # process flags directly indicate route
        elif name.startswith("ore_"):
            weights[i] = 1.5  # ore type strongly influences process
    return weights


def _prepare_distance_space(X_train: np.ndarray, X_query: np.ndarray,
                            feature_names: list[str]) -> tuple:
    """Map raw features into the exact space the KNN distance is computed in.

    Primary-metal grade (cu_grade/au_grade) is normalized to [0,1] on the
    training range so it's comparable to binary flags, then every column is
    scaled by sqrt(weight) so weighted-Euclidean distance == plain Euclidean
    on the returned matrices. Factored out of find_neighbors so the trace
    builder decomposes the *same* geometry the prediction uses.

    Returns:
        X_train_w, X_query_w: (N, F) matrices in the weighted-normalized space
        weights: (F,) per-feature weights (pre-sqrt)
    """
    weights = compute_feature_weights(feature_names)

    X_train_norm = X_train.astype(np.float64, copy=True)
    X_query_norm = X_query.astype(np.float64, copy=True)
    for i, name in enumerate(feature_names):
        if name in ("cu_grade", "au_grade"):
            lo = X_train[:, i].min()
            hi = X_train[:, i].max()
            if hi > lo:
                X_train_norm[:, i] = (X_train[:, i] - lo) / (hi - lo)
                X_query_norm[:, i] = np.clip(
                    (X_query[:, i] - lo) / (hi - lo), 0, 1)

    sqrt_w = np.sqrt(weights)
    return X_train_norm * sqrt_w, X_query_norm * sqrt_w, weights


def find_neighbors(X_train: np.ndarray, X_query: np.ndarray,
                   feature_names: list[str], k: int = 5) -> tuple:
    """Find k nearest training docs for each query doc.

    Uses weighted Euclidean distance. Cu grade is normalized to [0,1]
    range before distance computation so it's comparable to binary flags.

    Returns:
        indices: (N_query, k) array of training indices
        distances: (N_query, k) array of distances
    """
    X_train_w, X_query_w, _ = _prepare_distance_space(
        X_train, X_query, feature_names)
    dist_matrix = cdist(X_query_w, X_train_w, metric="euclidean")

    # Get k nearest
    k = min(k, X_train.shape[0])
    indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
    # Sort within the k nearest
    for i in range(len(indices)):
        order = np.argsort(dist_matrix[i, indices[i]])
        indices[i] = indices[i][order]

    distances = np.array([dist_matrix[i, indices[i]] for i in range(len(indices))])
    return indices, distances


def _neighbor_similarity_weights(distances: np.ndarray) -> np.ndarray:
    """Inverse-distance weights normalized to sum 1 — the exact weighting
    knn_vote_stages and score_connection_candidates use to turn neighbor
    distances into votes. Kept here so the trace reports the same numbers."""
    eps = 1e-6
    sim = 1.0 / (np.asarray(distances, dtype=np.float64) + eps)
    total = sim.sum()
    return sim / total if total > 0 else sim


def build_knn_trace(query_doc_id: str,
                    neighbor_indices: np.ndarray,
                    neighbor_distances: np.ndarray,
                    X_query_row: np.ndarray,
                    X_train: np.ndarray,
                    feature_names: list[str],
                    train_doc_ids: list[str],
                    y_train: np.ndarray,
                    stage_vocab: list[str],
                    train_connections: list[set],
                    predicted_stages: set | None = None,
                    predicted_edges: set | None = None,
                    min_stage_prob: float = 1e-9) -> dict:
    """Decompose one query's KNN prediction into an auditable decision trace.

    This is a pure re-derivation of the numbers already produced by
    find_neighbors + knn_vote_stages + score_connection_candidates — it
    changes no prediction. It exposes *why* each stage/edge was voted for:
    which neighbor projects contributed, with what weight, and which
    features made those projects near.

    Args:
        query_doc_id:       id of the doc being predicted.
        neighbor_indices:   (k,) train-array indices for this query's
                            neighbors, nearest first (a row of find_neighbors'
                            `indices`).
        neighbor_distances: (k,) matching weighted-Euclidean distances.
        X_query_row:        (F,) raw feature vector for the query doc.
        X_train:            (N_train, F) raw train features.
        feature_names:      (F,) feature column names.
        train_doc_ids:      (N_train,) train doc ids (index-aligned to X_train).
        y_train:            (N_train, S) binary stage presence for train docs.
        stage_vocab:        (S,) stage names.
        train_connections:  (N_train,) each a set of (src, dst) edges from that
                            doc's ground-truth flowsheet.
        predicted_stages:   the stage set the model actually emitted (marks
                            each voted stage `selected`); optional.
        predicted_edges:    the edge set the model actually emitted (marks each
                            voted edge `selected`); optional.
        min_stage_prob:     drop stages whose total vote is below this (default
                            keeps anything with a non-zero vote).

    Returns a JSON-serializable dict:
        {
          query_doc_id, k,
          neighbors: [{doc_id, rank, distance, similarity_weight,
                       feature_match: [{feature, query_value, neighbor_value,
                                        weight, contribution}]}],
          stage_votes: [{stage, probability, selected,
                         supporting_neighbors: [{doc_id, similarity_weight}]}],
          edge_votes: [{src, dst, score, selected,
                        supporting_neighbors: [{doc_id, similarity_weight}]}],
        }
    contribution is the per-feature share of squared distance
    (weight * normalized-diff^2); it sums across features to distance^2, so a
    small contribution means that feature made the two projects alike.
    """
    idx = np.asarray(neighbor_indices).astype(int)
    dist = np.asarray(neighbor_distances, dtype=np.float64)
    sim_w = _neighbor_similarity_weights(dist)

    # Same weighted-normalized geometry the distance was computed in, so the
    # per-feature contributions sum back to distance^2.
    X_train_w, X_query_w, weights = _prepare_distance_space(
        X_train, X_query_row.reshape(1, -1), feature_names)
    q_w = X_query_w[0]

    predicted_stages = predicted_stages or set()
    predicted_edges = predicted_edges or set()

    neighbors = []
    for rank, (train_i, d, w) in enumerate(zip(idx, dist, sim_w)):
        contribs = (q_w - X_train_w[train_i]) ** 2
        feature_match = [
            {
                "feature": feature_names[f],
                "query_value": float(X_query_row[f]),
                "neighbor_value": float(X_train[train_i, f]),
                "weight": float(weights[f]),
                "contribution": float(contribs[f]),
            }
            for f in range(len(feature_names))
        ]
        # Strongest agreements first: the features that made this a neighbor.
        feature_match.sort(key=lambda fm: fm["contribution"])
        neighbors.append({
            "doc_id": train_doc_ids[train_i],
            "rank": rank,
            "distance": float(d),
            "similarity_weight": float(w),
            "feature_match": feature_match,
        })

    # Stage votes — decomposition of knn_vote_stages: prob(stage) is the sum of
    # similarity weights of neighbors whose flowsheet contains that stage.
    stage_votes = []
    for j, stage in enumerate(stage_vocab):
        supporters = [
            {"doc_id": train_doc_ids[train_i], "similarity_weight": float(w)}
            for train_i, w in zip(idx, sim_w) if y_train[train_i, j] > 0
        ]
        prob = float(sum(s["similarity_weight"] for s in supporters))
        if prob <= min_stage_prob:
            continue
        stage_votes.append({
            "stage": stage,
            "probability": prob,
            "selected": stage in predicted_stages,
            "supporting_neighbors": supporters,
        })
    stage_votes.sort(key=lambda s: s["probability"], reverse=True)

    # Edge votes — decomposition of score_connection_candidates: only edges
    # whose endpoints are both in the predicted stage set are votable (that is
    # the exact gate the scorer applies).
    edge_votes = []
    if predicted_stages:
        edge_supporters: dict = {}
        for train_i, w in zip(idx, sim_w):
            for edge in train_connections[train_i]:
                src, dst = edge
                if src in predicted_stages and dst in predicted_stages:
                    edge_supporters.setdefault(edge, []).append(
                        {"doc_id": train_doc_ids[train_i],
                         "similarity_weight": float(w)})
        for (src, dst), supporters in edge_supporters.items():
            edge_votes.append({
                "src": src,
                "dst": dst,
                "score": float(sum(s["similarity_weight"] for s in supporters)),
                "selected": (src, dst) in predicted_edges,
                "supporting_neighbors": supporters,
            })
        edge_votes.sort(key=lambda e: e["score"], reverse=True)

    return {
        "query_doc_id": query_doc_id,
        "k": int(len(idx)),
        "neighbors": neighbors,
        "stage_votes": stage_votes,
        "edge_votes": edge_votes,
    }


def knn_vote_stages(indices: np.ndarray, distances: np.ndarray,
                    y_train: np.ndarray) -> np.ndarray:
    """Distance-weighted voting on stage presence from neighbors.

    Returns (N_query, S) array of probabilities [0, 1].
    """
    # Convert distance to similarity weight (inverse distance)
    # Add small epsilon to avoid division by zero for exact matches
    eps = 1e-6
    sim_weights = 1.0 / (distances + eps)
    # Normalize weights per query to sum to 1
    sim_weights = sim_weights / sim_weights.sum(axis=1, keepdims=True)

    n_query = indices.shape[0]
    n_stages = y_train.shape[1]
    probs = np.zeros((n_query, n_stages), dtype=np.float64)

    for i in range(n_query):
        neighbor_labels = y_train[indices[i]]  # (k, S)
        probs[i] = (sim_weights[i, :, np.newaxis] * neighbor_labels).sum(axis=0)

    return probs


def score_connection_candidates(indices: np.ndarray, distances: np.ndarray,
                                train_connections: list[set],
                                predicted_stages: set) -> dict:
    """Distance-weighted vote on edges from neighbor ground-truth flowsheets.

    Mirrors knn_vote_stages but for directed (src, dst) edges. Direction
    is preserved exactly as it appears in the neighbor flowsheet — recycle
    and backwards edges (e.g. flotation→regrind) are voted on identically
    to forward edges. Self-loops are kept: when multiple raw stage IDs
    consolidate to one canonical name, the connection between them
    legitimately becomes a self-loop in the canonical vocab.

    Returns dict of {(src, dst): score} with scores normalized to [0, 1]
    per query (sum of inverse-distance weights). Only emits edges where
    both endpoints are in predicted_stages.
    """
    eps = 1e-6
    sim_weights = 1.0 / (distances + eps)
    sim_weights = sim_weights / sim_weights.sum()

    edge_scores = {}
    for j, idx in enumerate(indices):
        for edge in train_connections[idx]:
            src, dst = edge
            if src in predicted_stages and dst in predicted_stages:
                edge_scores[edge] = edge_scores.get(edge, 0.0) + float(sim_weights[j])

    return edge_scores


def corpus_prior_scores(predicted_stages: set, transition_counts: dict,
                        joint_support: dict) -> dict:
    """Per-edge corpus prior P(edge | both stages co-occur in same doc) ∈ [0, 1].

    joint_support[(src, dst)] = number of training docs containing both
    stages. This is the correct denominator for the conditional probability;
    the previous min(stage_support[src], stage_support[dst]) was an upper
    bound on joint occurrence and depressed the prior for forward edges
    where both endpoints are common but rarely both appear in the same doc.

    Direction-preserving — recycles and self-loops included. Self-loops
    are kept because when multiple raw stage IDs consolidate to one
    canonical name, the connection between them legitimately becomes a
    self-loop in the canonical vocab.

    Returned as raw normalized scores so the caller can blend with KNN
    instead of hard-thresholding.
    """
    prior = {}
    for key, count in transition_counts.items():
        src, dst = key.split("|")
        if src not in predicted_stages or dst not in predicted_stages:
            continue
        denom = joint_support.get((src, dst), 0)
        if denom <= 0:
            continue
        prior[(src, dst)] = min(1.0, count / denom)
    return prior


def blend_connection_scores(knn_scores: dict, prior_scores: dict,
                            alpha: float) -> dict:
    """Blend KNN edge scores with a corpus prior.

    Mirror of blend_predictions() for the edge dict format. alpha=1.0
    is pure KNN, alpha=0.0 is pure prior. Edges present in either input
    are kept (missing side contributes 0).
    """
    if alpha >= 1.0:
        return dict(knn_scores)
    if alpha <= 0.0:
        return dict(prior_scores)

    keys = set(knn_scores.keys()) | set(prior_scores.keys())
    return {
        edge: alpha * knn_scores.get(edge, 0.0)
              + (1.0 - alpha) * prior_scores.get(edge, 0.0)
        for edge in keys
    }


def threshold_connections(scores: dict, threshold: float,
                          recycle_threshold: float | None = None) -> set:
    """Final binary cut on edge scores. Returns set of (src, dst) tuples.

    If recycle_threshold is provided, forward edges (src.order < dst.order
    in _STAGE_ORDER) use `threshold` and recycle/loop edges (src.order >=
    dst.order) use `recycle_threshold`. Stages not in _STAGE_ORDER are
    treated as forward (lenient).
    """
    if recycle_threshold is None:
        return {edge for edge, score in scores.items() if score >= threshold}

    max_order = max(_STAGE_ORDER.values())
    out = set()
    for (src, dst), score in scores.items():
        src_o = _STAGE_ORDER.get(src, max_order // 2)
        dst_o = _STAGE_ORDER.get(dst, max_order // 2 + 1)
        cutoff = recycle_threshold if src_o >= dst_o else threshold
        if score >= cutoff:
            out.add((src, dst))
    return out


def blend_predictions(xgb_probs: np.ndarray, knn_probs: np.ndarray,
                      alpha: float = 0.5) -> np.ndarray:
    """Blend XGBoost and KNN probabilities.

    Args:
        xgb_probs: (N, S) XGBoost probabilities
        knn_probs: (N, S) KNN vote probabilities
        alpha: weight for XGBoost (1-alpha for KNN)

    Returns (N, S) blended probabilities.
    """
    return alpha * xgb_probs + (1.0 - alpha) * knn_probs
