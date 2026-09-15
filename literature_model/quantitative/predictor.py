"""Quantitative flowsheet predictor: clean features -> stage predictions.

Builds up from Cu grade (matching the mini model) and adds features
incrementally. Loads graph data from S3 and Cu grade from a local
grade_map.json cache (built from pipeline results).

Usage:
    python quantitative/predictor.py --graphs-bucket BUCKET [--features cu_grade,ore_type,...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier

# Import shared utilities from classification/stage_predictor.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classification"))
from stage_predictor import _load_graphs_from_s3, reduce_stage_vocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GRADE_MAP_PATH = Path(__file__).parent / "grade_map.json"
AU_GRADE_MAP_PATH = Path(__file__).parent / "grade_map_au.json"
MANDATORY_STAGES = {"crusher", "mill", "flotation"}

# Core feature set — the validated baseline. Override with --features only
# when intentionally running an ablation.
CORE_FEATURES = ["cu_grade", "ore_type", "deposit_type", "process_flags",
                 "mining_method", "grades", "recovery", "tonnage", "economics",
                 "comminution"]

# ---------------------------------------------------------------------------
# Feature groups — each can be toggled on/off independently
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "cu_grade": ["cu_grade"],
    "au_grade": ["au_grade"],
    "ore_type": ["ore_oxide", "ore_sulfide", "ore_supergene", "ore_hypogene",
                 "ore_transition", "ore_refractory", "ore_free_milling"],
    "deposit_type": ["dep_porphyry", "dep_skarn", "dep_vms", "dep_iocg",
                     "dep_sedimentary", "dep_epithermal"],
    "mining_method": ["mining_open_pit", "mining_underground"],
    "process_flags": ["process_flotation", "process_leach"],
    # Targeted numeric features from enriched graphs
    "grades": ["grade_cu_mean", "grade_au_mean", "grade_ag_mean"],
    "recovery": ["recovery_mean"],
    "tonnage": ["tonnage_mean"],
    "economics": ["npv_mean", "capex_mean", "opex_mean"],
    "comminution": ["grind_size_mean"],
}

# Keyword lookup: (section_group, keyword) for each binary feature
_BINARY_LOOKUP = {
    "ore_oxide": ("geology", "oxide"),
    "ore_sulfide": [("geology", "sulfide"), ("geology", "sulphide")],
    "ore_supergene": ("geology", "supergene"),
    "ore_hypogene": ("geology", "hypogene"),
    "ore_transition": ("geology", "transition"),
    "ore_refractory": ("geology", "refractory"),
    "ore_free_milling": ("geology", "free_milling"),
    "dep_porphyry": ("geology", "porphyry"),
    "dep_skarn": ("geology", "skarn"),
    "dep_vms": ("geology", "vms"),
    "dep_iocg": ("geology", "iocg"),
    "dep_sedimentary": ("geology", "sedimentary"),
    "dep_epithermal": ("geology", "epithermal"),
    "mining_open_pit": ("mining_method", "open_pit"),
    "mining_underground": ("mining_method", "underground"),
    "process_flotation": ("metallurgical_testing", "flotation"),
    "process_leach": ("metallurgical_testing", "leach"),
}

# Targeted numeric features: feature_name -> (node_feature_key, section_fallback_order)
# Try metallurgical_testing first (most relevant), fall back to resource_estimate, then geology
_NUMERIC_LOOKUP = {
    "grade_cu_mean": ("grade_cu_mean", ["metallurgical_testing", "resource_estimate", "geology", "summary"]),
    "grade_au_mean": ("grade_au_mean", ["metallurgical_testing", "resource_estimate", "geology", "summary"]),
    "grade_ag_mean": ("grade_ag_mean", ["metallurgical_testing", "resource_estimate", "geology", "summary"]),
    "recovery_mean": ("recovery_mean", ["metallurgical_testing", "resource_estimate", "summary"]),
    "tonnage_mean": ("tonnage_mean", ["resource_estimate", "mining_method", "metallurgical_testing", "summary"]),
    "grind_size_mean": ("grind_size_mean", ["metallurgical_testing", "mining_method"]),
    "npv_mean": ("npv_mean", ["economics", "summary"]),
    "capex_mean": ("capex_mean", ["economics", "summary"]),
    "opex_mean": ("opex_mean", ["economics", "mining_method"]),
}


def _normalize_value(name: str, val: float) -> float:
    """Normalize extracted values to consistent units.

    Raw table values mix units (tonnes vs Mt, $ vs $M). Normalizations
    inferred from distribution analysis of 931 enriched graphs:

    Tonnage: normalize to Mt (millions of tonnes)
      - 0-2000: likely already Mt or small kt, leave as-is
      - 2K-50K: likely kt (throughput), divide by 1,000
      - 50K+: likely actual tonnes, divide by 1,000,000

    NPV/CAPEX: normalize to $M
      - 0-10: likely $B, multiply by 1,000
      - 10-10K: likely $M already, leave as-is
      - 10K+: likely $K or actual $, divide by 1,000

    OPEX: normalize to $/t
      - 0-1000: likely $/t or $M/yr, leave as-is
      - 1000+: likely $K/yr or actual $, divide by 1,000

    Grades, recovery, grind_size: already clean from regex range filters.
    """
    if val == 0.0:
        return 0.0

    if "tonnage" in name:
        if val >= 50_000:
            return val / 1_000_000.0
        elif val >= 2_000:
            return val / 1_000.0
        return val

    if "npv" in name or "capex" in name:
        if val < 10:
            return val * 1_000.0
        elif val > 10_000:
            return val / 1_000.0
        return val

    if "opex" in name:
        if val > 1_000:
            return val / 1_000.0
        return val

    return val


def get_feature_names(groups: list[str]) -> list[str]:
    """Build ordered feature name list from selected groups."""
    names = []
    for g in groups:
        names.extend(FEATURE_GROUPS[g])
    return names


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Fallback chain for grade_cu_mean — same order as _NUMERIC_LOOKUP entry
_CU_GRADE_FALLBACK_SECTIONS = (
    "metallurgical_testing", "resource_estimate", "geology", "summary",
)

# Au falls back slightly differently: head grade lives in met-testing /
# recovery_methods for gold projects, resource_estimate only as a last resort.
_AU_GRADE_FALLBACK_SECTIONS = (
    "metallurgical_testing", "recovery_methods",
    "resource_estimate", "geology", "summary",
)

# Au head-grade band used when building au_grade feature. Matches the filter
# from quantitative/build_au_grade_map.py so the feature and the cached map
# agree on what counts as a "valid" gold head grade.
AU_MIN_GRADE = 0.3
AU_MAX_GRADE = 30.0


def _index_sections(graph: dict) -> dict:
    """Index a graph's context nodes by section group."""
    sections = {}
    for node in graph.get("nodes", []):
        if node.get("type") == "context":
            sections[node["group"]] = node.get("features", {})
    return sections


def resolve_cu_grade(doc_id: str, sections: dict, grade_map: dict) -> float | None:
    """Resolve Cu grade from the static cache, falling back to the graph's
    enriched grade_cu_mean when the cache misses.

    The static grade_map.json comes from pipeline/pdf_extraction.extract_copper_grades,
    which is strict and returns None for ~32 labeled docs. The S3 graphs have
    grade_cu_mean from a more permissive table extractor — use it as a backfill
    so we don't drop training data unnecessarily.

    Returns None only if both sources fail.
    """
    grade = grade_map.get(doc_id)
    if grade is not None and grade > 0:
        return float(grade)
    for section in _CU_GRADE_FALLBACK_SECTIONS:
        val = sections.get(section, {}).get("grade_cu_mean", 0.0)
        if val and val > 0:
            return float(val)
    return None


def resolve_au_grade(doc_id: str, sections: dict, grade_map: dict) -> float | None:
    """Resolve Au head grade (g/t). Mirrors resolve_cu_grade.

    grade_map is expected to be the output of build_au_grade_map.py, which is
    already pre-filtered to AU_MIN_GRADE ≤ g ≤ AU_MAX_GRADE. On a cache miss we
    fall back to the enriched graph's grade_au_mean, but only if it lands in
    the same evaluable window — this keeps concentrate / oz-t values from
    silently polluting the training data.
    """
    grade = grade_map.get(doc_id)
    if grade is not None and grade > 0:
        return float(grade)
    for section in _AU_GRADE_FALLBACK_SECTIONS:
        val = sections.get(section, {}).get("grade_au_mean", 0.0)
        if val and AU_MIN_GRADE <= val <= AU_MAX_GRADE:
            return float(val)
    return None


def extract_features(graph: dict, feature_names: list[str],
                     grade_map: dict) -> np.ndarray:
    """Extract features for a single document graph."""
    sections = _index_sections(graph)
    doc_id = graph.get("document_id", "")
    vec = np.zeros(len(feature_names), dtype=np.float64)

    for i, name in enumerate(feature_names):
        if name == "cu_grade":
            grade = resolve_cu_grade(doc_id, sections, grade_map)
            vec[i] = grade if grade is not None else 0.0
        elif name == "au_grade":
            grade = resolve_au_grade(doc_id, sections, grade_map)
            vec[i] = grade if grade is not None else 0.0
        elif name in _NUMERIC_LOOKUP:
            feat_key, fallback_sections = _NUMERIC_LOOKUP[name]
            val = 0.0
            for section in fallback_sections:
                val = sections.get(section, {}).get(feat_key, 0.0)
                if val != 0.0:
                    break
            vec[i] = _normalize_value(name, val)
        elif name in _BINARY_LOOKUP:
            lookup = _BINARY_LOOKUP[name]
            if isinstance(lookup, list):
                for section, kw in lookup:
                    if sections.get(section, {}).get(f"kw_{kw}", 0) > 0:
                        vec[i] = 1.0
                        break
            else:
                section, kw = lookup
                if sections.get(section, {}).get(f"kw_{kw}", 0) > 0:
                    vec[i] = 1.0

    return vec


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_grade_map(mineral: str = "cu") -> dict:
    """Load head-grade mapping from local cache. mineral ∈ {"cu", "au"}."""
    path = AU_GRADE_MAP_PATH if mineral == "au" else GRADE_MAP_PATH
    if not path.exists():
        logger.warning(f"Grade map not found at {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def load_graphs(graphs_bucket: str, graphs_prefix: str = "graphs/",
                min_support: int = 3, vocab_version: str = "v1") -> dict:
    """Download graphs from S3 once. Returns cached data for build_features()."""
    graphs = _load_graphs_from_s3(graphs_bucket, graphs_prefix)
    stage_vocab, raw_to_canonical = reduce_stage_vocab(graphs, min_support,
                                                       vocab_version=vocab_version)
    return {
        "graphs": graphs,
        "stage_vocab": stage_vocab,
        "raw_to_canonical": raw_to_canonical,
    }


def build_features(graph_data: dict, feature_groups: list[str] | None = None) -> dict:
    """Extract features + labels from cached graphs. No S3 calls."""
    if feature_groups is None:
        feature_groups = ["cu_grade"]

    graphs = graph_data["graphs"]
    stage_vocab = graph_data["stage_vocab"]
    raw_to_canonical = graph_data["raw_to_canonical"]
    stage_to_idx = {s: i for i, s in enumerate(stage_vocab)}

    feature_names = get_feature_names(feature_groups)
    # Pick the right cached grade map for the primary-grade feature. au_grade
    # takes precedence if both are requested — mixing both is unusual and if
    # the caller really wants both, they should load the maps themselves.
    if "au_grade" in feature_groups:
        grade_map = load_grade_map("au")
        primary_grade = "au_grade"
    elif "cu_grade" in feature_groups:
        grade_map = load_grade_map("cu")
        primary_grade = "cu_grade"
    else:
        grade_map = {}
        primary_grade = None

    X_list, y_binary_list, y_counts_list, doc_ids = [], [], [], []
    all_transitions = []
    doc_connections = []
    skipped_no_grade = 0
    recovered_from_graph = 0

    for graph in graphs:
        doc_id = graph.get("document_id", "unknown")

        stage_counter = defaultdict(int)
        for node in graph.get("nodes", []):
            if node.get("type") != "stage":
                continue
            raw_id = node.get("stage_id", node["id"].replace("stg_", ""))
            canonical = raw_to_canonical.get(raw_id)
            if canonical and canonical in stage_to_idx:
                stage_counter[canonical] += 1

        if not stage_counter:
            continue

        if primary_grade:
            sections = _index_sections(graph)
            resolver = resolve_au_grade if primary_grade == "au_grade" else resolve_cu_grade
            grade = resolver(doc_id, sections, grade_map)
            if grade is None:
                skipped_no_grade += 1
                continue
            if not (grade_map.get(doc_id) and grade_map[doc_id] > 0):
                recovered_from_graph += 1

        vec = extract_features(graph, feature_names, grade_map)
        X_list.append(vec)
        doc_ids.append(doc_id)

        label_binary = np.zeros(len(stage_vocab), dtype=np.float32)
        label_counts = np.zeros(len(stage_vocab), dtype=np.float32)
        for stage, count in stage_counter.items():
            idx = stage_to_idx[stage]
            label_binary[idx] = 1.0
            label_counts[idx] = float(count)
        y_binary_list.append(label_binary)
        y_counts_list.append(label_counts)

        doc_edges = set()
        for edge in graph.get("edges", []):
            if edge.get("type") != "stage_transition":
                continue
            src_raw = edge["source"].replace("stg_", "")
            dst_raw = edge["target"].replace("stg_", "")
            src_can = raw_to_canonical.get(src_raw)
            dst_can = raw_to_canonical.get(dst_raw)
            if src_can and dst_can and src_can in stage_to_idx and dst_can in stage_to_idx:
                all_transitions.append((src_can, dst_can))
                doc_edges.add((src_can, dst_can))
        doc_connections.append(doc_edges)

    X = np.array(X_list, dtype=np.float64)
    y_binary = np.array(y_binary_list, dtype=np.float32)
    y_counts = np.array(y_counts_list, dtype=np.float32)

    # Log-compress targeted numeric features (wide value ranges even after normalization)
    for i, name in enumerate(feature_names):
        if name in _NUMERIC_LOOKUP:
            X[:, i] = np.sign(X[:, i]) * np.log1p(np.abs(X[:, i]))
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    coverage = {}
    for i, name in enumerate(feature_names):
        nonzero = int((X[:, i] != 0).sum())
        coverage[name] = nonzero / len(X) if len(X) > 0 else 0.0

    transition_counts = defaultdict(int)
    for src, dst in all_transitions:
        transition_counts[f"{src}|{dst}"] += 1

    if skipped_no_grade > 0 or recovered_from_graph > 0:
        label = primary_grade.replace("_grade", "").upper() if primary_grade else "grade"
        logger.info(
            f"{label} grade: {recovered_from_graph} recovered from graph fallback, "
            f"{skipped_no_grade} still missing (dropped)"
        )
    logger.info(f"{X.shape[0]} docs, {len(feature_names)} features ({','.join(feature_groups)})")

    return {
        "X": X,
        "y_binary": y_binary,
        "y_counts": y_counts,
        "stage_vocab": stage_vocab,
        "feature_names": feature_names,
        "doc_ids": doc_ids,
        "doc_connections": doc_connections,
        "transition_counts": dict(transition_counts),
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Train / val split (test set never returned)
# ---------------------------------------------------------------------------

def split_data(X, y_binary, y_counts, doc_ids, doc_connections,
               val_pct=0.2, test_pct=0.1, random_state=42) -> dict:
    """70/20/10 train/val/test split. Returns only train + val."""
    train_val_idx, _ = train_test_split(
        np.arange(len(X)),
        test_size=test_pct,
        random_state=random_state,
    )
    val_fraction = val_pct / (1.0 - test_pct)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=random_state,
    )

    def _subset(indices):
        return {
            "X": X[indices],
            "y_binary": y_binary[indices],
            "y_counts": y_counts[indices],
            "doc_ids": [doc_ids[i] for i in indices],
            "doc_connections": [doc_connections[i] for i in indices],
        }

    result = {"train": _subset(train_idx), "val": _subset(val_idx)}
    logger.info(f"Split: {len(train_idx)} train, {len(val_idx)} val "
                f"(test held out: {len(X) - len(train_val_idx)})")
    return result


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def fit_polynomial(X_train, y_train, degree=2) -> dict:
    """Polynomial regression (multi-output) matching the mini model."""
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    r2 = model.score(X_train, y_train)
    logger.info(f"Polynomial (degree={degree}) R²={r2:.3f}")
    return {"model": model, "type": "polynomial"}


def fit_xgboost(X_train, y_train, stage_vocab) -> dict:
    """Train per-stage XGBoost binary classifiers."""
    models = []
    for i, stage in enumerate(stage_vocab):
        y_col = y_train[:, i]
        n_pos = int(y_col.sum())
        n_neg = len(y_col) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        clf = XGBClassifier(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        clf.fit(X_train, y_col)
        models.append(clf)

    logger.info(f"Trained {len(models)} XGBoost classifiers")
    return {"models": models, "type": "xgboost", "stage_vocab": stage_vocab}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model_dict: dict, X: np.ndarray) -> np.ndarray:
    """Predict stage values. Returns (N, S) array."""
    if model_dict["type"] == "polynomial":
        return model_dict["model"].predict(X)
    else:
        models = model_dict["models"]
        probs = np.zeros((X.shape[0], len(models)), dtype=np.float64)
        for i, clf in enumerate(models):
            probs[:, i] = clf.predict_proba(X)[:, 1]
        return probs


def postprocess_predictions(raw: np.ndarray, stage_vocab: list[str],
                            model_type: str, threshold: float = 0.5
                            ) -> np.ndarray:
    """Convert raw predictions to binary predictions."""
    if model_type == "polynomial":
        # Polynomial outputs counts — convert to binary presence
        y_pred = (np.round(raw).clip(min=0) > 0).astype(np.float32)
    else:
        y_pred = (raw >= threshold).astype(np.float32)

    # Enforce mandatory stages
    for j, stage in enumerate(stage_vocab):
        if stage in MANDATORY_STAGES:
            y_pred[:, j] = np.maximum(y_pred[:, j], 1.0)

    return y_pred


def filter_connections(candidate_edges: dict, predicted_stages: set,
                       stage_support: dict, max_outgoing: int = 2,
                       min_cooccurrence_ratio: float = 0.30) -> set:
    """Filter candidate edges using co-occurrence ratio and outgoing cap.

    Args:
        candidate_edges: {(src, dst): corpus_count} for edges where both stages present
        predicted_stages: set of stage names predicted for this doc
        stage_support: {stage: n_docs_with_stage} from training corpus
        max_outgoing: max outgoing edges per stage
        min_cooccurrence_ratio: edge must appear in >= this fraction of docs
            that contain both src and dst stages
    """
    # Filter by co-occurrence ratio
    ratio_filtered = {}
    for (src, dst), count in candidate_edges.items():
        both_support = min(stage_support.get(src, 1), stage_support.get(dst, 1))
        ratio = count / both_support if both_support > 0 else 0
        if ratio >= min_cooccurrence_ratio:
            ratio_filtered[(src, dst)] = count

    # Cap outgoing edges per stage (keep top-N by corpus count)
    from collections import defaultdict
    outgoing = defaultdict(list)
    for (src, dst), count in ratio_filtered.items():
        outgoing[src].append((dst, count))

    final_edges = set()
    for src, dsts in outgoing.items():
        top = sorted(dsts, key=lambda x: -x[1])[:max_outgoing]
        for dst, _ in top:
            final_edges.add((src, dst))

    return final_edges


def predict_connections(y_pred_binary: np.ndarray, stage_vocab: list[str],
                        transition_counts: dict, stage_support: dict = None,
                        min_corpus_freq: int = 3, max_outgoing: int = 2,
                        min_cooccurrence_ratio: float = 0.30
                        ) -> list[set]:
    """Predict connections using filtered corpus-frequency baseline."""
    result = []
    for i in range(y_pred_binary.shape[0]):
        predicted_stages = {stage_vocab[j] for j in range(len(stage_vocab))
                           if y_pred_binary[i, j] > 0}

        # Build candidate edges
        candidates = {}
        for key, count in transition_counts.items():
            if count >= min_corpus_freq:
                src, dst = key.split("|")
                if src in predicted_stages and dst in predicted_stages:
                    candidates[(src, dst)] = count

        if stage_support:
            doc_edges = filter_connections(candidates, predicted_stages,
                                           stage_support, max_outgoing,
                                           min_cooccurrence_ratio)
        else:
            doc_edges = set(candidates.keys())

        result.append(doc_edges)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_experiment(graph_data, feature_groups, model_type, threshold=0.5,
                   k=5, alpha=0.5, edge_threshold=0.3, edge_alpha=1.0,
                   edge_recycle_threshold=None):
    """Run a single experiment: build features, split, train, eval. Returns metrics."""
    from quantitative.eval import (
        evaluate_stages, evaluate_connections, evaluate_ordering,
    )
    from quantitative.similarity import (
        find_neighbors, knn_vote_stages, blend_predictions,
        score_connection_candidates, corpus_prior_scores,
        blend_connection_scores, threshold_connections,
    )

    data = build_features(graph_data, feature_groups)
    splits = split_data(data["X"], data["y_binary"], data["y_counts"],
                        data["doc_ids"], data["doc_connections"])

    train = splits["train"]
    val = splits["val"]

    # KNN component (always computed for blended, also used standalone)
    indices, distances = find_neighbors(train["X"], val["X"],
                                        data["feature_names"], k=k)
    knn_probs = knn_vote_stages(indices, distances, train["y_binary"])

    if model_type == "knn":
        val_raw = knn_probs
    elif model_type == "blended":
        xgb_model = fit_xgboost(train["X"], train["y_binary"], data["stage_vocab"])
        xgb_probs = predict(xgb_model, val["X"])
        val_raw = blend_predictions(xgb_probs, knn_probs, alpha=alpha)
    elif model_type == "polynomial":
        model = fit_polynomial(train["X"], train["y_binary"])
        val_raw = predict(model, val["X"])
    else:
        model = fit_xgboost(train["X"], train["y_binary"], data["stage_vocab"])
        val_raw = predict(model, val["X"])

    effective_type = "xgboost" if model_type in ("xgboost", "blended", "knn") else model_type
    val_pred = postprocess_predictions(val_raw, data["stage_vocab"],
                                       effective_type, threshold)

    # Edge prediction — KNN scoring path used for ALL model types so we get
    # apples-to-apples comparison. Stage predictor decides which stages to
    # include; this layer wires them up using neighbor flowsheets. Direction
    # is preserved exactly as in the neighbor (recycle edges included).
    stage_vocab = data["stage_vocab"]

    # Train-only corpus stats (was previously computed over the full set in
    # build_features, which leaked val/test edges into the prior).
    train_transition_counts = defaultdict(int)
    for doc_edges in train["doc_connections"]:
        for src, dst in doc_edges:
            train_transition_counts[f"{src}|{dst}"] += 1

    # Joint support: number of train docs where BOTH stages appear. This is
    # the correct denominator for P(edge | both stages co-occur).
    y_train = train["y_binary"]
    joint_support = {}
    for j1, s1 in enumerate(stage_vocab):
        s1_mask = y_train[:, j1] > 0
        for j2, s2 in enumerate(stage_vocab):
            joint_support[(s1, s2)] = int((s1_mask & (y_train[:, j2] > 0)).sum())

    val_conn_pred = []
    val_pred_stage_sets = []
    for i in range(val_pred.shape[0]):
        predicted = {stage_vocab[j] for j in range(len(stage_vocab))
                     if val_pred[i, j] > 0}
        val_pred_stage_sets.append(predicted)
        knn_scores = score_connection_candidates(
            indices[i], distances[i], train["doc_connections"], predicted)
        if edge_alpha < 1.0:
            prior = corpus_prior_scores(predicted, train_transition_counts,
                                        joint_support)
            scores = blend_connection_scores(knn_scores, prior, edge_alpha)
        else:
            scores = knn_scores
        val_conn_pred.append(threshold_connections(
            scores, edge_threshold, recycle_threshold=edge_recycle_threshold))

    stage_metrics = evaluate_stages(val["y_binary"], val_pred,
                                    data["stage_vocab"])
    conn_metrics = evaluate_connections(val_conn_pred, val["doc_connections"],
                                       data["stage_vocab"])
    true_stage_sets = [
        {stage_vocab[j] for j in range(len(stage_vocab)) if val["y_binary"][i, j] > 0}
        for i in range(val_pred.shape[0])
    ]
    ordering_metrics = evaluate_ordering(
        val_conn_pred, val["doc_connections"],
        val_pred_stage_sets, true_stage_sets,
    )
    return stage_metrics, conn_metrics, ordering_metrics, data


V2_CHECKPOINT_DIR = (Path(__file__).resolve().parent.parent
                     / "classification" / "xgb_checkpoint_v2")


def load_v2_checkpoint_predictions(checkpoint_dir: Path = V2_CHECKPOINT_DIR) -> dict:
    """Load precomputed V2 stage predictions from oof/val/test JSON files.

    Returns:
        {
            "stage_vocab": sorted list of 23 V2 stages,
            "preds": {doc_id: {"split", "predicted", "ground_truth"}},
        }
    Predicted/ground_truth are sets of stage names.
    """
    with open(checkpoint_dir / "stage_vocab.json") as f:
        stage_vocab = json.load(f)

    preds = {}
    for split, fname in [("oof", "oof_predictions.json"),
                         ("val", "val_predictions.json"),
                         ("test", "test_predictions.json")]:
        path = checkpoint_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        per_doc = data["per_doc"] if isinstance(data, dict) else data
        for entry in per_doc:
            doc_id = entry["doc_id"]
            preds[doc_id] = {
                "split": split,
                "predicted": set(entry.get("xgb_stages", [])),
                "ground_truth": set(entry.get("ground_truth", [])),
            }
    logger.info(f"Loaded V2 predictions: "
                f"{sum(1 for p in preds.values() if p['split'] == 'oof')} oof / "
                f"{sum(1 for p in preds.values() if p['split'] == 'val')} val / "
                f"{sum(1 for p in preds.values() if p['split'] == 'test')} test")
    return {"stage_vocab": sorted(stage_vocab), "preds": preds}


def build_v2_doc_edges(graphs: list[dict], v2_stage_set: set) -> dict:
    """Map ground-truth graph edges to the V2 checkpoint's 23-stage vocab.

    The V2 checkpoint uses the V1 canonical mapping (CANONICAL_STAGES in
    classification/stage_predictor.py) — directory naming is misleading.
    """
    _, raw_to_canonical = reduce_stage_vocab(graphs, min_support=1, vocab_version="v1")

    doc_edges = {}
    for graph in graphs:
        doc_id = graph.get("document_id", "")
        edges = set()
        for edge in graph.get("edges", []):
            if edge.get("type") != "stage_transition":
                continue
            src_raw = edge["source"].replace("stg_", "")
            dst_raw = edge["target"].replace("stg_", "")
            src_can = raw_to_canonical.get(src_raw)
            dst_can = raw_to_canonical.get(dst_raw)
            if src_can in v2_stage_set and dst_can in v2_stage_set:
                edges.add((src_can, dst_can))
        doc_edges[doc_id] = edges
    return doc_edges


def run_v2_substrate_experiment(v2_data: dict, doc_edges: dict,
                                 edge_threshold: float = 0.30,
                                 edge_recycle_threshold: float | None = None,
                                 ) -> tuple[dict, dict]:
    """Edge prediction sitting on top of the V2 checkpoint stage predictions.

    Replaces the local stage classifier with the V2 checkpoint's actual
    predictions (val macro-F1=0.86). Train-only corpus stats come from the
    OOF split; val docs are scored with corpus_prior_scores -> threshold.
    """
    from quantitative.eval import evaluate_connections, evaluate_ordering
    from quantitative.similarity import corpus_prior_scores, threshold_connections

    stage_vocab = v2_data["stage_vocab"]
    v2_stage_set = set(stage_vocab)
    preds = v2_data["preds"]

    train_ids = [d for d, p in preds.items() if p["split"] == "oof"]
    val_ids = [d for d, p in preds.items() if p["split"] == "val"]

    # Train-only transition counts (from ground truth edges of OOF docs)
    train_transition_counts = defaultdict(int)
    for doc_id in train_ids:
        for src, dst in doc_edges.get(doc_id, set()):
            train_transition_counts[f"{src}|{dst}"] += 1

    # Joint support: train docs where both stages are in ground truth
    joint_support = {}
    train_gt_sets = [preds[d]["ground_truth"] for d in train_ids]
    for s1 in stage_vocab:
        for s2 in stage_vocab:
            joint_support[(s1, s2)] = sum(
                1 for gt in train_gt_sets if s1 in gt and s2 in gt
            )

    val_conn_pred, val_conn_true = [], []
    val_pred_stage_sets, val_true_stage_sets = [], []
    for doc_id in val_ids:
        predicted = preds[doc_id]["predicted"] & v2_stage_set
        scores = corpus_prior_scores(predicted, train_transition_counts, joint_support)
        edges = threshold_connections(
            scores, edge_threshold, recycle_threshold=edge_recycle_threshold,
        )
        val_conn_pred.append(edges)
        val_conn_true.append(doc_edges.get(doc_id, set()))
        val_pred_stage_sets.append(predicted)
        val_true_stage_sets.append(preds[doc_id]["ground_truth"] & v2_stage_set)

    conn_metrics = evaluate_connections(val_conn_pred, val_conn_true, stage_vocab)
    ordering_metrics = evaluate_ordering(
        val_conn_pred, val_conn_true, val_pred_stage_sets, val_true_stage_sets,
    )
    return conn_metrics, ordering_metrics


def run_v2_substrate_sweep(v2_data: dict, doc_edges: dict, args):
    """Sweep edge_threshold (and optionally recycle_threshold) on V2 substrate.

    Optimization target is reach_f1 — the directed flow correctness metric. The
    columns also show edge_f1 and tau for context, but ranking is by reach_f1.
    """
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    recycle_options = [None, 0.40, 0.50, 0.60]

    rows = []
    for t in thresholds:
        for rt in recycle_options:
            conn_m, ord_m = run_v2_substrate_experiment(
                v2_data, doc_edges, edge_threshold=t, edge_recycle_threshold=rt,
            )
            rows.append((
                t, rt,
                ord_m["reach_f1"],
                ord_m["reach_precision"],
                ord_m["reach_recall"],
                conn_m["mean_f1"],
                ord_m["kendall_tau"],
            ))

    rows.sort(key=lambda r: -r[2])
    print(f"\nV2 Substrate Sweep — corpus prior on V2 stage predictions "
          f"({len(thresholds)} thresholds x {len(recycle_options)} recycle thresholds)")
    print(f"Ranked by reach_f1 (directed flow correctness)\n")
    print(f"{'thr':>5} {'recyc':>6} {'reach_f1':>9} {'rprec':>7} {'rrec':>7} "
          f"{'edge_f1':>8} {'tau':>7}")
    print("-" * 56)
    for t, rt, rf1, rp, rr, ef1, tau in rows:
        rt_str = "—" if rt is None else f"{rt:.2f}"
        print(f"{t:>5.2f} {rt_str:>6} {rf1:>9.3f} {rp:>7.3f} {rr:>7.3f} "
              f"{ef1:>8.3f} {tau:>7.3f}")
    best = rows[0]
    rt_str = "—" if best[1] is None else f"{best[1]:.2f}"
    print(f"\nBest: threshold={best[0]:.2f}, recycle={rt_str}  "
          f"(reach_f1={best[2]:.3f}, prec={best[3]:.3f}, rec={best[4]:.3f}, "
          f"edge_f1={best[5]:.3f})")


def run_edge_sweep(graph_data, args):
    """Sweep --edge-threshold x --edge-alpha on the local-model substrate.

    Optimization target is reach_f1 (directed flow correctness). Kept for
    apples-to-apples comparison with the V2 substrate sweep — local-model
    predictions are usually weaker, so this is a baseline reference.
    """
    feature_groups = [g.strip() for g in args.features.split(",")]
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    alphas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

    rows = []
    print(f"\nSweeping {len(thresholds)} thresholds x {len(alphas)} alphas "
          f"= {len(thresholds) * len(alphas)} runs ({args.model}, {args.features})\n")
    for t in thresholds:
        for a in alphas:
            stage_m, conn_m, ord_m, _ = run_experiment(
                graph_data, feature_groups, args.model, args.threshold,
                args.k, args.alpha, edge_threshold=t, edge_alpha=a,
                edge_recycle_threshold=args.edge_recycle_threshold)
            rows.append((
                t, a,
                ord_m["reach_f1"],
                ord_m["reach_precision"],
                ord_m["reach_recall"],
                conn_m["mean_f1"],
                ord_m["kendall_tau"],
            ))

    rows.sort(key=lambda r: -r[2])
    print(f"Ranked by reach_f1 (directed flow correctness)\n")
    print(f"{'thr':>5} {'alpha':>6} {'reach_f1':>9} {'rprec':>7} {'rrec':>7} "
          f"{'edge_f1':>8} {'tau':>7}")
    print("-" * 56)
    for t, a, rf1, rp, rr, ef1, tau in rows:
        print(f"{t:>5.2f} {a:>6.2f} {rf1:>9.3f} {rp:>7.3f} {rr:>7.3f} "
              f"{ef1:>8.3f} {tau:>7.3f}")
    best = rows[0]
    print(f"\nBest: threshold={best[0]:.2f}, alpha={best[1]:.2f}  "
          f"(reach_f1={best[2]:.3f}, prec={best[3]:.3f}, rec={best[4]:.3f}, "
          f"edge_f1={best[5]:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Quantitative flowsheet predictor")
    parser.add_argument("--graphs-bucket", required=True)
    parser.add_argument("--graphs-prefix", default="graphs/")
    parser.add_argument("--features", default=",".join(CORE_FEATURES),
                        help="Comma-separated feature groups (default: full CORE_FEATURES baseline)")
    parser.add_argument("--model", default="blended",
                        choices=["polynomial", "xgboost", "knn", "blended"])
    parser.add_argument("--vocab-version", default="v1", choices=["v1", "v2"])
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for KNN/blended")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="XGBoost weight in blend (0=pure KNN, 1=pure XGBoost)")
    parser.add_argument("--edge-threshold", type=float, default=0.3,
                        help="Score cutoff for KNN edge prediction (0-1)")
    parser.add_argument("--edge-alpha", type=float, default=1.0,
                        help="KNN weight in edge blend (1=pure KNN, 0=pure corpus prior)")
    parser.add_argument("--edge-recycle-threshold", type=float, default=None,
                        help="Separate threshold for recycle edges (src.order >= dst.order). "
                             "If unset, uses --edge-threshold for all edges.")
    parser.add_argument("--compare", action="store_true",
                        help="Run all feature group combinations and print summary")
    parser.add_argument("--compare-models", action="store_true",
                        help="Compare all model types on best feature set")
    parser.add_argument("--sweep-edges", action="store_true",
                        help="Sweep --edge-threshold x --edge-alpha on val and rank by joint objective")
    parser.add_argument("--v2-substrate", action="store_true",
                        help="Use V2 checkpoint stage predictions (val macro-F1=0.86) "
                             "as the edge predictor's input. Combine with --sweep-edges "
                             "to sweep threshold x recycle_threshold.")
    args = parser.parse_args()

    # Load graphs once
    graph_data = load_graphs(args.graphs_bucket, args.graphs_prefix,
                             args.min_support, args.vocab_version)

    if args.v2_substrate:
        v2_data = load_v2_checkpoint_predictions()
        doc_edges = build_v2_doc_edges(graph_data["graphs"], set(v2_data["stage_vocab"]))
        if args.sweep_edges:
            run_v2_substrate_sweep(v2_data, doc_edges, args)
        else:
            conn_m, ord_m = run_v2_substrate_experiment(
                v2_data, doc_edges,
                edge_threshold=args.edge_threshold,
                edge_recycle_threshold=args.edge_recycle_threshold,
            )
            print(f"\n{'='*60}")
            print(f"  V2 Substrate — Edge Prediction on V2 Stage Predictions")
            print(f"{'='*60}")
            print(f"\nConnections ({conn_m['n_docs']} docs):")
            print(f"  Mean F1:        {conn_m['mean_f1']:.3f}")
            print(f"  Mean Precision: {conn_m['mean_precision']:.3f}")
            print(f"  Mean Recall:    {conn_m['mean_recall']:.3f}")
            per_edge = conn_m.get("per_edge", {})
            if per_edge:
                scored = [(e, m) for e, m in per_edge.items() if m["support"] >= 2]
                if scored:
                    print(f"\n  Top edges by support:")
                    print(f"  {'Edge':<45} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Sup':>4}")
                    print(f"  {'-' * 70}")
                    for edge, m in sorted(scored, key=lambda x: -x[1]["support"])[:12]:
                        label = f"{edge[0]} -> {edge[1]}"
                        print(f"  {label:<45} {m['f1']:>6.3f} {m['tp']:>4} "
                              f"{m['fp']:>4} {m['fn']:>4} {m['support']:>4}")
                    worst = sorted(scored, key=lambda x: (x[1]["f1"], -x[1]["support"]))[:5]
                    print(f"\n  Worst edges (lowest F1, support >= 2):")
                    for edge, m in worst:
                        label = f"{edge[0]} -> {edge[1]}"
                        print(f"  {label:<45} {m['f1']:>6.3f} {m['tp']:>4} "
                              f"{m['fp']:>4} {m['fn']:>4} {m['support']:>4}")
            print(f"\nFlow Structure Quality ({ord_m['n_docs']} docs):")
            print(f"  Reachability F1:        {ord_m['reach_f1']:.3f}  (primary)")
            print(f"    precision:            {ord_m['reach_precision']:.3f}")
            print(f"    recall:               {ord_m['reach_recall']:.3f}")
            print(f"  Kendall's Tau:          {ord_m['kendall_tau']:.3f}")
            print(f"  LCS Ratio:              {ord_m['lcs_ratio']:.3f}")
            print()
        return

    if args.compare:
        from quantitative.eval import print_report
        combos = [CORE_FEATURES[:i+1] for i in range(len(CORE_FEATURES))]
        print(f"\n{'Features':<60} {'Docs':>5} {'Macro F1':>9} {'Micro F1':>9}")
        print("-" * 87)
        for groups in combos:
            stage_m, conn_m, ord_m, data = run_experiment(
                graph_data, groups, args.model, args.threshold, args.k, args.alpha,
                edge_threshold=args.edge_threshold, edge_alpha=args.edge_alpha,
                edge_recycle_threshold=args.edge_recycle_threshold)
            label = ",".join(groups)
            print(f"{label:<60} {data['X'].shape[0]:>5} {stage_m['macro_f1']:>9.3f} {stage_m['micro_f1']:>9.3f}")
        print()
        print_report(stage_m, conn_m, f"{args.model.title()} ({label})", ordering_metrics=ord_m)

    elif args.compare_models:
        from quantitative.eval import print_report
        feature_groups = [g.strip() for g in args.features.split(",")]
        models = ["xgboost", "knn", "blended"]
        print(f"\n{'Model':<20} {'Macro F1':>9} {'Micro F1':>9} {'Conn F1':>9}")
        print("-" * 50)
        results = {}
        for m in models:
            stage_m, conn_m, ord_m, data = run_experiment(
                graph_data, feature_groups, m, args.threshold, args.k, args.alpha,
                edge_threshold=args.edge_threshold, edge_alpha=args.edge_alpha,
                edge_recycle_threshold=args.edge_recycle_threshold)
            print(f"{m:<20} {stage_m['macro_f1']:>9.3f} {stage_m['micro_f1']:>9.3f} {conn_m['mean_f1']:>9.3f}")
            results[m] = (stage_m, conn_m, ord_m)
        print()
        # Best by stage macro-F1 (full report) and best by edge F1 (per-edge breakdown)
        best_stage_name = max(results, key=lambda m: results[m][0]["macro_f1"])
        best_edge_name = max(results, key=lambda m: results[m][1]["mean_f1"])
        s_stage, s_conn, s_ord = results[best_stage_name]
        print_report(s_stage, s_conn, f"{best_stage_name.title()} - best stages ({args.features})",
                     ordering_metrics=s_ord)
        if best_edge_name != best_stage_name:
            e_stage, e_conn, e_ord = results[best_edge_name]
            print_report(e_stage, e_conn, f"{best_edge_name.title()} - best edges ({args.features})",
                         ordering_metrics=e_ord)

    elif args.sweep_edges:
        run_edge_sweep(graph_data, args)

    else:
        from quantitative.eval import print_report
        feature_groups = [g.strip() for g in args.features.split(",")]
        stage_m, conn_m, ord_m, data = run_experiment(
            graph_data, feature_groups, args.model, args.threshold, args.k, args.alpha,
            edge_threshold=args.edge_threshold, edge_alpha=args.edge_alpha,
            edge_recycle_threshold=args.edge_recycle_threshold)
        model_name = f"{args.model.title()} ({args.features})"
        print_report(stage_m, conn_m, model_name, ordering_metrics=ord_m)


if __name__ == "__main__":
    main()
