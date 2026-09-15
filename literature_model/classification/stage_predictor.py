"""XGBoost per-stage binary classifiers with SHAP interpretability.

Replaces the GAT classifier (macro-F1=0.05) with gradient-boosted trees on
flattened document features.  180 docs × ~1,100 features is ideal for XGBoost;
SHAP TreeExplainer directly answers "which NI 43-101 sections influence which
process stages."

Key enhancements over the original Binary Relevance baseline:
  - recovery_methods (Section 17) excluded from features to avoid circular prediction
  - Classifier chains exploit stage co-occurrence (flotation → thickener/filter)
  - TF-IDF SVD features (20-dim per section) capture vocabulary patterns
  - Per-stage hyperparameter tuning (small grid search)
  - Semi-supervised pseudo-labeling on ~700 unlabeled docs
  - VarianceThreshold removes near-constant features

Usage (SageMaker entry script):
    Hyperparameters: graphs-bucket, graphs-prefix, folds, min-support
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
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (copied from graph_pipeline.py / layout_gat.py for SageMaker self-containment)
# ---------------------------------------------------------------------------

ALL_SECTION_TYPES = [
    "climate", "data_verification", "drilling", "economics", "environmental",
    "exploration", "geology", "history", "infrastructure", "interpretation",
    "market", "metallurgical_testing", "mining_method", "property",
    "recovery_methods", "references", "resource_estimate", "sample_analysis",
    "summary",
]

# Feature sections: only sections that drive flowsheet design decisions
FEATURE_SECTION_TYPES = [
    "geology",                # Ore mineralogy, deposit type → determines process route
    "metallurgical_testing",  # Recovery rates, process response → most direct signal
    "recovery_methods",       # Process design context — why the flowsheet was designed this way
    "economics",              # NPV, CAPEX, OPEX → constrains which stages are viable
    "mining_method",          # Open pit vs underground, throughput → grinding/crushing config
    "infrastructure",         # Power, water, transport → enables/constrains equipment choices
    "climate",                # Water availability, rainfall → heap leach vs tank leach
]

DOMAIN_KEYWORDS = [
    # Equipment
    "crusher", "mill", "flotation", "leach", "thickener", "filter",
    "electrowinning", "gravity", "elution", "solvent_extraction",
    "cyclone", "screen", "kiln", "reactor", "adsorption", "precipitation",
    "stockpile", "hopper", "feeder", "conveyor", "tank", "agglomeration",
    "magnetic_separation", "merrill_crowe", "ion_exchange", "drying",
    "water_treatment", "regrind", "cell", "ore_sorting",
    # Au-process equipment
    "cyanide", "cyanidation", "cil", "cip", "carbon_in_leach",
    "carbon_in_pulp", "carbon", "dore", "roasting", "pressure_oxidation",
    "bio_oxidation", "heap_leach", "intensive_leach", "gravity_concentrate",
    "inline_leach",
    # Mineralogy (pruned to high-signal only — full list had near-zero ablation impact)
    "chalcopyrite", "pyrite", "arsenopyrite", "molybdenite",
    # Deposit
    "porphyry", "skarn", "vms", "iocg", "sedimentary", "epithermal",
    "orogenic", "breccia", "intrusive", "volcanic",
    # Ore
    "oxide", "sulfide", "sulphide", "supergene", "hypogene",
    "transition", "refractory", "free_milling", "oxide_gold", "sulfide_gold",
    # Economics
    "tonnage", "grade", "recovery", "npv", "irr", "capex", "opex",
    "payback", "cut_off", "measured", "indicated", "inferred",
    # Mining
    "open_pit", "underground", "block_cave", "stoping", "strip_ratio",
    # Climate
    "rainfall", "arid", "water_availability", "tailings_dam", "closure",
]

NUMERIC_FEATURES = [
    "num_count", "num_mean", "num_median", "num_max",
    "grade", "recovery", "tonnage", "cost", "npv", "irr",
    # Targeted table extraction: actual values by column type
    "grade_cu_min", "grade_cu_mean", "grade_cu_max",
    "grade_au_min", "grade_au_mean", "grade_au_max",
    "grade_ag_min", "grade_ag_mean", "grade_ag_max",
    "recovery_min", "recovery_mean", "recovery_max",
    "tonnage_min", "tonnage_mean", "tonnage_max",
    "grind_size_min", "grind_size_mean", "grind_size_max",
    "npv_min", "npv_mean", "npv_max",
    "capex_min", "capex_mean", "capex_max",
    "opex_min", "opex_mean", "opex_max",
]

TEXT_FEATURES = ["char_count", "num_pages"]

DEPOSIT_TYPES = [
    "porphyry", "skarn", "vms", "iocg", "sedimentary",
    "epithermal", "orogenic", "breccia", "intrusive", "volcanic",
]

ORE_TYPES = [
    "oxide", "sulfide", "sulphide", "supergene", "hypogene",
    "transition", "refractory", "free_milling",
]

# TF-IDF SVD dimensions per section
TFIDF_SVD_COMPONENTS = 20

# Stage vocabulary reduction: CATEGORY_MAP from layout_gat.py + CORE_WORDS additions
CANONICAL_STAGES = {
    # From CATEGORY_MAP
    "crusher": "crusher", "mill": "mill", "regrind": "regrind",
    "screen": "screen", "agglomeration": "agglomeration", "cyclone": "cyclone",
    "input": "input",
    "flotation": "flotation", "magnetic_separation": "magnetic_separation",
    "thickener": "thickener", "filter": "filter", "cell": "electrowinning",
    "gravity": "gravity",
    "leach": "leach", "adsorption": "adsorption", "precipitation": "product_recovery",
    "electrowinning": "electrowinning", "elution": "elution",
    "solvent_extraction": "solution_recovery", "merrill_crowe": "solution_recovery",
    "ion_exchange": "solution_recovery",
    "kiln": "kiln", "reactor": "kiln",
    "stockpile": "stockpile", "hopper": "feeder", "feeder": "feeder",
    "conveyor": "conveyor", "bin": "bin",
    "tank": "tank", "pond": "tailing", "tailing": "tailing",
    "drying": "drying", "water_treatment": "water_treatment",
    "ore_sorting": "ore_sorting", "dore": "product_recovery",
    "carbon_handling": "carbon_handling", "refinery": "product_recovery",
    "detox": "tailing",
}

# Synonyms for mapping raw stage_ids -> canonical (from graph_pipeline SYNONYMS)
STAGE_SYNONYMS = {
    "crushing": "crusher", "grinding": "mill", "screening": "screen",
    "thickening": "thickener", "filtering": "filter", "filtration": "filter",
    "leaching": "leach", "rougher": "flotation", "cleaner": "flotation",
    "scavenger": "flotation", "recleaner": "flotation", "flash": "flotation",
    "dewatering": "filter", "classification": "cyclone", "conditioning": "tank",
    "decantation": "thickener", "cic": "adsorption", "columns": "adsorption",
    "storage": "bin", "disposal": "tailing", "dryer": "filter",
    "water_treatment_plant": "water_treatment",
    "sx": "solution_recovery",
    "ix": "solution_recovery",
    "zinc_precip": "product_recovery",
    "gold_room": "product_recovery",
    "smelting": "product_recovery",
    "cementation": "solution_recovery",
}

REMOVABLE_WORDS = {
    "copper", "cu", "gold", "au", "iron", "fe", "lead", "pb", "zinc", "zn",
    "moly", "silver", "polymetallic", "pyrite",
}


# ---------------------------------------------------------------------------
# V2 Stage Vocabulary — 37 mid-level stages preserving sequence distinctions
# ---------------------------------------------------------------------------

V2_DIRECT_MAP = {
    # Comminution
    "sag_milling": "sag_mill", "sag_mill": "sag_mill", "sag": "sag_mill",
    "ball_milling": "ball_mill", "ball_mill": "ball_mill",
    "rod_milling": "ball_mill", "rod_mill": "ball_mill",
    "milling": "ball_mill", "grinding": "ball_mill",
    "hpgr": "hpgr",
    "regrind": "regrind", "regrind_mill": "regrind",
    "screen": "screen", "vibrating_screen": "screen", "screening": "screen",
    "grizzly": "screen",
    "cyclone": "cyclone", "hydrocyclone": "cyclone", "classification": "cyclone",
    "ciclones": "cyclone",
    # Concentration
    "gravity": "gravity_concentration", "gravity_concentration": "gravity_concentration",
    "dms": "gravity_concentration", "dms_plant": "gravity_concentration",
    "dense_media_separation": "gravity_concentration",
    "magnetic_separation": "magnetic_separation",
    # Hydromet
    "leach": "leach", "leaching": "leach", "heap_leach": "leach",
    "tank_leaching": "leach",
    "adsorption": "adsorption", "cip": "adsorption", "cic": "adsorption",
    "cil": "adsorption", "columns": "adsorption",
    "elution": "elution", "stripping": "elution", "acid_wash": "elution",
    "elution_regeneration": "elution", "elution_and_electrowinning": "elution",
    "elution_electrowinning": "elution",
    "solvent_extraction": "solvent_extraction", "sx": "solvent_extraction",
    "sx_extraction": "solvent_extraction",
    "electrowinning": "electrowinning", "cell": "electrowinning",
    "precipitation": "precipitation", "merrill_crowe": "precipitation",
    "cementation": "precipitation", "zinc_precip": "precipitation",
    "ion_exchange": "ion_exchange", "ix": "ion_exchange",
    "ccd": "ccd", "decantation": "ccd",
    # Pyromet
    "kiln": "kiln", "roasting": "kiln", "calciner": "kiln",
    "reactor": "kiln", "autoclave": "kiln", "pressure_oxidation": "kiln",
    "smelting": "smelting", "furnace": "smelting",
    "regeneration": "carbon_regeneration", "carbon_regeneration": "carbon_regeneration",
    # Solid-liquid sep
    "filter": "filter", "filter_press": "filter", "filtration": "filter",
    "dewatering": "filter",
    # Material handling
    "input": "input", "rom_stockpile": "input",
    "stockpile": "stockpile", "ore_stockpile": "stockpile",
    "bin": "bin", "fine_ore_bin": "bin", "concentrate_bin": "bin", "silo": "bin",
    "feeder": "feeder", "apron_feeder": "feeder", "hopper": "feeder",
    "conveyor": "conveyor",
    # Product / waste
    "tailing": "tailing", "tailings": "tailing", "pond": "tailing",
    "disposal": "tailing", "detox": "water_treatment",
    "water_treatment": "water_treatment", "cyanide_destruction": "water_treatment",
    "cyanide_detox": "water_treatment", "neutralization": "water_treatment",
    "solution_ponds": "solution_pond", "solution_pond": "solution_pond",
    "dore": "concentrate_product", "dore_product": "concentrate_product",
    "concentrate": "concentrate_product", "concentrate_product": "concentrate_product",
    "product": "concentrate_product", "cathode_product": "concentrate_product",
    "refinery": "concentrate_product", "gold_room": "concentrate_product",
    "refining": "concentrate_product",
    # Aux
    "tank": "tank", "conditioning": "tank", "agitation_tank": "tank",
    "agglomeration": "agglomeration",
    "ore_sorting": "ore_sorting",
}

V2_STAGE_VOCAB = sorted({
    "primary_crusher", "secondary_crusher", "hpgr",
    "sag_mill", "ball_mill", "regrind",
    "screen", "cyclone",
    "rougher_flotation", "cleaner_flotation",
    "gravity_concentration", "magnetic_separation",
    "leach", "adsorption", "elution", "solvent_extraction",
    "electrowinning", "precipitation", "ion_exchange", "ccd",
    "kiln", "smelting", "carbon_regeneration",
    "concentrate_thickener", "tailings_thickener", "filter",
    "input", "stockpile", "bin", "feeder", "conveyor",
    "concentrate_product", "tailing", "solution_pond", "water_treatment",
    "tank", "agglomeration", "ore_sorting",
})


def map_stage_v2(raw_id: str, order: float = 0.0,
                 edges_in: list = None, edges_out: list = None) -> str | None:
    """Map a raw stage ID to the V2 mid-level vocabulary.

    Uses order and connection context to disambiguate:
    - crusher → primary_crusher (order < 0.3) or secondary_crusher
    - flotation → rougher_flotation (default) or cleaner_flotation (if after regrind)
    - thickener → concentrate_thickener or tailings_thickener (based on output edges)
    - mill → sag_mill or ball_mill (based on raw ID)
    """
    sid = raw_id.lower().strip()

    # Strip metal prefixes
    tokens = sid.split("_")
    tokens = [t for t in tokens if t not in REMOVABLE_WORDS]
    if not tokens:
        return None
    sid_clean = "_".join(tokens)

    # Direct lookup
    if sid in V2_DIRECT_MAP:
        v2 = V2_DIRECT_MAP[sid]
    elif sid_clean in V2_DIRECT_MAP:
        v2 = V2_DIRECT_MAP[sid_clean]
    else:
        # Try last token
        for t in reversed(tokens):
            if t in V2_DIRECT_MAP:
                v2 = V2_DIRECT_MAP[t]
                break
        else:
            return None

    # Context-dependent disambiguation
    edges_in = edges_in or []
    edges_out = edges_out or []

    if v2 == "ball_mill" and "sag" in sid:
        v2 = "sag_mill"

    if v2 in ("sag_mill", "ball_mill") and "mill" == sid and "sag" not in sid:
        # Generic "mill" — use order to guess. Early = sag, late = ball
        v2 = "sag_mill" if order < 0.4 else "ball_mill"

    # Crusher disambiguation by order
    if sid in ("crusher", "crushing") or sid_clean in ("crusher", "crushing"):
        v2 = "primary_crusher" if order < 0.3 else "secondary_crusher"

    # Flotation disambiguation
    if sid in ("flotation",) or sid_clean in ("flotation",):
        # If regrind is in the incoming edges, this is a cleaner
        in_types = {e.get("type", "") for e in edges_in} if edges_in else set()
        if "regrind" in in_types or any("regrind" in str(e) for e in edges_in):
            v2 = "cleaner_flotation"
        else:
            v2 = "rougher_flotation"
    elif sid in ("rougher", "rougher_flotation"):
        v2 = "rougher_flotation"
    elif sid in ("cleaner", "cleaner_flotation", "recleaner", "scavenger",
                 "column_cleaning"):
        v2 = "cleaner_flotation"

    # Thickener disambiguation by output edges
    if sid in ("thickener", "thickening"):
        out_types = set()
        for e in edges_out:
            if isinstance(e, dict):
                out_types.add(e.get("target", "").replace("stg_", ""))
            elif isinstance(e, str):
                out_types.add(e)
        if out_types & {"tailing", "tailings", "pond", "disposal", "tsf"}:
            v2 = "tailings_thickener"
        elif out_types & {"filter", "bin", "output", "concentrate"}:
            v2 = "concentrate_thickener"
        else:
            # Default: if order > 0.7, likely tailings
            v2 = "tailings_thickener" if order > 0.7 else "concentrate_thickener"

    if v2 in V2_STAGE_VOCAB:
        return v2
    return None


# ---------------------------------------------------------------------------
# Stage vocabulary reduction (V1 — kept for backward compatibility)
# ---------------------------------------------------------------------------

def _map_stage(raw_id: str) -> str | None:
    """Map a raw stage_id to a canonical stage name."""
    sid = raw_id.lower().strip()
    # Direct match
    if sid in CANONICAL_STAGES:
        return CANONICAL_STAGES[sid]
    # Synonym match
    if sid in STAGE_SYNONYMS:
        return STAGE_SYNONYMS[sid]
    # Strip site-specific suffixes and metals
    tokens = sid.split("_")
    tokens = [t for t in tokens if t not in REMOVABLE_WORDS]
    if not tokens:
        return None
    # Try joined
    joined = "_".join(tokens)
    if joined in CANONICAL_STAGES:
        return CANONICAL_STAGES[joined]
    if joined in STAGE_SYNONYMS:
        return STAGE_SYNONYMS[joined]
    # Try last token (e.g. "copper_flotation" -> "flotation")
    for t in reversed(tokens):
        if t in CANONICAL_STAGES:
            return CANONICAL_STAGES[t]
        if t in STAGE_SYNONYMS:
            return STAGE_SYNONYMS[t]
    return None


def reduce_stage_vocab(graphs: list[dict], min_support: int = 3,
                       vocab_version: str = "v1") -> tuple[list[str], dict]:
    """Map raw stage IDs to canonical names, drop rare stages.

    Args:
        vocab_version: "v1" (23 stages) or "v2" (37 stages with sequence distinctions)

    Returns:
        stage_vocab: sorted list of canonical stage names with >= min_support docs
        raw_to_canonical: mapping from raw stage_id -> canonical name
    """
    raw_to_canonical = {}
    stage_doc_counts = defaultdict(set)

    for doc_idx, graph in enumerate(graphs):
        # Build per-node edge context for V2 disambiguation
        nodes = {}
        out_by = defaultdict(list)
        in_by = defaultdict(list)
        for node in graph.get("nodes", []):
            if node.get("type") == "stage":
                sid = node.get("stage_id", node["id"].replace("stg_", ""))
                order = node.get("features", {}).get("order_normalized", 0.5)
                nodes[node["id"]] = {"raw": sid, "order": order}
        for edge in graph.get("edges", []):
            if edge.get("type") == "stage_transition":
                src_raw = nodes.get(edge["source"], {}).get("raw", "")
                dst_raw = nodes.get(edge["target"], {}).get("raw", "")
                out_by[edge["source"]].append(dst_raw)
                in_by[edge["target"]].append(src_raw)

        for node in graph.get("nodes", []):
            if node.get("type") != "stage":
                continue
            nid = node["id"]
            raw_id = node.get("stage_id", nid.replace("stg_", ""))

            if vocab_version == "v2":
                info = nodes.get(nid, {"raw": raw_id, "order": 0.5})
                canonical = map_stage_v2(
                    info["raw"], order=info["order"],
                    edges_in=[{"type": r} for r in in_by.get(nid, [])],
                    edges_out=[{"target": r} for r in out_by.get(nid, [])],
                )
            else:
                canonical = _map_stage(raw_id)

            if canonical:
                raw_to_canonical[raw_id] = canonical
                stage_doc_counts[canonical].add(doc_idx)

    # Filter by min support
    stage_vocab = sorted(
        s for s, docs in stage_doc_counts.items() if len(docs) >= min_support
    )

    logger.info(
        f"Stage vocab ({vocab_version}): {len(stage_doc_counts)} canonical -> "
        f"{len(stage_vocab)} with >={min_support} support "
        f"(from {len(raw_to_canonical)} raw IDs)"
    )
    for s in stage_vocab:
        logger.info(f"  {s}: {len(stage_doc_counts[s])} docs")

    return stage_vocab, raw_to_canonical


# ---------------------------------------------------------------------------
# Feature flattening
# ---------------------------------------------------------------------------

def build_feature_names() -> list[str]:
    """Build ordered list of feature names for the flat vector (excludes recovery_methods)."""
    names = []
    for section in FEATURE_SECTION_TYPES:
        for feat in NUMERIC_FEATURES:
            names.append(f"{section}__{feat}")
        for kw in DOMAIN_KEYWORDS:
            names.append(f"{section}__kw_{kw}")
        for feat in TEXT_FEATURES:
            names.append(f"{section}__{feat}")
        names.append(f"{section}__present")
    # Document-level metadata
    for dt in DEPOSIT_TYPES:
        names.append(f"meta__deposit_{dt}")
    for ot in ORE_TYPES:
        names.append(f"meta__ore_{ot}")
    names.extend(["meta__grade_min", "meta__grade_mean", "meta__grade_max"])
    return names


def flatten_document(graph: dict) -> np.ndarray:
    """Convert a graph JSON to a flat feature vector."""
    n_per_section = len(NUMERIC_FEATURES) + len(DOMAIN_KEYWORDS) + len(TEXT_FEATURES) + 1
    n_sections = len(FEATURE_SECTION_TYPES)
    n_meta = len(DEPOSIT_TYPES) + len(ORE_TYPES) + 3
    vec = np.zeros(n_sections * n_per_section + n_meta, dtype=np.float32)

    section_idx = {s: i for i, s in enumerate(FEATURE_SECTION_TYPES)}

    # Extract context node features
    for node in graph.get("nodes", []):
        if node.get("type") != "context":
            continue
        group = node.get("group", "")
        if group not in section_idx:
            continue
        si = section_idx[group]
        offset = si * n_per_section
        feats = node.get("features", {})

        # Numeric features
        for j, feat in enumerate(NUMERIC_FEATURES):
            vec[offset + j] = float(feats.get(feat, 0.0))

        # Keyword counts
        kw_offset = offset + len(NUMERIC_FEATURES)
        for j, kw in enumerate(DOMAIN_KEYWORDS):
            vec[kw_offset + j] = float(feats.get(f"kw_{kw}", 0.0))

        # Text stats
        text_offset = kw_offset + len(DOMAIN_KEYWORDS)
        vec[text_offset] = float(feats.get("char_count", 0.0))
        vec[text_offset + 1] = float(feats.get("num_pages", 0.0))

        # Presence flag
        vec[text_offset + 2] = 1.0

    # Document-level metadata from geology / resource_estimate nodes
    meta_offset = n_sections * n_per_section
    for node in graph.get("nodes", []):
        if node.get("type") != "context":
            continue
        group = node.get("group", "")
        feats = node.get("features", {})

        if group == "geology":
            for j, dt in enumerate(DEPOSIT_TYPES):
                if feats.get(f"kw_{dt}", 0) > 0:
                    vec[meta_offset + j] = 1.0
            ore_offset = meta_offset + len(DEPOSIT_TYPES)
            for j, ot in enumerate(ORE_TYPES):
                if feats.get(f"kw_{ot}", 0) > 0:
                    vec[ore_offset + j] = 1.0

        if group == "resource_estimate":
            grade_offset = meta_offset + len(DEPOSIT_TYPES) + len(ORE_TYPES)
            num_mean = feats.get("num_mean", 0.0)
            num_max = feats.get("num_max", 0.0)
            num_min = feats.get("num_count", 0.0)  # approximate
            if num_mean > 0:
                vec[grade_offset] = min(vec[grade_offset], num_mean) if vec[grade_offset] > 0 else num_mean
                vec[grade_offset + 1] = num_mean
            if num_max > 0:
                vec[grade_offset + 2] = max(vec[grade_offset + 2], num_max)

    # Sanitize: replace inf/nan, clamp extreme values (some num_mean/num_max
    # in resource_estimate sections are very large raw numbers)
    np.nan_to_num(vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(vec, -1e9, 1e9, out=vec)
    return vec


def _extract_tfidf_vectors(graph: dict) -> dict[str, np.ndarray]:
    """Extract per-section TF-IDF vectors from a graph's context nodes."""
    tfidf = {}
    for node in graph.get("nodes", []):
        if node.get("type") != "context":
            continue
        group = node.get("group", "")
        if group not in FEATURE_SECTION_TYPES:
            continue
        feats = node.get("features", {})
        vec = feats.get("tfidf")
        if vec is not None and len(vec) > 0:
            tfidf[group] = np.array(vec, dtype=np.float32)
    return tfidf


# ---------------------------------------------------------------------------
# TF-IDF SVD
# ---------------------------------------------------------------------------

def fit_tfidf_svd(all_tfidf: list[dict[str, np.ndarray]]) -> dict[str, TruncatedSVD]:
    """Fit TruncatedSVD per section on all documents (labeled + unlabeled).

    Args:
        all_tfidf: list of per-doc dicts mapping section -> tfidf vector

    Returns:
        dict mapping section -> fitted TruncatedSVD
    """
    section_vectors = defaultdict(list)
    for doc_tfidf in all_tfidf:
        for section, vec in doc_tfidf.items():
            section_vectors[section].append(vec)

    svd_models = {}
    for section in FEATURE_SECTION_TYPES:
        vecs = section_vectors.get(section, [])
        if len(vecs) < TFIDF_SVD_COMPONENTS + 1:
            continue
        mat = np.array(vecs, dtype=np.float64)
        n_components = min(TFIDF_SVD_COMPONENTS, mat.shape[1], mat.shape[0] - 1)
        if n_components < 1:
            continue
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(mat)
        svd_models[section] = svd
        logger.info(f"  SVD {section}: {mat.shape[0]} docs, "
                    f"{n_components} components, "
                    f"explained_var={svd.explained_variance_ratio_.sum():.3f}")

    logger.info(f"Fitted TF-IDF SVD for {len(svd_models)} sections "
                f"({len(svd_models) * TFIDF_SVD_COMPONENTS} new features)")
    return svd_models


def transform_tfidf_svd(doc_tfidf: dict[str, np.ndarray],
                         svd_models: dict[str, TruncatedSVD]) -> np.ndarray:
    """Transform one document's TF-IDF vectors through fitted SVD models.

    Returns flat vector of length len(FEATURE_SECTION_TYPES) * TFIDF_SVD_COMPONENTS.
    """
    result = np.zeros(len(FEATURE_SECTION_TYPES) * TFIDF_SVD_COMPONENTS, dtype=np.float32)
    for si, section in enumerate(FEATURE_SECTION_TYPES):
        svd = svd_models.get(section)
        vec = doc_tfidf.get(section)
        if svd is not None and vec is not None:
            offset = si * TFIDF_SVD_COMPONENTS
            transformed = svd.transform(vec.reshape(1, -1))[0]
            n = len(transformed)
            result[offset:offset + n] = transformed
    return result


def build_svd_feature_names(svd_models: dict[str, TruncatedSVD]) -> list[str]:
    """Build feature names for SVD-transformed TF-IDF features."""
    names = []
    for section in FEATURE_SECTION_TYPES:
        for c in range(TFIDF_SVD_COMPONENTS):
            names.append(f"{section}__svd_{c}")
    return names


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_graphs_from_s3(graphs_bucket: str, graphs_prefix: str) -> list[dict]:
    """Load all graph JSONs from S3."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    skip_keys = {"cross_doc_edges.json", "stage_vocab.json",
                 "graphs_summary.json", "checkpoint.json",
                 "allowlist.json", "attempt_marker.json", "skip_list.json"}
    graphs = []
    for page in paginator.paginate(Bucket=graphs_bucket, Prefix=graphs_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            basename = key.split("/")[-1]
            if basename in skip_keys:
                continue
            try:
                resp = s3.get_object(Bucket=graphs_bucket, Key=key)
                graph = json.loads(resp["Body"].read())
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
    logger.info(f"Loaded {len(graphs)} graphs from s3://{graphs_bucket}/{graphs_prefix}")
    return graphs


def load_training_data(graphs_bucket: str, graphs_prefix: str = "graphs/",
                       min_support: int = 3, vocab_version: str = "v1"):
    """Load graphs, flatten features, build label matrix.

    Returns dict with:
        X: (N_labeled, D) feature matrix
        y: (N_labeled, S) binary label matrix
        X_unlabeled: (N_unlabeled, D) feature matrix for unlabeled docs
        tfidf_labeled: list of per-doc TF-IDF dicts (labeled)
        tfidf_unlabeled: list of per-doc TF-IDF dicts (unlabeled)
        tfidf_all: list of ALL per-doc TF-IDF dicts (for SVD fitting)
        stage_vocab: list of S canonical stage names
        feature_names: list of D feature names
        transition_counts: dict of "src|dst" -> count
        doc_ids: list of labeled document IDs
    """
    graphs = _load_graphs_from_s3(graphs_bucket, graphs_prefix)

    # Reduce stage vocabulary
    stage_vocab, raw_to_canonical = reduce_stage_vocab(graphs, min_support,
                                                       vocab_version=vocab_version)
    stage_to_idx = {s: i for i, s in enumerate(stage_vocab)}

    feature_names = build_feature_names()
    logger.info(f"Feature vector dimension: {len(feature_names)}")

    X_list, y_list, doc_ids = [], [], []
    X_unlabeled_list, unlabeled_ids = [], []
    tfidf_labeled, tfidf_unlabeled, tfidf_all = [], [], []
    all_transitions = []
    doc_connections = []  # per-doc ground truth edge sets

    for graph in graphs:
        doc_id = graph.get("document_id", "unknown")

        # Check if this doc has any labeled stages
        doc_stages = set()
        for node in graph.get("nodes", []):
            if node.get("type") != "stage":
                continue
            raw_id = node.get("stage_id", node["id"].replace("stg_", ""))
            canonical = raw_to_canonical.get(raw_id)
            if canonical and canonical in stage_to_idx:
                doc_stages.add(canonical)

        # Flatten features and extract TF-IDF for ALL docs
        vec = flatten_document(graph)
        doc_tfidf = _extract_tfidf_vectors(graph)
        tfidf_all.append(doc_tfidf)

        if not doc_stages:
            # Unlabeled doc — save for pseudo-labeling
            X_unlabeled_list.append(vec)
            tfidf_unlabeled.append(doc_tfidf)
            unlabeled_ids.append(doc_id)
            continue

        X_list.append(vec)
        tfidf_labeled.append(doc_tfidf)

        # Build label vector
        label = np.zeros(len(stage_vocab), dtype=np.float32)
        for s in doc_stages:
            label[stage_to_idx[s]] = 1.0
        y_list.append(label)
        doc_ids.append(doc_id)

        # Collect transitions (corpus-level and per-doc)
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

    X = np.array(X_list, dtype=np.float64)  # use float64 to avoid overflow
    y = np.array(y_list, dtype=np.float32)
    X_unlabeled = np.array(X_unlabeled_list, dtype=np.float64) if X_unlabeled_list else np.zeros((0, X.shape[1]), dtype=np.float64)

    # Log-compress large values (same approach as graph_pipeline.py)
    X = np.sign(X) * np.log1p(np.abs(X))
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if X_unlabeled.shape[0] > 0:
        X_unlabeled = np.sign(X_unlabeled) * np.log1p(np.abs(X_unlabeled))
        np.nan_to_num(X_unlabeled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    transition_counts = defaultdict(int)
    for src, dst in all_transitions:
        transition_counts[f"{src}|{dst}"] += 1

    logger.info(f"Training data: {X.shape[0]} labeled docs, {X_unlabeled.shape[0]} unlabeled, "
                f"{X.shape[1]} features, {len(stage_vocab)} stages, "
                f"{len(transition_counts)} transitions")
    logger.info(f"Feature range: [{X.min():.4f}, {X.max():.4f}], "
                f"nonzero: {(X != 0).sum()} / {X.size}")

    return {
        "X": X,
        "y": y,
        "X_unlabeled": X_unlabeled,
        "tfidf_labeled": tfidf_labeled,
        "tfidf_unlabeled": tfidf_unlabeled,
        "tfidf_all": tfidf_all,
        "stage_vocab": stage_vocab,
        "feature_names": feature_names,
        "transition_counts": dict(transition_counts),
        "doc_ids": doc_ids,
        "unlabeled_ids": unlabeled_ids,
        "raw_to_canonical": raw_to_canonical,
        "doc_connections": doc_connections,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _find_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                    n_pos: int = 50) -> float:
    """Two-pass threshold sweep to maximize F1 (or F-beta for rare stages).

    For stages with n_pos < 25, uses F-beta (beta=1.5) to bias toward recall,
    since threshold tuning on 3-4 validation examples is unreliable.
    """
    use_fbeta = n_pos < 25
    best_score, best_t = 0.0, 0.5

    def _score(y_t, y_p):
        if use_fbeta:
            return fbeta_score(y_t, y_p, beta=1.5, zero_division=0)
        return f1_score(y_t, y_p, zero_division=0)

    # Coarse pass
    for t in np.arange(0.15, 0.86, 0.05):
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        s = _score(y_true, y_pred)
        if s > best_score:
            best_score, best_t = s, t
    # Fine pass around best
    lo = max(0.05, best_t - 0.05)
    hi = min(0.95, best_t + 0.05)
    for t in np.arange(lo, hi + 0.005, 0.01):
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        s = _score(y_true, y_pred)
        if s > best_score:
            best_score, best_t = s, t
    return float(best_t)


def _build_cooccurrence_priors(y: np.ndarray) -> np.ndarray:
    """Build conditional probability matrix P(stage_j | stage_i) from labels.

    Args:
        y: (N, S) binary label matrix

    Returns:
        (S, S) matrix where entry [i, j] = P(stage_j present | stage_i present)
    """
    S = y.shape[1]
    cooc = np.zeros((S, S), dtype=np.float64)
    counts = y.sum(axis=0)  # per-stage counts
    for i in range(S):
        if counts[i] == 0:
            continue
        mask_i = y[:, i] == 1
        for j in range(S):
            cooc[i, j] = y[mask_i, j].sum() / counts[i]
    return cooc


def _compute_cooccurrence_features(y_doc: np.ndarray, cooc_matrix: np.ndarray) -> np.ndarray:
    """Compute co-occurrence score features for a set of documents.

    For each doc, cooc_score_j = mean(P(j|i) for stages i present in doc).

    Args:
        y_doc: (N, S) binary labels (or predicted labels) for documents
        cooc_matrix: (S, S) conditional probability matrix

    Returns:
        (N, S) co-occurrence score features
    """
    N, S = y_doc.shape
    scores = np.zeros((N, S), dtype=np.float64)
    for n in range(N):
        present = np.where(y_doc[n] > 0.5)[0]
        if len(present) > 0:
            scores[n] = cooc_matrix[present].mean(axis=0)
    return scores


def _apply_smote(X: np.ndarray, y: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling for rare stages (n_pos < 25 and >= 3).

    Returns resampled (X, y). Returns originals unchanged if SMOTE not applicable.
    """
    if n_pos >= 25 or n_pos < 3:
        return X, y
    k_neighbors = min(2, n_pos - 1)
    try:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        return smote.fit_resample(X, y)
    except ValueError:
        return X, y


def _determine_chain_order(y: np.ndarray, stage_names: list[str]) -> list[int]:
    """Order stages by frequency (most common first) for classifier chain."""
    freqs = y.sum(axis=0)
    order = np.argsort(-freqs).tolist()
    # Only include stages with >= 2 positives (trainable)
    order = [i for i in order if freqs[i] >= 2]
    logger.info(f"Chain order ({len(order)} stages): "
                + ", ".join(f"{stage_names[i]}({int(freqs[i])})" for i in order[:10])
                + ("..." if len(order) > 10 else ""))
    return order


def _tune_hyperparams(X_train: np.ndarray, y_train: np.ndarray,
                      scale_pos_weight: float, n_splits: int,
                      n_pos: int = 50) -> dict:
    """Two-phase grid search with early stopping.

    Phase 1: 36 combos (depth × lr × colsample) with early stopping.
    Phase 2: Best 3 param sets × 3 reg_alpha values (9 combos).

    Returns dict of best hyperparameters including n_estimators from early stopping.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # min_child_weight scales with support: rare stages get finer splits
    mcw = 1 if n_pos < 20 else 3 if n_pos < 50 else 5

    def _eval_params(params):
        """Evaluate a param set across folds, return (mean_f1, mean_best_trees)."""
        fold_f1s = []
        fold_trees = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            n_pos_fold = int(y_tr.sum())
            X_tr, y_tr = _apply_smote(X_tr, y_tr, n_pos_fold)
            clf = XGBClassifier(
                n_estimators=500, early_stopping_rounds=30,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                colsample_bytree=params["colsample_bytree"],
                reg_alpha=params.get("reg_alpha", 0.0),
                reg_lambda=params.get("reg_lambda", 1.0),
                min_child_weight=mcw,
                subsample=0.7, scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            clf.fit(X_tr, y_tr,
                    eval_set=[(X_train[val_idx], y_train[val_idx])],
                    verbose=False)
            best_trees = clf.best_iteration + 1 if hasattr(clf, "best_iteration") else 300
            fold_trees.append(best_trees)
            val_prob = clf.predict_proba(X_train[val_idx])[:, 1]
            t = _find_threshold(y_train[val_idx], val_prob, n_pos=n_pos)
            val_pred = (val_prob >= t).astype(int)
            fold_f1s.append(f1_score(y_train[val_idx], val_pred, zero_division=0))
        return float(np.mean(fold_f1s)), int(np.mean(fold_trees))

    # Phase 1: core hyperparams (36 combos)
    phase1_results = []
    for max_depth in [3, 4, 5, 6]:
        for learning_rate in [0.03, 0.05, 0.1]:
            for colsample in [0.5, 0.7, 0.9]:
                params = {"max_depth": max_depth, "learning_rate": learning_rate,
                          "colsample_bytree": colsample}
                mean_f1, mean_trees = _eval_params(params)
                phase1_results.append((mean_f1, mean_trees, params))

    phase1_results.sort(key=lambda x: x[0], reverse=True)

    # Phase 2: top 3 param sets × (reg_alpha, reg_lambda) grid. 15 combos.
    # reg_lambda added (was default 1.0 everywhere) because L2 shrinks the val/OOF
    # overfit gap on stages where SMOTE + shallow trees weren't enough.
    best_f1, best_trees, best_params = phase1_results[0]
    reg_grid = [
        (0.0, 1.0), (0.5, 1.0), (2.0, 1.0),
        (0.0, 3.0), (0.5, 3.0),
    ]
    for f1_score_val, trees, params in phase1_results[:3]:
        for reg_alpha, reg_lambda in reg_grid:
            p2_params = {**params, "reg_alpha": reg_alpha, "reg_lambda": reg_lambda}
            mean_f1, mean_trees = _eval_params(p2_params)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_trees = mean_trees
                best_params = p2_params

    best_params["n_estimators"] = best_trees
    return best_params


def train_stage_models(X: np.ndarray, y: np.ndarray, stage_names: list[str],
                       feature_names: list[str], folds: int = 5,
                       output_dir: str = "/opt/ml/model",
                       X_unlabeled: np.ndarray | None = None):
    """Train classifier chain of XGBClassifiers with hyperparameter tuning.

    Stages are ordered by frequency. Each stage's features are augmented with
    OOF predictions from prior stages in the chain. After initial training,
    pseudo-labeling expands the dataset with high-confidence unlabeled docs.

    Returns dict with models, metrics, thresholds, chain_order.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    N, S = y.shape

    # Step 6: Variance threshold — remove near-constant features
    var_selector = VarianceThreshold(threshold=0.001)
    X_sel = var_selector.fit_transform(X)
    kept_mask = var_selector.get_support()
    n_removed = X.shape[1] - X_sel.shape[1]
    sel_feature_names = [f for f, k in zip(feature_names, kept_mask) if k]
    logger.info(f"VarianceThreshold: {X.shape[1]} -> {X_sel.shape[1]} features "
                f"({n_removed} removed)")

    if X_unlabeled is not None and X_unlabeled.shape[0] > 0:
        X_unl_sel = var_selector.transform(X_unlabeled)
    else:
        X_unl_sel = None

    # Step 2: Determine chain order (most frequent stages first)
    chain_order = _determine_chain_order(y, stage_names)

    # Co-occurrence prior features from full label matrix
    cooc_matrix = _build_cooccurrence_priors(y)
    cooc_feats_full = _compute_cooccurrence_features(y, cooc_matrix)
    logger.info(f"Co-occurrence priors: {cooc_matrix.shape} matrix, "
                f"nonzero={int((cooc_matrix > 0).sum())}")

    # Build OOF chain features using classifier chain approach
    oof_chain_probs = np.zeros((N, S), dtype=np.float64)
    models = {}
    thresholds = {}
    per_stage_metrics = {}
    best_hyperparams = {}

    for chain_pos, si in enumerate(chain_order):
        stage = stage_names[si]
        y_col = y[:, si]
        n_pos = int(y_col.sum())
        n_neg = N - n_pos

        if n_pos < 2:
            logger.info(f"  {stage}: skipped (only {n_pos} positive)")
            continue

        scale_pos = n_neg / max(n_pos, 1)
        n_splits = min(folds, n_pos)
        if n_splits < 2:
            logger.info(f"  {stage}: skipped (can't stratify with {n_pos} positives)")
            continue

        # Augment X with chain features + co-occurrence priors
        prior_indices = chain_order[:chain_pos]
        aug_parts = [X_sel]
        if prior_indices:
            aug_parts.append(oof_chain_probs[:, prior_indices])
        aug_parts.append(cooc_feats_full)
        X_aug = np.hstack(aug_parts)

        # Step 4: Hyperparameter tuning
        hp = _tune_hyperparams(X_aug, y_col, scale_pos, n_splits, n_pos=n_pos)
        best_hyperparams[stage] = hp

        # CV with best hyperparams
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_f1s = []

        mcw = 1 if n_pos < 20 else 3 if n_pos < 50 else 5

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_aug, y_col)):
            # Compute co-occurrence from training fold only (avoid leakage)
            cooc_fold = _build_cooccurrence_priors(y[train_idx])
            cooc_train = _compute_cooccurrence_features(y[train_idx], cooc_fold)
            cooc_val = _compute_cooccurrence_features(y[val_idx], cooc_fold)

            # Build fold-specific augmented features
            fold_aug_parts_tr = [X_sel[train_idx]]
            fold_aug_parts_val = [X_sel[val_idx]]
            if prior_indices:
                fold_aug_parts_tr.append(oof_chain_probs[train_idx][:, prior_indices])
                fold_aug_parts_val.append(oof_chain_probs[val_idx][:, prior_indices])
            fold_aug_parts_tr.append(cooc_train)
            fold_aug_parts_val.append(cooc_val)
            X_fold_tr = np.hstack(fold_aug_parts_tr)
            X_fold_val = np.hstack(fold_aug_parts_val)

            # SMOTE on training fold for rare stages
            n_pos_fold = int(y_col[train_idx].sum())
            X_fold_tr_sm, y_fold_tr_sm = _apply_smote(
                X_fold_tr, y_col[train_idx], n_pos_fold)

            clf_fold = XGBClassifier(
                n_estimators=hp.get("n_estimators", 300),
                learning_rate=hp.get("learning_rate", 0.05),
                max_depth=hp.get("max_depth", 4),
                colsample_bytree=hp.get("colsample_bytree", 0.7),
                reg_alpha=hp.get("reg_alpha", 0.0),
                reg_lambda=hp.get("reg_lambda", 1.0),
                min_child_weight=mcw,
                subsample=0.7, scale_pos_weight=scale_pos,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            clf_fold.fit(X_fold_tr_sm, y_fold_tr_sm)
            val_prob = clf_fold.predict_proba(X_fold_val)[:, 1]
            threshold = _find_threshold(y_col[val_idx], val_prob, n_pos=n_pos)
            val_pred = (val_prob >= threshold).astype(int)
            fold_f1 = f1_score(y_col[val_idx], val_pred, zero_division=0)
            fold_f1s.append(fold_f1)

            # Record OOF predictions for chain
            oof_chain_probs[val_idx, si] = val_prob

        mean_f1 = float(np.mean(fold_f1s))
        per_stage_metrics[stage] = {
            "mean_f1": round(mean_f1, 4),
            "n_positive": n_pos,
            "fold_f1s": [round(f, 4) for f in fold_f1s],
            "best_hp": hp,
        }
        logger.info(f"  {stage}: F1={mean_f1:.4f} (n_pos={n_pos}, {n_splits} folds, "
                    f"depth={hp.get('max_depth')}, lr={hp.get('learning_rate')}, "
                    f"col={hp.get('colsample_bytree')}, mcw={mcw})")

        # Train final model on all labeled data (with SMOTE for rare stages)
        X_aug_final, y_col_final = _apply_smote(X_aug, y_col, n_pos)
        clf = XGBClassifier(
            n_estimators=hp.get("n_estimators", 300),
            learning_rate=hp.get("learning_rate", 0.05),
            max_depth=hp.get("max_depth", 4),
            colsample_bytree=hp.get("colsample_bytree", 0.7),
            reg_alpha=hp.get("reg_alpha", 0.0),
            reg_lambda=hp.get("reg_lambda", 1.0),
            min_child_weight=mcw,
            subsample=0.7, scale_pos_weight=scale_pos,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        clf.fit(X_aug_final, y_col_final)
        models[stage] = clf

        # Find threshold on full OOF predictions
        thresholds[stage] = _find_threshold(y_col, oof_chain_probs[:, si], n_pos=n_pos)

    # Compute aggregate metrics on OOF predictions (before pseudo-labeling)
    _log_aggregate_metrics(y, oof_chain_probs, stage_names, chain_order,
                           models, thresholds, "Pre-pseudo-label")

    # Step 5: Semi-supervised pseudo-labeling (2 rounds)
    if X_unl_sel is not None and X_unl_sel.shape[0] > 0:
        models, thresholds = _pseudo_label_rounds(
            X_sel, y, X_unl_sel, stage_names, chain_order, models,
            thresholds, best_hyperparams, folds, n_rounds=2,
            cooc_matrix=cooc_matrix,
        )

    # Save models
    for stage, clf in models.items():
        clf.save_model(str(out / f"xgb_{stage}.json"))
    with open(out / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(out / "stage_metrics.json", "w") as f:
        json.dump(per_stage_metrics, f, indent=2)
    with open(out / "chain_order.json", "w") as f:
        json.dump(chain_order, f, indent=2)
    with open(out / "var_selector.pkl", "wb") as f:
        pickle.dump(var_selector, f)

    logger.info(f"Saved {len(models)} models to {output_dir}")

    return {
        "models": models,
        "thresholds": thresholds,
        "per_stage_metrics": per_stage_metrics,
        "chain_order": chain_order,
        "var_selector": var_selector,
        "oof_probs": oof_chain_probs,
    }


def _log_aggregate_metrics(y: np.ndarray, oof_probs: np.ndarray,
                            stage_names: list[str], chain_order: list[int],
                            models: dict, thresholds: dict, prefix: str):
    """Log aggregate OOF metrics for active stages."""
    N = y.shape[0]
    active = [(pos, si) for pos, si in enumerate(chain_order)
              if stage_names[si] in models]
    if not active:
        return
    active_si = [si for _, si in active]
    active_names = [stage_names[si] for si in active_si]
    oof_true = y[:, active_si]
    oof_prob = oof_probs[:, active_si]
    active_thresholds = np.array([thresholds.get(s, 0.5) for s in active_names])
    oof_pred = (oof_prob >= active_thresholds).astype(int)

    macro_f1 = f1_score(oof_true, oof_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(oof_true, oof_pred, average="micro", zero_division=0)
    precision = precision_score(oof_true, oof_pred, average="macro", zero_division=0)
    recall_val = recall_score(oof_true, oof_pred, average="macro", zero_division=0)

    logger.info(f"\n{prefix} OOF metrics ({len(active)} active stages):")
    logger.info(f"  macro-F1={macro_f1:.4f}  micro-F1={micro_f1:.4f}  "
                f"P={precision:.4f}  R={recall_val:.4f}")


def evaluate_connections(y: np.ndarray, oof_probs: np.ndarray,
                         stage_names: list[str], thresholds: dict,
                         transition_counts: dict,
                         doc_connections: list[set]):
    """Evaluate predicted connections against per-doc ground truth edges.

    Computes two sets of metrics:
      - Unconditional edge P/R/F1: all predicted edges vs all ground truth edges
      - Conditional edge P/R/F1: restricted to edges where both endpoints are
        in both predicted AND ground truth stage sets (isolates wiring errors
        from stage prediction errors)

    Returns dict with metrics and per-doc breakdown.
    """
    S = y.shape[1]
    N = y.shape[0]

    # Build per-doc predicted stage sets from OOF probs
    threshold_vec = np.array([thresholds.get(stage_names[i], 0.5) for i in range(S)])
    pred_stages_matrix = (oof_probs >= threshold_vec).astype(int)

    total_tp, total_fp, total_fn = 0, 0, 0
    cond_tp, cond_fp, cond_fn = 0, 0, 0
    self_loop_tp, self_loop_total_true, self_loop_total_pred = 0, 0, 0
    docs_with_edges = 0

    for i in range(N):
        true_edges = doc_connections[i]
        if not true_edges:
            continue
        docs_with_edges += 1

        # Predicted stage set for this doc
        pred_stage_set = {stage_names[j] for j in range(S) if pred_stages_matrix[i, j]}
        true_stage_set = {stage_names[j] for j in range(S) if y[i, j] > 0}

        # Predicted edges: filter corpus transitions to predicted stage set
        pred_edges = set()
        for key in transition_counts:
            src, dst = key.split("|")
            if src in pred_stage_set and dst in pred_stage_set:
                pred_edges.add((src, dst))

        # Unconditional metrics
        tp = len(pred_edges & true_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Conditional metrics: only edges where both endpoints are in both sets
        shared_stages = pred_stage_set & true_stage_set
        cond_true = {(s, d) for s, d in true_edges if s in shared_stages and d in shared_stages}
        cond_pred = {(s, d) for s, d in pred_edges if s in shared_stages and d in shared_stages}
        cond_tp += len(cond_pred & cond_true)
        cond_fp += len(cond_pred - cond_true)
        cond_fn += len(cond_true - cond_pred)

        # Self-loop diagnostics
        true_self = {(s, d) for s, d in true_edges if s == d}
        pred_self = {(s, d) for s, d in pred_edges if s == d}
        self_loop_tp += len(pred_self & true_self)
        self_loop_total_true += len(true_self)
        self_loop_total_pred += len(pred_self)

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    edge_p, edge_r, edge_f1 = _prf(total_tp, total_fp, total_fn)
    cond_p, cond_r, cond_f1 = _prf(cond_tp, cond_fp, cond_fn)

    logger.info(f"\nConnection evaluation ({docs_with_edges} docs with edges):")
    logger.info(f"  Unconditional — P={edge_p:.4f}  R={edge_r:.4f}  F1={edge_f1:.4f}")
    logger.info(f"  Conditional   — P={cond_p:.4f}  R={cond_r:.4f}  F1={cond_f1:.4f}")
    logger.info(f"  Self-loops    — TP={self_loop_tp}  true={self_loop_total_true}  "
                f"pred={self_loop_total_pred}")

    return {
        "docs_with_edges": docs_with_edges,
        "unconditional": {"precision": edge_p, "recall": edge_r, "f1": edge_f1},
        "conditional": {"precision": cond_p, "recall": cond_r, "f1": cond_f1},
        "self_loops": {
            "tp": self_loop_tp,
            "total_true": self_loop_total_true,
            "total_pred": self_loop_total_pred,
        },
    }


def _pseudo_label_rounds(X_labeled: np.ndarray, y: np.ndarray,
                          X_unlabeled: np.ndarray, stage_names: list[str],
                          chain_order: list[int], models: dict,
                          thresholds: dict, best_hyperparams: dict,
                          folds: int, n_rounds: int = 2,
                          cooc_matrix: np.ndarray | None = None):
    """Run pseudo-labeling rounds to expand training data.

    High-confidence predictions (>0.85 positive, <0.15 negative) on unlabeled
    docs are added as pseudo-labels.
    """
    N_labeled = X_labeled.shape[0]
    S = y.shape[1]

    for round_num in range(n_rounds):
        # Predict on unlabeled docs using chain
        N_unl = X_unlabeled.shape[0]
        unl_probs = np.zeros((N_unl, S), dtype=np.float64)

        # Co-occurrence features for unlabeled docs (use labeled co-occurrence matrix)
        # For unlabeled prediction, use zeros (no labels available)
        unl_cooc = np.zeros((N_unl, S), dtype=np.float64)

        for chain_pos, si in enumerate(chain_order):
            stage = stage_names[si]
            if stage not in models:
                continue
            clf = models[stage]
            prior_indices = chain_order[:chain_pos]
            aug_parts = [X_unlabeled]
            if prior_indices:
                aug_parts.append(unl_probs[:, prior_indices])
            aug_parts.append(unl_cooc)
            X_aug_unl = np.hstack(aug_parts)
            unl_probs[:, si] = clf.predict_proba(X_aug_unl)[:, 1]

        # Select high-confidence pseudo-labels with per-stage adaptive thresholds
        pseudo_labels = np.full((N_unl, S), -1.0)  # -1 = unknown
        active_si = [si for si in chain_order if stage_names[si] in models]
        for si in active_si:
            n_pos_labeled = int(y[:, si].sum())
            if n_pos_labeled >= 30:
                hi_t, lo_t = 0.85, 0.15
            elif n_pos_labeled >= 15:
                hi_t, lo_t = 0.75, 0.25
            else:
                hi_t, lo_t = 0.65, 0.35
            pseudo_labels[unl_probs[:, si] > hi_t, si] = 1.0
            pseudo_labels[unl_probs[:, si] < lo_t, si] = 0.0

        # Accept docs with >= 80% stage coverage (fill unknowns with 0)
        coverage = np.mean(pseudo_labels[:, active_si] >= 0, axis=1)
        has_enough = coverage >= 0.8
        for si in active_si:
            mask = has_enough & (pseudo_labels[:, si] < 0)
            pseudo_labels[mask, si] = 0.0
        n_pseudo = int(has_enough.sum())

        if n_pseudo == 0:
            logger.info(f"  Pseudo-label round {round_num + 1}: no confident docs, stopping")
            break

        logger.info(f"  Pseudo-label round {round_num + 1}: {n_pseudo} docs added "
                    f"(from {N_unl} unlabeled)")

        # Build expanded dataset
        pseudo_X = X_unlabeled[has_enough]
        pseudo_y = np.zeros((n_pseudo, S), dtype=np.float32)
        for si in active_si:
            pseudo_y[:, si] = pseudo_labels[has_enough, si]

        X_expanded = np.vstack([X_labeled, pseudo_X])
        y_expanded = np.vstack([y, pseudo_y])

        # Recompute co-occurrence from expanded labels
        cooc_expanded = _build_cooccurrence_priors(y_expanded)
        cooc_feats_expanded = _compute_cooccurrence_features(y_expanded, cooc_expanded)

        # Retrain all models on expanded data
        oof_chain_probs = np.zeros((X_expanded.shape[0], S), dtype=np.float64)

        for chain_pos, si in enumerate(chain_order):
            stage = stage_names[si]
            if stage not in models:
                continue

            y_col = y_expanded[:, si]
            n_pos = int(y_col.sum())
            n_neg = X_expanded.shape[0] - n_pos
            scale_pos = n_neg / max(n_pos, 1)

            prior_indices = chain_order[:chain_pos]
            aug_parts = [X_expanded]
            if prior_indices:
                aug_parts.append(oof_chain_probs[:, prior_indices])
            aug_parts.append(cooc_feats_expanded)
            X_aug = np.hstack(aug_parts)

            hp = best_hyperparams.get(stage, {})
            mcw = 1 if n_pos < 20 else 3 if n_pos < 50 else 5
            clf = XGBClassifier(
                n_estimators=hp.get("n_estimators", 300),
                learning_rate=hp.get("learning_rate", 0.05),
                max_depth=hp.get("max_depth", 4),
                colsample_bytree=hp.get("colsample_bytree", 0.7),
                reg_alpha=hp.get("reg_alpha", 0.0),
                reg_lambda=hp.get("reg_lambda", 1.0),
                min_child_weight=mcw,
                subsample=0.7, scale_pos_weight=scale_pos,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            clf.fit(X_aug, y_col)
            models[stage] = clf

            # Update chain probs for downstream stages
            oof_chain_probs[:, si] = clf.predict_proba(X_aug)[:, 1]

            # Re-find threshold on original labeled data only
            orig_probs = oof_chain_probs[:N_labeled, si]
            n_pos_orig = int(y[:, si].sum())
            thresholds[stage] = _find_threshold(y[:, si], orig_probs, n_pos=n_pos_orig)

    return models, thresholds


# ---------------------------------------------------------------------------
# SHAP interpretability
# ---------------------------------------------------------------------------

def _build_feat_to_section(feature_names: list[str]) -> dict:
    """Map feature index -> section index for aggregating importance by section."""
    n_per_section = len(NUMERIC_FEATURES) + len(DOMAIN_KEYWORDS) + len(TEXT_FEATURES) + 1
    feat_to_section = {}
    for si, section in enumerate(FEATURE_SECTION_TYPES):
        start = si * n_per_section
        end = start + n_per_section
        for fi in range(start, end):
            if fi < len(feature_names):
                feat_to_section[fi] = si
    return feat_to_section


def _build_influence_matrix(stage_names: list[str], importance_per_stage: dict,
                            feat_to_section: dict) -> np.ndarray:
    """Aggregate per-feature importance into (n_sections × n_stages) matrix."""
    n_sections = len(FEATURE_SECTION_TYPES)
    matrix = np.zeros((n_sections, len(stage_names)), dtype=np.float64)
    for stage_idx, stage in enumerate(stage_names):
        imp = importance_per_stage.get(stage)
        if imp is None:
            continue
        for fi, sec_idx in feat_to_section.items():
            if fi < len(imp):
                matrix[sec_idx, stage_idx] += imp[fi]
    # Normalize columns to [0, 1]
    col_max = matrix.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    matrix /= col_max
    return matrix


def _save_and_log_influences(matrix: np.ndarray, stage_names: list[str],
                             method: str, output_dir: str) -> dict:
    """Save influence matrix and log top influences per stage."""
    out = Path(output_dir)
    result = {
        "sections": FEATURE_SECTION_TYPES,
        "stages": stage_names,
        "influence_matrix": matrix.tolist(),
        "method": method,
    }
    with open(out / "section_influences.json", "w") as f:
        json.dump(result, f, indent=2)
    n_sections = len(FEATURE_SECTION_TYPES)
    logger.info(f"{method} influence matrix saved: {n_sections} sections × {len(stage_names)} stages")

    for stage_idx, stage in enumerate(stage_names):
        col = matrix[:, stage_idx]
        top_sections = np.argsort(col)[::-1][:3]
        top_str = ", ".join(
            f"{FEATURE_SECTION_TYPES[si]}={col[si]:.3f}" for si in top_sections
        )
        logger.info(f"  {stage}: {top_str}")
    return result


def compute_shap_influences(models: dict, X: np.ndarray,
                            feature_names: list[str],
                            chain_order: list[int] | None = None,
                            stage_names: list[str] | None = None,
                            y: np.ndarray | None = None,
                            output_dir: str = "/opt/ml/model"):
    """Compute SHAP values via XGBoost's built-in Tree SHAP (pred_contribs).

    For chain models, each model's input is augmented with prior chain
    predictions (zeros padded) so pred_contribs gets the right feature count.
    Only base-feature SHAP values are used for section attribution.

    Returns (n_sections × n_stages) matrix of section importance.
    """
    import xgboost as xgb

    sorted_stages = sorted(models.keys())
    feat_to_section = _build_feat_to_section(feature_names)
    n_base = X.shape[1]
    N = X.shape[0]

    # Co-occurrence features (must match training augmentation)
    cooc_feats = np.zeros((N, 0), dtype=np.float64)
    if y is not None:
        cooc_matrix = _build_cooccurrence_priors(y)
        cooc_feats = _compute_cooccurrence_features(y, cooc_matrix)

    # Build chain predictions for augmentation
    chain_preds = {}
    if chain_order is not None and stage_names is not None:
        for chain_pos, si in enumerate(chain_order):
            stage = stage_names[si]
            if stage not in models:
                continue
            # Build augmented X for this stage
            prior_indices = chain_order[:chain_pos]
            aug_parts = [X]
            if prior_indices:
                chain_feats = np.column_stack([
                    chain_preds.get(chain_order[p], np.zeros(N))
                    for p in range(chain_pos)
                ])
                aug_parts.append(chain_feats)
            if cooc_feats.shape[1] > 0:
                aug_parts.append(cooc_feats)
            X_aug = np.hstack(aug_parts) if len(aug_parts) > 1 else X
            # Get predictions for downstream stages
            clf = models[stage]
            chain_preds[si] = clf.predict_proba(X_aug)[:, 1]

    importance_per_stage = {}
    for stage in sorted_stages:
        clf = models[stage]
        booster = clf.get_booster()

        # Determine augmented X for this model
        if chain_order is not None and stage_names is not None:
            si = stage_names.index(stage)
            chain_pos = chain_order.index(si) if si in chain_order else -1
            prior_indices = chain_order[:chain_pos] if chain_pos > 0 else []
            aug_parts = [X]
            if prior_indices:
                chain_feats = np.column_stack([
                    chain_preds.get(chain_order[p], np.zeros(N))
                    for p in range(chain_pos)
                ])
                aug_parts.append(chain_feats)
            if cooc_feats.shape[1] > 0:
                aug_parts.append(cooc_feats)
            X_aug = np.hstack(aug_parts) if len(aug_parts) > 1 else X
        else:
            X_aug = np.hstack([X, cooc_feats]) if cooc_feats.shape[1] > 0 else X

        dmat = xgb.DMatrix(X_aug)
        # pred_contribs returns (N, D+1) — last col is bias term
        contribs = booster.predict(dmat, pred_contribs=True)
        feature_shap = contribs[:, :-1]  # (N, D)
        # Only use base features for section attribution
        feature_shap = feature_shap[:, :n_base]
        importance_per_stage[stage] = np.mean(np.abs(feature_shap), axis=0)

    matrix = _build_influence_matrix(sorted_stages, importance_per_stage,
                                     feat_to_section)
    return _save_and_log_influences(matrix, sorted_stages, "tree_shap",
                                    output_dir)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_stages(models: dict, thresholds: dict, feature_vec: np.ndarray,
                   transition_counts: dict, stage_vocab: list[str],
                   chain_order: list[int] | None = None,
                   var_selector: VarianceThreshold | None = None,
                   svd_models: dict | None = None,
                   doc_tfidf: dict | None = None) -> dict:
    """Predict stages for a single document feature vector.

    If chain_order is provided, predictions follow chain order and each stage's
    features are augmented with prior predictions.

    Returns dict with stages, connections, probabilities.
    """
    # Apply variance selector
    if var_selector is not None:
        fv = var_selector.transform(feature_vec.reshape(1, -1))[0]
    else:
        fv = feature_vec

    # Apply SVD features
    if svd_models is not None and doc_tfidf is not None:
        svd_feats = transform_tfidf_svd(doc_tfidf, svd_models)
        if var_selector is not None:
            # SVD features were appended after variance selection during training
            fv = np.concatenate([fv, svd_feats])
        else:
            fv = np.concatenate([fv, svd_feats])

    stage_probs = {}
    S = len(stage_vocab)

    if chain_order is not None:
        # Chain prediction: predict in order, feed results forward
        chain_preds = np.zeros(S, dtype=np.float64)
        for chain_pos, si in enumerate(chain_order):
            stage = stage_vocab[si]
            if stage not in models:
                continue
            clf = models[stage]
            prior_indices = chain_order[:chain_pos]
            if prior_indices:
                chain_feats = chain_preds[prior_indices]
                x_aug = np.concatenate([fv, chain_feats])
            else:
                x_aug = fv
            prob = float(clf.predict_proba(x_aug.reshape(1, -1))[0, 1])
            chain_preds[si] = prob
            threshold = thresholds.get(stage, 0.5)
            if prob >= threshold:
                stage_probs[stage] = round(prob, 4)
    else:
        # Fallback: independent predictions
        for stage in stage_vocab:
            if stage not in models:
                continue
            clf = models[stage]
            prob = float(clf.predict_proba(fv.reshape(1, -1))[0, 1])
            threshold = thresholds.get(stage, 0.5)
            if prob >= threshold:
                stage_probs[stage] = round(prob, 4)

    if not stage_probs:
        # Fallback: take top 5 by probability
        all_probs = {}
        for stage in stage_vocab:
            if stage not in models:
                continue
            prob = float(models[stage].predict_proba(fv.reshape(1, -1))[0, 1])
            all_probs[stage] = prob
        top5 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        stage_probs = {s: round(p, 4) for s, p in top5}

    # Build connections from transition counts
    stage_set = set(stage_probs.keys())
    connections = []
    for key, count in transition_counts.items():
        src, dst = key.split("|")
        if src in stage_set and dst in stage_set:
            connections.append({
                "from": src, "to": dst,
                "confidence": round(count / 100.0, 4),
                "corpus_count": count,
            })

    # Topological sort
    ordered = _topological_sort(stage_set, connections)

    return {
        "stages": ordered,
        "connections": connections,
        "stage_probabilities": dict(sorted(stage_probs.items(),
                                           key=lambda x: x[1], reverse=True)),
    }


def _topological_sort(stages: set, connections: list) -> list:
    """Order stages by process flow using transition edges."""
    adjacency = defaultdict(list)
    in_degree = defaultdict(int)
    for stage in stages:
        in_degree[stage] = 0
    for conn in connections:
        adjacency[conn["from"]].append(conn["to"])
        in_degree[conn["to"]] += 1

    queue = sorted(s for s in stages if in_degree[s] == 0)
    ordered = []
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for neighbor in sorted(adjacency[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        queue.sort()

    remaining = sorted(s for s in stages if s not in set(ordered))
    ordered.extend(remaining)
    return ordered


# ---------------------------------------------------------------------------
# Neighbor features from cross-doc similarity
# ---------------------------------------------------------------------------

def _load_cross_doc_edges(graphs_bucket: str, graphs_prefix: str) -> dict:
    """Load cross_doc_edges.json from S3.

    Returns dict mapping doc_id -> list of {doc_id, overall_similarity, ...}.
    """
    s3 = boto3.client("s3")
    key = f"{graphs_prefix}cross_doc_edges.json"
    try:
        resp = s3.get_object(Bucket=graphs_bucket, Key=key)
        edges = json.loads(resp["Body"].read())
        logger.info(f"Loaded cross-doc edges for {len(edges)} documents")
        return edges
    except Exception as e:
        logger.warning(f"Could not load cross_doc_edges.json: {e}")
        return {}


def _compute_neighbor_features(X: np.ndarray, doc_ids: list[str],
                                edges: dict, k: int = 10) -> np.ndarray:
    """Compute neighbor-averaged features from cross-doc similarity.

    For each doc, weighted-average the features of its top-k labeled neighbors,
    then compute the element-wise difference (X_doc - X_neighbor_avg) as features.

    Args:
        X: (N, D) feature matrix
        doc_ids: list of N document IDs
        edges: cross-doc edge dict from S3
        k: max neighbors to use

    Returns:
        (N, D) neighbor difference features
    """
    N, D = X.shape
    doc_to_idx = {did: i for i, did in enumerate(doc_ids)}
    neighbor_feats = np.zeros((N, D), dtype=np.float64)

    for i, doc_id in enumerate(doc_ids):
        doc_edges = edges.get(doc_id, {})
        neighbors = doc_edges.get("neighbors", [])
        if not neighbors:
            continue

        # Filter to neighbors that are in our labeled set, take top-k
        valid = []
        for nb in neighbors:
            nb_id = nb.get("doc_id", "")
            if nb_id in doc_to_idx:
                valid.append((doc_to_idx[nb_id], nb.get("overall_similarity", 0.5)))
            if len(valid) >= k:
                break

        if not valid:
            continue

        indices, weights = zip(*valid)
        weights = np.array(weights, dtype=np.float64)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        else:
            weights = np.ones(len(weights)) / len(weights)

        neighbor_avg = np.average(X[list(indices)], axis=0, weights=weights)
        neighbor_feats[i] = X[i] - neighbor_avg

    n_with_neighbors = int((np.abs(neighbor_feats).sum(axis=1) > 0).sum())
    logger.info(f"Neighbor features: {n_with_neighbors}/{N} docs have neighbors, "
                f"{D} features")
    return neighbor_feats


# ---------------------------------------------------------------------------
# SageMaker entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost stage predictors")
    parser.add_argument("--graphs-bucket", type=str, required=True)
    parser.add_argument("--graphs-prefix", type=str, default="graphs/")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--test-pct", type=float, default=0.10,
                        help="Fraction of labeled docs for test set (default: 0.10)")
    parser.add_argument("--val-pct", type=float, default=0.30,
                        help="Fraction of labeled docs for validation set "
                             "(default: 0.30, was 0.20 — larger val reduces "
                             "sample-variance noise on the headline F1)")
    parser.add_argument("--vocab-version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="Stage vocabulary: v1 (23 stages) or v2 (37 stages)")
    args = parser.parse_args()

    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    logger.info(f"Loading data from s3://{args.graphs_bucket}/{args.graphs_prefix}")
    t0 = time.time()
    data = load_training_data(args.graphs_bucket, args.graphs_prefix,
                              min_support=args.min_support,
                              vocab_version=args.vocab_version)

    # Step 3: Fit TF-IDF SVD on ALL docs (labeled + unlabeled)
    logger.info("Fitting TF-IDF SVD...")
    svd_models = fit_tfidf_svd(data["tfidf_all"])

    # Transform and append SVD features to X and X_unlabeled
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

    # Update feature names
    feature_names = data["feature_names"] + build_svd_feature_names(svd_models)

    # Load cross-doc edges and compute neighbor features
    edges = _load_cross_doc_edges(args.graphs_bucket, args.graphs_prefix)
    if edges:
        neighbor_feats = _compute_neighbor_features(
            X, data["doc_ids"], edges, k=10)
        X = np.hstack([X, neighbor_feats])
        neighbor_names = [f"neighbor_diff_{i}" for i in range(neighbor_feats.shape[1])]
        feature_names = feature_names + neighbor_names

        if X_unlabeled.shape[0] > 0:
            neighbor_feats_unl = _compute_neighbor_features(
                X_unlabeled, data["unlabeled_ids"], edges, k=10)
            X_unlabeled = np.hstack([X_unlabeled, neighbor_feats_unl])

    logger.info(f"Total features: {len(feature_names)}")

    # --- 70 / 20 / 10 Train / Validation / Test split ---
    from sklearn.model_selection import train_test_split
    N = X.shape[0]
    holdout_pct = args.val_pct + args.test_pct
    X_test, y_test, test_doc_ids = None, None, []
    X_val, y_val, val_doc_ids = None, None, []

    if holdout_pct > 0 and N > 30:
        indices = np.arange(N)

        # Split: train (70%) vs holdout (30%)
        train_idx, holdout_idx = train_test_split(
            indices, test_size=holdout_pct, random_state=42,
        )

        # Split holdout: val (20%) vs test (10%)  — relative sizes within holdout
        if args.test_pct > 0 and args.val_pct > 0:
            test_frac_of_holdout = args.test_pct / holdout_pct
            val_idx, test_idx = train_test_split(
                holdout_idx, test_size=test_frac_of_holdout, random_state=42,
            )
        elif args.test_pct > 0:
            val_idx, test_idx = np.array([], dtype=int), holdout_idx
        else:
            val_idx, test_idx = holdout_idx, np.array([], dtype=int)

        if len(test_idx) > 0:
            X_test = X[test_idx]
            y_test = data["y"][test_idx]
            test_doc_ids = [data["doc_ids"][i] for i in test_idx]

        if len(val_idx) > 0:
            X_val = X[val_idx]
            y_val = data["y"][val_idx]
            val_doc_ids = [data["doc_ids"][i] for i in val_idx]

        # Remove held-out docs from training data
        X = X[train_idx]
        data["y"] = data["y"][train_idx]
        data["doc_ids"] = [data["doc_ids"][i] for i in train_idx]
        data["tfidf_labeled"] = [data["tfidf_labeled"][i] for i in train_idx]
        if "doc_connections" in data:
            data["doc_connections"] = [data["doc_connections"][i] for i in train_idx]

        logger.info(f"Split: {len(train_idx)} train ({1-holdout_pct:.0%}), "
                    f"{len(val_idx)} val ({args.val_pct:.0%}), "
                    f"{len(test_idx)} test ({args.test_pct:.0%})")
    else:
        logger.info(f"No split (N={N})")

    logger.info("Training XGBoost classifier chain...")
    result = train_stage_models(
        X, data["y"], data["stage_vocab"], feature_names,
        folds=args.folds, output_dir=output_dir,
        X_unlabeled=X_unlabeled,
    )

    # Evaluate connection accuracy (corpus-frequency baseline vs per-doc ground truth)
    conn_metrics = evaluate_connections(
        data["y"], result["oof_probs"], data["stage_vocab"],
        result["thresholds"], data["transition_counts"],
        data["doc_connections"],
    )
    out = Path(output_dir)
    with open(out / "connection_metrics.json", "w") as f:
        json.dump(conn_metrics, f, indent=2)

    # Save per-doc OOF predictions for hybrid LLM eval
    oof_probs = result["oof_probs"]
    oof_preds = []
    for i, doc_id in enumerate(data["doc_ids"]):
        pred_stages = {}
        gt_stages = []
        for j, stage in enumerate(data["stage_vocab"]):
            prob = float(oof_probs[i, j])
            if prob >= result["thresholds"].get(stage, 0.5):
                pred_stages[stage] = round(prob, 4)
            if data["y"][i, j] > 0:
                gt_stages.append(stage)
        oof_preds.append({
            "doc_id": doc_id,
            "ground_truth": gt_stages,
            "xgb_predicted": pred_stages,
            "xgb_stages": sorted(pred_stages, key=lambda s: pred_stages[s], reverse=True),
        })
    with open(out / "oof_predictions.json", "w") as f:
        json.dump(oof_preds, f, indent=2)
    logger.info(f"Saved OOF predictions for {len(oof_preds)} labeled docs")

    # --- Evaluate on held-out val and test sets ---
    def _eval_holdout(X_held, y_held, held_doc_ids, split_name):
        """Run XGBoost chain on held-out docs, save per-doc predictions."""
        if X_held is None or len(held_doc_ids) == 0:
            return []

        models = result["models"]
        thresholds = result["thresholds"]
        chain_order = result["chain_order"]
        var_selector = result["var_selector"]
        stage_names = data["stage_vocab"]
        S = len(stage_names)

        # Apply variance selector
        X_sel = var_selector.transform(X_held)

        # Build co-occurrence priors from training labels
        cooc_matrix = _build_cooccurrence_priors(data["y"])

        # Chain prediction on held-out
        N_h = X_sel.shape[0]
        held_probs = np.zeros((N_h, S), dtype=np.float64)
        for chain_pos, si in enumerate(chain_order):
            stage = stage_names[si]
            if stage not in models:
                continue
            clf = models[stage]
            prior_indices = chain_order[:chain_pos]
            cooc_feats = _compute_cooccurrence_features(y_held, cooc_matrix) if chain_pos == 0 else cooc_held
            if chain_pos == 0:
                cooc_held = cooc_feats

            parts = [X_sel]
            if prior_indices:
                parts.append(held_probs[:, prior_indices])
            parts.append(cooc_feats)
            X_aug = np.hstack(parts)

            held_probs[:, si] = clf.predict_proba(X_aug)[:, 1]

        # Build per-doc results
        held_preds = []
        y_true_list, y_pred_list = [], []
        for i, doc_id in enumerate(held_doc_ids):
            gt_stages = [stage_names[j] for j in range(S) if y_held[i, j] > 0]
            pred_stages = {}
            for j, stage in enumerate(stage_names):
                prob = float(held_probs[i, j])
                if prob >= thresholds.get(stage, 0.5):
                    pred_stages[stage] = round(prob, 4)

            pred_list = sorted(pred_stages, key=lambda s: pred_stages[s], reverse=True)
            correct = set(gt_stages) & set(pred_list)
            missed = set(gt_stages) - set(pred_list)
            false_pos = set(pred_list) - set(gt_stages)
            n_gt = len(gt_stages)
            n_pred = len(pred_list)
            n_correct = len(correct)
            doc_f1 = (2 * n_correct / (n_gt + n_pred)) if (n_gt + n_pred) > 0 else 0.0

            held_preds.append({
                "doc_id": doc_id,
                "ground_truth": sorted(gt_stages),
                "xgb_predicted": pred_stages,
                "xgb_stages": pred_list,
                "correct": sorted(correct),
                "missed": sorted(missed),
                "false_positive": sorted(false_pos),
                "doc_f1": round(doc_f1, 4),
            })
            y_true_list.append(y_held[i])
            y_pred_list.append(
                np.array([(1 if stage_names[j] in pred_stages else 0) for j in range(S)])
            )

        # Aggregate metrics
        y_t = np.array(y_true_list)
        y_p = np.array(y_pred_list)
        macro = f1_score(y_t, y_p, average="macro", zero_division=0)
        micro = f1_score(y_t, y_p, average="micro", zero_division=0)
        prec = precision_score(y_t, y_p, average="macro", zero_division=0)
        rec = recall_score(y_t, y_p, average="macro", zero_division=0)

        logger.info(f"\n{split_name} set ({len(held_doc_ids)} docs):")
        logger.info(f"  macro-F1={macro:.4f}  micro-F1={micro:.4f}  P={prec:.4f}  R={rec:.4f}")

        # Per-stage on held-out
        per_stage_held = {}
        for j, stage in enumerate(stage_names):
            support = int(y_t[:, j].sum())
            if support == 0:
                continue
            sf1 = f1_score(y_t[:, j], y_p[:, j], zero_division=0)
            per_stage_held[stage] = {"f1": round(float(sf1), 4), "support": support}

        return {
            "metrics": {
                "macro_f1": round(float(macro), 4),
                "micro_f1": round(float(micro), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "num_docs": len(held_doc_ids),
            },
            "per_stage": per_stage_held,
            "per_doc": held_preds,
        }

    val_results = _eval_holdout(X_val, y_val, val_doc_ids, "Validation")
    test_results = _eval_holdout(X_test, y_test, test_doc_ids, "Test")

    if val_results:
        with open(out / "val_predictions.json", "w") as f:
            json.dump(val_results, f, indent=2)
        logger.info(f"Saved validation predictions ({val_results['metrics']['num_docs']} docs)")

    if test_results:
        with open(out / "test_predictions.json", "w") as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Saved test predictions ({test_results['metrics']['num_docs']} docs)")

    logger.info("Computing SHAP influences...")
    # For SHAP, use base features only (before chain augmentation)
    # Pass the variance-selected features
    var_selector = result["var_selector"]
    X_shap = var_selector.transform(X)
    shap_feature_names = [f for f, k in zip(feature_names, var_selector.get_support()) if k]
    compute_shap_influences(
        result["models"], X_shap, shap_feature_names,
        chain_order=result["chain_order"],
        stage_names=data["stage_vocab"],
        y=data["y"],
        output_dir=output_dir,
    )

    # Save metadata for inference
    out = Path(output_dir)
    with open(out / "stage_vocab.json", "w") as f:
        json.dump(data["stage_vocab"], f, indent=2)
    with open(out / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(out / "transition_counts.json", "w") as f:
        json.dump(data["transition_counts"], f, indent=2)
    with open(out / "svd_models.pkl", "wb") as f:
        pickle.dump(svd_models, f)

    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.0f}s")
    logger.info(f"Models: {len(result['models'])}, stages: {len(data['stage_vocab'])}")


if __name__ == "__main__":
    main()
