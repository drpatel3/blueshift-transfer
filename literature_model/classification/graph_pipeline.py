"""Document context graph builder — runs on SageMaker GPU.

Reads per-document result JSONs, extracts section features, scores
table-text pairs with DeBERTa, fits corpus-wide TF-IDF, and builds
per-document graphs for Phase 3 GNN prediction.

Usage (SageMaker entry script):
    Channels: training (result JSONs), model (DeBERTa tar.gz)
    Hyperparameters: batch-size, relevance-threshold
"""

import os as _os
# Reduce glibc heap fragmentation — must be set before first malloc
_os.environ.setdefault("MALLOC_ARENA_MAX", "2")
_os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "0")
# Prevent PyTorch CUDA allocator from reserving large contiguous virtual
# address blocks — reduces driver-side host memory overhead.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                        "expandable_segments:False")
# Delay CUDA module loading to reduce initial driver memory footprint.
_os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import ctypes
import gc
import json
import logging
import os
import pickle
import math
import random
import re
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section grouping — maps NI 43-101 section numbers to context node types
# ---------------------------------------------------------------------------

SECTION_GROUPS = {
    2: "summary", 3: "summary",
    4: "property",
    5: "climate",
    6: "history",
    7: "geology", 8: "geology",
    9: "exploration",
    10: "drilling",
    11: "sample_analysis",
    12: "data_verification",
    13: "metallurgical_testing",
    14: "resource_estimate", 15: "resource_estimate",
    16: "mining_method",
    17: "recovery_methods",
    18: "infrastructure",
    19: "market",
    20: "environmental",
    21: "economics", 22: "economics",
    23: "interpretation",
    24: "references", 25: "references", 26: "references", 27: "references",
}

ALL_SECTION_TYPES = sorted(set(SECTION_GROUPS.values()))

# ---------------------------------------------------------------------------
# Domain keywords for feature extraction
# ---------------------------------------------------------------------------

EQUIPMENT_KEYWORDS = [
    "crusher", "mill", "flotation", "leach", "thickener", "filter",
    "electrowinning", "gravity", "elution", "solvent_extraction",
    "cyclone", "screen", "kiln", "reactor", "adsorption", "precipitation",
    "stockpile", "hopper", "feeder", "conveyor", "tank", "agglomeration",
    "magnetic_separation", "merrill_crowe", "ion_exchange", "drying",
    "water_treatment", "regrind", "cell", "ore_sorting",
]

MINERALOGY_KEYWORDS = [
    "chalcopyrite", "bornite", "pyrite", "chalcocite", "covellite",
    "enargite", "arsenopyrite", "molybdenite", "galena", "sphalerite",
    "magnetite", "hematite", "goethite", "malachite", "azurite", "chrysocolla",
]

DEPOSIT_KEYWORDS = [
    "porphyry", "skarn", "vms", "iocg", "sedimentary", "epithermal",
    "orogenic", "breccia", "intrusive", "volcanic",
]

ORE_KEYWORDS = [
    "oxide", "sulfide", "sulphide", "supergene", "hypogene",
    "transition", "refractory", "free_milling",
]

ECONOMICS_KEYWORDS = [
    "tonnage", "grade", "recovery", "npv", "irr", "capex", "opex",
    "payback", "cut_off", "measured", "indicated", "inferred",
]

MINING_KEYWORDS = [
    "open_pit", "underground", "block_cave", "stoping", "strip_ratio",
]

CLIMATE_KEYWORDS = [
    "rainfall", "arid", "water_availability", "tailings_dam", "closure",
]

DOMAIN_KEYWORDS = (
    EQUIPMENT_KEYWORDS + MINERALOGY_KEYWORDS + DEPOSIT_KEYWORDS +
    ORE_KEYWORDS + ECONOMICS_KEYWORDS + MINING_KEYWORDS + CLIMATE_KEYWORDS
)

# ---------------------------------------------------------------------------
# Normalization — copied from pipeline/normalize_ids.py (SageMaker self-contained)
# ---------------------------------------------------------------------------

SYNONYMS = {
    'cells': 'cell', 'tailings': 'tailing', 'tanks': 'tank',
    'crushing': 'crusher', 'grinding': 'mill', 'screening': 'screen',
    'thickening': 'thickener', 'filtering': 'filter', 'filtration': 'filter',
    'leaching': 'leach',
    'cyclones': 'cyclone', 'mills': 'mill', 'screens': 'screen',
    'feeders': 'feeder', 'hoppers': 'hopper', 'conveyors': 'conveyor',
    'grizzly': 'screen', 'grasshopper': None, 'furnace': 'kiln',
    'baghouse': 'filter', 'scrubber': 'filter', 'classification': 'cyclone',
    'conditioning': 'tank', 'cleaners': 'cleaner', 'cleaner': 'flotation',
    'rougher': 'flotation', 'decantation': 'thickener',
    'counter': None, 'current': None,
    'separation': 'thickener', 'solid': None, 'liquid': None,
    'vessel': 'tank', 'feed': 'feeder', 'reception': 'hopper',
    'cic': 'adsorption', 'columns': 'adsorption',
    'INPUT': 'input', 'OUTPUT': 'output',
    'jaw': None, 'cone': None, 'gyratory': None,
    'primary': None, 'secondary': None, 'tertiary': None,
    '1st': None, '2nd': None, '3rd': None,
    'i': None, 'ii': None, 'iii': None,
    'celda': 'cell', 'celdas': 'cell', 'chancadora': 'crusher',
    'molino': 'mill', 'bolas': None, 'ciclones': 'cyclone',
    'ciclon': 'cyclone', 'hidrociclon': 'cyclone', 'hydrocyclone': 'cyclone',
    'espesador': 'thickener', 'filtro': 'filter',
    'tolva': 'hopper', 'tolvas': 'hopper',
    'faja': 'conveyor', 'zaranda': 'screen',
    'relavera': 'tailing', 'relaves': 'tailing',
    'tanque': 'tank', 'tanques': 'tank',
    'limpieza': None, 'primaria': None, 'primarias': None,
    'secundaria': None, 'terciaria': None, 'recepcion': None,
    'vibratoria': None, 'de': None,
    'dewatering': 'filter', 'storage': 'bin', 'disposal': 'tailing',
    'drying': 'filter', 'dryer': 'filter', 'recycle': 'pond',
    'supply': 'tank', 'tmf': 'tailing',
    'shipping': None, 'handling': None, 'plant': None, 'facility': None,
    'collection': None, 'circuit': None, 'room': None, 'reagents': None,
    'final': None, 'coarse': None, 'fine': None, 'loaded': None,
    'barren': None, 'carbon': None, 'safety': None, 'sizing': None,
    'fines': None, 'attritioning': None, 'water': None,
    'scavenger': 'flotation', 'recleaner': 'flotation',
    'flash': 'flotation', 'jameson': None, 'incl': None,
    'piscina': 'pond', 'poza': 'pond',
    'contingencia': None, 'finos': None, 'gruesos': None,
    'prensa': None, 'quijada': None, 'conica': None,
    'concentrado': None, 'relave': 'tailing', 'agua': None,
    'clarificada': None, 'pirita': 'flotation',
    'cus': None, 'smbs': None, 'rom': 'input',
    'sorting': 'ore_sorting', 's': None, 'py': None, 'no': None,
}

REMOVABLE = {
    'copper', 'cu', 'gold', 'au', 'iron', 'fe', 'lead', 'pb', 'zinc', 'zn',
    'moly', 'silver', 'polymetallic', 'pyrite',
}

CORE_WORDS = [
    'input', 'output', 'regrind', 'flotation', 'crusher', 'mill', 'cyclone',
    'thickener', 'filter', 'stockpile', 'hopper', 'feeder', 'screen', 'tank',
    'conveyor', 'leach', 'precipitation', 'cell', 'reactor',
    'pond', 'bin', 'silo', 'kiln', 'furnace', 'adsorption',
    'tailing', 'ore_sorting', 'mol', 'sars',
]


def normalize_id(id_str: str) -> str:
    """Produce a simplified grouping key."""
    normalized = id_str.lower()
    normalized = re.sub(r'_unit$', '', normalized)
    normalized = re.sub(r'_\d+$', '', normalized)
    normalized = re.sub(r'_[a-b]$', '', normalized)
    tokens = normalized.split('_')
    tokens = [t for t in tokens if t not in REMOVABLE]
    if not tokens:
        return normalized
    normalized_tokens = []
    for t in tokens:
        if t in SYNONYMS:
            if SYNONYMS[t] is not None:
                normalized_tokens.append(SYNONYMS[t])
        else:
            normalized_tokens.append(t)
    if not normalized_tokens:
        return '_'.join(tokens)
    for core in CORE_WORDS:
        if core in normalized_tokens:
            return core
    joined = '_'.join(normalized_tokens)
    if joined in CORE_WORDS:
        return joined
    return joined


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def group_by_section(doc: dict) -> dict:
    """Group tables and page_text entries by SECTION_GROUPS key.

    Returns {group_name: {"tables": [...], "page_texts": [...]}}.
    """
    groups = defaultdict(lambda: {"tables": [], "page_texts": []})

    # Build page -> section_number map from page_text entries
    page_section_map = {}
    for pt in doc.get("page_text", []):
        sec = pt.get("section_number")
        page = pt.get("page")
        if sec and page:
            try:
                sec_int = int(sec)
                if sec_int in SECTION_GROUPS:
                    page_section_map[page] = SECTION_GROUPS[sec_int]
            except (ValueError, TypeError):
                pass

    # Assign page_text entries
    for pt in doc.get("page_text", []):
        sec = pt.get("section_number")
        group = None
        if sec:
            try:
                group = SECTION_GROUPS.get(int(sec))
            except (ValueError, TypeError):
                pass
        if group:
            groups[group]["page_texts"].append(pt)

    # Assign tables — use section_number if present, else page proximity
    for table in doc.get("tables", []):
        sec = table.get("section_number")
        group = None
        if sec:
            try:
                group = SECTION_GROUPS.get(int(sec))
            except (ValueError, TypeError):
                pass
        if not group:
            # Fallback: assign by page proximity
            t_page = table.get("page")
            if t_page and page_section_map:
                closest_page = min(page_section_map.keys(),
                                   key=lambda p: abs(p - t_page))
                if abs(closest_page - t_page) <= 3:
                    group = page_section_map[closest_page]
        if group:
            groups[group]["tables"].append(table)

    return dict(groups)


def _parse_numeric(cell_text: str) -> float | None:
    """Try to parse a numeric value from a table cell."""
    if not cell_text or not isinstance(cell_text, str):
        return None
    cleaned = cell_text.strip().replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


# Column-header patterns for targeted numeric extraction.
# Each entry: (regex_pattern, min_valid, max_valid)
_COLUMN_PATTERNS = {
    "grade_cu": (re.compile(r'\bcu\b|\bcopper\b|\bcu\s*[(%]', re.I), 0.01, 10.0),
    "grade_au": (re.compile(r'\bau\b|\bgold\b|g/t\s*au', re.I), 0.01, 100.0),
    "grade_ag": (re.compile(r'\bag\b|\bsilver\b|g/t\s*ag', re.I), 0.1, 1000.0),
    "recovery": (re.compile(r'\brecov\w*\b|\brec\s*%', re.I), 1.0, 100.0),
    "tonnage":  (re.compile(r'\btonn\w*\b|\btons\b|\bmt\b', re.I), 0.001, 1e9),
    "grind_size": (re.compile(r'\bp80\b|\bgrind\b|\bmicron\b|\bμm\b', re.I), 10.0, 500.0),
    "npv":      (re.compile(r'\bnpv\b', re.I), 0.0, 1e12),
    "capex":    (re.compile(r'\bcapex\b|\bcapital\s*cost', re.I), 0.0, 1e12),
    "opex":     (re.compile(r'\bopex\b|\boperating\s*cost', re.I), 0.0, 1e12),
}


def extract_numeric_features(tables: list[dict]) -> dict:
    """Extract numeric summary features from tables in a section group.

    Two extraction methods:
      1. Aggregate stats (legacy): mean/median/max of all parsed numbers
      2. Targeted extraction: match column headers to known patterns,
         extract values with range validation, store min/mean/max per type
    """
    all_nums = []
    header_flags = {"grade": 0, "recovery": 0, "tonnage": 0, "cost": 0, "npv": 0, "irr": 0}
    # Targeted: collect values per pattern type across all tables in section
    targeted_values = {name: [] for name in _COLUMN_PATTERNS}

    for table in tables:
        headers = table.get("headers", []) or []
        rows = table.get("rows", []) or []
        headers_text = " ".join(headers).lower()
        caption_text = (table.get("caption", "") or "").lower()
        combined_header = headers_text + " " + caption_text

        # Legacy: binary header flags
        for flag in header_flags:
            if flag in combined_header:
                header_flags[flag] = 1

        # Legacy: aggregate numeric stats (first 10 rows)
        for row in rows[:10]:
            for cell in row:
                val = _parse_numeric(cell)
                if val is not None:
                    all_nums.append(val)

        # Targeted: match column headers to patterns, extract from ALL rows
        for col_idx, header in enumerate(headers):
            header_str = str(header).lower()
            for name, (pattern, lo, hi) in _COLUMN_PATTERNS.items():
                if not pattern.search(header_str):
                    continue
                for row in rows:
                    if col_idx >= len(row):
                        continue
                    val = _parse_numeric(row[col_idx])
                    if val is not None and lo <= val <= hi:
                        targeted_values[name].append(val)

    # Legacy features
    features = {
        "num_count": len(all_nums),
        "num_mean": mean(all_nums) if all_nums else 0.0,
        "num_median": median(all_nums) if all_nums else 0.0,
        "num_max": max(all_nums) if all_nums else 0.0,
    }
    features.update(header_flags)

    # Targeted features: min/mean/max per type
    for name, values in targeted_values.items():
        if values:
            features[f"{name}_min"] = min(values)
            features[f"{name}_mean"] = mean(values)
            features[f"{name}_max"] = max(values)
        else:
            features[f"{name}_min"] = 0.0
            features[f"{name}_mean"] = 0.0
            features[f"{name}_max"] = 0.0

    return features


def extract_text_features(page_texts: list[dict], tables: list[dict]) -> dict:
    """Extract text-based features from a section group."""
    # Concatenate all text
    all_text_parts = []
    for pt in page_texts:
        text = pt.get("text", "")
        if text:
            all_text_parts.append(text)
    for table in tables:
        caption = table.get("caption", "")
        if caption:
            all_text_parts.append(caption)

    full_text = " ".join(all_text_parts).lower()

    # Count domain keywords
    keyword_counts = {}
    for kw in DOMAIN_KEYWORDS:
        # Handle multi-word keywords (e.g. open_pit -> "open pit" or "open_pit")
        pattern = kw.replace("_", "[_ ]")
        keyword_counts[f"kw_{kw}"] = len(re.findall(pattern, full_text))

    features = {
        "char_count": len(full_text),
        "num_pages": len(set(pt.get("page", 0) for pt in page_texts)),
    }
    features.update(keyword_counts)
    return features


def serialize_table(table: dict) -> str:
    """Flatten a table dict into a single string for model input."""
    caption = table.get("caption", "") or ""
    headers = " ".join(table.get("headers", []) or [])
    rows = table.get("rows", []) or []
    flat_rows = " ".join(
        " ".join(cell for cell in row) for row in rows[:5]
    )
    return f"{caption} | {headers} | {flat_rows}".strip()


# ---------------------------------------------------------------------------
# DeBERTa scoring
# ---------------------------------------------------------------------------

def load_deberta(model_dir: str, device: str):
    """Load DeBERTa model and tokenizer from a directory (possibly tar.gz)."""
    model_path = Path(model_dir)

    # If model_dir contains a tar.gz, extract it
    tar_files = list(model_path.glob("*.tar.gz"))
    if tar_files:
        extract_dir = model_path / "extracted"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True)
            logger.info(f"Extracting {tar_files[0]} ...")
            with tarfile.open(tar_files[0], "r:gz") as tar:
                tar.extractall(extract_dir)
        model_path = extract_dir

    # Find the actual model files (may be nested)
    config_files = list(model_path.rglob("config.json"))
    if config_files:
        model_path = config_files[0].parent

    logger.info(f"Loading DeBERTa from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def fit_tfidf(all_section_texts: list[str], max_features: int = 200) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on all section-group texts across corpus."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    vectorizer.fit(all_section_texts)
    logger.info(f"TF-IDF fitted on {len(all_section_texts)} section texts, "
                f"{len(vectorizer.vocabulary_)} features")
    return vectorizer


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def build_document_graph(doc: dict, section_features: dict, doc_scores: list,
                         tfidf_vectorizer: TfidfVectorizer,
                         section_texts: dict,
                         relevance_threshold: float = 0.5) -> dict:
    """Build a single document's context graph.

    Args:
        doc: Raw document dict
        section_features: {group_name: {numeric + text features}}
        doc_scores: List of {table_idx, page_number, group, score}
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        section_texts: {group_name: concatenated text}
        relevance_threshold: Minimum DeBERTa score for context→stage edges
    """
    doc_id = doc.get("document_id", doc.get("filename", "unknown"))
    nodes = []
    edges = []

    # --- Context nodes ---
    context_node_ids = []
    for group_name, features in section_features.items():
        # TF-IDF vector for this section
        text = section_texts.get(group_name, "")
        if text:
            tfidf_vec = tfidf_vectorizer.transform([text]).toarray()[0].tolist()
        else:
            tfidf_vec = [0.0] * len(tfidf_vectorizer.vocabulary_)

        # Section index (ordinal position in ALL_SECTION_TYPES)
        section_idx = ALL_SECTION_TYPES.index(group_name) if group_name in ALL_SECTION_TYPES else -1

        node_id = f"ctx_{group_name}"
        node_features = {
            **features,
            "tfidf": tfidf_vec,
            "section_index": section_idx,
        }
        nodes.append({
            "id": node_id,
            "type": "context",
            "group": group_name,
            "features": node_features,
        })
        context_node_ids.append(node_id)

    # --- Context cooccurrence edges (global TF-IDF cosine similarity) ---
    if len(context_node_ids) >= 2:
        # Use the global TF-IDF vectors already stored in node features
        tfidf_vectors = []
        for node in nodes:
            if node["type"] == "context":
                tfidf_vectors.append(node["features"]["tfidf"])
        tfidf_matrix = np.array(tfidf_vectors, dtype=np.float32)
        sim_matrix = cosine_similarity(tfidf_matrix)
        for i in range(len(context_node_ids)):
            for j in range(i + 1, len(context_node_ids)):
                sim = float(sim_matrix[i, j])
                if sim > COOCCURRENCE_THRESHOLD:
                    edges.append({
                        "source": context_node_ids[i],
                        "target": context_node_ids[j],
                        "type": "section_cooccurrence",
                        "weight": round(sim, 4),
                    })

    # --- Stage nodes (from flowsheets) ---
    stage_nodes = {}
    stage_sequence = []
    flowsheets_raw = doc.get("flowsheets", {}) or {}
    # flowsheets is a dict keyed by name; normalize to list of dicts
    if isinstance(flowsheets_raw, dict):
        flowsheets = list(flowsheets_raw.values())
    else:
        flowsheets = list(flowsheets_raw)
    all_connections = []

    for fs in flowsheets:
        stages = fs.get("stages", []) or []
        connections = fs.get("connections", []) or []
        all_connections.extend(connections)

        for idx, stage in enumerate(stages):
            stage_id = stage.get("id", "")
            norm_id = normalize_id(stage_id)
            if norm_id not in stage_nodes:
                stage_nodes[norm_id] = {
                    "original_ids": [],
                    "order": idx,
                    "in_degree": 0,
                    "out_degree": 0,
                }
            stage_nodes[norm_id]["original_ids"].append(stage_id)

        # Track sequence
        for stage in stages:
            norm = normalize_id(stage.get("id", ""))
            if norm not in stage_sequence:
                stage_sequence.append(norm)

    # Compute degrees from connections
    for conn in all_connections:
        parent = conn.get("parent_id") or conn.get("from_type", "")
        child = conn.get("child_id") or conn.get("to_type", "")
        parent_norm = normalize_id(parent) if parent else ""
        child_norm = normalize_id(child) if child else ""
        if parent_norm in stage_nodes:
            stage_nodes[parent_norm]["out_degree"] += 1
        if child_norm in stage_nodes:
            stage_nodes[child_norm]["in_degree"] += 1

    num_stages = len(stage_nodes)
    for norm_id, info in stage_nodes.items():
        order_normalized = info["order"] / max(num_stages, 1)
        is_terminal = 1 if info["out_degree"] == 0 else 0
        node_id = f"stg_{norm_id}"
        nodes.append({
            "id": node_id,
            "type": "stage",
            "stage_id": norm_id,
            "features": {
                "order_normalized": round(order_normalized, 4),
                "in_degree": info["in_degree"],
                "out_degree": info["out_degree"],
                "is_terminal": is_terminal,
            },
        })

    # --- Reverse lookup: equipment keyword → set of normalized stage IDs ---
    # (shared by stage transition weighting and context→stage edges)
    keyword_to_stages = defaultdict(set)
    for norm_id in stage_nodes:
        id_tokens = set(norm_id.split("_"))
        for token in id_tokens:
            keyword_to_stages[token].add(norm_id)
        for syn, canonical in SYNONYMS.items():
            if canonical in id_tokens:
                keyword_to_stages[syn].add(norm_id)

    # Count total keyword mentions per stage across all section texts
    stage_mentions = defaultdict(int)
    for text in section_texts.values():
        text_lower = text.lower()
        if not text_lower:
            continue
        for keyword, stage_ids in keyword_to_stages.items():
            hits = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            if hits > 0:
                for sid in stage_ids:
                    stage_mentions[sid] += hits

    # --- Stage transition edges (weighted by log-compressed mention counts) ---
    transition_counts = defaultdict(int)
    for conn in all_connections:
        parent = conn.get("parent_id") or conn.get("from_type", "")
        child = conn.get("child_id") or conn.get("to_type", "")
        if parent and child:
            key = (normalize_id(parent), normalize_id(child))
            transition_counts[key] += 1
    for (parent_norm, child_norm), count in transition_counts.items():
        weight = (math.log(1 + stage_mentions.get(parent_norm, 0), 1.8)
                  + math.log(1 + stage_mentions.get(child_norm, 0), 1.8))
        edges.append({
            "source": f"stg_{parent_norm}",
            "target": f"stg_{child_norm}",
            "type": "stage_transition",
            "weight": round(weight, 4),
        })

    # --- Context → Stage edges (DeBERTa scored + keyword matched) ---
    if doc_scores and stage_nodes:
        # Aggregate max DeBERTa score per group that passes threshold
        group_max_score = defaultdict(float)
        for score_entry in doc_scores:
            group = score_entry["group"]
            score = score_entry["score"]
            if score >= relevance_threshold:
                group_max_score[group] = max(group_max_score[group], score)

        # Match each qualifying group's text against stage keywords,
        # weight by keyword frequency density (mentions per 1K chars)
        for group, max_score in group_max_score.items():
            text = section_texts.get(group, "").lower()
            if not text:
                continue
            text_len = max(len(text), 1)
            # Count keyword hits per stage
            stage_hit_counts = defaultdict(int)
            for keyword, stage_ids in keyword_to_stages.items():
                hits = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
                if hits > 0:
                    for sid in stage_ids:
                        stage_hit_counts[sid] += hits
            if not stage_hit_counts:
                continue
            # Normalize: density per 1K chars, then scale to 0-1
            max_density = max(
                count / text_len * 1000 for count in stage_hit_counts.values()
            )
            for norm_id, count in stage_hit_counts.items():
                density = count / text_len * 1000
                weight = density / max_density if max_density > 0 else 0.0
                if weight < CTX_STAGE_THRESHOLD:
                    continue
                edges.append({
                    "source": f"ctx_{group}",
                    "target": f"stg_{norm_id}",
                    "type": "context_influences_stage",
                    "weight": round(weight, 4),
                })

    return {
        "document_id": doc_id,
        "nodes": nodes,
        "edges": edges,
        "stage_sequence": stage_sequence,
        "num_context_nodes": len(context_node_ids),
        "num_stage_nodes": len(stage_nodes),
        "num_edges": len(edges),
    }


# ---------------------------------------------------------------------------
# Main entry point (SageMaker)
# ---------------------------------------------------------------------------

def _safe_doc_name(doc_id: str) -> str:
    """Sanitize doc_id for use as a filename."""
    return re.sub(r'[^\w\-.]', '_', doc_id)


def _update_summary(summary_stats: dict, stage_vocab: dict, graph: dict):
    """Accumulate graph stats into running summary."""
    summary_stats["total_nodes"] += len(graph["nodes"])
    summary_stats["total_edges"] += len(graph["edges"])
    summary_stats["total_context_nodes"] += graph["num_context_nodes"]
    summary_stats["total_stage_nodes"] += graph["num_stage_nodes"]
    if graph["num_stage_nodes"] > 0:
        summary_stats["docs_with_stages"] += 1
    else:
        summary_stats["docs_context_only"] += 1
    for node in graph["nodes"]:
        if node["type"] == "context":
            summary_stats["section_coverage"][node["group"]] += 1
        elif node["type"] == "stage":
            sid = node["stage_id"]
            if sid not in stage_vocab:
                stage_vocab[sid] = len(stage_vocab)
    for edge in graph["edges"]:
        summary_stats["edge_type_counts"][edge["type"]] += 1


def _write_checkpoint(graphs_dir: Path, summary_stats: dict, stage_vocab: dict,
                      tfidf_vectorizer, completed_ids: set):
    """Write summary, stage_vocab, tfidf, and checkpoint file to disk."""
    # Checkpoint: list of completed doc IDs
    with open(graphs_dir / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(sorted(completed_ids), f)

    # Summary (make a serializable copy)
    summary_copy = dict(summary_stats)
    summary_copy["section_coverage"] = dict(summary_stats["section_coverage"])
    summary_copy["edge_type_counts"] = dict(summary_stats["edge_type_counts"])
    summary_copy["stage_vocab_size"] = len(stage_vocab)
    with open(graphs_dir / "graphs_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_copy, f, indent=2, ensure_ascii=False)

    with open(graphs_dir / "stage_vocab.json", "w", encoding="utf-8") as f:
        json.dump(stage_vocab, f, indent=2, ensure_ascii=False)

    if tfidf_vectorizer is not None:
        with open(graphs_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf_vectorizer, f)


def _release_memory():
    """Force Python + glibc to return freed memory to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # Not on Linux (e.g. local dev on Windows)


def _write_attempt_marker(graphs_dir: Path, doc_id: str, s3, bucket: str,
                          prefix: str):
    """Write current doc ID to attempt_marker.json before processing.

    If the process is OOM-killed, this file survives on S3 and the doc
    is skipped on the next run.
    """
    marker_path = graphs_dir / "attempt_marker.json"
    with open(marker_path, "w") as f:
        json.dump({"doc_id": doc_id}, f)
    if bucket:
        s3.upload_file(str(marker_path), bucket, prefix + "attempt_marker.json")


def _clear_attempt_marker(graphs_dir: Path, s3, bucket: str, prefix: str):
    """Remove attempt marker after successful processing."""
    marker_path = graphs_dir / "attempt_marker.json"
    if marker_path.exists():
        marker_path.unlink()
    if bucket:
        try:
            s3.delete_object(Bucket=bucket, Key=prefix + "attempt_marker.json")
        except Exception:
            pass


def _load_skip_ids(s3, bucket: str, prefix: str) -> set:
    """Load doc IDs that should be skipped (failed in previous runs)."""
    skip = set()
    if not bucket:
        return skip
    # Check for attempt marker (doc that was being processed when killed)
    try:
        resp = s3.get_object(Bucket=bucket, Key=prefix + "attempt_marker.json")
        marker = json.loads(resp["Body"].read())
        failed_id = marker.get("doc_id", "")
        if failed_id:
            skip.add(failed_id)
            logger.warning(f"Will skip {failed_id[:60]} (OOM in previous run)")
        # Clean up the marker
        s3.delete_object(Bucket=bucket, Key=prefix + "attempt_marker.json")
    except Exception:
        pass
    # Also load persistent skip list if it exists
    try:
        resp = s3.get_object(Bucket=bucket, Key=prefix + "skip_list.json")
        skip_list = json.loads(resp["Body"].read())
        skip.update(skip_list)
    except Exception:
        pass
    # Save updated skip list
    if skip:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sorted(skip), f)
            tmp = f.name
        s3.upload_file(tmp, bucket, prefix + "skip_list.json")
        os.unlink(tmp)
    return skip


COOCCURRENCE_THRESHOLD = 0.15  # Min cosine similarity for section cooccurrence edges
CTX_STAGE_THRESHOLD = 0.01    # Min keyword density weight for context→stage edges
BATCH_SIZE_DOCS = 10  # Documents per checkpoint batch
MAX_PAGE_DISTANCE = 5  # Only pair tables with page texts within this many pages
MAX_PAIRS_PER_DOC = 5000  # Safety cap after proximity filtering
TFIDF_SAMPLE_DOCS = 200  # Docs to sample for TF-IDF fitting (memory control)
MAX_DOCS_PER_RUN = 100   # Process at most this many docs per job (checkpoint & resume)


def _extract_doc_id(doc: dict) -> str:
    """Get document ID from a doc dict."""
    return doc.get("document_id", doc.get("filename", "unknown"))


def _process_single_doc(doc: dict, tokenizer, model, device: str,
                        batch_size: int, tfidf_vectorizer,
                        relevance_threshold: float) -> tuple[dict, int]:
    """Extract features, score pairs, and build graph for one document.

    Returns (graph_dict, num_pairs_scored).
    """
    groups = group_by_section(doc)

    # Extract section features and texts
    doc_features = {}
    section_texts = {}
    for group_name, group_data in groups.items():
        tables = group_data["tables"]
        page_texts = group_data["page_texts"]
        num_feats = extract_numeric_features(tables)
        text_feats = extract_text_features(page_texts, tables)
        doc_features[group_name] = {**num_feats, **text_feats, "num_tables": len(tables)}
        section_texts[group_name] = " ".join(
            pt.get("text", "") for pt in page_texts if pt.get("text")
        )

    # Build table-text pairs for DeBERTa scoring
    doc_pairs = []
    for group_name, group_data in groups.items():
        tables = group_data.get("tables", [])
        page_texts = group_data.get("page_texts", [])
        if not tables or not page_texts:
            continue
        for t_idx, table in enumerate(tables):
            table_str = serialize_table(table)
            if not table_str.strip():
                continue
            t_page = table.get("page") or table.get("page_number") or 0
            for pt in page_texts:
                text = pt.get("text", "")
                if not text or len(text.strip()) < 20:
                    continue
                pt_page = pt.get("page") or pt.get("page_number") or 0
                if abs(t_page - pt_page) > MAX_PAGE_DISTANCE:
                    continue
                doc_pairs.append((t_idx, t_page,
                                  group_name, table_str, text))

    # Cap pairs to prevent OOM on large documents
    if len(doc_pairs) > MAX_PAIRS_PER_DOC:
        random.seed(42)
        original_count = len(doc_pairs)
        doc_pairs = random.sample(doc_pairs, MAX_PAIRS_PER_DOC)
        logger.info(f"  Sampled {MAX_PAIRS_PER_DOC} pairs (was {original_count})")

    # Score pairs with DeBERTa (per-batch tokenization to limit memory)
    doc_scores = []
    use_cuda = device == "cuda"
    for i in range(0, len(doc_pairs), batch_size):
        batch = doc_pairs[i:i + batch_size]
        table_strs = [p[3] for p in batch]
        texts = [p[4] for p in batch]
        encodings = tokenizer(
            table_strs, texts,
            truncation=True, max_length=128,
            padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(
                input_ids=encodings["input_ids"].to(device),
                attention_mask=encodings["attention_mask"].to(device),
            )
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().tolist()
        for (t_idx, page, group, _, _), score in zip(batch, probs):
            doc_scores.append({
                "table_idx": t_idx,
                "page_number": page,
                "group": group,
                "score": round(score, 4),
            })
        del encodings, outputs, probs, table_strs, texts, batch
        # Release CUDA cached blocks every batch to prevent VMS growth
        if use_cuda:
            torch.cuda.empty_cache()

    # Free pairs before graph build
    num_pairs = len(doc_pairs)
    del doc_pairs
    _release_memory()

    # Build graph
    graph = build_document_graph(
        doc, doc_features, doc_scores, tfidf_vectorizer,
        section_texts, relevance_threshold,
    )
    return graph, num_pairs


def _log_memory(label: str, device: str = "cpu"):
    """Log current memory usage for diagnostics."""
    import psutil
    proc = psutil.Process()
    mem = proc.memory_info()
    rss_gb = mem.rss / (1024 ** 3)
    vms_gb = mem.vms / (1024 ** 3)
    msg = f"[MEM] {label}: RSS={rss_gb:.2f}GB VMS={vms_gb:.2f}GB"
    # Read cgroup memory stats if available (SageMaker)
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
            cgroup_used = int(f.read().strip()) / (1024 ** 3)
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            cgroup_limit = int(f.read().strip()) / (1024 ** 3)
        msg += f" cgroup={cgroup_used:.2f}/{cgroup_limit:.2f}GB"
    except (FileNotFoundError, PermissionError):
        pass
    if device == "cuda" and torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        msg += f" GPU_alloc={gpu_alloc:.2f}GB GPU_resv={gpu_reserved:.2f}GB"
    logger.info(msg)


def main():
    t0 = time.time()

    # SageMaker config — model via input channel, results streamed from S3
    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model_dir = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = int(os.environ.get("SM_HP_BATCH_SIZE", "8"))
    relevance_threshold = float(os.environ.get("SM_HP_RELEVANCE_THRESHOLD", "0.5"))
    checkpoint_bucket = os.environ.get("SM_HP_CHECKPOINT_BUCKET", "")
    checkpoint_prefix = os.environ.get("SM_HP_CHECKPOINT_PREFIX", "graphs/")
    results_bucket = os.environ.get("SM_HP_RESULTS_BUCKET", "")
    results_prefix = os.environ.get("SM_HP_RESULTS_PREFIX", "results/")

    logger.info(f"Device: {device}, batch_size: {batch_size}, "
                f"relevance_threshold: {relevance_threshold}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model channel: {model_dir}")
    logger.info(f"Results: s3://{results_bucket}/{results_prefix}")

    import boto3
    s3 = boto3.client("s3")

    output_path = Path(output_dir)
    graphs_dir = output_path / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Download checkpoint + saved TF-IDF from S3 (previous runs)
    checkpoint_path = graphs_dir / "checkpoint.json"
    tfidf_path = graphs_dir / "tfidf_vectorizer.pkl"
    completed_ids = set()
    saved_tfidf = None
    if checkpoint_bucket:
        ckpt_key = checkpoint_prefix + "checkpoint.json"
        logger.info(f"Checking for checkpoint at s3://{checkpoint_bucket}/{ckpt_key}")
        try:
            s3.download_file(checkpoint_bucket, ckpt_key, str(checkpoint_path))
            with open(checkpoint_path, encoding="utf-8") as f:
                completed_ids = set(json.load(f))
            logger.info(f"Resuming from checkpoint: {len(completed_ids)} docs already done")
        except Exception as e:
            logger.info(f"No existing checkpoint: {e}")

        # Try to load saved TF-IDF vectorizer (avoids re-reading 200 docs from S3)
        if completed_ids:
            tfidf_key = checkpoint_prefix + "tfidf_vectorizer.pkl"
            try:
                s3.download_file(checkpoint_bucket, tfidf_key, str(tfidf_path))
                with open(tfidf_path, "rb") as f:
                    saved_tfidf = pickle.load(f)
                logger.info(f"Loaded saved TF-IDF vectorizer from checkpoint "
                            f"({len(saved_tfidf.vocabulary_)} features)")
            except Exception as e:
                logger.info(f"No saved TF-IDF vectorizer, will re-fit: {e}")

    _log_memory("after checkpoint load", device)

    # 1. List result JSON keys from S3 (no bulk download)
    doc_keys = {}  # doc_id (S3-key stem) -> S3 key
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=results_bucket, Prefix=results_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                doc_id = key[len(results_prefix):].removesuffix(".json")
                doc_keys[doc_id] = key

    # Optional doc-allowlist (e.g. gold training corpus). Hyperparameter
    # SM_HP_ALLOWLIST_S3_KEY points to a JSON in checkpoint_bucket containing
    # {"result_keys": ["results/foo.json", ...]} — only those are processed.
    allowlist_key = os.environ.get("SM_HP_ALLOWLIST_S3_KEY", "").strip()
    if allowlist_key and checkpoint_bucket:
        try:
            resp = s3.get_object(Bucket=checkpoint_bucket, Key=allowlist_key)
            allow = json.loads(resp["Body"].read())
            allow_keys = set(allow.get("result_keys", []))
            before = len(doc_keys)
            doc_keys = {k: v for k, v in doc_keys.items() if v in allow_keys}
            logger.info(f"Allowlist {allowlist_key}: filtered {before} -> "
                        f"{len(doc_keys)} docs")
        except Exception as e:
            logger.warning(f"Failed to load allowlist {allowlist_key}: {e}")

    total_docs = len(doc_keys)
    logger.info(f"Found {total_docs} result JSONs in S3 ({time.time() - t0:.0f}s)")

    # 2. TF-IDF: use saved vectorizer on resume, otherwise fit from scratch
    if saved_tfidf is not None:
        tfidf_vectorizer = saved_tfidf
        del saved_tfidf
        logger.info("Skipping TF-IDF pass (using saved vectorizer from checkpoint)")
    else:
        sample_size = min(TFIDF_SAMPLE_DOCS, total_docs)
        sample_keys = random.Random(42).sample(list(doc_keys.values()), sample_size)
        logger.info(f"Pass 1: collecting TF-IDF corpus from {sample_size} sampled docs...")
        tfidf_corpus = []
        for s3_key in sample_keys:
            try:
                resp = s3.get_object(Bucket=results_bucket, Key=s3_key)
                doc = json.loads(resp["Body"].read())
                groups = group_by_section(doc)
                for group_data in groups.values():
                    section_text = " ".join(
                        pt.get("text", "") for pt in group_data["page_texts"] if pt.get("text")
                    )
                    if section_text.strip():
                        tfidf_corpus.append(section_text)
                del doc, groups, resp
            except Exception as e:
                logger.warning(f"Skipping {s3_key} for TF-IDF: {e}")

        logger.info(f"Collected {len(tfidf_corpus)} section texts ({time.time() - t0:.0f}s)")

        # 3. Fit TF-IDF vectorizer
        logger.info("Fitting TF-IDF...")
        tfidf_vectorizer = fit_tfidf(tfidf_corpus, max_features=200)
        del tfidf_corpus
    _release_memory()
    _log_memory("after TF-IDF", device)

    # 4. Load DeBERTa model from local model channel
    logger.info("Loading DeBERTa model...")
    tokenizer, model = load_deberta(model_dir, device)
    _release_memory()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Drop page cache from model files to free cgroup memory
    try:
        os.system("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null")
        logger.info("Dropped page caches after model load")
    except Exception:
        pass

    _log_memory("after model load", device)

    # 5. Initialize summary stats — restore from checkpoint if available
    summary_stats = None
    stage_vocab = None
    if checkpoint_bucket and completed_ids:
        try:
            resp = s3.get_object(Bucket=checkpoint_bucket,
                                 Key=checkpoint_prefix + "graphs_summary.json")
            summary_stats = json.loads(resp["Body"].read())
            # Convert plain dicts back to defaultdict(int)
            summary_stats["section_coverage"] = defaultdict(int, summary_stats.get("section_coverage", {}))
            summary_stats["edge_type_counts"] = defaultdict(int, summary_stats.get("edge_type_counts", {}))
            logger.info(f"Restored summary from checkpoint: {summary_stats['docs_with_stages']} with stages, "
                        f"{summary_stats['docs_context_only']} context-only")
        except Exception as e:
            logger.info(f"No saved summary, starting fresh: {e}")
        try:
            resp = s3.get_object(Bucket=checkpoint_bucket,
                                 Key=checkpoint_prefix + "stage_vocab.json")
            stage_vocab = json.loads(resp["Body"].read())
            logger.info(f"Restored stage vocab from checkpoint: {len(stage_vocab)} stages")
        except Exception as e:
            logger.info(f"No saved stage vocab, starting fresh: {e}")

    if summary_stats is None:
        summary_stats = {
            "total_docs": total_docs,
            "docs_with_stages": 0,
            "docs_context_only": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "total_context_nodes": 0,
            "total_stage_nodes": 0,
            "section_coverage": defaultdict(int),
            "edge_type_counts": defaultdict(int),
        }
    summary_stats["total_docs"] = total_docs  # Always update in case corpus grew
    if stage_vocab is None:
        stage_vocab = {}

    # 6. Pass 2: process pending docs (stream each from S3)
    skip_ids = _load_skip_ids(s3, checkpoint_bucket, checkpoint_prefix)
    all_pending = [did for did in doc_keys if did not in completed_ids]
    pending_ids = all_pending[:MAX_DOCS_PER_RUN]
    logger.info(f"Pending: {len(all_pending)} total, processing {len(pending_ids)} this run "
                f"({len(completed_ids)} already done, {len(skip_ids)} skipped)")

    total_scored = 0
    for batch_start in range(0, len(pending_ids), BATCH_SIZE_DOCS):
        batch_ids = pending_ids[batch_start:batch_start + BATCH_SIZE_DOCS]
        batch_num = batch_start // BATCH_SIZE_DOCS + 1
        total_batches = (len(pending_ids) + BATCH_SIZE_DOCS - 1) // BATCH_SIZE_DOCS
        logger.info(f"=== Batch {batch_num}/{total_batches} "
                    f"({len(batch_ids)} docs) ===")

        for doc_idx, doc_id in enumerate(batch_ids):
            # Skip docs that caused OOM in previous runs
            if doc_id in skip_ids:
                logger.warning(f"Skipping {doc_id[:60]} (failed in previous run)")
                continue

            _log_memory(f"batch{batch_num} doc{doc_idx} pre-load ({doc_id[:50]})", device)

            # Mark doc as in-progress before processing (survives OOM kill)
            _write_attempt_marker(graphs_dir, doc_id, s3, checkpoint_bucket,
                                  checkpoint_prefix)

            resp = s3.get_object(Bucket=results_bucket, Key=doc_keys[doc_id])
            raw = resp["Body"].read()
            doc = json.loads(raw)
            doc_size_kb = len(raw) / 1024
            del resp, raw

            _log_memory(f"batch{batch_num} doc{doc_idx} post-parse ({doc_size_kb:.0f}KB)", device)

            # Move model back to GPU before scoring
            if device == "cuda":
                model.to(device)

            graph, num_scored = _process_single_doc(
                doc, tokenizer, model, device, batch_size,
                tfidf_vectorizer, relevance_threshold,
            )
            total_scored += num_scored
            del doc

            _log_memory(f"batch{batch_num} doc{doc_idx} post-score ({num_scored} pairs)", device)

            # Write graph JSON immediately
            graph_path = graphs_dir / f"{_safe_doc_name(doc_id)}.json"
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False)

            _update_summary(summary_stats, stage_vocab, graph)
            completed_ids.add(doc_id)
            del graph
            _release_memory()
            if device == "cuda":
                torch.cuda.empty_cache()
                # Move model to CPU between docs to release driver DMA buffers
                model.cpu()
                torch.cuda.empty_cache()

            # Clear attempt marker on success
            _clear_attempt_marker(graphs_dir, s3, checkpoint_bucket,
                                  checkpoint_prefix)

            # Checkpoint after EVERY doc and sync to S3 immediately
            _write_checkpoint(graphs_dir, summary_stats, stage_vocab,
                              tfidf_vectorizer, completed_ids)
            if checkpoint_bucket:
                for fname in ["checkpoint.json", "graphs_summary.json",
                              "stage_vocab.json", "tfidf_vectorizer.pkl"]:
                    fp = graphs_dir / fname
                    if fp.exists():
                        s3.upload_file(str(fp), checkpoint_bucket,
                                       checkpoint_prefix + fname)

            _log_memory(f"batch{batch_num} doc{doc_idx} post-gc", device)

        elapsed = time.time() - t0
        logger.info(f"  Batch {batch_num} done: {len(completed_ids)}/{total_docs} docs, "
                    f"{total_scored:,} pairs scored, {elapsed:.0f}s elapsed")

    # Final summary write
    _write_checkpoint(graphs_dir, summary_stats, stage_vocab,
                      tfidf_vectorizer, completed_ids)
    if checkpoint_bucket:
        for fname in ["checkpoint.json", "graphs_summary.json",
                      "stage_vocab.json", "tfidf_vectorizer.pkl"]:
            fp = graphs_dir / fname
            if fp.exists():
                s3.upload_file(str(fp), checkpoint_bucket,
                               checkpoint_prefix + fname)

    # Remove checkpoint file only when ALL docs are done
    remaining = total_docs - len(completed_ids)
    if remaining == 0:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if checkpoint_bucket:
            try:
                s3.delete_object(Bucket=checkpoint_bucket,
                                 Key=checkpoint_prefix + "checkpoint.json")
            except Exception:
                pass

    elapsed = time.time() - t0
    logger.info(f"Run complete: {len(pending_ids)} docs processed in {elapsed:.0f}s, "
                f"{remaining} remaining")
    logger.info(f"  Context nodes: {summary_stats['total_context_nodes']}, "
                f"Stage nodes: {summary_stats['total_stage_nodes']}")
    logger.info(f"  Docs with stages: {summary_stats['docs_with_stages']}, "
                f"Context-only: {summary_stats['docs_context_only']}")
    logger.info(f"  Stage vocab: {len(stage_vocab)} unique stages")
    logger.info(f"  Output: {graphs_dir}")


if __name__ == "__main__":
    main()
