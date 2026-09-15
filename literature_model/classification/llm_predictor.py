"""LLM-powered flowsheet prediction — replaces or augments XGBoost.

The LLM traverses the full graph network to make decisions. Instead of
receiving summaries, it sees the actual graph topology: context nodes with
features, edge weights between sections and stages, cross-document similarity
links, and corpus-wide transition patterns. It then builds an explicit
decision tree — at each stage selection point, it evaluates the graph
evidence and decides the next processing step.

Two serialization modes:
  1. Full graph traversal — serialize every node, edge, and weight so the
     LLM can follow relationships (context → stage influence, cross-doc
     similarity, stage transitions) and reason through the network.
  2. Decision tree output — the LLM produces a tree of decisions, not just
     a flat stage list. Each branch explains which graph evidence led to
     the stage selection.

Usage:
    from llm_predictor import predict_flowsheet_llm

    result = predict_flowsheet_llm(
        index=index,
        head_grade=0.45,
        deposit_type="porphyry",
        ore_type="sulfide",
        mode="standalone",        # or "augment"
        xgb_result=None,          # pass XGBoost output for augment mode
    )
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants duplicated from stage_predictor to avoid importing heavy deps
# (stage_predictor imports imblearn, xgboost, etc. which may not be local)
CANONICAL_STAGES = {
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

FEATURE_SECTION_TYPES = [
    "geology", "metallurgical_testing", "economics",
    "mining_method", "infrastructure", "climate",
]

DEPOSIT_TYPES = [
    "porphyry", "skarn", "vms", "iocg", "sedimentary",
    "epithermal", "orogenic", "breccia", "intrusive", "volcanic",
]

ORE_TYPES = [
    "oxide", "sulfide", "sulphide", "supergene", "hypogene",
    "transition", "refractory", "free_milling",
]

# ---------------------------------------------------------------------------
# Stage descriptions extracted from TERMS.md (23 canonical stages)
# ---------------------------------------------------------------------------

STAGE_DESCRIPTIONS = {
    "crusher": "Size reduction using mechanical force (jaw, cone, gyratory crushers)",
    "mill": "Fine grinding using rotating drums with grinding media (SAG, ball, rod mills)",
    "regrind": "Secondary grinding to improve mineral liberation",
    "screen": "Size classification through mesh/apertures",
    "agglomeration": "Binding fine particles into larger masses for heap leaching",
    "cyclone": "Size classification using centrifugal force (hydrocyclones)",
    "input": "Process feed initialization (ROM stockpile)",
    "flotation": "Mineral separation using air bubbles and reagents (rougher, cleaner, scavenger)",
    "magnetic_separation": "Separation using magnetic properties of minerals",
    "thickener": "Solid-liquid separation by gravity settling",
    "filter": "Dewatering by filtration (filter press, vacuum filter)",
    "electrowinning": "Electrochemical metal recovery from solution",
    "gravity": "Separation based on density differences (jigs, spirals, shaking tables)",
    "leach": "Dissolving metals using chemical solutions (heap leach, tank leach)",
    "adsorption": "Metal recovery onto activated carbon or resin (CIC, CIP, CIL)",
    "elution": "Stripping adsorbed metals from carbon or resin",
    "kiln": "High-temperature processing (furnace, calciner, roaster)",
    "stockpile": "Bulk ore storage",
    "feeder": "Controlled material discharge (apron, belt feeders, hoppers)",
    "conveyor": "Belt transport systems",
    "tank": "Vessels for holding, mixing, or conditioning",
    "tailing": "Waste material disposal (tailings pond, TMF, detox)",
    "drying": "Moisture removal from concentrates or products",
    "water_treatment": "Treatment of process water, effluent, or cyanide solutions",
    "ore_sorting": "Sensor-based ore sorting",
    "carbon_handling": "Carbon management (screening, regeneration, storage)",
    "product_recovery": "Final metal product (dore, precipitation, refinery, smelting)",
    "solution_recovery": "Solution-phase metal recovery (SX, IX, Merrill-Crowe, cementation)",
    "bin": "Storage containers (ore bin, concentrate bin, silo)",
}

# Section labels for readable profiles
SECTION_LABELS = {
    "geology": "Geology",
    "metallurgical_testing": "Met Testing",
    "economics": "Economics",
    "mining_method": "Mining Method",
    "infrastructure": "Infrastructure",
    "climate": "Climate",
}

# Keyword groups for readable serialization
MINERALOGY_KW = {
    "chalcopyrite", "bornite", "pyrite", "chalcocite", "covellite",
    "enargite", "arsenopyrite", "molybdenite", "galena", "sphalerite",
    "magnetite", "hematite", "goethite", "malachite", "azurite", "chrysocolla",
}

DEPOSIT_KW = {
    "porphyry", "skarn", "vms", "iocg", "sedimentary", "epithermal",
    "orogenic", "breccia", "intrusive", "volcanic",
}

ORE_KW = {
    "oxide", "sulfide", "sulphide", "supergene", "hypogene",
    "transition", "refractory", "free_milling",
}

MINING_KW = {"open_pit", "underground", "block_cave", "stoping", "strip_ratio"}

CLIMATE_KW = {"rainfall", "arid", "water_availability", "tailings_dam", "closure"}


# ---------------------------------------------------------------------------
# Layer 1a: Document profile serialization (compact, for similar docs)
# ---------------------------------------------------------------------------

def serialize_document_profile(graph: dict) -> str:
    """Convert a document graph JSON into a human-readable text profile.

    Reads the 6 high-signal context nodes and stage nodes, converts numeric
    features and keyword counts into natural language statements.

    Returns ~200-400 token structured text block.
    """
    sections = {}
    stages = []
    connections = []

    for node in graph.get("nodes", []):
        if node.get("type") == "context":
            group = node.get("group", "")
            if group in SECTION_LABELS:
                sections[group] = node.get("features", {})
        elif node.get("type") == "stage":
            sid = node.get("stage_id", node.get("id", "").replace("stg_", ""))
            stages.append(sid)

    for edge in graph.get("edges", []):
        if edge.get("type") == "stage_transition":
            connections.append((edge.get("source", ""), edge.get("target", "")))

    lines = []
    doc_id = graph.get("doc_id", "unknown")
    lines.append(f"Project: {doc_id}")

    # Geology
    geo = sections.get("geology", {})
    if geo:
        minerals = [kw for kw in MINERALOGY_KW if geo.get(f"kw_{kw}", 0) > 0]
        deposits = [kw for kw in DEPOSIT_KW if geo.get(f"kw_{kw}", 0) > 0]
        ores = [kw for kw in ORE_KW if geo.get(f"kw_{kw}", 0) > 0]
        parts = []
        if deposits:
            parts.append(f"Deposit: {', '.join(deposits)}")
        if ores:
            parts.append(f"Ore: {', '.join(ores)}")
        if minerals:
            parts.append(f"Mineralogy: {', '.join(minerals)}")
        if parts:
            lines.append("  " + " | ".join(parts))

    # Economics / grade
    econ = sections.get("economics", {})
    if econ:
        parts = []
        for key, label in [("grade_cu_mean", "Cu grade"),
                           ("grade_au_mean", "Au grade"),
                           ("recovery_mean", "Recovery"),
                           ("tonnage_mean", "Tonnage"),
                           ("npv_mean", "NPV"),
                           ("capex_mean", "CAPEX")]:
            val = econ.get(key, 0)
            if val and val > 0:
                parts.append(f"{label}: {val:.2f}")
        if not parts:
            grade = econ.get("num_mean", 0)
            if grade > 0:
                parts.append(f"Avg numeric: {grade:.2f}")
        if parts:
            lines.append(f"  Economics: {', '.join(parts)}")

    # Met testing
    met = sections.get("metallurgical_testing", {})
    if met:
        parts = []
        recovery = met.get("recovery_mean", 0)
        if recovery > 0:
            parts.append(f"Recovery: {recovery:.1f}%")
        grind = met.get("grind_size_mean", 0)
        if grind > 0:
            parts.append(f"Grind P80: {grind:.0f}μm")
        equip = [kw for kw in [
            "flotation", "leach", "gravity", "magnetic_separation",
            "electrowinning", "adsorption", "solvent_extraction",
        ] if met.get(f"kw_{kw}", 0) > 0]
        if equip:
            parts.append(f"Methods tested: {', '.join(equip)}")
        if parts:
            lines.append(f"  Met Testing: {', '.join(parts)}")

    # Mining method
    mining = sections.get("mining_method", {})
    if mining:
        methods = [kw.replace("_", " ") for kw in MINING_KW
                   if mining.get(f"kw_{kw}", 0) > 0]
        if methods:
            lines.append(f"  Mining: {', '.join(methods)}")

    # Climate
    climate = sections.get("climate", {})
    if climate:
        conditions = [kw.replace("_", " ") for kw in CLIMATE_KW
                      if climate.get(f"kw_{kw}", 0) > 0]
        if conditions:
            lines.append(f"  Climate: {', '.join(conditions)}")

    # Infrastructure
    infra = sections.get("infrastructure", {})
    if infra:
        chars = infra.get("char_count", 0)
        if chars > 0:
            lines.append(f"  Infrastructure: {chars:.0f} chars of context")

    # Stages and connections
    if stages:
        lines.append(f"  Flowsheet stages: {' → '.join(stages)}")
    if connections:
        conn_strs = [f"{s}→{t}" for s, t in connections]
        lines.append(f"  Connections: {', '.join(conn_strs)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 1b: Full graph topology serialization (for LLM traversal)
# ---------------------------------------------------------------------------

# The 6 sections that XGBoost uses — only these are sent to the LLM
XGBOOST_SECTIONS = {
    "geology", "metallurgical_testing", "economics",
    "mining_method", "infrastructure", "climate",
}


def serialize_graph_topology(graph: dict) -> str:
    """Serialize the graph sections that inform XGBoost's predictions.

    Only includes:
    - The 6 context nodes that XGBoost uses (geology, met testing, economics,
      mining method, infrastructure, climate) with their features
    - Section cooccurrence edges between these 6 sections
    - Context→stage influence edges (DeBERTa scores)

    This focuses the LLM on the same evidence the XGBoost model sees.
    """
    lines = []
    doc_id = graph.get("doc_id", "unknown")
    lines.append(f"=== GRAPH: {doc_id} ===")

    # Index nodes by ID for edge resolution
    node_map = {}
    context_nodes = []
    stage_nodes = []

    for node in graph.get("nodes", []):
        nid = node.get("id", "")
        node_map[nid] = node
        if node.get("type") == "context":
            # Only include the 6 XGBoost feature sections
            if node.get("group", "") in XGBOOST_SECTIONS:
                context_nodes.append(node)
        elif node.get("type") == "stage":
            stage_nodes.append(node)

    # Context nodes: serialize all features as key=value pairs
    lines.append("")
    lines.append("CONTEXT NODES (NI 43-101 report sections):")
    for node in context_nodes:
        group = node.get("group", "unknown")
        feats = node.get("features", {})

        # Extract meaningful features (skip zeros)
        active_kw = {}
        numerics = {}
        for key, val in feats.items():
            if val == 0 or val == 0.0:
                continue
            if key.startswith("kw_"):
                active_kw[key[3:]] = val
            elif key == "tfidf":
                continue  # Skip raw TF-IDF vector (200-dim, not useful as text)
            else:
                numerics[key] = val

        lines.append(f"  [{group}]")
        if numerics:
            num_parts = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in sorted(numerics.items())]
            lines.append(f"    metrics: {', '.join(num_parts)}")
        if active_kw:
            kw_parts = [f"{k}({int(v)})" if v > 1 else k
                        for k, v in sorted(active_kw.items(),
                                           key=lambda x: x[1], reverse=True)]
            lines.append(f"    keywords: {', '.join(kw_parts)}")

    # Stage nodes
    if stage_nodes:
        lines.append("")
        lines.append("STAGE NODES (process equipment):")
        for node in stage_nodes:
            sid = node.get("stage_id", node.get("id", "").replace("stg_", ""))
            feats = node.get("features", {})
            order = feats.get("order", "?")
            in_deg = feats.get("in_degree", 0)
            out_deg = feats.get("out_degree", 0)
            terminal = feats.get("is_terminal", False)
            parts = [f"order={order}", f"in={in_deg}", f"out={out_deg}"]
            if terminal:
                parts.append("TERMINAL")
            lines.append(f"  [{sid}] {', '.join(parts)}")

    # Edges: grouped by type
    edges_by_type = defaultdict(list)
    for edge in graph.get("edges", []):
        etype = edge.get("type", "unknown")
        edges_by_type[etype].append(edge)

    # Section cooccurrence edges — only between the 6 XGBoost sections
    cooc = edges_by_type.get("section_cooccurrence", [])
    if cooc:
        relevant_cooc = []
        for edge in cooc:
            src_group = node_map.get(edge.get("source", ""), {}).get("group", "")
            tgt_group = node_map.get(edge.get("target", ""), {}).get("group", "")
            if src_group in XGBOOST_SECTIONS and tgt_group in XGBOOST_SECTIONS:
                relevant_cooc.append(edge)
        if relevant_cooc:
            lines.append("")
            lines.append("SECTION RELATIONSHIPS (cooccurrence weights):")
            for edge in sorted(relevant_cooc, key=lambda e: e.get("weight", 0), reverse=True)[:15]:
                src_group = node_map.get(edge["source"], {}).get("group", "")
                tgt_group = node_map.get(edge["target"], {}).get("group", "")
                lines.append(f"  {src_group} <-> {tgt_group} (weight={edge.get('weight', 0):.3f})")

    # Context → stage influence — only from the 6 XGBoost sections
    influence = edges_by_type.get("context_influences_stage", [])
    if influence:
        relevant_inf = []
        for edge in influence:
            src_group = node_map.get(edge.get("source", ""), {}).get("group", "")
            if src_group in XGBOOST_SECTIONS:
                relevant_inf.append(edge)
        if relevant_inf:
            lines.append("")
            lines.append("SECTION -> STAGE INFLUENCE (DeBERTa relevance scores):")
            for edge in sorted(relevant_inf, key=lambda e: e.get("weight", 0), reverse=True):
                src_group = node_map.get(edge["source"], {}).get("group", "")
                tgt_stage = node_map.get(edge["target"], {}).get("stage_id",
                            edge["target"].replace("stg_", ""))
                lines.append(f"  {src_group} -> {tgt_stage} (relevance={edge.get('weight', 0):.3f})")

    # Stage transition edges (stage → stage)
    transitions = edges_by_type.get("stage_transition", [])
    if transitions:
        lines.append("")
        lines.append("STAGE TRANSITIONS (process flow):")
        for edge in transitions:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            w = edge.get("weight", 1)
            src_stage = node_map.get(src, {}).get("stage_id",
                        src.replace("stg_", ""))
            tgt_stage = node_map.get(tgt, {}).get("stage_id",
                        tgt.replace("stg_", ""))
            lines.append(f"  {src_stage} → {tgt_stage} (count={w})")

    return "\n".join(lines)


def serialize_cross_doc_edges(target_doc_id: str,
                              cross_doc_data: dict,
                              graphs: list[dict],
                              top_k: int = 5) -> str:
    """Serialize cross-document similarity edges for a target document.

    Shows the LLM which other projects are most similar and why (per-section
    similarity scores), so it can traverse to analogous projects.
    """
    neighbors = cross_doc_data.get(target_doc_id, {}).get("neighbors", [])
    if not neighbors:
        return ""

    graph_by_id = {g.get("doc_id"): g for g in graphs}
    lines = [f"CROSS-DOCUMENT NEIGHBORS for {target_doc_id}:"]
    lines.append("(Projects with similar geology, metallurgy, and process design)")
    lines.append("")

    for nb in neighbors[:top_k]:
        nb_id = nb.get("doc_id", "unknown")
        overall = nb.get("overall_similarity", 0)
        strategy = nb.get("strategy_similarity", 0)
        section_sims = nb.get("section_similarities", {})

        lines.append(f"  → {nb_id} (overall={overall:.2f}, strategy_match={strategy:.2f})")

        # Show which sections are most similar
        top_sections = sorted(section_sims.items(),
                              key=lambda x: x[1], reverse=True)[:5]
        if top_sections:
            sim_parts = [f"{s}={v:.2f}" for s, v in top_sections]
            lines.append(f"    section similarity: {', '.join(sim_parts)}")

        # Include neighbor's flowsheet if available
        nb_graph = graph_by_id.get(nb_id)
        if nb_graph:
            nb_stages = []
            for node in nb_graph.get("nodes", []):
                if node.get("type") == "stage":
                    nb_stages.append(node.get("stage_id",
                                    node.get("id", "").replace("stg_", "")))
            if nb_stages:
                lines.append(f"    flowsheet: {' → '.join(nb_stages)}")
        lines.append("")

    return "\n".join(lines)


def serialize_transition_graph(transition_counts: dict) -> str:
    """Serialize the corpus-wide stage transition graph.

    This is the accumulated knowledge from all 967+ documents: which stages
    follow which, and how frequently. The LLM uses this as the backbone for
    ordering decisions.
    """
    lines = ["CORPUS STAGE TRANSITION GRAPH (accumulated from all documents):"]
    lines.append("Format: source → target (occurrence count across corpus)")
    lines.append("")

    # Parse and sort by count
    edges = []
    for key, count in transition_counts.items():
        parts = key.split("|") if "|" in key else key.split("→")
        if len(parts) == 2:
            edges.append((parts[0].strip(), parts[1].strip(), count))

    edges.sort(key=lambda x: x[2], reverse=True)

    # Group by source node for traversability
    by_source = defaultdict(list)
    for src, dst, count in edges:
        by_source[src].append((dst, count))

    for src in sorted(by_source, key=lambda s: sum(c for _, c in by_source[s]),
                      reverse=True):
        targets = sorted(by_source[src], key=lambda x: x[1], reverse=True)
        target_strs = [f"{dst}({count})" for dst, count in targets]
        lines.append(f"  {src} → {', '.join(target_strs)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 2: Corpus transition knowledge base (dynamic)
# ---------------------------------------------------------------------------

def build_corpus_knowledge(transition_counts: dict, stage_vocab: list,
                           graphs: list[dict]) -> str:
    """Build a text knowledge base from corpus-wide transition patterns.

    Extracts common pathway templates and conditional stage rules.
    Computed dynamically at inference time.

    Returns ~500-800 token text block for the system prompt.
    """
    lines = ["## Common Processing Pathways"]

    # Build directed graph from transition counts
    adjacency = defaultdict(list)
    for key, count in sorted(transition_counts.items(),
                             key=lambda x: x[1], reverse=True):
        parts = key.split("|") if "|" in key else key.split("→")
        if len(parts) == 2:
            src, dst = parts[0].strip(), parts[1].strip()
            adjacency[src].append((dst, count))

    # Extract top pathway templates via greedy walk from common start nodes
    pathways = _extract_pathways(adjacency, max_paths=8)
    for i, (path, total_weight) in enumerate(pathways, 1):
        lines.append(f"{i}. {' → '.join(path)} (corpus support: {total_weight})")

    # Conditional rules: stage co-occurrence with ore/deposit types
    lines.append("")
    lines.append("## Conditional Stage Rules (from corpus)")
    rules = _extract_conditional_rules(graphs)
    for rule in rules[:15]:
        lines.append(f"- {rule}")

    # Stage frequency
    lines.append("")
    lines.append("## Stage Frequency (labeled documents)")
    stage_counts = defaultdict(int)
    total_labeled = 0
    for graph in graphs:
        has_stages = False
        for node in graph.get("nodes", []):
            if node.get("type") == "stage":
                sid = node.get("stage_id", node.get("id", "").replace("stg_", ""))
                if sid in CANONICAL_STAGES:
                    stage_counts[CANONICAL_STAGES[sid]] += 1
                    has_stages = True
                elif sid in stage_vocab:
                    stage_counts[sid] += 1
                    has_stages = True
        if has_stages:
            total_labeled += 1

    for stage, count in sorted(stage_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / max(total_labeled, 1) * 100
        lines.append(f"  {stage}: {count} docs ({pct:.0f}%)")

    return "\n".join(lines)


def _extract_pathways(adjacency: dict, max_paths: int = 8) -> list:
    """Greedy walk from common start nodes to extract pathway templates."""
    # Find likely start nodes (high out-degree, low or zero in-degree)
    in_counts = defaultdict(int)
    for src, targets in adjacency.items():
        for dst, _ in targets:
            in_counts[dst] += 1

    start_candidates = []
    for node in adjacency:
        in_count = in_counts.get(node, 0)
        out_weight = sum(c for _, c in adjacency[node])
        start_candidates.append((node, out_weight - in_count * 10))

    start_candidates.sort(key=lambda x: x[1], reverse=True)

    pathways = []
    seen_starts = set()
    for start_node, _ in start_candidates:
        if start_node in seen_starts or len(pathways) >= max_paths:
            break
        seen_starts.add(start_node)

        # Greedy walk: always follow highest-weight edge
        path = [start_node]
        total_weight = 0
        visited = {start_node}
        current = start_node
        while current in adjacency:
            targets = [(dst, c) for dst, c in adjacency[current]
                       if dst not in visited]
            if not targets:
                break
            next_node, weight = max(targets, key=lambda x: x[1])
            path.append(next_node)
            total_weight += weight
            visited.add(next_node)
            current = next_node

        if len(path) >= 3:
            pathways.append((path, total_weight))

    pathways.sort(key=lambda x: x[1], reverse=True)
    return pathways[:max_paths]


def _extract_conditional_rules(graphs: list[dict]) -> list[str]:
    """Extract conditional stage rules from document features + stage labels."""
    # Track: for each (condition, stage) pair, count co-occurrences
    condition_stage = defaultdict(lambda: defaultdict(int))
    condition_total = defaultdict(int)

    for graph in graphs:
        # Get document conditions from geology node
        conditions = set()
        for node in graph.get("nodes", []):
            if node.get("type") != "context":
                continue
            feats = node.get("features", {})
            group = node.get("group", "")

            if group == "geology":
                for kw in ORE_KW:
                    if feats.get(f"kw_{kw}", 0) > 0:
                        conditions.add(f"ore={kw}")
                for kw in DEPOSIT_KW:
                    if feats.get(f"kw_{kw}", 0) > 0:
                        conditions.add(f"deposit={kw}")
            elif group == "climate":
                for kw in CLIMATE_KW:
                    if feats.get(f"kw_{kw}", 0) > 0:
                        conditions.add(f"climate={kw}")

        # Get document stages
        doc_stages = set()
        for node in graph.get("nodes", []):
            if node.get("type") == "stage":
                sid = node.get("stage_id", node.get("id", "").replace("stg_", ""))
                canonical = CANONICAL_STAGES.get(sid, sid)
                doc_stages.add(canonical)

        if not conditions or not doc_stages:
            continue

        for cond in conditions:
            condition_total[cond] += 1
            for stage in doc_stages:
                condition_stage[cond][stage] += 1

    # Generate rules for high-confidence associations
    rules = []
    for cond, total in sorted(condition_total.items(), key=lambda x: x[1], reverse=True):
        if total < 5:
            continue
        for stage, count in sorted(condition_stage[cond].items(),
                                   key=lambda x: x[1], reverse=True):
            pct = count / total * 100
            if pct >= 60 and count >= 5:
                rules.append(f"When {cond}: {stage} appears in {pct:.0f}% of cases ({count}/{total} docs)")

    # Sort by confidence descending
    rules.sort(key=lambda r: float(r.split("in ")[1].split("%")[0]), reverse=True)
    return rules


# ---------------------------------------------------------------------------
# Layer 3: Similar document retrieval
# ---------------------------------------------------------------------------

def retrieve_similar_profiles(head_grade: float, deposit_type: str,
                              ore_type: str, index: dict,
                              graphs: list[dict],
                              top_k: int = 5) -> list[str]:
    """Find K most similar documents and serialize their profiles.

    Uses the same scoring logic as flowsheet_predictor.predict_flowsheet()
    (grade distance + deposit match + ore match).
    """
    docs = index.get("docs", [])
    doc_scores = []

    for i, doc in enumerate(docs):
        score = 0.0
        meta = doc.get("metadata", {})

        # Grade similarity
        grade_vals = meta.get("grade_values", [])
        if grade_vals and head_grade > 0:
            min_dist = min(abs(g - head_grade) for g in grade_vals)
            score += 1.0 / (1.0 + min_dist * 5.0) * 2.0
        elif not grade_vals:
            score += 0.3

        # Deposit match
        if deposit_type:
            dep_norm = deposit_type.lower().strip()
            if dep_norm in meta.get("deposit_types", []):
                score += 1.5
            elif not meta.get("deposit_types"):
                score += 0.3

        # Ore match
        if ore_type:
            ore_norm = ore_type.lower().strip()
            doc_ore = meta.get("ore_types", [])
            if ore_norm in doc_ore:
                score += 1.0
            elif ore_norm == "mixed" and len(doc_ore) > 1:
                score += 0.7
            elif not doc_ore:
                score += 0.2

        # Boost labeled docs
        if doc.get("has_labels"):
            score *= 1.2

        doc_scores.append((i, score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Serialize top-K profiles
    profiles = []
    graph_by_id = {g.get("doc_id"): g for g in graphs} if graphs else {}

    for idx, score in doc_scores[:top_k]:
        doc = docs[idx]
        doc_id = doc.get("doc_id", "unknown")
        graph = graph_by_id.get(doc_id)

        if graph:
            profile = serialize_document_profile(graph)
        else:
            # Fallback: build minimal profile from index metadata
            profile = _profile_from_index(doc)

        profiles.append(profile)

    return profiles


def _profile_from_index(doc: dict) -> str:
    """Build a minimal profile from index metadata when graph JSON unavailable."""
    doc_id = doc.get("doc_id", "unknown")
    meta = doc.get("metadata", {})
    lines = [f"Project: {doc_id}"]

    deposits = meta.get("deposit_types", [])
    if deposits:
        lines.append(f"  Deposit: {', '.join(deposits)}")

    ores = meta.get("ore_types", [])
    if ores:
        lines.append(f"  Ore: {', '.join(ores)}")

    grades = meta.get("grade_values", [])
    if grades:
        lines.append(f"  Grade values: {', '.join(f'{g:.3f}' for g in grades[:5])}")

    stage_probs = doc.get("stage_probs", {})
    if stage_probs:
        active = [s for s, p in sorted(stage_probs.items(),
                                        key=lambda x: x[1], reverse=True) if p > 0.5]
        if active:
            lines.append(f"  Flowsheet stages: {' → '.join(active)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_stage_vocab_text(stage_vocab: list = None) -> str:
    """Build stage vocabulary section for system prompt."""
    lines = []
    stages = stage_vocab or sorted(STAGE_DESCRIPTIONS.keys())
    for stage in stages:
        desc = STAGE_DESCRIPTIONS.get(stage, "")
        lines.append(f"- **{stage}**: {desc}")
    return "\n".join(lines)


def _build_system_prompt(corpus_knowledge: str,
                         transition_graph: str,
                         stage_vocab: list = None) -> str:
    """Construct the system prompt with graph traversal instructions."""
    stage_text = _build_stage_vocab_text(stage_vocab)
    return f"""You are a mineral processing engineer designing process flowsheets. You have been given the full graph network from a corpus of NI 43-101 mining technical reports.

YOUR TASK: Traverse the graph to make engineering decisions. If you receive an ML model's predictions, you MUST actively correct them — add missing stages and remove unjustified ones based on the graph evidence. Do NOT rubber-stamp the ML output.

You will receive:
1. The target project's FULL GRAPH — every context node (report section) with features, every edge with weights, and the influence scores showing which sections drive which process stages.
2. CROSS-DOCUMENT NEIGHBORS — similar projects with their graphs and known flowsheets, linked by similarity scores so you can see WHY they're similar.
3. The CORPUS TRANSITION GRAPH — accumulated stage-to-stage flow patterns from all documents.

## How to Traverse the Graph
- Start at the context nodes. Read the geology section's keywords and numerics to determine ore type, deposit type, and mineralogy.
- Follow the "section → stage influence" edges — these DeBERTa scores tell you which report sections provide evidence for which process stages.
- Check cross-document neighbors — if a similar project (high geology + metallurgy similarity) uses a specific circuit, that's strong evidence.
- Validate connections against the corpus transition graph — high-count transitions are well-established process routes.

## Canonical Stage Vocabulary
{stage_text}

{corpus_knowledge}

{transition_graph}

## Output Format
Return ONLY valid JSON. Use ONLY the exact stage names provided in the input — do not rename or invent new names.

{{
  "connections": [
    {{"from": "stage_a", "to": "stage_b"}},
    {{"from": "stage_b", "to": "stage_c"}},
    {{"from": "stage_c", "to": "stage_b", "label": "recycle"}}
  ],
  "stages": ["stage_a", "stage_b", "stage_c"],
  "reasoning": {{
    "stage_a -> stage_b": "why this connection exists"
  }}
}}

## Rules
- The "from" and "to" fields MUST use the exact stage names from the input list
- Do NOT invent new stage names or IDs
- Do NOT add stages that are not in the input list
- Do NOT remove stages from the input list
- The "stages" field MUST be the exact same list as the input stages
- Include recycle loops where engineering practice requires them (e.g., cyclone -> ball_mill AND ball_mill -> cyclone)
- Include branching where needed (e.g., one stage feeding both concentrate and tails paths)
- Every stage must appear in at least one connection"""


def _build_user_prompt(head_grade: float, deposit_type: str, ore_type: str,
                       target_graph_text: str,
                       cross_doc_text: str,
                       similar_profiles: list[str],
                       mode: str = "standalone",
                       xgb_result: dict = None,
                       throughput_tpd: float = 0,
                       extra_context: str = "") -> str:
    """Construct the user prompt with full graph data for traversal."""
    lines = ["Design a process flowsheet by traversing the graph data below."]
    lines.append("Include equipment specifications (count, size) for each stage.")
    lines.append("")
    lines.append("## Target Project Parameters")
    if head_grade > 0:
        lines.append(f"- Head grade: {head_grade}% Cu")
    if deposit_type:
        lines.append(f"- Deposit type: {deposit_type}")
    if ore_type:
        lines.append(f"- Ore type: {ore_type}")
    if throughput_tpd > 0:
        lines.append(f"- Throughput: {throughput_tpd:,.0f} tpd")
    if extra_context:
        lines.append(f"- Additional context: {extra_context}")

    # Full graph topology for the target (or best-match) document
    if target_graph_text:
        lines.append("")
        lines.append("## Target Project Graph (traverse this)")
        lines.append(target_graph_text)

    # Cross-document edges
    if cross_doc_text:
        lines.append("")
        lines.append("## Cross-Document Similarity Graph")
        lines.append(cross_doc_text)

    # Similar project profiles (compact summaries for additional context)
    if similar_profiles:
        lines.append("")
        lines.append("## Similar Projects (compact profiles)")
        for i, profile in enumerate(similar_profiles, 1):
            lines.append(f"### Reference Project {i}")
            lines.append(profile)
            lines.append("")

    if mode == "augment" and xgb_result:
        lines.append("")
        lines.append("## STAGES TO CONNECT")
        lines.append("")
        stages = xgb_result.get("stages", [])
        lines.append("The following stages have been predicted for this project.")
        lines.append("Use ONLY these exact stage names in your connections.")
        lines.append("Do NOT add any stages. Do NOT rename them. Do NOT remove them.")
        lines.append("")
        for s in stages:
            lines.append(f"  - {s}")
        lines.append("")
        lines.append("Connect these stages into a process flowsheet.")
        lines.append("A stage can appear multiple times in the connections (e.g., cyclone -> ball_mill AND ball_mill -> cyclone for a recycle loop).")
        lines.append("Every stage must have at least one connection.")
        lines.append("Use the graph context above to determine the correct ordering.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM API call (Amazon Bedrock)
# ---------------------------------------------------------------------------

def _call_bedrock(system_prompt: str, user_prompt: str,
                  model_id: str = None, region: str = None) -> dict:
    """Call Amazon Bedrock with Claude and parse JSON response."""
    import boto3

    model_id = model_id or os.getenv(
        "LLM_PREDICTOR_MODEL",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
    )
    region = region or os.getenv(
        "LLM_PREDICTOR_REGION",
        os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    client = boto3.client("bedrock-runtime", region_name=region)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "temperature": 0.3,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
    }

    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]

    return _parse_json_response(text)


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and preamble."""
    import re
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first { ... last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    logger.error(f"Failed to parse JSON from LLM response: {text[:500]}")
    return {"stages": [], "connections": [], "reasoning": {},
            "decision_tree": [], "confidence": 0.0,
            "parse_error": text[:1000]}


def _call_llm(system_prompt: str, user_prompt: str) -> dict:
    """Route to the configured LLM provider (Bedrock or direct Anthropic API)."""
    provider = os.getenv("LLM_PREDICTOR_PROVIDER", "bedrock")

    if provider == "bedrock":
        return _call_bedrock(system_prompt, user_prompt)
    elif provider == "anthropic":
        # Direct Anthropic API
        import anthropic
        client = anthropic.Anthropic()
        model = os.getenv("LLM_PREDICTOR_MODEL", "claude-sonnet-4-20250514")
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return _parse_json_response(response.content[0].text)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def predict_flowsheet_llm(index: dict, head_grade: float,
                          deposit_type: str = "", ore_type: str = "",
                          mode: str = "standalone",
                          xgb_result: dict = None,
                          graphs: list[dict] = None,
                          cross_doc_data: dict = None,
                          top_k: int = 5,
                          throughput_tpd: float = 0,
                          extra_context: str = "") -> dict:
    """Predict a process flowsheet using LLM graph traversal.

    The LLM receives the full graph topology — nodes, edges, weights,
    cross-doc links — and traverses it to build a decision tree that
    produces the flowsheet with equipment specifications.

    Args:
        index: loaded inference_index.json (docs, transition_counts, etc.)
        head_grade: copper head grade in percent (e.g. 0.5 for 0.5% Cu)
        deposit_type: e.g. "porphyry", "skarn", "vms"
        ore_type: e.g. "sulfide", "oxide", "mixed"
        mode: "standalone" (LLM only) or "augment" (refine XGBoost output)
        xgb_result: XGBoost prediction dict (required for augment mode)
        graphs: list of document graph JSONs (full graph data)
        cross_doc_data: cross_doc_edges.json data (neighbor similarities)
        top_k: number of similar projects to include
        throughput_tpd: plant throughput in tonnes per day (for equipment sizing)
        extra_context: any additional project info (e.g. "gold credit expected")

    Returns:
        {
            "stages": [ordered list of stage IDs],
            "connections": [{"from": ..., "to": ..., "rationale": ...}],
            "stage_probabilities": {stage_id: confidence},
            "reasoning": {stage_id: "graph evidence for this stage"},
            "decision_tree": [{"decision": ..., "evidence": ..., "choice": ...}],
            "similar_documents": [doc IDs used as reference],
            "model": "llm" or "hybrid",
            "input": {...},
        }
    """
    transition_counts = index.get("transition_counts", {})
    stage_vocab = list(STAGE_DESCRIPTIONS.keys())
    graphs = graphs or []
    cross_doc_data = cross_doc_data or {}

    # Find the best-matching document to use as the traversal anchor
    target_graph, target_doc_id = _find_best_match_graph(
        head_grade, deposit_type, ore_type, index, graphs,
    )

    # Layer 1b: Full graph topology of the target document
    target_graph_text = ""
    if target_graph:
        target_graph_text = serialize_graph_topology(target_graph)

    # Cross-document edges for the target
    cross_doc_text = ""
    if target_doc_id and cross_doc_data:
        cross_doc_text = serialize_cross_doc_edges(
            target_doc_id, cross_doc_data, graphs, top_k=top_k,
        )

    # Layer 2: Corpus knowledge (conditional rules + stage frequency)
    corpus_knowledge = build_corpus_knowledge(
        transition_counts, stage_vocab, graphs,
    )

    # Corpus transition graph (full traversable structure)
    transition_graph_text = serialize_transition_graph(transition_counts)

    # Layer 3: Similar project profiles (compact, for additional context)
    similar_profiles = retrieve_similar_profiles(
        head_grade, deposit_type, ore_type, index, graphs, top_k=top_k,
    )

    # Build prompts
    system_prompt = _build_system_prompt(
        corpus_knowledge, transition_graph_text, stage_vocab,
    )
    user_prompt = _build_user_prompt(
        head_grade, deposit_type, ore_type,
        target_graph_text, cross_doc_text,
        similar_profiles, mode=mode, xgb_result=xgb_result,
        throughput_tpd=throughput_tpd, extra_context=extra_context,
    )

    logger.info(f"LLM prediction: mode={mode}, "
                f"system_tokens≈{len(system_prompt.split())}, "
                f"user_tokens≈{len(user_prompt.split())}, "
                f"target_doc={target_doc_id}")

    # Call LLM
    llm_output = _call_llm(system_prompt, user_prompt)

    # Validate and normalize output
    process_stages = llm_output.get("process_stages", [])
    stages = llm_output.get("stages", [])
    connections = llm_output.get("connections", [])
    reasoning = llm_output.get("reasoning", {})
    decision_tree = llm_output.get("decision_tree", [])
    equipment = llm_output.get("equipment", {})
    confidence = llm_output.get("confidence", 0.5)

    # In augment mode, stages are locked from XGBoost — pass through unchanged
    if mode == "augment" and xgb_result:
        stages = xgb_result.get("stages", [])
        # Only keep connections that reference the locked stage names
        locked = set(stages)
        connections = [c for c in connections
                       if c.get("from") in locked and c.get("to") in locked]
    elif process_stages:
        valid_stages = set(STAGE_DESCRIPTIONS.keys())
        process_stages = [ps for ps in process_stages
                          if ps.get("type", "") in valid_stages]
        stages = sorted(set(ps["type"] for ps in process_stages))
        stage_ids = {ps["id"] for ps in process_stages}
        connections = [c for c in connections
                       if c.get("from") in stage_ids and c.get("to") in stage_ids]
    else:
        valid_stages = set(STAGE_DESCRIPTIONS.keys())
        stages = [s for s in stages if s in valid_stages]

    # Build stage probabilities from confidence
    stage_probs = {s: confidence for s in stages}

    # Get similar doc IDs
    similar_doc_ids = []
    for profile in similar_profiles:
        for line in profile.split("\n"):
            if line.startswith("Project: "):
                similar_doc_ids.append(line.replace("Project: ", "").strip())
                break

    # Attempt layout
    positions = {}
    try:
        from layout_gat import layout_flowsheet
        positions = layout_flowsheet(
            model_path=None,
            stages=stages,
            connections=[{"from": c["from"], "to": c["to"],
                         "corpus_count": transition_counts.get(
                             f"{c['from']}|{c['to']}", 1)}
                         for c in connections],
        )
    except Exception:
        pass

    return {
        "process_stages": process_stages,
        "stages": stages,
        "connections": connections,
        "stage_probabilities": stage_probs,
        "reasoning": reasoning,
        "decision_tree": decision_tree,
        "equipment": equipment,
        "similar_documents": similar_doc_ids,
        "positions": positions,
        "model": "hybrid" if mode == "augment" else "llm",
        "input": {
            "head_grade": head_grade,
            "deposit_type": deposit_type,
            "ore_type": ore_type,
            "throughput_tpd": throughput_tpd,
        },
    }


def _find_best_match_graph(head_grade: float, deposit_type: str,
                           ore_type: str, index: dict,
                           graphs: list[dict]) -> tuple:
    """Find the graph JSON for the document most similar to the input params.

    Returns (graph_dict, doc_id) or (None, None) if no match.
    """
    if not graphs:
        return None, None

    docs = index.get("docs", [])
    graph_by_id = {g.get("doc_id"): g for g in graphs}

    best_doc_id = None
    best_score = -1.0

    for doc in docs:
        doc_id = doc.get("doc_id", "")
        if doc_id not in graph_by_id:
            continue

        score = 0.0
        meta = doc.get("metadata", {})

        grade_vals = meta.get("grade_values", [])
        if grade_vals and head_grade > 0:
            min_dist = min(abs(g - head_grade) for g in grade_vals)
            score += 1.0 / (1.0 + min_dist * 5.0) * 2.0

        if deposit_type:
            if deposit_type.lower() in meta.get("deposit_types", []):
                score += 1.5

        if ore_type:
            if ore_type.lower() in meta.get("ore_types", []):
                score += 1.0

        if doc.get("has_labels"):
            score *= 1.2

        if score > best_score:
            best_score = score
            best_doc_id = doc_id

    if best_doc_id:
        return graph_by_id[best_doc_id], best_doc_id
    return None, None


# ---------------------------------------------------------------------------
# Evaluation: predict from graph context (strips ground truth)
# ---------------------------------------------------------------------------

def strip_stage_nodes(graph: dict) -> dict:
    """Return a copy of the graph with stage nodes and stage_transition edges removed.

    This prevents the LLM from seeing the ground truth during evaluation.
    """
    blind = {k: v for k, v in graph.items() if k not in ("nodes", "edges")}
    blind["nodes"] = [n for n in graph.get("nodes", []) if n.get("type") != "stage"]
    blind["edges"] = [e for e in graph.get("edges", [])
                      if e.get("type") != "stage_transition"
                      and e.get("type") != "context_influences_stage"]
    blind["stage_sequence"] = []
    blind["num_stage_nodes"] = 0
    return blind


def extract_params_from_graph(graph: dict) -> dict:
    """Extract deposit_type, ore_type, head_grade from a graph's context nodes."""
    deposit_type = ""
    ore_type = ""
    head_grade = 0.0

    for node in graph.get("nodes", []):
        if node.get("type") != "context":
            continue
        feats = node.get("features", {})
        group = node.get("group", "")

        if group == "geology":
            for kw in DEPOSIT_TYPES:
                if feats.get(f"kw_{kw}", 0) > 0:
                    deposit_type = kw
                    break
            for kw in ORE_TYPES:
                if feats.get(f"kw_{kw}", 0) > 0:
                    ore_type = kw
                    break

        if group == "economics":
            grade = feats.get("grade_cu_mean", 0) or feats.get("num_mean", 0)
            if grade > 0:
                head_grade = grade

    return {
        "deposit_type": deposit_type,
        "ore_type": ore_type,
        "head_grade": head_grade,
    }


def predict_from_graph(graph: dict, all_graphs: list[dict],
                       cross_doc_data: dict = None,
                       transition_counts: dict = None,
                       top_k: int = 5,
                       exclude_doc_id: str = None,
                       exclude_doc_ids: set = None,
                       xgb_result: dict = None) -> dict:
    """Predict flowsheet from a graph's context — for evaluation.

    Strips stage nodes (ground truth) from the target graph, extracts
    project parameters from context nodes, and runs the LLM predictor.
    Similar docs exclude ALL held-out docs to prevent data leakage.

    Args:
        graph: full document graph JSON (will be stripped of stages)
        all_graphs: all graphs in corpus (for similar doc retrieval)
        cross_doc_data: cross-doc edges dict
        transition_counts: corpus transition counts
        top_k: number of similar docs as examples
        exclude_doc_id: single doc ID to exclude (the target)
        exclude_doc_ids: set of ALL doc IDs to exclude (val + test sets)
        xgb_result: XGBoost prediction dict (for hybrid/augment mode)

    Returns:
        Same schema as predict_flowsheet_llm()
    """
    doc_id = graph.get("doc_id", graph.get("document_id", "unknown"))
    exclude_doc_id = exclude_doc_id or doc_id

    # Extract parameters from context nodes
    params = extract_params_from_graph(graph)

    # Strip ground truth
    blind_graph = strip_stage_nodes(graph)
    blind_graph["doc_id"] = doc_id

    # Build corpus knowledge from transition counts
    transition_counts = transition_counts or {}
    stage_vocab = list(STAGE_DESCRIPTIONS.keys())
    corpus_knowledge = build_corpus_knowledge(
        transition_counts, stage_vocab, all_graphs,
    )

    # Build exclusion set and neighbor list FIRST
    excluded = exclude_doc_ids or set()
    if exclude_doc_id:
        excluded = excluded | {exclude_doc_id}
    neighbor_graphs = [g for g in all_graphs
                       if g.get("doc_id", g.get("document_id", "")) not in excluded]

    # Serialize the blind graph
    target_graph_text = serialize_graph_topology(blind_graph)

    # Cross-doc edges (filter out held-out docs)
    cross_doc_text = ""
    if cross_doc_data:
        filtered_cross = {}
        if doc_id in cross_doc_data:
            entry = cross_doc_data[doc_id]
            filtered_neighbors = [
                nb for nb in entry.get("neighbors", [])
                if nb.get("doc_id", "") not in excluded
            ]
            filtered_cross[doc_id] = {"neighbors": filtered_neighbors}
        cross_doc_text = serialize_cross_doc_edges(
            doc_id, filtered_cross, neighbor_graphs, top_k=top_k,
        )

    # Transition graph
    transition_graph_text = serialize_transition_graph(transition_counts)
    similar_profiles = []
    for g in neighbor_graphs[:top_k * 3]:
        # Only include labeled docs as examples
        has_stages = any(n.get("type") == "stage" for n in g.get("nodes", []))
        if has_stages:
            similar_profiles.append(serialize_document_profile(g))
        if len(similar_profiles) >= top_k:
            break

    # Build prompts
    mode = "augment" if xgb_result else "standalone"
    system_prompt = _build_system_prompt(
        corpus_knowledge, transition_graph_text, stage_vocab,
    )
    user_prompt = _build_user_prompt(
        params["head_grade"], params["deposit_type"], params["ore_type"],
        target_graph_text, cross_doc_text, similar_profiles,
        mode=mode, xgb_result=xgb_result,
    )

    logger.info(f"Eval predict {doc_id[:20]}...: "
                f"mode={mode}, grade={params['head_grade']:.2f}, "
                f"deposit={params['deposit_type']}, "
                f"ore={params['ore_type']}")

    # Call LLM
    llm_output = _call_llm(system_prompt, user_prompt)

    # In augment mode: lock stages from XGBoost, only take connections from LLM
    if xgb_result:
        stages = xgb_result.get("stages", [])
        locked = set(stages)
        connections = [c for c in llm_output.get("connections", [])
                       if c.get("from") in locked and c.get("to") in locked]
    else:
        stages = llm_output.get("stages", [])
        connections = llm_output.get("connections", [])

    return {
        "stages": stages,
        "connections": connections,
        "reasoning": llm_output.get("reasoning", {}),
        "decision_tree": llm_output.get("decision_tree", []),
        "equipment": llm_output.get("equipment", {}),
        "confidence": llm_output.get("confidence", 0.5),
        "doc_id": doc_id,
        "input_params": params,
    }
