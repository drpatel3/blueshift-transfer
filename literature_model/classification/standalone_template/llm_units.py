"""LLM equipment-count annotation layer — strict annotation only.

HARD CONSTRAINT: This layer MUST NOT mutate the stage or edge sets chosen by
the V2 XGBoost predictor. It only adds `{type, count}` commentary per stage,
one LLM call at a time, scoped to a single stage so the model cannot even
be asked to change the stage set. A post-hoc assertion verifies invariance
and raises on divergence.

Public API:
    predict_units(prediction, data, k=5, llm_model="gpt-4.1") -> dict
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


# Allowed equipment types per stage, derived from TERMS.md.
# The LLM is instructed to pick only from these. Keeps downstream output clean.
STAGE_EQUIPMENT_TYPES: dict[str, list[str]] = {
    "adsorption":        ["cil", "cip", "cic", "carbon_column"],
    "agglomeration":     ["agglomerator", "agglomeration_drum"],
    "bin":               ["ore_bin", "fine_ore_bin", "concentrate_bin", "silo"],
    "crusher":           ["jaw_crusher", "cone_crusher", "gyratory_crusher",
                          "primary_crusher", "secondary_crusher",
                          "tertiary_crusher"],
    "cyclone":           ["hydrocyclone", "cyclone_cluster", "classifier"],
    "electrowinning":    ["ew_cell", "electrowinning_cell"],
    "elution":           ["elution_column", "acid_wash_vessel", "strip_vessel"],
    "feeder":            ["apron_feeder", "belt_feeder", "reclaim_feeder"],
    "filter":            ["filter_press", "vacuum_filter", "pressure_filter",
                          "belt_filter"],
    "flotation":         ["rougher_cell", "cleaner_cell", "scavenger_cell",
                          "column_cell", "flash_flotation_cell"],
    "gravity":           ["knelson", "falcon", "spirals", "jig",
                          "shaking_table", "dms_cyclone"],
    "input":             ["rom_feed", "feed_source"],
    "kiln":              ["rotary_kiln", "calciner", "roaster",
                          "regeneration_kiln"],
    "leach":             ["leach_tank", "heap_leach_pad", "autoclave",
                          "pressure_leach_vessel"],
    "mill":              ["ball_mill", "sag_mill", "ag_mill", "rod_mill",
                          "hpgr", "tower_mill"],
    "product_recovery":  ["merrill_crowe_unit", "precipitation_reactor",
                          "dore_furnace", "refinery_unit"],
    "regrind":           ["regrind_mill", "regrind_ball_mill",
                          "isamill", "vertimill"],
    "screen":            ["vibrating_screen", "grizzly", "banana_screen",
                          "sizing_screen"],
    "solution_recovery": ["sx_mixer_settler", "ion_exchange_column",
                          "carbon_column"],
    "stockpile":         ["coarse_ore_stockpile", "rom_stockpile",
                          "surge_pile"],
    "tailing":           ["tailings_pond", "tailings_storage_facility",
                          "dry_stack_tailings"],
    "tank":              ["agitation_tank", "conditioning_tank",
                          "storage_tank", "neutralization_tank"],
    "thickener":         ["high_rate_thickener", "conventional_thickener",
                          "concentrate_thickener", "tailings_thickener",
                          "ccd_thickener"],
}


# Lexical keywords for matching tables to a stage. Superset of TERMS.md synonyms.
_STAGE_KEYWORDS: dict[str, list[str]] = {
    "adsorption":        ["cil", "cip", "cic", "carbon column",
                          "carbon-in-leach", "carbon-in-pulp", "adsorption"],
    "agglomeration":     ["agglomerat", "drum agglom"],
    "bin":               ["ore bin", "fine ore bin", "concentrate bin", "silo"],
    "crusher":           ["crusher", "crushing", "jaw", "cone", "gyratory",
                          "chancad"],
    "cyclone":           ["cyclone", "hydrocyclone", "classifier",
                          "classification", "ciclon"],
    "electrowinning":    ["electrowinning", "ew cell", "cathode"],
    "elution":           ["elution", "acid wash", "strip ", "stripping"],
    "feeder":            ["apron feeder", "belt feeder", "reclaim feeder"],
    "filter":            ["filter press", "filter", "vacuum filter",
                          "pressure filter", "filtration", "filtro"],
    "flotation":         ["flotation", "rougher", "cleaner", "scavenger",
                          "column cell"],
    "gravity":           ["gravity", "knelson", "falcon", "jig", "spiral",
                          "shaking table", "dms"],
    "input":             ["feed rate", "throughput", "plant feed",
                          "design feed"],
    "kiln":              ["kiln", "calciner", "roaster", "furnace",
                          "regeneration"],
    "leach":             ["leach", "heap leach", "tank leach", "autoclave",
                          "pressure oxidation", "lixiviaci"],
    "mill":              ["ball mill", "sag mill", "ag mill", "rod mill",
                          "grinding", "hpgr", "tower mill", "molino"],
    "product_recovery":  ["merrill crowe", "merrill-crowe", "precipitation",
                          "dore", "doré", "refinery", "cementation"],
    "regrind":           ["regrind", "isamill", "vertimill"],
    "screen":            ["screen", "grizzly", "sizing", "zaranda"],
    "solution_recovery": ["solvent extraction", "sx ", "sx/ew",
                          "ion exchange"],
    "stockpile":         ["stockpile", "rom"],
    "tailing":           ["tailing", "tailings", "tmf", "tsf"],
    "tank":              ["agitation tank", "conditioning tank",
                          "storage tank", "neutralization tank"],
    "thickener":         ["thickener", "thickening", "ccd", "espesador"],
}


def _rank_tables_for_stage(tables: list[dict], stage_name: str) -> list[tuple[int, dict]]:
    """Score tables by lexical match to stage keywords. Returns (score, table)."""
    keywords = _STAGE_KEYWORDS.get(stage_name, [stage_name.replace("_", " ")])
    scored: list[tuple[int, dict]] = []
    for t in tables:
        text_parts = [
            str(t.get("caption", "") or ""),
            " ".join(str(h) for h in (t.get("headers") or [])),
        ]
        for row in (t.get("rows") or [])[:5]:
            text_parts.append(" ".join(str(c) for c in row))
        blob = " ".join(text_parts).lower()
        score = sum(1 for kw in keywords if kw in blob)
        if score > 0:
            scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    return scored


def _serialize_table(table: dict, max_rows: int = 10) -> str:
    """Flatten a table into a compact text block for the LLM."""
    caption = (table.get("caption") or "").strip()
    headers = [str(h) for h in (table.get("headers") or [])]
    rows = (table.get("rows") or [])[:max_rows]

    lines = []
    if caption:
        lines.append(f"Table: {caption}")
    if headers:
        lines.append("  " + " | ".join(headers))
    for row in rows:
        cells = [str(c) for c in row]
        lines.append("  " + " | ".join(cells))
    if len(table.get("rows") or []) > max_rows:
        lines.append(f"  ... ({len(table['rows']) - max_rows} more rows)")
    return "\n".join(lines)


def _build_neighbor_context(neighbors: list[tuple[str, float]],
                            doc_tables: dict[str, list[dict]],
                            stage_name: str,
                            max_tables_per_neighbor: int = 5) -> str:
    """Build the neighbor-table context block for one stage."""
    blocks: list[str] = []
    for idx, (doc_id, grade) in enumerate(neighbors):
        tables = doc_tables.get(doc_id, [])
        if not tables:
            continue
        ranked = _rank_tables_for_stage(tables, stage_name)[:max_tables_per_neighbor]
        if not ranked:
            continue
        label = chr(ord("A") + idx)
        header = f"Project {label} (doc_id={doc_id}, Cu={grade:.3f}%):"
        body = "\n\n".join(_serialize_table(t) for _, t in ranked)
        blocks.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(blocks)


_SYSTEM_PROMPT = (
    "You are a mineral processing engineer. "
    "Respond with valid JSON only — no prose, no markdown fences."
)


def _call_bedrock(user_prompt: str, model_id: str | None = None) -> str:
    """Call Amazon Bedrock with Claude. Returns raw text content."""
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError(
            "boto3 not installed — run `pip install -r requirements.txt`"
        ) from e

    model_id = (
        model_id
        or os.getenv("LLM_UNITS_MODEL")
        or "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    region = (
        os.getenv("LLM_UNITS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )

    client = boto3.client("bedrock-runtime", region_name=region)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.1,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    resp = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"]


def _call_llm(user_prompt: str, model_override: str | None = None) -> str:
    """Route to Bedrock (the only supported provider). Returns raw text content.

    Model override:
        LLM_UNITS_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
    """
    return _call_bedrock(user_prompt, model_id=model_override)


def _build_prompt(stage_name: str, target_grade: float,
                  allowed_types: list[str], neighbor_context: str) -> str:
    return f"""You are sizing equipment for ONE specific stage of a copper processing plant.

Target project: Cu head grade = {target_grade:.3f}%.
Stage to annotate: "{stage_name}"

The following tables are from the {neighbor_context.count('Project ') or 0} most-analogous real NI 43-101 projects (nearest by head grade). They contain equipment specifications relevant to this stage:

{neighbor_context if neighbor_context else "(No relevant tables found for this stage. Infer typical equipment from stage name.)"}

TASK
Based on the tables above, decide what equipment the target project would realistically use for the "{stage_name}" stage: which types of equipment, and how many of each.

CONSTRAINTS
- Do NOT comment on whether this stage belongs in the flowsheet. That decision is already made — only produce equipment for this stage.
- "type" MUST be one of: {json.dumps(allowed_types)}
- "count" MUST be a positive integer.
- Prefer counts grounded in the neighbor tables above. If the tables show 2 ball mills for a similar-grade project, 2 is a defensible number. If no evidence exists, pick a conservative default of 1.

OUTPUT — return ONLY this JSON, no prose:
{{"units": [{{"type": "<one of the allowed types>", "count": <int>}}, ...]}}
"""


def _parse_units(raw: str, allowed_types: list[str]) -> list[dict[str, Any]]:
    """Parse LLM response → list of {type, count}. Drops invalid entries.

    Returns [] on unrecoverable parse failure — caller applies a safe default.
    """
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return []
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    raw_units = obj.get("units", []) if isinstance(obj, dict) else []
    if not isinstance(raw_units, list):
        return []

    allowed_set = set(allowed_types)
    cleaned: list[dict[str, Any]] = []
    for u in raw_units:
        if not isinstance(u, dict):
            continue
        utype = u.get("type")
        ucount = u.get("count")
        if utype not in allowed_set:
            continue
        try:
            ucount_int = int(ucount)
        except (TypeError, ValueError):
            continue
        if ucount_int < 1:
            continue
        cleaned.append({"type": utype, "count": ucount_int})
    return cleaned


def _safe_default(stage_name: str) -> list[dict[str, Any]]:
    """Fallback equipment when LLM is unavailable or returns garbage.

    Picks the first allowed type for the stage with count=1. Never fails.
    """
    types = STAGE_EQUIPMENT_TYPES.get(stage_name, [f"{stage_name}_unit"])
    return [{"type": types[0], "count": 1}]


def predict_units(prediction: dict, data: dict, k: int = 5,
                  llm_model: str | None = None,
                  max_tables_per_neighbor: int = 5) -> dict:
    """Annotate each XGBoost-chosen stage with {type, count} equipment units.

    INVARIANT: The stage list and edge list are read-only. Post-hoc assertion
    raises on divergence.
    """
    from predict import _find_closest_doc

    pre_stages = list(prediction["stages"])
    pre_edges = [(e["src"], e["dst"]) for e in prediction["edges"]]

    target_grade = float(prediction["cu_grade"])
    neighbors = _find_closest_doc(target_grade, data, k=k)
    if not neighbors:
        logger.warning("No KNN neighbors found; using safe defaults")
        neighbors = []

    doc_tables: dict[str, list[dict]] = data.get("doc_tables") or {}
    if not doc_tables:
        logger.warning(
            "doc_tables.json is missing or empty. LLM layer will operate "
            "without neighbor-table evidence; counts fall back to safe defaults. "
            "To populate, run the cache-rebuild step from the main repo."
        )

    annotated_stages: list[dict[str, Any]] = []
    llm_errors = 0

    for stage_name in pre_stages:
        allowed_types = STAGE_EQUIPMENT_TYPES.get(
            stage_name, [f"{stage_name}_unit"])

        neighbor_ctx = _build_neighbor_context(
            neighbors, doc_tables, stage_name,
            max_tables_per_neighbor=max_tables_per_neighbor)

        units: list[dict[str, Any]] = []
        if doc_tables and neighbor_ctx:
            prompt = _build_prompt(
                stage_name, target_grade, allowed_types, neighbor_ctx)
            try:
                raw = _call_llm(prompt, model_override=llm_model)
                units = _parse_units(raw, allowed_types)
            except Exception as e:
                logger.warning("LLM call failed for stage %s: %s",
                               stage_name, e)
                llm_errors += 1

        if not units:
            units = _safe_default(stage_name)

        annotated_stages.append({"name": stage_name, "units": units})

    post_stages = [s["name"] for s in annotated_stages]
    assert set(post_stages) == set(pre_stages), (
        f"LLM layer mutated the stage set. Before: {sorted(pre_stages)} "
        f"After: {sorted(post_stages)}")
    assert len(post_stages) == len(pre_stages), (
        f"LLM layer changed stage count. Before: {len(pre_stages)} "
        f"After: {len(post_stages)}")

    result = dict(prediction)
    result["stages"] = annotated_stages
    post_edges = [(e["src"], e["dst"]) for e in result["edges"]]
    assert pre_edges == post_edges, (
        f"LLM layer mutated the edge set. Before: {pre_edges} "
        f"After: {post_edges}")

    if llm_errors:
        result["_llm_errors"] = llm_errors
    return result
