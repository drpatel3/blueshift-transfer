"""Stage-level normalization for the mini digital twin pipeline.

Uses normalize_id() from normalize_ids for token-level work (suffixes,
Spanish translations, synonym mappings), then applies STAGE_MAP to
collapse into canonical process stage types.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from normalize_ids import normalize_id

# ---------------------------------------------------------------------------
# Stage normalization bank
# Maps normalize_id() outputs -> canonical process stage types.
# None = utility/noise stage to be filtered out.
# ---------------------------------------------------------------------------
STAGE_MAP = {
    # --- Input / feed ---
    "input": "input",
    "ore": "input",
    "ore_mining_sources": "input",
    "ore_receiving": "input",
    "ore_receiving_and_crushed_ore": "input",
    "open_pit_mining": "input",
    "underground_mine": "input",
    "feeding": "input",
    "feeder": "input",

    # --- Crushing ---
    "crusher": "crusher",
    "comminution": "crusher",

    # --- Ore sorting / pre-concentration ---
    "ore_sorting": "ore_sorting",
    "dms": "ore_sorting",

    # --- Stockpile / surge ---
    "stockpile": "stockpile",
    "crushed_ore_stockpiling": "stockpile",
    "ore_stockpiling": "stockpile",
    "ore_stockpiling_and_reclaim": "stockpile",
    "ore_surge": "stockpile",
    "conveying_and_surge": "stockpile",
    "crushed_ore_bins_2_3": "stockpile",
    "hpgr_stockpiling": "stockpile",

    # --- Conveying ---
    "conveyor": "conveyor",
    "crushed_ore_conveying": "conveyor",
    "material_conveying": "conveyor",
    "transfer_conveying": "conveyor",

    # --- Grinding ---
    "mill": "mill",
    "milling": "mill",
    "ball_milling": "mill",
    "rod_milling": "mill",
    "sag_ball_milling": "mill",
    "sag_milling": "mill",

    # --- Classification ---
    "cyclone": "cyclone",

    # --- Screening ---
    "screen": "screen",

    # --- Flotation (rougher, cleaner, scavenger) ---
    "flotation": "flotation",
    "cleaner": "flotation",
    "barite_cleaner": "flotation",
    "barite_recleaners": "flotation",
    "mo_cleaning": "flotation",
    "molybdenum_cleaning": "flotation",
    "cell": "flotation",

    # --- Regrind ---
    "regrind": "regrind",
    "regrinding": "regrind",

    # --- Gravity concentration ---
    "gravity_concentration": "gravity",
    "future_gravity": "gravity",
    "shaking_tables": "gravity",
    "shaking_tables_rg_cl": "gravity",
    "magnetite_scavenging": "gravity",

    # --- Leaching ---
    "leach": "leach",
    "agglomeration": "leach",
    "ccd_washing": "leach",
    "residue_washing": "leach",
    "acid_washing": "leach",
    "cyanide_recovery": "leach",
    "dynamic_ils": "leach",
    "antigua_ils": "leach",
    "south_dump_ils": "leach",
    "dynamic_pls": "leach",

    # --- Adsorption (CIP/CIL/carbon) ---
    "adsorption": "adsorption",
    "cip": "adsorption",
    "carbon_handling": "adsorption",
    "regeneration": "adsorption",
    "elution": "adsorption",
    "elution_and_heat_exchange": "adsorption",
    "washing_desorption_electrowinning": "adsorption",

    # --- Solvent extraction ---
    "sx": "solvent_extraction",
    "sx_extraction": "solvent_extraction",
    "sx_mixer_settlers": "solvent_extraction",
    "sx_stripping": "solvent_extraction",
    "hydromet_solvent_extraction": "solvent_extraction",
    "solvent_extraction_oxide": "solvent_extraction",
    "solvent_extraction_sulphide": "solvent_extraction",
    "organic_recovery": "solvent_extraction",

    # --- Electrowinning ---
    "electrowinning": "electrowinning",
    "electrowinning_oxide": "electrowinning",
    "electrowinning_sulphide": "electrowinning",
    "ew": "electrowinning",
    "hydromet_electrowinning": "electrowinning",
    "adr_ew": "electrowinning",

    # --- Thickening ---
    "thickener": "thickener",
    "clarification": "thickener",
    "deaeration": "thickener",
    "deoxygenation": "thickener",
    "pls_clarifier": "thickener",

    # --- Filtering ---
    "filter": "filter",

    # --- Precipitation / refining ---
    "precipitation": "precipitation",
    "merrill_crowe_and_deaeration": "precipitation",
    "merrill_crowe_recovery": "precipitation",
    "recovery_merrill_crowe": "precipitation",
    "pgm_cementation": "precipitation",
    "recovery": "precipitation",
    "recovery_refining": "precipitation",
    "refinery": "precipitation",
    "smelting": "precipitation",
    "smelting_and_dore": "precipitation",
    "sulfur_recovery_acid_production": "precipitation",

    # --- Tank / pond ---
    "tank": "tank",
    "pond": "tank",
    "solution_ponds": "tank",
    "pls_ponds": "tank",
    "raffinate_ponds": "tank",
    "return_dam": "tank",
    "dam_1112_mrl": "tank",
    "process_return": "tank",

    # --- Bin / silo / hopper ---
    "bin": "bin",
    "silo": "bin",
    "hopper": "bin",
    "pebble": "bin",

    # --- Tailing ---
    "tailing": "tailing",
    "tails_to_tsf": "tailing",
    "oxide_tails_to_tsf": "tailing",
    "discard_management": "tailing",
    "waste": "tailing",
    "sand": "tailing",
    "backfill": "tailing",
    "future_paste": "tailing",
    "treated_discharge": "tailing",
    "seepage": "tailing",
    "seepage_loss": "tailing",

    # --- Treatment (neutralization, detox, water treatment) ---
    "treatment": "treatment",
    "neutralization": "treatment",
    "neutralization_and_sart_solution": "treatment",
    "neutralization_gypsum": "treatment",
    "sart": "treatment",
    "cyanide_destruction": "treatment",
    "cyanide_detox": "treatment",
    "detox": "treatment",
    "sulphide_removal": "treatment",
    "acid_removal": "treatment",
    "mg_removal": "treatment",
    "removal": "treatment",

    # --- Output / product ---
    "output": "output",
    "concentrate": "output",
    "concentrate_product": "output",
    "bulk_concentrate_product": "output",
    "bullion_product": "output",
    "cathode_product": "output",
    "cathode_product_oxide": "output",
    "cathode_product_sulphide": "output",
    "cathodes_product": "output",
    "cathode_production": "output",
    "dore_product": "output",
    "gypsum_product": "output",
    "magnetite_concentrate": "output",
    "molybdenum_concentrate": "output",
    "molybdenum_concentrate_bagging": "output",
    "oxide_concentrate": "output",
    "precipitate_product": "output",
    "precious_metal": "output",
    "products": "output",
    "concentrate_sulfide": "output",
    "concentrate_enrichment": "output",
    "concentrate_bagging_loading": "output",
    "concentrate_dispatch": "output",
    "concentrate_loadout": "output",
    "concentrate_transport": "output",
    "packaging_dispatch": "output",
    "port": "output",

    # --- Utility / noise (filtered out) ---
    "circuit": None,
    "process": None,
    "processing": None,
    "process_management": None,
    "process_system": None,
    "room": None,
    "reagents": None,
    "reagent_preparation": None,
    "reagent_and_air_services": None,
    "pump_station": None,
    "pump_station_locations": None,
    "reticulation": None,
    "spray_reticulation": None,
    "shaft_15": None,
    "shaft_1ter": None,
    "south_dump_fingerways": None,
}

# Canonical stage types — for fallback substring matching
CANONICAL_STAGES = [
    "input", "crusher", "ore_sorting", "stockpile", "conveyor",
    "mill", "cyclone", "screen", "flotation", "regrind", "gravity",
    "leach", "adsorption", "solvent_extraction", "electrowinning",
    "thickener", "filter", "precipitation", "tank", "bin",
    "tailing", "treatment", "output",
]


def normalize_stage(stage_id):
    """Normalize a raw stage ID to a canonical process stage type.

    Applies normalize_id() for token-level normalization, then looks up
    in STAGE_MAP. Falls back to substring matching against CANONICAL_STAGES.
    Returns None for utility/noise stages that should be filtered out.
    """
    normed = normalize_id(stage_id)

    # Direct lookup
    if normed in STAGE_MAP:
        return STAGE_MAP[normed]

    # Fallback: check if any canonical stage appears as a substring
    for canonical in CANONICAL_STAGES:
        if canonical in normed:
            return canonical

    # Unknown — return as-is so it's visible for future mapping
    return normed
