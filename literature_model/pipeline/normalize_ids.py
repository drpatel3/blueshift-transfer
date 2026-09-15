import json
import logging
import os
import re

import config

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# English semantic equivalents -> canonical form
SYNONYMS = {
    # Singular/plural
    'cells': 'cell',
    'tailings': 'tailing',
    'tanks': 'tank',

    # Verb/noun forms
    'crushing': 'crusher',
    'grinding': 'mill',
    'screening': 'screen',
    'thickening': 'thickener',
    'filtering': 'filter',
    'filtration': 'filter',
    'leaching': 'leach',

    # Equipment variants
    'cyclones': 'cyclone',
    'mills': 'mill',
    'screens': 'screen',
    'feeders': 'feeder',
    'hoppers': 'hopper',
    'conveyors': 'conveyor',
    'grizzly': 'screen',
    'grasshopper': None,  # remove descriptor
    'furnace': 'kiln',

    # Map to existing groups
    'baghouse': 'filter',
    'scrubber': 'filter',
    'classification': 'cyclone',
    'conditioning': 'tank',
    'cleaners': 'cleaner',
    'cleaner': 'flotation',  # cleaner circuits are flotation
    'rougher': 'flotation',
    'decantation': 'thickener',
    'counter': None,  # remove from counter_current_decantation
    'current': None,
    'separation': 'thickener',  # solid_liquid_separation
    'solid': None,
    'liquid': None,
    'vessel': 'tank',
    'feed': 'feeder',
    'reception': 'hopper',
    'cic': 'adsorption',
    'columns': 'adsorption',
    'INPUT': 'input',
    'OUTPUT': 'output',

    # Crusher types -> crusher
    'jaw': None,  # will be removed
    'cone': None,
    'gyratory': None,

    # Positional prefixes -> remove
    'primary': None,
    'secondary': None,
    'tertiary': None,

    # Ordinals -> remove
    '1st': None,
    '2nd': None,
    '3rd': None,
    'i': None,
    'ii': None,
    'iii': None,

    # Spanish -> English translations
    'celda': 'cell',
    'celdas': 'cell',
    'chancadora': 'crusher',
    'molino': 'mill',
    'bolas': None,  # remove (ball mill -> mill)
    'ciclones': 'cyclone',
    'ciclon': 'cyclone',
    'hidrociclon': 'cyclone',
    'hydrocyclone': 'cyclone',
    'espesador': 'thickener',
    'filtro': 'filter',
    'tolva': 'hopper',
    'tolvas': 'hopper',
    'faja': 'conveyor',
    'zaranda': 'screen',
    'relavera': 'tailing',
    'relaves': 'tailing',
    'tanque': 'tank',
    'tanques': 'tank',

    # Spanish descriptors -> remove
    'limpieza': None,  # cleaner
    'primaria': None,
    'primarias': None,
    'secundaria': None,
    'terciaria': None,
    'recepcion': None,
    'vibratoria': None,
    'de': None,

    # Process actions -> equipment type
    'dewatering': 'filter',
    'storage': 'bin',
    'disposal': 'tailing',
    'drying': 'filter',
    'dryer': 'filter',
    'recycle': 'pond',
    'supply': 'tank',
    'tmf': 'tailing',

    # Descriptor removals
    'shipping': None,
    'handling': None,
    'plant': None,
    'facility': None,
    'collection': None,
    'circuit': None,
    'room': None,
    'reagents': None,
    'final': None,
    'coarse': None,
    'fine': None,
    'loaded': None,
    'barren': None,
    'carbon': None,
    'safety': None,
    'sizing': None,
    'fines': None,
    'attritioning': None,
    'water': None,

    # Flotation circuit terms
    'scavenger': 'flotation',
    'recleaner': 'flotation',
    'flash': 'flotation',
    'jameson': None,
    'incl': None,

    # Additional Spanish
    'piscina': 'pond',
    'poza': 'pond',
    'contingencia': None,
    'finos': None,
    'gruesos': None,
    'prensa': None,
    'quijada': None,
    'conica': None,
    'concentrado': None,
    'relave': 'tailing',
    'agua': None,
    'clarificada': None,
    'pirita': 'flotation',

    # Abbreviations
    'cus': None,
    'smbs': None,
    'rom': 'input',
    'sorting': 'ore_sorting',
    's': None,
    'py': None,
    'no': None,
}

REMOVABLE = {
    # Metals / commodities — stripped for grouping, not from stored names
    'copper', 'cu', 'gold', 'au', 'iron', 'fe', 'lead', 'pb', 'zinc', 'zn',
    'moly', 'silver', 'polymetallic', 'pyrite',
}

# Core equipment/process words - if found, use as the normalized ID
# Order matters: first match wins (e.g. thickener before tailing so tailings_thickener -> thickener)
CORE_WORDS = [
    'input', 'output', 'regrind', 'flotation', 'crusher', 'mill', 'cyclone', 'thickener',
    'filter', 'stockpile', 'hopper', 'feeder', 'screen', 'tank',
    'conveyor', 'leach', 'precipitation', 'cell', 'reactor',
    'pond', 'bin', 'silo', 'kiln', 'furnace', 'adsorption',
    'tailing', 'ore_sorting', 'mol', 'sars',
]

def normalize_id(id_str: str) -> str:
    """Produce a simplified grouping key. Original IDs stay intact in results.json."""
    normalized = id_str.lower()

    # Remove _unit suffix
    normalized = re.sub(r'_unit$', '', normalized)

    # Remove numeric and letter suffixes (_1, _2, _01, _a, _b, etc.)
    normalized = re.sub(r'_\d+$', '', normalized)
    normalized = re.sub(r'_[a-b]$', '', normalized)

    # Split into tokens
    tokens = normalized.split('_')

    # Strip commodity prefixes for grouping only
    tokens = [t for t in tokens if t not in REMOVABLE]
    if not tokens:
        return normalized  # all tokens were noise, return original lowered form

    # Apply synonym mappings
    normalized_tokens = []
    for t in tokens:
        if t in SYNONYMS:
            if SYNONYMS[t] is not None:
                normalized_tokens.append(SYNONYMS[t])
            # else: skip token (remove it)
        else:
            normalized_tokens.append(t)

    if not normalized_tokens:
        return '_'.join(tokens)  # all tokens mapped to None, return pre-synonym form

    # First CORE_WORD match wins
    for core in CORE_WORDS:
        if core in normalized_tokens:
            return core

    # Multi-word CORE_WORDS (e.g. ore_sorting)
    joined = '_'.join(normalized_tokens)
    if joined in CORE_WORDS:
        return joined

    return joined

def build_standard_id_library(results_path: str, dictionary_path: str = None):
    """
    Build a standard library of IDs using semantic normalization.
    Persists and merges with existing dictionary so it grows across pipeline runs.

    Returns:
        tuple: (id_mapping, normalized_groups)
    """
    if dictionary_path is None:
        dictionary_path = str(config.NORMALIZATION_DICT_PATH)

    # Load existing dictionary if it exists
    existing_groups = {}
    if os.path.exists(dictionary_path):
        try:
            with open(dictionary_path, 'r') as f:
                existing_groups = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing_groups = {}

    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}, existing_groups
    if not data:
        return {}, existing_groups

    # Collect all IDs from stages and units
    all_ids = []
    for _, flowsheet in data.items():
        for stage in flowsheet.get('stages', []):
            all_ids.append(stage['id'])
        for unit in flowsheet.get('units', []):
            all_ids.append(unit['id'])

    # Remove duplicates while preserving order
    unique_ids = list(dict.fromkeys(all_ids))

    # Group by normalized form
    normalized_groups = {}  # normalized_id -> [original_ids]
    for original_id in unique_ids:
        norm = normalize_id(original_id)
        if norm not in normalized_groups:
            normalized_groups[norm] = []
        normalized_groups[norm].append(original_id)

    # Merge new IDs into existing dictionary (union, no duplicates)
    for norm, originals in normalized_groups.items():
        if norm in existing_groups:
            merged = list(dict.fromkeys(existing_groups[norm] + originals))
            existing_groups[norm] = merged
        else:
            existing_groups[norm] = originals

    # Save accumulated dictionary
    with open(dictionary_path, 'w') as f:
        json.dump(existing_groups, f, indent=2)

    # Build mapping: first encountered ID becomes the standard
    id_mapping = {}
    for norm, originals in existing_groups.items():
        standard = originals[0]
        for orig in originals:
            id_mapping[orig] = standard

    return id_mapping, existing_groups

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mapping, groups = build_standard_id_library(str(config.RESULTS_PATH))

    logger.info(f"Total unique IDs: {len(mapping)}")
    logger.info(f"Standard IDs (after semantic matching): {len(groups)}")

    logger.info("\nSemantic groups (IDs that match):")
    for norm, originals in sorted(groups.items()):
        if len(originals) > 1:
            logger.info(f"  {norm}: {originals}")

    logger.info("\nStandalone IDs (no matches):")
    for norm, originals in sorted(groups.items()):
        if len(originals) == 1:
            logger.info(f"  {norm}")