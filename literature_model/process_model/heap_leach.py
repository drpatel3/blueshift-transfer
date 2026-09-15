"""Whole-ore heap leach — primary-crushed ore stacked on a lined pad and
irrigated with H2SO4 PLS. Distinct from `leach.py` (concentrate POX/Albion):

- Heap leach feed is RUN-OF-MINE ORE (after primary crushing only), not
  flotation concentrate. Low capex per tonne but lower extraction.
- Suited to OXIDE Cu and SUPERGENE CHALCOCITE deposits at low grade
  (0.2-0.6% Cu). Not suited to chalcopyrite — extraction <30% over
  industrial cycle times.
- Examples: BHP Spence (chalcocite), Escondida BMOH oxide, Codelco Radomiro
  Tomic, Lomas Bayas, Mantoverde.

Sources:
    Schlitt, W.J. (1992) "Heap leach kinetics and column simulation",
        Mining Engineering 44(8):885-889.
    Mular, A.L. (2002) "Mining Cost Estimating Guide" (SME), Vol 2 §4.2 —
        heap pad capex correlations.
    SME Mineral Processing Handbook (2019), Ch. 22.
    Project disclosures: BHP Spence growth option, Codelco RT-Sulfuros
        feasibility 43-101 reports.
"""
from dataclasses import dataclass


# Acid consumption is dominated by GANGUE dissolution (carbonates, mafics),
# not Cu mineral. Industry typical for chalcocite/oxide ore at 0.4% Cu:
# 10-25 kg H2SO4 per tonne ore. Cu reaction itself is ~3 kg/kg Cu (oxidative)
# so a 0.4% Cu ore contributes ~1.2 kg/t-ore from Cu, ~14 kg/t-ore from
# gangue. Refit per ore mineralogy (high-carbonate ore can hit 50+ kg/t).
DEFAULT_ACID_KG_PER_T_ORE = 15.0  # mid-range chalcocite

# Industrial extraction floors at full residence time (12-18 month cycle).
# Schlitt 1992 column data + project disclosures:
#   - Oxide ore (chrysocolla, malachite): 75-90%
#   - Supergene chalcocite (Cu2S): 65-85%
#   - Mixed oxide/sulfide: 60-75%
#   - Chalcopyrite (CuFeS2): <30% — heap leach is NOT viable
DEFAULT_EXTRACTION_FRACTION = 0.70  # supergene chalcocite mid-range

# PLS Cu concentration feeding SX. Industry typical 2-6 g/L; below 2 g/L
# SX kinetics get marginal, above 6 g/L organic loading saturates.
DEFAULT_PLS_CU_GPL = 4.0


@dataclass
class HeapLeachParams:
    """Whole-ore heap leach kinetics + sizing inputs.

    Defaults are supergene chalcocite at 0.4% Cu, mid-range gangue acid
    consumption, 12-month primary cycle. Override per deposit:
        - Oxide ore        : extraction_fraction=0.85, acid 20 kg/t
        - Mixed oxide/sulf : extraction_fraction=0.65, acid 18 kg/t
        - Chalcopyrite     : DO NOT USE — set leach_enabled (concentrate POX) instead
    """
    extraction_fraction: float = DEFAULT_EXTRACTION_FRACTION
    acid_kg_per_t_ore: float = DEFAULT_ACID_KG_PER_T_ORE
    pls_cu_gpl: float = DEFAULT_PLS_CU_GPL
    # Cycle time only enters capex sizing (pad area = ore-on-heap inventory).
    primary_cycle_days: float = 365.0
    heap_height_m: float = 8.0
    irrigation_rate_l_m2_h: float = 10.0


def size_heap(ore_tph_metric: float, cu_grade: float,
              p: HeapLeachParams) -> dict:
    """Compute Cu extraction, acid demand, and PLS flow for a whole-ore
    heap leach. Pad area is sized off ore-on-heap inventory (ore_tph *
    cycle_time / (density * heap_height)).
    """
    if ore_tph_metric <= 0:
        return {
            "X_Cu": 0.0, "cu_leached_tph": 0.0, "cu_residue_tph": 0.0,
            "acid_kg_per_h": 0.0, "pls_m3_per_h": 0.0,
            "pad_area_m2": 0.0, "ore_on_heap_t": 0.0,
        }
    cu_in_ore_tph = ore_tph_metric * cu_grade
    cu_leached_tph = cu_in_ore_tph * p.extraction_fraction
    # acid_kg_per_t_ore is kg acid per metric tonne of ore, ore_tph_metric
    # is t/h, so the product is already kg/h — NO extra 1000 factor.
    acid_kg_per_h = ore_tph_metric * p.acid_kg_per_t_ore
    # PLS volume from Cu loading: kg Cu / h * 1000 / (g Cu / L) = L/h -> m3/h
    pls_m3_per_h = (cu_leached_tph * 1_000_000.0 / p.pls_cu_gpl) / 1000.0
    # Pad sizing — ore inventory at steady state = ore_tph * 24h * cycle_days
    ore_on_heap_t = ore_tph_metric * 24.0 * p.primary_cycle_days
    bulk_density_t_m3 = 1.7   # crushed Cu ore typical
    pad_area_m2 = ore_on_heap_t / (bulk_density_t_m3 * p.heap_height_m)
    return {
        "X_Cu": p.extraction_fraction,
        "cu_leached_tph": cu_leached_tph,
        "cu_residue_tph": cu_in_ore_tph - cu_leached_tph,
        "acid_kg_per_h": acid_kg_per_h,
        "pls_m3_per_h": pls_m3_per_h,
        "pad_area_m2": pad_area_m2,
        "ore_on_heap_t": ore_on_heap_t,
        "primary_cycle_days": p.primary_cycle_days,
    }


if __name__ == "__main__":
    p = HeapLeachParams()
    diag = size_heap(ore_tph_metric=2000.0, cu_grade=0.004, p=p)
    print("Whole-ore heap leach: 2000 tph @ 0.4% Cu, chalcocite defaults")
    for k, v in diag.items():
        print(f"  {k:25s} {v}")
