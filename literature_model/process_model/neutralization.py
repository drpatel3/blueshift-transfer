"""Two-stage lime neutralization — SME Mineral Processing Handbook Ch. 20.

Sources:
    Eq. 20.64  — Stage 1 (pH 4-5): H2SO4 + Ca(OH)2 -> CaSO4.2H2O
    Eq. 20.65  — Stage 1 Fe/As co-precipitation:
                 2 Fe2(SO4)3 + 2 H3AsO4 + H2SO4 + 7 CaCO3 + 13 H2O ->
                     2 FeAsO4 + 2 Fe(OH)3 + 7 CaSO4.2H2O + 7 CO2
                 (Fe:As >= 3:1 mol required)
    Eq. 20.66  — Stage 2 (pH 6-8): MSO4 + Ca(OH)2 + 2 H2O -> M(OH)2 + CaSO4.2H2O

Returns lime demand (kg/h), gypsum production (kg/h), tank volumes
(residence time x stages x flow), and a flag if Fe:As < 3:1.
"""
from dataclasses import dataclass


# Molar masses, g/mol
M_H2SO4 = 98.08
M_CA_OH2 = 74.09
M_GYPSUM = 172.17    # CaSO4.2H2O
M_FE = 55.85
M_AS = 74.92
M_CACO3 = 100.09
M_FEASO4 = 194.86
M_FE_OH3 = 106.87


@dataclass
class NeutralizationParams:
    """Operating point and reagent costs."""
    residence_time_h: float = 1.0   # per stage
    num_stages_each: int = 2        # 2 tanks each for stage 1, stage 2
    lime_purity: float = 0.95       # commercial Ca(OH)2 (~95% active)
    gypsum_solids_kg_per_t_solution: float = 5.0  # carried for tail balance


def stage1_lime_demand_kg_per_h(h2so4_kg_per_h: float,
                                fe_kg_per_h: float = 0.0,
                                as_kg_per_h: float = 0.0,
                                purity: float = 0.95) -> dict:
    """Stage-1 lime demand from acid neutralization (Eq 20.64) plus Fe/As
    co-precipitation (Eq 20.65). Returns a dict of mass flows.

    Eq 20.65 lumps lime use as 7 CaCO3 (= 7 mol Ca(OH)2 equivalent) per
    2 mol Fe + 2 mol As + 1 mol H2SO4. Here we attribute the H2SO4 portion
    to Eq 20.64 (already counted) and add the residual 7 mol Ca(OH)2 per
    2 mol Fe / 2 mol As pair.
    """
    # Eq 20.64: 1 mol H2SO4 -> 1 mol Ca(OH)2 -> 1 mol gypsum
    n_h2so4 = h2so4_kg_per_h * 1000.0 / M_H2SO4    # mol/h
    lime_eq64_mol = n_h2so4
    gypsum_eq64_mol = n_h2so4

    # Eq 20.65 (Fe/As co-precip): per 2 mol Fe and 2 mol As, consume 7 mol
    # of base (less the 1 mol of H2SO4 already counted) and produce 7 mol
    # gypsum, 2 mol FeAsO4, 2 mol Fe(OH)3.
    n_fe = fe_kg_per_h * 1000.0 / M_FE
    n_as = as_kg_per_h * 1000.0 / M_AS
    fe_as_mol = min(n_fe, n_as) / 2.0  # rate-limiting pair count
    fe_as_warn = None
    if n_as > 0 and n_fe / n_as < 3.0:
        fe_as_warn = (f"Fe:As mol ratio {n_fe/n_as:.2f} < 3:1 minimum; "
                      "increase Fe addition or limit As loading")

    lime_eq65_mol = 6.0 * fe_as_mol     # 7 mol total - 1 mol already in eq 64 = 6
    gypsum_eq65_mol = 7.0 * fe_as_mol
    feaso4_mol = 2.0 * fe_as_mol
    fe_oh3_mol = 2.0 * fe_as_mol

    lime_total_mol = lime_eq64_mol + lime_eq65_mol
    lime_kg_per_h = lime_total_mol * M_CA_OH2 / 1000.0 / purity

    gypsum_total_mol = gypsum_eq64_mol + gypsum_eq65_mol

    return {
        "lime_kg_per_h_stage1": lime_kg_per_h,
        "gypsum_kg_per_h_stage1": gypsum_total_mol * M_GYPSUM / 1000.0,
        "feaso4_kg_per_h": feaso4_mol * M_FEASO4 / 1000.0,
        "fe_oh3_kg_per_h": fe_oh3_mol * M_FE_OH3 / 1000.0,
        "fe_as_warn": fe_as_warn,
    }


def stage2_lime_demand_kg_per_h(metals_sulfate_kg_per_h: dict,
                                purity: float = 0.95) -> dict:
    """Stage-2 base-metal precipitation (Eq 20.66): 1 mol MSO4 -> 1 mol
    Ca(OH)2 + 1 mol gypsum + 1 mol M(OH)2.

    `metals_sulfate_kg_per_h` is a dict {metal_symbol: kg/h MSO4}; the
    caller supplies sulfate-form mass flows. Lime is summed across metals.
    """
    # Approximate metal-sulfate molar mass: skip exact M lookup, treat all
    # MSO4 at average molar mass 175 g/mol (Cu=159.6, Zn=161.4, Ni=154.8;
    # close enough for demand estimation).
    M_MSO4_AVG = 175.0
    total_mso4_mol = sum(metals_sulfate_kg_per_h.values()) * 1000.0 / M_MSO4_AVG
    lime_kg = total_mso4_mol * M_CA_OH2 / 1000.0 / purity
    gypsum_kg = total_mso4_mol * M_GYPSUM / 1000.0
    return {
        "lime_kg_per_h_stage2": lime_kg,
        "gypsum_kg_per_h_stage2": gypsum_kg,
        "m_oh2_kg_per_h": total_mso4_mol * 90.0 / 1000.0,  # M(OH)2 ~ 90 avg
    }


def total_tank_volume_m3(slurry_m3_per_h: float, p: NeutralizationParams) -> float:
    """Sum of stage-1 + stage-2 tank volumes. Each side has p.num_stages_each
    tanks of (slurry * residence) volume."""
    one_side = slurry_m3_per_h * p.residence_time_h * p.num_stages_each
    return 2.0 * one_side   # stage 1 + stage 2


def size_neutralization(h2so4_kg_per_h: float,
                        fe_kg_per_h: float,
                        as_kg_per_h: float,
                        metals_sulfate_kg_per_h: dict,
                        slurry_m3_per_h: float,
                        p: NeutralizationParams) -> dict:
    """Full two-stage sizing. Stage 1 handles acid + Fe/As; Stage 2 handles
    base-metal sulfates."""
    s1 = stage1_lime_demand_kg_per_h(h2so4_kg_per_h, fe_kg_per_h, as_kg_per_h,
                                     p.lime_purity)
    s2 = stage2_lime_demand_kg_per_h(metals_sulfate_kg_per_h, p.lime_purity)
    total_lime = s1["lime_kg_per_h_stage1"] + s2["lime_kg_per_h_stage2"]
    total_gypsum = s1["gypsum_kg_per_h_stage1"] + s2["gypsum_kg_per_h_stage2"]
    return {
        **s1, **s2,
        "total_lime_kg_per_h": total_lime,
        "total_gypsum_kg_per_h": total_gypsum,
        "tank_volume_total_m3": total_tank_volume_m3(slurry_m3_per_h, p),
        "tank_volume_per_stage_m3": slurry_m3_per_h * p.residence_time_h,
        "num_stages_total": 2 * p.num_stages_each,
    }


if __name__ == "__main__":
    # Smoke test: 5000 m3/h leach effluent with 30 g/L acid, 1 g/L Fe, 0.3 g/L As,
    # plus 2 g/L Cu / Zn sulfate residuals.
    h2so4 = 30.0 * 5000.0   # 150,000 kg/h
    fe = 1.0 * 5000.0       # 5,000 kg/h Fe
    arsen = 0.3 * 5000.0    # 1,500 kg/h As
    base_metals = {"Cu": 2.0 * 5000.0, "Zn": 2.0 * 5000.0}  # kg/h MSO4

    p = NeutralizationParams()
    diag = size_neutralization(h2so4, fe, arsen, base_metals, 5000.0, p)
    print("Two-stage neutralization, 5000 m3/h leach effluent:")
    for k, v in diag.items():
        print(f"  {k:30s} {v}")
