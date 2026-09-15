"""Atmospheric / pressure tank leach model — shrinking-core kinetics.

Sources:
    Sohn & Wadsworth (1979) "Rate Processes of Extractive Metallurgy",
    shrinking-core fluid-solid reaction model.
    Implementation pedigree: ported from the prior optimization_model
    (`optimization_model/internal_dev/main/physical_parameters/leaching.py:14-84`).
    Stoichiometry from SME Mineral Processing Handbook Ch. 20: H2SO4/Cu = 6.17 kg/kg.

Scope: stirred-tank atmospheric or autoclave leach on chalcopyrite /
chalcocite Cu sulfides with O2 oxidant. Heap-leach column kinetics
(Schlitt 1992, irrigation rate, percolation depth) are NOT in this
module and remain a deferred build.

Equation (shrinking-core, ash-diffusion-controlled with first-order O2):

    dX_Cu/dt = 3 * k0 * exp(-Ea / (R * T)) * P_O2^n * (1 - X_Cu)^(2/3)

where X_Cu = Cu extraction fraction (0..1), T in K, P_O2 in bar.
Integrated to a residence time t_h gives X_Cu(t_h).
"""
from dataclasses import dataclass
import math


R_GAS = 8.314                 # J/(mol*K)
H2SO4_PER_CU_KG_PER_KG = 6.17 # SME Ch. 20: 6.17 kg H2SO4 per kg Cu leached

# Refractory-Cu cap on leach extraction. Real Cu sulfide ore always contains a
# small fraction of Cu locked in accessory phases (silicates, iron oxides,
# composite particles) that no realistic residence time will liberate. The
# shrinking-core integrated form `1 - (1-kt)^3` mathematically approaches 1.0
# as kt grows, but industrial autoclaves do not.
#
# Industry benchmarks at full residence time:
#   - POX chalcopyrite, 200-230 C, 12 bar O2 (Phelps Dodge El Abra,
#     Sherritt Bagdad pilot): 92-97% Cu extraction.
#   - Albion atmospheric oxidative leach (Glencore McArthur River
#     equivalent): 95-98% on fine-ground concentrate.
#   - Atmospheric chalcocite tank leach: 85-95%.
# 0.95 sits at the middle of the POX band and is conservative for Albion.
# Adjust per ore mineralogy via LeachParams.max_extraction when modelling a
# specific deposit.
MAX_LEACH_EXTRACTION = 0.95


@dataclass
class LeachParams:
    """Shrinking-core kinetics + tank sizing inputs.

    Defaults are autoclave chalcopyrite (200 C, 12 bar O2, 4 h, Ea=70 kJ/mol)
    which gives ~99% Cu extraction. For atmospheric tank chalcocite, set
    temperature_C ~ 80, P_O2_bar = 1.0, Ea_kj_per_mol ~ 35, residence_time_h
    ~ 24 (chalcocite oxidizes in air, no autoclave required).

    k0 is in 1/s (matching optimization_model's per-second integration); the
    closed-form below converts residence_time_h to seconds.
    """
    k0: float = 500.0              # pre-exponential, 1/s (optimization_model default)
    Ea_kj_per_mol: float = 70.0    # activation energy, chalcopyrite low end
    n_O2: float = 0.7              # O2 partial-pressure exponent (0.5-1.0)
    P_O2_bar: float = 12.0         # autoclave default; 1.0 for atmospheric
    temperature_C: float = 200.0   # autoclave default
    residence_time_h: float = 4.0  # per stage
    num_stages: int = 1            # stirred-tank stages (CSTR-in-series)
    max_extraction: float = MAX_LEACH_EXTRACTION  # refractory-Cu cap


def kinetic_constant(p: LeachParams) -> float:
    """Effective rate constant k(T) = k0 * exp(-Ea / RT) * P_O2^n, in 1/s."""
    T_K = p.temperature_C + 273.15
    return p.k0 * math.exp(-p.Ea_kj_per_mol * 1000.0 / (R_GAS * T_K)) * p.P_O2_bar ** p.n_O2


def cu_extraction(p: LeachParams) -> float:
    """Integrate dX/dt = 3*k*(1-X)^(2/3) from X=0 over residence time.

    Closed-form: X(t) = 1 - (1 - k * t)^3, t in seconds.
    Multi-stage approximated as plug-flow (total volume = num_stages * V_per).
    Result is capped at p.max_extraction to reflect the refractory-Cu floor
    that no industrial leach can break through (see MAX_LEACH_EXTRACTION
    docstring above).
    """
    k_per_s = kinetic_constant(p)
    t_total_s = p.residence_time_h * p.num_stages * 3600.0
    arg = 1.0 - k_per_s * t_total_s
    x_kinetic = 1.0 if arg <= 0.0 else 1.0 - arg ** 3
    return min(p.max_extraction, x_kinetic)


def acid_consumption_kg_per_h(cu_leached_tph: float) -> float:
    """SME Ch. 20 stoichiometry: 6.17 kg H2SO4 per kg Cu leached."""
    return cu_leached_tph * 1000.0 * H2SO4_PER_CU_KG_PER_KG


def total_tank_volume_m3(slurry_m3_per_h: float, p: LeachParams) -> float:
    """Tank volume = slurry flow * residence time per stage * num_stages.

    Capex sizes off this number (atmospheric-tank power-law in tea.py).
    """
    return slurry_m3_per_h * p.residence_time_h * p.num_stages


def per_stage_volume_m3(slurry_m3_per_h: float, p: LeachParams) -> float:
    return slurry_m3_per_h * p.residence_time_h


def size_leach(cu_in_tph: float, slurry_m3_per_h: float,
               p: LeachParams) -> dict:
    """Compute Cu extraction, acid demand, and tank volumes."""
    X_Cu = cu_extraction(p)
    cu_leached_tph = cu_in_tph * X_Cu
    acid_kg_per_h = acid_consumption_kg_per_h(cu_leached_tph)
    return {
        "X_Cu": X_Cu,
        "cu_leached_tph": cu_leached_tph,
        "cu_residue_tph": cu_in_tph - cu_leached_tph,
        "acid_kg_per_h": acid_kg_per_h,
        "tank_volume_total_m3": total_tank_volume_m3(slurry_m3_per_h, p),
        "tank_volume_per_stage_m3": per_stage_volume_m3(slurry_m3_per_h, p),
        "num_stages": p.num_stages,
        "k_eff": kinetic_constant(p),
    }


if __name__ == "__main__":
    # Smoke test: 16 tph Cu, 5000 m3/h slurry, autoclave 200 C / 12 bar O2
    p = LeachParams()
    diag = size_leach(cu_in_tph=16.0, slurry_m3_per_h=5000.0, p=p)
    print("Atmospheric/autoclave leach (autoclave defaults):")
    for k, v in diag.items():
        print(f"  {k:25s} {v}")

    # Atmospheric chalcocite: 80 C, 1 bar O2, 24 h, Ea=35
    p_atm = LeachParams(temperature_C=80.0, P_O2_bar=1.0, Ea_kj_per_mol=35.0,
                        residence_time_h=24.0)
    diag2 = size_leach(cu_in_tph=16.0, slurry_m3_per_h=5000.0, p=p_atm)
    print("\nAtmospheric chalcocite (80 C, 1 bar O2, 24 h, Ea=35):")
    for k, v in diag2.items():
        print(f"  {k:25s} {v}")
