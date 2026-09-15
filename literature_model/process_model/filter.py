"""Horizontal vacuum belt filter (Perry Ch. 18, 7th ed. 1999).

All equations sourced from `references/references.md` — Ruth constant-pressure
filtration (Eq. 18-71), compressibility (Eq. 18-74), minimum cake thickness
(Table 18-8). Residual cake moisture is NOT computed here — Perry provides no
kinetics equation for it; callers must supply it as a process spec.
"""
from dataclasses import dataclass
import math


@dataclass
class BeltFilterParams:
    """Belt vacuum filter parameters.

    alpha, r come from a leaf-test fit per Perry Fig. 18-107a
    (slope = μαw/(2P), intercept = μr/P). Defaults below are placeholder
    order-of-magnitude values for Cu sulfide concentrate — replace with
    leaf-test values before relying on absolute sizing.
    """
    alpha: float = 1.0e11           # specific cake resistance at operating dP, m/kg
    r: float = 1.0e10               # medium resistance, 1/m
    dP_pa: float = 70_000.0         # vacuum ΔP, Pa (≈0.7 atm)
    t_form_s: float = 60.0          # cake-formation residence on belt, s
    t_cycle_s: float = 200.0        # total cycle time (form + dewater + discharge), s
    rho_cake_dry_kg_m3: float = 2000.0   # dry cake bulk density (for thickness)
    mu_pas: float = 1.0e-3          # filtrate viscosity (water), Pa·s
    min_thickness_mm: float = 3.0   # Perry Table 18-8, horizontal belt


def alpha_from_compressibility(alpha_prime: float, dP_pa: float, s: float) -> float:
    """Perry Eq. 18-74: α = α' · P^s. s ∈ [0.1, 0.8] for most industrial slurries."""
    return alpha_prime * (dP_pa ** s)


def solids_per_filtrate_volume(solids_tph: float, water_tph: float) -> float:
    """w in Perry Eq. 18-71: mass dry cake per unit filtrate volume, kg/m³.

    Approximation: treats feed water tph as filtrate volume m³/h (ρ_w = 1 t/m³).
    Small amount of water retained in cake is ignored for sizing.
    """
    if water_tph <= 0:
        return 0.0
    return (solids_tph * 1000.0) / water_tph


def filtrate_per_area(t_s: float, w_kg_m3: float, p: BeltFilterParams) -> float:
    """Perry Eq. 18-71 solved for V/A at time t (constant-pressure).

    θ/(V/A) = [μαw / (2·dP)]·(V/A) + μr/dP
    → a·x² + b·x − t = 0, with x = V/A.
    """
    a = p.mu_pas * p.alpha * w_kg_m3 / (2.0 * p.dP_pa)
    b = p.mu_pas * p.r / p.dP_pa
    if a <= 0:
        return 0.0
    return (-b + math.sqrt(b * b + 4.0 * a * t_s)) / (2.0 * a)


def cake_loading(V_per_A: float, w_kg_m3: float) -> float:
    """Dry cake mass per unit area, kg/m². W/A = w · V/A."""
    return w_kg_m3 * V_per_A


def cake_thickness_mm(W_per_A: float, rho_cake_dry_kg_m3: float) -> float:
    """Cake thickness in mm from areal loading and bulk density."""
    return 1000.0 * W_per_A / rho_cake_dry_kg_m3


def required_area(solids_tph: float, W_per_A: float, t_cycle_s: float) -> float:
    """Belt area (m²) from throughput and per-cycle cake loading.

    Dry solids rate per unit belt area = (W/A) / t_cycle.
    """
    if W_per_A <= 0 or t_cycle_s <= 0:
        return float("inf")
    solids_kg_s = solids_tph * 1000.0 / 3600.0
    rate_per_area = W_per_A / t_cycle_s
    return solids_kg_s / rate_per_area
