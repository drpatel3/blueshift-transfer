"""Gyratory crusher model (Pothina et al. 2007, Min. & Met. Proc. 24(3)).

Bond's third theory for energy, plus the Pothina amperage-constant scaling that
links blast fragmentation (burden B, spacing S, CSS) to crusher power draw.
"""
from dataclasses import dataclass
import math


@dataclass
class CrusherParams:
    Wi: float          # Bond work index, kWh/t
    V: float = 4160.0  # line voltage, V
    PF: float = 0.85   # power factor
    LF: float = 1.0    # lump factor (material coarseness)
    DE: float = 0.95   # drive efficiency
    A_idle: float = 30.0  # idle amperage draw, A


def bond_energy(Wi: float, F80: float, P80: float) -> float:
    """Bond (1952): specific energy in kWh/t. F80, P80 in microns."""
    return Wi * 10.0 * (1.0 / math.sqrt(P80) - 1.0 / math.sqrt(F80))


def amperage_constant(a: float) -> float:
    """Pothina Eq. (24): A_srmt = -22.24 + 0.3348 * a.
    `a` is the Lynch breakage factor derived from B, S, CSS and Kuz-Ram."""
    return -22.24 + 0.3348 * a


def idle_power(p: CrusherParams) -> float:
    """Eq. (12): P0 = V * A_idle * PF, watts."""
    return p.V * p.A_idle * p.PF


def crushing_power(p: CrusherParams, E_mt_kwh_per_t: float, rate_tph: float) -> float:
    """Eq. (11): P_srhr = (E_mt * LF / DE) * RC. Returns kW."""
    return (E_mt_kwh_per_t * p.LF / p.DE) * rate_tph


def total_power(p: CrusherParams, F80: float, P80: float, rate_tph: float) -> float:
    """Total hourly power draw, kW (Eq. 13)."""
    E = bond_energy(p.Wi, F80, P80)
    return idle_power(p) / 1000.0 + crushing_power(p, E, rate_tph)


def annual_cost(C_bmt: float, C_cmt: float, tpy: float) -> float:
    """Eq. (14) objective: C_tyear = (C_bmt + C_cmt) * tpy."""
    return (C_bmt + C_cmt) * tpy
