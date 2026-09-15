"""Thickener model (Galvez et al. 2014, Min. Eng.).

Richardson-Zaki hindered settling, King (2001) steady-state ideal thickener,
Parkinson-Mular cost correlation. Units: m, m3/h, t/h, t/m3.
"""
from dataclasses import dataclass
import math


@dataclass
class ThickenerParams:
    vTF: float        # terminal settling velocity, m/h (Richardson-Zaki)
    rF: float         # Richardson-Zaki volume-fraction coeff
    n: float          # Richardson-Zaki exponent
    rho_s: float = 2.70   # solids density, t/m3


def settling_flux(C_M: float, p: ThickenerParams) -> float:
    """Richardson-Zaki (Eq. 24): solids flux psi_M at concentration C_M."""
    return p.vTF * C_M * (1.0 - p.rF * C_M) ** p.n


def max_solids_flux(p: ThickenerParams) -> float:
    """Galvez Eq. (27): maximum solids flux the thickener can handle."""
    n = p.n
    return (4.0 * p.vTF / p.rF) * n * (n - 1.0) ** (n - 1.0) / (n + 1.0) ** (n + 1.0)


def required_area(W_solids_tph: float, p: ThickenerParams, safety: float = 1.25) -> float:
    """Minimum thickener area (m2) so feed flux < max. W_solids_tph = t/h feed."""
    f_max = max_solids_flux(p)  # m/h * t/m3 effectively
    return safety * W_solids_tph / (p.rho_s * f_max)


def underflow_concentration(C_D: float, f_e: float, psi_M: float) -> float:
    """King (2001) Eq. (22): C_M at the bottom sediment."""
    return C_D * (f_e - psi_M) / f_e


def diameter_from_area(area_m2: float) -> float:
    return math.sqrt(4.0 * area_m2 / math.pi)
