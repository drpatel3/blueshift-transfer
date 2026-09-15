"""Hydrocyclone classifier model (Plitt 1976, updated by Samaeili et al. 2017).

All lengths in cm, flows in L/min, pressures in kPa, densities in g/cm3 — Plitt's
original units. Correction factors C1..C4 default to Samaeili's tuned values
(Table 3) and can be re-fit per plant.
"""
from dataclasses import dataclass
import math


@dataclass
class CycloneGeom:
    Dc: float    # body diameter, cm
    Di: float    # inlet diameter, cm
    Dx: float    # vortex finder diameter, cm
    Du: float    # apex (spigot) diameter, cm
    h: float     # free vortex height, cm


@dataclass
class PlittCoeffs:
    C1: float = 0.0387   # separation diameter
    C2: float = 51.27    # pressure drop
    C3: float = 0.1953   # flow fraction
    C4: float = 0.1762   # sharpness


def cut_size_plitt(g: CycloneGeom, Q: float, cv: float,
                   rho_p: float, rho_f: float, mu: float = 1.0,
                   C1: float = 0.0387) -> float:
    """Plitt Eq. (10): d50 in um. Q in L/min, cv = vol% solids, densities g/cm3."""
    num = 50.5 * (g.Dc ** 0.46) * (g.Di ** 0.6) * (g.Dx ** 1.21) * math.exp(0.063 * cv)
    # Plitt's original includes mu^0.5 in the numerator
    num *= math.sqrt(mu)
    denom = (g.Du ** 0.71) * (g.h ** 0.38) * (Q ** 0.45) * (rho_p - rho_f)
    return C1 * num / denom


def pressure_drop(g: CycloneGeom, Q: float, cv: float,
                  C2: float = 51.27) -> float:
    """Plitt Eq. (11): dP in kPa."""
    num = 1.88 * (Q ** 1.78) * math.exp(0.0055 * cv)
    denom = (g.Dc ** 0.37) * (g.Di ** 0.94) * (g.h ** 0.28) * ((g.Du ** 2 + g.Dx ** 2) ** 0.87)
    return C2 * num / denom


def flow_split_S(g: CycloneGeom, H: float, cv: float,
                 C3: float = 0.1953) -> float:
    """Plitt Eq. (13): S = Q_overflow / Q_underflow. H = pressure head, m of fluid."""
    num = 1.9 * ((g.Du / g.Dx) ** 3.31) * (g.h ** 0.54)
    num *= ((g.Du ** 2 + g.Dx ** 2) ** 0.36) * math.exp(0.0054 * cv)
    denom = (H ** 0.24) * (g.Dc ** 1.11)
    return C3 * num / denom


def water_recovery_Rv(S: float) -> float:
    """Eq. (12): fraction of feed water reporting to underflow."""
    return S / (S + 1.0)


def sharpness_m(g: CycloneGeom, Rv: float, Q: float,
                C4: float = 0.1762) -> float:
    """Plitt Eq. (14): slope of efficiency curve."""
    exponent = -1.58 * Rv * ((g.Dx ** 2 / Q) ** 0.15)
    return C4 * 1.94 * math.exp(exponent)


def grade_efficiency(x: float, d50: float, m: float) -> float:
    """Plitt Eq. (15): fraction of particles of size x reporting to underflow."""
    return 1.0 - math.exp(-0.693 * (x / d50) ** m)


def arterburn_capacity(Dc_cm: float, Pf_kpa: float) -> float:
    """Arterburn (1982) per Galvez 2014 Eq. (14): Qe in m3/h (Dc inches in original
    Arterburn; here assume cm and rescale). Use as a sanity check on sizing."""
    # Original: Qe = 0.000199 * Dc^1.87 * Pf^0.5 (Dc inches, Pf kPa, Qe m3/h)
    Dc_in = Dc_cm / 2.54
    return 0.000199 * (Dc_in ** 1.87) * math.sqrt(Pf_kpa)
