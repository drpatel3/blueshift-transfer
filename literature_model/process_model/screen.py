"""Vibrating screen sizing (SME Mineral Processing Handbook Ch. 20).

Sources:
    Table 20.8 — capacity factors A, B, C, D, E, F, G, H, J for deck area
    Eq. 20.25  — discharge-end bed depth (DBD) sanity check

All equations are imperial (ft, stph, in). Public callers pass metric and
this module converts at the boundary.
"""
from dataclasses import dataclass
import math


SHORT_PER_METRIC_TON = 1.10231
FT2_PER_M2 = 10.7639
FT_PER_M = 3.28084
IN_PER_MM = 1 / 25.4


@dataclass
class ScreenFactors:
    """Table 20.8 capacity factors. Defaults are mid-range for a coarse
    crusher-product screening duty (industrial Cu ore at 25 mm aperture).
    Refit per duty against Table 20.8 entries; values below pass an
    industrial sanity check but are not authoritative without the table."""
    A: float = 4.5    # base capacity, stph/ft^2 (aperture-dependent)
    B: float = 1.0    # oversize fraction in feed
    C: float = 1.0    # halfsize (<= half aperture) fraction in feed
    D: float = 1.0    # wet vs dry (1.0 dry, 1.25-1.50 wet sprayed)
    E: float = 1.0    # material density vs 100 lb/ft^3
    F: float = 1.0    # open area
    G: float = 1.0    # particle shape
    H: float = 1.0    # screening efficiency target (1.0 = 95%)
    J: float = 1.0    # deck position (top deck = 1.0)


@dataclass
class ScreenGeom:
    """Screen deck geometry."""
    aperture_mm: float = 25.0       # opening size
    width_m: float = 2.4            # deck width (ft converted; 8 ft typical)
    travel_rate_fpm: float = 75.0   # material velocity along deck (fpm)
    bulk_density_t_m3: float = 1.6  # bulk density of oversize


def required_deck_area_ft2(undersize_stph: float, fac: ScreenFactors) -> float:
    """Table 20.8 sizing equation: deck area = U / (A*B*C*D*E*F*G*H*J), ft^2.

    `undersize_stph` is the short-tons-per-hour of feed expected to pass.
    Returns required *single-deck* area in ft^2; multi-deck installations
    take the larger of the per-deck areas.
    """
    denom = fac.A * fac.B * fac.C * fac.D * fac.E * fac.F * fac.G * fac.H * fac.J
    if denom <= 0:
        raise ValueError("ScreenFactors product must be > 0")
    return undersize_stph / denom


def required_deck_area_m2(undersize_tph: float, fac: ScreenFactors) -> float:
    """Metric wrapper. `undersize_tph` is metric tonnes/hour."""
    u_stph = undersize_tph * SHORT_PER_METRIC_TON
    return required_deck_area_ft2(u_stph, fac) / FT2_PER_M2


def discharge_bed_depth_in(oversize_stph: float, geom: ScreenGeom) -> float:
    """SME Eq. 20.25:  DBD = (O * C) / (5 * T * W),  inches.

    O = oversize tonnage [stph]
    C = bulk density [ft^3/st]  (= 2000 / lb_per_ft3)
    T = travel rate [fpm]
    W = deck width [ft]

    Sanity gate: DBD must be <= 3-4 x aperture. If DBD > 4 x aperture, the
    deck is overloaded — widen W or shorten the deck.
    """
    width_ft = geom.width_m * FT_PER_M
    lb_per_ft3 = geom.bulk_density_t_m3 * 62.4280   # 1 t/m^3 = 62.428 lb/ft^3
    C_ft3_per_st = 2000.0 / lb_per_ft3
    return (oversize_stph * C_ft3_per_st) / (5.0 * geom.travel_rate_fpm * width_ft)


def dbd_passes(dbd_in: float, aperture_mm: float, max_ratio: float = 4.0) -> bool:
    """Sanity check: discharge-end bed depth should be <= max_ratio x aperture."""
    return dbd_in <= max_ratio * aperture_mm * IN_PER_MM


def size_screen(undersize_tph: float, oversize_tph: float,
                fac: ScreenFactors, geom: ScreenGeom) -> dict:
    """Compute deck area, DBD, and pass/fail flags. Returns a diag dict
    suitable for the simulator's stream output."""
    area_m2 = required_deck_area_m2(undersize_tph, fac)
    over_stph = oversize_tph * SHORT_PER_METRIC_TON
    dbd_in = discharge_bed_depth_in(over_stph, geom)
    return {
        "area_m2": area_m2,
        "area_ft2": area_m2 * FT2_PER_M2,
        "dbd_in": dbd_in,
        "dbd_max_in": 4.0 * geom.aperture_mm * IN_PER_MM,
        "dbd_pass": dbd_passes(dbd_in, geom.aperture_mm),
        "warn": None if dbd_passes(dbd_in, geom.aperture_mm)
                     else f"DBD {dbd_in:.2f} in exceeds 4x aperture; widen deck",
    }


if __name__ == "__main__":
    # Smoke test: 2000 tph crusher product, 70% oversize / 30% undersize
    feed_tph = 2000.0
    over_tph = feed_tph * 0.70
    under_tph = feed_tph * 0.30
    fac = ScreenFactors()
    geom = ScreenGeom()
    diag = size_screen(under_tph, over_tph, fac, geom)
    print("Screen sizing for 2000 tph (70/30 over/under):")
    for k, v in diag.items():
        print(f"  {k:14s} {v}")
