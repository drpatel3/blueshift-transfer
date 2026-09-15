"""Steady-state chain of unit operations.

Connects crusher -> SAG -> hydrocyclone -> thickener -> SX by mass balance on
solids, water, and (for SX) Cu concentration. Intentionally simple: each stage
is a pure function that takes an inlet Stream and returns outlet Stream(s).
"""
from dataclasses import dataclass, replace
from typing import Optional
import math

from . import (crusher, sag, hydrocyclone as hc, thickener as th, sx,
               flotation as fl, filter as flt)


@dataclass
class Stream:
    solids_tph: float           # t/h of dry solids
    water_tph: float            # t/h of water
    F80_um: Optional[float] = None   # 80%-passing size, microns
    cu_grade: float = 0.0       # mass fraction Cu in solids
    cu_aq_gpl: float = 0.0      # dissolved Cu in aqueous, g/L
    organic_tph: float = 0.0    # t/h of organic phase (SX)
    cu_org_gpl: float = 0.0     # Cu in organic, g/L

    @property
    def solids_pct(self) -> float:
        total = self.solids_tph + self.water_tph
        return 100.0 * self.solids_tph / total if total else 0.0

    @property
    def cv(self) -> float:
        """volume % solids, approx (assumes rho_s = 2.7, rho_w = 1.0)."""
        vs = self.solids_tph / 2.7
        vw = self.water_tph / 1.0
        total = vs + vw
        return 100.0 * vs / total if total else 0.0
    


def run_crusher(feed: Stream, params: crusher.CrusherParams,
                P80_um: float) -> tuple[Stream, float]:
    """Returns (product stream, power draw kW). Mass is conserved."""
    kw = crusher.total_power(params, feed.F80_um, P80_um, feed.solids_tph)
    out = replace(feed, F80_um=P80_um)
    return out, kw


def run_sag(feed: Stream, params: sag.SAGParams,
            Ecs_kwh_per_t: float) -> tuple[Stream, float]:
    """SAG+BM grinding. Bond third-theory inversion (SME EQ 20.10):
    W = 10*Wi*(1/sqrt(P80) - 1/sqrt(F80))  ->  P80 = [10/(W/Wi + 10/sqrt(F80))]^2.
    t10/A/b retained in sag.py for future population-balance use."""
    inv_sqrt_P80 = Ecs_kwh_per_t / (10.0 * params.Wi) + 1.0 / math.sqrt(feed.F80_um)
    P80 = (1.0 / inv_sqrt_P80) ** 2
    kw = Ecs_kwh_per_t * feed.solids_tph
    return replace(feed, F80_um=P80), kw


def run_hydrocyclone(feed: Stream, geom: hc.CycloneGeom,
                     pressure_kpa: float,
                     coeffs: hc.PlittCoeffs = hc.PlittCoeffs(),
                     rho_p: float = 2.70, rho_f: float = 1.00,
                     mu: float = 1.0) -> tuple[Stream, Stream, dict]:
    """Returns (underflow, overflow, diagnostics). Splits solids by grade
    efficiency evaluated at F80 as a proxy for the whole distribution."""
    Q_lpm = (feed.solids_tph / rho_p + feed.water_tph) * 1000.0 / 60.0
    cv = feed.cv
    # Approximate head from pressure: H (m) = P (kPa) / 9.81
    H = pressure_kpa / 9.81
    d50 = hc.cut_size_plitt(geom, Q_lpm, cv, rho_p, rho_f, mu, coeffs.C1)
    dp = hc.pressure_drop(geom, Q_lpm, cv, coeffs.C2)
    S = hc.flow_split_S(geom, H, cv, coeffs.C3)
    Rv = hc.water_recovery_Rv(S)
    m = hc.sharpness_m(geom, Rv, Q_lpm, coeffs.C4)
    y = hc.grade_efficiency(feed.F80_um, d50, m) if feed.F80_um else 0.5

    uf = Stream(solids_tph=feed.solids_tph * y,
                water_tph=feed.water_tph * Rv,
                F80_um=feed.F80_um, cu_grade=feed.cu_grade)
    of = Stream(solids_tph=feed.solids_tph * (1 - y),
                water_tph=feed.water_tph * (1 - Rv),
                F80_um=feed.F80_um, cu_grade=feed.cu_grade)
    diag = {"d50_um": d50, "dp_kpa": dp, "S": S, "Rv": Rv, "m": m,
            "solids_to_uf": y}
    return uf, of, diag


def run_thickener(feed: Stream, params: th.ThickenerParams,
                  target_uf_pct: float = 60.0) -> tuple[Stream, Stream, dict]:
    """Galvez thickener: all solids to underflow, water split to hit target %."""
    area = th.required_area(feed.solids_tph, params)
    diameter = th.diameter_from_area(area)
    # Water in underflow from target % solids
    w_uf = feed.solids_tph * (100.0 / target_uf_pct - 1.0)
    w_uf = min(w_uf, feed.water_tph)  # clamp
    uf = Stream(solids_tph=feed.solids_tph, water_tph=w_uf,
                F80_um=feed.F80_um, cu_grade=feed.cu_grade)
    of = Stream(solids_tph=0.0, water_tph=feed.water_tph - w_uf)
    diag = {"area_m2": area, "diameter_m": diameter}
    return uf, of, diag


def run_sx(pls: Stream, barren_organic: Stream,
           mp: sx.MixerParams, iso: sx.IsothermParams,
           pH: float = 1.8, cu_bo: float = 0.5) -> tuple[Stream, Stream]:
    """One extraction stage at steady state. Returns (raffinate, loaded_organic).
    PLS aqueous Cu is pls.cu_aq_gpl; BO organic Cu is barren_organic.cu_aq_gpl."""
    X_i = pls.cu_aq_gpl
    Y_i = barren_organic.cu_aq_gpl

    def Y_star_fn(X):
        return extraction_isotherm_safe(X, iso, pH, X_i, cu_bo)

    def X_star_fn(Y):
        # invert isotherm: X* = B*Y / (A - Y)
        A = iso.a * iso.ML
        B = ((10.0 ** -pH) ** iso.b / iso.ML ** iso.c) * (iso.d * X_i + iso.f * cu_bo)
        return B * Y / max(A - Y, 1e-6)

    Q_Ai = (pls.solids_tph / 2.7 + pls.water_tph)  # m3/h
    Q_Oi = barren_organic.water_tph  # treat organic as non-solid stream, m3/h

    X_m, Y_m = sx.steady_state_mixer(X_i, Y_i, X_star_fn, Y_star_fn,
                                     Q_Ai, Q_Oi, mp)
    raffinate = replace(pls, cu_aq_gpl=X_m)
    loaded = replace(barren_organic, cu_aq_gpl=Y_m)
    return raffinate, loaded


def run_flotation(feed: Stream,
                  fp: fl.FittingParams, cp: fl.CellParams, op: fl.OperatingParams,
                  gangue_recovery: float = 0.03,
                  conc_solids_pct: float = 35.0,
                  frother_type: int = 2,
                  scavenger_recovery: float = 0.0,
                  recovery_cap: float = 1.0) -> tuple[Stream, Stream, dict]:
    """Rougher/cleaner flotation. Cu recovery from flotation.recovery(),
    gangue recovery supplied (default 3% entrainment). Concentrate grade
    falls out of the mass balance. Water split by target conc % solids.

    scavenger_recovery: optional scavenger-bank Cu recovery on rougher tails
    (parallel-failure combined: R_total = R + (1-R)*R_scav). Industry Cu
    chalcopyrite scavengers run ~0.40-0.55 on rougher tails. Default 0
    (no scavenger).

    recovery_cap: hard upper bound on combined R_Cu (default 1.0 = no cap).
    Industry rougher+scavenger banks deliver up to ~0.90 on chalcopyrite.
    """
    res = fl.recovery(fp, cp, op, frother_type=frother_type)
    R_cu = res["recovery"]
    if scavenger_recovery > 0.0:
        R_cu = R_cu + (1.0 - R_cu) * scavenger_recovery
    R_cu = min(R_cu, recovery_cap)
    cu_mass = feed.solids_tph * feed.cu_grade
    gangue_mass = feed.solids_tph * (1.0 - feed.cu_grade)
    cu_to_conc = cu_mass * R_cu
    gangue_to_conc = gangue_mass * gangue_recovery
    conc_solids = cu_to_conc + gangue_to_conc
    tails_solids = feed.solids_tph - conc_solids
    conc_grade = cu_to_conc / conc_solids if conc_solids > 0 else 0.0
    tails_grade = (cu_mass - cu_to_conc) / tails_solids if tails_solids > 0 else 0.0
    w_conc = conc_solids * (100.0 / conc_solids_pct - 1.0)
    w_conc = min(w_conc, feed.water_tph)
    conc = Stream(solids_tph=conc_solids, water_tph=w_conc,
                  F80_um=feed.F80_um, cu_grade=conc_grade)
    tails = Stream(solids_tph=tails_solids, water_tph=feed.water_tph - w_conc,
                   F80_um=feed.F80_um, cu_grade=tails_grade)
    diag = {**res, "R_cu": R_cu, "conc_grade": conc_grade,
            "tails_grade": tails_grade, "upgrade_ratio": conc_grade / feed.cu_grade
            if feed.cu_grade > 0 else 0.0}
    return conc, tails, diag


def run_filter(feed: Stream, params: flt.BeltFilterParams,
               target_moisture_pct: float) -> tuple[Stream, Stream, dict]:
    """Horizontal vacuum belt filter (Perry Ch. 18).

    Sizes belt area via Ruth constant-pressure filtration (Eq. 18-71);
    checks cake thickness against Table 18-8 minimum (3 mm for belt).
    Cake moisture is a user-supplied SPEC — Perry provides no residual-
    moisture kinetics equation, so this is NOT computed from dewatering time.
    """
    w = flt.solids_per_filtrate_volume(feed.solids_tph, feed.water_tph)
    V_per_A = flt.filtrate_per_area(params.t_form_s, w, params)
    W_per_A = flt.cake_loading(V_per_A, w)
    thickness_mm = flt.cake_thickness_mm(W_per_A, params.rho_cake_dry_kg_m3)
    area_m2 = flt.required_area(feed.solids_tph, W_per_A, params.t_cycle_s)

    warn = None
    if thickness_mm < params.min_thickness_mm:
        warn = (f"cake thickness {thickness_mm:.2f} mm < "
                f"{params.min_thickness_mm} mm (Perry Table 18-8)")

    # Water balance set by the supplied moisture spec, clamped to available water.
    m = target_moisture_pct / 100.0
    water_cake = feed.solids_tph * m / (1.0 - m)
    water_cake = min(water_cake, feed.water_tph)
    cake = Stream(solids_tph=feed.solids_tph, water_tph=water_cake,
                  F80_um=feed.F80_um, cu_grade=feed.cu_grade)
    filtrate = Stream(solids_tph=0.0, water_tph=feed.water_tph - water_cake)
    diag = {"area_m2": area_m2, "thickness_mm": thickness_mm,
            "W_per_A_kg_m2": W_per_A, "V_per_A_m3_m2": V_per_A,
            "w_kg_m3": w, "target_moisture_pct": target_moisture_pct,
            "warn": warn}
    return cake, filtrate, diag


def extraction_isotherm_safe(X, iso, pH, cu_pls, cu_bo):
    X = max(X, 0.0)
    return sx.extraction_isotherm(X, iso, pH, cu_pls, cu_bo)
