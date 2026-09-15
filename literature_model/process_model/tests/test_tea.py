"""Unit tests for process_model.tea — per-stage Perry/Mular capex.

Uses a fabricated minimal sim_result dict so tests don't depend on the full
throughput simulator. Checks arithmetic within tight tolerance against
hand-computed values using the BlueShift Excel's capex/opex conventions.
"""
from dataclasses import dataclass
import math

from process_model.tea import (
    TEAParams, evaluate, _irr, perry_cost, select_basis,
    tailings_storage_capex,
    CRUSHER_BASES, MOTOR_BASES, AGITATED_TANK_BASIS, ATM_TANK_BASIS,
    EW_CELL_BASIS, FLOT_FROTH_VOLUME_MULTIPLIER, FLOTATION_CELL_BASIS,
    select_flot_basis,
    SAG_MILL_BASIS, BM_MILL_BASIS, HYDROCYCLONE_BASIS,
    THICKENER_BASIS, BELT_FILTER_BASIS,
    TAILINGS_MIN_USD_1988, TAILINGS_EXP,
    SHORT_PER_METRIC_TON,
    M_S_INDEX_PERRYS, M_S_INDEX_2025, M_S_INDEX_1988,
    smelter_as_penalty_per_t_cu, DEFAULT_CU_PRICE_PER_T,
    AS_PENALTY_FREE_PCT, AS_PENALTY_REJECT_PCT, AS_PENALTY_AT_REJECT_USD,
)
from process_model.topology import Topology


# Minimal Stream stand-in — TEA reads solids_tph, water_tph, cu_grade
@dataclass
class _S:
    solids_tph: float
    water_tph: float = 0.0
    cu_grade: float = 0.0


def _toy_sim(conc_tph=75.0, conc_grade=0.20, ore_tph=2000.0,
             crusher_kw=300.0, sag_kw=5000.0, bm_kw=15000.0, flot_kw=900.0,
             thickener_dia=20.0, filter_area=40.0,
             flot_volume_m3=1500.0,
             bm_disch_solids=7000.0, bm_disch_water=5700.0,
             cyc_of_water_tph=5000.0,
             topology=None):
    """Build a minimal sim_result dict mirroring simulate_cu_sulfide output."""
    return {
        "streams": {
            "ore": _S(ore_tph, 0.0, 0.008),
            "filter_cake": _S(conc_tph, 0.0, conc_grade),
            "cyc_of": _S(2000.0, cyc_of_water_tph, 0.008),
            "bm_discharge": _S(bm_disch_solids, bm_disch_water, 0.008),
        },
        "power_kw": {
            "crusher_kw": crusher_kw,
            "sag_kw": sag_kw,
            "bm_kw": bm_kw,
            "flotation_kw": flot_kw,
        },
        "conc_thickener": {"diameter_m": thickener_dia, "area_m2": 314.0},
        "belt_filter": {"area_m2": filter_area},
        "flotation": {"total_volume_m3": flot_volume_m3,
                      "num_cells": 6, "cell_volume_m3": 250.0},
        "topology": topology if topology is not None else Topology(),
    }


# ---------------------------------------------------------------- constants
def test_inflation_factor_matches_sheet():
    assert math.isclose(M_S_INDEX_2025 / M_S_INDEX_PERRYS, 2.383, rel_tol=1e-4)


# ---------------------------------------------------------------- perry_cost
def test_perry_cost_formula():
    basis_usd, basis_cap, exp = ATM_TANK_BASIS
    c = perry_cost(basis_usd, basis_cap, exp, actual_cap=1.0, inflation=1.0)
    expected = basis_usd * (1.0 / basis_cap) ** exp
    assert math.isclose(c, expected, rel_tol=1e-9)


def test_perry_cost_applies_inflation():
    c_inflated = perry_cost(9300.0, 0.38, 0.53, 1.0, inflation=2.383)
    c_raw = perry_cost(9300.0, 0.38, 0.53, 1.0, inflation=1.0)
    assert math.isclose(c_inflated / c_raw, 2.383, rel_tol=1e-9)


def test_perry_cost_zero_capacity():
    assert perry_cost(9300.0, 0.38, 0.53, 0.0) == 0.0


# ---------------------------------------------------------------- basis selection
def test_select_basis_small_capacity_uses_small_tier():
    chosen = select_basis(CRUSHER_BASES, 5.0)
    assert chosen[1] == 7.5


def test_select_basis_large_capacity_uses_large_tier():
    chosen = select_basis(CRUSHER_BASES, 300.0)
    assert chosen[1] == 74.6


# ---------------------------------------------------------------- revenue
def test_revenue_arithmetic():
    """Concentrate at 30% Cu — clears the smelter full-payable threshold,
    so the standard payable arithmetic applies untouched."""
    tea = TEAParams()
    sim = _toy_sim(conc_tph=75.0, conc_grade=0.30)
    r = evaluate(sim, tea)

    # payable = (22.5 - 0.75) * 0.965 = 20.98875 tph
    assert math.isclose(r["breakdown"]["payable_cu_tph"], 20.98875, rel_tol=1e-4)
    expected_gross = 20.98875 * 12500.0 * 7884.0
    assert math.isclose(r["breakdown"]["gross_revenue"], expected_gross, rel_tol=1e-6)
    assert math.isclose(r["breakdown"]["tc_cost"], 75 * 22 * 7884, rel_tol=1e-6)
    assert math.isclose(r["breakdown"]["rc_cost"], 20.98875 * 50 * 7884, rel_tol=1e-6)


def test_as_penalty_schedule_brackets():
    """The general smelter As schedule: clean below the penalty-free limit,
    a linear ramp to $500/t Cu at the rejection limit, and a Cu-price
    (unsaleable) penalty at/above the rejection limit."""
    # Clean concentrate — no penalty.
    assert smelter_as_penalty_per_t_cu(AS_PENALTY_FREE_PCT) == 0.0
    assert smelter_as_penalty_per_t_cu(0.05) == 0.0
    # Mid-bracket — half-way between free (0.20%) and reject (0.50%) -> half of
    # the $500/t Cu top-of-ramp deduction.
    mid = (AS_PENALTY_FREE_PCT + AS_PENALTY_REJECT_PCT) / 2.0
    assert math.isclose(smelter_as_penalty_per_t_cu(mid),
                        AS_PENALTY_AT_REJECT_USD * 0.5, rel_tol=1e-9)
    # At/above the rejection limit — concentrate is unsaleable.
    assert smelter_as_penalty_per_t_cu(AS_PENALTY_REJECT_PCT) == DEFAULT_CU_PRICE_PER_T
    assert smelter_as_penalty_per_t_cu(0.80) == DEFAULT_CU_PRICE_PER_T


def test_as_rich_concentrate_pays_more_than_clean_at_same_grade():
    """Sulfide concentrate revenue must drop as the concentrate As grade rises.
    A penalty-free clean concentrate (As <= 0.20%) dodges the deduction; a
    mid-bracket As grade pays a proportional penalty."""
    base = _toy_sim(conc_tph=75.0, conc_grade=0.30)
    sim_clean = {**base, "conc_arsenic_pct": 0.05}
    sim_dirty = {**base, "conc_arsenic_pct": 0.35}

    r_clean = evaluate(sim_clean, TEAParams())
    r_dirty = evaluate(sim_dirty, TEAParams())

    assert r_clean["breakdown"]["impurity_cost"] == 0.0
    assert r_dirty["breakdown"]["impurity_cost"] > 0.0
    assert (r_dirty["breakdown"]["net_concentrate_revenue"]
            < r_clean["breakdown"]["net_concentrate_revenue"])
    # Magnitude check: 0.35% As -> half-ramp -> $250/t Cu * payable * hours.
    expected_dirty = 20.98875 * smelter_as_penalty_per_t_cu(0.35) * 7884.0
    assert math.isclose(r_dirty["breakdown"]["impurity_cost"], expected_dirty,
                        rel_tol=1e-6)


def test_explicit_impurity_penalty_overrides_as_schedule():
    """Caller-set TEAParams.impurity_penalty_per_t_cu must bypass the As
    schedule — lets contract scenarios force a specific penalty."""
    sim = {**_toy_sim(conc_tph=75.0, conc_grade=0.30),
           "conc_arsenic_pct": 0.55}   # would otherwise reject
    r = evaluate(sim, TEAParams(impurity_penalty_per_t_cu=42.0))
    expected = 20.98875 * 42.0 * 7884.0
    assert math.isclose(r["breakdown"]["impurity_cost"], expected,
                        rel_tol=1e-6)


def test_byproduct_credits_paid_only_on_sulfide_route():
    """Mo + Au + Ag credit the sulfide concentrate route only. Hydromet
    routes (leach or heap_leach on) and zero-concentrate cases get $0."""
    sim_sulfide = {**_toy_sim(conc_tph=75.0, conc_grade=0.30),
                   "mineralogy": "chalcopyrite",
                   "byproducts": {"mo_ppm": 250.0,
                                  "au_g_per_t": 0.10,
                                  "ag_g_per_t": 2.0}}
    r_sulfide = evaluate(sim_sulfide, TEAParams())
    assert r_sulfide["breakdown"]["byproduct_revenue"] > 0.0
    assert r_sulfide["breakdown"]["mo_revenue"] > 0.0

    # Hydromet route — leach_enabled forces concentrate revenue to zero
    # and byproduct revenue to zero (Cu reports as cathode only).
    leach_topo = Topology(leach_enabled=True, sx_enabled=True,
                          ew_enabled=True, regrind_enabled=True,
                          cleaner_enabled=True, filter_enabled=False)
    sim_hydromet = {**_toy_sim(conc_tph=75.0, conc_grade=0.30,
                               topology=leach_topo),
                    "mineralogy": "chalcopyrite",
                    "byproducts": {"mo_ppm": 250.0, "au_g_per_t": 0.10,
                                   "ag_g_per_t": 2.0}}
    r_hydro = evaluate(sim_hydromet, TEAParams())
    assert r_hydro["breakdown"]["byproduct_revenue"] == 0.0


def test_byproduct_revenue_arithmetic():
    """Mo/Au/Ag revenue scales linearly with ore tph * head grade *
    recovery * price."""
    tea = TEAParams()
    sim = {**_toy_sim(conc_tph=75.0, conc_grade=0.30, ore_tph=2000.0),
           "mineralogy": "chalcopyrite",
           "byproducts": {"mo_ppm": 250.0, "au_g_per_t": 0.10,
                          "ag_g_per_t": 2.0}}
    r = evaluate(sim, tea)
    expected_mo = 2000.0 * 250.0e-6 * tea.mo_recovery * tea.mo_price_per_t * 7884
    expected_au = 2000.0 * 0.10e-6 * tea.au_payable * tea.au_price_per_t * 7884
    expected_ag = 2000.0 * 2.0e-6 * tea.ag_payable * tea.ag_price_per_t * 7884
    assert math.isclose(r["breakdown"]["mo_revenue"], expected_mo, rel_tol=1e-6)
    assert math.isclose(r["breakdown"]["au_revenue"], expected_au, rel_tol=1e-6)
    assert math.isclose(r["breakdown"]["ag_revenue"], expected_ag, rel_tol=1e-6)


def test_smelter_min_grade_refuses_low_concentrate():
    """Concentrate below the smelter min grade (default 18% Cu) is refused
    — payable goes to zero and no concentrate revenue is booked."""
    tea = TEAParams()
    sim = _toy_sim(conc_tph=120.0, conc_grade=0.10)
    r = evaluate(sim, tea)
    assert r["breakdown"]["payable_cu_tph"] == 0.0
    assert r["breakdown"]["gross_revenue"] == 0.0


def test_smelter_partial_payable_in_penalty_band():
    """Concentrate between 18% and 25% Cu attracts a linear payable factor."""
    tea = TEAParams()
    # conc_grade = 0.215 sits midway between 0.18 and 0.25 -> factor = 0.5
    sim = _toy_sim(conc_tph=80.0, conc_grade=0.215)
    r = evaluate(sim, tea)
    cu_in_conc = 80.0 * 0.215
    deductible = 0.01 * 80.0
    expected_payable = (cu_in_conc - deductible) * 0.965 * 0.5
    assert math.isclose(r["breakdown"]["payable_cu_tph"],
                        expected_payable, rel_tol=1e-6)


# ---------------------------------------------------------------- opex
def test_opex_arithmetic():
    tea = TEAParams()
    sim = _toy_sim()
    r = evaluate(sim, tea)

    total_kw = 300 + 5000 + 15000 + 900
    grind_kw = 5000 + 15000
    ore_tph = 2000.0
    hrs = tea.hours_per_year

    assert math.isclose(r["breakdown"]["power_cost"],
                        total_kw * hrs * 0.05, rel_tol=1e-6)
    assert math.isclose(r["breakdown"]["media_cost"],
                        grind_kw * hrs * 0.30, rel_tol=1e-6)
    assert math.isclose(r["breakdown"]["reagent_cost"],
                        ore_tph * hrs * 0.50, rel_tol=1e-6)
    assert math.isclose(r["breakdown"]["water_cost"],
                        5000.0 * hrs * 0.30, rel_tol=1e-6)


def test_om_opex_is_pct_of_capex():
    tea = TEAParams()
    sim = _toy_sim()
    r = evaluate(sim, tea)
    assert math.isclose(r["breakdown"]["om_opex"],
                        r["total_capex"] * tea.om_opex_pct_capex,
                        rel_tol=1e-9)


# ---------------------------------------------------------------- per-stage capex
def test_per_stage_capex_lines_present():
    """Every toggleable + non-toggleable stage has its own breakdown line."""
    sim = _toy_sim()
    r = evaluate(sim, TEAParams())
    b = r["breakdown"]
    for key in ("crusher_capex", "crusher_motor_capex",
                "sag_capex", "bm_capex",
                "cyclone_capex", "flotation_capex",
                "thickener_capex", "filter_capex",
                "tailings_capex"):
        assert key in b, f"missing {key}"
        assert b[key] >= 0.0


def test_sag_capex_uses_mular_basis():
    sim = _toy_sim(sag_kw=5000.0)
    b = evaluate(sim, TEAParams())["breakdown"]
    expected = SAG_MILL_BASIS[0] * (5000.0 / SAG_MILL_BASIS[1]) ** SAG_MILL_BASIS[2]
    assert math.isclose(b["sag_capex"], expected, rel_tol=1e-9)


def test_filter_capex_uses_belt_filter_basis():
    sim = _toy_sim(filter_area=40.0)
    b = evaluate(sim, TEAParams())["breakdown"]
    expected = BELT_FILTER_BASIS[0] * (40.0 / BELT_FILTER_BASIS[1]) ** BELT_FILTER_BASIS[2]
    assert math.isclose(b["filter_capex"], expected, rel_tol=1e-9)


def test_thickener_capex_uses_diameter_basis():
    sim = _toy_sim(thickener_dia=20.0)
    b = evaluate(sim, TEAParams())["breakdown"]
    expected = THICKENER_BASIS[0] * (20.0 / THICKENER_BASIS[1]) ** THICKENER_BASIS[2]
    assert math.isclose(b["thickener_capex"], expected, rel_tol=1e-9)


def test_lang_factor_applied_to_equipment_only():
    """installed = equipment × Lang(route) + tailings + balance-of-plant.

    A default (flotation) toy sim takes the sulfide Lang factor; tailings,
    heap pad and the balance-of-plant line are already-installed and bypass
    Lang. The toy sim has no heap pad, so that term is zero here.
    """
    tea = TEAParams()
    sim = _toy_sim()
    r = evaluate(sim, tea)
    b = r["breakdown"]
    expected = (b["equipment_capex"] * tea.lang_factor_sulfide
                + b["tailings_capex"] + b["balance_of_plant_capex"])
    assert math.isclose(b["installed_capex"], expected, rel_tol=1e-9)
    assert math.isclose(r["total_capex"], b["installed_capex"], rel_tol=1e-9)
    # Default flotation route picks the sulfide (greenfield) Lang factor.
    assert b["lang_factor_applied"] == tea.lang_factor_sulfide
    # Balance-of-plant scales with ore feed (flotation route).
    assert math.isclose(b["balance_of_plant_capex"],
                        2000.0 * 24.0 * tea.balance_of_plant_per_tpd,
                        rel_tol=1e-9)


def test_tailings_adverse_factor_scales_linearly():
    sim = _toy_sim()
    r1 = evaluate(sim, TEAParams(tailings_adverse_factor=1.0))
    r3 = evaluate(sim, TEAParams(tailings_adverse_factor=3.0))
    assert math.isclose(r3["breakdown"]["tailings_capex"],
                        3.0 * r1["breakdown"]["tailings_capex"], rel_tol=1e-9)


# ---------------------------------------------------------------- topology toggles change capex
def test_bm_off_zeroes_bm_capex_and_lowers_total():
    """Turning the ball mill off must remove the BM capex line entirely."""
    on  = _toy_sim(topology=Topology(ball_mill_enabled=True, cyclone_enabled=True))
    off = _toy_sim(bm_kw=0.0,
                   topology=Topology(ball_mill_enabled=False, cyclone_enabled=False))
    r_on = evaluate(on, TEAParams())
    r_off = evaluate(off, TEAParams())
    assert r_off["breakdown"]["bm_capex"] == 0.0
    assert r_on["breakdown"]["bm_capex"] > 0.0
    # Total capex must drop too (Lang applies, so the delta is amplified).
    assert r_off["total_capex"] < r_on["total_capex"]


def test_cyclone_off_zeroes_cyclone_capex():
    on  = _toy_sim(topology=Topology(cyclone_enabled=True, ball_mill_enabled=True))
    off = _toy_sim(topology=Topology(cyclone_enabled=False, ball_mill_enabled=True))
    r_on = evaluate(on, TEAParams())
    r_off = evaluate(off, TEAParams())
    assert r_on["breakdown"]["cyclone_capex"] > 0.0
    assert r_off["breakdown"]["cyclone_capex"] == 0.0
    assert r_off["total_capex"] < r_on["total_capex"]


def test_thickener_off_zeroes_thickener_capex():
    on  = _toy_sim(topology=Topology(thickener_enabled=True))
    off = _toy_sim(topology=Topology(thickener_enabled=False))
    r_on = evaluate(on, TEAParams())
    r_off = evaluate(off, TEAParams())
    assert r_on["breakdown"]["thickener_capex"] > 0.0
    assert r_off["breakdown"]["thickener_capex"] == 0.0
    assert r_off["total_capex"] < r_on["total_capex"]


def test_filter_off_zeroes_filter_capex():
    on  = _toy_sim(topology=Topology(filter_enabled=True))
    off = _toy_sim(topology=Topology(filter_enabled=False))
    r_on = evaluate(on, TEAParams())
    r_off = evaluate(off, TEAParams())
    assert r_on["breakdown"]["filter_capex"] > 0.0
    assert r_off["breakdown"]["filter_capex"] == 0.0
    assert r_off["total_capex"] < r_on["total_capex"]


# ---------------------------------------------------------------- grade sensitivity
def test_capex_grows_with_concentrate_flow():
    """Higher head grade -> more concentrate -> larger conc thickener and
    filter -> higher capex on those lines. Simulated here by directly
    scaling the diagnostics that the simulator would produce."""
    low  = _toy_sim(thickener_dia=15.0, filter_area=20.0)
    high = _toy_sim(thickener_dia=25.0, filter_area=60.0)
    r_low  = evaluate(low,  TEAParams())
    r_high = evaluate(high, TEAParams())
    assert r_high["breakdown"]["thickener_capex"] > r_low["breakdown"]["thickener_capex"]
    assert r_high["breakdown"]["filter_capex"]    > r_low["breakdown"]["filter_capex"]
    assert r_high["total_capex"] > r_low["total_capex"]


# ---------------------------------------------------------------- cashflow
def test_fcf_and_npv_pretax():
    tea = TEAParams()
    sim = _toy_sim()
    r = evaluate(sim, tea)

    assert math.isclose(r["annual_fcf"],
                        r["annual_revenue"] - r["annual_opex"], rel_tol=1e-9)
    pv = sum(1.0 / (1 + tea.discount_rate) ** t
             for t in range(1, tea.plant_life_years + 1))
    expected_npv = -r["total_capex"] + r["annual_fcf"] * pv
    assert math.isclose(r["npv"], expected_npv, rel_tol=1e-9)


def test_irr_solves_npv_zero():
    tea = TEAParams()
    sim = _toy_sim()
    r = evaluate(sim, tea)
    if math.isnan(r["irr"]):
        return
    pv = sum(1.0 / (1 + r["irr"]) ** t
             for t in range(1, tea.plant_life_years + 1))
    resid = -r["total_capex"] + r["annual_fcf"] * pv
    assert abs(resid) < 1.0


def test_unprofitable_returns_negative_npv():
    tea = TEAParams(cu_price_per_t=500.0)
    sim = _toy_sim()
    r = evaluate(sim, tea)
    assert r["npv"] < 0


def test_integration_with_real_simulator():
    """End-to-end smoke: a viable cleaner+regrind concentrator at 1.5% Cu
    feed produces a smelter-spec concentrate and books positive revenue."""
    from process_model.throughput import simulate_cu_sulfide
    topo = Topology(regrind_enabled=True, cleaner_enabled=True)
    sim = simulate_cu_sulfide(2000.0, 0.015, topology=topo)
    r = evaluate(sim)
    assert math.isfinite(r["npv"])
    assert r["total_capex"] > 0
    assert r["annual_revenue"] > 0


def test_default_sulfide_capex_in_defensible_band():
    """Phase 2B/2C regression guard: a 2000 tph default sulfide concentrator
    must land in the post-rework capex band (route Lang 6.0 + balance-of-plant),
    well above the pre-Phase-2B level. Real 50 ktpd Cu concentrators run
    $1.4-2.4B; the concentrator-only scope (no greenfield tailings dam) sits
    below that, but must clear $0.8B."""
    from process_model.throughput import simulate_cu_sulfide
    sim = simulate_cu_sulfide(2000.0, 0.008, topology=Topology())
    r = evaluate(sim)
    assert r["total_capex"] > 0.8e9
    assert r["breakdown"]["balance_of_plant_capex"] > 0.0


def test_hydromet_stages_add_capex_and_cathode_revenue():
    from process_model.throughput import simulate_cu_sulfide
    t_no = Topology()
    t_yes = Topology(regrind_enabled=True, cleaner_enabled=True,
                     leach_enabled=True, sx_enabled=True, ew_enabled=True,
                     neutralization_enabled=True)
    sim_no = simulate_cu_sulfide(2000.0, 0.008, topology=t_no)
    sim_yes = simulate_cu_sulfide(2000.0, 0.008, topology=t_yes)
    r_no = evaluate(sim_no)
    r_yes = evaluate(sim_yes)
    # All four hydromet capex lines non-zero when on, zero when off
    for k in ("leach_capex", "sx_capex", "ew_capex", "neut_capex"):
        assert r_yes["breakdown"][k] > 0, f"{k} should be > 0 when stage enabled"
        assert r_no["breakdown"][k] == 0, f"{k} should be 0 when stage disabled"
    # Cathode revenue appears
    assert r_yes["breakdown"]["cathode_revenue"] > 0
    assert r_no["breakdown"]["cathode_revenue"] == 0
    # Hydromet opex appears
    assert r_yes["breakdown"]["hydromet_opex"] > 0
    assert r_no["breakdown"]["hydromet_opex"] == 0


def test_screen_regrind_cleaner_capex_lines_toggle():
    from process_model.throughput import simulate_cu_sulfide
    t_no = Topology()
    t_yes = Topology(screen_enabled=True, regrind_enabled=True,
                     cleaner_enabled=True)
    r_no = evaluate(simulate_cu_sulfide(2000.0, 0.008, topology=t_no))
    r_yes = evaluate(simulate_cu_sulfide(2000.0, 0.008, topology=t_yes))
    for k in ("screen_capex", "regrind_capex", "cleaner_capex"):
        assert r_yes["breakdown"][k] > 0
        assert r_no["breakdown"][k] == 0


def test_cleaner_capex_uses_cleaner_cell_basis():
    """Cleaner cells are smaller than rougher cells and must select their own
    flotation-cell cost tier."""
    from process_model.throughput import simulate_cu_sulfide
    t = Topology(regrind_enabled=True, cleaner_enabled=True)
    sim = simulate_cu_sulfide(2000.0, 0.008, topology=t)
    r = evaluate(sim)
    clean = sim["cleaner"]
    fb = select_flot_basis(clean["cell_volume_m3"])
    expected = (perry_cost(fb[0], fb[1], fb[2], clean["cell_volume_m3"])
                * clean["num_cells"])
    assert math.isclose(r["breakdown"]["cleaner_capex"], expected, rel_tol=1e-9)


def test_regrind_power_counts_in_media_cost():
    """Grinding media opex should include SAG, primary BM, and regrind power."""
    from process_model.throughput import simulate_cu_sulfide
    t = Topology(regrind_enabled=True, cleaner_enabled=True)
    sim = simulate_cu_sulfide(2000.0, 0.008, topology=t)
    tea = TEAParams()
    r = evaluate(sim, tea)
    p = sim["power_kw"]
    expected = (p.get("sag_kw", 0.0) + p.get("bm_kw", 0.0)
                + p.get("regrind_kw", 0.0)) * tea.hours_per_year * tea.media_cost_per_kwh_grind
    assert math.isclose(r["breakdown"]["media_cost"], expected, rel_tol=1e-9)


# ---------------------------------------------------------------- parallel_cost
def test_parallel_cost_n1_matches_perry_below_cap():
    from process_model.tea import parallel_cost
    cost, n = parallel_cost(50_000_000.0, 16_500.0, 0.85, 10_000.0, 22_000.0)
    expected = perry_cost(50_000_000.0, 16_500.0, 0.85, 10_000.0)
    assert n == 1
    assert math.isclose(cost, expected, rel_tol=1e-9)


def test_parallel_cost_splits_above_cap():
    from process_model.tea import parallel_cost
    # 33 MW BM with 22 MW cap -> 2 units of 16.5 MW each.
    cost, n = parallel_cost(50_000_000.0, 16_500.0, 0.85, 33_000.0, 22_000.0)
    assert n == 2
    expected_per_unit = perry_cost(50_000_000.0, 16_500.0, 0.85, 16_500.0)
    assert math.isclose(cost, 2 * expected_per_unit, rel_tol=1e-9)


def test_parallel_cost_zero_capacity_returns_zero():
    from process_model.tea import parallel_cost
    cost, n = parallel_cost(50_000_000.0, 16_500.0, 0.85, 0.0, 22_000.0)
    assert cost == 0.0 and n == 0


def test_high_throughput_parallelizes_bm_and_cyclones():
    """At 4000 tph the BM should split into 2 units and cyclones into many."""
    from process_model.throughput import simulate_cu_sulfide
    sim = simulate_cu_sulfide(4000.0, 0.008)
    b = evaluate(sim)["breakdown"]
    assert b["n_bm"] == 2, f"expected 2 BMs at 4000 tph, got {b['n_bm']}"
    assert b["n_cyclone"] >= 30, f"expected many cyclones, got {b['n_cyclone']}"


def test_default_throughput_no_mill_parallelization():
    """At the 2000 tph default the SAG/BM stay single-train (within caps)."""
    from process_model.throughput import simulate_cu_sulfide
    sim = simulate_cu_sulfide(2000.0, 0.008)
    b = evaluate(sim)["breakdown"]
    assert b["n_sag"] == 1 and b["n_bm"] == 1
    # Cyclone always parallelizes at 2000 tph (~8300 m^3/h vs ~400 m^3/h cap).
    assert b["n_cyclone"] >= 20
