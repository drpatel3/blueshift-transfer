"""Techno-economic evaluation of the Cu-sulfide flowsheet.

Scope: concentrator-only (run-of-mine ore in, Cu concentrate out). No mine,
no labor / G&A. Tailings storage included (O'Hara Eq 6.3.145). Add labor
and G&A before using the absolute NPV for an investment decision; the
present output supports relative comparison across flowsheet variants.

Per-stage capex
---------------

Capex is built stage-by-stage so every Topology() boolean is a real cost
lever. All stages except tailings storage are equipment-only Perry's
Table 9-50 power-laws (Mular for mills, where Perry tabulates by
diameter/volume but the simulator only exposes installed motor kW); they
get the BlueShift Lang factor 5.0 uplift to convert equipment cost to
installed cost. Tailings storage uses O'Hara 1992 Eq 6.3.145, which is
already integrated installed cost and bypasses Lang.

| Stage           | Sizing var (sim)          | Source                           | Toggleable |
|-----------------|---------------------------|----------------------------------|------------|
| Primary crusher | crusher_kw                | Perry T9-50 jaw two-tier         | no         |
| Crusher motor   | crusher_kw                | Perry T9-50 motor two-tier       | no         |
| SAG mill        | sag_kw                    | Mular & Poulter (2002), kW^0.69  | no         |
| Ball mill       | bm_kw                     | Mular & Poulter (2002), kW^0.69  | yes (BM)   |
| Hydrocyclone    | feed slurry m³/h          | Perry T9-50 hydrocyclone         | yes (cyc)  |
| Flotation bank  | per-cell volume * num_cells | Arfania et al. (2017), V^0.452 | no         |
| Conc thickener  | conc_thickener.diameter_m | Perry T9-50 / Parkinson-Mular    | yes        |
| Belt filter     | belt_filter.area_m2       | Perry T9-50 vacuum belt filter   | yes        |
| Tailings        | ore tpd (short)           | O'Hara 1992 Eq 6.3.145           | no         |

Throughput vs grade
-------------------

Throughput (ore_tph) drives crusher kW, mill kW, cyclone slurry flow,
flotation slurry volume, tailings tonnage. Grade enters through
conc-side equipment: `conc_thickener.diameter_m` and
`belt_filter.area_m2` are sized on flotation concentrate flow, which
depends on head grade and recovery. So head grade now moves capex.

Bases below are in 2025 USD (M&S = 2383) so we set inflation=1.0 when
calling perry_cost. Where a literature basis was published in earlier
dollars (Mular 2002, Parkinson-Mular ~1991), the basis_usd value below
already includes the M&S uplift to 2025; the comment cites the source
and the inflation factor used. Numerical values should be verified
against the source PDFs before any external use of absolute NPV.

Financial skeleton: BlueShift methodology (25-yr pre-tax DCF at 8%, 7884
op-hours, $0.05/kWh, 3.5% O&M, Lang 5.0). Concentrate sale net of TC/RC.

================================================================
SINGLE SOURCE OF TRUTH
================================================================

Every monetary constant in the process_model package lives in this file
(see the COST BASES block immediately below). Stages elsewhere (`crusher`,
`sag`, `hydrocyclone`, `flotation`, `thickener`, `filter`) carry only
physics and sizing — no dollar figures.

Dependency map (where each constant is consumed):

    CRUSHER_BASES, MOTOR_BASES               -> evaluate(): crusher + motor
    SAG_MILL_BASIS, BM_MILL_BASIS            -> evaluate(): SAG + BM lines
    HYDROCYCLONE_BASIS                       -> evaluate(): cyclone line
    AGITATED_TANK_BASIS, FROTH_MULTIPLIER    -> evaluate(): flotation line
    THICKENER_BASIS                          -> evaluate(): conc thickener
    BELT_FILTER_BASIS                        -> evaluate(): belt filter
    TAILINGS_MIN_USD_1988, TAILINGS_EXP      -> tailings_storage_capex()
    ATM_TANK_BASIS, EW_CELL_BASIS            -> centralized for future
                                                leach/SX/EW expansion
    M_S_INDEX_*                              -> inflation factor anchors
    DEFAULT_*                                -> TEAParams field defaults
    Sizing variables (read-only consumers of simulator output):
        crusher_kw, sag_kw, bm_kw            <- power_kw[*]
        bm_discharge slurry m^3/h            <- streams["bm_discharge"]
        flot total_volume_m3                 <- sim_result["flotation"]
        thickener diameter_m                 <- sim_result["conc_thickener"]
        filter area_m2                       <- sim_result["belt_filter"]
        ore tpd, conc_tph                    <- streams["ore"], ["filter_cake"]
        cyclone water_tph                    <- streams["cyc_of"]
        topology booleans                    <- sim_result["topology"]

External callers: `optimizer.py` reads TEAParams + evaluate(); `throughput.py`
imports nothing from here (TEA is downstream of the simulator).
"""
from dataclasses import dataclass, field, replace
from typing import Optional


# ============================================================================
# COST BASES — every monetary constant in the process_model package
# ============================================================================
#
# Capex bases use the (basis_usd, basis_capacity, exponent) tuple form, all
# in **2025 USD** (M&S=2383); pass them through perry_cost(..., inflation=1.0).
# Where a literature basis was published in earlier dollars, the basis_usd
# value below already includes the M&S uplift to 2025; the citation comment
# states the year and uplift factor.
#
# Numerical values should be verified against the source PDFs before any
# external use of absolute NPV.

# ---------------------------------------------------------------- M&S indices
# Marshall & Swift Equipment Cost Index. Pending verification against an
# authoritative issue before external citation.
M_S_INDEX_PERRYS = 1000.0   # Perry T9-50 publication basis (symbolic)
M_S_INDEX_2025 = 2383.0
M_S_INDEX_1988 = 852.0      # Q3 1988 — O'Hara grinding-equation basis
M_S_INDEX_2002 = 1104.0     # 2002 — Mular & Poulter basis year
M_S_INDEX_1991 = 931.0      # 1991 — Parkinson-Mular thickener basis year
M_S_INDEX_2013 = 1561.0     # 2013 — Arfania et al. flotation-cell basis year

SHORT_PER_METRIC_TON = 1.10231

# ---------------------------------------------------------------- crusher (Perry T9-50)
# Jaw crusher capex vs shaft power (kW), two-tier. Perry basis (M&S=1000)
# uplifted by 2.383 to 2025.
CRUSHER_BASES = [
    (34_000.0  * 2.383,  7.5,  0.65),   # small: kW <= ~40
    (284_000.0 * 2.383, 74.6,  0.81),   # large: kW > 40
]
# Crusher drive motor vs kW. Perry T9-50, two-tier, M&S=1000 -> 2025.
MOTOR_BASES = [
    (12_300.0 * 2.383,  7.5, 0.56),
    (19_300.0 * 2.383, 52.0, 0.77),
]

# ---------------------------------------------------------------- mills (Phase 2A vendor-anchored refit)
# SAG / ball mill installed equipment cost on motor kW.
#
# Phase 2A refit (CAPEX_REWORK_PLAN.md §2A): the prior Mular & Poulter (2002)
# fit `C_2025_USD = 14,005 * kW^0.6915` was anchored on 1990s-era 1-8 MW mills
# and underweights the diseconomies of scale at modern 16-22 MW mega-mills.
# Real Cu plants at 50 ktpd run $5,000-8,000/kW installed; the old fit gave
# ~$3,200/kW installed (Lang x5).
#
# New basis is vendor-anchored on 2024 USD quotes for modern Cu-plant mills:
#   - 16.5 MW ball mill equipment: ~$50M (Outotec / FLSmidth / Metso)
#   - 22 MW SAG mill equipment   : ~$60M
# Higher exponent (0.85) captures scale diseconomy. Validation:
#   - 5 MW SAG: $50M * (5000/16500)^0.85 = $18.4M  (matches midsize quotes)
#   - 16.5 MW BM: $50M                              (matches modern Cu plants)
#   - 22 MW SAG: $50M * (22000/16500)^0.85 = $63.7M (matches upper-end quotes)
# Same basis for SAG and BM at scale (vendor catalog prices converge per kW
# beyond ~10 MW). Refit when CostMine 2024 data is on hand.
# Source: references/equipment_basis_2026.md §1 (SAG), §2 (BM).
SAG_MILL_BASIS = (50_000_000.0, 16_500.0, 0.85)
BM_MILL_BASIS  = (50_000_000.0, 16_500.0, 0.85)

# Max single-unit caps for parallelization (references/equipment_basis_2026.md §4):
# - SAG: 28 MW gearless (Cobre Panama, world's largest in commercial service)
# - BM:  22 MW gearless (Cobre Panama, largest commercial BM)
# Above these, the model splits into n parallel mills via parallel_cost().
MAX_SINGLE_SAG_KW = 28_000.0
MAX_SINGLE_BM_KW  = 22_000.0

# ---------------------------------------------------------------- hydrocyclone (Perry T9-50, 2025-bumped)
# Single-unit equipment vs feed slurry volumetric flow m^3/h. Perry row at
# M&S=1000 ($12k @ 100 m^3/h, exp 0.55) -> 2024 (x 2.383) = $28.6k -> 2025
# (+5% mining-equipment inflation) ~= $30k. Single-unit basis only; vendor max
# diameter ~840 mm caps a real cyclone at ~400 m^3/h, so high-throughput
# circuits run a parallel cluster (parallelization is a Phase 2B follow-up).
# Source: references/equipment_basis_2026.md §3.
HYDROCYCLONE_BASIS = (30_000.0, 100.0, 0.55)
# Single-unit max throughput at the Krebs gMAX 840 mm vendor ceiling (~400 m^3/h
# slurry on typical Cu pulp). Above this the circuit runs a parallel cluster.
MAX_SINGLE_CYCLONE_M3PH = 400.0

# ---------------------------------------------------------------- flotation (Arfania et al. 2017)
# Mechanical flotation cell capex per single cell vs cell volume m^3.
# Arfania, Sayadi & Khalesi (2017), "Cost modelling for flotation machines",
# J. South. Afr. Inst. Min. Metall. 117(1):153-158, DOI 10.17159/2411-9717/
# 2017/v117n1a13. Univariate exponential regression on 10 industrial Standard
# Flotation Machines:
#     CC_2013_USD = 25,351.28 * V^0.452     (R^2 = 0.9561, V = 0.28-158.6 m^3)
# Inflated 2013 -> 2025 via M&S 1561 -> 2383 (factor 1.527):
#     CC_2025_USD = 38,711 * V^0.452        (per cell, equipment-only)
# Refactor to perry_cost form at a basis V = 100 m^3:
#     basis = 38,711 * 100^0.452 = $310,841
# Cell-volume fit range is 0.28-158.6 m^3; modern industrial cells (e.g. FLS
# 660 m^3, Outotec TankCell e500) extrapolate beyond the regression and are
# handled by the Phase 2A large-cell tier below. Cell volume is a discrete
# equipment spec (default e500 = 500 m^3), not an optimizer variable.
FLOTATION_CELL_BASIS = (310_841.0, 100.0, 0.452)
# Phase 2A large-cell tier (CAPEX_REWORK_PLAN.md §2A): modern Cu plants run
# Outotec TankCell e300/e500/e660 cells beyond Arfania's 158 m^3 fit ceiling.
# Vendor list price for an installed e500 (~500 m^3) is ~$1.5M equipment.
# Lower exponent (0.50) reflects ~sqrt(volume) tank-shell scaling above the
# 200 m^3 threshold where standard mechanical-cell geometry gives way to
# tank-cell geometry. Selected by `select_flot_basis(cell_volume_m3)` below.
FLOTATION_CELL_BASIS_LARGE = (1_500_000.0, 500.0, 0.50)
FLOT_BASIS_CROSSOVER_M3 = 158.0    # use Arfania below, large-cell tier above
FLOT_FROTH_VOLUME_MULTIPLIER = 1.2  # carried for sizing physics in throughput.py
# Legacy Perry T9-50 vertical agitated tank vs volume m^3. Kept available
# for general agitated tanks (leach reactors, conditioners) that aren't
# flotation machines.
AGITATED_TANK_BASIS = (12_300.0 * 2.383, 3.8, 0.50)

# ---------------------------------------------------------------- thickener (Parkinson-Mular 1991)
# Equipment vs tank diameter m. Parkinson-Mular form C = a_t * D^1.6;
# basis $5,000 @ 10 m (1991 USD) x 2.56 (M&S 931 -> 2383) = $510k (2025).
# Reference: Parkinson & Mular, "Mineral Processing Equipment Costs and
# Preliminary Capital Cost Estimations" (1991), via Wills 7th ed.
THICKENER_BASIS = (510_000.0, 10.0, 1.6)

# ---------------------------------------------------------------- belt filter (Perry T9-50)
# Horizontal vacuum belt filter vs belt area m^2. Perry row $62,000 @ 10 m^2
# exp 0.58 (M&S=1000) x 2.383 = $147,750 (2025).
BELT_FILTER_BASIS = (147_750.0, 10.0, 0.58)

# ---------------------------------------------------------------- tailings (O'Hara 1992 Eq 6.3.145)
# Minimum initial tailings storage at a favorable site. 1988 USD, T = short
# tons of ore per day, integrated installed (bypass Lang). O'Hara notes
# realistic projects run 2-5x; controlled by tailings_adverse_factor.
TAILINGS_MIN_USD_1988 = 20_000.0
TAILINGS_EXP = 0.5

# ---------------------------------------------------------------- atmospheric tank (Perry T9-50)
# Vs volume m^3. Used for leach tanks, SX mixer-settlers, and neutralization
# tanks via the aliases below — same Perry row, different stage labels in
# the breakdown.
ATM_TANK_BASIS = (9_300.0 * 2.383, 0.38, 0.53)
LEACH_TANK_BASIS = ATM_TANK_BASIS
SX_MIXER_SETTLER_BASIS = ATM_TANK_BASIS
NEUTRALIZATION_TANK_BASIS = ATM_TANK_BASIS

# ---------------------------------------------------------------- regrind mill (Mular & Poulter 2002)
# Same correlation as the primary BM (sized on installed motor kW) but
# typically smaller mill (1-3 MW). Toggleable when regrind_enabled=True.
REGRIND_MILL_BASIS = BM_MILL_BASIS

# ---------------------------------------------------------------- electrowinning cell (Perry T9-50)
# Vs total electrode area m^2 (Perry-style power-law from
# process_cost_integration.py:662-665). Active when ew_enabled=True.
EW_CELL_BASIS = (383_000.0 * 2.383, 94.0, 0.81)

# ---------------------------------------------------------------- vibrating screen (vendor-quote anchored)
# PLACEHOLDER until a proper Perry T9-50 row or Mular-Halbe-Barratt 2002
# coefficient is verified. Anchored on Metso/FLSmidth 2024-2025 vendor list
# prices for double-deck banana screens: ~$120-200k for an 8 m^2 deck.
# Basis: $150,000 @ 8 m^2 deck area, exp 0.55 (2025 USD). Refit when the
# Mular handbook is on hand.
SCREEN_BASIS = (150_000.0, 8.0, 0.55)

# ---------------------------------------------------------------- heap leach pad (Mular 2002 + project disclosures)
# Whole-ore heap pad capex covers: liner system (HDPE / GCL composite),
# pad earthworks, irrigation drip system, PLS pond, ripios stacking
# infrastructure. Does NOT cover SX/EW (separate lines via SX_/EW_BASIS).
#
# Sizing: power-law on annual ore throughput (t/yr design). Industry
# anchor points (2024 USD, total project disclosed):
#   - BHP Spence growth option, ~16 Mt/yr heap : ~$200M
#   - Codelco Radomiro Tomic, ~30 Mt/yr        : ~$400M
#   - Lomas Bayas Mantoverde, ~14 Mt/yr        : ~$130M
# Power-law fit at $200M @ 16 Mt/yr design, exp 0.85 (slight scale
# economy). Already integrated installed cost — bypasses Lang factor
# (heap pad is mostly civil + earthworks, not equipment).
# Source: Mular 2002 Vol 2 §4.2; SME Mineral Processing Handbook (2019)
# Ch. 22; recent NI 43-101 disclosures.
HEAP_PAD_BASIS = (200_000_000.0, 16_000_000.0, 0.85)  # ($, t/yr design, exp)

# ---------------------------------------------------------------- BlueShift unit rates / financial knobs
# Defaults for TEAParams. Override per-call by constructing TEAParams(...)
# with explicit keyword arguments.
#
# Commodity / smelter terms — concentrate sale.
# TC/RC reset Apr 2026 to current annual benchmark from BlueShift's
# 2018-vintage $80/t-conc + 8c/lb. 2024 Antofagasta-Jiangxi benchmark
# settled at $21.25/t TC and 2.125c/lb RC (Reuters 2023-11-30); 2025
# benchmark settled at $21.25/t TC again (Fastmarkets 2024-11-22). 2025
# spot has run negative on tight smelter capacity but the annual contract
# basis is what merchant concentrators model; refit when the next
# benchmark settles.
DEFAULT_CU_PRICE_PER_T     = 12_500.0   # USD / t Cu metal
DEFAULT_TC_PER_T_CONC      =     22.0   # USD / t conc — 2024-25 annual benchmark
DEFAULT_RC_PER_T_CU        =     50.0   # USD / t Cu  — ~2.3 c/lb (2024-25 benchmark)
DEFAULT_PAYABLE_FRACTION   =      0.965
DEFAULT_DEDUCTIBLE_UNITS   =      0.01  # 1 unit Cu deduction
# Smelter arsenic-penalty schedule. Cu smelters price As as a function of
# the *concentrate* As grade, not the ore type: below a penalty-free limit
# the concentrate is clean; above it, a per-unit deduction applies; above a
# hard maximum the concentrate is unsaleable (rejected) and a producer must
# route the Cu to a hydrometallurgical circuit instead of the smelter. This
# is the documented reason the Andean enargite-tennantite belt (El Abra,
# Spence Hypogene, Chuquicamata/Ministro Hales) built POX / roasting rather
# than selling concentrate.
#
# `smelter_as_penalty_per_t_cu()` implements that schedule generally — it is
# keyed to concentrate As grade, NOT to any plant or mineralogy name. The
# per-deposit As grade is a disclosed ore property supplied as an input
# (see throughput.AS_CONC_PCT_BY_MINERALOGY and CuSulfideParams.conc_arsenic_pct).
# Full source notes + the engineering-judgment flags are in
# references/references.md (§Smelter arsenic-penalty schedule).
#
# Brackets:
#   As <= 0.20 %  : penalty-free. ~0.2 % (2000 ppm) is the widely-quoted
#                   commercial clean-concentrate As limit (AusIMM "Copper
#                   Concentrate Marketing 101").
#   0.20–0.50 %   : linear deduction ramping 0 -> $500/t Cu. $500/t Cu
#                   (~2.3 c/lb) anchors to the top-quartile of disclosed
#                   As-rich Cu concentrate deductions (Doyle 2010, SEG SP 18,
#                   the project's established penalty source); the ramp shape
#                   between limits is engineering judgment, not a fitted curve.
#   As >= 0.50 %  : rejected — concentrate exceeds the ~0.5 % As limit applied
#                   by the dominant concentrate-import market (China), the
#                   de-facto commercial rejection threshold. Modelled as a
#                   penalty equal to the Cu price, i.e. the concentrate carries
#                   no net payable value, so the sulfide-concentrate route
#                   cannot win against a cathode-producing route.
AS_PENALTY_FREE_PCT       =    0.20   # %, clean-concentrate As limit
AS_PENALTY_REJECT_PCT     =    0.50   # %, max saleable As (smelter/import spec)
AS_PENALTY_AT_REJECT_USD  =  500.0    # $/t Cu at the rejection threshold


def smelter_as_penalty_per_t_cu(conc_arsenic_pct: float,
                                cu_price_per_t: float = DEFAULT_CU_PRICE_PER_T
                                ) -> float:
    """Smelter As deduction in USD / t payable Cu as a function of the
    concentrate As grade (percent). General contract schedule — see the
    comment block above for bracket sources. For rejected concentrate
    (As >= the rejection limit) returns the Cu price actually in use, so the
    deduction exactly zeroes the concentrate's net payable value in
    evaluate() regardless of the price assumption."""
    if conc_arsenic_pct <= AS_PENALTY_FREE_PCT:
        return 0.0
    if conc_arsenic_pct >= AS_PENALTY_REJECT_PCT:
        return cu_price_per_t   # unsaleable — reject
    frac = ((conc_arsenic_pct - AS_PENALTY_FREE_PCT)
            / (AS_PENALTY_REJECT_PCT - AS_PENALTY_FREE_PCT))
    return AS_PENALTY_AT_REJECT_USD * frac


DEFAULT_IMPURITY_PENALTY_PER_T_CU =  0.0   # default clean concentrate

# Byproduct credits — paid by the smelter on the sulfide concentrate
# route only. POX and heap routes get zero (Mo/Au/Ag report to the
# leach residue or solution where commercial recovery is rare).
#
# Mo: recovered to MoS2 concentrate via a separate Mo flotation cleaner
#   on the bulk Cu rougher concentrate. Real-plant flotation recovery
#   ~50-70% (Sutulov 1976; Damjanović & Goode 2000); 0.55 is mid-range
#   conservative. Mo concentrate price $25/lb = $55,000/t Mo, USGS MCS
#   2025 average for technical-grade MoO3 (after roasting). The roaster
#   TC + refining penalty is folded into the price (gross-of-roast Mo
#   metal price minus ~$5/lb roast cost ~ $44,000/t net to the producer);
#   we use the net figure directly.
#
# Au: 95% payable in concentrate, refined separately. Price $2,500/oz =
#   $80,376,755/t (USGS MCS 2025 average 2024 = $2,386/oz; 2025 spot
#   running higher; rounding to $2,500 for forward-looking flat DCF).
#
# Ag: 85% payable in concentrate. Price $30/oz = $964,521/t (USGS MCS
#   2025 average 2024 = $28.27/oz; rounding to $30 for forward-looking).
DEFAULT_MO_RECOVERY        =      0.55
DEFAULT_MO_PRICE_PER_T     = 44_000.0   # USD / t Mo, net of roast TC
DEFAULT_AU_PAYABLE         =      0.95
DEFAULT_AU_PRICE_PER_T     = 80_376_755.0   # $2,500/oz × 32,150.7 oz/t
DEFAULT_AG_PAYABLE         =      0.85
DEFAULT_AG_PRICE_PER_T     =    964_521.0   # $30/oz × 32,150.7 oz/t


# Operating-rate basis
DEFAULT_HOURS_PER_YEAR     =   7884.0   # capacity factor 0.9 x 8760
# Utility unit rates
DEFAULT_POWER_COST_PER_KWH =      0.05  # $50 / MWh
DEFAULT_WATER_COST_PER_M3  =      0.30
DEFAULT_HEAT_COST_PER_KWH  =      0.02  # $20 / MWh natural gas
# Capex / opex factors
DEFAULT_OM_OPEX_PCT_CAPEX  =      0.035
# Legacy single Lang factor (Perry chemical-plant basis). Retained as a
# back-compat default; evaluate() now selects a route-specific factor below.
DEFAULT_LANG_FACTOR        =      5.0
# Route-dependent Lang factor (Phase 2C). Greenfield Cu concentrators carry
# heavier installation multipliers than the Perry chemical-plant 5.0. Mular,
# Halls & Barratt (2002), "Mining and Mineral Processing Plant Design," ch.1,
# and Sayadi, Khalesi & Borji (2014), Minerals Eng. 55:96-102, give a
# greenfield-mining Lang RANGE of 5-7 for a flotation concentrator and ~5 for
# a hydromet/heap operation (fewer installed mechanical lines; civil-heavy pad
# already on the O'Hara integrated line). The texts give the range, not a
# single number: 6.0 is the midpoint of 5-7 (author judgment within the cited
# band, NOT a value either source hands you directly). Selected on
# topo.flotation_enabled.
DEFAULT_LANG_FACTOR_SULFIDE  =    6.0   # midpoint of cited 5-7 greenfield band
DEFAULT_LANG_FACTOR_HYDROMET =    5.0   # heap / SX-EW (Perry default, unchanged)
# Balance-of-plant (Phase 2B). The per-stage Perry/Mular/Arfania bases capture
# the major EQUIPMENT, and the Lang factor above covers each item's OWN local
# piping / E&I / installation. Neither captures the plant-wide INTER-AREA
# infrastructure that is not attached to any single costed equipment line.
# Bottom-up estimate for a 48,000 tpd (2000 tph) greenfield Cu flotation plant,
# 2024 USD. Each line is the midpoint of the order-of-magnitude vendor range in
# CAPEX_REWORK_PLAN.md Phase 2B (which is an internal estimate scaled from
# Mular et al. 2002 ch.1, NOT a primary-sourced table) — treat as a defensible
# order-of-magnitude gap-filler, not an equation-faithful value:
#   inter-area conveyors / belt galleries      $40M  ->  $  833/tpd
#   tailings + reclaim slurry mains & pumps    $50M  ->  $1,042/tpd
#   reagent storage / distribution mains       $20M  ->  $  417/tpd
#   raw + reclaim water supply systems         $50M  ->  $1,042/tpd
#   main MCC / E&I house / control room        $30M  ->  $  625/tpd
#   -----------------------------------------------------------------
#   total                                     $190M  ->  $3,960/tpd of ore feed
# Scoped to plant-wide mains (tailings line to the TSF, reclaim mains, area
# conveyors) — distinct from the equipment-local piping the Lang factor already
# covers, so no double-count. Already an installed basis, so it bypasses Lang.
# Excludes tailings storage itself (O'Hara 6.3.145 line). Applied only to
# flotation routes: a heap operation has no mill-area inter-stage mains; its
# civil works are already on the O'Hara heap-pad line. NOTE: on a hybrid
# (flotation + tails leach) the dominant flotation concentrator sets a single
# site-wide Lang 6.0 over all equipment, including the leach add-on — a
# deliberate simplification (the plant is primarily a concentrator), at the
# cost of slightly over-installing the SX/EW/leach equipment.
DEFAULT_BALANCE_OF_PLANT_PER_TPD = 3_960.0   # USD / tpd ore feed (installed)
# Reagents & consumables (not in BlueShift Cu sheet; carried over)
DEFAULT_REAGENT_PER_T_ORE  =      0.50  # USD / t ore
DEFAULT_MEDIA_PER_KWH_GRIND=      0.30  # USD / kWh of grinding
# Ore PURCHASE — merchant-concentrator framing.
#
# This TEA represents a third-party concentrator buying ore at the gate,
# not a mine-owned operation. So the supplier (the mine) recovers all of
# its own capex (fleet, pre-strip, mine sustaining), opex (drill+blast,
# haulage, mine G&A) and margin through the ore purchase price we pay
# them. None of those line items appear elsewhere in this file — they
# are folded into the Cu-value coefficient below.
#
# Two-component price:
#   $/t-ore = base_per_t + cu_grade * per_t_cu
#
#   base_per_t  = $10/t-ore — gate handling, weighing, sampling, short-haul
#                 logistics from supplier siding to plant. Independent of
#                 grade.
#   per_t_cu    = $6,000/t Cu in ore. Bottom-up build from the Wood
#                 Mackenzie 2024 Cu cost curve (mid-quartile open-pit
#                 porphyry), with stronger capex amortization to reflect
#                 a partially-greenfield supplier:
#                   C1 cash cost (drill+blast, haul, mine G&A, energy)
#                       — Wood Mac 2024, $0.85-1.20/lb Cu
#                       midpoint  = $2,275/t Cu
#                   Sustaining capex (1-2% of installed mine capex/yr)
#                                 = $700/t Cu
#                   New-build mine capex amortization (closer to full
#                   recovery on a 25-yr life — greenfield-leaning mid)
#                                 = $1,900/t Cu
#                   Royalty (Chilean sliding scale 5-14% on margin —
#                   Jones Day 2023; rough ~3% NSR equivalent)
#                                 = $400/t Cu
#                   Mine's return on equity (15% on AISC)
#                                 = $725/t Cu
#                                 ----------
#                   Total          = $6,000/t Cu.
#                 Defensible bands:
#                   ~$3,800/t Cu : mature sunk-capex mine (Codelco / BHP
#                                  existing operations)
#                   ~$5,000/t Cu : mid-life mine, partial capex recovery
#                   ~$6,000/t Cu : default — partially-greenfield supplier
#                                  with closer-to-full capex amortization
#                   ~$6,750/t Cu : full-greenfield mine demanding full
#                                  capex payback over 25-yr life
#                 Override `ore_purchase_per_t_cu` per-deposit when modelling
#                 a specific supplier contract.
#
# Effective $/t-ore by grade (illustrative at default $12,500/t Cu):
#   0.4% Cu : $10 + 0.004 * 6000 = $34/t ore
#   0.8% Cu : $10 + 0.008 * 6000 = $58/t ore
#   1.0% Cu : $10 + 0.010 * 6000 = $70/t ore
#   2.0% Cu : $10 + 0.020 * 6000 = $130/t ore
DEFAULT_ORE_PURCHASE_BASE_PER_T = 10.0     # USD / t ore (gate handling + short-haul)
DEFAULT_ORE_PURCHASE_PER_T_CU   = 6000.0   # USD / t Cu in ore (Wood Mac AISC + capex amort + 15% margin)
# Tailings adverse-site multiplier (1.0 = favorable)
DEFAULT_TAILINGS_ADVERSE   =      1.0
# Hydromet reagent unit rates.
# - Lime $200/t: USGS Mineral Commodity Summaries 2025, Lime chapter — 2024
#   US average price (16 Mt produced @ $3.2B = $200/t implied avg).
#   https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-lime.pdf
# - Sulfuric acid $150/t: 2025 contract prices for smelter-byproduct H2SO4
#   delivered to Chilean Cu hydromet plants vary $50-200/t depending on
#   distance to smelter. Mid-range placeholder, NOT a hard citation —
#   verify against the destination plant's logistics before external use.
# - Oxygen $100/t: industrial-gas cryogenic ASU vendor quote band $80-120/t
#   (Air Liquide / Linde long-term supply). PLACEHOLDER.
# - SX organic $7/L: extractant blend (e.g. LIX 84-IC at ~$8/L; Acorga M5640
#   at ~$6/L) per BASF / Cytec product datasheets. PLACEHOLDER, refit per
#   actual extractant + diluent ratio.
DEFAULT_LIME_COST_PER_T    =    200.0   # USD / t Ca(OH)2 — USGS MCS 2025
DEFAULT_ACID_COST_PER_T    =    150.0   # USD / t H2SO4 — placeholder
DEFAULT_O2_COST_PER_T      =    100.0   # USD / t O2 — placeholder
DEFAULT_ORGANIC_COST_PER_L =      7.0   # USD / L SX extractant — placeholder
# Cathode revenue mode — when ew_enabled=True, Cu sells as cathode and
# bypasses TC/RC. Payable fraction goes to 1.0; concentrate deductions skip.
DEFAULT_CATHODE_PAYABLE    =      1.000
# Cashflow horizon
DEFAULT_PLANT_LIFE_YEARS   =     25
DEFAULT_DISCOUNT_RATE      =      0.08


# ---------------------------------------------------------------- helpers
def perry_cost(basis_usd: float, basis_cap: float, exp: float,
               actual_cap: float, inflation: float = 1.0) -> float:
    """Power-law capex helper. With basis_usd already in target-year USD,
    pass inflation=1.0 (default). For Perry-era basis values (M&S=1000),
    pass inflation=M_S_INDEX_2025 / M_S_INDEX_PERRYS."""
    if actual_cap <= 0:
        return 0.0
    return basis_usd * (actual_cap / basis_cap) ** exp * inflation


def select_basis(bases, actual_cap: float):
    """Pick the basis whose basis_capacity most closely brackets actual_cap.
    Matches the BlueShift sheet's two-tier small/large selection."""
    candidates = sorted(bases, key=lambda b: b[1])
    chosen = candidates[0]
    for b in candidates:
        if actual_cap >= b[1]:
            chosen = b
    return chosen


def parallel_cost(basis_usd: float, basis_cap: float, exp: float,
                  required_cap: float, max_single_cap: float,
                  inflation: float = 1.0) -> tuple:
    """Parallel-train capex helper: when required_cap exceeds max_single_cap
    for one unit, split into n parallel units of equal size and sum their
    power-law cost.

    Returns (total_cost_usd, n_units). For n=1 this matches perry_cost().
    Per-unit cost rises with (req/n)^exp; total cost rises near-linearly in n
    once n >= 2, capturing the diseconomy of running multiple smaller trains
    versus one impossibly-large mega-unit.

    Source: references/equipment_basis_2026.md §4."""
    import math
    if required_cap <= 0 or max_single_cap <= 0:
        return 0.0, 0
    n_units = max(1, math.ceil(required_cap / max_single_cap))
    per_unit_cap = required_cap / n_units
    cost_per_unit = perry_cost(basis_usd, basis_cap, exp, per_unit_cap, inflation)
    return cost_per_unit * n_units, n_units


def _ohara_1988(basis: float, exp: float, ore_tpd_short: float,
                m_s_target: float = M_S_INDEX_2025) -> float:
    """O'Hara 1992 power-law (1988 USD basis) inflated to target M&S year."""
    if ore_tpd_short <= 0:
        return 0.0
    return basis * ore_tpd_short ** exp * m_s_target / M_S_INDEX_1988


def tailings_storage_capex(ore_tph_metric: float,
                           adverse_factor: float = 1.0,
                           hours_per_day: float = 24.0,
                           m_s_target: float = M_S_INDEX_2025) -> float:
    """Initial tailings storage capex via O'Hara 1992 Eq 6.3.145
    ($20,000 T^0.5, 1988 USD). Minimum x adverse_factor; O'Hara notes
    realistic projects often run 2-5x. Integrated installed — no Lang."""
    ore_tpd_short = ore_tph_metric * hours_per_day * SHORT_PER_METRIC_TON
    base = _ohara_1988(TAILINGS_MIN_USD_1988, TAILINGS_EXP,
                       ore_tpd_short, m_s_target)
    return base * adverse_factor


def _slurry_m3_per_h(stream) -> float:
    """Volumetric flow m^3/h from a Stream, assuming rho_s=2.7, rho_w=1.0."""
    return stream.solids_tph / 2.7 + stream.water_tph


def select_flot_basis(cell_volume_m3: float):
    """Phase 2A: pick Arfania (small-cell) vs vendor-anchored large-cell
    basis based on per-cell volume. Crossover at FLOT_BASIS_CROSSOVER_M3."""
    if cell_volume_m3 > FLOT_BASIS_CROSSOVER_M3:
        return FLOTATION_CELL_BASIS_LARGE
    return FLOTATION_CELL_BASIS


def tails_thickener_diameter_m(tails_tph: float,
                               unit_area_t_per_m2_per_h: float = 0.30) -> float:
    """Sizing for tailings thickener from solids loading.
    Cu rougher tails typical unit area ~0.25-0.40 t/m^2/h (Wills 7th ed.,
    Outotec Mineral Processing Handbook). Conservative midpoint 0.30."""
    if tails_tph <= 0 or unit_area_t_per_m2_per_h <= 0:
        return 0.0
    area_m2 = tails_tph / unit_area_t_per_m2_per_h
    import math
    return math.sqrt(4.0 * area_m2 / math.pi)


# ---------------------------------------------------------------- parameters
@dataclass
class TEAParams:
    """BlueShift TEA knobs. Defaults reference the DEFAULT_* constants in the
    COST BASES block at the top of this file — change a number there to move
    every TEAParams() instantiation in the codebase."""
    # --- commodity / smelter terms (concentrate sale)
    cu_price_per_t: float = DEFAULT_CU_PRICE_PER_T
    tc_per_t_conc: float = DEFAULT_TC_PER_T_CONC
    rc_per_t_cu: float = DEFAULT_RC_PER_T_CU
    payable_fraction: float = DEFAULT_PAYABLE_FRACTION
    deductible_units: float = DEFAULT_DEDUCTIBLE_UNITS
    # $/t-Cu smelter impurity deduction on concentrate sale. Defaults to 0
    # (clean concentrate); evaluate() resolves it from the concentrate As
    # grade via smelter_as_penalty_per_t_cu() when left at the default. Set
    # explicitly on TEAParams to bypass the As-schedule lookup.
    impurity_penalty_per_t_cu: float = DEFAULT_IMPURITY_PENALTY_PER_T_CU

    # --- byproduct credits (sulfide concentrate route only)
    mo_recovery:        float = DEFAULT_MO_RECOVERY
    mo_price_per_t:     float = DEFAULT_MO_PRICE_PER_T
    au_payable:         float = DEFAULT_AU_PAYABLE
    au_price_per_t:     float = DEFAULT_AU_PRICE_PER_T
    ag_payable:         float = DEFAULT_AG_PAYABLE
    ag_price_per_t:     float = DEFAULT_AG_PRICE_PER_T
    # Smelter min-grade payable schedule. Cu concentrate below ~18% Cu is
    # typically refused by modern smelters; 18-25% attracts progressive
    # penalties; above 25% gets full payable. Industry-standard merchant
    # concentrate spec; refit per destination smelter contract.
    smelter_min_grade: float = 0.18
    smelter_full_payable_grade: float = 0.25

    # --- BlueShift financial basis
    hours_per_year: float = DEFAULT_HOURS_PER_YEAR
    power_cost_per_kwh: float = DEFAULT_POWER_COST_PER_KWH
    water_cost_per_m3: float = DEFAULT_WATER_COST_PER_M3
    heat_cost_per_kwh: float = DEFAULT_HEAT_COST_PER_KWH
    om_opex_pct_capex: float = DEFAULT_OM_OPEX_PCT_CAPEX
    # Legacy single Lang factor — superseded by the route-specific pair below
    # inside evaluate(); kept for back-compat with callers that read it.
    lang_factor: float = DEFAULT_LANG_FACTOR
    lang_factor_sulfide: float = DEFAULT_LANG_FACTOR_SULFIDE
    lang_factor_hydromet: float = DEFAULT_LANG_FACTOR_HYDROMET
    # Balance-of-plant installed $/tpd of ore feed (flotation routes only).
    balance_of_plant_per_tpd: float = DEFAULT_BALANCE_OF_PLANT_PER_TPD
    m_s_inflation: float = M_S_INDEX_2025 / M_S_INDEX_PERRYS

    # --- reagent & consumable scalars
    reagent_cost_per_t_ore: float = DEFAULT_REAGENT_PER_T_ORE
    media_cost_per_kwh_grind: float = DEFAULT_MEDIA_PER_KWH_GRIND
    # --- hydromet reagent unit rates (active when leach/SX/EW/neutralization on)
    lime_cost_per_t: float = DEFAULT_LIME_COST_PER_T
    acid_cost_per_t: float = DEFAULT_ACID_COST_PER_T
    o2_cost_per_t: float = DEFAULT_O2_COST_PER_T
    organic_cost_per_l: float = DEFAULT_ORGANIC_COST_PER_L
    # --- cathode revenue mode (used when ew_enabled=True)
    cathode_payable_fraction: float = DEFAULT_CATHODE_PAYABLE

    # --- ore purchase (merchant-concentrator). Supplier's full project
    # cost (mine capex + mine opex + margin) is folded into per_t_cu;
    # concentrator only sees ore-gate price.
    # Total $/t-ore = ore_purchase_base_per_t + cu_grade * ore_purchase_per_t_cu.
    ore_purchase_base_per_t: float = DEFAULT_ORE_PURCHASE_BASE_PER_T
    ore_purchase_per_t_cu:   float = DEFAULT_ORE_PURCHASE_PER_T_CU

    # --- tailings adverse-condition multiplier on the O'Hara 6.3.145
    # minimum. 1.0 = "favorable site". Realistic projects often run 2-5x.
    tailings_adverse_factor: float = DEFAULT_TAILINGS_ADVERSE

    # --- cashflow horizon
    plant_life_years: int = DEFAULT_PLANT_LIFE_YEARS
    discount_rate: float = DEFAULT_DISCOUNT_RATE


# ---------------------------------------------------------------- IRR util
def _irr(capex0: float, fcf_annual: float, life: int,
         lo: float = -0.9, hi: float = 50.0, tol: float = 1e-6) -> float:
    """Bisection IRR for a flat-annuity cashflow; nan if not bracketed."""
    def _npv(r):
        return -capex0 + fcf_annual * sum(1.0 / (1.0 + r) ** t
                                          for t in range(1, life + 1))
    if _npv(lo) * _npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        v = _npv(mid)
        if abs(v) < tol:
            return mid
        if v * _npv(lo) < 0:
            hi = mid
        else:
            lo = mid
    return mid


# ---------------------------------------------------------------- main entry
def evaluate(sim_result: dict, tea: Optional[TEAParams] = None) -> dict:
    """Compute NPV and itemized cashflows for a simulator result.

    `sim_result` is the dict returned by `throughput.simulate_cu_sulfide`.
    """
    tea = tea or TEAParams()
    streams = sim_result["streams"]
    power = sim_result["power_kw"]
    hours = tea.hours_per_year

    # Resolve the smelter As penalty from the concentrate As grade via the
    # general schedule when the caller has left tea.impurity_penalty_per_t_cu
    # at the default 0.0. An explicit non-zero TEAParams value bypasses the
    # schedule (lets tests and contract overrides force a specific penalty
    # regardless of the simulator's concentrate As grade).
    if (tea.impurity_penalty_per_t_cu == DEFAULT_IMPURITY_PENALTY_PER_T_CU
            and "conc_arsenic_pct" in sim_result):
        tea = replace(
            tea,
            impurity_penalty_per_t_cu=smelter_as_penalty_per_t_cu(
                sim_result["conc_arsenic_pct"], tea.cu_price_per_t),
        )

    # Topology booleans gate stage capex. If sim_result lacks "topology"
    # (older callers / tests), default to all-on.
    topo = sim_result.get("topology")
    bm_on = getattr(topo, "ball_mill_enabled", True)
    cyc_on = getattr(topo, "cyclone_enabled", True)
    thick_on = getattr(topo, "thickener_enabled", True)
    filt_on = getattr(topo, "filter_enabled", True)
    screen_on = getattr(topo, "screen_enabled", False)
    regrind_on = getattr(topo, "regrind_enabled", False)
    cleaner_on = getattr(topo, "cleaner_enabled", False)
    leach_on = getattr(topo, "leach_enabled", False)
    heap_leach_on = getattr(topo, "heap_leach_enabled", False)
    sx_on = getattr(topo, "sx_enabled", False)
    ew_on = getattr(topo, "ew_enabled", False)
    neut_on = getattr(topo, "neutralization_enabled", False)

    # ---------- revenue (concentrate sale, net of TC/RC)
    # Sulfide vs hydromet are mutually exclusive at the revenue level: in the
    # hydromet route the concentrate is consumed by the leach (it's the leach
    # feed), so there is no concentrate sale — Cu reports as cathode only.
    cake = streams["filter_cake"]
    conc_tph = cake.solids_tph
    cu_in_conc_tph = conc_tph * cake.cu_grade
    if leach_on or heap_leach_on:
        # Either hydromet route — Cu reports as cathode only, no concentrate
        # sale. (Concentrate-leach: flotation conc consumed by tank leach.
        # Heap-leach: no flotation at all, conc_tph is zero by construction.)
        deductible_tph = 0.0
        payable_cu_tph = 0.0
        gross_revenue = 0.0
        tc_cost = 0.0
        rc_cost = 0.0
        impurity_cost = 0.0
        net_concentrate_revenue = 0.0
    else:
        # Smelter min-grade payable schedule. Real Cu smelters (Boliden,
        # Aurubis, etc.) refuse very low-grade concentrate and apply a
        # progressive payable discount on borderline material:
        #   below smelter_min_grade  : refused (payable -> 0)
        #   between min and full     : payable ramps linearly 0 -> 1
        #   above smelter_full_grade : full payable, normal terms
        # This forces the optimizer to enable cleaner+regrind on low-feed-
        # grade ore, because a bare-rougher concentrate falls below spec.
        if cake.cu_grade < tea.smelter_min_grade:
            grade_payable_factor = 0.0
        elif cake.cu_grade >= tea.smelter_full_payable_grade:
            grade_payable_factor = 1.0
        else:
            grade_payable_factor = ((cake.cu_grade - tea.smelter_min_grade)
                                    / (tea.smelter_full_payable_grade
                                       - tea.smelter_min_grade))
        deductible_tph = tea.deductible_units * conc_tph
        payable_cu_tph = (max(0.0, cu_in_conc_tph - deductible_tph)
                          * tea.payable_fraction * grade_payable_factor)
        gross_revenue = payable_cu_tph * tea.cu_price_per_t * hours
        tc_cost = conc_tph * tea.tc_per_t_conc * hours
        rc_cost = payable_cu_tph * tea.rc_per_t_cu * hours
        impurity_cost = payable_cu_tph * tea.impurity_penalty_per_t_cu * hours
        net_concentrate_revenue = (gross_revenue - tc_cost - rc_cost
                                   - impurity_cost)

    # Cathode revenue (when EW is on): full Cu price, no TC/RC, payable
    # fraction is essentially 1.0. Simulator reports cathode_tph in balance.
    cathode_tph = sim_result.get("balance", {}).get("cu_in_cathode_tph", 0.0)
    cathode_revenue = (cathode_tph * tea.cathode_payable_fraction
                       * tea.cu_price_per_t * hours)

    # Byproduct credits — Mo/Au/Ag report to Cu concentrate. Credited only
    # on the sulfide route (concentrate sold to smelter). On hydromet
    # routes the byproducts go to leach residue / raffinate where
    # commercial recovery is uncommon; on heap leach the gangue is never
    # liberated to release them. Both branches return zero. Override
    # per-deposit by setting CuSulfideParams.{mo,au,ag}_grade or by
    # changing the recovery / price knobs on TEAParams.
    byp = sim_result.get("byproducts", {})
    if (leach_on or heap_leach_on) or conc_tph <= 0.0:
        mo_revenue = 0.0
        au_revenue = 0.0
        ag_revenue = 0.0
    else:
        ore_feed_tph = streams["ore"].solids_tph
        mo_tph = ore_feed_tph * byp.get("mo_ppm", 0.0) * 1e-6 * tea.mo_recovery
        au_tph = ore_feed_tph * byp.get("au_g_per_t", 0.0) * 1e-6 * tea.au_payable
        ag_tph = ore_feed_tph * byp.get("ag_g_per_t", 0.0) * 1e-6 * tea.ag_payable
        mo_revenue = mo_tph * tea.mo_price_per_t * hours
        au_revenue = au_tph * tea.au_price_per_t * hours
        ag_revenue = ag_tph * tea.ag_price_per_t * hours
    byproduct_revenue = mo_revenue + au_revenue + ag_revenue

    net_revenue = net_concentrate_revenue + cathode_revenue + byproduct_revenue

    # ---------- opex driven by simulator
    ore_tph = streams["ore"].solids_tph
    sag_kw = power.get("sag_kw", 0.0)
    bm_kw = power.get("bm_kw", 0.0)
    crusher_kw = power.get("crusher_kw", 0.0)
    flot_kw = power.get("flotation_kw", 0.0)
    regrind_kw = power.get("regrind_kw", 0.0)
    grinding_kw = sag_kw + bm_kw + regrind_kw
    total_kw = sum(power.values())

    process_water_tph = (streams["cyc_of"].water_tph
                        if "cyc_of" in streams else 0.0)
    water_m3_per_h = process_water_tph
    water_cost = water_m3_per_h * hours * tea.water_cost_per_m3

    power_cost = total_kw * hours * tea.power_cost_per_kwh
    reagent_cost = ore_tph * hours * tea.reagent_cost_per_t_ore
    media_cost = grinding_kw * hours * tea.media_cost_per_kwh_grind
    ore_cu_grade = streams["ore"].cu_grade
    ore_purchase_cost = ore_tph * hours * (
        tea.ore_purchase_base_per_t
        + ore_cu_grade * tea.ore_purchase_per_t_cu
    )

    # ---------- per-stage equipment capex (Perry / Mular, 2025 USD)
    # Crusher: Perry T9-50 jaw two-tier, sized on shaft kW.
    cb = select_basis(CRUSHER_BASES, crusher_kw)
    crusher_capex = perry_cost(cb[0], cb[1], cb[2], crusher_kw)
    mb = select_basis(MOTOR_BASES, crusher_kw)
    crusher_motor_capex = perry_cost(mb[0], mb[1], mb[2], crusher_kw)

    # SAG mill: Mular & Poulter on installed kW. Parallelizes above
    # MAX_SINGLE_SAG_KW (28 MW Cobre Panama ceiling).
    sb = SAG_MILL_BASIS
    sag_capex, n_sag = parallel_cost(sb[0], sb[1], sb[2], sag_kw,
                                      MAX_SINGLE_SAG_KW)

    # Ball mill: Mular & Poulter on installed kW. Zero when BM is off
    # (bm_kw will be 0 from simulator), so the topology toggle bites here.
    # Parallelizes above MAX_SINGLE_BM_KW (22 MW Cobre Panama ceiling).
    bb = BM_MILL_BASIS
    if bm_on:
        bm_capex, n_bm = parallel_cost(bb[0], bb[1], bb[2], bm_kw,
                                        MAX_SINGLE_BM_KW)
    else:
        bm_capex, n_bm = 0.0, 0

    # Hydrocyclone: Perry T9-50 on feed slurry m^3/h. Cyclone is on the
    # closed-circuit BM discharge; size off bm_discharge stream if present.
    # Parallelizes above MAX_SINGLE_CYCLONE_M3PH (~400 m^3/h, Krebs gMAX 840 mm).
    cyclone_feed_m3_per_h = 0.0
    if cyc_on and "bm_discharge" in streams:
        cyclone_feed_m3_per_h = _slurry_m3_per_h(streams["bm_discharge"])
    hb = HYDROCYCLONE_BASIS
    if cyc_on:
        cyclone_capex, n_cyclone = parallel_cost(
            hb[0], hb[1], hb[2], cyclone_feed_m3_per_h,
            MAX_SINGLE_CYCLONE_M3PH)
    else:
        cyclone_capex, n_cyclone = 0.0, 0

    # Flotation bank: Arfania et al. 2017 per-cell capex * num_cells.
    # Sizing is single-cell volume m^3 (NOT bank total) — the paper's
    # regression is fit per-machine on 10 industrial Standard Flotation cells.
    flot_sim = sim_result.get("flotation", {})
    flot_volume_m3 = flot_sim.get("total_volume_m3", 0.0)
    cell_volume_m3 = flot_sim.get("cell_volume_m3", 0.0)
    num_cells = flot_sim.get("num_cells", 0)
    fb = select_flot_basis(cell_volume_m3)
    cost_per_cell = perry_cost(fb[0], fb[1], fb[2], cell_volume_m3)
    flotation_capex = cost_per_cell * num_cells

    # Concentrate thickener: Parkinson-Mular on diameter (when on).
    thick_diam = sim_result.get("conc_thickener", {}).get("diameter_m", 0.0)
    tb = THICKENER_BASIS
    thickener_capex = (perry_cost(tb[0], tb[1], tb[2], thick_diam)
                       if thick_on else 0.0)

    # Phase 2A: Tailings thickener — always sized on whatever stream feeds the
    # final tails (rougher_tails, combined_tails, or fall back to ore_tph as
    # a proxy). Sizing rule: 0.30 t/m^2/h unit area (Wills 7th ed.). Same
    # Parkinson-Mular basis as the conc thickener but on a much larger
    # diameter (35-45 m typical for 50 ktpd Cu). Always on — no toggle.
    if "combined_tails" in streams:
        tails_tph = streams["combined_tails"].solids_tph
    elif "rougher_tails" in streams:
        tails_tph = streams["rougher_tails"].solids_tph
    else:
        tails_tph = max(0.0, ore_tph - conc_tph)
    tails_thick_diam = tails_thickener_diameter_m(tails_tph)
    tails_thickener_capex = perry_cost(tb[0], tb[1], tb[2], tails_thick_diam)

    # Belt filter: Perry T9-50 vacuum belt on m^2 (when on).
    filter_area = sim_result.get("belt_filter", {}).get("area_m2", 0.0)
    fbb = BELT_FILTER_BASIS
    filter_capex = (perry_cost(fbb[0], fbb[1], fbb[2], filter_area)
                    if filt_on else 0.0)

    # Vibrating screen: placeholder vendor-quote curve, sized on deck area.
    screen_area_m2 = sim_result.get("screen", {}).get("area_m2", 0.0)
    sb_screen = SCREEN_BASIS
    screen_capex = (perry_cost(sb_screen[0], sb_screen[1], sb_screen[2],
                               screen_area_m2)
                    if screen_on else 0.0)

    # Regrind mill: Mular & Poulter on installed kW, scaled by the chosen
    # mill technology (ball / tower / IsaMill — see REGRIND_MILL_PROPS).
    from .throughput import REGRIND_MILL_PROPS as _RMP
    rgb = REGRIND_MILL_BASIS
    regrind_mill_type = getattr(topo, "regrind_mill_type", "ball")
    regrind_capex_factor = float(_RMP.get(regrind_mill_type,
                                           _RMP["ball"])["capex_factor"])
    regrind_capex = (perry_cost(rgb[0], rgb[1], rgb[2], regrind_kw)
                     * regrind_capex_factor
                     if regrind_on else 0.0)

    # Cleaner flotation bank: Arfania per-cell * num_cells, selected on the
    # cleaner cell volume instead of reusing the rougher-cell basis.
    cleaner_diag = sim_result.get("cleaner", {})
    cleaner_cell_vol = cleaner_diag.get("cell_volume_m3", 0.0)
    cleaner_n_cells = cleaner_diag.get("num_cells", 0)
    cleaner_fb = select_flot_basis(cleaner_cell_vol)
    cleaner_capex = (perry_cost(cleaner_fb[0], cleaner_fb[1], cleaner_fb[2],
                                cleaner_cell_vol)
                     * cleaner_n_cells if cleaner_on else 0.0)

    # Leach tank: atmospheric tank power-law on total tank volume.
    leach_diag = sim_result.get("leach", {})
    leach_volume = leach_diag.get("tank_volume_total_m3", 0.0)
    ltb = LEACH_TANK_BASIS
    leach_capex = (perry_cost(ltb[0], ltb[1], ltb[2], leach_volume)
                   if leach_on else 0.0)

    # SX mixer-settler: atmospheric tank basis on SX tank volume.
    sx_diag = sim_result.get("sx", {})
    sx_volume = sx_diag.get("tank_volume_m3", 0.0)
    sxb = SX_MIXER_SETTLER_BASIS
    sx_capex = (perry_cost(sxb[0], sxb[1], sxb[2], sx_volume)
                if sx_on else 0.0)

    # EW cell: Perry-style power-law on total electrode area.
    ew_diag = sim_result.get("ew", {})
    ew_area = ew_diag.get("total_electrode_area_m2", 0.0)
    ewb = EW_CELL_BASIS
    ew_capex = (perry_cost(ewb[0], ewb[1], ewb[2], ew_area)
                if ew_on else 0.0)

    # Neutralization tanks: atmospheric tank basis on total volume.
    neut_diag = sim_result.get("neutralization", {})
    neut_volume = neut_diag.get("tank_volume_total_m3", 0.0)
    nb = NEUTRALIZATION_TANK_BASIS
    neut_capex = (perry_cost(nb[0], nb[1], nb[2], neut_volume)
                  if neut_on else 0.0)

    # Heap-leach pad: Mular 2002-anchored civil works on annual ore tonnage.
    # Already integrated installed cost — bypasses Lang factor.
    heap_diag = sim_result.get("heap_leach", {})
    if heap_leach_on:
        annual_ore_tonnes = ore_tph * hours
        hpb = HEAP_PAD_BASIS
        heap_pad_capex = perry_cost(hpb[0], hpb[1], hpb[2], annual_ore_tonnes)
    else:
        heap_pad_capex = 0.0

    equipment_capex = (crusher_capex + crusher_motor_capex
                       + sag_capex + bm_capex
                       + cyclone_capex + flotation_capex
                       + thickener_capex + tails_thickener_capex
                       + filter_capex
                       + screen_capex + regrind_capex + cleaner_capex
                       + leach_capex + sx_capex + ew_capex + neut_capex)

    # ---------- installed capex
    # Per-stage equipment lines x Lang for piping/E&I/control/installation.
    # Tailings (O'Hara Eq 6.3.145), heap pad (Mular 2002 civil works) and the
    # balance-of-plant line are already integrated installed and bypass Lang.
    tailings_capex = tailings_storage_capex(
        ore_tph, adverse_factor=tea.tailings_adverse_factor)

    # Phase 2C — route-dependent Lang factor. A greenfield flotation
    # concentrator carries a heavier installation multiplier than a heap/SX-EW
    # operation (Mular et al. 2002 ch.1; Sayadi et al. 2014).
    flotation_on = getattr(topo, "flotation_enabled", True)
    lang = (tea.lang_factor_sulfide if flotation_on
            else tea.lang_factor_hydromet)

    # Phase 2B — balance-of-plant (flotation routes only; see constant note).
    balance_of_plant_capex = (ore_tph * 24.0 * tea.balance_of_plant_per_tpd
                              if flotation_on else 0.0)

    installed_capex = (equipment_capex * lang
                       + tailings_capex + heap_pad_capex
                       + balance_of_plant_capex)
    total_capex = installed_capex   # BlueShift: no separate EPCM/contingency

    # ---------- hydromet opex (active only when respective stages on)
    # Acid demand = leach acid_kg_per_h driven by Cu leached.
    leach_acid_kg_per_h = leach_diag.get("acid_kg_per_h", 0.0)
    leach_acid_cost = (leach_acid_kg_per_h / 1000.0 * hours
                       * tea.acid_cost_per_t) if leach_on else 0.0
    # O2 demand: rough proxy of 1 kg O2 per kg Cu leached (stoichiometric
    # for chalcopyrite is ~1.5 kg O2 / kg Cu via 4Cu + O2 = 2Cu2O routes;
    # midpoint placeholder).
    cu_leached_tph = leach_diag.get("cu_leached_tph", 0.0)
    leach_o2_cost = (cu_leached_tph * hours * tea.o2_cost_per_t
                     if leach_on else 0.0)
    # Heap-leach acid: dominated by GANGUE dissolution (carbonates, mafics),
    # not Cu — typically 10-25 kg H2SO4 per tonne ore vs ~6 kg/kg Cu in
    # concentrate POX. Heap leach has no oxygen demand (atmospheric).
    heap_acid_kg_per_h = heap_diag.get("acid_kg_per_h", 0.0)
    heap_acid_cost = (heap_acid_kg_per_h / 1000.0 * hours
                      * tea.acid_cost_per_t) if heap_leach_on else 0.0
    # Lime demand from neutralization stage (sum of stage 1 + stage 2).
    lime_kg_per_h = neut_diag.get("total_lime_kg_per_h", 0.0)
    neut_lime_cost = (lime_kg_per_h / 1000.0 * hours
                      * tea.lime_cost_per_t) if neut_on else 0.0
    # SX organic make-up: very small, ~1% of inventory per year. Inventory
    # is approximated as 0.5 m^3 organic per 1 m^3 SX tank.
    sx_organic_inventory_l = sx_volume * 500.0   # 0.5 m^3 per m^3 tank
    sx_organic_cost = (sx_organic_inventory_l * 0.01 * tea.organic_cost_per_l
                       if sx_on else 0.0)

    # ---------- O&M opex (% of capex) and totals
    om_opex = total_capex * tea.om_opex_pct_capex
    hydromet_opex = (leach_acid_cost + leach_o2_cost + heap_acid_cost
                     + neut_lime_cost + sx_organic_cost)
    annual_opex = (power_cost + water_cost + reagent_cost + media_cost
                   + ore_purchase_cost + om_opex + hydromet_opex)

    # ---------- cashflow & NPV (pre-tax, BlueShift method)
    life = tea.plant_life_years
    fcf_annual = net_revenue - annual_opex
    r = tea.discount_rate
    pv_factor = sum(1.0 / (1.0 + r) ** t for t in range(1, life + 1))
    npv = -total_capex + fcf_annual * pv_factor
    irr = _irr(total_capex, fcf_annual, life)
    payback = total_capex / fcf_annual if fcf_annual > 0 else float("inf")

    return {
        "npv": npv,
        "irr": irr,
        "total_capex": total_capex,
        "annual_revenue": net_revenue,
        "annual_opex": annual_opex,
        "annual_fcf": fcf_annual,
        "payback_years": payback,
        "breakdown": {
            "gross_revenue": gross_revenue,
            "net_concentrate_revenue": net_concentrate_revenue,
            "cathode_revenue": cathode_revenue,
            "tc_cost": tc_cost,
            "rc_cost": rc_cost,
            "impurity_cost": impurity_cost,
            "byproduct_revenue": byproduct_revenue,
            "mo_revenue": mo_revenue,
            "au_revenue": au_revenue,
            "ag_revenue": ag_revenue,
            "payable_cu_tph": payable_cu_tph,
            "cathode_tph": cathode_tph,
            "conc_tph": conc_tph,
            "power_cost": power_cost,
            "water_cost": water_cost,
            "reagent_cost": reagent_cost,
            "media_cost": media_cost,
            "ore_purchase_cost": ore_purchase_cost,
            "leach_acid_cost": leach_acid_cost,
            "leach_o2_cost": leach_o2_cost,
            "neut_lime_cost": neut_lime_cost,
            "sx_organic_cost": sx_organic_cost,
            "hydromet_opex": hydromet_opex,
            "om_opex": om_opex,
            "crusher_capex": crusher_capex,
            "crusher_motor_capex": crusher_motor_capex,
            "sag_capex": sag_capex,
            "n_sag": n_sag,
            "bm_capex": bm_capex,
            "n_bm": n_bm,
            "cyclone_capex": cyclone_capex,
            "n_cyclone": n_cyclone,
            "flotation_capex": flotation_capex,
            "thickener_capex": thickener_capex,
            "tails_thickener_capex": tails_thickener_capex,
            "tails_thickener_diameter_m": tails_thick_diam,
            "filter_capex": filter_capex,
            "screen_capex": screen_capex,
            "regrind_capex": regrind_capex,
            "cleaner_capex": cleaner_capex,
            "leach_capex": leach_capex,
            "heap_pad_capex": heap_pad_capex,
            "heap_acid_cost": heap_acid_cost,
            "sx_capex": sx_capex,
            "ew_capex": ew_capex,
            "neut_capex": neut_capex,
            "tailings_capex": tailings_capex,
            "balance_of_plant_capex": balance_of_plant_capex,
            "lang_factor_applied": lang,
            "flot_volume_m3": flot_volume_m3,
            "flot_cell_volume_m3": cell_volume_m3,
            "flot_num_cells": num_cells,
            "cyclone_feed_m3_per_h": cyclone_feed_m3_per_h,
            "thickener_diameter_m": thick_diam,
            "filter_area_m2": filter_area,
            "equipment_capex": equipment_capex,
            "installed_capex": installed_capex,
        },
    }


def print_tea_report(tea_result: dict) -> None:
    print("=== TEA (per-stage Perry/Mular capex) ===")
    fmt = lambda x: f"{x:>18,.0f}" if abs(x) >= 1.0 else f"{x:>18.4f}"
    print(f"  {'NPV':<24s} {fmt(tea_result['npv'])} USD")
    irr = tea_result["irr"]
    irr_s = f"{irr*100:>17.2f} %" if irr == irr else f"{'nan':>17s} %"
    print(f"  {'IRR':<24s} {irr_s}")
    print(f"  {'Total capex (installed)':<24s} {fmt(tea_result['total_capex'])} USD")
    print(f"  {'Annual revenue (net)':<24s} {fmt(tea_result['annual_revenue'])} USD/yr")
    print(f"  {'Annual opex':<24s} {fmt(tea_result['annual_opex'])} USD/yr")
    print(f"  {'Annual FCF (pre-tax)':<24s} {fmt(tea_result['annual_fcf'])} USD/yr")
    print(f"  {'Payback (yr)':<24s} {tea_result['payback_years']:>17.2f}")
    print("  -- breakdown --")
    for k, v in tea_result["breakdown"].items():
        print(f"    {k:<25s} {fmt(v)}")
