"""Steady-state throughput simulator for a reference Cu sulfide flowsheet.

Topology: crusher -> SAG -> (sump) -> ball mill -> cyclone -> rougher flotation
-> concentrate thickener -> filter.

Grinding is a two-stage split:
  * SAG grinds fresh feed to an intermediate P80 at its own Ecs (kWh/t fresh).
  * Ball mill runs closed-circuit with the cyclone. Total BM Ecs comes from
    Bond's law inverted between SAG discharge P80 and target flotation P80
    (on a fresh-feed basis — correct on total energy regardless of CL).
  * Circulating load is an input; it sizes the BM/cyclone but does not change
    total grinding power.

Filter is approximated (Darcy placeholder pending dedicated module).
"""
from dataclasses import dataclass, field, replace
from typing import Optional
import math

from . import (crusher as cr, sag, hydrocyclone as hc, thickener as th,
               flotation as fl, filter as flt,
               screen as scr, leach as lch, heap_leach as hlc,
               ew as ew_mod, neutralization as neut)
from .flowsheet import (
    Stream, run_crusher, run_sag, run_hydrocyclone, run_thickener,
    run_flotation, run_filter,
)


# Regrind-mill technology factors used by the regrind block + tea capex line.
# Energy factor multiplies the ball-mill Bond Ecs at the regrind P80 target;
# capex factor multiplies the BM-mill basis (Mular & Poulter 2002) on
# installed kW. Citations: Mazzinghy et al 2014 "Vertical mill simulation
# applied to iron ores" (tower / Vertimill); Burford & Clark 2007 AusIMM
# "IsaMill ultra-fine grinding for fine-grind cleaner circuits"; Jankovic
# 2003 "A review of regrinding and fine grinding technology" (Min Eng).
# Tower mill: ~25% lower kWh/t at P80 ~30 µm vs ball; ~85% capex per kW.
# IsaMill: ~40% lower kWh/t at P80 <25 µm vs ball; ~140% capex per kW
# (specialized ceramic media + smaller installed base).
# Per-mineralogy properties driving route selection. Mineralogy is a
# first-class lever because the type of Cu mineral in the feed dictates
# whether the ore can flot at all (oxide cannot, sulfide can), how
# refractory it is to autoclave leach (chalcopyrite needs POX; chalcocite
# leaches readily in atmospheric/heap acid), and what fraction is amenable
# to acid heap leach (oxide and chalcocite high; chalcopyrite poor).
#
# Fields:
#   flot_R_max     : hard upper bound on rougher+scavenger Cu recovery.
#                    For oxide this is ~0 (no sulfide surface for xanthate
#                    collectors). Chalcopyrite/bornite ~0.90 (industry).
#                    Chalcocite ~0.85 (slightly lower kinetic constant per
#                    Wills 7th ed. Ch. 12.2 collector chemistry).
#   pox_max        : autoclave leach extraction ceiling (LeachParams.max_extraction).
#                    Chalcopyrite is the autoclave's primary use case; oxide
#                    has no sulfide to oxidize so POX is irrelevant (= 0).
#   heap_X         : whole-ore acid heap extraction (HeapLeachParams).
#                    Chalcocite and oxide both heap well; chalcopyrite is
#                    refractory in acid heap (~0.30 max even with bioleach).
#
# Sources:
#   Wills' Mineral Processing Technology 7th ed., Ch. 12.2 (Cu mineral
#   floatability), Ch. 15 (heap-leach amenability by mineralogy).
#   SME Mineral Processing Handbook (2019) Ch. 18 (Cu flotation collector
#   chemistry by mineral) and Ch. 21 (heap leach Cu mineralogy).
#   Chalcocite heap_X recentered to 0.72 — the mean of the two disclosed
#   chalcocite heap operations (BHP Spence 0.73, Lomas Bayas 0.70), inside the
#   0.70-0.85 chalcocite acid-heap range of Schlesinger et al., Extractive
#   Metallurgy of Copper 5th ed. Ch. 15. The prior 0.70 sat at the bottom of
#   the disclosed pair; 0.72 centers the model on both (no per-deposit knob).
MINERALOGY_PROPS: dict[str, dict[str, float]] = {
    "chalcopyrite":     {"flot_R_max": 0.90, "pox_max": 0.95, "heap_X": 0.30},
    "chalcocite":       {"flot_R_max": 0.85, "pox_max": 0.95, "heap_X": 0.72},
    "chalcocite/oxide": {"flot_R_max": 0.50, "pox_max": 0.85, "heap_X": 0.80},
    "oxide":            {"flot_R_max": 0.05, "pox_max": 0.00, "heap_X": 0.85},
    "bornite":          {"flot_R_max": 0.92, "pox_max": 0.95, "heap_X": 0.30},
    "IOCG":             {"flot_R_max": 0.90, "pox_max": 0.95, "heap_X": 0.30},
}


# Byproduct head grades by deposit class. Real plants disclose these per
# project; defaults below come from the public deposit-model literature
# and are used when CuSulfideParams.{mo,au,ag}_grade is left as None.
#
#   chalcopyrite porphyry — Sinclair (2007) USGS Open-File 2007-1321
#       median Cu-Mo porphyry: Mo 200-300 ppm, Au 0.05-0.15 g/t, Ag 1-3 g/t
#       (Cobre Panama / Cerro Verde / Antamina / QB2 disclosures bracket
#       this range).
#   bornite — usually intergrown with chalcopyrite in the porphyry-Cu
#       hypogene zone; same default grades.
#   chalcocite — supergene blanket. Mo and Au are essentially stripped
#       during the supergene oxidation cycle (Sillitoe 2010 Econ Geol
#       105:3-41; Sillitoe & Perelló 2005). Disclosed Spence / Mantos
#       Blancos / Mantoverde production reports book no Mo or Au credit
#       and only minor Ag. Defaults to zero — set per-project for the
#       rare hypogene chalcocite where Mo persists.
#   chalcocite/oxide and oxide — fully leached, byproducts gone. Zero.
#   IOCG — Olympic Dam-style: no Mo, but Au+Ag enriched (Cu-Au-Ag-U-REE
#       signature; BHP 2024 ASR discloses 0.5 g/t Au, 1.7 g/t Ag for OD).
#       U/REE outside concentrator scope.
BYPRODUCT_GRADES: dict[str, dict[str, float]] = {
    "chalcopyrite":     {"mo_ppm": 250.0, "au_g_per_t": 0.10, "ag_g_per_t": 2.0},
    "chalcocite":       {"mo_ppm":   0.0, "au_g_per_t": 0.00, "ag_g_per_t": 0.0},
    "chalcocite/oxide": {"mo_ppm":   0.0, "au_g_per_t": 0.00, "ag_g_per_t": 0.0},
    "oxide":            {"mo_ppm":   0.0, "au_g_per_t": 0.00, "ag_g_per_t": 0.0},
    "bornite":          {"mo_ppm": 250.0, "au_g_per_t": 0.10, "ag_g_per_t": 2.0},
    "IOCG":             {"mo_ppm":   0.0, "au_g_per_t": 0.50, "ag_g_per_t": 1.7},
}

# Default Cu-concentrate arsenic grade (percent) by mineralogy class. As is
# a disclosed ore property, not a tuning knob; tea.smelter_as_penalty_per_t_cu()
# converts it to a $/t-Cu smelter deduction. A per-project disclosed value on
# CuSulfideParams.conc_arsenic_pct overrides these class defaults. These class
# defaults are engineering-judgment central values within the literature ranges
# below — set the per-project field when a deposit's concentrate As is disclosed.
# Full notes in tea.py references/references.md (§Smelter arsenic-penalty schedule).
#   chalcopyrite / bornite : clean porphyry concentrate, ~0.05% As — well
#       under the 0.2% penalty-free limit. $0 penalty.
#   chalcocite : Andean supergene chalcocite carries the enargite-tennantite
#       association — concentrates from the belt (El Abra, Spence Hypogene,
#       Chuquicamata/Ministro Hales) are reported in the ~0.1-1% As range
#       (enargite Cu3AsS4; Lattanzi et al. 2008 enargite review; Filippou et al.
#       2007 As in Cu metallurgy). 0.55% is the mid-point of that range and
#       sits above the ~0.5% import rejection limit — why the belt built
#       POX/roasting rather than selling concentrate. Mid-point is judgment.
#   chalcocite/oxide : partly oxidised, lower sulfide-As load — 0.30%
#       (penalised but saleable). Interpolated engineering judgment, no
#       single disclosure; routing-insensitive (this class routes to heap).
#   oxide : does not flot (flot_R_max 0.05); concentrate As is moot — 0.0.
#   IOCG : Olympic Dam-class concentrate is low-As (U/F is the relevant
#       penalty, outside scope) — ~0.05%, $0.
AS_CONC_PCT_BY_MINERALOGY: dict[str, float] = {
    "chalcopyrite":     0.05,
    "chalcocite":       0.55,
    "chalcocite/oxide": 0.30,
    "oxide":            0.00,
    "bornite":          0.05,
    "IOCG":             0.05,
}


REGRIND_MILL_PROPS: dict[str, dict[str, float | str]] = {
    "ball":    {"energy_factor": 1.00, "capex_factor": 1.00,
                "label": "Ball-mill\nregrind"},
    "tower":   {"energy_factor": 0.75, "capex_factor": 0.85,
                "label": "Tower mill\n(Vertimill)"},
    "isamill": {"energy_factor": 0.60, "capex_factor": 1.40,
                "label": "IsaMill\n(stirred fine)"},
}


@dataclass
class CuSulfideParams:
    """All tunables for the reference Cu sulfide chain."""
    # Drives flotation/leach/heap extraction ceilings via MINERALOGY_PROPS.
    mineralogy: str = "chalcopyrite"
    crusher: cr.CrusherParams = field(
        default_factory=lambda: cr.CrusherParams(Wi=14.0))
    crusher_P80_um: float = 150_000.0  # 150 mm

    sag_params: sag.SAGParams = field(
        default_factory=lambda: sag.SAGParams(A=70.0, b=0.7, Wi=14.0,
                                              ta=0.5, PLI=7.0))
    sag_Ecs_kwh_t: float = 2.8   # ~2000 um P80 at Wi=14 from Bond inversion
    bm_params: sag.SAGParams = field(
        default_factory=lambda: sag.SAGParams(A=70.0, b=0.7, Wi=14.0,
                                              ta=0.5, PLI=7.0))
    target_flot_P80_um: float = 150.0  # flotation feed P80 = BM/cyclone O/F P80
    circulating_load: float = 2.5      # CL = U/F solids / fresh solids
    cyclone_geom: hc.CycloneGeom = field(
        default_factory=lambda: hc.CycloneGeom(
            Dc=50.0, Di=12.0, Dx=15.0, Du=8.0, h=120.0))
    cyclone_pressure_kpa: float = 70.0
    cyclone_feed_solids_pct: float = 55.0  # BM discharge / cyclone feed
    flot_feed_solids_pct: float = 30.0     # legacy; feed water now derives from flot_op.slurry_fraction (see _flot_dilution_water_per_t)
    flot_fit: fl.FittingParams = field(
        default_factory=lambda: fl.FittingParams(
            b=2.0, alpha=0.05, coverage=0.525, bubble_f=0.90,
            detach_f=0.7, bulk_zone=0.5))
    flot_cell: fl.CellParams = field(
        default_factory=lambda: fl.CellParams(
            specific_gravity=2.8, permitivity=6.9e-10, dielectric=78.5,
            pe=4.0, cell_area=25.0, bbl_ratio=1.2))
    # contact_angle is a fixed mineral-surface PROPERTY, not a tuned variable:
    # 55 deg is a representative advancing contact angle for well-collected
    # chalcopyrite with a xanthate collector (chalcopyrite-xanthate range
    # ~40-80 deg; Chander & Fuerstenau, and Fuerstenau & Pradip, flotation
    # surface chemistry). At this value the rougher recovery sits on the
    # mineralogy floatability plateau (~89%, against the 0.90 flot_R_max cap),
    # which matches disclosed Cu plant recoveries. It was removed from the DOE
    # in bound-desaturation Attempt 2 (a property cannot be "dialed").
    flot_op: fl.OperatingParams = field(
        default_factory=lambda: fl.OperatingParams(
            particle_size=75, contact_angle=55, bubble_z_pot=-35,
            particle_z_pot=-20, sp_power=1.2, sp_gas_rate=1.5,
            air_fraction=0.25, slurry_fraction=0.30, num_cells=6,
            ret_time=18, cell_volume=250, froth_height=0.15, frother_conc=50))
    # Calibrated so the rougher concentrate still needs cleaner upgrading.
    gangue_recovery: float = 0.10
    # Applied to rougher tails: R_total = R + (1-R)*R_scav.
    scavenger_recovery_on_tails: float = 0.50
    rougher_recovery_cap: float = 0.90
    # Liberation-limited recovery ceiling (grind-vs-recovery trade-off).
    # flot_R_max / rougher_recovery_cap is the ceiling at a FINE, well-liberated
    # grind. Coarse feed is composite/poorly-liberated and detaches more, so the
    # achievable ceiling falls above a reference P80 (sulfide flotation recovery
    # declines for coarse particles >~150 um; Trahar 1981, "A rational
    # interpretation of the role of particle size in flotation", Int. J. Miner.
    # Process. 8:289-327; Wills 7th ed. Ch. 12). This restores the grind-vs-
    # recovery trade-off so target_flot_P80 lands at an interior optimum rather
    # than the coarsest bound (bound-desaturation Attempt 3).
    liberation_p80_ref_um: float = 150.0          # full-liberation reference grind
    coarse_recovery_penalty_per_um: float = 0.0013  # ceiling fraction lost per um above ref (~6.5pp at 200 um)
    liberation_floor: float = 0.5                 # ceiling never below 50% of flot_R_max
    # Froth-flooding penalty (bound-desaturation Attempt 4).
    # Above a critical superficial gas velocity (Jg), froth becomes unstable:
    # bubble coalescence and flooding reduce froth recovery, so the achievable
    # Cu recovery rolls over and falls. Gorain, Franzidis & Manlapig (1998),
    # "Studies on impeller type, impeller speed and air flow rate in an
    # industrial scale flotation cell," Int. J. Miner. Process. 53:215-236
    # define Sb = 6·Jg/db; recovery rate constant peaks then falls with Jg
    # above the flooding threshold. For TankCell-class mechanical cells the
    # practical ceiling is ~1.8-2.2 cm/s (Wills 7th ed. Ch. 12). The penalty
    # factor is unity at/below jg_ref and falls linearly above it.
    # The upper bound of sp_gas_rate is widened to 2.2 cm/s so the rollover
    # region [1.6, 2.2] cm/s is inside the search space; the validated
    # operating point (1.5 cm/s) sits just below jg_ref and is unchanged.
    froth_flood_jg_ref: float = 1.6    # cm/s onset of froth instability
    froth_flood_slope: float = 0.25    # fraction of cap lost per cm/s above ref
    froth_flood_floor: float = 0.70    # cap never below 70% due to flooding alone
    # Turbulent-detachment penalty on sp_power (bound-desaturation Attempt 5).
    # At very high cell-specific power (ε > ~1.0 kW/m³) turbulent eddies exceed
    # the inertial detachment threshold, stripping collected particles from
    # bubble surfaces. Schubert (1999), "Nanobubbles, hydrophobicity, heterocoagulation
    # and hydrodynamics in flotation", Int. J. Miner. Process. 56:257-273;
    # Pyke, Fornasiero & Ralston (2003), J. Colloid Interface Sci. 265:141-151.
    # Below sp_power_ref: factor = 1.0 (no penalty). Above: cap multiplier falls.
    sp_power_ref: float = 1.0       # kW/m³ — onset of significant detachment
    sp_power_detach_slope: float = 0.35  # cap fraction lost per kW/m³ above ref
    sp_power_detach_floor: float = 0.90  # floor (never below 90% of cap from this)
    # CL-sharpness coupling (bound-desaturation Attempt 6).
    # Higher circulating load → more cyclone passes per unit time → sharper
    # size classification → finer, more uniform flotation feed → better
    # liberation at same target P80. Napier-Munn et al. (1996), "Mineral
    # Comminution Circuits," JKMRC Monograph, Ch. 5.
    # circulating_load removed from the DOE (Attempt 6): CL affects BM feed
    # mass (and thus power) but the flotation-feed F80 is set by target_flot_P80_um
    # independently of CL in the current model. The classification-sharpness
    # benefit (higher CL → narrower PSD → better liberation) cannot be
    # represented without modifying the cyclone model or the kinetic rate
    # constant — both off-limits. Fixed at 2.5 (industry standard for a
    # BM-cyclone circuit; Wills 7th ed. Ch. 5).
    # (Unused params below retained for documentation — removed from optimizer).
    flot_size_from_grind: bool = True  # override flot_op.particle_size with cyc O/F F80
    flot_autosize: bool = True
    flot_residence_time_min: float = 18.0
    flot_cell_volume_m3: float = 500.0   # discrete equipment spec, not optimized
    flot_froth_factor: float = 1.2
    conc_thickener: th.ThickenerParams = field(
        default_factory=lambda: th.ThickenerParams(
            vTF=8.0, rF=0.4, n=4.6))
    conc_uf_pct: float = 65.0
    # Filter alpha/r defaults are placeholders until leaf-test values exist.
    belt_filter: flt.BeltFilterParams = field(
        default_factory=lambda: flt.BeltFilterParams())
    filter_cake_moisture_pct: float = 9.0  # spec; Perry has no kinetics for this
    mill_water_ratio: float = 0.33       # ~75% solids into SAG
    cyclone_water_ratio: float = 2.3     # ~30% solids into flot

    # Optional pre-SAG screen: undersize bypasses SAG and feeds the BM circuit.
    screen_factors: scr.ScreenFactors = field(
        default_factory=scr.ScreenFactors)
    screen_geom: scr.ScreenGeom = field(
        default_factory=scr.ScreenGeom)
    screen_undersize_fraction: float = 0.30   # fraction finer than deck aperture
    screen_aperture_um: float = 10_000.0      # 10 mm pre-SAG screen deck

    # Regrind mill technology is selected via Topology.regrind_mill_type.
    regrind_target_P80_um: float = 30.0
    regrind_Wi: float = 14.0
    # Cleaner uses rougher kinetics with cleaner-regime bubbles and froth.
    cleaner_flot_fit: fl.FittingParams = field(
        default_factory=lambda: fl.FittingParams(
            b=2.0, alpha=0.05, coverage=0.525, bubble_f=0.60,
            detach_f=0.7, bulk_zone=0.5))
    cleaner_froth_height_m: float = 0.08
    # Empirical liberation correction applied on top of cleaner kinetics.
    cleaner_liberation_P80_slope: float = 0.0020   # multiplier shift per um
    # Cleaner bank retunes gangue mass pull to target a saleable merchant
    # concentrate grade; the solved bank R_g is physically clamped below.
    target_cleaner_conc_grade_pct: float = 26.0
    cleaner_residence_time_min: float = 12.0
    cleaner_cell_volume_m3: float = 50.0   # cleaner cells smaller than rougher

    leach_params: lch.LeachParams = field(default_factory=lch.LeachParams)
    leach_tails_cu_grade_floor: float = 0.001  # gangue Cu still in residue
    ew_params: ew_mod.EWParams = field(default_factory=ew_mod.EWParams)
    neut_params: neut.NeutralizationParams = field(
        default_factory=neut.NeutralizationParams)
    heap_leach_params: hlc.HeapLeachParams = field(
        default_factory=hlc.HeapLeachParams)
    # Order-of-magnitude neutralization loadings; refit per ore body.
    leach_fe_kg_per_t_ore: float = 2.0
    leach_as_kg_per_t_ore: float = 0.05

    # Byproduct head grades. None falls back to BYPRODUCT_GRADES[mineralogy].
    # Set explicitly per project to use disclosed reserve-statement values.
    mo_grade_ppm: Optional[float] = None
    au_g_per_t:   Optional[float] = None
    ag_g_per_t:   Optional[float] = None

    # Disclosed Cu-concentrate arsenic grade (percent). None falls back to
    # AS_CONC_PCT_BY_MINERALOGY[mineralogy]. Drives the smelter As penalty.
    conc_arsenic_pct: Optional[float] = None


def _feed_stream(tph: float, cu_grade_frac: float,
                 ore_F80_um: float = 300_000.0) -> Stream:
    return Stream(solids_tph=tph, water_tph=0.0,
                  F80_um=ore_F80_um, cu_grade=cu_grade_frac)


def _run_heap_leach_route(ore: Stream, crushed: Stream,
                          p: "CuSulfideParams",
                          topo: "Topology",
                          out: dict, power: dict) -> dict:
    """Whole-ore heap-leach branch. Crusher feeds the heap pad directly;
    no SAG / BM / cyclone / flotation / cleaner / thickener / filter.
    Cathode product flows from the leach -> SX -> EW chain on the heap PLS.

    Returns the same dict schema as the main simulator path so downstream
    consumers (`tea.py`, `visualize.py`) need no special-casing.
    """
    # Heap-leach Cu extraction + acid demand on whole crushed ore
    heap_diag = hlc.size_heap(crushed.solids_tph, crushed.cu_grade,
                              p.heap_leach_params)
    cu_in_ore_tph = crushed.solids_tph * crushed.cu_grade
    cu_leached_tph = heap_diag["cu_leached_tph"]
    pls_m3_per_h = heap_diag["pls_m3_per_h"]
    cu_in_pls_gpl = (cu_leached_tph * 1_000_000.0
                     / max(pls_m3_per_h, 1e-9)) / 1000.0  # g/L
    out["heap_pls"] = Stream(solids_tph=0.0,
                             water_tph=pls_m3_per_h,
                             cu_grade=0.0,
                             cu_aq_gpl=cu_in_pls_gpl)
    # Heap residue stays on the pad — accounted as final tails for mass balance
    cu_in_residue_tph = cu_in_ore_tph - cu_leached_tph
    residue_grade = (cu_in_residue_tph / max(crushed.solids_tph, 1e-9)
                     if crushed.solids_tph > 0 else 0.0)
    tails = Stream(solids_tph=crushed.solids_tph, water_tph=0.0,
                   F80_um=crushed.F80_um, cu_grade=residue_grade)
    out["heap_residue"] = tails
    # No flotation concentrate on this route — provide an empty filter_cake
    # so tea.py's revenue path can read a zero-tonnage concentrate without
    # special-casing the route.
    empty_conc = Stream(solids_tph=0.0, water_tph=0.0, cu_grade=0.0,
                        F80_um=crushed.F80_um)
    out["filter_cake"] = empty_conc
    out["rougher_conc"] = empty_conc
    out["rougher_tails"] = tails
    # Set zero power for skipped stages so tea.py kW lookups stay numeric
    power["sag_kw"] = 0.0
    power["bm_kw"] = 0.0
    power["flotation_kw"] = 0.0

    # SX + EW + neutralization on the heap PLS
    sx_diag = {"note": "sx disabled"}
    ew_diag = {"note": "ew disabled"}
    neut_diag = {"note": "neutralization disabled"}
    cathode_tph = 0.0
    cu_to_sx_raffinate_tph = 0.0
    cu_to_ew_spent_tph = 0.0
    if topo.sx_enabled:
        # Closed-circuit SX-EW recycle closure (see _run_concentrate_hydromet
        # for the full citation): heap raffinate recycles back to the leach
        # pad and EW spent electrolyte recycles to SX strip, so steady-state
        # PLS-to-cathode recovery is set by the ~3% electrolyte bleed, not
        # single-pass SX/EW efficiency. Wills & Finch 8th ed. Ch. 12.4 /
        # Ch. 15; Schlesinger et al. 5th ed. Ch. 17.
        sx_pass_eff, ew_pass_eff, electrolyte_bleed = 0.95, 0.95, 0.03
        circuit_pass_eff = sx_pass_eff * ew_pass_eff
        sx_ew_recovery = circuit_pass_eff / (
            1.0 - (1.0 - circuit_pass_eff) * (1.0 - electrolyte_bleed))
        cu_to_organic_tph = cu_leached_tph * sx_ew_recovery
        cu_to_sx_raffinate_tph = cu_leached_tph - cu_to_organic_tph
        sx_diag = {"cu_transfer_eff": sx_pass_eff,
                   "sx_ew_circuit_recovery": sx_ew_recovery,
                   "electrolyte_bleed": electrolyte_bleed,
                   "cu_to_organic_tph": cu_to_organic_tph,
                   "cu_in_raffinate_tph": cu_to_sx_raffinate_tph,
                   "tank_volume_m3": pls_m3_per_h * 0.5}
        if topo.ew_enabled:
            cu_to_ew_spent_tph = 0.0
            ew_diag = ew_mod.size_ew(cu_to_organic_tph, p.ew_params)
            cathode_tph = ew_diag["cathode_cu_tph"]
            power["ew_kw"] = ew_diag["total_power_kw"]
            out["cathode"] = Stream(solids_tph=cathode_tph, water_tph=0.0,
                                    cu_grade=1.0)
    if topo.neutralization_enabled:
        acid_kg_per_h = heap_diag["acid_kg_per_h"]
        fe_kg = p.leach_fe_kg_per_t_ore * ore.solids_tph
        as_kg = p.leach_as_kg_per_t_ore * ore.solids_tph
        mso4_residual = {"Cu": 100.0, "Zn": 50.0}
        neut_diag = neut.size_neutralization(
            h2so4_kg_per_h=acid_kg_per_h,
            fe_kg_per_h=fe_kg, as_kg_per_h=as_kg,
            metals_sulfate_kg_per_h=mso4_residual,
            slurry_m3_per_h=pls_m3_per_h,
            p=p.neut_params)

    # Mass balance — Cu leaves as cathode + heap residue + SX/EW losses
    cu_in = ore.solids_tph * ore.cu_grade
    cu_out_total = (cathode_tph + cu_in_residue_tph
                    + cu_to_sx_raffinate_tph + cu_to_ew_spent_tph)
    mb = {
        "cu_in_tph": cu_in,
        "cu_in_conc_tph": 0.0,
        "cu_in_cathode_tph": cathode_tph,
        "cu_in_tails_tph": cu_in_residue_tph,
        "cu_lost_sx_raffinate_tph": cu_to_sx_raffinate_tph,
        "cu_lost_ew_spent_tph": cu_to_ew_spent_tph,
        "cu_closure_err": cu_in - cu_out_total,
        "overall_cu_recovery": cathode_tph / cu_in if cu_in > 0 else 0.0,
        "conc_grade_pct": 0.0,
        "total_power_kw": sum(power.values()),
    }
    return {"streams": out, "power_kw": power, "balance": mb,
            "cyclone": {"note": "cyclone disabled (heap-leach route)"},
            "flotation": {"note": "flotation disabled (heap-leach route)",
                          "total_volume_m3": 0.0, "num_cells": 0,
                          "cell_volume_m3": 0.0},
            "conc_thickener": {"note": "thickener disabled (heap-leach route)"},
            "belt_filter": {"note": "filter disabled (heap-leach route)"},
            "screen": {"note": "screen disabled (heap-leach route)"},
            "regrind": {"note": "regrind disabled (heap-leach route)"},
            "cleaner": {"note": "cleaner disabled (heap-leach route)"},
            "leach": {"note": "concentrate-leach disabled (heap-leach route)"},
            "heap_leach": heap_diag, "sx": sx_diag,
            "ew": ew_diag, "neutralization": neut_diag,
            "topology": topo, "mineralogy": p.mineralogy,
            "conc_arsenic_pct": _conc_arsenic_pct(p),
            "byproducts": _byproduct_grades(p)}


def _byproduct_grades(p: CuSulfideParams) -> dict[str, float]:
    """Return Mo/Au/Ag head grades for the simulator's mineralogy. Per-
    project overrides on CuSulfideParams take precedence over the
    BYPRODUCT_GRADES default table."""
    defaults = BYPRODUCT_GRADES.get(p.mineralogy, BYPRODUCT_GRADES["chalcopyrite"])
    return {
        "mo_ppm":     p.mo_grade_ppm if p.mo_grade_ppm is not None else defaults["mo_ppm"],
        "au_g_per_t": p.au_g_per_t if p.au_g_per_t   is not None else defaults["au_g_per_t"],
        "ag_g_per_t": p.ag_g_per_t if p.ag_g_per_t   is not None else defaults["ag_g_per_t"],
    }


def _conc_arsenic_pct(p: CuSulfideParams) -> float:
    """Return the Cu-concentrate As grade (percent) for the simulator's
    mineralogy. A per-project disclosed CuSulfideParams.conc_arsenic_pct
    overrides the AS_CONC_PCT_BY_MINERALOGY class default."""
    if p.conc_arsenic_pct is not None:
        return p.conc_arsenic_pct
    return AS_CONC_PCT_BY_MINERALOGY.get(p.mineralogy, 0.0)


def _apply_mineralogy(p: CuSulfideParams) -> CuSulfideParams:
    miner_props = MINERALOGY_PROPS.get(p.mineralogy,
                                       MINERALOGY_PROPS["chalcopyrite"])
    return replace(
        p,
        leach_params=replace(p.leach_params,
                             max_extraction=miner_props["pox_max"]),
        heap_leach_params=replace(p.heap_leach_params,
                                  extraction_fraction=miner_props["heap_X"]),
    )


def _run_crushing(feed_tph: float, feed_grade: float,
                  p: CuSulfideParams) -> tuple[dict, dict, Stream, Stream]:
    out: dict = {}
    power: dict = {}
    ore = _feed_stream(feed_tph, feed_grade)
    out["ore"] = ore
    crushed, kw = run_crusher(ore, p.crusher, p.crusher_P80_um)
    out["crushed"] = crushed
    power["crusher_kw"] = kw
    return out, power, ore, crushed


def _run_screen_and_sag(crushed: Stream, p: CuSulfideParams,
                        topo: "Topology", out: dict,
                        power: dict) -> tuple[Stream, float, float, float, dict]:
    if topo.screen_enabled:
        undersize_tph = crushed.solids_tph * p.screen_undersize_fraction
        oversize_tph = crushed.solids_tph - undersize_tph
        screen_diag = scr.size_screen(
            undersize_tph=undersize_tph, oversize_tph=oversize_tph,
            fac=p.screen_factors, geom=p.screen_geom)
        screen_diag["undersize_tph"] = undersize_tph
        screen_diag["oversize_tph"] = oversize_tph
        screen_diag["aperture_um"] = p.screen_aperture_um
    else:
        undersize_tph = 0.0
        oversize_tph = crushed.solids_tph
        screen_diag = {"note": "screen disabled"}

    sag_feed = Stream(solids_tph=oversize_tph,
                      water_tph=oversize_tph * p.mill_water_ratio,
                      F80_um=crushed.F80_um, cu_grade=crushed.cu_grade)
    sag_out, sag_kw = run_sag(sag_feed, p.sag_params, p.sag_Ecs_kwh_t)
    out["sag_discharge"] = sag_out
    power["sag_kw"] = sag_kw
    if topo.screen_enabled and oversize_tph > 0:
        bm_fresh_F80_um = (
            (oversize_tph * sag_out.F80_um
             + undersize_tph * p.screen_aperture_um)
            / crushed.solids_tph
        )
    else:
        bm_fresh_F80_um = sag_out.F80_um
    return sag_out, bm_fresh_F80_um, undersize_tph, oversize_tph, screen_diag


def _run_grinding(crushed: Stream, sag_out: Stream, bm_fresh_F80_um: float,
                  undersize_tph: float, oversize_tph: float,
                  p: CuSulfideParams, topo: "Topology",
                  out: dict, power: dict) -> tuple[Stream, dict]:
    fresh = crushed.solids_tph
    Wi_bm = p.bm_params.Wi

    if topo.ball_mill_enabled:
        if topo.screen_enabled and undersize_tph > 0:
            Ecs_oversize_total = 10.0 * Wi_bm * (
                1.0 / math.sqrt(p.target_flot_P80_um)
                - 1.0 / math.sqrt(crushed.F80_um))
            Ecs_oversize_bm = max(0.0, Ecs_oversize_total - p.sag_Ecs_kwh_t)
            Ecs_undersize_bm = 10.0 * Wi_bm * (
                1.0 / math.sqrt(p.target_flot_P80_um)
                - 1.0 / math.sqrt(p.screen_aperture_um))
            bm_kw = (oversize_tph * Ecs_oversize_bm
                     + undersize_tph * Ecs_undersize_bm)
        else:
            Ecs_total_fresh = 10.0 * Wi_bm * (
                1.0 / math.sqrt(p.target_flot_P80_um)
                - 1.0 / math.sqrt(crushed.F80_um)
            )
            Ecs_bm_fresh = max(0.0, Ecs_total_fresh - p.sag_Ecs_kwh_t)
            bm_kw = Ecs_bm_fresh * fresh
        power["bm_kw"] = bm_kw

        if topo.cyclone_enabled:
            CL = p.circulating_load
            uf_solids = fresh * CL
            bm_feed_solids = fresh + uf_solids
            bm_feed_F80 = (fresh * bm_fresh_F80_um
                           + uf_solids * p.target_flot_P80_um) / bm_feed_solids
            bm_feed_water = bm_feed_solids * (100.0 / p.cyclone_feed_solids_pct - 1.0)
            out["bm_feed"] = Stream(solids_tph=bm_feed_solids,
                                    water_tph=bm_feed_water,
                                    F80_um=bm_feed_F80,
                                    cu_grade=sag_out.cu_grade)
            bm_discharge = Stream(solids_tph=bm_feed_solids,
                                  water_tph=bm_feed_water,
                                  F80_um=p.target_flot_P80_um,
                                  cu_grade=sag_out.cu_grade)
            out["bm_discharge"] = bm_discharge

            _, _, cyc_diag = run_hydrocyclone(bm_discharge, p.cyclone_geom,
                                              p.cyclone_pressure_kpa)
            Rv = cyc_diag["Rv"]
            uf_water = bm_feed_water * Rv
            of_water_raw = bm_feed_water - uf_water
            of_water_target = fresh * _flot_dilution_water_per_t(p)
            of_water = max(of_water_raw, of_water_target)
            out["cyc_uf"] = Stream(solids_tph=uf_solids, water_tph=uf_water,
                                   F80_um=bm_feed_F80,
                                   cu_grade=sag_out.cu_grade)
            flot_in = Stream(solids_tph=fresh, water_tph=of_water,
                             F80_um=p.target_flot_P80_um,
                             cu_grade=sag_out.cu_grade)
            out["cyc_of"] = flot_in
        else:
            cyc_diag = {"note": "cyclone disabled (open-circuit grinding)"}
            bm_water = fresh * _flot_dilution_water_per_t(p)
            out["bm_feed"] = Stream(solids_tph=fresh, water_tph=bm_water,
                                    F80_um=bm_fresh_F80_um,
                                    cu_grade=sag_out.cu_grade)
            flot_in = Stream(solids_tph=fresh, water_tph=bm_water,
                             F80_um=p.target_flot_P80_um,
                             cu_grade=sag_out.cu_grade)
            out["bm_discharge"] = flot_in
            out["cyc_of"] = flot_in
    else:
        power["bm_kw"] = 0.0
        cyc_diag = {"note": "ball mill disabled (SAG-only grinding)"}
        sag_water = fresh * _flot_dilution_water_per_t(p)
        flot_in = Stream(solids_tph=fresh, water_tph=sag_water,
                         F80_um=sag_out.F80_um,
                         cu_grade=sag_out.cu_grade)
        out["cyc_of"] = flot_in
    return flot_in, cyc_diag


def _flot_dilution_water_per_t(p: CuSulfideParams) -> float:
    """t water per t solids to reach the flotation pulp density.

    slurry_fraction (m3 solids / m3 slurry — the same value the kinetic model
    uses as the cell pulp density) sets the dilution-water demand:
        water/solids (t/t) = (1 - phi) / (SG * phi),  SG = 2.7 t/m3.
    Deriving the feed water from the optimizer's slurry_fraction (instead of a
    fixed %-solids target) makes dilution pay its real price — makeup water on
    the TEA water line and cell volume on the capex line — rather than giving
    the kinetic benefit for free (bound-desaturation Attempt 7). Pulp-density
    vs capacity trade-off: Wills 7th ed. Ch. 12.
    """
    phi = p.flot_op.slurry_fraction
    return (1.0 - phi) / (2.7 * phi)


def _run_rougher_flotation(flot_in: Stream, p: CuSulfideParams,
                           topo: "Topology", out: dict,
                           power: dict) -> tuple[Stream, Stream, dict]:
    if topo.flotation_enabled:
        flot_op = p.flot_op
        if p.flot_size_from_grind and flot_in.F80_um:
            flot_op = replace(flot_op, particle_size=flot_in.F80_um)
        if p.flot_autosize:
            slurry_m3_h = flot_in.solids_tph / 2.7 + flot_in.water_tph
            # Cell volume holds slurry plus dispersed air: effective pulp
            # volume is V*(1 - air_fraction), so the bank sizes up by
            # 1/(1 - air_fraction). Air now costs installed volume (capex +
            # power) in proportion to what it displaces, instead of being a
            # free rate-constant lever (bound-desaturation Attempt 7).
            # Effective-volume sizing basis: Wills 7th ed. Ch. 12.
            total_vol_m3 = (slurry_m3_h * p.flot_residence_time_min / 60.0
                            * p.flot_froth_factor
                            / (1.0 - flot_op.air_fraction))
            n_cells = max(1, math.ceil(total_vol_m3 / p.flot_cell_volume_m3))
            flot_op = replace(flot_op, num_cells=n_cells,
                              cell_volume=p.flot_cell_volume_m3,
                              ret_time=p.flot_residence_time_min)
        miner_props = MINERALOGY_PROPS.get(p.mineralogy,
                                           MINERALOGY_PROPS["chalcopyrite"])
        # Liberation-limited ceiling: coarse rougher feed cannot reach the
        # fine-grind floatability cap (see CuSulfideParams notes). Uses the
        # actual flotation feed size (cyclone O/F F80), falling back to the
        # grind target. Unity at/below the reference grind, so the validated
        # 150 um operating point is unchanged.
        p80_feed = flot_in.F80_um or p.target_flot_P80_um
        liberation = 1.0 - max(0.0, p80_feed - p.liberation_p80_ref_um) \
            * p.coarse_recovery_penalty_per_um
        liberation = max(liberation, p.liberation_floor)
        # Froth-flooding penalty: above the critical Jg, froth destabilises
        # and recovery falls back. Applied as a multiplier on the effective cap
        # (same pattern as liberation — post-kinetics ceiling, not inside the
        # PhD-derived kinetic model). Unity at Jg <= froth_flood_jg_ref.
        jg = flot_op.sp_gas_rate
        froth_stability = 1.0 - max(0.0, jg - p.froth_flood_jg_ref) \
            * p.froth_flood_slope
        froth_stability = max(froth_stability, p.froth_flood_floor)
        # Turbulent-detachment penalty: high specific power strips collected
        # particles from bubble surfaces (Schubert 1999; Pyke et al. 2003).
        sp = flot_op.sp_power
        detachment = 1.0 - max(0.0, sp - p.sp_power_ref) * p.sp_power_detach_slope
        detachment = max(detachment, p.sp_power_detach_floor)
        # Note: circulating_load removed from DOE (Attempt 6); it is fixed at
        # CuSulfideParams.circulating_load default (2.5). The CL-sharpness
        # benefit cannot be represented in the current model (see param notes).
        eff_cap = min(p.rougher_recovery_cap,
                      miner_props["flot_R_max"]) * liberation * froth_stability \
                  * detachment
        conc, tails, flot_diag = run_flotation(
            flot_in, p.flot_fit, p.flot_cell, flot_op,
            gangue_recovery=p.gangue_recovery,
            scavenger_recovery=p.scavenger_recovery_on_tails,
            recovery_cap=eff_cap)
        flot_diag["total_volume_m3"] = flot_op.cell_volume * flot_op.num_cells
        flot_diag["num_cells"] = flot_op.num_cells
        flot_diag["cell_volume_m3"] = flot_op.cell_volume
        flot_diag["liberation_factor"] = liberation
        flot_diag["froth_stability_factor"] = froth_stability
        flot_diag["detachment_factor"] = detachment
        # Surface the scavenger recovery applied to rougher tails so the
        # scavenger bank is visible in diagnostics and the flowsheet diagram
        # (the recovery itself is already folded into `tails` via run_flotation:
        # R_total = R_rougher + (1 - R_rougher) * scavenger_recovery).
        flot_diag["scavenger_recovery_on_tails"] = p.scavenger_recovery_on_tails
        out["rougher_conc"] = conc
        out["rougher_tails"] = tails
        power["flotation_kw"] = flot_diag["power_kw"]
    else:
        conc = Stream(solids_tph=0.0, water_tph=0.0, F80_um=flot_in.F80_um,
                      cu_grade=0.0)
        tails = flot_in
        flot_diag = {"note": "flotation disabled (hydromet route)",
                     "total_volume_m3": 0.0, "num_cells": 0,
                     "cell_volume_m3": 0.0}
        out["rougher_conc"] = conc
        out["rougher_tails"] = tails
        power["flotation_kw"] = 0.0
    return conc, tails, flot_diag


def _run_regrind_and_cleaner(conc: Stream, tails: Stream, flot_in: Stream,
                             p: CuSulfideParams, topo: "Topology",
                             out: dict,
                             power: dict) -> tuple[Stream, Stream, dict, dict]:
    regrind_diag = {"note": "regrind disabled"}
    if topo.regrind_enabled:
        F80 = conc.F80_um or p.target_flot_P80_um
        mill_props = REGRIND_MILL_PROPS.get(topo.regrind_mill_type,
                                            REGRIND_MILL_PROPS["ball"])
        Ecs_ball = 10.0 * p.regrind_Wi * (
            1.0 / math.sqrt(p.regrind_target_P80_um) - 1.0 / math.sqrt(F80))
        Ecs = max(0.0, Ecs_ball * mill_props["energy_factor"])
        regrind_kw = Ecs * conc.solids_tph
        power["regrind_kw"] = regrind_kw
        conc = Stream(solids_tph=conc.solids_tph, water_tph=conc.water_tph,
                      F80_um=p.regrind_target_P80_um, cu_grade=conc.cu_grade)
        out["regrind_discharge"] = conc
        regrind_diag = {"Ecs_kwh_t": Ecs, "regrind_kw": regrind_kw,
                        "F80_in": F80, "P80_out": p.regrind_target_P80_um,
                        "mill_type": topo.regrind_mill_type,
                        "energy_factor": mill_props["energy_factor"]}

    cleaner_diag = {"note": "cleaner disabled"}
    if topo.cleaner_enabled:
        delta_p80 = p.regrind_target_P80_um - 30.0
        cleaner_op = replace(
            p.flot_op,
            particle_size=p.regrind_target_P80_um,
            ret_time=p.cleaner_residence_time_min,
            cell_volume=p.cleaner_cell_volume_m3,
            num_cells=max(1, int(p.flot_op.num_cells)),
            froth_height=p.cleaner_froth_height_m,
        )
        R_cu_kin = fl.recovery(p.cleaner_flot_fit, p.flot_cell, cleaner_op)["recovery"]
        liberation = max(0.85, min(1.05,
                                   1.0 - p.cleaner_liberation_P80_slope * delta_p80))
        R_cu_eff = max(0.80, min(0.99, R_cu_kin * liberation))
        N = max(1, int(getattr(topo, "n_cleaner_stages", 1)))
        cu_in_flot = flot_in.solids_tph * flot_in.cu_grade
        gangue_in_flot = flot_in.solids_tph * (1.0 - flot_in.cu_grade)
        R_R_cu = ((conc.solids_tph * conc.cu_grade) / cu_in_flot
                  if cu_in_flot > 0 else 0.0)
        gangue_in_rougher_conc = conc.solids_tph * (1.0 - conc.cu_grade)
        R_R_g = (gangue_in_rougher_conc / gangue_in_flot
                 if gangue_in_flot > 0 else 0.0)
        c_R = max(0.0, min(0.999999, conc.cu_grade))
        target_grade = max(0.01, min(0.60,
                                     p.target_cleaner_conc_grade_pct / 100.0))
        target_R_g_recycle = (
            R_cu_eff * c_R * (1.0 - target_grade)
            / max(1e-9, target_grade * (1.0 - c_R))
        )
        R_g_required = (
            target_R_g_recycle * (1.0 - R_R_g)
            / max(1e-9, 1.0 - target_R_g_recycle * R_R_g)
        )
        R_g_eff = max(0.04, min(0.20, R_g_required))
        R_g_per_stage = R_g_eff ** (1.0 / N)
        denom_cu = max(1e-9, 1.0 - R_R_cu * (1.0 - R_cu_eff))
        denom_g = max(1e-9, 1.0 - R_R_g * (1.0 - R_g_eff))
        R_cu_recycle = min(0.999, R_cu_eff / denom_cu)
        R_g_recycle = min(0.999, R_g_eff / denom_g)
        cu_in = conc.solids_tph * conc.cu_grade
        gangue_in = conc.solids_tph * (1.0 - conc.cu_grade)
        cu_to_cleaner_conc = cu_in * R_cu_recycle
        gangue_to_cleaner_conc = gangue_in * R_g_recycle
        clean_solids = cu_to_cleaner_conc + gangue_to_cleaner_conc
        clean_grade = cu_to_cleaner_conc / clean_solids if clean_solids > 0 else 0.0
        cu_in_final_tails_target = cu_in_flot - cu_to_cleaner_conc
        gangue_in_final_tails_target = gangue_in_flot - gangue_to_cleaner_conc
        tails_solids = max(0.0, gangue_in_final_tails_target
                           + cu_in_final_tails_target)
        tails_grade = (cu_in_final_tails_target / tails_solids
                       if tails_solids > 0 else 0.0)
        tails = Stream(solids_tph=tails_solids, water_tph=tails.water_tph,
                       F80_um=tails.F80_um, cu_grade=tails_grade)
        cleaner_tails = Stream(solids_tph=0.0, water_tph=0.0,
                               F80_um=conc.F80_um, cu_grade=0.0)
        w_clean = clean_solids * (100.0 / 50.0 - 1.0)
        w_clean = min(w_clean, conc.water_tph)
        cleaner_conc = Stream(solids_tph=clean_solids, water_tph=w_clean,
                              F80_um=conc.F80_um, cu_grade=clean_grade)
        slurry_m3_h = conc.solids_tph / 2.7 + conc.water_tph
        # Same gas-holdup effective-volume basis as the rougher bank.
        clean_total_vol = (slurry_m3_h * p.cleaner_residence_time_min / 60.0
                           * p.flot_froth_factor
                           / (1.0 - p.flot_op.air_fraction))
        clean_n_cells = max(1, math.ceil(clean_total_vol / p.cleaner_cell_volume_m3))
        cleaner_diag = {
            "R_cu_per_pass": R_cu_eff, "R_g_per_pass": R_g_eff,
            "R_g_per_stage": R_g_per_stage, "n_stages": N,
            "R_cu_with_recycle": R_cu_recycle,
            "R_g_with_recycle": R_g_recycle,
            "rougher_R_cu": R_R_cu, "rougher_R_g": R_R_g,
            "target_conc_grade": target_grade,
            "R_g_required": R_g_required,
            "regrind_P80_um": p.regrind_target_P80_um,
            "conc_grade": clean_grade,
            "total_volume_m3": clean_n_cells * p.cleaner_cell_volume_m3 * N,
            "num_cells": clean_n_cells * N,
            "cell_volume_m3": p.cleaner_cell_volume_m3,
        }
        out["cleaner_conc"] = cleaner_conc
        out["cleaner_tails"] = cleaner_tails
        out["combined_tails"] = tails
        conc = cleaner_conc
    return conc, tails, regrind_diag, cleaner_diag


def _run_dewatering(conc: Stream, p: CuSulfideParams, topo: "Topology",
                    out: dict) -> tuple[Stream, dict, dict]:
    if topo.thickener_enabled:
        c_uf, c_of, th_diag = run_thickener(conc, p.conc_thickener,
                                            target_uf_pct=p.conc_uf_pct)
        out["conc_thickener_uf"] = c_uf
        out["conc_thickener_of"] = c_of
    else:
        c_uf = conc
        th_diag = {"note": "thickener disabled (no dewatering before filter)"}

    if topo.filter_enabled:
        cake, filtrate, filt_diag = run_filter(c_uf, p.belt_filter,
                                               p.filter_cake_moisture_pct)
        out["filter_cake"] = cake
        out["filter_filtrate"] = filtrate
    else:
        cake = c_uf
        out["filter_cake"] = cake
        filt_diag = {"note": "filter disabled (concentrate ships at upstream moisture)"}
    return cake, th_diag, filt_diag


def _run_concentrate_hydromet(final_conc: Stream, ore: Stream,
                              p: CuSulfideParams, topo: "Topology",
                              out: dict,
                              power: dict) -> tuple[dict, dict, dict, dict, float, float, float]:
    leach_diag = {"note": "leach disabled"}
    sx_diag = {"note": "sx disabled"}
    ew_diag = {"note": "ew disabled"}
    neut_diag = {"note": "neutralization disabled"}
    cathode_tph = 0.0
    cu_to_sx_raffinate_tph = 0.0
    cu_to_ew_spent_tph = 0.0

    if topo.leach_enabled:
        leach_feed = final_conc
        feed_slurry_m3_h = leach_feed.solids_tph / 2.7 + leach_feed.water_tph
        cu_in_feed_tph = leach_feed.solids_tph * leach_feed.cu_grade
        leach_diag = lch.size_leach(cu_in_feed_tph, feed_slurry_m3_h,
                                     p.leach_params)
        cu_leached_tph = leach_diag["cu_leached_tph"]
        out["leach_residue"] = Stream(
            solids_tph=leach_feed.solids_tph,
            water_tph=leach_feed.water_tph,
            F80_um=leach_feed.F80_um,
            cu_grade=max(p.leach_tails_cu_grade_floor,
                         (cu_in_feed_tph - cu_leached_tph)
                         / max(leach_feed.solids_tph, 1e-9)))
        out["leach_pls"] = Stream(
            solids_tph=0.0, water_tph=leach_feed.water_tph,
            cu_grade=0.0, cu_aq_gpl=cu_leached_tph * 1000.0
                          / max(feed_slurry_m3_h, 1e-9))
        tails_slurry_m3_h = feed_slurry_m3_h

        if topo.sx_enabled:
            # SX-EW is a closed circuit: SX raffinate recycles to the
            # autoclave leach liquor and EW spent electrolyte recycles to
            # the SX strip stage, so single-pass SX/EW efficiency badly
            # understates steady-state concentrate-to-cathode recovery.
            # Apply the same steady-state recycle closure already used for
            # the cleaner-tails loop (Wills & Finch, Mineral Processing
            # Technology 8th ed. Ch. 12.4 recycle algebra; SX-EW circuit
            # context Ch. 15): R_ss = R / (1 - (1-R)(1-b)). The only
            # permanent Cu sink is the electrolyte bleed taken for
            # impurity control (~3%, Schlesinger et al., Extractive
            # Metallurgy of Copper 5th ed. Ch. 17). Per-pass SX and EW
            # efficiencies stay at 0.95; the bleed sets the steady-state
            # circuit loss.
            sx_pass_eff, ew_pass_eff, electrolyte_bleed = 0.95, 0.95, 0.03
            circuit_pass_eff = sx_pass_eff * ew_pass_eff
            sx_ew_recovery = circuit_pass_eff / (
                1.0 - (1.0 - circuit_pass_eff) * (1.0 - electrolyte_bleed))
            cu_to_organic_tph = cu_leached_tph * sx_ew_recovery
            cu_to_sx_raffinate_tph = cu_leached_tph - cu_to_organic_tph
            sx_diag = {"cu_transfer_eff": sx_pass_eff,
                       "sx_ew_circuit_recovery": sx_ew_recovery,
                       "electrolyte_bleed": electrolyte_bleed,
                       "cu_to_organic_tph": cu_to_organic_tph,
                       "cu_in_raffinate_tph": cu_to_sx_raffinate_tph,
                       "tank_volume_m3": tails_slurry_m3_h * 0.5}

            if topo.ew_enabled:
                # Circuit recycle is already folded into sx_ew_recovery;
                # EW just plates the steady-state cathode tonnage.
                cu_to_ew_spent_tph = 0.0
                ew_diag = ew_mod.size_ew(cu_to_organic_tph, p.ew_params)
                cathode_tph = ew_diag["cathode_cu_tph"]
                power["ew_kw"] = ew_diag["total_power_kw"]
                out["cathode"] = Stream(solids_tph=cathode_tph, water_tph=0.0,
                                        cu_grade=1.0)

        if topo.neutralization_enabled:
            acid_kg_per_h = leach_diag["acid_kg_per_h"]
            fe_kg = p.leach_fe_kg_per_t_ore * ore.solids_tph
            as_kg = p.leach_as_kg_per_t_ore * ore.solids_tph
            mso4_residual = {"Cu": 100.0, "Zn": 50.0}
            neut_diag = neut.size_neutralization(
                h2so4_kg_per_h=acid_kg_per_h,
                fe_kg_per_h=fe_kg, as_kg_per_h=as_kg,
                metals_sulfate_kg_per_h=mso4_residual,
                slurry_m3_per_h=tails_slurry_m3_h,
                p=p.neut_params)

    return (leach_diag, sx_diag, ew_diag, neut_diag, cathode_tph,
            cu_to_sx_raffinate_tph, cu_to_ew_spent_tph)


def _build_mass_balance(ore: Stream, final_conc: Stream, tails: Stream,
                        topo: "Topology", leach_diag: dict,
                        cathode_tph: float, cu_to_sx_raffinate_tph: float,
                        cu_to_ew_spent_tph: float, power: dict) -> dict:
    cu_in = ore.solids_tph * ore.cu_grade
    cu_out_cathode = cathode_tph
    if topo.leach_enabled:
        cu_out_conc = 0.0
        cu_unleached = max(0.0, (final_conc.solids_tph * final_conc.cu_grade
                                  - leach_diag.get("cu_leached_tph", 0.0)))
        cu_out_tails = tails.solids_tph * tails.cu_grade + cu_unleached
    else:
        cu_out_conc = final_conc.solids_tph * final_conc.cu_grade
        cu_out_tails = tails.solids_tph * tails.cu_grade
    cu_out_total = (cu_out_conc + cu_out_cathode + cu_out_tails
                    + cu_to_sx_raffinate_tph + cu_to_ew_spent_tph)
    return {
        "cu_in_tph": cu_in,
        "cu_in_conc_tph": cu_out_conc,
        "cu_in_cathode_tph": cu_out_cathode,
        "cu_in_tails_tph": cu_out_tails,
        "cu_lost_sx_raffinate_tph": cu_to_sx_raffinate_tph,
        "cu_lost_ew_spent_tph": cu_to_ew_spent_tph,
        "cu_closure_err": cu_in - cu_out_total,
        "overall_cu_recovery": (cu_out_conc + cu_out_cathode) / cu_in
                                if cu_in > 0 else 0.0,
        "conc_grade_pct": final_conc.cu_grade * 100.0,
        "total_power_kw": sum(power.values()),
    }


def simulate_cu_sulfide(feed_tph: float, feed_grade: float,
                        params: Optional[CuSulfideParams] = None,
                        topology: Optional["Topology"] = None) -> dict:
    """Run the reference Cu sulfide chain and return per-stage Streams + diagnostics.

    `topology` (process_model.topology.Topology) gates the optional stages:
    ball_mill_enabled, cyclone_enabled, thickener_enabled, filter_enabled.
    Default Topology() reproduces the original SAG+BM closed-circuit chain.
    """
    from .topology import Topology as _Topology, _is_valid
    p = _apply_mineralogy(params or CuSulfideParams())
    topo = topology or _Topology()
    if not _is_valid(topo):
        raise ValueError(f"invalid process topology: {topo}")

    out, power, ore, crushed = _run_crushing(feed_tph, feed_grade, p)
    if topo.heap_leach_enabled:
        return _run_heap_leach_route(ore, crushed, p, topo, out, power)

    sag_out, bm_fresh_F80_um, undersize_tph, oversize_tph, screen_diag = (
        _run_screen_and_sag(crushed, p, topo, out, power)
    )
    flot_in, cyc_diag = _run_grinding(
        crushed, sag_out, bm_fresh_F80_um, undersize_tph, oversize_tph,
        p, topo, out, power)
    conc, tails, flot_diag = _run_rougher_flotation(
        flot_in, p, topo, out, power)
    conc, tails, regrind_diag, cleaner_diag = _run_regrind_and_cleaner(
        conc, tails, flot_in, p, topo, out, power)
    final_conc, th_diag, filt_diag = _run_dewatering(conc, p, topo, out)
    (leach_diag, sx_diag, ew_diag, neut_diag, cathode_tph,
     cu_to_sx_raffinate_tph, cu_to_ew_spent_tph) = _run_concentrate_hydromet(
        final_conc, ore, p, topo, out, power)
    mb = _build_mass_balance(
        ore, final_conc, tails, topo, leach_diag, cathode_tph,
        cu_to_sx_raffinate_tph, cu_to_ew_spent_tph, power)

    return {"streams": out, "power_kw": power, "balance": mb,
            "cyclone": cyc_diag, "flotation": flot_diag,
            "conc_thickener": th_diag, "belt_filter": filt_diag,
            "screen": screen_diag, "regrind": regrind_diag,
            "cleaner": cleaner_diag, "leach": leach_diag,
            "heap_leach": {"note": "heap leach disabled"},
            "sx": sx_diag,
            "ew": ew_diag, "neutralization": neut_diag,
            "topology": topo, "mineralogy": p.mineralogy,
            "conc_arsenic_pct": _conc_arsenic_pct(p),
            "byproducts": _byproduct_grades(p)}

def print_report(sim: dict) -> None:
    print("=== Streams ===")
    for name, s in sim["streams"].items():
        cu_tph = s.solids_tph * s.cu_grade
        print(f"  {name:22s}  solids={s.solids_tph:9.2f} tph  water={s.water_tph:9.2f} tph  "
              f"%sol={s.solids_pct:5.1f}  Cu={s.cu_grade*100:5.2f}%  Cu_tph={cu_tph:7.3f}")
    print("=== Power ===")
    for k, v in sim["power_kw"].items():
        print(f"  {k:22s}  {v:10.1f} kW")
    print("=== Mass balance ===")
    for k, v in sim["balance"].items():
        print(f"  {k:22s}  {v:10.4f}")


if __name__ == "__main__":
    import sys
    sim = simulate_cu_sulfide(2000.0, 0.008)
    print_report(sim)
    if "--tea" in sys.argv:
        from .tea import evaluate, print_tea_report
        print_tea_report(evaluate(sim))
