"""Mechanical-cell flotation model.

Faithful port of the VBA `Function FlotRec` (Blueshift source, see
optimization_model/data/Function FlotRec...docx). Components:
  - Szyszkowski surface tension with MIBC/PPG400/Octanol/Pentanol
  - Lu 2-compartment energy dissipation
  - Abrahamson (1975) collision kernel beta
  - Luttrell-Yoon (1991) modified p_collision
  - Work-of-adhesion-based p_detachment
  - Finch-Dobby (1990) froth recovery
  - Do & Yoon (eq. 32) single-CSTR collection recovery (VBA active line)

Fitting defaults from VBA: b=2.3, alpha=0.05, bubble_f=0.5, detach_f=0.5,
bulk_zone=0.5, coverage=0.5, impeller_zone=15, energy_barrier=1e-18 J placeholder.

Units: SI internally (m, kg, s, W/m3). Inputs: particle size in microns,
superficial gas velocity in cm/s, retention time in minutes, frother in ppm.
"""
from dataclasses import dataclass
import math


WATER_DENSITY = 1000.0   # kg/m3
AIR_DENSITY = 1.225      # kg/m3
WATER_VISCOSITY = 0.001  # Pa s
PI = math.pi


@dataclass
class FittingParams:
    """Empirically fit froth + hydrodynamic constants."""
    b: float              # froth plug-flow exponent
    alpha: float          # froth attachment decay, 1/s
    coverage: float       # bubble coverage factor (unused scalar, kept for parity)
    bubble_f: float       # bubble-diameter multiplier on Sauter prediction
    detach_f: float       # detachment-velocity multiplier
    bulk_zone: float      # bulk/mean energy dissipation ratio
    energy_barrier_j: float = 1e-18  # DLVO attachment barrier, J (placeholder until full DLVO)
    drag_beta: float = 1.0           # hydrodynamic drag correction on attachment


@dataclass
class CellParams:
    """Physical cell + slurry constants."""
    specific_gravity: float   # solids SG (water=1)
    permitivity: float        # medium permittivity, F/m
    dielectric: float         # relative dielectric
    pe: float                 # Peclet number (axial dispersion)
    cell_area: float          # cross-sectional area, m2
    bbl_ratio: float          # top/bottom bubble-diameter ratio


@dataclass
class OperatingParams:
    """Operator-adjustable variables."""
    particle_size: float      # um
    contact_angle: float      # deg
    bubble_z_pot: float       # mV
    particle_z_pot: float     # mV
    sp_power: float           # kW/m3
    sp_gas_rate: float        # cm/s (superficial gas velocity)
    air_fraction: float       # m3 air / m3 cell
    slurry_fraction: float    # m3 solids / m3 slurry (aq phase)
    num_cells: int            # cells in series
    ret_time: float           # bank-total residence time in the stage, min
    cell_volume: float        # per-cell volume, m3
    froth_height: float       # m
    frother_conc: float       # ppm


# Frother adsorption parameters (Szyszkowski)
_FROTHER = {
    2: ("MIBC",     0.000005, 230.0,     102170.0),
    3: ("PPG400",   0.000001, 1700000.0, 134170.0),
    4: ("Octanol",  0.000008, 2200.0,    130230.0),
    5: ("Pentanol", 0.000006, 55.0,      88150.0),
}


def surface_tension(frother_type: int, frother_conc_ppm: float,
                    temperature_c: float = 23.0) -> float:
    """Szyszkowski surface tension, N/m. frother_type: 2=MIBC,3=PPG400,4=Octanol,5=Pentanol."""
    sigma0 = 0.07243  # pure water @ 23 C
    if frother_type not in _FROTHER:
        return sigma0
    _, gamma, k, mw = _FROTHER[frother_type]
    c_mol_per_L = frother_conc_ppm / mw
    return sigma0 - 8.314 * (273.15 + temperature_c) * gamma * math.log(k * c_mol_per_L + 1.0)


def bubble_diameter(surface_tension_nm: float, e_impeller: float,
                    bubble_f: float) -> float:
    """Sauter bubble diameter from impeller energy dissipation, m."""
    return bubble_f * (2.11 * surface_tension_nm /
                       (WATER_DENSITY * e_impeller ** 0.66)) ** 0.6


def energy_dissipation(sp_power_wm3: float, air_fraction: float,
                       slurry_fraction: float, particle_dens: float,
                       bulk_zone: float, impeller_zone: float = 15.0) -> tuple:
    """Returns (e_mean, e_bulk, e_impeller) in W/kg."""
    total_dens = (air_fraction * AIR_DENSITY
                  + (1.0 - air_fraction) * slurry_fraction * particle_dens
                  + (1.0 - slurry_fraction) * WATER_DENSITY)
    e_mean = sp_power_wm3 / total_dens
    return e_mean, bulk_zone * e_mean, impeller_zone * e_mean


def abrahamson_beta(particle_diam: float, bubble_diam: float,
                    e_bulk: float, particle_dens: float) -> float:
    """Abrahamson (1975) collision kernel, m3/s."""
    kin_visc = WATER_VISCOSITY / WATER_DENSITY
    collision_diam = 0.5 * (particle_diam + bubble_diam)
    u1 = (0.4 * (e_bulk ** (4 / 9)) * (particle_diam ** (7 / 9))
          * (kin_visc ** (-1 / 3))
          * (particle_dens / WATER_DENSITY - 1.0) ** (2 / 3)) ** 2
    u2 = 2.0 * (e_bulk * bubble_diam) ** (2 / 3)
    return (2.0 ** 1.5) * (PI ** 0.5) * (collision_diam ** 2) * math.sqrt(u1 + u2), u1, u2


def p_attachment(mass_particle: float, u1: float, drag_beta: float,
                 eb: float) -> float:
    kinetic_e = 0.5 * mass_particle * u1 / (drag_beta ** 2)
    return math.exp(-eb / kinetic_e) if kinetic_e > 0 else 0.0


def p_collision(particle_diam: float, bubble_diam: float,
                u2: float) -> float:
    """Luttrell-Yoon modified collision probability."""
    kin_visc = WATER_VISCOSITY / WATER_DENSITY
    re = math.sqrt(u2) * bubble_diam / kin_visc
    x = math.sqrt(1.5 * (1.0 + (3.0 / 16.0 * re) / (1.0 + 0.249 * re ** 0.56))) \
        * (particle_diam / bubble_diam)
    val = math.tanh(x) ** 2
    return min(val, 1.0)


def p_detachment(particle_diam: float, bubble_diam: float,
                 mass_particle: float, e_impeller: float,
                 detach_f: float, surface_tension_nm: float,
                 contact_angle_deg: float, eb: float) -> float:
    kin_visc = WATER_VISCOSITY / WATER_DENSITY
    u_detach = detach_f * (particle_diam + bubble_diam) * math.sqrt(e_impeller / kin_visc)
    kinetic_e = 0.5 * mass_particle * u_detach ** 2
    work_adh = surface_tension_nm * PI * (particle_diam / 2.0) ** 2 \
        * (1.0 - math.cos(contact_angle_deg * PI / 180.0)) ** 2
    return math.exp(-(work_adh + eb) / kinetic_e) if kinetic_e > 0 else 0.0


def froth_recovery(bubble_diam: float, particle_diam: float,
                   bbl_ratio: float, specific_gravity: float,
                   sp_gas_rate_ms: float, froth_height: float,
                   num_cells: int, ret_time_min: float,
                   cell_volume: float, cell_area: float,
                   particle_dens: float, fp: FittingParams) -> float:
    """Finch-Dobby froth recovery = attachment (rmax * exp(-alpha*t_f)) + entrainment."""
    top_bubble = bubble_diam / bbl_ratio
    coverage_factor = 2.0 * (bubble_diam / particle_diam) ** 2
    buoyant = bubble_diam * ((1.0 - 0.001275)
                             / ((specific_gravity - 1.0) * coverage_factor)) ** (1.0 / 3.0)
    pfr = math.exp(fp.b * particle_diam / buoyant)
    rmax = bubble_diam / top_bubble
    t_froth = froth_height / sp_gas_rate_ms * pfr
    r_attach = rmax * math.exp(-fp.alpha * t_froth)

    air_q = sp_gas_rate_ms * cell_area
    water_q = cell_volume / (ret_time_min / num_cells * 60.0)
    r_water_max = (air_q / water_q) / ((1.0 / 0.2) - 1.0)
    if r_water_max > 1.0:
        r_water_max = 0.1
    r_entrain = r_water_max * math.exp(-0.0325 * (particle_dens - WATER_DENSITY) / 1000.0
                                       - 0.063 * particle_diam * 1.0e6)
    return min(r_attach + r_entrain, 1.0)


def collection_recovery(rate_const_1pm: float, ret_time_min: float,
                        num_cells: int, pe: float) -> float:
    """Single-CSTR collection-zone recovery, Do & Yoon eq. 32.
    The original VBA's dispersion branch is commented out; CSTR is the active line."""
    return 1.0 - 1.0 / (1.0 + rate_const_1pm * ret_time_min)


def collection_recovery_dispersion(rate_const_1pm: float, ret_time_min: float,
                                   num_cells: int, pe: float) -> float:
    """N-cell axial-dispersion alternative (VBA branch, commented out in source)."""
    Aa = (1.0 + 4.0 * rate_const_1pm * ret_time_min / (num_cells * pe)) ** 0.5
    denom = ((1.0 + Aa) ** 2 * math.exp(Aa * pe / 2.0)
             - (1.0 - Aa) ** 2 * math.exp(-Aa * pe / 2.0))
    return 1.0 - 4.0 * Aa * math.exp(pe / 2.0) / denom


def bank_recovery(recovery_i: float, num_cells: int) -> float:
    """Overall bank recovery from per-cell recovery (parallel failure model)."""
    return 1.0 - (1.0 - recovery_i) ** num_cells


def power_kw(op: OperatingParams) -> float:
    """Total installed power over the bank, kW."""
    return op.sp_power * op.cell_volume * op.num_cells


def water_m3_per_pass(op: OperatingParams) -> float:
    """Water throughput for one bank residence, m3."""
    particle_dens = 0.0  # dens cancels; keep for clarity
    # solids tonnage per pass / water-to-solids ratio
    solids_t = (op.slurry_fraction * op.cell_volume * op.num_cells
                * WATER_DENSITY) / 1000.0
    water_m3_per_t = ((1.0 - op.slurry_fraction)
                      / (op.slurry_fraction * WATER_DENSITY)) * 1000.0
    return water_m3_per_t * solids_t


def recovery(fp: FittingParams, cp: CellParams, op: OperatingParams,
             frother_type: int = 2) -> dict:
    """Full mechanical-cell flotation recovery.

    Returns dict with overall bank recovery, per-cell recovery, froth recovery,
    rate constant (1/min), power (kW), and water (m3/pass).
    """
    particle_diam = op.particle_size * 1.0e-6  # um -> m
    sp_power_wm3 = op.sp_power * 1000.0        # kW/m3 -> W/m3
    sp_gas_ms = op.sp_gas_rate / 100.0         # cm/s -> m/s
    particle_dens = cp.specific_gravity * 1000.0

    sigma = surface_tension(frother_type, op.frother_conc)
    _, e_bulk, e_imp = energy_dissipation(sp_power_wm3, op.air_fraction,
                                          op.slurry_fraction, particle_dens,
                                          fp.bulk_zone)
    d_b = bubble_diameter(sigma, e_imp, fp.bubble_f)

    vol_p = (4.0 / 3.0) * PI * (particle_diam / 2.0) ** 3
    vol_b = (4.0 / 3.0) * PI * (d_b / 2.0) ** 3
    mass_p = particle_dens * vol_p
    n_bubble = op.air_fraction / vol_b

    beta, u1, u2 = abrahamson_beta(particle_diam, d_b, e_bulk, particle_dens)
    eb, drag_beta = fp.energy_barrier_j, fp.drag_beta

    p_att = p_attachment(mass_p, u1, drag_beta, eb)
    p_col = p_collision(particle_diam, d_b, u2)
    p_det = p_detachment(particle_diam, d_b, mass_p, e_imp, fp.detach_f,
                         sigma, op.contact_angle, eb)

    r_froth = froth_recovery(d_b, particle_diam, cp.bbl_ratio, cp.specific_gravity,
                             sp_gas_ms, op.froth_height, op.num_cells,
                             op.ret_time, op.cell_volume, cp.cell_area,
                             particle_dens, fp)

    k_1pm = beta * n_bubble * p_att * p_col * (1.0 - p_det) * 60.0
    r_coll = collection_recovery(k_1pm, op.ret_time, op.num_cells, cp.pe)
    r_i = r_coll * r_froth / (r_coll * r_froth + 1.0 - r_coll)
    # Treat the rougher bank as a single mixing stage with bank-total
    # residence time. NumCells is a mechanical/sizing parameter (cell volume,
    # froth water flow, equipment count) — NOT a series-stacking multiplier.
    # Departure from the literal VBA `1 - (1 - r_i)^N`, which assumes a
    # particle passes through every cell sequentially.
    r_total = r_i

    return {
        "recovery": r_total,
        "recovery_per_cell": r_i,
        "froth_recovery": r_froth,
        "collection_recovery": r_coll,
        "rate_const_1pm": k_1pm,
        "bubble_diam_m": d_b,
        "power_kw": power_kw(op),
        "water_m3_per_pass": water_m3_per_pass(op),
    }
