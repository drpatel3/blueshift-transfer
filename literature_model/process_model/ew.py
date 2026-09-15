"""Electrowinning (EW) cell model — SME Mineral Processing Handbook Ch. 20.

Sources:
    Eq. 20.70  — Faraday's law on cathode mass deposited
    Eq. 20.71  — Cell solution resistance (R = rho * L / A)
    p. 499     — Limiting current density i_L = 18.6 * [Cu+] g/L; operate at
                  i_op <= 40% of i_L to avoid powdery deposits.

Outputs: cathode Cu mass, cell voltage, energy demand kWh/kg Cu, electrode
area required to plate the target tph.
"""
from dataclasses import dataclass
import math


F_FARADAY = 96485.0      # C/mol
M_CU = 63.5              # g/mol
Z_CU = 2                 # electrons per Cu^2+
SECONDS_PER_HOUR = 3600.0


@dataclass
class EWParams:
    """EW operating point. Defaults are typical Cu cathode tankhouse."""
    cu_concentration_gpl: float = 45.0    # advance electrolyte Cu, g/L
    current_efficiency: float = 0.92      # CE (Eq 20.70 fraction)
    i_op_fraction_of_iL: float = 0.40     # operate at <= 40% of i_L
    cell_voltage_overhead: float = 1.96   # V; thermodynamic + overpotentials
    electrolyte_resistivity: float = 0.5  # ohm-cm (typical Cu sulfate raffinate)
    electrode_spacing_cm: float = 5.0     # anode-cathode gap
    electrode_area_per_cell_m2: float = 1.2  # single-cathode plate area
    cathodes_per_cell: int = 60           # commercial tankhouse cell


def limiting_current_density(cu_gpl: float) -> float:
    """SME p. 499: i_L (A/m^2) = 18.6 * [Cu+] g/L."""
    return 18.6 * cu_gpl


def operating_current_density(p: EWParams) -> float:
    """Operating i_op = i_op_fraction * i_L."""
    return p.i_op_fraction_of_iL * limiting_current_density(p.cu_concentration_gpl)


def cathode_mass_per_cell_per_h(p: EWParams) -> float:
    """Eq. 20.70: W = (I * t * M * CE) / (z * F).

    Per cathode plate per hour, in grams. Scales linearly with electrode
    area and current density.
    """
    i_op = operating_current_density(p)
    I_per_plate = i_op * p.electrode_area_per_cell_m2  # A
    t = SECONDS_PER_HOUR
    return (I_per_plate * t * M_CU * p.current_efficiency) / (Z_CU * F_FARADAY)


def cell_resistance_ohm(p: EWParams) -> float:
    """Eq. 20.71: R = rho * L / A.

    rho = resistivity (ohm-cm), L = electrode spacing (cm), A = effective
    cross-section (cm^2 = electrode_area_m2 * 10^4).
    """
    A_cm2 = p.electrode_area_per_cell_m2 * 1.0e4
    return p.electrolyte_resistivity * p.electrode_spacing_cm / A_cm2


def cell_voltage(p: EWParams) -> float:
    """V_cell = I*R + overhead.  Overhead lumps thermodynamic decomposition
    voltage + anodic + cathodic overpotentials; SME suggests ~1.96 V."""
    i_op = operating_current_density(p)
    I_per_plate = i_op * p.electrode_area_per_cell_m2
    return I_per_plate * cell_resistance_ohm(p) + p.cell_voltage_overhead


def specific_energy_kwh_per_kg_cu(p: EWParams) -> float:
    """SME check value ~ 2.1 kWh/kg Cu cathode. Computed from V*I*t / W."""
    V = cell_voltage(p)
    i_op = operating_current_density(p)
    I_per_plate = i_op * p.electrode_area_per_cell_m2
    W_g_per_h = cathode_mass_per_cell_per_h(p)
    if W_g_per_h <= 0:
        return float("inf")
    energy_kwh_per_h = V * I_per_plate / 1000.0
    return energy_kwh_per_h / (W_g_per_h / 1000.0)


def size_ew(target_cu_tph: float, p: EWParams) -> dict:
    """Compute cells, total electrode area, total power."""
    g_per_h_per_cathode = cathode_mass_per_cell_per_h(p)
    g_per_h_per_cell = g_per_h_per_cathode * p.cathodes_per_cell
    target_g_per_h = target_cu_tph * 1.0e6
    if g_per_h_per_cell <= 0:
        raise ValueError("zero cathode plating rate")
    num_cells = math.ceil(target_g_per_h / g_per_h_per_cell)

    total_electrode_area_m2 = (num_cells * p.cathodes_per_cell
                               * p.electrode_area_per_cell_m2)
    V = cell_voltage(p)
    i_op = operating_current_density(p)
    total_kw = (num_cells * p.cathodes_per_cell
                * i_op * p.electrode_area_per_cell_m2 * V / 1000.0)

    return {
        "num_cells": num_cells,
        "cathodes_total": num_cells * p.cathodes_per_cell,
        "total_electrode_area_m2": total_electrode_area_m2,
        "cell_voltage_V": V,
        "i_op_A_per_m2": i_op,
        "i_L_A_per_m2": limiting_current_density(p.cu_concentration_gpl),
        "specific_energy_kwh_per_kg": specific_energy_kwh_per_kg_cu(p),
        "total_power_kw": total_kw,
        "cathode_cu_tph": target_cu_tph,
    }


if __name__ == "__main__":
    # Smoke test: plate 12 tph Cu (default Cu sulfide chain produces ~13)
    p = EWParams()
    diag = size_ew(target_cu_tph=12.0, p=p)
    print("EW sizing for 12 tph Cu cathode:")
    for k, v in diag.items():
        print(f"  {k:30s} {v}")
