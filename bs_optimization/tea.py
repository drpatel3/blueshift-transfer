# TEA Model Constants and Variables

import math

# ECONOMIC CONSTANTS (Row 35-43)
# Adjusted to realistic industry values
water_cost = 0.3  # $/m3 (industrial process water)
electricity_cost = 50  # $/MWh (industrial rates: $50-80/MWh)
heat_cost = 20  # $/MWh (from natural gas)
plant_lifetime = 25  # years
interest_rate = 0.08  # 8% discount rate
o_m_opex = 0.035  # 3.5% of CAPEX for O&M
capacity_factor = 0.9  # 90% annual operating capacity
lang_factor = 5  # Lang factor for installed costs

# Time conversions
year_in_days = 365  # days
capacity_days = year_in_days * capacity_factor  # max days operating at capacity

# Cost indices
perrys_ms_index = 1000  # Perry's Table M&S Index (1990)
inflation_factor = 2.4  # from M&S index - 1990

# ELEMENT DATA (Rows 6, 15, 27, 28)
# Element concentrations (ppm), atomic weights (g/mol), oxide molecular weights (g/mol)
elements = {
    'Au': {'concentration_ppm': 55, 'atomic_weight': 197, 'oxide_mw': 442},
    'Cu': {'concentration_ppm': 5800, 'atomic_weight': 64, 'oxide_mw': 80},
    'Pb': {'concentration_ppm': 20000, 'atomic_weight': 207, 'oxide_mw': 223},
    'Zc': {'concentration_ppm': 30000, 'atomic_weight': 65, 'oxide_mw': 81}
}

# GRINDING PARAMETERS (Rows 55-67)
# Feed and product sizes (default values)
feed_size = 10000  # microns (F)
product_size = 100  # microns (P)
p1 = 1000  # microns
g = 2.5  # g/rev (assumed)

# Calculated exponents
product_size_power = product_size ** 0.5  # P^(0.5)
feed_size_power = feed_size ** 0.5  # F^(0.5)
g_power = g ** 0.82  # G^(0.82)
p1_power = p1 ** 0.23  # P1^(0.23)


work_index = 13  # kWh/t
work = work_index * 10 * (1/math.sqrt(product_size) - 1/math.sqrt(feed_size))

# JAW CRUSHER COST BASIS (Rows 73-74, from Perry's Table 9-50)
jaw_crusher_small = {
    'cost': 34000,  # USD
    'power': 7.5,  # kW
    'exponent': 0.65
}

jaw_crusher_large = {
    'cost': 284000,  # USD
    'power': 74.6,  # kW
    'exponent': 0.81
}


# MOTOR COST BASIS (Rows 84-85, from Perry's Table 9-50)
motor_small = {
    'cost': 12300,  # USD
    'power': 7.5,  # kW
    'exponent': 0.56
}

motor_large = {
    'cost': 19300,  # USD
    'power': 52,  # kW
    'exponent': 0.77
}

# FROTH FLOTATION PARAMETERS (Rows 95-128)
flotation_work = 20  # kWh/t (assumed); < 10 for ores
water_flow_rate_multiplier = 4  # x Feed Flow Rate
feed_density = 2.7  # tonnes/m3 (typical copper ore density, changed from 0.0015 coal fly ash)
froth_factor_excess = 0.2  # Assumed excess
residence_time_flotation = 15  # minutes

# VERTICAL AGITATED TANK COST BASIS (Row 134, from Perry's Table 9-50)
vertical_agitated_tank_basis = {
    'cost': 12300,  # USD
    'volume': 3.8,  # m3
    'exponent': 0.5
}

# LEACHING PARAMETERS (Rows 150-171)
sl_ratio = 0.05  # S:L Ratio (solid to liquid)
acid_concentration = 0.15  # 15% acid in solution (typical for leaching, not pure acid)
sulfuric_acid_density = 1380  # kg/m3
residence_time_leaching = 180  # minutes

# ATMOSPHERIC TANK COST BASIS (Row 177, from Perry's Table 9-50)
atm_tank_basis = {
    'cost': 9300,  # USD
    'volume': 0.38,  # m3
    'exponent': 0.53
}

# SOLVENT EXCHANGE PARAMETERS (Rows 186-187)
sf_heavies_vs_lights = 5  # Separation Factor (Heavies vs. Lights)
purity = 0.999  # 3N purity

# CALCULATED VALUES
# Mass factors for elements (oxide MW / atomic weight)
mass_factor_au = elements['Au']['oxide_mw'] / elements['Au']['atomic_weight']  # J6
mass_factor_cu = elements['Cu']['oxide_mw'] / elements['Cu']['atomic_weight']  # J15
mass_factor_pb = elements['Pb']['oxide_mw'] / elements['Pb']['atomic_weight']  # J27
mass_factor_zc = elements['Zc']['oxide_mw'] / elements['Zc']['atomic_weight']  # J28

# Time unit conversions
hours_per_year = year_in_days * 24  # D42
minutes_per_year = hours_per_year * 60  # D43
seconds_per_year = minutes_per_year * 60  # D44
days_per_op_year = year_in_days * capacity_factor  # D46
hours_per_op_year = days_per_op_year * 24  # D47
minutes_per_op_year = hours_per_op_year * 60  # D48
seconds_per_op_year = minutes_per_op_year * 60  # D49

# Flotation residence time in seconds
residence_time_flotation_sec = residence_time_flotation * 60  # E104

# Leaching residence time in seconds
residence_time_leaching_sec = residence_time_leaching * 60  # E165


# COST ESTIMATION FUNCTIONS
def equipment_cost(equipment_type, capacity, basis=1):
    """
    Calculate equipment cost using power law scaling.
    Args:
        equipment_type: 'jaw_crusher', 'motor', 'vertical_tank', or 'atm_tank'
        capacity: Required capacity (kW for crushers/motors, m3 for tanks)
        basis: 1 for small basis, 2 for large basis (jaw_crusher and motor only)
    Returns:
        Cost in USD
    """
    equipment_data = {
        'jaw_crusher': [jaw_crusher_small, jaw_crusher_large],
        'motor': [motor_small, motor_large],
        'vertical_tank': [vertical_agitated_tank_basis],
        'atm_tank': [atm_tank_basis]
    }

    if equipment_type not in equipment_data:
        raise ValueError(f"Unknown equipment type: {equipment_type}")

    basis_idx = basis - 1
    basis_data = equipment_data[equipment_type][basis_idx]

    capacity_key = 'power' if equipment_type in ['jaw_crusher', 'motor'] else 'volume'

    return basis_data['cost'] * ((capacity / basis_data[capacity_key]) ** basis_data['exponent'])


# FLOTATION PROCESS CALCULATIONS
def flotation_rates(throughput_tpa):
    """
    Calculate all flotation flow rates.

    Args:
        throughput_tpa: Throughput in tonnes per annum

    Returns:
        dict with keys:
            - 'feed_rate': Feed volumetric flow rate (m3/s)
            - 'water_rate': Water volumetric flow rate (m3/s)
            - 'total_rate': Total volumetric flow rate (m3/s)
    """
    feed_rate = throughput_tpa / feed_density / seconds_per_op_year
    water_rate = water_flow_rate_multiplier * feed_rate
    total_rate = feed_rate + water_rate

    return {
        'feed_rate': feed_rate,
        'water_rate': water_rate,
        'total_rate': total_rate
    }


def flotation_volume_required(throughput_tpa):
    """
    Calculate flotation tank volume required (with froth excess factor).

    Args:
        throughput_tpa: Throughput in tonnes per annum

    Returns:
        Volume in m3
    """
    rates = flotation_rates(throughput_tpa)
    base_volume = rates['total_rate'] * residence_time_flotation_sec
    return (1 + froth_factor_excess) * base_volume


# LEACHING PROCESS CALCULATIONS
def leaching_rates(throughput_tpa, pb_loss_ppm=20000, density_kg_m3=None):
    """
    Calculate all leaching flow rates.

    Args:
        throughput_tpa: Throughput in tonnes per annum
        pb_loss_ppm: Lead loss in ppm (default 20000 from element data)
        density_kg_m3: Acid density (default: nitric_acid_density)

    Returns:
        dict with keys:
            - 'solids_rate': Solids volumetric flow rate (m3/s)
            - 'acid_rate': Acid volumetric flow rate (m3/s) - actual pure acid
            - 'solution_rate': Acid solution volumetric flow rate (m3/s)
            - 'total_rate': Total volumetric flow rate (m3/s)
    """
    if density_kg_m3 is None:
        density_kg_m3 = sulfuric_acid_density

    # Calculate feed rate from flotation
    feed_rate = throughput_tpa / feed_density / seconds_per_op_year

    # Calculate solids rate (accounting for Pb loss)
    loss_fraction = pb_loss_ppm / 1000000
    solids_rate = feed_rate * (1 - loss_fraction)

    # Calculate acid solution rate (liquid phase in slurry)
    solids_mass_rate = solids_rate * feed_density * 1000  # kg/s
    solution_mass_rate = ((1 - sl_ratio) / sl_ratio) * solids_mass_rate  # kg/s of acid solution
    solution_rate = solution_mass_rate / density_kg_m3  # m3/s of acid solution

    # Calculate actual acid consumption (only the acid in the solution, not the water)
    acid_mass_rate = solution_mass_rate * acid_concentration  # kg/s of pure acid
    acid_rate = acid_mass_rate / density_kg_m3  # m3/s of pure acid equivalent

    # Calculate total rate
    total_rate = solids_rate + solution_rate

    return {
        'solids_rate': solids_rate,
        'acid_rate': acid_rate,  # Pure acid consumption
        'solution_rate': solution_rate,  # Total acid solution volume
        'total_rate': total_rate
    }


def leaching_volume_required(throughput_tpa, pb_loss_ppm=20000, density_kg_m3=None):
    """
    Calculate leaching tank volume required.

    Args:
        throughput_tpa: Throughput in tonnes per annum
        pb_loss_ppm: Lead loss in ppm (default 20000 from element data)
        density_kg_m3: Acid density (default: nitric_acid_density)

    Returns:
        Volume in m3
    """
    rates = leaching_rates(throughput_tpa, pb_loss_ppm, density_kg_m3)
    return rates['total_rate'] * residence_time_leaching_sec


# SOLVENT EXCHANGE CALCULATIONS
def solvent_exchange_rate(throughput_tpa, pb_zc_loss_ppm=50000, density_kg_m3=None):
    """
    Calculate volumetric rate for solvent exchange.
    Accounts for Pb, Zc, and other losses.

    Args:
        throughput_tpa: Throughput in tonnes per annum
        pb_zc_loss_ppm: Combined Pb + Zc + other losses in ppm (default 50000)
        density_kg_m3: Acid density (default: nitric_acid_density)

    Returns:
        Volumetric flow rate in m3/s
    """
    # Get leaching rates (which includes acid rate)
    leaching_data = leaching_rates(throughput_tpa, pb_loss_ppm=20000, density_kg_m3=density_kg_m3)
    acid_rate = leaching_data['acid_rate']

    # Calculate feed rate from flotation
    feed_rate = throughput_tpa / feed_density / seconds_per_op_year

    # Apply losses
    loss_fraction = pb_zc_loss_ppm / 1000000
    return acid_rate + feed_rate * (1 - loss_fraction)


def solvent_exchange_separation_factor_power(stage):
    """
    Calculate separation factor power for given stage.

    Args:
        stage: Stage number (1-5)

    Returns:
        Separation factor power (1/SF^stage)
    """
    return 1 / (sf_heavies_vs_lights ** stage)

