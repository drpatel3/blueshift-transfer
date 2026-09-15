import numpy as np
from scipy.optimize import differential_evolution, minimize
from process_cost_integration import (
    GrindingStage, FlotationStage, LeachingStage,
    SolventExtractionStage, ElectrowinningStage
)
from flotation_recovery import flot_rec
from leaching import leaching_copper_recovery
from solvent_extraction import solvent_extraction
import warnings
warnings.filterwarnings('ignore')


def calculate_npv(capex, opex_per_year, copper_tons_per_year,
                  copper_price=13000, mine_life_years=25, discount_rate=0.08):
    """
    Calculate Net Present Value (NPV) for copper production project.

    Args:
        capex: Total upfront capital cost ($)
        opex_per_year: Annual operating cost ($/year)
        copper_tons_per_year: Copper production (tons/year)
        copper_price: Price per ton Cu ($/ton) - default $13,000
        mine_life_years: Mine lifetime (years) - default 25
        discount_rate: Discount rate (fraction) - default 0.08 (8%)

    Returns:
        float: NPV in dollars
    """
    annual_revenue = copper_tons_per_year * copper_price
    annual_cash_flow = annual_revenue - opex_per_year

    # Calculate discounted cash flows
    npv = -capex  # Year 0: CAPEX outflow

    for year in range(1, mine_life_years + 1):
        discount_factor = (1 + discount_rate) ** year
        npv += annual_cash_flow / discount_factor

    return npv


def calculate_irr(capex, opex_per_year, copper_tons_per_year,
                  copper_price=13000, mine_life_years=25):
    """
    Calculate Internal Rate of Return (IRR) using Newton-Raphson method.

    Args:
        capex: Total upfront capital cost ($)
        opex_per_year: Annual operating cost ($/year)
        copper_tons_per_year: Copper production (tons/year)
        copper_price: Price per ton Cu ($/ton)
        mine_life_years: Mine lifetime (years)

    Returns:
        float: IRR as a fraction (e.g., 0.15 = 15%)
    """
    annual_revenue = copper_tons_per_year * copper_price
    annual_cash_flow = annual_revenue - opex_per_year

    # discount rate
    rate = 0.08

    # Newton-Raphson iteration
    for _ in range(100):
        npv = -capex
        npv_derivative = 0

        for year in range(1, mine_life_years + 1):
            discount_factor = (1 + rate) ** year
            npv += annual_cash_flow / discount_factor
            npv_derivative -= year * annual_cash_flow / ((1 + rate) ** (year + 1))

        if abs(npv) < 1:  # Converged
            break

        rate = rate - npv / npv_derivative

    return rate


def calculate_payback_period(capex, opex_per_year, copper_tons_per_year,
                              copper_price=13000):
    """
    Calculate simple payback period in years.

    Args:
        capex: Total upfront capital cost ($)
        opex_per_year: Annual operating cost ($/year)
        copper_tons_per_year: Copper production (tons/year)
        copper_price: Price per ton Cu ($/ton)

    Returns:
        float: Payback period in years
    """
    annual_revenue = copper_tons_per_year * copper_price
    annual_cash_flow = annual_revenue - opex_per_year

    if annual_cash_flow <= 0:
        return float('inf')  # Never pays back

    return capex / annual_cash_flow


class ElectrolyzerStage:
    """
    Electrolyzer stage: Produces sulfuric acid for leaching, reducing acid costs.

    The electrolyzer generates H2SO4 through water electrolysis, drastically reducing
    the need to purchase acid. This impacts OPEX through reduced acid costs and
    increased power/water costs, plus CAPEX for the electrolyzer equipment.
    """

    def __init__(self, power_price_per_kwh=0.05, water_price_per_m3=0.3):
        """
        Args:
            power_price_per_kwh: Electricity price ($/kWh), default $50/MWh = $0.05/kWh
            water_price_per_m3: Water price ($/m³), default $0.3/m³
        """
        self.power_price = power_price_per_kwh
        self.water_price = water_price_per_m3

    def calculate_capex(self, acid_required_tpa):
        """
        Calculate CAPEX for electrolyzer based on required acid production capacity.

        Args:
            acid_required_tpa: Acid required (tons/year)

        Returns:
            float: Installed CAPEX ($)
        """
        # Convert to kg/s
        kg_per_second = (acid_required_tpa * 1000) / (365 * 24 * 3600)

        # Calculate molar flow rate
        molar_mass_H2SO4 = 98.08  # g/mol
        molar_flow_rate = (kg_per_second * 1000) / molar_mass_H2SO4  # mol/s

        # Calculate current required
        F = 96485  # C/mol
        z = 2  # electron transfer for H2SO4
        current_required = molar_flow_rate * F * z  # Amperes

        # Calculate area required
        current_density = 4000  # A/m²
        area_required = current_required / current_density  # m²

        # CAPEX calculation with power-law scaling
        capex_per_m2 = 1500  # $/m²
        basis_area = 50  # m²
        exponent = 0.85
        capex_cost = capex_per_m2 * basis_area * ((area_required / capex_per_m2) ** exponent)

        return capex_cost

    def calculate_opex(self, acid_required_tpa):
        """
        Calculate OPEX for electrolyzer (power + water costs).

        Args:
            acid_required_tpa: Acid required (tons/year)

        Returns:
            dict with breakdown of operating costs
        """
        # Convert to kg/s
        kg_per_second = (acid_required_tpa * 1000) / (365 * 24 * 3600)

        # Calculate molar flow rate
        molar_mass_H2SO4 = 98.08  # g/mol
        molar_flow_rate = (kg_per_second * 1000) / molar_mass_H2SO4  # mol/s

        # Energy calculation
        F = 96485  # C/mol
        z = 2  # electron transfer
        current_required = molar_flow_rate * F * z  # Amperes
        cell_potential = 2  # Volts
        power_required = (current_required * cell_potential) / 1000  # kW

        operating_capacity = 0.9
        operating_hours_per_year = 365 * 24 * operating_capacity  # hours/year
        energy_required = power_required * operating_hours_per_year  # kWh/year
        energy_cost = energy_required * self.power_price  # $/year

        # Water calculation
        acid_concentration = 0.5  # M (mol/L)
        acid_solution_volumetric_flow = molar_flow_rate / (acid_concentration * 1000)  # m³/s
        water_volumetric_flow = acid_solution_volumetric_flow * 1.2185  # m³/s (from Excel model)

        operating_seconds_per_year = 365 * 24 * 3600 * operating_capacity  # s/year
        annual_water_use = water_volumetric_flow * operating_seconds_per_year  # m³/year
        water_cost = annual_water_use * self.water_price  # $/year

        return {
            'power_cost': energy_cost,
            'water_cost': water_cost,
            'energy_kwh_per_year': energy_required,
            'water_m3_per_year': annual_water_use,
            'acid_produced_tpa': acid_required_tpa,
            'total': energy_cost + water_cost
        }


class OptimalTEA:
    """Optimization framework for copper production process."""

    def __init__(self, target_cu_tons=10000, feed_grade=0.0058, concentrate_grade=0.28,
                 ore_feed_tpa=None, use_electrolyzer=False, feed_particle_size=1000):
        """
        Initialize optimization parameters.

        Args:
            target_cu_tons: Target copper production (tons/year)
            feed_grade: Feed ore copper grade (fraction, e.g., 0.0058 = 0.58%)
            concentrate_grade: Concentrate copper grade (fraction, e.g., 0.28 = 28%)
            ore_feed_tpa: Fixed ore feed rate (tons/year). If None, will be calculated dynamically.
            use_electrolyzer: If True, use electrolyzer to produce acid (reduces acid cost significantly)
            feed_particle_size: Feed particle size entering the grinder (microns). Product size must be <= this.
        """
        self.target_cu_tons = target_cu_tons
        self.feed_grade = feed_grade
        self.concentrate_grade = concentrate_grade
        self.fixed_ore_feed = ore_feed_tpa
        self.use_electrolyzer = use_electrolyzer
        self.feed_particle_size = feed_particle_size

        # Define optimization bounds for 23 parameters
        # [Grinding (1), Flotation (8), Leaching (6), SX (4), EW (4)]
        self.bounds = [
            # Grinding parameters
            # Product size must be <= feed size (grinder reduces particle size)
            # Minimum is 25 microns (practical lower limit for grinding)
            (25, feed_particle_size),       # product_size (microns)

            # Flotation parameters
            (0.8, 1.2),      # sp_power (kW/m³)
            (0.8, 1.5),      # sp_gas_rate (cm/s)
            (10, 50),        # frother_conc (g/l)
            (15, 20),        # ret_time (min)
            (0.10, 0.25),    # air_fraction
            (0.15, 0.30),    # slurry_fraction
            (20, 55),        # contact_angle (degrees)
            (100, 250),      # cell_volume (m³)
            # NOTE: num_cells is CALCULATED from throughput and cell_volume

            # Leaching parameters
            (190, 190),      # T (°C)
            (12, 15),        # P_O2 (bar)
            (70, 90),        # Ea (kJ/mol)
            (0.5, .6),       # n (reaction order)
            (0.1, 0.3),      # H_plus (mol/L)
            (2, 12),         # leach_time (hours)

            # Solvent Extraction parameters
            (2, 3),          # K_ex (equilibrium constant)
            (0.05, 0.10),    # RH (extractant concentration, M)
            (.8, 1.0),    # O_A (organic/aqueous ratio)
            (1.5, 1.8),      # pH

            # Electrowinning parameters
            (4000,4000),      # current_density (A/m²)
            (0.90, 0.98),    # current_efficiency
            (1.8, 2.2),      # cell_potential (V)
            (0.8, 0.98)     # ew_recovery
        ]

        # Initialize process stage objects (grinding will be updated with optimized params)
        self.grinding = None  # Will be initialized with optimized product_size
        self.flotation = FlotationStage()

    
        if self.use_electrolyzer:
            self.electrolyzer = ElectrolyzerStage()
            # With electrolyzer: acid cost is $0 (we produce it ourselves)
            # The cost is captured in electrolyzer CAPEX/OPEX
            self.leaching = LeachingStage(acid_price_per_tonne=0, acid_recovery_fraction=0)
            # SX still needs organic solvent ($5/L) regardless of electrolyzer
            self.sx = SolventExtractionStage(solvent_price_per_litre=5.0, solvent_loss_fraction=0.01)
        else:
            # Without electrolyzer: buy acid at market price
            self.leaching = LeachingStage(acid_price_per_tonne=420, acid_recovery_fraction=0.5)
            # SX uses same organic solvent ($5/L) in both cases
            self.sx = SolventExtractionStage(solvent_price_per_litre=5.0, solvent_loss_fraction=0.01)

        self.ew = ElectrowinningStage()

    def unpack_parameters(self, params):
        """Unpack parameter vector into stage-specific dictionaries."""
        grinding_params = {
            'product_size': params[0]
        }

        flotation_params = {
            'sp_power': params[1],
            'sp_gas_rate': params[2],
            'frother_conc': params[3],
            'ret_time': params[4],
            'air_fraction': params[5],
            'slurry_fraction': params[6],
            'particle_size': params[0],  # Use grinding product_size output!
            'contact_angle': params[7],
            'cell_volume': params[8]
            # NOTE: num_cells will be calculated in simulate_process based on throughput
        }

        leaching_params = {
            'T': params[9],
            'P_O2': params[10],
            'Ea': params[11],
            'n': params[12],
            'H_plus': params[13],
            'leach_time': params[14]
        }

        sx_params = {
            'K_ex': params[15],
            'RH': params[16],
            'O_A': params[17],
            'pH': params[18]
        }

        ew_params = {
            'current_density': params[19],
            'current_efficiency': params[20],
            'cell_potential': params[21],
            'ew_recovery': params[22]
        }

        return grinding_params, flotation_params, leaching_params, sx_params, ew_params

    def calculate_required_feed(self, grinding_params, flotation_params, leaching_params, sx_params, ew_params):
        """
        Calculate ore feed required to achieve target Cu production.
        Works backwards from target through recovery chain using actual parameter values.
        """
        # Conservative estimates based on parameter ranges
        # These will be refined in the full simulation

        # Flotation recovery: assume mid-range (will be calculated exactly later)
        flot_recovery_est = 0.90

        # Leaching recovery: use actual parameters to estimate
        # Typical recovery at good conditions
        leach_recovery_est = 0.92

        # SX recovery estimate (from equilibrium)
        H_plus = 10**(-sx_params['pH'])
        denominator = 1 + sx_params['K_ex'] * (sx_params['RH'] ** 2) / (H_plus ** 2) * sx_params['O_A']
        sx_recovery_est = max(0.01, min(0.90, 1 - (1 / denominator)))  # Bound between 1-99%

        # EW recovery (from parameters)
        ew_recovery_est = ew_params['ew_recovery']

        # Overall recovery
        overall_recovery = flot_recovery_est * leach_recovery_est * sx_recovery_est * ew_recovery_est

        # Ensure we don't divide by zero or get unrealistic values
        overall_recovery = max(0.01, min(0.98, overall_recovery))

        # Calculate required ore feed to produce target_cu_tons
        # Cu in ore = ore_feed × feed_grade
        # Cu produced = ore_feed × feed_grade × overall_recovery
        # Therefore: ore_feed = target_cu / (feed_grade × overall_recovery)

        ore_feed_tons = self.target_cu_tons / (self.feed_grade * overall_recovery)

        return ore_feed_tons

    def simulate_process(self, params):
        """
        Simulate entire process chain and calculate costs.

        Returns:
            dict: Contains costs, production, recoveries, and constraint violations
        """
        # Unpack parameters
        grinding_params, flotation_params, leaching_params, sx_params, ew_params = self.unpack_parameters(params)

        # Calculate required ore feed by working backwards from target production
        # Uses estimated recoveries based on operating parameters
        ore_feed_tpa = self.calculate_required_feed(grinding_params, flotation_params, leaching_params, sx_params, ew_params)

        # Stage 1: Grinding - initialize with optimized product size
        grinding_stage = GrindingStage(product_size_um=grinding_params['product_size'])
        grinding_capex = grinding_stage.calculate_capex(ore_feed_tpa)
        grinding_opex = grinding_stage.calculate_opex(ore_feed_tpa)

        # Stage 2: Flotation
        cu_in_feed_kg = ore_feed_tpa * 1000 * self.feed_grade

        # Calculate required number of cells based on throughput, retention time, and cell volume
        # Logic:
        #   - total_rate (m³/s) = volumetric flow rate through flotation system
        #   - retention_time (s) = how long material stays in the system
        #   - total_volume_in_system = flow_rate × retention_time × (1 + froth_excess)
        #   - num_cells = total_volume_in_system / cell_volume
        # This ensures we have enough cell capacity to handle the continuous throughput
        # while maintaining the required retention time
        from tea import flotation_rates, froth_factor_excess
        import math

        ret_time_sec = flotation_params['ret_time'] * 60  # Convert minutes to seconds
        rates = flotation_rates(ore_feed_tpa)

        # Total volume that must be held in the flotation system at any given time
        # This accounts for the continuous flow × residence time × froth excess factor
        total_volume_in_system = rates['total_rate'] * ret_time_sec * (1 + froth_factor_excess)

        # Number of cells = total volume needed / volume per cell (round up)
        num_cells_required = math.ceil(total_volume_in_system / flotation_params['cell_volume'])

        # Add calculated num_cells to flotation_params (minimum 1 cell)
        flotation_params['num_cells'] = max(1, num_cells_required)

        # Prepare flotation function inputs (from flotation_recovery.py)
        fitting_params = {
            'b': 2,
            'alpha': 0.08,
            'coverage': 0.525,  
            'bubble_f': 0.825,
            'detach_f': 0.7,
            'bulk_zone': 0.5
        }
        flot_constants = {'specific_gravity': 4.2, 'grade': self.concentrate_grade, 'permitivity': 8.854e-12,
                         'dielectric': 80, 'pe': 4, 'cell_area': 200, 'bbl_ratio': 10,
                         'feed_tonnage': ore_feed_tpa}
        opt_vars = {
            'particle_size': flotation_params['particle_size'],
            'contact_angle': flotation_params['contact_angle'],
            'bubble_z_pot': -0.03,
            'particle_z_pot': -50,
            'sp_power': flotation_params['sp_power'],
            'sp_gas_rate': flotation_params['sp_gas_rate'],
            'air_fraction': flotation_params['air_fraction'],
            'slurry_fraction': flotation_params['slurry_fraction'],
            'num_cells': flotation_params['num_cells'],
            'ret_time': flotation_params['ret_time'],
            'cell_volume': flotation_params['cell_volume'],  # Now optimized (100-250 m³)
            'froth_height': 0.165,
            'frother_conc': flotation_params['frother_conc']
        }
        inputs_dict = {'frother_type': 4, 'water_or_particle': 'Particle', 'copper_grade': self.concentrate_grade}

        flotation_result = flot_rec(fitting_params, flot_constants, opt_vars, inputs_dict)
        flotation_recovery_raw = flotation_result[0]  # Extract recovery from tuple

        # Cap flotation recovery at 95% (physically realistic maximum)
        flotation_recovery = min(flotation_recovery_raw, 0.95)

        cu_in_concentrate_kg = cu_in_feed_kg * flotation_recovery
        concentrate_tpa = (cu_in_concentrate_kg / 1000) / self.concentrate_grade

        flotation_capex = self.flotation.calculate_capex(
            ore_feed_tpa,
            ret_time=flotation_params['ret_time'],
            num_cells=flotation_params['num_cells'],
            cell_volume=flotation_params['cell_volume']
        )
        flotation_opex_dict = self.flotation.calculate_opex(
            ore_feed_tpa,
            sp_power=flotation_params['sp_power'],
            ret_time=flotation_params['ret_time'],
            num_cells=flotation_params['num_cells'],
            frother_conc=None,  # Frother cost excluded
            cell_volume=flotation_params['cell_volume']
        )
        flotation_opex = flotation_opex_dict['total']

        # Stage 3: Leaching
        # leaching_copper_recovery returns (X_Cu_final, solution)
        leaching_result_tuple = leaching_copper_recovery(
            Ea=leaching_params['Ea'],
            T=leaching_params['T'],
            P_O2=leaching_params['P_O2'],
            n=leaching_params['n'],
            H_plus=leaching_params['H_plus'],
            MFeS2=0.3,  # Typical mass fraction of FeS2
            X_Cu=0.0,   # Initial copper conversion
            Phi=0.1,    # Mass of solids per solution volume (kg/L)
            k0=250,     # Pre-exponential factor (tuned to give 20-95% recovery range)
            leach_time=leaching_params['leach_time']
        )
        leaching_recovery = leaching_result_tuple[0]  # Extract X_Cu_final

        cu_leached_kg = cu_in_concentrate_kg * leaching_recovery

        # Leaching costs
        leaching_capex = self.leaching.calculate_capex(
            concentrate_tpa,
            leach_time=leaching_params['leach_time']
        )
        leaching_opex_dict = self.leaching.calculate_opex(cu_leached_kg)
        leaching_opex = leaching_opex_dict['acid_cost']

        # Stage 4: Solvent Extraction
        cu_in_leach_solution = cu_leached_kg  # mol of Cu
        cu_out_raffinate = solvent_extraction(
            C_Cu_in_aq=cu_in_leach_solution,
            K_ex=sx_params['K_ex'],
            RH=sx_params['RH'],
            O_A=sx_params['O_A'],
            pH=sx_params['pH']
        )

        cu_extracted_kg = cu_in_leach_solution - cu_out_raffinate
        sx_recovery = cu_extracted_kg / cu_in_leach_solution if cu_in_leach_solution > 0 else 0

        # SX costs
        cu_leached_tpa = cu_leached_kg / 1000
        sx_capex = self.sx.calculate_capex(cu_leached_tpa)
        solution_volume_m3 = cu_leached_kg * 0.001  # Estimate solution volume
        sx_opex_dict = self.sx.calculate_opex(cu_extracted_kg, solution_volume_m3)
        sx_opex = sx_opex_dict['solvent_cost']

        # Electrolyzer costs (if enabled) - calculate AFTER leaching costs
        if self.use_electrolyzer:
            # Electrolyzer produces acid ONLY for leaching (SX uses organic solvent)
            leaching_acid_tpa = leaching_opex_dict['acid_net_consumption_tpa']

            # Total acid requirement for electrolyzer (only leaching)
            acid_required_tpa = leaching_acid_tpa

            electrolyzer_capex = self.electrolyzer.calculate_capex(acid_required_tpa)
            electrolyzer_opex_dict = self.electrolyzer.calculate_opex(acid_required_tpa)
            electrolyzer_opex = electrolyzer_opex_dict['total']

            # Store breakdown for debugging
            electrolyzer_opex_dict['leaching_acid_tpa'] = leaching_acid_tpa
            electrolyzer_opex_dict['total_acid_tpa'] = acid_required_tpa
        else:
            electrolyzer_capex = 0
            electrolyzer_opex = 0
            electrolyzer_opex_dict = {}

        # Stage 5: Electrowinning
        cu_plated_kg = cu_extracted_kg * ew_params['ew_recovery']
        cu_plated_tons = cu_plated_kg / 1000

        ew_capex = self.ew.calculate_capex(cu_extracted_kg)
        # calculate_opex uses default parameters (E_eq=0.34, eta_cath=0.1, eta_anode=0.3, IR=0.5)
        # Our optimization parameters (current_density, current_efficiency, cell_potential) affect recovery
        # but don't directly map to the OPEX calculation parameters
        ew_opex_dict = self.ew.calculate_opex(cu_plated_tons)
        ew_opex = ew_opex_dict['total']

        # Ore feed cost (mining/processing raw material)
        ore_cost_per_tonne = 15  # $/tonne of ore feed
        ore_feed_opex = ore_feed_tpa * ore_cost_per_tonne

        # Total costs
        total_capex = grinding_capex + flotation_capex + leaching_capex + sx_capex + ew_capex + electrolyzer_capex

        # O&M cost: 3.5% of total CAPEX per year
        om_cost = total_capex * 0.035

        total_opex = ore_feed_opex + grinding_opex + flotation_opex + leaching_opex + sx_opex + ew_opex + electrolyzer_opex + om_cost
        total_cost = total_capex + total_opex

        # Calculate energy costs breakdown
        energy_costs = {
            'grinding': grinding_opex,  # Grinding OPEX is primarily energy
            'flotation': flotation_opex_dict.get('power', 0),
            'ew': ew_opex_dict.get('power_cost', 0),
            'electrolyzer': electrolyzer_opex_dict.get('power_cost', 0) if self.use_electrolyzer else 0
        }
        total_energy_cost = sum(energy_costs.values())

        # Calculate water costs breakdown
        water_costs = {
            'flotation': flotation_opex_dict.get('water', 0),
            'electrolyzer': electrolyzer_opex_dict.get('water_cost', 0) if self.use_electrolyzer else 0
        }
        total_water_cost = sum(water_costs.values())

        # Calculate constraint violation (both over and under production)
        production_shortfall = max(0, self.target_cu_tons - cu_plated_tons)
        production_excess = max(0, cu_plated_tons - self.target_cu_tons)
        production_deviation = abs(cu_plated_tons - self.target_cu_tons)

        return {
            'total_cost': total_cost,
            'total_capex': total_capex,
            'total_opex': total_opex,
            'ore_feed_tpa': ore_feed_tpa,
            'cu_produced_tons': cu_plated_tons,
            'production_shortfall': production_shortfall,
            'production_excess': production_excess,
            'production_deviation': production_deviation,
            'recoveries': {
                'flotation': flotation_recovery,
                'leaching': leaching_recovery,
                'sx': sx_recovery,
                'ew': ew_params['ew_recovery'],
                'overall': (cu_plated_tons / (ore_feed_tpa * self.feed_grade)) if ore_feed_tpa > 0 else 0
            },
            'stage_costs': {
                'ore_feed': {'capex': 0, 'opex': ore_feed_opex},
                'grinding': {'capex': grinding_capex, 'opex': grinding_opex},
                'flotation': {'capex': flotation_capex, 'opex': flotation_opex},
                'leaching': {'capex': leaching_capex, 'opex': leaching_opex},
                'sx': {'capex': sx_capex, 'opex': sx_opex},
                'ew': {'capex': ew_capex, 'opex': ew_opex},
                'electrolyzer': {'capex': electrolyzer_capex, 'opex': electrolyzer_opex, 'enabled': self.use_electrolyzer},
                'om': {'capex': 0, 'opex': om_cost}
            },
            'energy_costs': {
                'grinding': energy_costs['grinding'],
                'flotation': energy_costs['flotation'],
                'ew': energy_costs['ew'],
                'electrolyzer': energy_costs['electrolyzer'],
                'total': total_energy_cost
            },
            'water_costs': {
                'flotation': water_costs['flotation'],
                'electrolyzer': water_costs['electrolyzer'],
                'total': total_water_cost
            },
            'calculated_parameters': {
                'grinding': grinding_params,
                'flotation': flotation_params,
                'leaching': leaching_params,
                'sx': sx_params,
                'ew': ew_params
            }
        }

    def objective_function(self, params):
        """
        Objective function for optimization: MAXIMIZE NPV while PRODUCING EXACTLY target amount.
        Returns -NPV because scipy.optimize minimizes (maximizing NPV = minimizing -NPV).

        CRITICAL: Must produce EXACTLY target_cu_tons/year. No more, no less.
        Heavy penalties applied for ANY deviation from target (both over and under production).
        """
        result = self.simulate_process(params)

        # Calculate NPV
        capex = result['total_capex']
        opex_per_year = result['total_opex']
        copper_tons_per_year = result['cu_produced_tons']

        # NPV calculation with 25-year mine life, $13,000/ton Cu, 8% discount rate
        npv = calculate_npv(
            capex=capex,
            opex_per_year=opex_per_year,
            copper_tons_per_year=copper_tons_per_year,
            copper_price=13000,
            mine_life_years=25,
            discount_rate=0.08
        )

        # CRITICAL: Apply MASSIVE penalties for ANY deviation from exact target
        production_deviation = abs(copper_tons_per_year - self.target_cu_tons)

        # Quadratic penalty to heavily discourage deviation
        # $50M per ton deviation + $10M per ton^2 for exponential discouragement
        linear_penalty = production_deviation * 5e11
        quadratic_penalty = (production_deviation ** 2) * 5e11

        total_penalty = linear_penalty + quadratic_penalty

        # Penalized NPV
        penalized_npv = npv - total_penalty

        # Return negative penalized NPV (to maximize via minimization)
        return -penalized_npv

    def run_optimization(self, method='differential_evolution', maxiter=30, verbose=True):
        """
        Run optimization to find parameters that maximize NPV.

        Args:
            method: Optimization method ('differential_evolution' or 'minimize')
            maxiter: Maximum iterations
            verbose: If True, print progress information

        Returns:
            dict: Optimization results including optimal parameters and NPV
        """
        # Create a unique identifier for this optimization run
        opt_id = f"[{'ELEC' if self.use_electrolyzer else 'NO-ELEC'}]"

        if verbose:
            print("="*80)
            print(f"{opt_id} COPPER PRODUCTION OPTIMIZATION - MAXIMIZE NPV")
            print("="*80)
            print(f"\nEconomic Parameters:")
            print(f"  Copper price: $13,000/ton")
            print(f"  Mine lifetime: 25 years")
            print(f"  Discount rate: 8%")
            print(f"\nTarget Production: {self.target_cu_tons:,.0f} tons Cu/year")
            print(f"Feed Grade: {self.feed_grade*100:.2f}%")
            print(f"Concentrate Grade: {self.concentrate_grade*100:.1f}%")
            print(f"\nOptimizing {len(self.bounds)} parameters across 5 process stages...")
            print(f"Method: {method}")
            print(f"\n{opt_id} Running optimization (this may take several minutes)...\n")

        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function,
                self.bounds,
                maxiter=maxiter,
                popsize=10,
                tol=0.01,
                atol=0,
                seed=42,
                workers=1,
                disp=False  # Set to False to prevent interleaved output when running multiple optimizations
            )
            optimal_params = result.x
            optimal_cost = result.fun
        else:
            # Use minimize with random initial guess
            x0 = np.array([np.mean(b) for b in self.bounds])
            result = minimize(
                self.objective_function,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds,
                options={'maxiter': maxiter, 'disp': False}  # Set to False to prevent interleaved output
            )
            optimal_params = result.x
            optimal_cost = result.fun

        # Simulate with optimal parameters
        optimal_result = self.simulate_process(optimal_params)

        return {
            'optimal_params': optimal_params,
            'optimal_cost': optimal_cost,
            'result': optimal_result,
            'optimization_output': result
        }

    def analyze_results(self, optimization_output):
        """
        Print detailed analysis of optimization results.
        """
        result = optimization_output['result']

        # Use calculated parameters (includes num_cells) instead of raw unpacked parameters
        grinding_params = result['calculated_parameters']['grinding']
        flotation_params = result['calculated_parameters']['flotation']
        leaching_params = result['calculated_parameters']['leaching']
        sx_params = result['calculated_parameters']['sx']
        ew_params = result['calculated_parameters']['ew']

        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)

        # Production summary
        print("\n--- PRODUCTION SUMMARY ---")
        print(f"Ore Feed Required: {result['ore_feed_tpa']:,.0f} tons/year")
        print(f"Copper Produced: {result['cu_produced_tons']:,.2f} tons/year")
        print(f"Target: {self.target_cu_tons:,.0f} tons/year")

        deviation = result['cu_produced_tons'] - self.target_cu_tons
        achievement_pct = (result['cu_produced_tons']/self.target_cu_tons)*100

        print(f"Achievement: {achievement_pct:.2f}%")
        if abs(deviation) > 0.01:  # More than 10 kg deviation
            if deviation > 0:
                print(f"DEVIATION: +{deviation:.2f} tons OVER target (EXCESS PRODUCTION)")
            else:
                print(f"DEVIATION: {deviation:.2f} tons UNDER target (SHORTFALL)")
        else:
            print(f"DEVIATION: {deviation:.4f} tons (EXCELLENT - within target)")

        # Recovery summary
        print("\n--- RECOVERY RATES ---")
        print(f"Flotation: {result['recoveries']['flotation']*100:.2f}%")
        print(f"Leaching: {result['recoveries']['leaching']*100:.2f}%")
        print(f"Solvent Extraction: {result['recoveries']['sx']*100:.2f}%")
        print(f"Electrowinning: {result['recoveries']['ew']*100:.2f}%")
        print(f"Overall: {result['recoveries']['overall']*100:.2f}%")

        # Economic analysis
        capex = result['total_capex']
        opex_per_year = result['total_opex']
        cu_produced = result['cu_produced_tons']

        npv = calculate_npv(capex, opex_per_year, cu_produced, copper_price=13000, mine_life_years=25, discount_rate=0.08)
        irr = calculate_irr(capex, opex_per_year, cu_produced, copper_price=13000, mine_life_years=25)
        payback = calculate_payback_period(capex, opex_per_year, cu_produced, copper_price=13000)
        annual_revenue = cu_produced * 13000
        annual_cash_flow = annual_revenue - opex_per_year

        print("\n--- ECONOMIC ANALYSIS ---")
        print(f"NPV (25 years, 8% discount): ${npv:,.2f}")
        print(f"IRR: {irr*100:.2f}%")
        print(f"Payback Period: {payback:.2f} years")
        print(f"\nAnnual Revenue: ${annual_revenue:,.2f}/year")
        print(f"Annual OPEX: ${opex_per_year:,.2f}/year")
        print(f"Annual Cash Flow: ${annual_cash_flow:,.2f}/year")

        print("\n--- COST BREAKDOWN ---")
        print(f"Total CAPEX: ${capex:,.2f}")
        print(f"Total OPEX: ${opex_per_year:,.2f}/year")
        print(f"\nCost per ton Cu: ${(capex/25 + opex_per_year)/cu_produced:,.2f}/ton (amortized)")
        print(f"Revenue per ton Cu: $13,000/ton")
        print(f"Profit per ton Cu: ${13000 - (capex/25 + opex_per_year)/cu_produced:,.2f}/ton")

        print("\n--- STAGE-WISE COSTS ---")
        for stage, costs in result['stage_costs'].items():
            if stage != 'om':  # O&M shown separately in OPEX breakdown
                print(f"{stage.capitalize():15s} CAPEX: ${costs['capex']:>12,.2f}  OPEX: ${costs['opex']:>12,.2f}/year")

        print("\n--- OPEX BREAKDOWN BY CATEGORY ---")
        energy_total = result['energy_costs']['total']
        water_total = result['water_costs']['total']
        om_total = result['stage_costs']['om']['opex']
        other_opex = opex_per_year - energy_total - water_total - om_total

        print(f"  Energy:        ${energy_total:>12,.2f}/year  ({energy_total/opex_per_year*100:5.1f}%)")
        print(f"  Water:         ${water_total:>12,.2f}/year  ({water_total/opex_per_year*100:5.1f}%)")
        print(f"  O&M (3.5% CAPEX): ${om_total:>9,.2f}/year  ({om_total/opex_per_year*100:5.1f}%)")
        print(f"  Other:         ${other_opex:>12,.2f}/year  ({other_opex/opex_per_year*100:5.1f}%)")
        print(f"  ----------------------------------------")
        print(f"  TOTAL OPEX:    ${opex_per_year:>12,.2f}/year")

        # Optimal parameters
        print("\n--- OPTIMAL PARAMETERS ---")
        print("\nGrinding:")
        for key, value in grinding_params.items():
            print(f"  {key:20s}: {value:.4f}")

        print("\nFlotation:")
        for key, value in flotation_params.items():
            print(f"  {key:20s}: {value:.4f}")

        print("\nLeaching:")
        for key, value in leaching_params.items():
            print(f"  {key:20s}: {value:.4f}")

        print("\nSolvent Extraction:")
        for key, value in sx_params.items():
            print(f"  {key:20s}: {value:.4f}")

        print("\nElectrowinning:")
        for key, value in ew_params.items():
            print(f"  {key:20s}: {value:.4f}")

        print("\n" + "="*80)


def compare_with_and_without_electrolyzer(target_cu_tons=10000, feed_grade=0.006,
                                          concentrate_grade=0.3, maxiter=50, feed_particle_size=150):
    """
    Compare NPV optimization with and without electrolyzer for acid production.

    WITHOUT electrolyzer: Buy acid at market price ($100/tonne)
    WITH electrolyzer: Produce acid using electricity and water
    """
    print("\n" + "="*80)
    print("ELECTROLYZER COMPARISON - NPV OPTIMIZATION")
    print("="*80)
    print(f"Target Production: {target_cu_tons:,.0f} tonnes Cu/year")
    print(f"Feed Grade: {feed_grade*100:.2f}%")
    print(f"Concentrate Grade: {concentrate_grade*100:.1f}%")
    print(f"Feed Particle Size: {feed_particle_size:.0f} microns")
    print("="*80)

    # WITHOUT electrolyzer
    print("\n### SCENARIO 1: WITHOUT ELECTROLYZER (Buy Acid) ###")
    optimizer_no = OptimalTEA(
        target_cu_tons=target_cu_tons,
        feed_grade=feed_grade,
        concentrate_grade=concentrate_grade,
        use_electrolyzer=False,
        feed_particle_size=feed_particle_size
    )
    results_no = optimizer_no.run_optimization(method='differential_evolution', maxiter=maxiter)
    optimizer_no.analyze_results(results_no)
    result_no = results_no['result']
    npv_no = calculate_npv(result_no['total_capex'], result_no['total_opex'], result_no['cu_produced_tons'])

    # WITH electrolyzer
    print("\n" + "="*80)
    print("### SCENARIO 2: WITH ELECTROLYZER (Produce Acid) ###")
    optimizer_yes = OptimalTEA(
        target_cu_tons=target_cu_tons,
        feed_grade=feed_grade,
        concentrate_grade=concentrate_grade,
        use_electrolyzer=True,
        feed_particle_size=feed_particle_size
    )
    results_yes = optimizer_yes.run_optimization(method='differential_evolution', maxiter=maxiter)
    optimizer_yes.analyze_results(results_yes)
    result_yes = results_yes['result']
    npv_yes = calculate_npv(result_yes['total_capex'], result_yes['total_opex'], result_yes['cu_produced_tons'])

    # Calculate additional metrics
    irr_no = calculate_irr(result_no['total_capex'], result_no['total_opex'], result_no['cu_produced_tons'])
    payback_no = calculate_payback_period(result_no['total_capex'], result_no['total_opex'], result_no['cu_produced_tons'])

    irr_yes = calculate_irr(result_yes['total_capex'], result_yes['total_opex'], result_yes['cu_produced_tons'])
    payback_yes = calculate_payback_period(result_yes['total_capex'], result_yes['total_opex'], result_yes['cu_produced_tons'])

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nWithout Electrolyzer:")
    print(f"  Total CAPEX: ${result_no['total_capex']:,.2f}")
    print(f"  Total OPEX: ${result_no['total_opex']:,.2f}/year")
    print(f"  Acid Purchase OPEX: ${result_no['stage_costs']['leaching']['opex']:,.2f}/year")
    print(f"  NPV: ${npv_no:,.2f}")
    print(f"  IRR: {irr_no*100:.2f}%")
    print(f"  Payback Period: {payback_no:.2f} years")

    print(f"\nWith Electrolyzer:")
    print(f"  Total CAPEX: ${result_yes['total_capex']:,.2f}")
    print(f"  Total OPEX: ${result_yes['total_opex']:,.2f}/year")
    print(f"  Acid Purchase OPEX: ${result_yes['stage_costs']['leaching']['opex']:,.2f}/year")
    print(f"  Electrolyzer OPEX: ${result_yes['stage_costs']['electrolyzer']['opex']:,.2f}/year")
    print(f"  NPV: ${npv_yes:,.2f}")
    print(f"  IRR: {irr_yes*100:.2f}%")
    print(f"  Payback Period: {payback_yes:.2f} years")

    # Delta analysis
    capex_delta = result_yes['total_capex'] - result_no['total_capex']
    opex_delta = result_yes['total_opex'] - result_no['total_opex']
    npv_delta = npv_yes - npv_no

    print(f"\nDelta (With - Without):")
    print(f"  CAPEX Change: ${capex_delta:+,.2f}")
    print(f"  OPEX Change: ${opex_delta:+,.2f}/year")
    print(f"  NPV Change: ${npv_delta:+,.2f}")
    print(f"  NPV Change: ${(npv_delta / npv_no)*100:+,.2f}%")
    print(f"  IRR Change: {(irr_yes - irr_no)*100:+.2f}%")
    print(f"  Payback Change: {payback_yes - payback_no:+.2f} years")

    if npv_delta > 0:
        print(f"\n✓ RECOMMENDATION: Use electrolyzer (NPV improves by ${npv_delta:,.2f})")
    else:
        print(f"\n✗ RECOMMENDATION: Buy acid (NPV decreases by ${abs(npv_delta):,.2f} with electrolyzer)")

    print("="*80 + "\n")

    return {
        'npv_without': npv_no,
        'npv_with': npv_yes,
        'npv_improvement': npv_delta,
        'irr_without': irr_no,
        'irr_with': irr_yes,
        'payback_without': payback_no,
        'payback_with': payback_yes,
        'capex_delta': capex_delta,
        'opex_delta': opex_delta
    }


def main():
    """Main function to run optimization."""
    # Create optimizer with default target (10,000 tons Cu/year)
    optimizer = OptimalTEA(
        target_cu_tons=10000,
        feed_grade=0.0055,  
        concentrate_grade=0.25,  
        use_electrolyzer=False
    )

    # Run optimization
    results = optimizer.run_optimization(
        method='differential_evolution',
        maxiter=50  # Adjust for speed vs accuracy trade-off
    )

    # Analyze and display results
    optimizer.analyze_results(results)

    return results


if __name__ == "__main__":
    # Run electrolyzer comparison


    comparison = compare_with_and_without_electrolyzer(
        target_cu_tons=10000,
        feed_grade=.006,
        concentrate_grade=0.3,
        maxiter=30
    )