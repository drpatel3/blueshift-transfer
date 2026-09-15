"""
Process-to-Cost Integration Module - Grinding Stage

Starting with grinding to establish the pattern for connecting
physical operations to TEA costs.
"""

import numpy as np
from tea import (
    feed_size, product_size, p1, g,
    work_index, work,
    jaw_crusher_small, jaw_crusher_large,
    motor_small, motor_large,
    hours_per_op_year, electricity_cost,
    lang_factor, inflation_factor,
    equipment_cost,
    flotation_work, water_flow_rate_multiplier,
    residence_time_flotation, feed_density, froth_factor_excess,
    seconds_per_op_year, residence_time_flotation_sec,
    water_cost, flotation_volume_required, flotation_rates,
    sl_ratio, acid_concentration, sulfuric_acid_density,
    residence_time_leaching, residence_time_leaching_sec, leaching_volume_required, leaching_rates
)
from grind_break import grind_break
from flotation_recovery import flot_rec
from leaching import leaching_copper_recovery, calculate_acid_consumption
from solvent_extraction import solvent_extraction
from electrowinning import cell_voltage


class GrindingStage:
    """
    Grinding stage: Connects particle size reduction to costs.

    Physical Parameters → Cost Calculation:
    - feed_size, product_size → work (kWh/t) → OPEX (electricity)
    - work × throughput → required_power_kw → CAPEX (equipment)
    """

    def __init__(self, feed_size_um=10000, product_size_um=100, p1_um=1000, g_val=2.5):
        """
        Args:
            feed_size_um: Feed particle size (microns)
            product_size_um: Product particle size (microns)
            p1_um: Reference size (microns)
            g_val: g value (g/rev)
        """
        self.feed_size = feed_size_um
        self.product_size = product_size_um
        self.p1 = p1_um
        self.g = g_val

        # Calculate work index and specific work using Bond's equation
        self.work_index = self._calculate_work_index()
        self.work = self._calculate_work()

    def _calculate_work_index(self):
        """Bond work index (kWh/t)."""
        work_index = 13
        return work_index
    
    def _calculate_work(self):
        """Specific grinding work (kWh/t)."""
        product_size_power = self.product_size ** 0.5
        feed_size_power = self.feed_size ** 0.5

        return 10 * self.work_index * ((1 / product_size_power) - (1 / feed_size_power))

    def calculate_capex(self, throughput_tpa):
        """
        CAPEX = equipment cost for jaw crusher + motor

        Steps:
        1. Calculate required power from throughput and work
        2. Select equipment basis (small vs large)
        3. Scale cost using power-law
        4. Apply inflation and lang factors
        """
        # Step 1: Required power
        required_power_kw = throughput_tpa * self.work / hours_per_op_year

        # Step 2: Select basis
        if required_power_kw <= jaw_crusher_small['power']:
            basis = 1
        else:
            basis = 2

        # Step 3: Equipment costs with power-law scaling
        crusher_cost = equipment_cost('jaw_crusher', required_power_kw, basis)
        motor_cost = equipment_cost('motor', required_power_kw, basis if basis == 1 else 2)

        # Step 4: Apply factors
        total_equipment_cost = (crusher_cost + motor_cost) * inflation_factor
        installed_cost = total_equipment_cost * lang_factor

        return installed_cost

    def calculate_opex(self, throughput_tpa):
        """
        OPEX = annual electricity cost

        Steps:
        1. Calculate annual energy from throughput and work
        2. Convert to cost using electricity price
        """
        # Annual energy consumption
        annual_power_kwh = throughput_tpa * self.work
        annual_power_mwh = annual_power_kwh / 1000

        # Annual cost
        return annual_power_mwh * electricity_cost

    def process(self, tonnage_input, top_size_input, bottom_size_input, selection_input, break_intensity):
        """
        Calculate grinding product using population balance model.

        Uses grind_break from equation_flow_official.py to calculate particle
        size distribution after grinding.

        Args:
            tonnage_input: Feed tonnage array by size class
            top_size_input: Top size for each size class (microns)
            bottom_size_input: Bottom size for each size class (microns)
            selection_input: Selection function values
            break_intensity: Breakage intensity (number of cycles)

        Returns:
            dict with keys:
                - 'ground_tonnage': Product tonnage distribution
                - 'total_feed': Total feed tonnage
                - 'total_product': Total product tonnage
                - 'p80_size': P80 particle size (microns)
        """
        # Run grind_break equation
        ground_tonnage = grind_break(tonnage_input, top_size_input, bottom_size_input,
                                     selection_input, break_intensity)

        total_feed = sum(tonnage_input)
        total_product = np.sum(ground_tonnage)

        # Calculate P80 particle size (80% passing size)
        cumulative_mass = 0
        p80_size = 0
        for tonnage, bottom, top in zip(ground_tonnage, bottom_size_input, top_size_input):
            cumulative_mass += tonnage
            if cumulative_mass >= 0.8 * total_product:
                p80_size = (top + bottom) / 2
                break

        return {
            'ground_tonnage': ground_tonnage,
            'total_feed': total_feed,
            'total_product': total_product,
            'p80_size': p80_size,
            'mass_balance_ok': abs(total_feed - total_product) < 0.01
        }


class FlotationStage:
    """
    Flotation stage: Separates valuable minerals using bubble attachment.

    Physical Parameters → Cost Calculation:
    - throughput → tank volume → CAPEX (vertical agitated tank)
    - throughput → power + water → OPEX (electricity + water)
    """

    def __init__(self, flotation_work_kwh_per_t=20):
        """
        Args:
            flotation_work_kwh_per_t: Specific flotation work (kWh/t)
        """
        self.flotation_work = flotation_work_kwh_per_t

    def calculate_capex(self, throughput_tpa, ret_time=None, num_cells=1, cell_volume=None):
        """
        CAPEX = vertical agitated tank cost

        Steps:
        1. Calculate required tank volume from throughput and retention time
        2. Divide by number of cells to get volume per cell
        3. Scale cost using power-law
        4. Apply inflation and lang factors

        Args:
            throughput_tpa: Throughput in tonnes per annum
            ret_time: Retention time (minutes) - if None, uses default from tea.py
            num_cells: Number of cells (default 1)
            cell_volume: Volume per cell (m³) - if provided, overrides calculated volume
        """
        # Step 1: Calculate required volume based on retention time OR use provided cell_volume
        if cell_volume is not None:
            # Use provided cell volume (from optimization)
            volume_per_cell = cell_volume
        else:
            # Calculate volume from throughput and retention time
            if ret_time is None:
                ret_time_sec = residence_time_flotation_sec
            else:
                ret_time_sec = ret_time * 60  # Convert minutes to seconds

            rates = flotation_rates(throughput_tpa)
            total_volume_required = rates['total_rate'] * ret_time_sec * (1 + froth_factor_excess)
            # Divide total volume by number of cells
            volume_per_cell = total_volume_required / num_cells

        # Step 2: Equipment cost with power-law scaling per cell
        # Apply 1.2x volume factor for flotation cells (accounts for froth zone)
        # Formula from Excel line 136: 12300 * (Volume * 1.2 / 3.8)^0.5
        flotation_volume_adjusted = volume_per_cell * 1.2
        single_cell_cost = equipment_cost('vertical_tank', flotation_volume_adjusted)

        # Step 3: Total cost = cost per cell × number of cells, apply factors
        total_cost = single_cell_cost * num_cells * inflation_factor * lang_factor

        return total_cost

    def calculate_opex(self, throughput_tpa, sp_power=None, ret_time=None,
                       num_cells=1, frother_conc=None, frother_price_per_kg=5, cell_volume=None):
        """
        OPEX = annual power cost + water cost + frother reagent cost

        Steps:
        1. Calculate annual power consumption from sp_power and tank volume
        2. Calculate annual water consumption
        3. Calculate frother reagent cost
        4. Sum costs

        Args:
            throughput_tpa: Throughput in tonnes per annum
            sp_power: Specific power (kW/m³) - if None, uses default calculation
            ret_time: Retention time (minutes) - needed to calculate volume
            num_cells: Number of cells
            frother_conc: Frother concentration (g/L) - if None, no frother cost
            frother_price_per_kg: Frother reagent price ($/kg, default 5)
            cell_volume: Volume per cell (m³) - if provided, overrides calculated volume
        """
        # Always calculate rates (needed for water and frother costs)
        rates = flotation_rates(throughput_tpa)

        # Calculate tank volume
        if cell_volume is not None:
            # Use provided cell volume (from optimization)
            volume_per_cell = cell_volume
            total_volume = volume_per_cell * num_cells
        else:
            # Calculate volume from throughput and retention time
            if ret_time is None:
                ret_time_sec = residence_time_flotation_sec
            else:
                ret_time_sec = ret_time * 60

            total_volume_required = rates['total_rate'] * ret_time_sec * (1 + froth_factor_excess)
            # Divide by number of cells
            volume_per_cell = total_volume_required / num_cells
            total_volume = total_volume_required

        # Power cost from sp_power
        if sp_power is None:
            annual_power_kwh = throughput_tpa * self.flotation_work
        else:
            power_kw = sp_power * total_volume
            annual_power_kwh = power_kw * hours_per_op_year

        annual_power_mwh = annual_power_kwh / 1000
        power_cost = annual_power_mwh * electricity_cost

        # Water cost (unchanged)
        annual_water_m3 = rates['water_rate'] * seconds_per_op_year
        water_cost_annual = annual_water_m3 * water_cost

        # Frother reagent cost (simple: concentration × flow × price)
        if frother_conc is not None:
            frother_kg_per_m3 = frother_conc / 1000  # g/L to kg/m³
            annual_solution_m3 = rates['total_rate'] * seconds_per_op_year
            annual_frother_kg = frother_kg_per_m3 * annual_solution_m3
            annual_frother_cost = annual_frother_kg * frother_price_per_kg
        else:
            annual_frother_cost = 0

        return {
            'power': power_cost,
            'water': water_cost_annual,
            'frother': annual_frother_cost,
            'total': power_cost + water_cost_annual + annual_frother_cost
        }

    def process(self, fitting_parameters, flotation_constants, optimizable_vars, inputs):
        """
        Calculate flotation recovery using physics-based model.

        Uses flot_rec from equation_flow_official.py to calculate recovery.
        This determines how much concentrate goes to downstream stages.

        Args:
            fitting_parameters: Dict with keys: b, alpha, coverage, bubble_f, detach_f, bulk_zone
            flotation_constants: Dict with keys: specific_gravity, grade, permitivity,
                                dielectric, pe, cell_area, bbl_ratio
            optimizable_vars: Dict with keys: particle_size, contact_angle, bubble_z_pot,
                             particle_z_pot, sp_power, sp_gas_rate, air_fraction, slurry_fraction,
                             num_cells, ret_time, cell_volume, froth_height, frother_conc
            inputs: Dict with keys: frother_type, water_or_particle

        Returns:
            dict with keys:
                - 'recovery': Flotation recovery (0-1)
                - 'feed_tonnage': Feed tonnage to flotation
                - 'concentrate_tonnage': Concentrate tonnage (feed * recovery)
                - 'tails_tonnage': Tailings tonnage (feed * (1-recovery))
                - 'particle_size': Particle size used (microns)
                - 'contact_angle': Contact angle used (degrees)
        """
        # Calculate recovery using physics-based model
        recovery = flot_rec(fitting_parameters, flotation_constants, optimizable_vars, inputs)

        # Get feed tonnage from flotation constants (if provided)
        feed_tonnage = flotation_constants.get('feed_tonnage', 0)

        # Calculate mass balance
        concentrate_tonnage = feed_tonnage * recovery
        tails_tonnage = feed_tonnage * (1 - recovery)

        return {
            'recovery': recovery,
            'feed_tonnage': feed_tonnage,
            'concentrate_tonnage': concentrate_tonnage,
            'tails_tonnage': tails_tonnage,
            'particle_size': optimizable_vars['particle_size'],
            'contact_angle': optimizable_vars['contact_angle']
        }


class LeachingStage:
    """
    Leaching stage: Dissolves copper from concentrate using acid.

    Physical Parameters → Cost Calculation:
    - concentrate_tonnage → tank volume → CAPEX (atmospheric tank)
    - copper_leached → acid consumption → OPEX (acid cost)

    Key dependency: Uses concentrate tonnage from flotation recovery.
    """

    def __init__(self, acid_price_per_tonne=420, acid_recovery_fraction=0.92):
        """
        Args:
            acid_price_per_tonne: Sulfuric acid price ($/tonne)
                                 Industrial H2SO4: $50-150/tonne
                                 Concentrated HNO3: $300-500/tonne
            acid_recovery_fraction: Fraction of acid recovered and recycled (0-1)
                                   Typical industrial: 0.90-0.95
                                   Net consumption = stoichiometric × (1 - recovery)
        """
        self.acid_price = acid_price_per_tonne
        self.acid_recovery = acid_recovery_fraction

    def calculate_capex(self, concentrate_tpa, leach_time=None):
        """
        CAPEX = atmospheric tank cost

        Steps:
        1. Calculate required tank volume from concentrate throughput and leach time
        2. Scale cost using power-law
        3. Apply inflation and lang factors

        Args:
            concentrate_tpa: Concentrate throughput (NOT ore feed!)
            leach_time: Leaching time (hours) - if None, uses default from tea.py
        """
        # Calculate volume based on leach time
        if leach_time is None:
            leach_time_sec = residence_time_leaching_sec
        else:
            leach_time_sec = leach_time * 3600  # hours to seconds

        rates = leaching_rates(concentrate_tpa)
        volume_required = rates['total_rate'] * leach_time_sec

        # Equipment cost with power-law scaling
        tank_cost = equipment_cost('atm_tank', volume_required)

        # Apply factors
        installed_cost = tank_cost * inflation_factor * lang_factor

        return installed_cost

    def calculate_opex(self, copper_leached_kg):
        """
        OPEX = annual acid cost (accounting for acid recovery/recycling)

        Steps:
        1. Calculate stoichiometric acid consumption
        2. Apply acid recovery factor to get net consumption
        3. Convert to annual cost

        Args:
            copper_leached_kg: Copper leached per year (kg)

        Returns:
            dict with acid cost breakdown
        """
        # Calculate stoichiometric acid consumption
        acid_result = calculate_acid_consumption(copper_leached_kg)

        # Stoichiometric acid requirement (convert kg to tonnes)
        stoichiometric_acid_kg = acid_result['acid_consumption_kg']
        stoichiometric_acid_tonnes = stoichiometric_acid_kg / 1000

        # Net consumption after recovery (only makeup acid needed)
        net_acid_tonnes = stoichiometric_acid_tonnes #* (1 - self.acid_recovery)

        # Cost (only pay for makeup acid, not recycled acid)
        acid_cost = net_acid_tonnes * self.acid_price

        return {
            'acid_stoichiometric_tpa': stoichiometric_acid_tonnes,
            'acid_net_consumption_tpa': net_acid_tonnes,
            'acid_recovery_fraction': self.acid_recovery,
            'acid_per_kg_cu': acid_result['acid_per_kg_cu'],
            'acid_cost': acid_cost,
            'total': acid_cost
        }

    def process(self, concentrate_tpa, copper_in_concentrate_tpa, Ea=80, T=190,
                P_O2=12, n=0.75, H_plus=0.2, MFeS2=0.5, leach_time=12):
        """
        Calculate leaching recovery using kinetic model.

        Uses leaching_copper_recovery from equation_flow_official.py.

        Args:
            concentrate_tpa: Concentrate feed (tonnes/year)
            copper_in_concentrate_tpa: Copper in concentrate (tonnes/year)
            Ea: Activation energy (kJ/mol)
            T: Temperature (C)
            P_O2: Oxygen partial pressure (bar)
            n: Reaction order
            H_plus: H+ concentration
            MFeS2: FeS2 mass fraction
            leach_time: Leaching time (hours)

        Returns:
            dict with keys:
                - 'leach_recovery': Leaching recovery (0-1)
                - 'copper_leached_tpa': Copper leached (tonnes/year)
                - 'copper_leached_kg': Copper leached (kg/year)
                - 'solution': Leaching solution data
        """
        # Run leaching kinetics model
        X_Cu = 0  # Initial conversion
        leach_recovery, solution = leaching_copper_recovery(
            Ea, T, P_O2, n, H_plus, MFeS2, X_Cu, leach_time=leach_time
        )

        # Calculate copper leached
        copper_leached_tpa = copper_in_concentrate_tpa * leach_recovery
        copper_leached_kg = copper_leached_tpa * 1000

        return {
            'leach_recovery': leach_recovery,
            'concentrate_feed_tpa': concentrate_tpa,
            'copper_in_concentrate_tpa': copper_in_concentrate_tpa,
            'copper_leached_tpa': copper_leached_tpa,
            'copper_leached_kg': copper_leached_kg,
            'solution': solution
        }


class SolventExtractionStage:
    """
    Solvent Extraction (SX) stage: Purifies copper from leach solution using organic extractants.

    Physical Parameters → Cost Calculation:
    - solution_volume → mixer-settler volume → CAPEX (tank-like equipment)
    - copper_extracted → organic solvent consumption → OPEX (solvent makeup cost)

    Key dependency: Uses copper solution from leaching stage.
    """

    def __init__(self, solvent_price_per_litre=8, solvent_loss_fraction=0.01):
        """
        Args:
            solvent_price_per_litre: Organic extractant price ($/L)
                                    LIX reagents: $3-8/L
            solvent_loss_fraction: Fraction of solvent lost per cycle (0-1)
                                    Typical: 0.005-0.02 (0.5-2% losses)
        """
        self.solvent_price = solvent_price_per_litre
        self.solvent_loss = solvent_loss_fraction

    def calculate_capex(self, leach_concentrate_tpa):
        """
        CAPEX = mixer-settler equipment cost

        Uses atmospheric tank cost basis as proxy for mixer-settlers.
        Calculates based on total throughput (aqueous + organic streams).

        Steps:
        1. Calculate total throughput based on O/A ratio
        2. Scale cost using power-law
        3. Apply inflation and lang factors

        Args:
            leach_concentrate_tpa: Concentrate from leaching (tonnes/year)
        """
        # Feed to solvent ratio (O/A = 1 means equal organic and aqueous volumes)
        oa_ratio = 1
        aqueous_tpa = leach_concentrate_tpa
        organic_tpa = aqueous_tpa * oa_ratio
        total_throughput = aqueous_tpa + organic_tpa

        # CAPEX using power-law scaling
        # From Perry's Table 9-50: Atm Tank basis
        sx_capex = 9300 * ((total_throughput / 0.38) ** 0.53)
        sx_capex = sx_capex * inflation_factor * lang_factor

        return sx_capex

    def calculate_opex(self, copper_extracted_kg, solution_volume_m3):
        """
        OPEX = solvent makeup cost for solvent extraction

        Steps:
        1. Calculate solvent inventory from solution volume
        2. Calculate annual solvent losses
        3. Convert to cost

        Args:
            copper_extracted_kg: Copper extracted per year (kg)
            solution_volume_m3: Solution volume processed (m3/year)

        Returns:
            dict with solvent cost breakdown
        """
        # Typical O/A ratio is 1:1, so organic volume ≈ aqueous volume
        organic_volume_m3 = solution_volume_m3
        organic_volume_litres = organic_volume_m3 * 1000

        # Annual solvent makeup (only replace losses)
        annual_solvent_losses_litres = organic_volume_litres * self.solvent_loss

        # Cost
        solvent_cost = annual_solvent_losses_litres * self.solvent_price

        return {
            'organic_volume_litres': organic_volume_litres,
            'solvent_losses_litres': annual_solvent_losses_litres,
            'solvent_loss_fraction': self.solvent_loss,
            'solvent_cost': solvent_cost,
            'total': solvent_cost
        }

    def process(self, copper_leached_kg, K_ex=10, RH=0.2, O_A=1.2, pH=1.1):
        """
        Calculate solvent extraction recovery using equilibrium model.

        Uses solvent_extraction from solvent_extraction.py.

        Args:
            copper_leached_kg: Copper from leaching (kg/year)
            K_ex: Extraction equilibrium constant (dimensionless)
            RH: Extractant concentration (molar)
            O_A: Organic/aqueous ratio (dimensionless)
            pH: Solution pH

        Returns:
            dict with keys:
                - 'sx_recovery': SX recovery (0-1)
                - 'copper_feed_kg': Copper feed from leaching (kg/year)
                - 'copper_extracted_kg': Copper extracted to organic (kg/year)
                - 'copper_raffinate_kg': Copper remaining in raffinate (kg/year)
                - 'copper_feed_moles': Copper feed (moles)
                - 'copper_extracted_moles': Copper extracted (moles)
        """
        # Convert copper to moles
        MW_Cu = 63.546  # g/mol
        copper_feed_moles = copper_leached_kg * 1000 / MW_Cu  # kg to g to mol

        # Run solvent extraction model
        copper_raffinate_moles = solvent_extraction(copper_feed_moles, K_ex, RH, O_A, pH)
        copper_extracted_moles = copper_feed_moles - copper_raffinate_moles

        # Calculate recovery
        sx_recovery = copper_extracted_moles / copper_feed_moles if copper_feed_moles > 0 else 0

        # Convert back to kg
        copper_extracted_kg = copper_extracted_moles * MW_Cu / 1000
        copper_raffinate_kg = copper_raffinate_moles * MW_Cu / 1000

        return {
            'sx_recovery': sx_recovery,
            'copper_feed_kg': copper_leached_kg,
            'copper_extracted_kg': copper_extracted_kg,
            'copper_raffinate_kg': copper_raffinate_kg,
            'copper_feed_moles': copper_feed_moles,
            'copper_extracted_moles': copper_extracted_moles,
            'K_ex': K_ex,
            'RH': RH,
            'O_A': O_A,
            'pH': pH
        }


class ElectrowinningStage:
    """
    Electrowinning (EW) stage: Plates pure copper metal from purified solution using electrolysis.

    Physical Parameters → Cost Calculation:
    - copper_plated → cell area/number → CAPEX (electrowinning cells)
    - copper_plated + cell_voltage → power consumption → OPEX (electricity cost)

    Key dependency: Uses purified copper solution from SX stage.
    """

    def __init__(self, current_density=250, current_efficiency=0.9, cell_cost_per_m2=5000):
        """
        Args:
            current_density: Current density (A/m2), typical: 200-300 A/m2
            current_efficiency: Current efficiency (0-1), typical: 0.90-0.98
            cell_cost_per_m2: Cost per unit cathode area ($/m2) - 1990s basis
                             Will be adjusted for inflation
                             Typical 1990s: $1500-2500/m2
                             Current (2020s): $3500-6000/m2
        """
        self.current_density = current_density
        self.current_efficiency = current_efficiency
        self.cell_cost_per_m2 = cell_cost_per_m2

    def calculate_capex(self, copper_feed_kg_per_year, cu_concentration_kg_per_m3=45, residence_time_hours=36):
        """
        CAPEX = electrowinning cell equipment cost

        Steps:
        1. Convert copper feed to molar flow rate (mol/s)
        2. Calculate required current using Faraday's law: I = n*F*molar_rate
        3. Calculate electrode area: area = current / current_density
        4. Scale CAPEX based on electrode area

        Args:
            copper_feed_kg_per_year: Copper from SX (kg/year)
            cu_concentration_kg_per_m3: Copper concentration in electrolyte (kg/m3), typical 40-50
            residence_time_hours: Residence time in tank (hours), typical 24-48

        Returns:
            Installed CAPEX ($)
        """
        # Step 1: Convert copper feed to molar flow rate
        # copper_feed_kg_per_year (kg/year) → mol/s
        MW_Cu = 63.546  # g/mol
        copper_mol_per_year = (copper_feed_kg_per_year * 1000) / MW_Cu  # kg → g → mol
        copper_mol_per_second = copper_mol_per_year / seconds_per_op_year  # mol/s

        z = 2  # electrons per Cu²⁺
        F = 96485  # C/mol (Faraday constant)
        req_current_amps = copper_mol_per_second * z * F  # Amps

        # Calculate electrode area required
        current_density = 4000  # A/m² (higher density for industrial EW)
        area_req_m2 = req_current_amps / current_density  # m²

        # CAPEX scaling based on electrode area
        basis_area_m2 = 94  # m² basis
        cost_coefficient = 383000  # $ at basis
        ew_exponent = 0.81
        ew_capex = cost_coefficient * ((area_req_m2 / basis_area_m2) ** ew_exponent)
        ew_capex = ew_capex * lang_factor * inflation_factor

        return ew_capex

    def calculate_opex(self, copper_plated_tpa, E_eq=0.34, n_cath=0.1, n_anode=0.3, IR=0.5):
        """
        OPEX = annual power cost for electrowinning

        Steps:
        1. Calculate required current from Faraday's law
        2. Calculate cell voltage
        3. Calculate power consumption
        4. Convert to annual cost

        Args:
            copper_plated_tpa: Copper production rate (tonnes/year)
            E_eq: Equilibrium potential (V), default 0.34 V for Cu
            n_cath: Cathode overpotential (V), typical 0.1 V
            n_anode: Anode overpotential (V), typical 0.3 V
            IR: Resistance losses (V), typical 0.5 V

        Returns:
            dict with power cost breakdown
        """
        MW_Cu = 63.546  # g/mol
        F = 96485  # C/mol
        n = 2  # electrons per Cu

        # Annual copper production
        copper_kg_per_year = copper_plated_tpa * 1000

        # Operating time per year (seconds)
        operating_seconds = seconds_per_op_year

        # Required current (Amperes)
        required_current = (copper_kg_per_year * 1000 * n * F) / (operating_seconds * MW_Cu * self.current_efficiency)

        # Cell voltage
        V_cell = cell_voltage(E_eq, n_cath, n_anode, IR)

        # Power (Watts)
        power_watts = required_current * V_cell
        power_kw = power_watts / 1000

        # Annual energy consumption
        annual_hours = hours_per_op_year
        annual_energy_kwh = power_kw * annual_hours
        annual_energy_mwh = annual_energy_kwh / 1000

        # Annual cost
        power_cost = annual_energy_mwh * electricity_cost

        return {
            'required_current': required_current,
            'cell_voltage': V_cell,
            'power_kw': power_kw,
            'annual_energy_kwh': annual_energy_kwh,
            'annual_energy_mwh': annual_energy_mwh,
            'power_cost': power_cost,
            'total': power_cost
        }

    def process(self, copper_extracted_kg, ew_recovery=0.95):
        """
        Calculate electrowinning production.

        Simple recovery model: assumes fixed recovery based on operating conditions.
        In practice, recovery depends on current efficiency and operating time.

        Args:
            copper_extracted_kg: Copper from SX (kg/year)
            ew_recovery: Electrowinning recovery (0-1), typical 0.93-0.98

        Returns:
            dict with keys:
                - 'ew_recovery': EW recovery (0-1)
                - 'copper_feed_kg': Copper feed from SX (kg/year)
                - 'copper_plated_kg': Copper plated (kg/year)
                - 'copper_plated_tpa': Copper plated (tonnes/year)
                - 'copper_remaining_kg': Copper remaining in solution (kg/year)
        """
        # Calculate copper plated
        copper_plated_kg = copper_extracted_kg * ew_recovery
        copper_remaining_kg = copper_extracted_kg * (1 - ew_recovery)
        copper_plated_tpa = copper_plated_kg / 1000

        return {
            'ew_recovery': ew_recovery,
            'copper_feed_kg': copper_extracted_kg,
            'copper_plated_kg': copper_plated_kg,
            'copper_plated_tpa': copper_plated_tpa,
            'copper_remaining_kg': copper_remaining_kg
        }

