#need to know acid or base flow rate

#electrolyzer needs to know total amount of acid required from leaching and solvent exchange

variables = {
"sl_ratio": .05,
"acid_concentration": .5,
"kg_feed": 1000,
"acid_flow_rate": 6.17*1000, # total from leaching and solvent exchange
"molar_flow_rate": .5 * 6.17*1000,
"F": 96485,  # C/mol
"z": 2, # mol electron transfer for suplhuric acid
"current_required": .5 * 6.17*1000 * 96485 * 2,
"cell_potential": 2,
"sulph_acid_density": 1830, #kg/m^3
"recovery_rate": .85,
"grade": .03
}

#electrolyzer - energy cost = energy_required * $50mw/h

def electrolyzer(variables=variables):
    def energy_use(variables=variables, F = 96485, z=2):
        recovery_rate = .85
        grade = .03
        tons_feed_per_year = 100000 # tons/year
        kg_feed_per_second = (tons_feed_per_year * 1000) / (365 * 24 * 3600) # kg/s
        acid_flow_rate = 6.17*kg_feed_per_second*grade*recovery_rate # kg H2SO4/s
        molar_mass_H2SO4 = 98.08 # g/mol
        molar_flow_rate = (acid_flow_rate * 1000) / molar_mass_H2SO4 # mol/s
        F = 96485 # C/mol
        z = 2 # mol electron transfer for sulphuric acid
        area_required = molar_flow_rate * z * F
        current_required = molar_flow_rate * F * z # Amperes
        cell_potential = 2 # Volts
        power_required = (current_required * cell_potential)/1000 # kW
        operating_capacity = 0.9
        operating_hours_per_year = 365 * 24 * operating_capacity # hours/year
        energy_required = power_required * operating_hours_per_year # kWh/year

        return energy_required, area_required

    def water_use(tons_feed_year=100000):
        """
        Calculate water usage for electrolyzer based on stoichiometry.

        For H2SO4 electrolysis: H2SO4 -> H2 + SO2 + O2 + H2O
        Water is produced during electrolysis.

        However, water is also needed for the acid solution preparation.
        The relationship from the Excel model shows:
        - Acid solution flow rate (m3/s) is based on concentration and molar flow
        - Water flow rate = (Acid solution mass flow - Acid mass flow)

        Returns:
            annual_water_use (float): Water consumption in m³/year
        """
        recovery_rate = .85
        grade = .03
        tons_feed_year = tons_feed_year
        kg_feed_per_second = (tons_feed_year * 1000) / (365 * 24 * 3600)

        # Acid flow rate calculation
        acid_flow_rate = 6.17*kg_feed_per_second*grade*recovery_rate  # kg H2SO4/s

        # Convert to molar flow rate
        molar_mass_H2SO4 = 98.08  # g/mol
        molar_flow_rate = (acid_flow_rate * 1000) / molar_mass_H2SO4  # mol/s

        # Calculate acid solution flow rate based on concentration
        acid_concentration = 0.5  # M (mol/L) = 500 mol/m³
        acid_solution_volumetric_flow = molar_flow_rate / (acid_concentration * 1000)  # m³/s

        # Calculate mass flow rate of acid solution
        # Assuming density of 0.5 M H2SO4 solution ≈ 1030 kg/m³
        acid_solution_density = 1030  # kg/m³
        acid_solution_mass_flow = acid_solution_volumetric_flow * acid_solution_density  # kg/s

        # Water flow rate = total solution - acid
        water_flow_rate = acid_solution_mass_flow - acid_flow_rate  # kg/s

        # Convert to volumetric flow (water density = 1000 kg/m³)
        water_density = 1000  # kg/m³
        water_volumetric_flow = water_flow_rate / water_density  # m³/s

        # Calculate annual water use based on operating capacity
        operating_capacity = 0.9
        operating_seconds_per_year = 365 * 24 * 3600 * operating_capacity  # s/year
        annual_water_use = water_volumetric_flow * operating_seconds_per_year  # m³/year

        return annual_water_use, water_flow_rate


    def capex(area_required):
        capex_cost = 1500 * 50 * ((area_required / 1500) ** 0.85)
        return capex_cost

    def cost(energy_required, water_required, area_required, water_cost_per_m3=0.3):
        """
        Calculate costs for electrolyzer including CAPEX and OPEX.

        Args:
            energy_required (float): Energy consumption in kWh/year
            water_required (float): Water consumption in m³/year
            area_required (float): Electrolyzer area in m²
            water_cost_per_m3 (float): Water cost in $/m³ (default 0.3 from Excel model)

        Returns:
            tuple: (capex_cost, energy_cost, water_cost, opex_total, cost_total) all in $
        """
        capex_cost = capex(area_required)
        energy_cost_per_mwh = 50  # $/MWh
        energy_cost = (energy_required / 1000) * energy_cost_per_mwh  # $/year
        water_cost = water_required * water_cost_per_m3  # $/year
        opex_total = water_cost + energy_cost  # $/year
        cost_total = capex_cost + opex_total  # $
        return capex_cost, energy_cost, water_cost, opex_total, cost_total
    energy_required, area_required = energy_use()
    water_required, water_flow_rate = water_use()
    capex_cost, energy_cost, water_cost, opex_total, cost_total = cost(energy_required, water_required, area_required)

    return energy_required, water_required, capex_cost, opex_total, cost_total, water_flow_rate

