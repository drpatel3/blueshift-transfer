import math
import scipy.integrate

variables = {
    "Ea": (70, 90),  # kJ/mol
    "T": (180, 202),  # °C
    "P_O2": (10, 15),  # bar
    "n": (0.5, 1),  # dimensionless
    "Phi": 10,  # kg/L
    "H_plus": (.1-.3),  # -log(pH)
    "P(O2_eff)": (10-15)  # effective oxygen pressure
}

def leaching_copper_recovery(Ea, T, P_O2, n, H_plus, MFeS2, X_Cu, Phi=.1, k0=500, R=8.314, leach_time=12):
    """
    Calculate total copper recovery after a given leaching time.

    Args:
        Ea: Activation energy (kJ/mol)
        T: Temperature (°C)
        P_O2: Partial pressure of oxygen (bar)
        n: Stoichiometry parameter (dimensionless)
        Phi: Mass of solids per solution volume (kg/L)
        H_plus: H+ ion concentration, typically expressed as -log(pH)
        MFeS2: Mass fraction of FeS2 (dimensionless)
        X_Cu: Initial copper conversion (not used, kept for compatibility)
        k0: Pre-exponential factor (1/s), default=1.0
        R: Universal gas constant (J/mol·K), default=8.314
        leach_time: Leaching time in hours

    Returns:
        X_Cu_final: Final copper recovery fraction (0-1)
        solution: ODE solution object
    """
    from scipy.integrate import solve_ivp

    MFeS2 = .86 * 1000 * .58
    T_K = T + 273.15
    Ea_J = Ea * 1000
    P_O2_eff = P_O2

    def copper_leaching_rate(t, y):
        """
        ODE function for solve_ivp.

        Equation: dX_Cu/dt = 3*k0*exp(-Ea/RT)*(P_O2_eff)^n*(1-X_Cu)^(2/3)

        Args:
            t: time (seconds)
            y: state vector [X_Cu]

        Returns:
            [dX_Cu/dt]
        """
        X_Cu_current = y[0]

        # Prevent X_Cu from exceeding 1.0
        if X_Cu_current >= 1.0:
            return [0.0]

        # Calculate the leaching rate
        exponential_term = math.exp(-Ea_J / (R * T_K))
        pressure_term = P_O2_eff ** n
        conversion_term = (1 - X_Cu_current) ** (2/3)

        dX_Cu_dt = 3 * k0 * exponential_term * pressure_term * conversion_term
        return [dX_Cu_dt]

    def max_conversion(t, y):
        """Event function to stop integration when X_Cu reaches 1.0"""
        return y[0] - 1.0

    max_conversion.terminal = True
    max_conversion.direction = 1

    X_Cu_initial = [0.0]
    t_span = (0, leach_time * 3600)

    solution = solve_ivp(copper_leaching_rate, t_span, X_Cu_initial,
                        events=max_conversion, dense_output=False)

    X_Cu_final = min(solution.y[0][-1], 1.0)  # Cap at 100%

    return X_Cu_final, solution


def calculate_acid_consumption(Cu_recovered_kg):
    """
    Calculate acid consumption for copper leaching based on chalcopyrite stoichiometry.

    Stoichiometry for chalcopyrite (CuFeS2):
    CuFeS2 + 4H2SO4 + O2 -> CuSO4 + FeSO4 + 4S + 4H2O

    Args:
        Cu_recovered_kg: Mass of copper recovered (kg)

    Returns:
        dict with acid consumption data
    """
    MW_Cu = 63.546  # g/mol
    MW_H2SO4 = 98.079  # g/mol

    # Stoichiometric ratio: 4 mol H2SO4 per 1 mol Cu
    acid_stoich_ratio = 4 * MW_H2SO4 / MW_Cu  # kg H2SO4 / kg Cu = 6.17

    # Total acid consumption
    acid_consumption_kg = Cu_recovered_kg * acid_stoich_ratio

    return {
        "acid_consumption_kg": acid_consumption_kg,
        "acid_per_kg_cu": acid_stoich_ratio,
        "Cu_recovered_kg": Cu_recovered_kg
    }
feed_density = 2.7
seconds_per_op_year = 28382400
sulfuric_acid_density = 1380
sl_ratio = .05
acid_concentration = 0.15 
inflation_factor = 2.4
lang_factor = 5

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
