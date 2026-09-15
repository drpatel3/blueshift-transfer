import math
import numpy as np

def electrowinning_concentration(C_Cu_in=1005, C_Cu_out=1000, eta_CE=.98, n_var=2, f_var=96485):
    """
    Calculate output copper concentration from electrowinning.

    Equation: C_Cu_out = C_Cu_in - (eta_CE * I) / (n * F * F)

    Args:
        C_Cu_in: Input copper concentration (g/mol or appropriate units)
        eta_CE: Current efficiency (dimensionless, 0-1)
        I: Current (Amperes)
        n: Number of electrons per Cu (2 e- per Cu)
        F: Faraday constant (96485 C/mol)
        M_Cu: Molecular weight of Cu (63.55 g/mol)

    Returns:
        float: Output copper concentration (C_Cu_out)
    """
    # Calculate concentration change due to electrowinning
    C_Cu_out = 1000
    I = (n_var*f_var*(C_Cu_in-C_Cu_out))/eta_CE
    C_Cu_in = C_Cu_out + (eta_CE * I) / (n_var * f_var)
    recovery_rate = (C_Cu_in - C_Cu_out) / C_Cu_out

    return C_Cu_out, C_Cu_in, recovery_rate

def cell_voltage(E_eq = .34, eta_cath = .1, eta_anode = .3, IR = .5):
    """
    Calculate cell voltage for electrowinning.

    Equation: V_cell = E_eq + eta_cath + eta_anode + IR

    Args:
        E_eq: Nernst equilibrium potential (V)
        eta_cath: Overpotentials at cathode (V)
        eta_anode: Overpotentials at anode (V)
        I: Current (A)
        R: Resistance (Ohms)

    Returns:
        float: Cell voltage (V)
    """
    V_cell = E_eq + eta_cath + eta_anode + IR

    return V_cell

def nernst_potential(E_Cu_0=.34, a_Cu2=1, R=8.314, T=298.15, F=96485, n=2):
    """
    Calculate Nernst equilibrium potential.

    Equation: E_eq = E_Cu^0 + (RT/2F) * ln(a_Cu2+)

    Args:
        E_Cu_0: Standard potential for Cu (V), typically 0.34 V
        a_Cu2: Activity of Cu2+ ions (dimensionless)
        R: Universal gas constant (J/mol·K), default=8.314
        T: Temperature (K), default=298.15
        F: Faraday constant (C/mol), default=96485
        n: Number of electrons (2 for Cu), default=2

    Returns:
        float: Nernst equilibrium potential (V)
    """
    RT_nF = 0
    E_eq = E_Cu_0 + (RT_nF) * math.log(a_Cu2)

    return E_eq

def overpotential(a, j, b):
    """
    Calculate overpotential as function of current density.

    Equation: eta = a * ln(j) + b

    Args:
        a: Tafel slope coefficient (V)
        j: Current density (I/A)
        b: Constant (V)

    Returns:
        float: Overpotential (V)
    """
    eta = a * math.log(j) + b

    return eta
