"""Solvent extraction mixer-settler model (Moreno, Perez-Correa, Otero 2009).

Dynamic CSTR mixer + non-ideal settler (fast branch = N CSTRs in series,
slow branch = plug-flow delay followed by two Cholette-Cloutier two-zone tanks).
State vector and ODE right-hand side are built for scipy.integrate.odeint / solve_ivp.

All concentrations in g/L, volumes in m3, flows in m3/h.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class MixerParams:
    V_Am: float      # aqueous holdup, m3
    V_Om: float      # organic holdup, m3
    K_E: float       # extraction mass-transfer coefficient, 1/h  (use K_S for stripping)
    eta: float       # efficiency η_E or η_S
    stage: str = "extract"  # "extract" or "strip"


@dataclass
class IsothermParams:
    # Extraction: Y* = A*X*/(X*+B); A = a*ML; B from (10^-pH)^b / ML^c * (d*[Cu]_PLS + f*[Cu]_BO)
    # Stripping : Y* = C*X* + D;    C = g*ML; D = h*vol%^m / [H2SO4]^n + p
    ML: float        # max organic load, g/L
    a: float = 0.99
    b: float = 1.02
    c: float = 1.01
    d: float = 35.15
    f: float = 27.15
    g: float = 0.11
    h: float = 444.49
    m: float = 0.10
    n: float = 0.81
    p: float = -8.91


def extraction_isotherm(X_star: float, iso: IsothermParams,
                        pH: float, cu_pls: float, cu_bo: float) -> float:
    """Eq. (27)-(29): loaded organic in equilibrium with aqueous X*."""
    A = iso.a * iso.ML
    B = ((10.0 ** -pH) ** iso.b / iso.ML ** iso.c) * (iso.d * cu_pls + iso.f * cu_bo)
    return A * X_star / (X_star + B)


def stripping_isotherm(X_star: float, iso: IsothermParams,
                       vol_pct: float, h2so4_le: float) -> float:
    """Eq. (30)-(32): linear stripping isotherm."""
    C = iso.g * iso.ML
    D = iso.h * (vol_pct ** iso.m) / (h2so4_le ** iso.n) + iso.p
    return C * X_star + D


def mixer_rhs(X_m: float, Y_m: float,
              X_i: float, Y_i: float,
              X_star: float, Y_star: float,
              Q_Ai: float, Q_Oi: float,
              mp: MixerParams) -> tuple:
    """Moreno Eqs. (3)-(8). Returns (dX_m/dt, dY_m/dt).
    X = aqueous [Cu], Y = organic [Cu]. Sign on transfer term flips for strip."""
    # Pseudo-equilibrium with efficiency (Eqs. 7-8)
    if mp.stage == "extract":
        Y_E = (1.0 - mp.eta) * Y_i + mp.eta * Y_star
        dX = (Q_Ai / mp.V_Am) * (X_i - X_m) - mp.K_E * (Y_E - Y_m)
        dY = (Q_Oi / mp.V_Om) * (Y_i - Y_m) + mp.K_E * (Y_E - Y_m)
    else:  # strip
        X_S = (1.0 - mp.eta) * X_i + mp.eta * X_star
        dX = (Q_Ai / mp.V_Am) * (X_i - X_m) + mp.K_E * (X_S - X_m)
        dY = (Q_Oi / mp.V_Om) * (Y_i - Y_m) - mp.K_E * (X_S - X_m)
    return dX, dY


def efficiency_from_gradient(eps: float, out_conc: float, in_conc: float,
                             eq_conc: float) -> float:
    """Eqs. (35)-(36): η = (ε + (out-in)) / (eq - in)."""
    return (eps + (out_conc - in_conc)) / (eq_conc - in_conc)


def steady_state_mixer(X_i: float, Y_i: float,
                       X_star_fn, Y_star_fn,
                       Q_Ai: float, Q_Oi: float,
                       mp: MixerParams,
                       tol: float = 1e-6, max_iter: int = 500) -> tuple:
    """Fixed-point iteration to the mixer steady state (dX/dt = dY/dt = 0).
    X_star_fn(Y) -> X*, Y_star_fn(X) -> Y*. Returns (X_m, Y_m)."""
    X_m, Y_m = X_i, Y_i
    for _ in range(max_iter):
        X_star = X_star_fn(Y_m)
        Y_star = Y_star_fn(X_m)
        dX, dY = mixer_rhs(X_m, Y_m, X_i, Y_i, X_star, Y_star,
                           Q_Ai, Q_Oi, mp)
        X_m += 0.05 * dX
        Y_m += 0.05 * dY
        if abs(dX) < tol and abs(dY) < tol:
            break
    return X_m, Y_m
