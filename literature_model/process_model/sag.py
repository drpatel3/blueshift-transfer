"""SAG mill model (Asghari, VandGhorbany, Nakhaei 2019, Part. Sci. Tech.).

Bond ball-mill work index from locked-cycle test, JK drop-weight breakage
curve, and shape descriptors from image analysis of the circulating load.
"""
from dataclasses import dataclass
import math


@dataclass
class SAGParams:
    A: float       # JK breakage fit parameter
    b: float       # JK breakage fit parameter
    Wi: float      # Bond ball-mill work index, kWh/t
    ta: float      # abrasion parameter (t10/10 from tumble test)
    PLI: float     # Point Load Index, MPa


def bond_wi(Pi: float, Gi: float, P80: float, F80: float) -> float:
    """Bond Wi from locked-cycle grindability (Asghari Eq. 1).
    Pi = control screen size (um), Gi = net g/rev, P80/F80 in um."""
    num = 44.5
    denom = (Pi ** 0.23) * (Gi ** 0.82)
    size_term = 10.0 / math.sqrt(P80) - 10.0 / math.sqrt(F80)
    return (num / denom) / size_term


def t10(A: float, b: float, Ecs: float) -> float:
    """JK drop-weight (Eq. 2): t10 = A*(1 - exp(-b*Ecs)). Ecs in kWh/t."""
    return A * (1.0 - math.exp(-b * Ecs))


def specific_energy_from_t10(A: float, b: float, t10_target: float) -> float:
    """Invert JK curve to get Ecs required for a target t10."""
    if t10_target >= A:
        raise ValueError("t10 target exceeds asymptote A")
    return -math.log(1.0 - t10_target / A) / b


def aspect_ratio(width: float, length: float) -> float:
    return width / length


def circularity(area: float, perimeter: float) -> float:
    """Eq. (4): C = 4*pi*A / P^2. 1.0 = perfect circle."""
    return 4.0 * math.pi * area / (perimeter ** 2)
