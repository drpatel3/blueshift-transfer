import math
import numpy as np

upper_lower_bounds = {
    "K_ex": (2, 6),
    "RH": (0.05, 0.10),
    "H_plus": (0.01, 0.05),
    "O_A": (0.8, 1.2),  # O/A ratio, default ~1.0
    "pH": (1.8, 2.5)
}

# Default target purity for copper SX
TARGET_PURITY = 0.993  # 99.3%


def calculate_num_mixers(O_A, K_ex, RH, pH, target_purity=TARGET_PURITY):
    """
    Calculate number of mixer-settler stages required to achieve target purity.

    Uses the extraction factor E from the SX equation to determine
    how many stages are needed to reach the target purity level.

    For single stage: recovery = E / (1 + E)
    For N stages: overall_recovery = 1 - (1 / (1 + E))^N

    Args:
        O_A: Organic/aqueous ratio (feed to solvent ratio)
        K_ex: Extraction equilibrium constant
        RH: Extractant concentration (M)
        pH: pH of aqueous solution
        target_purity: Target copper purity (0-1), default 0.993 (99.3%)

    Returns:
        dict with:
            - num_mixers: Number of mixer-settler stages (integer)
            - extraction_factor: Extraction factor E
            - single_stage_recovery: Recovery per single stage
            - overall_recovery: Expected overall recovery with num_mixers stages
    """
    H_plus = 10**(-pH)

    # Extraction factor from SX equation
    # E = K_ex * [RH]^2 / [H+]^2 * (O/A)
    E = K_ex * (RH ** 2) / (H_plus ** 2) * O_A

    # Single stage recovery
    single_stage_recovery = E / (1 + E)

    # Calculate required number of stages to achieve target purity
    # For countercurrent extraction: (1 - overall_recovery) = (1 - single_stage_recovery)^N
    # Solving for N: N = ln(1 - target_purity) / ln(1 - single_stage_recovery)

    if single_stage_recovery >= target_purity:
        # Single stage is sufficient
        num_mixers = 1
    else:
        # Calculate stages needed
        num_mixers = math.ceil(
            math.log(1 - target_purity) / math.log(1 - single_stage_recovery)
        )

    # Ensure minimum of 2 stages (industry standard for SX)
    num_mixers = max(2, num_mixers)

    # Calculate actual overall recovery with this number of stages
    overall_recovery = 1 - (1 - single_stage_recovery) ** num_mixers

    return {
        'num_mixers': num_mixers,
        'extraction_factor': E,
        'single_stage_recovery': single_stage_recovery,
        'overall_recovery': overall_recovery,
        'target_purity': target_purity,
        'O_A': O_A
    }


def solvent_extraction(C_Cu_in_aq, K_ex=10, RH=0.5, O_A=1.0, pH=1.1):
    """
    Calculate output copper concentration from solvent extraction.

    Equation: C_Cu_out^aq = C_Cu_in^aq / (1 + K_ex * [RH]^2 / [H+]^2 * (O/A))

    Args:
        C_Cu_in_aq: Amount of Cu in feed from leaching
        K_ex: Extraction equilibrium constant
        RH: Extractant concentration (M)
        O_A: Organic/aqueous ratio
        pH: pH of aqueous solution

    Returns:
        float: Copper concentration in raffinate (C_Cu_out^aq)
    """

    H_plus = 10**(-pH)
    denominator = 1 + K_ex * (RH ** 2) / (H_plus ** 2) * O_A
    C_Cu_out_aq = C_Cu_in_aq / denominator

    return C_Cu_out_aq
