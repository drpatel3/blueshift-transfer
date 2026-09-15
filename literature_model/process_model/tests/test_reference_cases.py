"""Stage-level regression cases.

Pins the per-stage outputs of the default sulfide chain at a Cerro Verde-like
feed (2000 tph, 0.4% Cu chalcopyrite) within +/-3% bands. Catches drift in any
single stage's physics without making the optimizer or TEA the gate.

Baseline captured 2026-06-23 against the current process_model code.
"""
from dataclasses import replace

import pytest

from process_model.throughput import simulate_cu_sulfide, CuSulfideParams
from process_model.topology import Topology


TOL = 0.03  # +/-3% bands on every pinned numeric


def _approx(actual: float, expected: float, tol: float = TOL) -> bool:
    if expected == 0.0:
        return abs(actual) < 1e-9
    return abs(actual - expected) / abs(expected) <= tol


@pytest.fixture(scope="module")
def sim():
    p = replace(CuSulfideParams(), mineralogy="chalcopyrite")
    return simulate_cu_sulfide(2000.0, 0.004, p, topology=Topology())


# (stream_name, solids_tph, water_tph, F80_um, cu_grade_frac)
PINNED_STREAMS = (
    ("ore",                 2000.00,    0.00, 300000.0, 0.00400),
    ("crushed",             2000.00,    0.00, 150000.0, 0.00400),
    ("sag_discharge",       2000.00,  660.00,   1961.0, 0.00400),
    ("bm_feed",             7000.00, 5727.27,    667.4, 0.00400),
    ("bm_discharge",        7000.00, 5727.27,    150.0, 0.00400),
    ("cyc_uf",              5000.00,  248.64,    667.4, 0.00400),
    ("cyc_of",              2000.00, 5478.63,    150.0, 0.00400),
    ("rougher_conc",         205.90,  382.38,    150.0, 0.03252),
    ("rougher_tails",       1794.10, 5096.25,    150.0, 0.00073),
    ("conc_thickener_uf",    205.90,  110.87,    150.0, 0.03252),
    ("filter_cake",          205.90,   20.36,    150.0, 0.03252),
)


@pytest.mark.parametrize("name,solids,water,F80,cu", PINNED_STREAMS)
def test_stream_outputs_within_band(sim, name, solids, water, F80, cu):
    s = sim["streams"][name]
    assert _approx(s.solids_tph, solids), (
        f"{name}.solids_tph {s.solids_tph:.3f} drifted from pinned {solids:.3f} "
        f"(>{TOL*100:.0f}%)")
    assert _approx(s.water_tph, water), (
        f"{name}.water_tph {s.water_tph:.3f} drifted from pinned {water:.3f}")
    if F80 is not None and s.F80_um is not None:
        assert _approx(s.F80_um, F80), (
            f"{name}.F80_um {s.F80_um:.3f} drifted from pinned {F80:.3f}")
    assert _approx(s.cu_grade, cu), (
        f"{name}.cu_grade {s.cu_grade:.5f} drifted from pinned {cu:.5f}")


# (power_key, kW)
PINNED_POWER = (
    ("crusher_kw",     329.0),
    ("sag_kw",        5600.0),
    ("bm_kw",        16538.9),
    # 3000 -> 3600 kW (5 -> 6 cells): deliberate re-pin after the gas-holdup
    # sizing basis (bound-desaturation Attempt 7) — the bank sizes up by
    # 1/(1 - air_fraction), so the default 25% gas holdup now costs its real
    # installed volume/power.
    ("flotation_kw",  3600.0),
)


@pytest.mark.parametrize("key,kw", PINNED_POWER)
def test_power_within_band(sim, key, kw):
    assert _approx(sim["power_kw"][key], kw), (
        f"power_kw[{key}] = {sim['power_kw'][key]:.2f} drifted from pinned "
        f"{kw:.2f} (>{TOL*100:.0f}%)")


def test_total_power_within_band(sim):
    # Re-pinned 25467.9 -> 26067.9 with flotation_kw 3000 -> 3600 (Attempt 7).
    total = sum(sim["power_kw"].values())
    assert _approx(total, 26067.9), (
        f"total power {total:.1f} drifted from pinned 26067.9")


def test_balance_within_band(sim):
    bal = sim["balance"]
    assert _approx(bal["overall_cu_recovery"], 0.8370), (
        f"overall_cu_recovery {bal['overall_cu_recovery']:.4f} drifted from "
        f"pinned 0.8370")
    assert _approx(bal["conc_grade_pct"], 3.25), (
        f"conc_grade_pct {bal['conc_grade_pct']:.3f} drifted from pinned 3.25")
    # Mass balance must remain closed regardless of physics tweaks.
    assert abs(bal["cu_closure_err"]) < 1e-6, (
        f"cu_closure_err = {bal['cu_closure_err']:.3e} (must be ~0)")
