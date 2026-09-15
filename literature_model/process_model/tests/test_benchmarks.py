"""Regression gates for the success-metric benchmark harness.

Reporting-only metrics (M1, M2) are not gated here — they're recorded by
hand into PROGRESS.md and drive follow-up plans, not CI failures.
"""
from process_model.benchmarks import run_m3a_closure, run_m3b_de_stability


def test_m3a_closure_under_threshold():
    """Mass balance must close to <0.1% across all (grade, route) combos."""
    out = run_m3a_closure(verbose=False)
    assert out["pass"], (
        f"M3a closure failed: max ratio {out['max_ratio']:.2e} "
        f">= 1.0e-03 threshold"
    )
    assert out["max_ratio"] < 1e-3


def test_m3b_de_stability_smoke():
    """Cheap DE convergence smoke test at one grade.

    Smoke threshold (20%) is wider than the recorded reading threshold (5%);
    the recorded reading is taken from `--full` and pasted into PROGRESS.md.
    """
    out = run_m3b_de_stability(grades=(0.010,), maxiter=20, popsize=8,
                               verbose=False)
    assert out["max_spread"] < 0.20, (
        f"M3b smoke spread {out['max_spread']*100:.2f}% > 20% threshold "
        f"— DE convergence is unstable even at one grade with quick params"
    )
