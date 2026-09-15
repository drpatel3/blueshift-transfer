"""Tests for the topology-selecting NPV optimizer."""
import numpy as np

from process_model.optimizer import (
    Optimizer, OptimizationResult, TopologyResult,
    BOUNDS, NAMES, DECISION_VARS, _apply,
)
from process_model.topology import Topology
from process_model.throughput import CuSulfideParams
from process_model.tea import TEAParams, evaluate
from process_model.throughput import simulate_cu_sulfide


def _vec(**overrides):
    """Build a decision vector aligned to the current DECISION_VARS (midpoint
    default), with name overrides. Robust to variables being added/removed."""
    return np.array([overrides.get(name, 0.5 * (lo + hi))
                     for name, lo, hi in DECISION_VARS])


# Where each decision-variable NAME lands after _apply, for round-trip checks.
def _decoded(p, t):
    return {
        "sp_power": p.flot_op.sp_power,
        "sp_gas_rate": p.flot_op.sp_gas_rate,
        "frother_conc": p.flot_op.frother_conc,
        "air_fraction": p.flot_op.air_fraction,
        "slurry_fraction": p.flot_op.slurry_fraction,
        "contact_angle": p.flot_op.contact_angle,
        "flot_residence_time_min": p.flot_residence_time_min,
        "circulating_load": p.circulating_load,
        "target_flot_P80_um": p.target_flot_P80_um,
        "tailings_adverse_factor": t.tailings_adverse_factor,
    }


def test_decision_vars_align():
    """Bounds, names, and the decoder must agree on length and order."""
    assert len(BOUNDS) == len(NAMES) == len(DECISION_VARS)
    # Lower bound < upper bound for every variable
    for name, lo, hi in DECISION_VARS:
        assert lo < hi, f"{name}: lo={lo} >= hi={hi}"


def test_apply_round_trip():
    """Every variable currently in the DOE must decode (by name) to its x
    value; this is robust to DOE membership changes."""
    base = CuSulfideParams()
    tea_base = TEAParams()
    vals = {name: 0.5 * (lo + hi) for name, lo, hi in DECISION_VARS}
    x = np.array([vals[name] for name, _, _ in DECISION_VARS])
    p, t = _apply(x, base, tea_base)
    decoded = _decoded(p, t)
    for name in NAMES:
        assert decoded[name] == vals[name], (
            f"{name} did not round-trip: got {decoded[name]}, want {vals[name]}")


def test_p80_changes_recovery_and_grinding_power():
    """P80 cascades through Bond kW and the recovery ceiling:
    - Finer P80 always demands more BM kW (Bond's law, true across the range).
    - Recovery is hump-shaped in P80: improves with finer grind in the coarse
      regime (liberation-limited) and falls back at very fine grind (slimes
      effect in the kinetics model). We test the coarse-regime direction
      (100 vs 175 um) where finer grind genuinely wins on both kW and recovery.
      The full grind-vs-NPV trade-off is validated by the liberation-factor
      sweep in _bound_probe and the grade-sweep tests."""
    base = CuSulfideParams()
    tea_base = TEAParams()
    # Compare within the productive grind regime (100-175 um), not extreme ends.
    p_c, _ = _apply(_vec(target_flot_P80_um=175.0), base, tea_base)
    p_f, _ = _apply(_vec(target_flot_P80_um=100.0), base, tea_base)
    sim_c = simulate_cu_sulfide(2000.0, 0.008, p_c, topology=Topology())
    sim_f = simulate_cu_sulfide(2000.0, 0.008, p_f, topology=Topology())

    # Finer P80 -> higher BM kW (Bond: more energy to grind finer)
    assert sim_f["power_kw"]["bm_kw"] > sim_c["power_kw"]["bm_kw"], (
        f"finer P80 should require more BM kW: "
        f"fine={sim_f['power_kw']['bm_kw']:.0f} vs "
        f"coarse={sim_c['power_kw']['bm_kw']:.0f}"
    )
    # Liberation cap: coarser grind (175 um > 150 um ref) gets a penalised
    # ceiling, so its liberation_factor < 1. Finer grind is at the full ceiling.
    lib_c = sim_c["flotation"]["liberation_factor"]
    lib_f = sim_f["flotation"]["liberation_factor"]
    assert lib_c < lib_f, (
        f"coarser P80 should have a lower liberation factor: "
        f"coarse={lib_c:.3f} vs fine={lib_f:.3f}"
    )


def test_optimizer_respects_bounds():
    """Every coordinate of the result must be within its declared bounds."""
    opt = Optimizer()
    tr = opt.run_one(Topology(), maxiter=10, popsize=8, seed=42)
    for i, (lo, hi) in enumerate(BOUNDS):
        x = tr.best_x[i]
        assert lo - 1e-9 <= x <= hi + 1e-9, (
            f"{NAMES[i]} = {x} outside [{lo}, {hi}]"
        )


def test_optimizer_beats_or_matches_hand_tuned_baseline():
    """DE should find an NPV at least as good as the hand-tuned operating
    point on a viable cleaner+regrind concentrator topology. 5% slack
    accounts for DE stochasticity at low maxiter. Use cleaner+regrind +
    1.5% Cu feed so the resulting concentrate clears the smelter min-grade
    payable threshold (bare rougher fails it on most realistic feeds)."""
    topo = Topology(regrind_enabled=True, cleaner_enabled=True,
                    n_cleaner_stages=3)
    base_sim = simulate_cu_sulfide(2000.0, 0.015, topology=topo)
    base_npv = evaluate(base_sim, TEAParams())["npv"]
    opt = Optimizer(feed_tph=2000.0, feed_grade=0.015)
    tr = opt.run_one(topo, maxiter=20, popsize=10, seed=42)
    # Tolerance: 5% of magnitude in either direction; handles negative NPVs.
    tol = 0.05 * abs(base_npv)
    assert tr.best_npv >= base_npv - tol, (
        f"DE NPV ${tr.best_npv/1e9:.2f}B substantially worse than baseline "
        f"${base_npv/1e9:.2f}B (tolerance ${tol/1e9:.2f}B)"
    )


def test_optimizer_runs_full_topology_sweep():
    """Smoke test the outer enumeration loop. Use tight maxiter to keep
    test fast (~20-30 s)."""
    opt = Optimizer()
    res = opt.run(maxiter=5, popsize=6, seed=42)
    assert isinstance(res, OptimizationResult)
    # Default sweeps the curated archetype set, not all 500+ topologies.
    from process_model.topology import archetypes
    assert len(res.results) == len(archetypes())
    assert all(isinstance(r, TopologyResult) for r in res.results)
    # Sorted high-NPV first
    npvs = [r.best_npv for r in res.results]
    assert npvs == sorted(npvs, reverse=True)
    # Winner accessor
    assert res.winner.best_npv == max(npvs)


def test_simulator_failure_returns_finite_penalty():
    """The objective function must never return NaN/inf, even if the
    simulator crashes for some operating point — DE will get stuck
    otherwise. Forge a vector that will cause issues by setting
    impossibly-low slurry fraction (close to lower bound)."""
    from process_model.optimizer import _objective
    base = CuSulfideParams()
    tea_base = TEAParams()
    # Vector at the lower-bound corner
    x = np.array([lo for _, lo, _ in DECISION_VARS])
    val = _objective(x, base=base, tea_base=tea_base, topo=Topology(),
                     feed_tph=2000.0, feed_grade=0.008)
    assert np.isfinite(val), f"objective returned {val} (not finite)"
