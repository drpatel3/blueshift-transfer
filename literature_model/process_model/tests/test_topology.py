"""Tests for topology refactor of throughput.simulate_cu_sulfide."""
import math
import pytest

from process_model.topology import (
    Topology, all_topologies, topology_label, _is_valid,
)
from process_model.throughput import simulate_cu_sulfide


def test_default_topology_is_all_on():
    t = Topology()
    assert t.thickener_enabled
    assert t.filter_enabled
    assert t.ball_mill_enabled
    assert t.cyclone_enabled


def test_invalid_combinations_filtered():
    """BM-off + cyclone-on is degenerate and must be excluded."""
    bad = Topology(ball_mill_enabled=False, cyclone_enabled=True)
    assert not _is_valid(bad)
    # And it must not appear in the enumerated list
    assert bad not in all_topologies()


def test_enumeration_count():
    """11 booleans × 3 regrind mill types -> ~6k raw combos; validity
    rules (flotation mandatory, chain dependencies, regrind type forced
    canonical when regrind off) cut to ~1000-1500. Exact count is not
    load-bearing but should be in this range."""
    topos = all_topologies()
    assert 50 < len(topos) < 4000
    # Defaults must be present
    assert Topology() in topos


def test_archetypes_subset_of_all():
    from process_model.topology import archetypes
    arch = archetypes()
    assert len(arch) >= 8
    all_topos = all_topologies()
    for a in arch:
        assert a in all_topos


def test_default_simulator_matches_implicit_topology():
    """Calling simulate_cu_sulfide without a topology kwarg must produce
    the same NPV-relevant numbers as passing Topology() explicitly. Regression
    guard for the topology refactor."""
    sim_implicit = simulate_cu_sulfide(2000.0, 0.008)
    sim_explicit = simulate_cu_sulfide(2000.0, 0.008, topology=Topology())
    a, b = sim_implicit["balance"], sim_explicit["balance"]
    assert math.isclose(a["overall_cu_recovery"], b["overall_cu_recovery"], rel_tol=1e-12)
    assert math.isclose(a["conc_grade_pct"], b["conc_grade_pct"], rel_tol=1e-12)
    assert math.isclose(a["total_power_kw"], b["total_power_kw"], rel_tol=1e-12)


def test_all_topologies_run_without_crashing():
    """Every enumerated topology must produce a finite NPV-input."""
    for t in all_topologies():
        sim = simulate_cu_sulfide(2000.0, 0.008, topology=t)
        R = sim["balance"]["overall_cu_recovery"]
        assert math.isfinite(R), f"{topology_label(t)} produced non-finite recovery"
        # Even a degenerate topology shouldn't recover negative or >100%
        assert -1e-6 <= R <= 1.0 + 1e-6


def test_filter_off_passes_thickener_uf_through():
    """With filter disabled, the 'filter_cake' key should equal the
    thickener underflow stream."""
    t = Topology(filter_enabled=False)
    sim = simulate_cu_sulfide(2000.0, 0.008, topology=t)
    cake = sim["streams"]["filter_cake"]
    uf = sim["streams"]["conc_thickener_uf"]
    assert math.isclose(cake.solids_tph, uf.solids_tph, rel_tol=1e-12)
    assert math.isclose(cake.water_tph, uf.water_tph, rel_tol=1e-12)


def test_thickener_off_passes_flotation_conc_through():
    """With thickener disabled, downstream concentrate should be the
    flotation concentrate at ~35% solids (no water removed)."""
    t = Topology(thickener_enabled=False, filter_enabled=False)
    sim = simulate_cu_sulfide(2000.0, 0.008, topology=t)
    cake = sim["streams"]["filter_cake"]
    rc = sim["streams"]["rougher_conc"]
    assert math.isclose(cake.solids_tph, rc.solids_tph, rel_tol=1e-12)
    assert math.isclose(cake.water_tph, rc.water_tph, rel_tol=1e-12)


def test_sag_only_recovery_is_lower_than_sag_bm():
    """Without the ball mill, flotation feed F80 stays at SAG product
    (~700 µm) — recovery should be substantially worse than the closed-
    circuit baseline."""
    sim_default = simulate_cu_sulfide(2000.0, 0.008, topology=Topology())
    sim_sag_only = simulate_cu_sulfide(
        2000.0, 0.008,
        topology=Topology(ball_mill_enabled=False, cyclone_enabled=False))
    R_default = sim_default["balance"]["overall_cu_recovery"]
    R_sag = sim_sag_only["balance"]["overall_cu_recovery"]
    assert R_sag < R_default, (
        f"SAG-only recovery {R_sag:.3f} should be worse than "
        f"SAG+BM {R_default:.3f}"
    )


def test_topology_label_format():
    assert topology_label(Topology()).startswith("[sulfide]")
    assert "rougher / thickener+filter" in topology_label(Topology())
    hydromet = Topology(leach_enabled=True, sx_enabled=True, ew_enabled=True)
    assert topology_label(hydromet).startswith("[hydromet]")
    assert "leach+SX+EW" in topology_label(hydromet)


def test_flotation_off_is_invalid():
    """Pure whole-ore hydromet is no longer offered: flotation is mandatory."""
    bad = Topology(flotation_enabled=False, leach_enabled=True,
                   sx_enabled=True, ew_enabled=True)
    assert not _is_valid(bad)
    assert bad not in all_topologies()


def test_simulator_rejects_invalid_direct_topology():
    """Direct simulator callers must not bypass topology validity rules."""
    bad = Topology(flotation_enabled=False, leach_enabled=True,
                   sx_enabled=True, ew_enabled=True)
    with pytest.raises(ValueError, match="invalid process topology"):
        simulate_cu_sulfide(2000.0, 0.008, topology=bad)


def test_simulate_returns_topology_in_output():
    """The simulator must echo the topology used so downstream code (TEA,
    optimizer) can read it back from the result dict."""
    t = Topology(thickener_enabled=False)
    sim = simulate_cu_sulfide(2000.0, 0.008, topology=t)
    assert sim["topology"] == t


def test_full_hydromet_chain_mass_balance():
    """Full hydromet chain (concentrate-leach + SX + EW + neut) closes
    mass balance and reports cathode-only revenue (no concentrate sale)."""
    t = Topology(regrind_enabled=True, cleaner_enabled=True,
                 leach_enabled=True, sx_enabled=True, ew_enabled=True,
                 neutralization_enabled=True, screen_enabled=True,
                 filter_enabled=False)
    sim = simulate_cu_sulfide(2000.0, 0.008, topology=t)
    b = sim["balance"]
    assert abs(b["cu_closure_err"]) < 1e-6
    # Cathode is non-zero when EW is on
    assert b["cu_in_cathode_tph"] > 0
    # In hydromet route the concentrate is consumed by leach, not sold.
    assert b["cu_in_conc_tph"] == 0.0
    # Overall recovery = cathode/feed; bounded but no strict ordering vs
    # sulfide-only because leach + SX + EW each take their cut.
    assert 0.3 < b["overall_cu_recovery"] < 1.0


def test_invalid_combos_filtered():
    bad_combos = [
        Topology(cleaner_enabled=True, regrind_enabled=False),
        Topology(sx_enabled=True, leach_enabled=False),
        Topology(ew_enabled=True, sx_enabled=False, leach_enabled=True),
        Topology(neutralization_enabled=True, leach_enabled=False),
    ]
    for t in bad_combos:
        assert not _is_valid(t), f"should reject: {t}"


def test_cleaner_grade_is_realistic():
    """Cleaner conc grade at 1.5% Cu feed should clear smelter spec, matching
    industry-typical mid- to high-grade Cu plants (Cobre Panama 24-26%
    at 0.4% feed, Escondida 28-30% at 0.8%, Antamina 30-38% at 0.9%).
    1.5% feed sits above the disclosed band so the model is allowed to
    deliver a saleable concentrate under the setpoint-tuned cleaner block.
    Low-grade merchant setpoint tracking is covered separately below."""
    t = Topology(regrind_enabled=True, cleaner_enabled=True,
                 n_cleaner_stages=3)
    sim = simulate_cu_sulfide(2000.0, 0.015, topology=t)
    grade = sim["balance"]["conc_grade_pct"]
    assert 25.0 < grade < 55.0, f"cleaner grade {grade:.1f}% outside expected band"


def test_cleaner_grade_tracks_merchant_setpoint():
    """Setpoint-tuned cleaner mass pull should flatten the concentrate-grade
    response across the disclosed chalcopyrite feed-grade range."""
    t = Topology(regrind_enabled=True, cleaner_enabled=True,
                 n_cleaner_stages=3)
    for feed_grade in (0.004, 0.008, 0.009):
        sim = simulate_cu_sulfide(2000.0, feed_grade, topology=t)
        grade = sim["balance"]["conc_grade_pct"]
        assert 25.0 < grade < 30.0, (
            f"{feed_grade*100:.1f}% feed cleaner grade {grade:.1f}% "
            "outside merchant setpoint band"
        )
