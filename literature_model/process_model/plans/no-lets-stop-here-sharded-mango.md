# Plan — Add grind P80 as the 11th decision variable in the optimizer

## Context

Phase 1 (topology selection) is **complete and shipped**: `process_model/topology.py`,
`process_model/optimizer.py`, 32/32 tests passing, 9 visualizations + 10 grade-sweep
flowsheets generated.

While reviewing the grade-sweep diagrams, the user noticed every PNG shows the same
flowsheet layout. Two reasons:

1. The grade-sweep script used `Topology()` defaults at every grade — didn't actually
   re-run the optimizer per grade.
2. **Even if it had**, the leaderboard would not change with grade. Capex is throughput-
   based (O'Hara equations), recovery is grade-independent in the simulator (set by
   physics, not feed grade), so the topology ranking is grade-invariant.

To get genuine grade-dependent variability — different grades selecting different
operating points — the optimizer needs at least one decision variable whose tradeoff
*depends on grade*. The cleanest such variable is **flotation feed P80**: finer grind
gives better recovery (extra revenue scales with grade) but costs more grinding power
(cost is grade-independent). At low grade, finer grind doesn't pay; at high grade,
it does. Same equipment, different operating point.

This change does **not** address the Option C lumped-capex caveat (topology choice
still won't differentiate by grade — that's a follow-up). But it does deliver
grade-dependent operating-point variability, which is the immediate user request.

## Design choice (user-confirmed)

- **Add `target_flot_P80_um` as the 11th continuous decision variable.**
- **Bounds: 75–200 µm** (industrial Cu rougher band per Wills' Mineral Processing
  Technology, 7th ed., Ch. 12).
- All cascading wiring already exists in `throughput.py` — no simulator changes needed.
- No topology change, no capex change, no Option B work — purely an optimizer-side
  extension.

## Why no `throughput.py` changes

`target_flot_P80_um` is already a field on `CuSulfideParams` (default 150 µm) and is
used in three places that all do the right thing automatically:

- `throughput.py:141` — Bond Ecs computation: `Ecs_total = 10·Wi·(1/√P80 − 1/√F80)`.
  Finer P80 raises grinding kW, which raises power_cost and media_cost in `tea.py`.
- `throughput.py:154,162,177,189` — sets the F80 of streams flowing into flotation.
- Flotation reads `flot_op.particle_size = flot_in.F80_um` (already in place via
  `flot_size_from_grind=True`), so kinetics responds to P80 changes through the
  Abrahamson collision kernel + Luttrell-Yoon p_collision + work-of-adhesion p_detach.

So the only file that needs to change is `optimizer.py`. Tests and visualizations
auto-pick up the new variable via the `DECISION_VARS` tuple.

## Files to modify

| File | Action |
|---|---|
| `process_model/optimizer.py` | Add 11th tuple to `DECISION_VARS`; update `_apply()` to set `p.target_flot_P80_um` from `x[10]` |
| `process_model/test_optimizer.py` | Update `test_decision_vars_align` (10 → 11), `test_apply_round_trip` (add P80 to test vector), add new `test_p80_changes_recovery_and_grinding_power` |
| `process_model/visualize.py` | Update `run_all` grade-sweep block to re-run the optimizer per grade (so the diagrams show *optimized* operating points, not defaults) |
| `process_model/PROGRESS.md` | Add Phase 1.7 entry |

## Existing functions to reuse, not rewrite

- `process_model/optimizer.py:DECISION_VARS, BOUNDS, NAMES, _apply` — the structure
  already supports adding a row; bounds/names auto-derive.
- `process_model/optimizer.py:Optimizer.run` — drives the per-grade sweep. Already
  takes `feed_grade` via constructor.
- `dataclasses.replace` (already imported in `optimizer.py:18`) — used to mutate
  `CuSulfideParams` with the new field.
- `process_model/visualize.py:plot_flowsheet` — already accepts a `sim_result`;
  re-render at each grade with the optimized sim.

## Implementation sketch (~30 lines of real code)

**`optimizer.py`:**
```python
DECISION_VARS = (
    # ... existing 10 ...
    ("target_flot_P80_um",     75.0, 200.0),  # µm, flotation feed P80
)

def _apply(x, base, tea_base):
    flot_op = replace(base.flot_op, ...)        # unchanged
    p = replace(base,
                flot_op=flot_op,
                flot_residence_time_min=float(x[3]),
                flot_cell_volume_m3=float(x[7]),
                circulating_load=float(x[8]),
                target_flot_P80_um=float(x[10]))  # NEW
    t = replace(tea_base, tailings_adverse_factor=float(x[9]))
    return p, t
```

**`test_optimizer.py` — new test:**
```python
def test_p80_changes_recovery_and_grinding_power():
    """Verify the new decision variable cascades through the simulator."""
    base = CuSulfideParams()
    tea_base = TEAParams()
    x_coarse = np.array([... default ..., 200.0])  # P80 at upper bound
    x_fine   = np.array([... default ...,  75.0])  # P80 at lower bound
    p_c, _ = _apply(x_coarse, base, tea_base)
    p_f, _ = _apply(x_fine,   base, tea_base)
    sim_c = simulate_cu_sulfide(2000, 0.008, p_c, topology=Topology())
    sim_f = simulate_cu_sulfide(2000, 0.008, p_f, topology=Topology())
    # Finer grind -> more BM kW
    assert sim_f["power_kw"]["bm_kw"] > sim_c["power_kw"]["bm_kw"]
    # Finer grind -> higher recovery
    R_f = sim_f["balance"]["overall_cu_recovery"]
    R_c = sim_c["balance"]["overall_cu_recovery"]
    assert R_f > R_c
```

**`visualize.py` — fix the grade-sweep CLI block:**
Replace the current "render `Topology()` with default params at each grade" loop with:
```python
for g in grades:
    grade_opt = Optimizer(feed_tph=feed_tph, feed_grade=g)
    res = grade_opt.run(maxiter=40, popsize=10, seed=42)
    win = res.winner
    p, t = _apply(win.best_x, CuSulfideParams(), TEAParams())
    sim = simulate_cu_sulfide(feed_tph, g, p, topology=win.topology)
    plot_flowsheet(topology=win.topology, sim_result=sim,
                   save=os.path.join(grade_dir, f"flowsheet_{g*100:.1f}pct.png"))
```
Trade-off: this adds ~10× compute time to the grade sweep (12 topologies × 10 grades).
With `maxiter=40, popsize=10` per call, expect ~5–8 minutes total. Acceptable for an
on-demand viz; not for inner-loop testing. CI tests will still use the cheap path.

## Verification

End-to-end acceptance criteria:

1. **Tests pass:**
   ```
   python -m pytest process_model/test_optimizer.py -v
   python -m pytest process_model/ -q   # full regression
   ```
   Expect 33/33 tests passing (32 prior + 1 new P80 cascade test).

2. **Optimizer CLI shows P80 in the operating-point output:**
   ```
   python -m process_model.optimizer
   ```
   The "best continuous params" section must list `target_flot_P80_um` with
   a value in [75, 200]. Likely lands somewhere around 100–130 µm at 0.8% Cu
   based on the recovery–power tradeoff.

3. **Visualization battery shows the new variable:**
   ```
   python -m process_model.visualize
   ```
   - `04_operating_point.png` should show 11 rows instead of 10
   - `05_param_sensitivity.png` should have an 11th subplot
   - `07_tornado.png` should show P80 with a non-trivial swing
   - `06_de_convergence.png` and `08_flowsheet.png` unchanged structure

4. **The user's actual ask — grade-dependent variability:**
   Re-run the grade sweep and inspect:
   ```
   python -m process_model.visualize
   ```
   The per-grade flowsheets in `process_model/figures/grade_sweep/` should now
   show *different optimized operating points* across grades. Specifically:
   - At 0.2–0.4% Cu: optimizer should pick **coarser** P80 (closer to 200 µm),
     since the marginal recovery gain doesn't pay for grinding power
   - At 1.5–2.0% Cu: optimizer should pick **finer** P80 (closer to 75 µm),
     since chasing recovery is worth the extra grinding cost
   - The Cu recovery on each diagram should differ correspondingly (e.g. ~75% at
     0.2% Cu, ~85% at 2.0% Cu)
   - The conc grade and tonnage on each diagram will also differ
   - Topology will likely still be the same across grades — that's the Option C
     caveat from the prior plan, not a regression here.

5. **Sanity: the new decision variable can't just hit a bound on every run.**
   If the optimizer always picks 75 µm or always picks 200 µm, that's a signal
   that one direction's tradeoff dominates — worth investigating whether the
   bounds are realistic or whether the cost coefficients (media $/kWh, power
   $/kWh) need recalibration.

## Out of scope (deferred)

- **Option B capex breakout** (Mular sub-fractions for flotation/thickener/filter/
  dryer). Still the right move when topology selection needs to differentiate by
  grade. Not in this change.
- **Particle size distribution** beyond the F80/P80 scalar. Real plants think in
  terms of full PSD; we still don't.
- **Variable feed throughput** as a decision variable. Capex scales with tph, so
  this would let the optimizer pick plant size — bigger scope change.
- **Grade-coupled mining cost** ($/t × grade premium for richer feed). Affects
  breakeven but not topology choice.
