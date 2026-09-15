# PLAN: TEA-Driven Optimization for the Cu Sulfide Digital Twin

## Context

`process_model/` now simulates a reference Cu-sulfide plant end-to-end
(crusher → SAG → BM closed-circuit w/ cyclone → rougher flotation → thickener
→ filter), with correct mass balance and Bond-basis total power. The current
outputs are **engineering metrics** (recovery %, concentrate grade, total
kW). That is not what the business cares about.

The objective is **to maximize financial gain (NPV)**, not recovery or
throughput. Pushing recovery from 88% → 92% often destroys value once the
marginal concentrate revenue is netted against extra power, reagent, capex,
and TC/RC. Any optimizer we build must trade those off in dollars.

Inspiration is PROMMIS (NETL) — Pyomo + IDAES wrapping first-principles
equations, with UKy lab data supplying parameter values (not the model).
We already have the equations, so PROMMIS is a structural reference, not a
dependency.

**This document defines three deliverables, treated as three separate
projects**:

1. **Shared foundation — the TEA module.** Built once. Feeds both
   optimization paths. Must exist before either path has a usable objective
   function.
2. **Path B — lightweight Pyomo/scipy wrapper** over our existing
   pure-function stages. Days of work. Solves the NPV optimization for a
   fixed topology.
3. **Path A — full PROMMIS-style IDAES port.** Weeks of work. Self-contained
   rewrite. Unlocks equation-oriented solving, automatic sensitivity, and
   (separately) a superstructure MILP for topology selection.

Path B and Path A are **not** sequential phases of one effort — they are
alternative implementations of the same optimization layer, with very
different cost/benefit profiles. We build the TEA module first, run Path B
to prove the NPV objective works, then decide whether Path A is warranted.

---

## Project 1 — TEA Module (shared foundation)

### Goal
Turn a simulator output (the `dict` returned by `simulate_cu_sulfide`) into
a single NPV number, plus itemized cashflows for diagnostics.

### Scope
New file: `process_model/tea.py`. Pure functions. No dependency on Pyomo or
any solver — just arithmetic over simulator outputs.

### Components

1. **Revenue** (per hour, then annualized at plant availability):
   - `payable_cu_tph = conc_tph × conc_grade × payable_fraction`
     (payable_fraction ≈ 0.965, deductible ≈ 1 unit of Cu)
   - `gross_revenue = payable_cu_tph × cu_price_per_t`
   - `tc_rc_deduction = conc_tph × TC + payable_cu_tph × RC`
   - Penalty terms (As, Bi) — stub to zero for now, flagged for later
2. **Opex** from simulator outputs:
   - Power cost: `sum(power_kw) × hours × $/kWh`
   - Reagents: collector/frother/lime g/t × ore_tph × $/t (dose tables
     parameterized per flowsheet)
   - Grinding media & liners: $/kWh on mill power (industry scalar)
   - Labor, maintenance: %-of-capex per year
   - Water, tailings: $/t ore scalar
3. **Capex** from equipment sizing:
   - Mill capex from kW via Mular-Poulter scaling: `C = a × kW^b`
   - Flotation: cell_volume × N_cells × unit_cost
   - Cyclone: count × unit_cost
   - Thickener: diameter (already in `thickener.py` via `th.cost`) — reuse
     the existing `cost()` function, don't re-implement
   - Filter, conveyors, crushers: reference-cost + scaling exponent
   - Installation factor (~2.5× installed), EPCM, contingency
4. **Cashflow & NPV**:
   - Inputs: mine_life_years, discount_rate, tax_rate, depreciation
     (straight-line), sustaining_capex_pct
   - `fcf_t = (revenue - opex - depreciation) × (1 - tax) + depreciation -
     sustaining_capex`
   - `NPV = -capex_0 + Σ fcf_t / (1 + r)^t`

### Interface

```python
@dataclass
class TEAParams:
    cu_price_per_t: float = 9500.0 * 2204.6  # ~$9.5k/lb Cu in $/t
    tc_per_t_conc: float = 75.0
    rc_per_t_cu: float = 0.075 * 2204.6
    payable_fraction: float = 0.965
    power_cost_per_kwh: float = 0.08
    hours_per_year: float = 8000
    mine_life_years: int = 20
    discount_rate: float = 0.08
    tax_rate: float = 0.25
    sustaining_capex_pct: float = 0.02
    # ... reagent doses, unit costs, installation factors

def evaluate(sim_result: dict, tea: TEAParams) -> dict:
    """Returns {npv, capex, annual_opex, annual_revenue, payback_years, irr}."""
```

### Files to add
- `process_model/tea.py` — module
- `process_model/PROGRESS.md` — add TEA entry to status table

### Verification
- Unit tests in `tests/test_tea.py`: feed a known sim result, check NPV, capex,
  annual cashflow match hand calculation within 0.1%.
- Integration: `python -m process_model.throughput --tea` prints
  NPV/capex/opex alongside the existing streams/power report.

---

## Project 2 — Path B: Lightweight Optimization Wrapper

### Goal
Find the `CuSulfideParams` values that maximize NPV subject to physical and
commercial constraints. Minimal new dependencies. Keep the pure-function
physics intact.

### Scope
New file: `process_model/optimize.py`. Uses `scipy.optimize` (already
transitively available) or a minimal Pyomo NLP. No IDAES.

### Design
1. **Decision vector** — a subset of `CuSulfideParams` fields flagged as
   tunable:
   - `sag_Ecs_kwh_t`, `circulating_load`, `target_flot_P80_um`
   - `flot_op.num_cells`, `flot_op.ret_time`, `flot_op.air_fraction`
   - `flot_op.sp_power`, reagent dose (when added)
2. **Objective**: `-NPV` (minimize). Wrap
   `simulate_cu_sulfide → tea.evaluate → -npv` as a single callable.
3. **Constraints** — penalty terms or `scipy.optimize.minimize` constraints:
   - Concentrate grade ≥ smelter minimum (e.g., 22% Cu)
   - Penalty-element limits (placeholder)
   - Total grinding power ≤ installed capacity
   - Recovery ≥ contractual floor (if any)
4. **Solver**: `scipy.optimize.differential_evolution` for global coverage
   on ~6–10 vars (the physics is non-convex), followed by a local
   `minimize(method="SLSQP")` polish. If we want gradients, wrap with Pyomo
   and IPOPT — this is the natural upgrade path.
5. **Sensitivity**: one-at-a-time perturbation around the optimum, printed
   as a tornado chart of ∂NPV/∂var.

### Files to add
- `process_model/optimize.py`
- `process_model/PROGRESS.md` — Path B entry

### Verification
- Run `python -m process_model.optimize` — prints optimized parameter vector,
  optimized NPV, and the delta vs baseline NPV.
- Sanity: optimized NPV ≥ baseline NPV (otherwise the optimizer regressed).
- Sensitivity report surfaces the top 3 NPV drivers — these should make
  physical sense (typically Cu price, grind P80, recovery).

### Effort
~2–4 days: TEA tests (0.5d), optimizer scaffolding (1d), decision vector +
constraints wiring (1d), sensitivity + reporting (0.5d), documentation
(0.5d).

### Explicit non-goals
- No topology decisions (cleaner vs no cleaner, etc.) — fixed circuit.
- No stochastic optimization (Cu price uncertainty) — deterministic only.
- No multi-objective Pareto front — single NPV objective.

---

## Project 3 — Path A: Full PROMMIS-Style IDAES Port

### Goal
Rebuild the simulator in the PROMMIS pattern: Pyomo `ConcreteModel` + IDAES
`FlowsheetBlock` with each stage as a `UnitModelBlockData` subclass, solved
with IPOPT. Unlocks:
- Equation-oriented solves (no per-stage fixed-point iteration)
- Automatic Jacobians / gradients for fast optimization
- Built-in sequential-decomposition recycle handling
- A later superstructure MILP for topology selection (separate follow-on)

### Scope
**This is a separate project, not a follow-on of Path B.** It is the
structural analog of PROMMIS's `src/prommis/uky/uky_flowsheet.py`. Days of
design up front; weeks of implementation.

### Components
1. **Dependencies**: `pyomo`, `idaes-pse`, `idaes get-extensions` (IPOPT
   binary). Non-trivial install footprint.
2. **Property package** (`CuSulfideParameterBlock`): components Cu, Fe, S,
   gangue, H₂O; phases solid/liquid; molecular weights; densities.
   Mirrors PROMMIS's `CoalRefuseParameters`.
3. **Unit models** — one IDAES class per stage:
   - `CrusherUnitModel`, `SAGUnitModel`, `BallMillUnitModel`,
     `HydrocycloneUnitModel`, `FlotationCellBank`, `ThickenerUnitModel`,
     `FilterUnitModel`
   - Each declares `Var`s, `Constraint`s, and `inlet`/`outlet` `Port`s.
   - Equations come straight from existing `process_model/*.py` files
     (crusher.py, sag.py, etc.) — transliterated from pure-functions into
     Pyomo constraints. Physics is unchanged.
4. **Flowsheet assembly**: `Arc` connectors, `TransformationFactory(
   "network.expand_arcs").apply_to(m)`, tear streams on cyclone U/F.
5. **Initialization**: `SequentialDecomposition` with tear guesses from a
   Path B (or current `throughput.py`) run.
6. **Costing**: reuse TEA module as Pyomo `Expression`s on the model, or
   adopt `QGESSCosting` pattern from PROMMIS.
7. **Optimization**: unfix decision vars, add `Objective(expr=m.fs.npv,
   sense=maximize)`, re-solve with IPOPT.
8. **Superstructure layer (follow-on)**: separate `superstructure/`
   subpackage with Gurobi MILP over stages × candidate techs, fed by
   reduced-order surrogates fitted from Path A simulator runs. PROMMIS
   treats this as its own module (`docs/superstructure/`) — we would too.

### Files to add (high level)
- `process_model/idaes/property_package.py`
- `process_model/idaes/unit_models/<stage>.py` × 7
- `process_model/idaes/flowsheet.py`
- `process_model/idaes/costing.py`
- `process_model/idaes/README.md` — install + run instructions
- Later: `process_model/superstructure/`

### Verification
- Build `m`, call `idaes.core.util.DOF` → DOF = 0 for square solve.
- IPOPT square-solve reproduces `throughput.py` streams within 0.5% on a
  reference case (regression test).
- Unfix decision vars, add objective, re-solve → NPV ≥ Path B NPV on the
  same problem (sanity — equation-oriented solver shouldn't do worse).
- Integration test: `pytest tests/test_idaes_flowsheet.py`.

### Effort
~2–4 weeks first cut: property package (3d), unit model refactors (5–8d),
flowsheet + recycle init (2d), costing wiring (2d), optimization problem
+ tests (2d), debugging IDAES-isms (ongoing).

### Explicit non-goals (for this phase)
- Superstructure MILP — deliberately deferred to its own follow-on project.
- Dynamic / unsteady-state simulation.
- Replacing `process_model/*.py` pure-functions — those stay as the
  reference implementation and Path B's substrate.

---

## Decision Framework

- **Always build Project 1 (TEA) first.** Both paths need it; it has the
  highest value per day of work. Without it, every optimization is
  engineering vanity.
- **Then build Path B.** It will answer "where does NPV actually come
  from?" in days, with no new deps. In ~80% of cases the answer is
  obvious enough that Path A is unjustified effort.
- **Build Path A only if** (a) Path B hits solver limits (>20 decision
  vars, non-smooth constraints), OR (b) we want the superstructure layer
  for topology selection, OR (c) external stakeholders expect a PROMMIS/
  IDAES-shaped deliverable.
- Path A does not supersede Path B — they can coexist. Path B stays as the
  fast exploratory tool; Path A becomes the production artifact.

---

## Critical files to reference

- `process_model/throughput.py` — current end-to-end simulator, entry point
  for TEA + optimization
- `process_model/flowsheet.py` — pure-function `run_*` stages (Path A ports
  these directly)
- `process_model/thickener.py` — already has a `cost()` function; TEA
  module must **reuse**, not duplicate
- `process_model/PROGRESS.md` — status board; update as each project lands
- `CLAUDE.md` — project coding preferences (simple algorithms, minimal
  cross-file changes)

## Verification (end-to-end, all three projects)

1. `python -m process_model.throughput` → baseline sim (already works).
2. After Project 1: `python -m process_model.throughput --tea` → prints
   NPV. Hand-check against a spreadsheet once.
3. After Project 2: `python -m process_model.optimize` → optimized NPV ≥
   baseline NPV, sensitivity report is physically sensible.
4. After Project 3 (if built): `pytest tests/test_idaes_flowsheet.py` →
   IDAES square solve reproduces throughput.py within 0.5%; optimization
   solve reproduces Path B NPV within 1%.
