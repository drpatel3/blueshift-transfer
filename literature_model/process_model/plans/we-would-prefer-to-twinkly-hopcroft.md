# Add missing process stages to `process_model`

## Context

The Cu-sulfide chain in `throughput.py` covers crusher → SAG+BM → cyclone → rougher flotation → thickener → vacuum belt filter. The user's expanded stage list adds:

- **Sulfide-flotation extensions**: ball-mill regrind, vibrating screen, scavenger / cleaner flotation
- **Hydromet branch**: atmospheric leach, SX mixer-settler, EW cell, neutralization
- **Concentrate dewatering**: pressure filter (in addition to the existing belt filter)
- **(Heap leach)**: deferred — no source available yet

Adding these stages serves two purposes:

1. The optimizer can pick between sulfide-flotation and hydromet routes per ore grade.
2. More stages = more capex lines, which drags the unrealistic 159% IRR down toward defensible territory (the absolute-NPV gap is mostly methodology — taxes, labor, contingency — but stage coverage is the prerequisite).

Sources confirmed by exploration:

- **SME Mineral Processing Handbook** equations transcribed in `process_model/references/SME_handbook_equations_bounds.xlsx` cover: SAG+BM (Eq 20.10–20.14, already wired), vibrating screen (Table 20.8 + Eq 20.25), gravity thickener (Eq 20.32, already wired), EW cell (Eq 20.70 / 20.71 / i_L cap), neutralization (Eq 20.64 / 20.65 / 20.66).
- **`optimization_model/internal_dev/main/physical_parameters/leaching.py:14-84`** has shrinking-core atmospheric-leach kinetics: `dX_Cu/dt = 3·k₀·exp(-Ea/RT)·(P_O₂)^n·(1-X_Cu)^(2/3)`, Ea = 70-90 kJ/mol, P_O₂ = 10-15 bar, H₂SO₄/Cu = 6.17 kg/kg.
- **`process_model/sx.py`** already has Moreno (2009) mixer-settler ODEs and a `run_sx()` shim in `flowsheet.py:104` — never called from `throughput.py`.
- **Pressure filter**: NOT in SME xlsx, NOT in optimization_model. Defer.
- **Heap leach**: NOT in SME xlsx, NOT in optimization_model (only atmospheric). Defer.

User-confirmed answers:
- Capex sources: Perry T9-50 power-laws where they exist; clearly-flagged vendor-quote placeholders elsewhere.
- Stages without an available source: defer (don't stub).

## Goal

Build all stages with a confirmed source, wired into a topology-selectable simulator + per-stage capex in `tea.py`. Defer pressure filter and heap leach. Hydromet route runs in parallel to sulfide flotation — the topology picks one or the other (or both, where physically valid).

## Phasing (one commit per phase)

### Phase 1A — physics modules (4 new files, no wiring)
- `process_model/screen.py` — SME Table 20.8 (deck area = U / (A·B·C·D·E·F·G·H·J)) + Eq 20.25 (DBD ≤ 3-4× aperture). Imperial units in equations; convert at module boundary.
- `process_model/leach.py` — shrinking-core atmospheric, ported from `optimization_model/.../leaching.py:14-84`. Cite Sohn & Wadsworth (1979) shrinking-core via the optimization_model pedigree. No heap-leach branch.
- `process_model/ew.py` — SME Eq 20.70 (Faraday's law on cathode mass), Eq 20.71 (cell voltage = I·R + 1.96 V overpotential), `i_L = 18.6·[Cu g/L]` cap with i_op ≤ 0.40·i_L gate.
- `process_model/neutralization.py` — SME Eq 20.64 / 20.65 / 20.66 stoichiometry. Lime demand for stage-1 (pH 4-5, acid + Fe/As) and stage-2 (pH 6-8, base metal MSO₄). Tank volume from residence × #stages × flow.

### Phase 1B — capex constants in `tea.py`
- `SCREEN_BASIS` — Perry T9-50 vibrating screen row if found; else flagged placeholder ($/m² of deck, 2025 USD, with comment that it needs vendor verification).
- `LEACH_TANK_BASIS` — reuse existing `ATM_TANK_BASIS` for autoclave/leach tank (Perry T9-50 atm tank already in 2025 USD).
- `SX_MIXER_SETTLER_BASIS` — `ATM_TANK_BASIS` per stage × #stages (3-stage E-W, 1-stage S typical).
- `EW_CELL_BASIS` — already exists, just plumbed in.
- `NEUTRALIZATION_TANK_BASIS` — `ATM_TANK_BASIS` per stage × #stages.
- `REGRIND_MILL_BASIS` — reuse `BM_MILL_BASIS` (Mular & Poulter 2002) sized on regrind kW.
- Cleaner / scavenger flotation reuses `FLOTATION_CELL_BASIS` (Arfania 2017) per cell × num_cells.
- New opex defaults (added next to existing `DEFAULT_*` block): `DEFAULT_LIME_COST_PER_T`, `DEFAULT_ORGANIC_COST_PER_L` (SX extractant), `DEFAULT_O2_COST_PER_T` (autoclave). Use placeholder values with citations to recent BC/Chile mine reports if found, else flag.

### Phase 1C — wire stages into `throughput.py`
- After crusher: optional `run_screen` (returns oversize/undersize streams).
- After rougher flot: optional regrind branch (BM regrind on rougher conc → cleaner feed).
- After regrind: optional cleaner flotation (reuse `run_flotation` with cleaner-feed kinetics).
- New parallel hydromet branch (input = ore stream, branch-gated by topology):
  - `run_leach` → leach residue + PLS
  - `run_sx` (already exists in `flowsheet.py:104`, just call it)
  - `run_ew` → cathode + spent electrolyte
  - `run_neutralization` → neutralized tails (gypsum + Fe/As + base metal hydroxides)
- The `filter_cake` key still resolves to whichever final concentrate stream applies; TEA reads it for revenue. For hydromet, the "filter_cake" becomes cathode_Cu (no TC/RC, full payable).

### Phase 1D — topology + optimizer + tests
- `topology.py`: extend `Topology` with `screen_enabled`, `regrind_enabled`, `cleaner_enabled`, `leach_enabled`, `sx_enabled`, `ew_enabled`, `neutralization_enabled`. Add `_is_valid` rules:
  - `cleaner_enabled` ⇒ `regrind_enabled` (no cleaning of unrelished conc)
  - `ew_enabled` ⇒ `sx_enabled` ⇒ `leach_enabled` (chained dependency)
  - `neutralization_enabled` ⇒ `leach_enabled` (only neutralize leach effluent)
  - At least one of (rougher-flot, leach) must be on (something has to recover Cu)
- `optimizer.py`: add bounds for new continuous parameters (regrind P80, cleaner kinetics knob, leach temperature, leach time, EW current density fraction). Topology enumeration grows from 12 to ~40-60 valid combos; a maxiter knock-down or a coarser DE may be needed to keep CLI runtime under 5 minutes.
- `tea.py`: add lines for screen, leach, sx, ew, neutralization, regrind, cleaner. Cathode-revenue branch: when EW is on, revenue = cathode tph × full Cu price (no TC/RC, payable_fraction = 1.0).
- Tests: extend `test_topology.py` with new validity rules and stage-toggle smoke tests; extend `test_tea.py` with capex-line existence and toggle-zeroes-line assertions for each new stage.

## Files to create / modify

**New files:**
- `process_model/screen.py`
- `process_model/leach.py`
- `process_model/ew.py`
- `process_model/neutralization.py`

**Modified:**
- `process_model/tea.py` — new constants, new evaluate() lines, hydromet revenue branch
- `process_model/throughput.py` — wire new run_* calls under topology gates
- `process_model/flowsheet.py` — add `run_screen`, `run_leach`, `run_ew`, `run_neutralization` shims (callers of the new modules)
- `process_model/topology.py` — extend Topology dataclass + validity rules + label generator
- `process_model/optimizer.py` — bounds for new decision variables, expand topology enumeration
- `process_model/test_topology.py` — new tests
- `process_model/test_tea.py` — new tests
- `process_model/PROGRESS.md` — Phase 2 entry

## Existing functions / constants to reuse

- `process_model/flowsheet.py:104` `run_sx()` — already implemented, never called. Wire it.
- `process_model/sx.py` Moreno (2009) ODE solver — full SX physics, just needs harness.
- `process_model/tea.py` `ATM_TANK_BASIS = (9_300 × 2.383, 0.38, 0.53)` — reused for leach, SX, neutralization tanks.
- `process_model/tea.py` `EW_CELL_BASIS = (383_000 × 2.383, 94.0, 0.81)` — already in file, plumb it.
- `process_model/tea.py` `BM_MILL_BASIS` — reuse for regrind.
- `process_model/tea.py` `FLOTATION_CELL_BASIS` (Arfania 2017) — reuse for scavenger/cleaner.
- `process_model/tea.py` `perry_cost`, `select_basis` — capex helpers.
- `process_model/tea.py` `DEFAULT_*` constants — pattern for new opex unit rates.

## Out of scope (deferred, NOT in this build)

- **Pressure filter**: no source agreed yet. Belt filter remains the only dewatering option.
- **Heap leach**: only atmospheric stirred-tank leach in this build. Heap-pad column kinetics + irrigation rate + percolation belong to a separate physics module.
- **Scavenger flotation as a third bank**: cleaner_enabled covers most use cases; adding scavenger triples the test surface for marginal gain. Defer until grade-sweep shows it's needed.
- **Methodology fixes** (post-tax DCF, labor / G&A, EPCM, contingency): real IRR-shrinkers but separate task. Note in PROGRESS that stage coverage alone won't get IRR to defensible level.

## Verification

End-to-end checks per phase:

1. **Phase 1A**: each new module has a `__main__` smoke block that prints sample inputs/outputs at typical operating points. `python -m process_model.screen` etc. should run without import errors.
2. **Phase 1B**: `python -c "from process_model.tea import *; print(SCREEN_BASIS, EW_CELL_BASIS, ...)"`. Plus a unit test per new constant in `test_tea.py`.
3. **Phase 1C**: `python -m process_model.throughput --tea` runs cleanly with all new stages on (full sulfide chain + full hydromet chain in parallel). Mass balance closes. New TEA breakdown lines appear with non-zero values.
4. **Phase 1D**:
   - `pytest process_model/ -x` — all existing 41 tests + new ones pass.
   - `python -m process_model.optimizer` — leaderboard shows topology mix including hydromet variants. Hydromet topologies should win at low grade (Cu price - TC/RC bites less); flotation should win at high grade. Capex now spans a wider range ($120M to $400M+).
   - `python -m process_model.visualize --grade-sweep` — topology winner shifts across grades.
5. Commit + push after each phase passes its checks.

## Sequencing call

Execute Phase 1A → 1B → 1C → 1D, one commit per phase, so we can review numbers at each step (capex line emergence at 1B, mass-balance closure at 1C, optimizer behavior at 1D). Total: ~4 commits, 41 → ~60+ tests, ~1000 new LOC across modules.
