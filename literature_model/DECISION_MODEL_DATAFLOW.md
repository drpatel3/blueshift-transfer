# Decision-Model Data Flow — process_model Package

## Entry Point & Parameters

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 1 | `__main__.main()` | CLI args: `--grade` (frac), `--tph` (float), `--maxiter`, `--popsize`, `--workers` | Parse CLI; instantiate Optimizer with feed_tph, feed_grade | `Optimizer(base_params, tea_params, feed_tph, feed_grade)` | Step 2 |
| 2 | `optimizer.Optimizer.run()` | `topologies` (list[Topology] \| None), maxiter, popsize, workers | Enumerate topology archetypes (default ~12); defer each to run_one() | `OptimizationResult(feed_tph, feed_grade, results[])` | Step 3 |
| 3 | `optimizer.Optimizer.run_one(topo)` | `topo: Topology`, maxiter, popsize, seed, workers | Set up DE objective closure with base_params, tea_params, topo | `TopologyResult(topology, label, best_x[], best_npv, irr, capex, cu_recovery, conc_grade_pct, de_message)` | Step 4 |

## Topology Gating

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 4 | `topology.all_topologies()` / `topology.archetypes()` | (none) | Enumerate boolean stage toggles (11 bools × 3 mill types); validate via `_is_valid()`; filter degenerate combos | `list[Topology]` with fields: `flotation_enabled`, `ball_mill_enabled`, `cyclone_enabled`, `regrind_enabled`, `cleaner_enabled`, `thickener_enabled`, `filter_enabled`, `leach_enabled`, `heap_leach_enabled`, `sx_enabled`, `ew_enabled`, `neutralization_enabled`, `regrind_mill_type`, `n_cleaner_stages` | Step 3 (Optimizer loop) |
| 5 | `topology.topology_label(topo)` | `topo: Topology` | Generate human-readable label for leaderboard display (e.g., "[sulfide] SAG+BM closed / rougher+cleaner(2-stage) / thickener+filter") | `label: str` | Leaderboard print |

## Optimization Inner Loop (Differential Evolution)

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 6 | `optimizer._objective(x[], *, base, tea_base, topo, feed_tph, feed_grade)` | Decision vector `x[]` (10-D); `base: CuSulfideParams`; `tea_base: TEAParams`; `topo: Topology`; feed conditions | Catch exceptions; apply decision vector to base params; run simulator; evaluate TEA; return `-NPV` for minimization | `-NPV: float` (or 1e12 penalty on failure) | scipy.optimize.differential_evolution |
| 7 | `optimizer._apply(x[], base, tea_base)` | Decision vector `x[]` per `DECISION_VARS`: (sp_power, sp_gas_rate, frother_conc, flot_residence_time_min, air_fraction, slurry_fraction, target_flot_P80_um); `base: CuSulfideParams`; `tea_base: TEAParams` | Zip `NAMES[]` with `x[]`; mutate `base.flot_op` (OperatingParams), `base` flotation residence/P80/circulating_load; return mutated params | `(CuSulfideParams, TEAParams)` | Step 6 |
| 8 | `optimizer.DECISION_VARS` constant | (none) | Define 10-D decision space with bounds (name, lo_bound, hi_bound) | Tuple of 10 (name, float, float) tuples | Step 7 |

## Simulator (Throughput Chain)

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 9 | `throughput.simulate_cu_sulfide(feed_tph, feed_grade, params, topology)` | feed_tph (float), feed_grade (frac), `params: CuSulfideParams` (with mineralogy + all stage configs), `topology: Topology` | Route to heap-leach OR sulfide-flotation path; apply mineralogy via `_apply_mineralogy()` | `sim_dict: dict` with keys: `streams`, `power_kw`, `balance`, `topology`, `mineralogy`, unit diagnostics | Step 6 → Step 10 (TEA evaluation) |

### Comminution Path (Sulfide)

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 10 | `throughput._run_crushing(feed_tph, feed_grade, p)` | feed_tph, feed_grade, `p: CuSulfideParams` (crusher specs) | Construct raw ore Stream (F80=300mm, Cu grade); call `run_crusher()`; return ore + crushed Stream + crusher_kw | `out['ore']`, `out['crushed']` Stream; `power['crusher_kw']` | Step 11 (SAG) or Step 21 (heap-leach) |
| 11 | `flowsheet.run_crusher(feed, params, P80_um)` | `feed: Stream` (solids_tph, water_tph, F80_um, cu_grade); `CrusherParams`; `P80_um` target | Calc Bond total_power = kW; return new Stream with updated F80, same Cu grade | `(Stream, kW)` | Step 10 consumer |
| 12 | `throughput._run_screen_and_sag(crushed, p, topo, out, power)` | `crushed: Stream`; `p: CuSulfideParams` (SAG specs); `topo: Topology` (screen_enabled, ball_mill_enabled) | If screen_enabled: split by undersize fraction, size screen; SAG the oversize; Bond P80 inversion; return SAG discharge + cyclone fresh F80 target | `sag_out: Stream`, `bm_fresh_F80_um`, undersize_tph, oversize_tph, `screen_diag` dict | Step 13 |
| 13 | `flowsheet.run_sag(feed, params, Ecs_kwh_per_t)` | `feed: Stream` (F80_um); `SAGParams`; `Ecs_kwh_per_t` (Bond specific energy) | Invert Bond 3rd-theory: P80 = [10/(Ecs/Wi + 10/√F80)]²; calc total kW = Ecs × solids_tph | `(Stream with new F80, kW)` | Step 12 consumer |
| 14 | `throughput._run_grinding(crushed, sag_out, bm_fresh_F80_um, undersize, oversize, p, topo, out, power)` | SAG discharge + screen inputs; `p.target_flot_P80_um`; `topo.cyclone_enabled`, `topo.ball_mill_enabled`; `topo.regrind_enabled`, `topo.regrind_mill_type` | If cyclone_enabled: ball-mill closed-circuit via cyclone on cyclone inlet stream to hit target P80; else open-circuit BM; optionally regrind; handle circulating load (CL=2.5); apply liberation-ceiling penalty + froth-flooding penalty + sp_power detachment penalty | `flot_in: Stream` (F80=target_flot_P80), `cyc_diag` dict with d50_um, dp_kpa, solids_to_uf%, Rv, m, etc. | Step 15 |
| 15 | `flowsheet.run_hydrocyclone(feed, geom, pressure_kpa, coeffs, rho_p, rho_f, mu)` | `feed: Stream` (F80_um, solids_tph, water_tph); `CycloneGeom` (Dc, Di, Dx, Du, h); pressure_kpa; Plitt coefficients | Calc Q_lpm; cut-size d50 via Plitt; sharpness m; water recovery Rv; grade efficiency y @ F80; split feed into UF (coarse) + OF (fine) | `(Stream_UF, Stream_OF, diag_dict)` | Step 14 consumer |

### Flotation Path

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 16 | `throughput._run_rougher_flotation(flot_in, p, topo, out, power)` | `flot_in: Stream` (solids_tph, cu_grade, F80=target); `p.flot_op: OperatingParams` (sp_power, sp_gas_rate, frother_conc, air_fraction, slurry_fraction, contact_angle, num_cells, ret_time, cell_volume); `p.gangue_recovery`, `p.scavenger_recovery_on_tails`; `p` mineralogy → flot_R_max | Route via flotation kinetics; calc recovery R_rougher (capped by mineralogy flot_R_max + liberation penalty + froth-flooding penalty + sp_power detachment penalty); apply kinetic model (FittingParams); split conc/tails; size conc tank; apply scavenger on tails (R_scav=0.50 on tails → overall R = R + (1-R)×R_scav) | `conc: Stream` (cu_grade raised, solids by R), `tails: Stream` (solids by 1-R, lower cu_grade), `flot_diag` dict (cell count, volume, kinetics) | Step 17 (regrind/cleaner) or Step 18 (dewatering) |
| 17 | `throughput._run_regrind_and_cleaner(conc, tails, flot_in, p, topo, out, power)` | Rougher conc; `topo.regrind_enabled`, `topo.regrind_mill_type` ("ball"/"tower"/"isamill"), `topo.cleaner_enabled`, `topo.n_cleaner_stages` (1-3); `p.regrind_target_P80_um` (30 µm); `p.target_cleaner_conc_grade_pct` (26%) | If regrind_enabled: ball-mill on conc to 30 µm (energy_factor per mill type); if cleaner_enabled: N-stage cleaner flotation banks on regrind discharge; each stage @ different kinetics (lower Jg, higher bubble size, lower ret_time); final conc grade resolved to ~26% via gangue-recovery target; tails rejoin rougher tails for leach/neutralization | `final_conc: Stream` (higher cu_grade ~26%), `tails: Stream` (regrind tails merged), `regrind_diag` dict (cell count, kW by mill type), `cleaner_diag` dict | Step 18 |
| 18 | `throughput._run_dewatering(conc, p, topo, out)` | Regrind/cleaner conc (or rougher conc if no cleaner); `topo.thickener_enabled`, `topo.filter_enabled`; `p.conc_thickener: ThickenerParams`; `p.belt_filter: BeltFilterParams`; `p.filter_cake_moisture_pct` (9%); `p.conc_uf_pct` (65% solids target) | If thickener_enabled: size thickener by Galvez equation on settling rate; all solids to UF @ 65% target, excess water to OF. If filter_enabled: size belt filter area by vacuum-filtration Darcy law; calc filter-cake dry solids + moisture | `final_conc: Stream` (filter_cake, ~9% moisture), `th_diag` dict (diameter_m), `filt_diag` dict (area_m2) | Step 19 or Step 22 (concentrate-hydromet) |
| 19 | `flowsheet.run_thickener(feed, params, target_uf_pct)` | `feed: Stream`; `ThickenerParams` (vTF, rF, n for settling-rate model); target_uf_pct (65) | Calc area via settling = vTF × (1/(rF×solids_tph))^n; water to UF from target %; clamp water to available | `(Stream_UF, Stream_OF, diag_dict with area_m2, diameter_m)` | Step 18 consumer |

### Hydromet & EW Path

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 20 | `throughput._run_concentrate_hydromet(final_conc, ore, p, topo, out, power)` | `final_conc: Stream` (solids_tph, cu_grade ~26%); `ore: Stream` (for byproduct calc); `p: CuSulfideParams` with leach/SX/EW/neut params; `topo.leach_enabled`, `topo.sx_enabled`, `topo.ew_enabled`, `topo.neutralization_enabled` | If leach_enabled: autoclave POX on conc (LeachParams.max_extraction from mineralogy); calc residue; if SX/EW: closed-circuit SX-EW recycle (95%×95% pass eff; 3% electrolyte bleed → ~97% circuit recovery); EW cathode sizing; if neutralization: size neutralization tanks on acid load | `leach_diag`, `sx_diag`, `ew_diag`, `neut_diag` dicts; `cathode_tph`, `cu_to_sx_raffinate_tph`, `cu_to_ew_spent_tph` floats | Step 21 (mass balance) |
| 21 | `throughput._run_heap_leach_route(ore, crushed, p, topo, out, power)` | `crushed: Stream` (whole ore, primary crush only); `p: CuSulfideParams` with `heap_leach_params`; `topo.sx_enabled`, `topo.ew_enabled`, `topo.neutralization_enabled` | Size heap: calc cu_leached_tph via HeapLeachParams.extraction_fraction (from mineralogy heap_X); PLS cu_aq_gpl; if SX/EW: same closed-circuit recycle as concentrate path; if neutralization: size on acid + Fe/As loads | Same output dicts as sulfide path (all-zero for flotation/thickener/filter/regrind) | Step 21 (mass balance) |

### Mass Balance Assembly

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 22 | `throughput._build_mass_balance(ore, final_conc, tails, topo, leach_diag, cathode_tph, cu_raffinate, cu_ew_spent, power)` | `ore: Stream` (solids_tph, cu_grade); `final_conc: Stream` (or conc Stream if no dewater); `tails: Stream`; `cathode_tph`; power dict; `topo` for route ID | Cu in = ore.solids_tph × ore.cu_grade; Cu to conc = conc_tph × conc_grade; Cu to cathode (if leach/heap); Cu to raffinate/ew_spent losses; closure_err = Cu_in - (Cu_conc + Cu_cathode + Cu_tails + losses); overall_Cu_recovery = (Cu_conc + Cu_cathode) / Cu_in | `balance: dict` with cu_in_tph, cu_in_conc_tph, cu_in_cathode_tph, cu_in_tails_tph, cu_lost_*, cu_closure_err, overall_cu_recovery, conc_grade_pct, total_power_kw | Step 10 (simulator return) |

## Financial Evaluation (TEA)

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 23 | `tea.evaluate(sim_result, tea_params)` | `sim_result: dict` (from Step 9) with streams, power, balance, topology, byproducts; `tea_params: TEAParams` (prices, costs, payability, discounting) | Read topology to gate capex on stage booleans; calc revenue (conc sale net TC/RC + cathode + byproducts); calc opex (power, water, reagent, media, ore purchase); calc stage capex (Perry/Mular 2025 USD for crusher/mills/cyclone/flot/thickener/filter/tailings); DCF NPV @ 8% discount, 25 yr, 7884 h/yr | `res: dict` with npv, irr, capex lines, opex lines, revenue lines | Step 6 (objective) → Step 3 (TopologyResult) |
| 24 | `tea.TEAParams` dataclass | (intrinsic) | Fixed financial constants: cu_price_per_t, tc_per_t_conc, rc_per_t_cu, power_cost_per_kwh, water_cost_per_m3, reagent_cost_per_t_ore, media_cost_per_kwh_grind, discount_rate (8%), hours_per_year (7884), tailings_adverse_factor, smelter min/full payable grades, cathode payable fraction, deductible units, Mo/Au/Ag prices & payability, impurity penalty (As schedule) | Used in evaluate() for all financial calcs | Step 23 |

## Benchmark & Reporting

| Step | Module.Function | Input Packet | Operation | Output Packet | Next Consumer |
|------|---|---|---|---|---|
| 25 | `benchmark_scorecard.render()` | (none) | Call `run_m1_route_selection()` (test 10 disclosed plants; count correct route picks); call `run_m2_metallurgy()` (test recovery vs model); compute mean |ΔR| live from M2 harness | PNG figure with M1 score (8/10 pass), M2 median |ΔR| (≤5pp pass) | File output |
| 26 | `optimizer.print_leaderboard(opt_result)` | `OptimizationResult` with results[] TopologyResult list | Sort by NPV desc; print table: rank, topology label, NPV ($B), IRR, capex ($M), R_Cu%, conc_grade% | Console output (stdout) | User |

---

## Legend

| Term | Definition |
|------|-----------|
| `tph` | metric tonnes per hour |
| `frac` | mass fraction (unitless, 0–1) |
| `%` | percentage (0–100) |
| `um` / `µm` | microns (1e-6 m) |
| `kW` | kilowatts (power) |
| `kWh/t` | kilowatt-hours per tonne (specific energy) |
| `m3/h` | cubic meters per hour (volumetric flow) |
| `g/L` | grams per liter (aqueous concentration) |
| `pp` | percentage points (for error metrics) |
| `Stream` | dataclass: solids_tph, water_tph, F80_um, cu_grade (frac), cu_aq_gpl (for leach), cu_org_gpl (for SX), organic_tph |
| `Topology` | dataclass: boolean stage toggles + discrete mill type + cleaner stage count; gates the equipment on/off |
| `CuSulfideParams` | dataclass: all simulator knobs (SAG/BM specs, flotation operating params, thickener/filter/regrind/cleaner configs, leach/heap/SX/EW params, mineralogy class, byproduct grades) |
| `TEAParams` | dataclass: all financial constants (Cu price, power cost, capex bases, discount rate, payability schedules) |
| `OptimizationResult` | dataclass: feed conditions + list of TopologyResult (one per topology tested) |
| `TopologyResult` | dataclass: topology, label, best_x[] (optimal decision vector), best_npv, irr, capex, cu_recovery, conc_grade_pct |
| `NPV` | Net Present Value, 25-year pre-tax DCF @ 8% discount rate, USD |
| `IRR` | Internal Rate of Return (%) |
| `R` / `R_Cu` | Cu recovery (overall solids balance: (Cu_conc + Cu_cathode) / Cu_in) |
| `flot_R_max` | mineralogy-dependent flotation recovery ceiling (oxide 0.05, chalcocite 0.85, chalcopyrite 0.90) |
| `P80` | 80%-passing particle size (microns) |
| `TC/RC` | Treatment Charge / Refining Charge (smelter deductions on concentrate) |
| `capex` | capital expenditure (USD) |
| `opex` | operating expenditure (USD) |
| `POX` / `Albion` | Pressure oxidation autoclave (leach technology for refractory Cu sulfides) |
| `SX/EW` | Solvent extraction / Electrowinning (hydromet Cu recovery from solution) |

---

## Notes

- **Decision vector** (optimizer DOE): 10 continuous variables (sp_power, sp_gas_rate, frother_conc, flot_residence_time_min, air_fraction, slurry_fraction, target_flot_P80_um) × 3 discrete toggles (regrind_mill_type, n_cleaner_stages) nested in Topology enumeration.
- **Topology** determines routing: SULFIDE (flotation + optional regrind/cleaner + dewatering), CONCENTRATE-LEACH HYDROMET (flotation → POX → SX/EW), or WHOLE-ORE HEAP LEACH (primary crush only → SX/EW).
- **Closed-circuit SX-EW**: raffinate recycles to leach feed; EW spent electrolyte recycles to SX strip; steady-state recovery ~97% (vs. 90% single-pass) per Wills 8th ed. Ch. 15.
- **Mineralogy** (chalcopyrite / chalcocite / oxide / IOCG / bornite) gates flotation ceiling, POX max extraction, and heap-leach extraction via lookups in MINERALOGY_PROPS and BYPRODUCT_GRADES.
- **Liberation penalty** (target_flot_P80_um trade-off): coarse grind → lower recovery ceiling due to poor liberation (Trahar 1981). Penalty applied as fraction of flot_R_max above liberation_p80_ref_um.
- **Froth-flooding penalty**: high sp_gas_rate (>1.6 cm/s) → bubble coalescence → lower kinetic rate. Penalty applied above jg_ref.
- **sp_power detachment penalty**: high specific power (>1.0 kW/m³) → turbulent eddies → particle detachment. Penalty applied above sp_power_ref.
- **Capex sources**: Perry T9-50 (crusher, cyclone, thickener, filter), Mular & Poulter 2002 refined to 2024 vendor quotes (SAG/BM mills), Arfania et al. 2017 (flotation cells), O'Hara 1992 Eq 6.3.145 (tailings storage). All uplifted to 2025 USD (M&S=2383).
- **Payability schedule**: smelter rejects conc below min grade; ramps payable fraction linearly between min and full-payable grades; applies TC/RC deductions and As penalty.
- **TEA scope**: concentrator-only (run-of-mine in, Cu concentrate + cathode + byproducts out). No mine, labor, G&A; tailings storage included.
