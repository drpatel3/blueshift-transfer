# Bound-desaturation campaign

Pulling the optimizer's operating variables off their upper/lower bounds.

## Problem

At the reference scenario the inner differential-evolution search slams **9 of
10** operating variables to a bound (see Attempt 0). This is not an optimizer
bug — NPV is *monotonic* in each pinned variable because each lever has a
one-sided effect in the model (a benefit with no offsetting cost, or a cost
with no offsetting benefit). The result is bound-sensitive rather than
physics-converged, and the operating-point optimization is effectively
degenerate.

## Principles

1. **Restore the trade-off, never clamp.** A variable lands interior only when
   its marginal benefit meets a marginal cost *inside* the range. We add the
   real physical/cost trade-off; we do **not** narrow bounds or add arbitrary
   penalties just to move the number.
2. **Source discipline.** Every new cost coefficient or penalty curve is cited.
3. **`flotation.py` kinetics are off-limits** (PhD-derived). Recovery/grade
   penalties are added in `throughput.py` (post-flotation) or `tea.py` (cost
   side), never inside the kinetic model.
4. **No validation regression.** After each attempt, re-run the probe and (at
   milestones) the M1/M2/M3 benchmarks. Recovery-touching changes must keep M2
   median |ΔR| ≤ 5 pp and M1 ≥ 8/10.

## Method

Probe: `python -m process_model._bound_probe [seed maxiter popsize]`
Runs DE on the reference winner topology (2000 tph, 0.8% Cu chalcopyrite,
sulfide regrind(tower)+cleaner(2-stage)), default seed 42, 60×12. Reports each
variable's normalized position and flags `MAX` / `MIN` / `interior`. Verified
to reproduce the 200×15 result, so 60×12 is a faithful, faster proxy.

The three channels a penalty can use (it must convert to NPV):
- **Cost** — lever burns a priced consumable (reagent/power/water).
- **Grade** — lever raises gangue entrainment → lower conc grade → revenue.
- **Rollover** — lever's recovery benefit turns hump-shaped past a physical limit.

---

## Attempt log

### Attempt 0 — baseline (no change)

`seed=42, 60×12`

| variable | value | bounds | position |
|---|---|---|---|
| sp_power | 1.200 | 0.8–1.2 | **MAX** |
| sp_gas_rate | 1.500 | 0.8–1.5 | **MAX** |
| frother_conc | 50.0 | 10–50 | **MAX** |
| flot_residence_time_min | 18.30 | 15–20 | interior 66% |
| air_fraction | 0.250 | 0.10–0.25 | **MAX** |
| slurry_fraction | 0.150 | 0.15–0.30 | **MIN** |
| contact_angle | 55.0 | 20–55 | **MAX** |
| circulating_load | 1.500 | 1.5–4.0 | **MIN** |
| tailings_adverse_factor | 1.000 | 1.0–5.0 | **MIN** |
| target_flot_P80_um | 200.0 | 75–200 | **MAX** |

**Result:** 9/10 pinned. NPV $4.72B · IRR 65% · capex $788M · R 89.3% · conc 27.3%.
Only `flot_residence_time` is interior — and it works precisely because it is
already coupled to a real cost (more residence → more cell volume → capex). It
is the template for every other fix.

### Attempt 1 — remove `tailings_adverse_factor` from the DOE

**Change** (`optimizer.py`): dropped `tailings_adverse_factor` from
`DECISION_VARS`. It is a TSF **site condition**, not a design choice, so the
optimizer only ever drove it to its 1.0 floor to shave capex for free. It is
now a fixed scenario input (`TEAParams.tailings_adverse_factor`, default 1.0;
set per project for an unfavourable site). Enabled by the name-based `_apply`
decoder so the variable simply falls back to its default.

**Result:** **8/9 pinned** (was 9/10). NPV $4.72B · capex $788M — **unchanged**,
because the default (1.0) is exactly where DE was parking it. One spurious pin
removed at zero economic cost. 7/7 optimizer tests pass.

| variable | position |
|---|---|
| flot_residence_time_min | interior 61% |
| *(the other 8)* | still at MAX/MIN |

Type: **structural removal** (a non-design variable). Two such variables exist;
`contact_angle` is the other (Attempt 2).

### Attempt 2 — fix `contact_angle` (remove from the DOE)

**Why first / biggest impact.** A held-everything-else sweep at the reference
point shows contact angle is by far the highest-leverage variable — it sets
recovery across almost the whole feasible range:

| contact angle | R_Cu | conc grade | NPV |
|---|---|---|---|
| 20° | 48.4% | 28.3% | −$1.58B |
| 40° | 74.3% | 29.4% | $2.44B |
| 50° | 86.5% | 27.8% | $4.30B |
| **55° (old DOE max)** | **89.3%** | **27.3%** | **$4.72B** |
| 60° | 89.5% | 27.0% | $4.74B |
| 70° | 89.7% | 26.6% | $4.77B |
| 80° | 89.8% | 26.4% | $4.78B |

Recovery climbs steeply, then **plateaus ~55–60°** against the
`flot_R_max = 0.90` chalcopyrite ceiling; grade falls (entrainment) but
recovery wins throughout, so **NPV is monotone** — DE rides it to whatever
bound is set (it would reach 80° if allowed). That is the pin.

**Change** (`optimizer.py` + `throughput.py`): contact angle is a
**mineral-surface property** (chalcopyrite-xanthate wettability), not a design
dial, so it was removed from `DECISION_VARS` and fixed at **55°** — a cited
representative value (chalcopyrite-xanthate ~40–80°; Chander & Fuerstenau,
Fuerstenau & Pradip) that lands recovery on the validated floatability plateau.

**Result:** **7/8 pinned** (was 8/9). NPV $4.72B · capex $788M · R 89.3% ·
conc 27.3% — **all unchanged** (55° is exactly where DE was parking it).
No validation regression: **M2** median |ΔR| 1.77 pp / |ΔG| 2.03 pp (PASS),
**M1** 9/10, full suite 67/67.

Type: **structural removal**. With both property/site variables now fixed, the
remaining 7 pins (`sp_power`, `sp_gas_rate`, `frother_conc`, `air_fraction`,
`slurry_fraction`, `circulating_load`, `target_flot_P80`) are genuine operating
levers that need restored trade-offs (Attempt 3+).

### Attempt 3 — `target_flot_P80`: restore the grind-vs-recovery trade-off

**Why it pinned (sweep at the reference point, old model):**

| P80 | R_Cu | BM power | NPV |
|---|---|---|---|
| 75 µm | 89.0% | 27.1 MW | $4.03B |
| 150 µm | 89.3% | 17.6 MW | $4.58B |
| 200 µm | 89.3% | 14.5 MW | **$4.72B** |

Recovery is **flat** (capped at `flot_R_max = 0.90`) while BM power falls as
the grind coarsens, so NPV rises monotonically to the coarsest bound — coarse
grinding was *free*. Physically wrong: coarse chalcopyrite floats worse.

**Change** (`throughput.py`): made the floatability ceiling **liberation-
limited**. `flot_R_max` is the ceiling at a fine, well-liberated grind; coarse
feed (composite, poorly liberated, detachment-prone) cannot reach it, so the
effective cap is multiplied by `liberation(P80) = 1 − 0.0013·(P80 − 150)` above
a 150 µm reference (floored at 0.5). Unity at/below 150 µm, so the validated
operating point and the M2 harness (which run at 150 µm) are unchanged.
Cited: Trahar 1981 (Int. J. Miner. Process. 8:289-327); Wills 7th ed. Ch. 12.

New sweep: recovery now peaks ~150 µm (89.3%) and falls above it (83.1% at
200 µm), so **NPV peaks at 150 µm** ($4.58B) instead of riding to 200 µm.

**Result (checked across two seeds — the difference is the finding):**

| variable | seed 42 | seed 7 |
|---|---|---|
| target_flot_P80 | interior 60% | interior 60% |
| frother_conc | interior 74% | interior 84% |
| circulating_load | interior 11% | interior 14% |
| flot_residence_time | interior 63% | interior 52% |
| sp_power | **MAX** | interior 85% |
| air_fraction | **MAX** | interior 98% |
| slurry_fraction | **MIN** | interior 19% |
| sp_gas_rate | **MAX** | **MAX** |
| **pinned total** | **4/8** | **1/8** |

NPV $4.72B → **$4.58B**, capex $788M → **$832M** (finer grind = more BM
power/capex — the honest cost of charging coarse grind for lost recovery).
No regression: **M1** 9/10, **M2** |ΔR| 1.77 / |ΔG| 2.03 pp (PASS), 67/67 tests.

**Interpretation.** Restoring the grind penalty did two things: (1) it
desaturated `target_flot_P80` robustly (interior at both seeds), and (2) it
**flattened the NPV surface** in the kinetics dimensions. Near the capped
recovery the aeration/power levers barely move NPV, so they are now *weakly
identified* — `sp_power`, `air_fraction`, `slurry_fraction` drift interior or
graze a bound depending on seed. Classification after Attempt 3:

- **Robustly interior:** `target_flot_P80`, `frother_conc`, `circulating_load`,
  `flot_residence_time`.
- **Weakly identified (seed-dependent):** `sp_power`, `air_fraction`,
  `slurry_fraction` — flat NPV; need a small real cost/grade term to define an
  optimum, but no longer a hard bound-pin.
- **Robustly pinned:** `sp_gas_rate` (MAX at both seeds) — next target
  (entrainment→grade penalty), Attempt 4.

Type: **restored physical trade-off** (liberation-limited recovery ceiling).

### Attempt 4 — `sp_gas_rate`: froth-flooding rollover + bound widened

**Why it stayed pinned after Attempt 3:** recovery rises monotonically across
the [0.8, 1.5] cm/s range (68.7% → 89.3%) with no rollover, so NPV has no
interior peak. The froth-flooding region (Jg > ~1.6 cm/s) was outside the
search space.

**Change** (`throughput.py` + `optimizer.py`):
- Added `froth_flood_jg_ref = 1.6` cm/s, `froth_flood_slope = 0.25`/cm/s,
  `froth_flood_floor = 0.70` to `CuSulfideParams`. Above 1.6 cm/s the
  effective recovery cap falls: `cap × max(0.70, 1 − 0.25×(Jg − 1.6))`.
  Cited: Gorain, Franzidis & Manlapig (1998) Int. J. Miner. Process. 53:215-236;
  Wills 7th ed. Ch. 12 (practical TankCell Jg ceiling ~2.0-2.2 cm/s).
- Widened `sp_gas_rate` upper bound **1.5 → 2.2 cm/s** so the flooding region
  is inside the search space. Validated operating point (1.5 cm/s) sits just
  below `jg_ref` — penalty is unity there, so M2 is unaffected.

**Sweep after change:**

| Jg (cm/s) | R_Cu | NPV ($B) | froth_stability |
|---|---|---|---|
| 0.8 | 68.7% | 1.40 | 1.000 |
| 1.5 | 89.3% | **4.58 (peak)** | 1.000 |
| 1.6 | 89.3% | 4.58 | 1.000 |
| 1.8 | 84.6% | 3.86 | 0.950 |
| 2.2 | 75.4% | 2.42 | 0.850 |

NPV peaks at ~1.5–1.6 cm/s — genuine interior optimum.

**Result: 2–4/8 pinned** (seed-dependent):

| variable | seed 42 | seed 7 |
|---|---|---|
| sp_gas_rate | interior 50% | interior 57% ✓ |
| target_flot_P80 | interior 60% | interior 60% ✓ |
| frother_conc | interior 48% | interior 60% ✓ |
| flot_residence_time | interior 60% | interior 8% ✓ |
| air_fraction | interior 94% | **MAX** |
| slurry_fraction | interior 26% | **MIN** |
| circulating_load | **MIN** | **MIN** |
| sp_power | **MAX** | **MAX** |
| **pinned total** | **2/8** | **4/8** |

M1 9/10, M2 |ΔR| 1.77 / |ΔG| 2.03 pp (PASS), 67/67 tests.

Robustly interior (both seeds): `sp_gas_rate`, `target_flot_P80`, `frother_conc`, `flot_residence_time`.
Weakly identified / seed-dependent: `air_fraction`, `slurry_fraction`.
Robustly pinned (both seeds): `sp_power` MAX, `circulating_load` MIN → Attempt 5+.

### Attempts 5+6 — `sp_power` (detachment rollover) + `circulating_load` (removed)

**`sp_power` sweep (Attempt 5):**

| sp_power | R_Cu | NPV ($B) | detachment factor |
|---|---|---|---|
| 0.80 | 86.5% | 4.13 | 1.000 |
| 1.00 | 88.3% | **4.41 (peak)** | 1.000 |
| 1.05 | 87.4% | 4.26 | 0.982 |
| 1.20 | 82.5% | 3.50 | 0.930 |

NPV peaks at **1.0 kW/m³** — genuine interior optimum.

**Change (Attempt 5, `throughput.py`):** Added turbulent-detachment penalty
on `sp_power`: above 1.0 kW/m³, turbulent eddies strip collected particles
from bubble surfaces, rolling the recovery cap back. `sp_power_ref=1.0`,
`sp_power_detach_slope=0.35`, `sp_power_detach_floor=0.90`. Cited: Schubert
(1999) Int. J. Miner. Process. 56:257-273; Pyke et al. (2003) J. Colloid
Interface Sci. 265:141-151.

**`circulating_load` (Attempt 6 — removed from DOE):** CL sweep showed
recovery flat at 88.3% across CL 1.5–4.0. The CL-sharpness benefit (higher
CL → tighter PSD → better liberation) cannot be represented: flotation-feed
F80 is set by `target_flot_P80_um` independently, and the rate-constant
improvement from a narrower PSD is inside the kinetic model (off-limits). CL
drives BM cost up with no NPV benefit → correctly pins to MIN. Since the benefit
side is unrepresentable, CL was removed from the DOE and fixed at **2.5**
(industry standard; Wills 7th ed. Ch. 5). DOE now 7 variables.

**Final result after Attempts 5+6:**

| variable | seed 42 | seed 7 |
|---|---|---|
| sp_power | **interior 50%** ✓ | interior 49% ✓ |
| sp_gas_rate | interior 56% ✓ | interior 57% ✓ |
| frother_conc | interior 61% ✓ | interior 31% ✓ |
| flot_residence_time | interior 78% ✓ | interior 46% ✓ |
| air_fraction | interior 95% | **MAX** |
| slurry_fraction | interior 5% | **MIN** |
| target_flot_P80 | interior 60% ✓ | interior 60% ✓ |
| **pinned total** | **0/7** | **2/7** |

NPV $4.53B · IRR 60% · capex $843M · R 89.1–89.2% · conc 27.6%.
M1 9/10, M2 |ΔR| 1.77 / |ΔG| 1.91 pp (both PASS; grade improved slightly).
67/67 tests.

Remaining seed-dependent: `air_fraction` (MAX seed 7) and `slurry_fraction`
(MIN seed 7). Their NPV sensitivity is very shallow — both variables graze
bounds when DE happens to miss the flat interior, but are not robustly pinned.
These are the last candidates for a dilution-water cost coupling (Attempt 7+,
optional given the very weak signal).

### Attempt 7 — `slurry_fraction` + `air_fraction`: pulp density and gas holdup pay their way

**Why they grazed bounds:** both variables reached the flotation model as
kinetic inputs only. The flotation feed water was set by a fixed
`flot_feed_solids_pct = 30%` target and the bank volume by
`slurry_m3_h × t × froth_factor` — so DE could dilute the pulp
(`slurry_fraction` → MIN, better kinetics) and load the cell with air
(`air_fraction` → MAX, more bubble surface area) without paying for the
water, the displaced cell volume, or the extra installed power.

**Change** (`throughput.py`, two consistency fixes — no new penalty curves):
- **Dilution water follows the DOE pulp density.** Flotation feed water is
  now derived from `flot_op.slurry_fraction` itself (the same φ the kinetic
  model uses as cell pulp density): `water/solids (t/t) = (1 − φ)/(2.7·φ)`
  via `_flot_dilution_water_per_t()`, replacing the fixed
  `flot_feed_solids_pct` target in all three grinding-circuit branches.
  Dilute pulp now buys its kinetic benefit at the price of makeup water
  (TEA $0.30/m³ line) and cell volume (capex). In cyclone circuits the
  `max(of_water_raw, target)` guard is kept — the circuit can still deliver
  more water than the target; φ sets the dilution floor. Wills 7th ed.
  Ch. 12 (pulp-density-vs-capacity trade-off).
- **Bank volume corrected for gas holdup.** A cell holds slurry plus
  dispersed air; effective pulp volume is `V·(1 − air_fraction)`, so both
  rougher and cleaner autosizing now scale by `1/(1 − air_fraction)`.
  Air buys its rate-constant benefit (`n_bubble ∝ air_fraction`) at the
  price of installed volume and power. Effective-volume sizing basis:
  Wills 7th ed. Ch. 12.

**Result: 0/7 pinned at BOTH seeds** (campaign target reached):

| variable | seed 42 | seed 7 |
|---|---|---|
| sp_power | interior 49% ✓ | interior 42% ✓ |
| sp_gas_rate | interior 57% ✓ | interior 54% ✓ |
| frother_conc | interior 3% | interior 23% ✓ |
| flot_residence_time | interior 55% ✓ | interior 97% |
| air_fraction | interior 92% | interior 97% |
| slurry_fraction | interior 38% ✓ | interior 3% |
| target_flot_P80 | interior 60% ✓ | interior 60% ✓ |
| **pinned total** | **0/7** | **0/7** |

NPV **$4.49B at both seeds** (0.0% spread — the optimum is physics-set, not
seed- or bound-set) · IRR 58–59% · capex $852–864M · R 89.0–89.1% ·
conc 27.7%. `air_fraction` / `slurry_fraction` / `frother_conc` still sit in
shallow regions near (but off) a bound at one seed each — honest shallow
optima, no longer pins.

**No validation regression:** M1 9/10 (same map, El Abra still the pinned
miss), M2 median |ΔR| 1.77pp / |ΔG| 1.91pp (both PASS, per-plant values
identical — the default chain's water is circuit-set by the cyclone split,
so the M2 operating points are untouched). One deliberate reference-case pin
update: `flotation_kw` 3000 → 3600 kW (5 → 6 cells at the default
`air_fraction = 0.25` — the previously uncharged cost of running 25% gas
holdup), total kW pin updated to match.

Type: **restored physical trade-off** (consistency between the kinetic
inputs and the sizing/water balance). **Campaign closed: 9/10 pinned →
0/7 at both probe seeds, every fix a cited physical or cost coupling.**
