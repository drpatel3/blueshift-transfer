# Process model capex rework — handoff for next session

**Status:** physics + topology + per-stage capex are in place; total capex is
roughly 10× too low vs. real Cu concentrators. Need to refit major-equipment
bases against current vendor-anchored data and add missing balance-of-plant
lines. NOT an inflation problem — M&S is correctly applied throughout.

---

## Where the project is right now

Repo: `C:\Users\Ryan Murray\Desktop\development\literature_platform\dev\process_model\`

Latest commit (push pending after this handoff): `b1c1d58` — route-choice
restructure (sulfide / hydromet / hybrid topology toggles).

**Working state:**
- 47/47 tests pass
- Topology has 12 booleans (`flotation_enabled`, `leach_enabled`, etc.); 9
  curated archetypes for the optimizer's default leaderboard sweep
- Sulfide route, hydromet route, and hybrid all simulate cleanly with mass
  balance closing
- `tea.py` is the single source of truth for all monetary constants (cost
  bases + DEFAULT_* opex unit rates); `TEAParams` defaults reference them

**Numbers at 2000 tph / 0.8% Cu (default sulfide):**
- NPV $2.18B, IRR 151%, total installed capex **$144M**
- Equipment subtotal $26.2M × Lang 5 + tailings $13M = $144M

**Numbers at 2000 tph / 0.8% Cu (full hybrid chain):**
- NPV $4.31B, IRR 93%, total installed capex $481M

---

## The problem the next session needs to solve

**Capex is ~10× too low.** Real Cu concentrators at 2000 tph (~50 ktpd) cost
$1.5–2.0B installed. We're at $144M.

### Benchmarks (concentrator only, no mine, no smelter)

| Plant | Capacity (ktpd) | Concentrator capex | Year | Linear-scaled to 50 ktpd |
|---|---:|---:|---|---:|
| Quellaveco (Anglo, Peru) | 127 | $3.5B | 2022 | $1.4B |
| Cobre Panamá (FQM) | 85 | $4.0B | 2019 | $2.4B |
| Las Bambas (MMG, Peru) | 140 | $5.0B | 2016 | $1.8B |
| Carrapateena (OZ) | 12 | $1.0B | 2019 | $4.2B (small-plant penalty) |

Realistic target for our 2000 tph default: **$1.5–2.0B installed**.

> **Scope note (2026-06-09):** the disclosed figures above are *full-project*
> totals — they bundle the greenfield tailings-dam construction, EPCM +
> contingency, and owners' infrastructure (camps, roads, power line) that this
> concentrator-only model deliberately omits. After Phase 2B/2C the model lands
> ~$0.8–0.9B at 2000 tph; the ~$0.5B residual to the band low end is precisely
> those three out-of-scope items (TSF dam $200–500M, EPCM 15–25%, owners' infra
> 10–20%). Compare the model on a concentrator-equipment + BOP basis, not
> against these full-project disclosures.

### Where the gap is — line-by-line

| Line | Current | Vendor / SME 2024 reality | Off by |
|---|---:|---:|---:|
| SAG mill 5.6 MW equipment | $5.3M | $30–50M | 6× |
| Ball mill 16.5 MW equipment | $11.3M | $80–130M | 8× |
| Flotation 15 cells × 150 m³ | $5.6M | $25–50M | 5× |
| Concentrate thickener (4 m dia) | $0.1M | OK for conc-side | OK |
| Belt filter | $1.1M | OK for vacuum belt | OK |
| Tailings thickener | **MISSING** | $5–15M (35–45 m dia) | missing |
| Conveyors / belt galleries | **MISSING** | $20–40M | missing |
| Slurry pumps / piping | **MISSING** | $30–60M | missing |
| Reagent storage / dosing | **MISSING** | $10–20M | missing |
| Water supply + reclaim | **MISSING** | $30–80M | missing |
| MCC / E&I house / control room | **MISSING** | $20–40M | missing |
| Tailings pipeline + dam construction | $13M (O'Hara 6.3.145 minimum) | $200–500M (greenfield) | huge |
| Lang factor | 5.0 (Perry chemical plant) | 6–7 (mining greenfield) | low |

---

## Root causes (in order of impact)

### 1. Mular & Poulter 2002 mill formula extrapolates badly to mega-mills

Current: `C_USD_2025 = 14,005 × kW^0.6915` (per cell, applied to SAG and BM
separately). Source: `tea.py:SAG_MILL_BASIS` and `BM_MILL_BASIS`, both at
`(4_920_000, 5_000, 0.6915)`.

The Mular fit was on 1990s-era 1–8 MW mills. At 16.5 MW (modern BM), the
exponent under-weights diseconomies of scale. Modern industry rule of thumb
is **$5,000–8,000/kW installed** for mega-mills; we're getting ~$3,200/kW
installed (Lang × 5).

**Fix candidates:**
- Refit on Sayadi et al. (2014) "A parametric cost model for mineral
  grinding mills," *Minerals Engineering* 55:96–102 — paywalled; Elsevier.
- Recent vendor-quote anchored basis (2024 USD): $4M @ 1000 kW with exp 0.85
  for ball mills, similar for SAG. Higher exponent captures the scale
  diseconomy.
- Or use Doll, A.G. (2013), "A simple estimation method of SAG mill power
  draw" — has a $/kW basis.
- Or pay for CostMine 2024 Equipment Cost Calculator subscription
  (~$2k/year) for ground-truth values.

### 2. Major missing capex lines

The per-stage Perry/Mular/Arfania approach captured the *equipment* lines but
left out everything between them. A real plant has tailings thickener,
conveyors, slurry pumps, reagent dosing, water systems, MCC, control house,
buildings — roughly **30–50% of total installed capex** at greenfield Cu
plants.

**Fix:** add a `BALANCE_OF_PLANT_BASIS` (or similar), sized as $/tpd of ore
feed. Industry rule of thumb (Mular handbook 2002, scaled to 2024 USD):
- Brownfield: $15–25k/tpd
- Greenfield: $30–50k/tpd

For 50 ktpd that's $750M–$2.5B at greenfield. That single line plausibly
fills most of the gap.

### 3. Lang factor of 5.0 is light for greenfield mining

Source: `tea.py:DEFAULT_LANG_FACTOR = 5.0`. Mining-industry references
(Mular handbook ch. 1, Sayadi et al. 2014):
- Brownfield concentrator (existing infrastructure): 3–4
- Greenfield concentrator (remote): 5–7
- Greenfield with tailings dam construction: 6–8

Our default is OK for hydromet, low for greenfield sulfide. Consider
splitting `lang_factor_sulfide = 6.0` and `lang_factor_hydromet = 5.0`.

### 4. Flotation Arfania 2017 fit ceiling is 158 m³

Source: `tea.py:FLOTATION_CELL_BASIS = (310_841, 100, 0.452)` from Arfania,
Sayadi & Khalesi (2017) — fit on 10 mechanical cells, range 0.28–158.6 m³.
Modern Cu plants use 300–660 m³ TankCell e300/e500/e660. Our optimizer caps
cell volume at 158 m³ to stay in the fit, which forces more (smaller) cells.

For a 2000 tph plant the bank ends up 15 cells × 150 m³ = 2,250 m³ total.
Real Cu plants run **6 cells × 500 m³ = 3,000 m³**. Vendor list price for
Outotec TankCell e500: **~$2–4M per cell installed** — so 6 × $3M = $18M for
the bank, vs. our $5.6M equipment / $28M installed. Close to right
order-of-magnitude after Lang, but Arfania's small-cell fit is genuinely
hitting its ceiling.

**Fix:** add a higher cell-volume basis from a 2024 vendor curve, OR
explicitly scale the existing per-cell cost up by a factor (1.4×) for
mega-cells beyond Arfania's fit range, with a citation note.

---

## Inflation IS already applied (not the root cause)

`tea.py` M&S anchors:
- `M_S_INDEX_PERRYS = 1000` (Perry's symbolic basis)
- `M_S_INDEX_2025 = 2383`
- `M_S_INDEX_1988 = 852` (O'Hara grinding)
- `M_S_INDEX_2002 = 1104` (Mular & Poulter)
- `M_S_INDEX_1991 = 931` (Parkinson-Mular thickener)
- `M_S_INDEX_2013 = 1561` (Arfania flotation)

Each basis tuple already includes the M&S uplift to 2025 USD. M&S captures
general industrial inflation correctly.

**However**, mining-specific cost escalation has run ~20–30% ahead of M&S in
2020–2025 due to steel prices, mining-country labor, vendor consolidation.
CostMine's Mining Cost Service Index would catch this; it's paywalled.

Adding a `MINING_PREMIUM_FACTOR = 1.25` would tighten this — but it's a
~25% effect, not the 10× we need. **Inflation is not the dominant problem.**

---

## Recommended plan for next session

### Phase 2A — Refit major-equipment bases (biggest single fix) [COMPLETE 2026-04-28]

**Done:**
- `SAG_MILL_BASIS` and `BM_MILL_BASIS` refit to vendor-anchored
  `(50_000_000, 16_500, 0.85)` (2024 USD).
- `FLOTATION_CELL_BASIS_LARGE = (1_500_000, 500, 0.50)` added; `select_flot_basis()`
  routes cells > 158 m³ to the large-cell tier.
- Tails thickener line added: sized via `tails_thickener_diameter_m()`
  (0.30 t/m²/h unit area, Wills 7th ed.) and priced via the existing
  Parkinson-Mular `THICKENER_BASIS`.
- Optimizer `flot_cell_volume_m3` upper bound lifted 158 → 500 m³.

**Result at 2000 tph / 0.8% Cu (default sulfide):** total capex $144M → $498M,
NPV $2.18B → $1.69B, IRR 151% → 41%.
Route-selecting grade sweep figures: `process_model/figures/grade_sweep_phase2a/`.



1. **SAG/BM mills**: replace `SAG_MILL_BASIS` and `BM_MILL_BASIS` with a
   higher-exponent fit anchored to 2024 vendor quotes:
   - Try `(50_000_000.0, 16_500.0, 0.85)` for 16.5 MW basis BM, exp 0.85.
   - Validate at 5 MW: $50M × (5000/16500)^0.85 = $50M × 0.36 = $18M (about
     right for a 5 MW SAG).
   - Validate at 16.5 MW: $50M (about right for a modern BM).
   - Cite to Mular Handbook 2002 + 2024 vendor data inflation.

2. **Flotation**: extend Arfania to large cells via a second-tier basis:
   - `FLOTATION_CELL_BASIS_LARGE = (1_500_000, 500, 0.50)` — sized for
     500 m³ TankCells, $1.5M per cell. Source: vendor list price.
   - Use small basis for cells ≤ 158 m³, large basis above.
   - Or replace entirely with a higher exponent fit.

3. **Add tailings thickener**: sized on tails tph (1968 t/h dry ≈ 35–40 m
   dia). Already have `THICKENER_BASIS` for the Parkinson-Mular form; just
   need to call it on the tails stream as well as the concentrate stream.

### Phase 2B — Add balance-of-plant capex line [COMPLETE 2026-06-09]

**Done (bottom-up):** `DEFAULT_BALANCE_OF_PLANT_PER_TPD = 3_960.0` USD/tpd of
ore feed — sum of the bottom-up line items below ($190M / 48,000 tpd): inter-area
conveyors $40M, tailings+reclaim slurry mains & pumps $50M, reagent storage/mains
$20M, raw+reclaim water systems $50M, main MCC/E&I/control $30M. Already-installed
basis (bypasses Lang); scoped to plant-wide mains distinct from the equipment-local
piping the Lang factor already covers (no double-count); applied to flotation
routes only. Wired into `evaluate()` as `balance_of_plant_capex` and exposed in the
breakdown. Default sulfide installed capex $513M → $803M.

Single line, $/tpd of ore feed, sized to fill the gap between
explicit-equipment subtotal and a defensible total. Two sub-options:

- **Bottom-up**: line items for tailings dam ($150M), conveyors ($40M),
  slurry pumps ($50M), reagent ($20M), water systems ($50M), MCC ($30M).
  Sum: $340M for 50 ktpd → ~$7k/tpd of equipment, × 1.0 (already installed).
- **Top-down**: just `BALANCE_OF_PLANT_PER_TPD = 25_000` USD/tpd of ore at
  greenfield, citing Mular ch. 1.

Recommend bottom-up for transparency.

### Phase 2C — Adjust Lang factor by route [COMPLETE 2026-06-09]

**Done:** `DEFAULT_LANG_FACTOR_SULFIDE = 6.0` (greenfield flotation concentrator,
midpoint of the cited Mular ch.1 / Sayadi 2014 5–7 band) and
`DEFAULT_LANG_FACTOR_HYDROMET = 5.0` (heap/SX-EW, Perry default unchanged), added
to `TEAParams` and selected in `evaluate()` on `topo.flotation_enabled`. The
legacy single `lang_factor` field is retained for back-compat. Heap economics
(and the El Abra POX miss) are unperturbed since heap stays at 5.0.

- `lang_factor_sulfide = 6.0` (greenfield concentrator)
- `lang_factor_hydromet = 5.0` (existing default)

Choose the right one based on `topo.flotation_enabled`.

### Phase 2D — Optional mining-specific premium

`MINING_PREMIUM_FACTOR = 1.25` applied to all 2025-USD bases as a
post-inflation step. Document explicitly as a flag-only knob with citation
to CostMine Mining Cost Service Index trends 2020–2025. NOT in the tested
default until verified.

### Verification after Phase 2A–2C

Total capex at 2000 tph default sulfide chain should land **$1.0–1.8B**
(was $144M). IRR drops from 151% toward defensible (likely 20–40%).

Hydromet route should be **$1.5–2.5B** for 50 ktpd (was $754M). IRR drops.

---

## Files / lines that need to change

- `process_model/tea.py`:
  - Lines ~126–135 (`SAG_MILL_BASIS`, `BM_MILL_BASIS`) — refit
  - Lines ~138–157 (`FLOTATION_CELL_BASIS`) — extend or add large-cell tier
  - Add new `BALANCE_OF_PLANT_PER_TPD` constant
  - Update `evaluate()` to add tailings thickener and BOP lines
  - Update `DEFAULT_LANG_FACTOR` or add per-route variants
- `process_model/test_tea.py`:
  - Add: at default 2000 tph, total_capex > $1B (regression guard)
  - Update fixture if mill capex assertions present
- `process_model/PROGRESS.md`:
  - Document Phase 2 entry with citations

---

## What to bring up first in the next session

1. Read this file: `process_model/CAPEX_REWORK_PLAN.md`
2. Read `process_model/tea.py` to confirm the current bases
3. Confirm the user wants the Phase 2A → 2D plan (or wants to scope down)
4. Decide on mill-basis source: refit-from-vendor vs. wait-for-CostMine vs.
   Sayadi 2014 paper paywall
5. Build Phase 2A first, commit, run optimizer leaderboard, sanity-check
   numbers against benchmark plants
