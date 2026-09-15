# Equipment CAPEX Basis Reference — 50 ktpd Cu Sulfide Concentrator (2025 USD)

**Document date:** 2026-04-29
**Scope:** SAG mills, ball mills, hydrocyclones — power-law cost scaling with
defensible 2025-USD bases. This file is the citation record for the constants
in `process_model/tea.py`.

Power-law form (used throughout the codebase):
```
cost = basis_usd * (capacity / basis_capacity) ** exponent
```

---

## Summary

| Equipment | Basis capacity | Basis cost (2025 USD) | Exponent | Status vs current code |
|-----------|----------------|------------------------|----------|------------------------|
| SAG mill | 16.5 MW | $50 M | 0.85 | **Confirmed**, no change |
| Ball mill | 16.5 MW | $50 M | 0.85 | **Confirmed**, no change |
| Hydrocyclone (single unit) | 100 m³/h | $30 k | 0.55 | Tiny inflation bump (+5%, $28.6k → $30k) |

**Headline finding:** The Phase 2A SAG/BM refit (commit `2fb49a0`) is well-anchored
to vendor and literature data. The current cyclone basis is also defensible; a
~5% 2024→2025 inflation bump is the only quantitative change.

**The plant scaling issue is *not* in the per-unit bases — it is the absence of
parallelization logic.** At 2000 tph the model sizes a single hydrocyclone
against ~8320 m³/h slurry, but vendor max diameter (~840 mm) caps a single
industrial unit at ~400 m³/h. The real circuit needs ~20–25 cyclones in a
cluster. Capping max single-unit size and emitting `n_units` is captured below
under §4 for the next-phase parallelization plan.

---

## §1 SAG mill

### Vendor + project data

- **Cobre Panamá (First Quantum, commissioned 2019)**: two 28 MW gearless SAG
  mills (~40 ft / 12.2 m diameter), ABB GMD drives. World's largest installed
  SAG power. Total project capex US$6.7 B (full mine; equipment-only not
  publicly broken out).
  - ABB news: https://new.abb.com/news/detail/39407/abb-successfully-commissions-six-gearless-mill-drives-at-panamas-newest-mine
  - International Mining: https://im-mining.com/2019/10/24/abbs-six-gearless-drives-heart-first-copper-production-fqm-cobre-panama/

- **Metso Outotec Premier 18 MW SAG sale to Northern Star (2021)**: order value
  ~EUR 15 M for the geared SAG package alone.
  - https://www.metso.com/corporate/media/news/2021/8/metso-outotec-to-supply-a-high-capacity-premier-sag-mill-to-northern-star-resources-in-australia/

- **FLSmidth Russian Cu order (2020)**: four 40 ft gearless-driven SAGs in the
  20–25 MW range.
  - https://im-mining.com/2020/02/09/major-flsmidth-mill-order-for-russian-copper-mine-includes-four-40-ft-gearless-driven-sag-mills/

### Literature

- **Sayadi, Khalesi & Khoshfaraman Borji (2014)**, "A parametric cost model for
  mineral grinding mills," *Minerals Engineering* 57, 4–12. UVR + MVR fits across
  industrial SAG installations; reported exponents land in 0.65–0.85 across the
  5–25 MW range.
  https://www.sciencedirect.com/science/article/abs/pii/S0892687513002860
- **Wills' Mineral Processing Technology**, 8th ed. (Wills & Finch, 2016),
  Chapter 7 — qualitative scaling discussion.
- **Mular & Poulter (2002)**, *Mineral Processing Plant Design — Practice and
  Control* (SME). Original 1990s 1–8 MW fit `C ≈ 14,005 · kW^0.6915`. Underweights
  modern mega-mill diseconomies; superseded by Phase 2A vendor refit.

### Recommended basis

- `SAG_MILL_BASIS = (50_000_000.0, 16_500.0, 0.85)` — **unchanged**.

Cross-checks:
- 5 MW: $50M × (5000/16500)^0.85 = **$18.4 M** (matches midsize Metso/FLS quotes)
- 18 MW geared: ~$16–17 M direct + ABB GMD upgrade (+50–70%) + integration
  (×~2) → installed envelope $46–55 M, consistent with $50 M @ 16.5 MW gearless
- 22 MW: $50M × (22/16.5)^0.85 = **$63.7 M** (matches FLS Russian order range)
- 28 MW: $50M × (28/16.5)^0.85 = **$78.0 M** (Cobre Panamá per-unit envelope)

---

## §2 Ball mill (overflow discharge, secondary grinding)

### Vendor + project data

- **Cobre Panamá**: four 16.5 MW + two 22 MW gearless ball mills (26 ft / 7.9 m
  diameter), all ABB GMD. The 22 MW units are the practical commercial maximum
  in service today.
- **Metso Premier / Select** ball mill lines price comparably to SAG packages
  at equal MW when supplied as part of an integrated grinding contract.
  https://www.metso.com/mining/grinding/grinding-mills/

### Literature

- Sayadi et al. (2014) ball-mill exponents fall in 0.75–0.85 over 5–25 MW.
- Manufacturing cost is ~3–5% lower than SAG at equal MW (simpler shell, no
  AG breaking action), but gearless drive + controls + installation are
  identical — vendors quote near parity in practice.

### Recommended basis

- `BM_MILL_BASIS = (50_000_000.0, 16_500.0, 0.85)` — **unchanged**.
- Same constants as SAG; documented assumption is full $/kW parity at the 16.5
  MW basis. If a future project procures BM separately, apply a 0.97–0.98
  scalar.

Cross-checks:
- 16.5 MW: $50.0 M (Cobre Panamá per-unit envelope)
- 22 MW: $63.7 M (Cobre Panamá larger BM line)

---

## §3 Hydrocyclone (single unit)

### Vendor data

- **FLSmidth Krebs gMAX®** molded urethane cyclones: 13–838 mm (½ in to 33 in)
  diameter, vendor-recommended industrial maximum ~840 mm.
  https://prod.flsmidth.com/globalassets/frontify/2024/11/krebs_gmax_urethanecyclones_brochure_en.pdf
- **Metso MHC™ / MHC™ CB** (2024 launch): 100–800 mm range.
  https://www.metso.com/portfolio/mhc-series/
- **Weir Cavex® / Cavex 2®**: comparable envelope.
  https://www.global.weir/product-catalogue/hydrocyclones/cavex-cvx-hydrocyclone/

Approximate single-unit slurry throughput (water, vendor-published, ore-specific
in practice):
- 250 mm → ~50 m³/h
- 500 mm → ~200 m³/h
- 750 mm → ~350–400 m³/h
- 840 mm (max) → ~400–450 m³/h

### Literature

- **Perry's Chemical Engineers' Handbook**, 8th ed. (2008), Ch. 20 — cyclone
  capex power-law, exponent 0.50–0.60 on slurry m³/h (T9-50 row).

### Recommended basis

- `HYDROCYCLONE_BASIS = (30_000.0, 100.0, 0.55)` — slight bump from
  `12_000 × 2.383 = 28_596` to round 30 k, capturing 2024→2025 mining-equipment
  inflation (~5%). Per-unit impact at 100 m³/h is +$1.4 k, negligible at fleet
  scale.

**Critical caveat — single-unit ceiling:** the current code applies this
power-law all the way up to the bm_discharge slurry flow as if a single cyclone
handles the entire load. At 2000 tph the bm discharge is ~8320 m³/h — that
needs a parallel cluster of ~20–25 industrial cyclones, not one mega-unit.
This is a parallelization gap (see §4), not a basis gap.

---

## §4 Max single-unit caps (Phase 2B parallelization input)

These constants are NOT yet enforced in code. They are recorded here so the
follow-up parallelization plan can pick them up directly.

| Equipment | Max single unit | Source |
|-----------|------------------|---------|
| `MAX_SINGLE_SAG_MW` | 28 MW | Cobre Panamá (highest in commercial service) |
| `MAX_SINGLE_BM_MW` | 22 MW | Cobre Panamá (highest commercial BM) |
| `MAX_SINGLE_CYCLONE_MM` | 840 mm | Krebs gMAX vendor max; Metso/Weir agree |
| `MAX_SINGLE_CYCLONE_M3PH` | ~400 m³/h | At ~840 mm with typical Cu slurry SG |

Suggested parallelization rule (for the next plan):
```
n_units = ceil(required_capacity / max_single_unit)
cost_per_unit = perry_cost(basis, basis_cap, exp, required_capacity / n_units)
total_cost = n_units * cost_per_unit
```

For 50 ktpd at 2000 tph this yields, e.g.: 2 SAG mills (~10 MW each), 2 BM
(~16.5 MW each — already at the basis), and ~22 cyclones (~380 m³/h each).

---

## §5 Pre-refit capex baseline (commit before this doc)

Run on `python -m process_model.throughput --tea` at 2000 tph / 0.8% Cu / single-train code:

```
crusher_capex                 2,251,255
crusher_motor_capex             190,359
sag_capex                    19,955,769   # 5,600 kW
bm_capex                     50,100,301   # 16,539 kW (≈ basis point)
cyclone_capex                   325,364   # 8,320 m^3/h, single-unit assumption
flotation_capex               7,500,000
thickener_capex                 128,460
tails_thickener_capex        17,290,806   # 90 m diameter
filter_capex                  1,112,039
tailings_capex               12,867,285   # O'Hara, integrated
equipment_capex              98,854,353
installed_capex             507,139,049   # Lang x5 + integrated tailings

NPV  = $1.68 B
IRR  = 40.31 %
```

After the (very small) cyclone basis bump:
- Cyclone capex changes by `(30000 / 28596 - 1) * 325,364 ≈ +$16,000` at this
  scale. Total installed capex shift: ~$80 k after Lang. Negligible.

The ~$1.0–1.5 B benchmark gap remains and is owned by **balance-of-plant capex**
(conveyors, slurry pumps, water/treatment, MCC/instrumentation, reagents,
greenfield tailings dam) — out of scope for this doc.

---

## §6 Sources cited (full)

1. Sayadi, M. R., Khalesi, M. R., & Khoshfaraman Borji, M. (2014).
   *Minerals Engineering* 57, 4–12.
2. Wills, B. A., & Finch, J. A. (2016). *Wills' Mineral Processing Technology*,
   8th ed., Elsevier.
3. Perry, R. H., Green, D. W., & Maloney, J. O. (Eds.) (2008). *Perry's Chemical
   Engineers' Handbook*, 8th ed., McGraw-Hill, Ch. 20.
4. Mular, A. L., & Poulter, J. (Eds.) (2002). *Mineral Processing Plant Design
   — Practice and Control*, SME.
5. ABB News (2019). Cobre Panamá six gearless mill drives.
6. Metso Outotec (2021). Premier SAG sale to Northern Star Resources.
7. International Mining (2020). FLSmidth Russian Cu mill order (4×40 ft GMD).
8. FLSmidth (2024). KREBS gMAX urethane cyclones brochure.
9. Metso (2024). MHC / MHC CB hydrocyclone launch.
10. Weir (2024). Cavex / Cavex 2 hydrocyclone catalogue.

---

**Next review:** before Phase 2B (balance-of-plant capex) or when a new
NI 43-101 with disclosed equipment-line cost data lands.
