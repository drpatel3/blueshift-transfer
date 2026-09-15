# process_model — source references

Equations, tables, and parameter sources grounding each module. Add a new
section per module as we build; keep entries tight — equations, numbered
items, and a short "not in this source" list.

---

# Filter — Perry Ch. 18 (7th ed., 1999)

## Ruth constant-pressure filtration (Eq. 18-71)

```
θ / (V/A) = [μ·α·w / (2·(P − P1))] · (V/A)  +  μ·r / (P − P1)
```

- `θ` time (s), `V` filtrate vol (m³), `A` area (m²), `μ` filtrate viscosity (Pa·s)
- `α` specific cake resistance (m/kg), `r` medium resistance (1/m)
- `w` dry cake mass per filtrate volume (kg/m³)
- `P` total ΔP (Pa), `P1 = μ·r·V/(A·θ)` = medium ΔP

Leaf-test fit (Fig. 18-107a): plot θ/(V/A) vs V/A → slope `= μαw/(2P)`, intercept `= μr/P`.
→ `α = 2P·slope/(μw)`, `r = P·intercept/μ`.

## Compressibility (Eq. 18-74)

```
α = α′ · P^s
```
`s = 0` rigid, `s = 1` highly compressible, `s = 0.1–0.8` most industrial slurries.
Fit from runs at ≥3 pressures, log α vs log P.

## Washing (Eqs. 18-62, 18-63)

- 18-62: W vs cake thickness (wash volume vs thickness, graphical)
- 18-63: log R vs N — R = % solute remaining after wash, N = wash ratio
- Wash time ∝ (cake thickness)² at fixed wash/solids ratio.

## Minimum cake thickness for discharge (Table 18-8)

Horizontal belt: **3–5 mm**. Below 3 mm → cake won't discharge.

## NOT in Perry — do not cite Perry for these

- Residual cake moisture vs dewatering time (exponential form is Wakeman-Tarleton, not Perry)
- Belt-zone time split (form:dewater:discharge)
- Vacuum pump power, air-to-filtrate leakage ratio
- Numerical α default for Cu concentrate (Perry gives the fit procedure, not a value — needs leaf test)

---

# TEA economic parameters — Wood Mac, AusIMM, Jones Day, smelter literature

Citations for monetary coefficients in `tea.py`, `leach.py`, and the
cleaner block in `throughput.py`. Add a new sub-section per coefficient;
include the source, the number it provides, and how it maps to the model.

## Ore purchase price — merchant-concentrator framing

Default: $10/t-ore base + $6,000/t-Cu. Bottom-up build:

| Component | $/t Cu | Source |
|---|---:|---|
| C1 cash cost (mid porphyry) | 2,275 | Wood Mac 2024 Cu cost curve, $0.85-1.20/lb midpoint |
| Sustaining capex (1-2% installed/yr) | 700 | Industry rule of thumb (CRU, AME) |
| New-build capex amortization (25-yr) | 1,900 | Greenfield-leaning mid (sunk-capex = $0, full-greenfield = $2,000) |
| Royalty paid by mine | 400 | Chile sliding 5-14% on margin — Jones Day 2023; ~3% NSR equiv |
| Mine ROE (15% on AISC) | 725 | Standard porphyry-Cu equity hurdle |
| **Total** | **6,000** | `tea.py: DEFAULT_ORE_PURCHASE_PER_T_CU` |

Defensible bands per supplier type:
- $3,800/t Cu : mature sunk-capex mine (Codelco / BHP existing operations)
- $5,000/t Cu : mid-life mine, partial capex recovery
- $6,000/t Cu : default — partially-greenfield supplier with closer-to-full capex amortization
- $6,750/t Cu : full-greenfield mine demanding full capex payback

Refs:
- AusIMM "Copper Concentrate Marketing 101"
  https://www.ausimm.com/bulletin/bulletin-articles/copper-concentrate-marketing-101/
- Jones Day, "Chile New Mining Royalty Law" (2023)
  https://www.jonesday.com/-/media/files/publications/2023/07/chile-new-mining-royalty-law/
- Wikipedia, "Net smelter return"
  https://en.wikipedia.org/wiki/Net_smelter_return

NOTE: merchant Cu *ore-purchase at scale* is not a standard industry
structure — see search-agent finding logged in `project_tea_state.md`.
The defaults above are constructed from cited mine cost curves rather
than from an observed merchant ore contract.

## Smelter min-grade payable schedule

`tea.py: smelter_min_grade = 0.18`, `smelter_full_payable_grade = 0.25`.

- Below 18% Cu : refused (payable factor → 0)
- 18-25% : linear ramp 0 → 1
- ≥25% : full payable

AusIMM "Copper Concentrate Marketing 101" confirms all terms negotiated;
no fixed schedule exists. The 18%/25% breakpoints are typical industry
spec — modern smelters (Pasar, Aurubis Olen, Boliden Harjavalta) refuse
or heavily penalise concentrate below ~18% Cu.

Refs:
- AusIMM "Copper Concentrate Marketing 101"
- KAPA Solutions "TC/RC Pricing Method"
  https://kaparesolutions.com/wp-content/uploads/2025/10/KAPA_TCRC_Pricing_Method.pdf
- Boliden, "Copper Smelters Revenue Stream" (2008 CMD)
  https://investors.boliden.com/sites/boliden-ir/files/files/cmd%20and%20other%20events/2008/13-copper-smelters-revenue-stream-ulf-soderstrom-president-ba-market.pdf

## Leach Cu extraction cap — refractory-Cu floor

`leach.py: MAX_LEACH_EXTRACTION = 0.95`. Override per `LeachParams.max_extraction`.

| Process | Industrial extraction at full residence | Source |
|---|---:|---|
| POX chalcopyrite, 200-230°C, 12 bar O2 | 92-97% | Phelps Dodge El Abra / Sherritt Bagdad pilot |
| Albion atmospheric oxidative | 95-98% | Glencore McArthur River equivalent |
| Atmospheric chalcocite tank leach | 85-95% | Industry typical |

0.95 sits at the middle of the POX band. Cap exists because the
shrinking-core integrated form `1 - (1-kt)^3` mathematically approaches
1.0 as kt grows but real ore always has a refractory-Cu fraction locked
in accessory phases (silicates, iron oxides, composite particles) that
no residence time will liberate.

## Cleaner R_Cu and gangue carryover — regrind P80 scaling

`throughput.py: cleaner_R_cu = 0.92, cleaner_gangue_recovery = 0.20` at
the reference regrind P80 of 30 μm; scaled by `_cleaner_factors()` based
on the `regrind_target_P80_um` parameter.

| Regrind P80 | R_Cu | R_g | Resulting cleaner conc grade (typical) |
|---:|---:|---:|---|
| 15 μm | 0.94 | 0.13 | 35-40% Cu |
| 30 μm (default) | 0.92 | 0.20 | 25-30% Cu |
| 50 μm | 0.88 | 0.30 | 18-22% Cu |

Calibrated against industry-typical mid-grade Cu concentrators:
- Cobre Panama 24-26%
- Escondida 28-30%
- Cerro Verde 28-32%
- Antamina 30-38%

Single-stage equivalent — real plants run 2-3 cleaner stages compounding
to the same overall result. No published per-P80 fit; coefficients are
engineering judgment anchored to the operating points above.

## Rougher gangue recovery (entrainment)

`throughput.py: gangue_recovery = 0.10` (was 0.03).

Real Cu rougher banks run 0.08-0.15 (entrainment + true flotation of
NSG). Source: Wills' Mineral Processing Technology 7th ed., Ch. 12.5.
The previous 0.03 default produced a rougher concentrate already at
20-30% Cu — physically inconsistent with industry rougher operation.

## Smelter arsenic-penalty schedule

`tea.py: smelter_as_penalty_per_t_cu(conc_arsenic_pct, cu_price_per_t)`
with `AS_PENALTY_FREE_PCT = 0.20`, `AS_PENALTY_REJECT_PCT = 0.50`,
`AS_PENALTY_AT_REJECT_USD = 500.0`. Replaces the prior flat per-mineralogy
penalty (`SMELTER_IMPURITY_PENALTY_PER_T_CU`, removed) with a schedule keyed
to the *concentrate As grade* — a disclosed ore property — not to a plant
or mineralogy name. Concentrate As is supplied per deposit on
`CuSulfideParams.conc_arsenic_pct`, defaulting to the class table
`throughput.AS_CONC_PCT_BY_MINERALOGY`.

| As in conc | Penalty ($/t payable Cu) | Basis |
|---|---|---|
| ≤ 0.20 % | 0 (clean) | ~0.2 % (2000 ppm) widely-quoted clean-concentrate As limit — AusIMM "Copper Concentrate Marketing 101" |
| 0.20–0.50 % | linear 0 → 500 | endpoint $500/t Cu (~2.3 c/lb) = top-quartile disclosed As-rich deduction (Doyle 2010, SEG SP 18 — established project source); ramp shape is engineering judgment |
| ≥ 0.50 % | = Cu price (unsaleable) | ~0.5 % As is the de-facto rejection threshold set by the dominant import market (China); concentrate carries no net payable value |

Concentrate As class defaults (`AS_CONC_PCT_BY_MINERALOGY`):

| Class | As % | Basis |
|---|---:|---|
| chalcopyrite / bornite | 0.05 | clean porphyry concentrate; below the 0.2 % free limit → $0 |
| chalcocite | 0.55 | mid-point of the ~0.1-1 % enargite-belt range (El Abra, Spence Hypogene, Chuquicamata/Ministro Hales). Lattanzi et al. 2008 (enargite review); Filippou et al. 2007 (As in Cu metallurgy). Mid-point is engineering judgment |
| chalcocite/oxide | 0.30 | interpolated judgment, no single disclosure; routing-insensitive (routes to heap) |
| oxide | 0.00 | does not flot (flot_R_max 0.05); moot |
| IOCG | 0.05 | Olympic Dam-class low-As concentrate (U/F is the relevant penalty, out of scope) |

Effect: high-As chalcocite (≥ 0.5 %) makes the sulfide-concentrate route
commercially unviable, so the model routes those deposits to a cathode-
producing route (heap / POX) — the documented reason the Andean enargite
belt built POX/roasting. The class defaults are central values within the
literature ranges, not fitted constants; supply a disclosed per-project As
grade where one exists.

NOTE — citation precision still open (flagged by tea-auditor, pending user
confirmation): exact China import-standard number/clause; the Doyle 2010
chapter/page within SEG SP 18; and full bibliographic entries for Lattanzi
2008 and Filippou 2007. The values are defensible within the cited ranges;
the precise document identifiers should be confirmed by the SME.

Refs:
- AusIMM "Copper Concentrate Marketing 101"
  https://www.ausimm.com/bulletin/bulletin-articles/copper-concentrate-marketing-101/
- Doyle (2010), SEG Special Publication 18 — As-rich Cu concentrate penalty disclosures
- Lattanzi et al. (2008), "Enargite oxidation: A review," Earth-Science Reviews
- Filippou, St. Germain & Grammatikopoulos (2007), "Recovery of metal values
  from copper–arsenic minerals and other related resources," Min. Proc. Ext. Met. Rev.
