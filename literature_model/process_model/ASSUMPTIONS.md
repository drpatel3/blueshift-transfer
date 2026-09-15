# Process Model — Assumptions

Assumptions and equations embedded in `process_model/`. Notation: P80/F80 = 80%-passing product/feed size (µm); tph = tonnes/h; R = recovery (fraction); g = grade.

---

## 1. Scope & structure

- Concentrator-only: ROM ore in, Cu concentrate **or** cathode out. No mine, labor, or G&A; supplier mine cost folded into the ore-purchase coefficient.
- Three mutually-exclusive product routes: (1) sulfide flotation → concentrate sale; (2) concentrate leach (POX/Albion) → cathode; (3) whole-ore heap leach → cathode.
- Always-on stages: crusher, SAG, rougher flotation.
- Byproducts (Mo/Au/Ag) credited only on the sulfide route; zero on hydromet/heap.
- Each stage is a pure function `Stream → Stream`; disabled stages pass stream through at 0 power.
- Ore solids SG = 2.7 in all slurry-volume calcs; flotation cell SG = 2.8; water = 1.0 t/m³.
- `total_capex = installed_capex` — no separate EPCM/contingency line.

---

## 2. Comminution

**Bond third theory** (crusher, SAG, ball, regrind):
$$E_{cs} = 10\,W_i\left(\tfrac{1}{\sqrt{P_{80}}} - \tfrac{1}{\sqrt{F_{80}}}\right)\ \text{kWh/t},\qquad kW = E_{cs}\cdot tph$$

Inverted for product size: $\tfrac{1}{\sqrt{P_{80}}} = \tfrac{E_{cs}}{10W_i} + \tfrac{1}{\sqrt{F_{80}}}$ (SME Eq 20.10).

**Two-stage grind split:** SAG grinds fresh feed at fixed `sag_Ecs = 2.8 kWh/t`; BM Ecs is the Bond remainder to target P80 → `bm_kw = max(0, Ecs_total − sag_Ecs)·tph`. Circulating load sizes BM/cyclone but does **not** change total grinding power.

- Wi = 14.0 kWh/t (all stages); crusher P80 = 150 mm; ore F80 = 300 mm.
- target flotation P80 = 150 µm; circulating load CL = 2.5.
- regrind target P80 = 30 µm.
- SAG/BM JK params: A=70, b=0.7, ta=0.5, PLI=7 (descriptors; no power model in sag.py).
- Crusher (Pothina 2007): idle `P0 = V·A_idle·PF`; crush `P = E·LF/DE·RC`; V=4160, PF=0.85, DE=0.95, LF=1.0, A_idle=30.

**Regrind mill factors** (× ball-mill kWh/t and capex/kW): ball 1.00/1.00; tower 0.75/0.85; isamill 0.60/1.40. *(Mazzinghy 2014; Burford & Clark 2007; Jankovic 2003.)*

---

## 3. Classification (hydrocyclone, Plitt 1976 / Samaeili 2017)

- d50: $d_{50}=C_1\cdot\frac{50.5\,D_c^{0.46}D_i^{0.6}D_x^{1.21}e^{0.063cv}\sqrt{\mu}}{D_u^{0.71}h^{0.38}Q^{0.45}(\rho_p-\rho_f)}$
- flow split $S=Q_o/Q_u$ (Eq 13); water recovery to UF $R_v=S/(S+1)$
- sharpness $m=C_4\cdot1.94\,e^{-1.58R_v(D_x^2/Q)^{0.15}}$
- grade efficiency $Eff(x)=1-e^{-0.693(x/d_{50})^m}$
- Coeffs: C1=0.0387, C2=51.27, C3=0.1953, C4=0.1762; µ=1.0.
- Geometry (cm): Dc=50, Di=12, Dx=15, Du=8, h=120; P=70 kPa.
- Closed-circuit: all fresh feed reports to overflow at steady state; efficiency evaluated at F80 as proxy for full distribution; fallback Eff=0.5 if F80 unknown.
- Cyclone-feed solids 55%; flotation-feed solids 30%.

---

## 4. Flotation (port of Blueshift VBA `FlotRec`)

Mechanistic rate constant from collision/attachment/detachment:
$$k = \beta\,n_b\,P_{att}P_{col}(1-P_{det})\cdot 60\ \text{[1/min]}$$
$$R_{coll}=1-\tfrac{1}{1+k\tau}\quad(\text{single CSTR}),\qquad R_i=\frac{R_{coll}R_{froth}}{R_{coll}R_{froth}+1-R_{coll}}$$

- Surface tension (Szyszkowski): $\sigma=\sigma_0-RT\,\Gamma\ln(kc+1)$, $\sigma_0=0.07243$ N/m @ 23°C.
- Bubble dia (Sauter): $d_b=f_{bub}[2.11\sigma/(\rho_w\varepsilon_{imp}^{0.66})]^{0.6}$; impeller zone $\varepsilon_{imp}=15\,\varepsilon_{mean}$.
- Collision kernel β: Abrahamson (1975). Attachment $P_{att}=e^{-E_b/E_{kin}}$; collision $P_{col}=\min(\tanh^2 x,1)$ (Luttrell–Yoon); detachment $P_{det}=e^{-(W_{adh}+E_b)/E_{kin}}$.
- Froth recovery (Finch–Dobby): $R_{froth}=\min(R_{attach}+R_{entrain},1)$.

**Structural:** rougher bank treated as a **single mixing stage** with bank-total residence time — `R_total = R_i` (deliberate departure from VBA `1−(1−R_i)^N`; NumCells is sizing-only). Scavenger combine: `R_total = R + (1−R)·R_scav`.

- VBA fit: b=2.3, alpha=0.05, bubble_f / detach_f / bulk_zone / coverage = 0.5.
- `energy_barrier_j = 1e-18` J — **placeholder** until full DLVO.
- Rougher recovery cap = 0.90; scavenger-on-tails = 0.50; gangue (entrainment) recovery = 0.10 (calibrated so conc still needs cleaning); conc solids 35%.
- Default frother = MIBC (type 2).

**Cleaner circuit (grade-targeting recycle algebra):**
$$R_{g,target}=\frac{R_{cu,eff}\,c_R(1-g)}{g(1-c_R)},\qquad R_{recycle}=\frac{R_{eff}}{1-R_R(1-R_{eff})}\ \text{(Wills 8th ed.)}$$
- liberation factor $= \text{clamp}(1-0.0020\cdot\Delta P_{80},\,0.85,\,1.05)$; per-stage Cu recovery clamped [0.80, 0.99]; per-stage gangue recovery clamped [0.04, 0.20].
- target cleaner conc grade = 26% Cu; cleaner cell 50 m³; residence 12 min; each stage rejects ~50% gangue; Cu assumed full internal recycle (no per-stage Cu loss).

**Auto-sizing:** `total_vol = Q·(t_res/60)·froth_factor`; `n_cells = ceil(total_vol/cell_vol)`; cell 500 m³, t_res 18 min, froth_factor 1.2.

**Recovery ceilings by mineralogy** (`flot_R_max`): chalcopyrite 0.90, chalcocite 0.85, chalcocite/oxide 0.50, oxide 0.05, bornite 0.92, IOCG 0.90. *(Wills 7th ed. Ch 12.2; SME Handbook 2019 Ch 18.)*

---

## 5. Leaching

**Tank leach (shrinking-core, Sohn & Wadsworth 1979):**
$$k(T)=k_0\,e^{-E_a/RT}\,P_{O_2}^{\,n},\qquad \tfrac{dX}{dt}=3k(1-X)^{2/3},\qquad X(t)=1-(1-kt)^3$$
- k0=500, Ea=70 kJ/mol (chalcopyrite low end), n=0.7, P_O2=12 bar, T=200°C, 4 h/stage, 1 stage.
- Extraction hard cap `MAX_LEACH_EXTRACTION = 0.95`.
- Acid: $\dot m_{acid}=cu_{leached}\cdot1000\cdot6.17$ kg/h (H₂SO₄/Cu = 6.17, SME Ch 20).
- Multi-stage approximated as plug flow (total V = stages × per-stage V).

**Heap leach (fixed empirical extraction, no kinetics):**
$$cu_{leached}=ore_{tph}\cdot g\cdot E,\qquad pls_{m^3/h}=\frac{cu_{leached}\cdot10^6}{pls\_gpl}/1000$$
- E = 0.70 (supergene chalcocite mid-range); acid = 15 kg/t ore (gangue-driven); PLS = 4 g/L Cu; heap 8 m; bulk density 1.7 t/m³; cycle 365 d. Pad area = ore inventory / (ρ·height).
- Not used for chalcopyrite (heap extraction <30%).

**POX/heap extraction ceilings** (`pox_max` / `heap_X`): chalcopyrite 0.95/0.30, chalcocite 0.95/0.70, chalcocite/oxide 0.85/0.80, oxide 0.00/0.85, bornite 0.95/0.30, IOCG 0.95/0.30.

---

## 6. SX–EW

**SX (Moreno et al. 2009)** — dynamic CSTR mixer + non-ideal settler. Extraction isotherm:
$$Y^*=\frac{A\,X^*}{X^*+B},\quad A=a\cdot ML,\quad B=\frac{(10^{-pH})^b}{ML^c}(d\,[Cu]_{PLS}+f\,[Cu]_{BO})$$
Strip isotherm linear: $Y^*=C X^*+D$. Fixed regression coeffs a=0.99, b=1.02, c=1.01, d=35.15, f=27.15, g=0.11, h=444.49, m=0.10, n=0.81, p=−8.91. Default pH=1.8, barren-organic Cu=0.5 g/L. Steady state by fixed-point iteration (tol 1e-6).

**EW (SME Handbook Ch 20)** — Faraday cathode mass:
$$W=\frac{I\,t\,M_{Cu}\,CE}{zF},\quad I=i_{op}A,\quad i_{op}=0.40\,i_L,\quad i_L=18.6\,[Cu]_{g/L}$$
- Cu in electrolyte 45 g/L; CE = 0.92; cell voltage = I·R + 1.96 V; ρ_elec=0.5 Ω·cm; spacing 5 cm; 1.2 m²/plate; 60 cathodes/cell. Check value ~2.1 kWh/kg Cu.

**Steady-state closed recycle** (both hydromet branches):
$$R_{ss}=\frac{R}{1-(1-R)(1-b)},\qquad R=\eta_{SX}\eta_{EW}=0.95\times0.95,\ b=0.03$$
- 3% electrolyte bleed is the only permanent Cu sink; EW spent loss folded into R_ss. *(Wills 8th ed Ch 12.4/15; Schlesinger 5th ed Ch 17.)*

---

## 7. Dewatering & neutralization

**Thickener (Galvez 2014, Richardson–Zaki):** flux $\psi_M=v_{TF}C_M(1-r_F C_M)^n$; max flux $\psi_{max}=\frac{4v_{TF}}{r_F}\frac{n(n-1)^{n-1}}{(n+1)^{n+1}}$; area $A=SF\cdot W/(\rho_s\psi_{max})$, SF=1.25. Tailings thickener unit area 0.30 t/m²/h. Conc UF 65%; tails/thickener default UF 60%.

**Belt filter (Perry Ch 18, Ruth Eq 18-71):** $\theta/(V/A)=\frac{\mu\alpha w}{2\Delta P}(V/A)+\frac{\mu r}{\Delta P}$. Cake moisture is a **user-supplied spec (9%)**, not computed — Perry gives no residual-moisture kinetics. α=1e11, r=1e10, ΔP=70 kPa, min cake 3 mm — **placeholder** until leaf-test values.

**Neutralization (two-stage lime, SME Ch 20 Eq 20.64–20.66):** lime/gypsum/precipitate from stoichiometry, mol/h = kg/h·1000/MW. lime purity 0.95; 2 stages/side; 1 h/stage. Fe/As co-precip rate-limited by min(Fe, As); warn if Fe:As < 3:1. Stage-2 base metals lumped at avg MW 175. Residual sulfate load fixed {Cu 100, Zn 50} kg/h.

**Screen (SME Ch 20):** area = U/(A·B…J); base capacity A=4.5 stph/ft² — **placeholder** Table 20.8 mid-range. DBD sanity gate ≤ 4× aperture.

---

## 8. Mineralogy-keyed ore properties

Mineralogy string selects flotation/POX/heap ceilings (§4–5), byproduct grades, and concentrate As.

**Byproduct head grades** (used when not overridden): chalcopyrite/bornite Mo 250 ppm, Au 0.10, Ag 2.0 g/t; IOCG Au 0.50, Ag 1.7; chalcocite/oxide types zero. *(Sinclair 2007; Sillitoe 2010; BHP 2024 ASR.)*

**Concentrate arsenic** (% As): chalcopyrite 0.05, chalcocite 0.55, chalcocite/oxide 0.30, oxide 0.00, bornite 0.05, IOCG 0.05. *(Lattanzi 2008; Filippou 2007.)* Feeds the smelter As penalty.

- Unknown mineralogy keys default to chalcopyrite. Per-project overrides take precedence over tables.

---

## 9. TEA — capex

**Power-law equipment cost:** $C=C_{basis}(cap/cap_{basis})^{exp}\cdot f_{infl}$; parallel trains when `cap_req > cap_max`: $n=\lceil cap_{req}/cap_{max}\rceil$, $C=n\cdot perry\_cost(cap_{req}/n)$.

**Installed:** $C_{installed}=C_{equipment}\cdot Lang + C_{tailings}+C_{heap}$. Lang = 5.0. Tailings & heap pad bypass Lang (already integrated installed).

- M&S inflation default = 2383/1000 = 2.383 (Perry basis → 2025).
- SAG/ball mill: $50M @ 16.5 MW, exp 0.85 (vendor-anchored 2024); single-train caps 28 MW SAG / 22 MW BM (Cobre Panama).
- Flotation cell: Arfania 2017 `25351·V^0.452` below 158 m³; large-cell tier $1.5M @ 500 m³, exp 0.50 above.
- Thickener (Parkinson–Mular): $510k @ 10 m, exp 1.6. Belt filter $147.75k @ 10 m², exp 0.58.
- Heap pad $200M @ 16 Mt/yr, exp 0.85. Tailings (O'Hara 1992 Eq 6.3.145): $20,000·T^0.5 (1988 USD)·adverse_factor.
- Screen basis ($150k @ 8 m²) — **placeholder**.

---

## 10. TEA — revenue, opex, cashflow

**Concentrate (sulfide):**
$$Rev_{net}=tph_{Cu,pay}\,P_{Cu}H - tph_{conc}\,TC\,H - tph_{Cu,pay}(RC+pen_{As})H$$
$$tph_{Cu,pay}=\max(0,\,tph_{Cu,conc}-0.01\,tph_{conc})\cdot0.965\cdot k_{grade}$$
Grade payable: $k_{grade}=0$ if g<18%; linear ramp to 1 over 18–25%; 1 above 25%.

**Cathode (hydromet):** $Rev=tph_{cathode}\cdot1.0\cdot P_{Cu}H$ (no TC/RC); concentrate & byproduct revenue zeroed.

**Arsenic penalty** ($/t Cu): $0 below 0.20% As; linear to $500 at 0.50%; full rejection (penalty = Cu price) at ≥0.50%. *(AusIMM; Doyle 2010 — ramp shape is engineering judgment.)*

**Cashflow (pre-tax flat annuity):**
$$NPV=-C_{total}+(Rev_{net}-Opex)\sum_{t=1}^{L}(1+r)^{-t},\quad r=0.08,\ L=25\,\text{yr}$$
IRR by bisection over r∈[−0.9, 50].

**Prices / rates:** Cu $12,500/t; TC $22/t conc; RC $50/t Cu; payable 0.965; Mo $44,000/t (R=0.55); Au $80.4M/t (0.95 pay); Ag $964.5k/t (0.85 pay).

**Opex:** power kW·H·$0.05/kWh; water $0.30/m³; reagent $0.50/t ore; grinding media $0.30/kWh; O&M 0.035·capex; hours/yr 7884 (0.9 cap factor).

**Ore purchase** (merchant framing): $/t = 10.0 + g_{Cu}·6000. *(Wood Mac 2024 bottom-up; bands $3,800–$6,750/t Cu.)*

- Hydromet reagent unit costs (acid $150/t, O₂ $100/t, organic $7/L) — **placeholders**; lime $200/t. O₂ proxy = 1 kg O₂/kg Cu leached — placeholder midpoint.

---

## 11. Optimizer (differential evolution)

**Objective:** minimize $-NPV$ at fixed 8% discount; failure penalty 1e12 on exception, non-finite NPV, or overall R < 0.05.

**Decision variables (box bounds only, no nonlinear constraints):**

| var | lo–hi | | var | lo–hi |
|---|---|---|---|---|
| sp_power (kW/m³) | 0.8–1.2 | | slurry_fraction | 0.15–0.30 |
| sp_gas_rate (cm/s) | 0.8–1.5 | | contact_angle (°) | 20–55 |
| frother_conc (ppm) | 10–50 | | circulating_load | 1.5–4.0 |
| flot_residence (min) | 15–20 | | tailings_adverse | 1.0–5.0 |
| air_fraction | 0.10–0.25 | | target_flot_P80 (µm) | 75–200 |

**DE hyperparameters:** maxiter 200, popsize 15, seed 42, tol 0.01, polish True (CLI overrides maxiter 80 / popsize 12). Defaults feed 2000 tph @ 0.8% Cu. Winner = max NPV across topology sweep.

---

## 12. Topology validity rules

- `flotation XOR heap_leach`; `leach XOR heap_leach`.
- `cyclone` requires `ball_mill`; `cleaner` requires `regrind`.
- `leach` requires `flotation AND cleaner` (autoclaves run on ~25–30% Cu cleaner conc, not 1–3% rougher froth).
- `sx` requires a leach path; `ew` requires `sx`; `neutralization` requires a leach path.
- `heap_leach` forces OFF: ball_mill, cyclone, regrind, cleaner, thickener, filter (bypasses fine grind/dewatering).
- `regrind_mill_type ∈ {ball, tower, isamill}`; `n_cleaner_stages ∈ {1,2,3}`.
- Heap route suits low-grade oxide / supergene chalcocite at <0.5% Cu.

---

## 13. Mass balance

- Cu in: $cu_{in}=ore_{tph}\cdot g$. Closure check: $cu_{in}-\sum cu_{out}$ (conc + cathode + tails + raffinate + bleed losses).
- Overall recovery $R=(cu_{conc}+cu_{cathode})/cu_{in}$.
- If leach on: concentrate Cu = 0; unleached Cu rejoins tails.
- Heap route: residue stays on pad, counted as final tails; emits zero concentrate streams.

---

## Flagged placeholders / engineering judgment

flotation `energy_barrier_j` (1e-18); filter α/r and screen capacity factors (non-authoritative); hydromet acid/O₂/organic unit costs and the 1 kg O₂/kg Cu proxy; arsenic penalty ramp shape; flotation bank deliberately collapsed to one mixing stage. **Absolute NPV values should be verified against source PDFs before external use.**
