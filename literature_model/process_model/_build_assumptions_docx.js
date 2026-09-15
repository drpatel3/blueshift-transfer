const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, TableOfContents, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, Header, Footer, PageBreak,
} = require("docx");

// ---- helpers -------------------------------------------------------------
const CONTENT_W = 9360; // US Letter, 1" margins

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
// rich paragraph: array of [text, {b,i}] segments or plain string
function para(parts, opts = {}) {
  const runs = (typeof parts === "string" ? [[parts]] : parts).map(
    ([t, o = {}]) => new TextRun({ text: t, bold: o.b, italics: o.i, font: o.f })
  );
  return new Paragraph({ spacing: { after: 120 }, children: runs, ...opts });
}
function bullet(parts, level = 0) {
  const runs = (typeof parts === "string" ? [[parts]] : parts).map(
    ([t, o = {}]) => new TextRun({ text: t, bold: o.b, italics: o.i })
  );
  return new Paragraph({ numbering: { reference: "bullets", level }, children: runs });
}
// equation: monospaced, indented, slight spacing
function eq(text) {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    indent: { left: 360 },
    children: [new TextRun({ text, font: "Consolas", size: 21 })],
  });
}
// source note (italic, small)
function src(text) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
  });
}
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const mk = (text, w, isHeader) =>
    new TableCell({
      borders,
      width: { size: w, type: WidthType.DXA },
      shading: isHeader ? { fill: "D5E8F0", type: ShadingType.CLEAR } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: !!isHeader, size: 19 })] })],
    });
  const headRow = new TableRow({
    tableHeader: true,
    children: headers.map((t, i) => mk(t, widths[i], true)),
  });
  const bodyRows = rows.map(
    (r) => new TableRow({ children: r.map((t, i) => mk(t, widths[i], false)) })
  );
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [headRow, ...bodyRows],
  });
}

// ---- document ------------------------------------------------------------
const children = [];

// Title block
children.push(
  new Paragraph({
    spacing: { after: 60 },
    children: [new TextRun({ text: "Process Model — Assumptions", bold: true, size: 40 })],
  }),
  new Paragraph({
    spacing: { after: 240 },
    children: [new TextRun({
      text: "Assumptions and equations embedded in process_model/. Notation: P_80 / F_80 = 80%-passing product / feed size (µm); tph = tonnes/h; R = recovery (fraction); g = grade.",
      italics: true, size: 20, color: "555555",
    })],
  }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
);

// 1. Scope
children.push(h1("1.  Scope & structure"));
[
  "Concentrator-only: ROM ore in, Cu concentrate or cathode out. No mine, labor, or G&A; supplier mine cost folded into the ore-purchase coefficient.",
  "Three mutually-exclusive product routes: (1) sulfide flotation → concentrate sale; (2) concentrate leach (POX/Albion) → cathode; (3) whole-ore heap leach → cathode.",
  "Always-on stages: crusher, SAG, rougher flotation.",
  "Byproducts (Mo/Au/Ag) credited only on the sulfide route; zero on hydromet/heap.",
  "Each stage is a pure function Stream → Stream; disabled stages pass stream through at 0 power.",
  "Ore solids SG = 2.7 in all slurry-volume calcs; flotation cell SG = 2.8; water = 1.0 t/m³.",
  "total_capex = installed_capex — no separate EPCM/contingency line.",
].forEach((t) => children.push(bullet(t)));

// 2. Comminution
children.push(h1("2.  Comminution"));
children.push(para([["Bond third theory", { b: true }], [" (crusher, SAG, ball, regrind):"]]));
children.push(eq("E_cs = 10·Wi·(1/√P_80 − 1/√F_80)   kWh/t        kW = E_cs · tph"));
children.push(para([["Inverted for product size:  1/√P_80 = E_cs/(10·Wi) + 1/√F_80   (SME Eq 20.10)."]]));
children.push(para([["Two-stage grind split: ", { b: true }], ["SAG grinds fresh feed at fixed sag_Ecs = 2.8 kWh/t; BM Ecs is the Bond remainder to target P_80 → bm_kw = max(0, Ecs_total − sag_Ecs)·tph. Circulating load sizes BM/cyclone but does not change total grinding power."]]));
[
  "Wi = 14.0 kWh/t (all stages); crusher P_80 = 150 mm; ore F_80 = 300 mm.",
  "target flotation P_80 = 150 µm; circulating load CL = 2.5; regrind target P_80 = 30 µm.",
  "SAG/BM JK params: A=70, b=0.7, ta=0.5, PLI=7 (descriptors; no power model in sag.py).",
  "Crusher (Pothina 2007): idle P0 = V·A_idle·PF; crush P = E·LF/DE·RC; V=4160, PF=0.85, DE=0.95, LF=1.0, A_idle=30.",
].forEach((t) => children.push(bullet(t)));
children.push(para([["Regrind mill factors", { b: true }], [" (× ball-mill kWh/t and capex/kW): ball 1.00/1.00; tower 0.75/0.85; isamill 0.60/1.40."]]));
children.push(src("Mazzinghy 2014; Burford & Clark 2007; Jankovic 2003."));

// 3. Classification
children.push(h1("3.  Classification (hydrocyclone, Plitt 1976 / Samaeili 2017)"));
children.push(eq("d50 = C1 · 50.5·Dc^0.46·Di^0.6·Dx^1.21·e^(0.063·cv)·√µ / [Du^0.71·h^0.38·Q^0.45·(ρp−ρf)]"));
children.push(eq("S = Qo/Qu (flow split, Eq 13)        Rv = S/(S+1)   (water recovery to UF)"));
children.push(eq("m = C4·1.94·e^(−1.58·Rv·(Dx²/Q)^0.15)        Eff(x) = 1 − e^(−0.693·(x/d50)^m)"));
[
  "Coeffs: C1=0.0387, C2=51.27, C3=0.1953, C4=0.1762; µ=1.0.",
  "Geometry (cm): Dc=50, Di=12, Dx=15, Du=8, h=120; P=70 kPa.",
  "Closed-circuit: all fresh feed reports to overflow at steady state; efficiency evaluated at F_80 as proxy for full distribution; fallback Eff=0.5 if F_80 unknown.",
  "Cyclone-feed solids 55%; flotation-feed solids 30%.",
].forEach((t) => children.push(bullet(t)));

// 4. Flotation
children.push(h1("4.  Flotation (port of Blueshift VBA FlotRec)"));
children.push(para([["Mechanistic rate constant from collision / attachment / detachment:"]]));
children.push(eq("k = β·n_b·P_att·P_col·(1−P_det)·60     [1/min]"));
children.push(eq("R_coll = 1 − 1/(1+k·τ)   (single CSTR)"));
children.push(eq("R_i = R_coll·R_froth / (R_coll·R_froth + 1 − R_coll)"));
[
  "Surface tension (Szyszkowski): σ = σ0 − R·T·Γ·ln(k·c + 1), σ0 = 0.07243 N/m @ 23°C.",
  "Bubble dia (Sauter): d_b = f_bub·[2.11·σ/(ρw·ε_imp^0.66)]^0.6; impeller zone ε_imp = 15·ε_mean.",
  "Collision kernel β: Abrahamson (1975). Attachment P_att = e^(−E_b/E_kin); collision P_col = min(tanh²x, 1) (Luttrell–Yoon); detachment P_det = e^(−(W_adh+E_b)/E_kin).",
  "Froth recovery (Finch–Dobby): R_froth = min(R_attach + R_entrain, 1).",
].forEach((t) => children.push(bullet(t)));
children.push(para([["Structural: ", { b: true }], ["rougher bank treated as a single mixing stage with bank-total residence time — R_total = R_i (deliberate departure from VBA 1−(1−R_i)^N; NumCells is sizing-only). Scavenger combine: R_total = R + (1−R)·R_scav."]]));
[
  "VBA fit: b=2.3, alpha=0.05, bubble_f / detach_f / bulk_zone / coverage = 0.5.",
  "energy_barrier_j = 1e-18 J — placeholder until full DLVO.",
  "Rougher recovery cap = 0.90; scavenger-on-tails = 0.50; gangue (entrainment) recovery = 0.10 (calibrated so conc still needs cleaning); conc solids 35%.",
  "Default frother = MIBC (type 2).",
].forEach((t) => children.push(bullet(t)));
children.push(para([["Cleaner circuit (grade-targeting recycle algebra):", { b: true }]]));
children.push(eq("R_g,target = R_cu,eff·c_R·(1−g) / [g·(1−c_R)]"));
children.push(eq("R_recycle = R_eff / [1 − R_R·(1−R_eff)]     (Wills 8th ed.)"));
[
  "liberation factor = clamp(1 − 0.0020·ΔP_80, 0.85, 1.05); per-stage Cu recovery clamped [0.80, 0.99]; per-stage gangue recovery clamped [0.04, 0.20].",
  "target cleaner conc grade = 26% Cu; cleaner cell 50 m³; residence 12 min; each stage rejects ~50% gangue; Cu assumed full internal recycle (no per-stage Cu loss).",
].forEach((t) => children.push(bullet(t)));
children.push(para([["Auto-sizing: ", { b: true }], ["total_vol = Q·(t_res/60)·froth_factor; n_cells = ceil(total_vol/cell_vol); cell 500 m³, t_res 18 min, froth_factor 1.2."]]));
children.push(para([["Recovery ceilings by mineralogy (flot_R_max):", { b: true }]]));
children.push(table(
  ["mineralogy", "flot_R_max"],
  [["chalcopyrite", "0.90"], ["chalcocite", "0.85"], ["chalcocite/oxide", "0.50"],
   ["oxide", "0.05"], ["bornite", "0.92"], ["IOCG", "0.90"]],
  [4680, 4680]));
children.push(src("Wills 7th ed. Ch 12.2; SME Handbook 2019 Ch 18."));

// 5. Leaching
children.push(h1("5.  Leaching"));
children.push(para([["Tank leach (shrinking-core, Sohn & Wadsworth 1979):", { b: true }]]));
children.push(eq("k(T) = k0·e^(−Ea/RT)·P_O2^n     dX/dt = 3k·(1−X)^(2/3)     X(t) = 1 − (1−k·t)³"));
[
  "k0=500, Ea=70 kJ/mol (chalcopyrite low end), n=0.7, P_O2=12 bar, T=200°C, 4 h/stage, 1 stage.",
  "Extraction hard cap MAX_LEACH_EXTRACTION = 0.95.",
  "Acid: ṁ_acid = cu_leached·1000·6.17 kg/h (H₂SO₄/Cu = 6.17, SME Ch 20).",
  "Multi-stage approximated as plug flow (total V = stages × per-stage V).",
].forEach((t) => children.push(bullet(t)));
children.push(para([["Heap leach (fixed empirical extraction, no kinetics):", { b: true }]]));
children.push(eq("cu_leached = ore_tph·g·E       pls_m3/h = (cu_leached·1e6 / pls_gpl)/1000"));
[
  "E = 0.70 (supergene chalcocite mid-range); acid = 15 kg/t ore (gangue-driven); PLS = 4 g/L Cu; heap 8 m; bulk density 1.7 t/m³; cycle 365 d. Pad area = ore inventory / (ρ·height).",
  "Not used for chalcopyrite (heap extraction <30%).",
].forEach((t) => children.push(bullet(t)));
children.push(para([["POX / heap extraction ceilings (pox_max / heap_X):", { b: true }]]));
children.push(table(
  ["mineralogy", "pox_max", "heap_X"],
  [["chalcopyrite", "0.95", "0.30"], ["chalcocite", "0.95", "0.70"], ["chalcocite/oxide", "0.85", "0.80"],
   ["oxide", "0.00", "0.85"], ["bornite", "0.95", "0.30"], ["IOCG", "0.95", "0.30"]],
  [4680, 2340, 2340]));

// 6. SX-EW
children.push(h1("6.  SX–EW"));
children.push(para([["SX (Moreno et al. 2009)", { b: true }], [" — dynamic CSTR mixer + non-ideal settler. Extraction isotherm:"]]));
children.push(eq("Y* = A·X* / (X* + B)     A = a·ML     B = (10^−pH)^b / ML^c · (d·[Cu]_PLS + f·[Cu]_BO)"));
children.push(para([["Strip isotherm linear: Y* = C·X* + D. Fixed regression coeffs a=0.99, b=1.02, c=1.01, d=35.15, f=27.15, g=0.11, h=444.49, m=0.10, n=0.81, p=−8.91. Default pH=1.8, barren-organic Cu=0.5 g/L. Steady state by fixed-point iteration (tol 1e-6)."]]));
children.push(para([["EW (SME Handbook Ch 20)", { b: true }], [" — Faraday cathode mass:"]]));
children.push(eq("W = I·t·M_Cu·CE / (z·F)     I = i_op·A     i_op = 0.40·i_L     i_L = 18.6·[Cu]_g/L"));
children.push(bullet("Cu in electrolyte 45 g/L; CE = 0.92; cell voltage = I·R + 1.96 V; ρ_elec=0.5 Ω·cm; spacing 5 cm; 1.2 m²/plate; 60 cathodes/cell. Check value ~2.1 kWh/kg Cu."));
children.push(para([["Steady-state closed recycle (both hydromet branches):", { b: true }]]));
children.push(eq("R_ss = R / [1 − (1−R)·(1−b)]     R = η_SX·η_EW = 0.95×0.95     b = 0.03"));
children.push(bullet("3% electrolyte bleed is the only permanent Cu sink; EW spent loss folded into R_ss."));
children.push(src("Wills 8th ed Ch 12.4/15; Schlesinger 5th ed Ch 17."));

// 7. Dewatering & neutralization
children.push(h1("7.  Dewatering & neutralization"));
children.push(para([["Thickener (Galvez 2014, Richardson–Zaki):", { b: true }]]));
children.push(eq("ψ_M = vTF·C_M·(1−rF·C_M)^n       ψ_max = (4·vTF/rF)·n·(n−1)^(n−1)/(n+1)^(n+1)"));
children.push(eq("A = SF·W/(ρs·ψ_max),  SF=1.25"));
children.push(bullet("Tailings thickener unit area 0.30 t/m²/h. Conc UF 65%; tails/thickener default UF 60%."));
children.push(para([["Belt filter (Perry Ch 18, Ruth Eq 18-71):", { b: true }]]));
children.push(eq("θ/(V/A) = µ·α·w/(2·ΔP)·(V/A) + µ·r/ΔP"));
children.push(bullet("Cake moisture is a user-supplied spec (9%), not computed — Perry gives no residual-moisture kinetics. α=1e11, r=1e10, ΔP=70 kPa, min cake 3 mm — placeholder until leaf-test values."));
children.push(para([["Neutralization (two-stage lime, SME Ch 20 Eq 20.64–20.66):", { b: true }], [" lime/gypsum/precipitate from stoichiometry, mol/h = kg/h·1000/MW. lime purity 0.95; 2 stages/side; 1 h/stage. Fe/As co-precip rate-limited by min(Fe, As); warn if Fe:As < 3:1. Stage-2 base metals lumped at avg MW 175. Residual sulfate load fixed {Cu 100, Zn 50} kg/h."]]));
children.push(para([["Screen (SME Ch 20):", { b: true }], [" area = U/(A·B…J); base capacity A=4.5 stph/ft² — placeholder Table 20.8 mid-range. DBD sanity gate ≤ 4× aperture."]]));

// 8. Mineralogy
children.push(h1("8.  Mineralogy-keyed ore properties"));
children.push(para("Mineralogy string selects flotation/POX/heap ceilings (§4–5), byproduct grades, and concentrate As."));
children.push(para([["Byproduct head grades (used when not overridden):", { b: true }]]));
children.push(table(
  ["mineralogy", "Mo (ppm)", "Au (g/t)", "Ag (g/t)"],
  [["chalcopyrite / bornite", "250", "0.10", "2.0"], ["IOCG", "0", "0.50", "1.7"],
   ["chalcocite / oxide types", "0", "0", "0"]],
  [3360, 2000, 2000, 2000]));
children.push(src("Sinclair 2007; Sillitoe 2010; BHP 2024 ASR."));
children.push(para([["Concentrate arsenic (% As):", { b: true }]]));
children.push(table(
  ["mineralogy", "% As"],
  [["chalcopyrite", "0.05"], ["chalcocite", "0.55"], ["chalcocite/oxide", "0.30"],
   ["oxide", "0.00"], ["bornite", "0.05"], ["IOCG", "0.05"]],
  [4680, 4680]));
children.push(src("Lattanzi 2008; Filippou 2007. Feeds the smelter As penalty."));
[
  "Unknown mineralogy keys default to chalcopyrite.",
  "Per-project overrides take precedence over tables.",
].forEach((t) => children.push(bullet(t)));

// 9. TEA capex
children.push(h1("9.  TEA — capex"));
children.push(para([["Power-law equipment cost:", { b: true }]]));
children.push(eq("C = C_basis·(cap/cap_basis)^exp · f_infl"));
children.push(eq("parallel trains when cap_req > cap_max:  n = ceil(cap_req/cap_max),  C = n·perry_cost(cap_req/n)"));
children.push(para([["Installed:", { b: true }]]));
children.push(eq("C_installed = C_equipment·Lang + C_tailings + C_heap     Lang = 5.0"));
children.push(bullet("Tailings & heap pad bypass Lang (already integrated installed)."));
[
  "M&S inflation default = 2383/1000 = 2.383 (Perry basis → 2025).",
  "SAG/ball mill: $50M @ 16.5 MW, exp 0.85 (vendor-anchored 2024); single-train caps 28 MW SAG / 22 MW BM (Cobre Panama).",
  "Flotation cell: Arfania 2017 25351·V^0.452 below 158 m³; large-cell tier $1.5M @ 500 m³, exp 0.50 above.",
  "Thickener (Parkinson–Mular): $510k @ 10 m, exp 1.6. Belt filter $147.75k @ 10 m², exp 0.58.",
  "Heap pad $200M @ 16 Mt/yr, exp 0.85. Tailings (O'Hara 1992 Eq 6.3.145): $20,000·T^0.5 (1988 USD)·adverse_factor.",
  "Screen basis ($150k @ 8 m²) — placeholder.",
].forEach((t) => children.push(bullet(t)));

// 10. TEA revenue/opex
children.push(h1("10.  TEA — revenue, opex, cashflow"));
children.push(para([["Concentrate (sulfide):", { b: true }]]));
children.push(eq("Rev_net = tph_Cu,pay·P_Cu·H − tph_conc·TC·H − tph_Cu,pay·(RC + pen_As)·H"));
children.push(eq("tph_Cu,pay = max(0, tph_Cu,conc − 0.01·tph_conc)·0.965·k_grade"));
children.push(bullet("Grade payable: k_grade = 0 if g<18%; linear ramp to 1 over 18–25%; 1 above 25%."));
children.push(para([["Cathode (hydromet):", { b: true }], [" Rev = tph_cathode·1.0·P_Cu·H (no TC/RC); concentrate & byproduct revenue zeroed."]]));
children.push(para([["Arsenic penalty ($/t Cu):", { b: true }], [" $0 below 0.20% As; linear to $500 at 0.50%; full rejection (penalty = Cu price) at ≥0.50%."]]));
children.push(src("AusIMM; Doyle 2010 — ramp shape is engineering judgment."));
children.push(para([["Cashflow (pre-tax flat annuity):", { b: true }]]));
children.push(eq("NPV = −C_total + (Rev_net − Opex)·Σ_{t=1..L} (1+r)^−t     r = 0.08, L = 25 yr"));
children.push(bullet("IRR by bisection over r ∈ [−0.9, 50]."));
[
  ["Prices / rates: ", { b: true }, "Cu $12,500/t; TC $22/t conc; RC $50/t Cu; payable 0.965; Mo $44,000/t (R=0.55); Au $80.4M/t (0.95 pay); Ag $964.5k/t (0.85 pay)."],
  ["Opex: ", { b: true }, "power kW·H·$0.05/kWh; water $0.30/m³; reagent $0.50/t ore; grinding media $0.30/kWh; O&M 0.035·capex; hours/yr 7884 (0.9 cap factor)."],
  ["Ore purchase (merchant framing): ", { b: true }, "$/t = 10.0 + g_Cu·6000 (Wood Mac 2024 bottom-up; bands $3,800–$6,750/t Cu)."],
  ["Hydromet reagent unit costs ", { b: false }, "(acid $150/t, O₂ $100/t, organic $7/L) — placeholders; lime $200/t. O₂ proxy = 1 kg O₂/kg Cu leached — placeholder midpoint."],
].forEach(([a, o, b]) => children.push(bullet([[a, o], [b]])));

// 11. Optimizer
children.push(h1("11.  Optimizer (differential evolution)"));
children.push(para([["Objective:", { b: true }], [" minimize −NPV at fixed 8% discount; failure penalty 1e12 on exception, non-finite NPV, or overall R < 0.05."]]));
children.push(para([["Decision variables (box bounds only, no nonlinear constraints):", { b: true }]]));
children.push(table(
  ["variable", "lo–hi", "variable", "lo–hi"],
  [
    ["sp_power (kW/m³)", "0.8–1.2", "slurry_fraction", "0.15–0.30"],
    ["sp_gas_rate (cm/s)", "0.8–1.5", "contact_angle (°)", "20–55"],
    ["frother_conc (ppm)", "10–50", "circulating_load", "1.5–4.0"],
    ["flot_residence (min)", "15–20", "tailings_adverse", "1.0–5.0"],
    ["air_fraction", "0.10–0.25", "target_flot_P80 (µm)", "75–200"],
  ],
  [2700, 1980, 2700, 1980]));
children.push(para([["DE hyperparameters: ", { b: true }], ["maxiter 200, popsize 15, seed 42, tol 0.01, polish True (CLI overrides maxiter 80 / popsize 12). Defaults feed 2000 tph @ 0.8% Cu. Winner = max NPV across topology sweep."]]));

// 12. Topology
children.push(h1("12.  Topology validity rules"));
[
  "flotation XOR heap_leach; leach XOR heap_leach.",
  "cyclone requires ball_mill; cleaner requires regrind.",
  "leach requires flotation AND cleaner (autoclaves run on ~25–30% Cu cleaner conc, not 1–3% rougher froth).",
  "sx requires a leach path; ew requires sx; neutralization requires a leach path.",
  "heap_leach forces OFF: ball_mill, cyclone, regrind, cleaner, thickener, filter (bypasses fine grind/dewatering).",
  "regrind_mill_type ∈ {ball, tower, isamill}; n_cleaner_stages ∈ {1,2,3}.",
  "Heap route suits low-grade oxide / supergene chalcocite at <0.5% Cu.",
].forEach((t) => children.push(bullet(t)));

// 13. Mass balance
children.push(h1("13.  Mass balance"));
children.push(eq("cu_in = ore_tph·g       closure = cu_in − Σ cu_out  (conc + cathode + tails + raffinate + bleed)"));
children.push(eq("R_overall = (cu_conc + cu_cathode) / cu_in"));
[
  "If leach on: concentrate Cu = 0; unleached Cu rejoins tails.",
  "Heap route: residue stays on pad, counted as final tails; emits zero concentrate streams.",
].forEach((t) => children.push(bullet(t)));

// Placeholders
children.push(h1("Flagged placeholders / engineering judgment"));
children.push(para([["flotation energy_barrier_j (1e-18); filter α/r and screen capacity factors (non-authoritative); hydromet acid/O₂/organic unit costs and the 1 kg O₂/kg Cu proxy; arsenic penalty ramp shape; flotation bank deliberately collapsed to one mixing stage. "], ["Absolute NPV values should be verified against source PDFs before external use.", { b: true }]]));

// ---- assemble ------------------------------------------------------------
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2E5496" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 270 }, spacing: { after: 40 } } } },
      ] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Process Model — Assumptions    ·    ", size: 16, color: "888888" }),
          new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "888888" })],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("ASSUMPTIONS.docx", buf);
  console.log("wrote ASSUMPTIONS.docx", buf.length, "bytes");
});
