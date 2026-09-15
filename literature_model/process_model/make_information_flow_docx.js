// Generates process_model/INFORMATION_FLOW.docx
// One-page narrative + tables explaining how information moves through the model.
// Appendix carries the ASCII block diagram.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, PageOrientation, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageBreak,
} = require("C:/Users/Ryan Murray/AppData/Roaming/npm/node_modules/docx");

const border = { style: BorderStyle.SINGLE, size: 4, color: "888888" };
const cellBorders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

const P = (text, opts = {}) => new Paragraph({
  spacing: { after: 80 },
  ...opts,
  children: Array.isArray(text) ? text : [new TextRun(text)],
});

const H1 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: t, bold: true, size: 26 })],
});

const H2 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 100, after: 60 },
  children: [new TextRun({ text: t, bold: true, size: 22 })],
});

const cell = (text, opts = {}) => new TableCell({
  borders: cellBorders,
  margins: cellMargins,
  width: { size: opts.width, type: WidthType.DXA },
  shading: opts.header ? { fill: "E6EEF5", type: ShadingType.CLEAR } : undefined,
  children: [new Paragraph({
    spacing: { after: 0 },
    children: [new TextRun({ text, bold: !!opts.header, size: 18 })],
  })],
});

function makeTable(cols, rows) {
  const widths = cols.map(c => c.width);
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        children: cols.map(c => cell(c.header, { header: true, width: c.width })),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((v, i) => cell(v, { width: widths[i] })),
      })),
    ],
  });
}

// ---------- Content ----------

const title = new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: "Process Model — Information Flow", bold: true, size: 32 })],
});
const subtitle = new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 160 },
  children: [new TextRun({
    text: "One-page companion to ASSUMPTIONS.md — how inputs become an NPV",
    italics: true, size: 20, color: "555555",
  })],
});

const intro = P([
  new TextRun({ text: "Question this doc answers: ", bold: true }),
  new TextRun("given a ROM ore description, how does the model decide a flowsheet and turn it into an NPV? Each stage is a pure function "),
  new TextRun({ text: "Stream → Stream", italics: true }),
  new TextRun(" (mass, water, size, Cu). The optimizer searches over which stages are on and their operating points; the TEA module converts the steady-state streams into cashflow."),
]);

// Structure block
const structure = [
  H1("1. Structure: three layers"),
  P([
    new TextRun("The model is split into three layers that communicate only through the "),
    new TextRun({ text: "Stream", italics: true }),
    new TextRun(" contract and a topology bit-vector. Stages compute physics, the flowsheet wires them by mass balance, and the decision layer searches over which stages are on and how they are tuned. Each layer can be re-fit without touching the others."),
  ]),

  H2("1.1 Stage layer — first-principles units"),
  P([
    new TextRun("Every unit operation is implemented as a pure function "),
    new TextRun({ text: "Stream → (Stream, kW)", italics: true }),
    new TextRun(" with a small frozen dataclass holding its design parameters. Stages hold no cross-call state; if a stage is disabled by topology its input passes through unchanged at 0 kW. The kW return is the only side-channel and is forwarded to TEA for power opex."),
  ]),
  makeTable(
    [
      { header: "Unit", width: 1500 },
      { header: "Param dataclass", width: 1900 },
      { header: "Reads from Stream", width: 2200 },
      { header: "Writes to Stream", width: 2200 },
      { header: "kW model", width: 1560 },
    ],
    [
      ["Crusher",         "CrusherParams",      "F80",                          "F80 → P80 (150 mm)",       "Bond Wi crushing"],
      ["SAG + BM",        "SAGParams, BMParams","F80, solids_tph",              "P80 (µm), water (CL)",     "Bond Ecs + ω"],
      ["Hydrocyclone",    "CycloneParams",      "solids/water, P80",            "UF/OF split, d50",         "Negligible"],
      ["Flotation (Ro)",  "FlotOp, FlotKin",    "P80, cu_grade, mineralogy",    "cu_grade↑ (conc) / cu_grade↓ (tails)", "P = ρ·V·sp_power"],
      ["Regrind + Cleaner","RegrindParams, CleanerParams","conc P80, cu_grade","P80↓, target g=26% Cu",     "Bond Ecs (mill type)"],
      ["Tank leach (POX)","LeachParams",        "conc cu_grade, mineralogy",    "cu_aq_gpl, residue",       "Compressor + agitator"],
      ["Heap leach",      "HeapLeachParams",    "ROM cu_grade, mineralogy",     "cu_aq_gpl (PLS)",          "Pumps only"],
      ["SX–EW",           "SXParams, EWParams", "cu_aq_gpl (PLS)",              "cathode tph, raffinate",   "Faraday CE × I"],
      ["Thickener/Filter","ThickenerParams, FilterParams","solids/water",       "% solids ↑ to ~91% cake",  "Rake + vacuum"],
      ["Neutralization",  "NeutParams",         "raffinate acid",               "gypsum, neutral effluent", "Agitator only"],
    ]
  ),

  H2("1.2 Flowsheet layer — mass-balance wiring"),
  P([
    new TextRun("flowsheet.py is the only place where Streams are routed between stages. It owns the four conservation rules (solids, water, F80, Cu) and the recycle algebra around the cleaner and the SX organic loop. It does "),
    new TextRun({ text: "not", italics: true }),
    new TextRun(" decide which stages to run — that is the decision layer's job. It only enforces that whatever topology is handed to it is wired in a way that closes mass."),
  ]),
  P([
    new TextRun({ text: "Topology gates. ", bold: true }),
    new TextRun("Before any physics call, topology.py checks the on/off bits against the §12 validity rules: flotation XOR heap, concentrate-leach requires a cleaner, SX requires a leach path producing PLS, EW requires SX, neutralization requires SX. Invalid combos return None and never reach flowsheet.py."),
  ]),
  makeTable(
    [
      { header: "Boundary", width: 2200 },
      { header: "What balances", width: 3600 },
      { header: "Gate / recycle rule", width: 3560 },
    ],
    [
      ["Crusher → SAG",         "solids_tph, water = 0",                       "Always on"],
      ["BM ↔ Cyclone",          "CL = 2.5 (UF returns to BM)",                 "cyclone_enabled forces CL>1"],
      ["Cyclone OF → Rougher",  "F80 = d50_OF, water from slurry %",           "flotation_enabled (always true unless heap)"],
      ["Rougher conc → Cleaner","Cu mass conserved; Wills recycle to g=26%",   "cleaner_enabled; n_cleaner_stages compounds gangue rejection"],
      ["Rougher tails",         "Cu and solids to thickener or heap",          "Routed to TSF; or to tank leach if leach_enabled"],
      ["Cleaner conc → POX",    "Cu_in = Cu_aq + Cu_residue",                  "leach_enabled requires cleaner_enabled"],
      ["PLS → SX → EW",         "Cu mass: aq → organic → cathode",             "sx_enabled + ew_enabled; bleed b=3%"],
      ["EW raffinate → Neut.",  "H2SO4 + Ca(OH)2 → CaSO4·2H2O",                "neutralization_enabled requires sx_enabled"],
      ["Heap pad → SX",         "Whole-ore Cu, no flotation in chain",         "heap_leach_enabled excludes flotation_enabled"],
    ]
  ),

  H2("1.3 Decision layer — topology + DE search"),
  P([
    new TextRun("optimizer.py drives an outer enumeration over valid topologies and an inner differential-evolution search over 10 continuous operating variables. The objective is −NPV at 8% discount; failed simulator calls are penalised at 1e12 so DE walks around infeasible regions without crashing. tea.py is the only consumer of the steady-state result: it turns Streams + kW + capex into revenue, opex, and cashflow."),
  ]),
  P([
    new TextRun({ text: "What the optimizer can and cannot move. ", bold: true }),
    new TextRun("Continuous knobs are the 10 box-bounded variables below. Discrete equipment choices (regrind mill type, cleaner stage count, cell volume) are set on Params and held fixed inside a single DE run; they vary across the outer topology enumeration. Prices, ore $/t, and discount rate are scenario inputs, not decision variables."),
  ]),
  makeTable(
    [
      { header: "Element", width: 2200 },
      { header: "Variable / action", width: 3600 },
      { header: "Bounds / source", width: 3560 },
    ],
    [
      ["Outer loop",            "Enumerate Topology() bit-vector",              "topology.all_topologies() filtered by _is_valid"],
      ["Inner loop",            "scipy.differential_evolution, 10-D",           "DECISION_VARS in optimizer.py"],
      ["sp_power",              "Flotation cell specific power (kW/m³)",        "0.8 – 1.2 (Wills Ch. 12)"],
      ["sp_gas_rate",           "Superficial gas velocity (cm/s)",              "0.8 – 1.5"],
      ["frother_conc",          "Frother dose (ppm)",                           "10 – 50"],
      ["flot_residence_time",   "Bank-total τ (min)",                           "15 – 20"],
      ["air_fraction",          "Cell air hold-up",                             "0.10 – 0.25"],
      ["slurry_fraction",       "Solids volume fraction",                       "0.15 – 0.30"],
      ["contact_angle",         "Effective hydrophobicity (°)",                 "20 – 55"],
      ["circulating_load",      "BM/cyclone CL",                                "1.5 – 4.0"],
      ["tailings_adverse_factor","TSF capex scalar",                            "1.0 – 5.0"],
      ["target_flot_P80_um",    "Grind target into rougher (µm)",               "75 – 200"],
      ["Objective",             "−NPV at 8% (TEA pre-tax annuity, 25 yr)",       "tea.evaluate(...)['npv']"],
      ["Feasibility",           "Topology gates + 1e12 penalty on sim failure", "topology._is_valid, optimizer._FAILURE_PENALTY"],
    ]
  ),
];

// Input→output table
const ioTable = [
  H1("2. End-to-end input → output"),
  makeTable(
    [
      { header: "Stage", width: 1700 },
      { header: "Inputs (what flows in)", width: 3500 },
      { header: "Outputs (what flows out)", width: 2400 },
      { header: "Drives in TEA", width: 1760 },
    ],
    [
      ["ROM ore", "tph, Cu grade, mineralogy, byproducts", "Feed Stream (F80=300mm)", "Ore purchase $/t"],
      ["Crusher", "Feed F80, target P80 (150mm)", "Coarser Stream, kW", "Power, capex (power-law)"],
      ["SAG + BM", "Wi, Ecs split, CL=2.5, target P80 (µm)", "Ground Stream, kW", "Power, mill capex"],
      ["Hydrocyclone", "Plitt geometry, slurry %", "UF/OF streams, d50, Rv", "Capex (small)"],
      ["Flotation (Ro)", "k from Blueshift port, τ, Rmax by mineralogy", "Conc + tails, R, kW", "Cell capex, power"],
      ["Regrind + Cleaner", "ΔP80, target g=26% Cu, recycle algebra", "Cleaner conc + middlings", "Capex, power"],
      ["Tank leach (POX)", "k(T), P_O2, t; pox_max ceiling", "Cu in PLS, residue", "O2, acid, autoclave capex"],
      ["Heap leach", "E, acid kg/t, pad geometry", "PLS, residue on pad", "Pad capex, acid opex"],
      ["SX–EW", "Moreno isotherm, Faraday CE, bleed b=3%", "Cathode tph, raffinate", "Reagents, EW power"],
      ["Dewater / Neut.", "Galvez flux, Ruth filter, lime stoich.", "Conc cake (9% H2O), gypsum", "Lime, capex"],
      ["TEA", "All streams, kW, capex, Cu/Mo/Au/Ag prices, TC/RC, As", "NPV, IRR, payback", "—"],
    ]
  ),
];

// Information flow narrative
const flow = [
  H1("3. How information flows"),
  P([
    new TextRun({ text: "Forward pass. ", bold: true }),
    new TextRun("Stream object carries (solids_tph, water_tph, F80, cu_grade, cu_aq_gpl, organic). Each enabled stage reads it, returns a new Stream and a kW draw. Disabled stages pass through at 0 power. Cu mass is conserved end-to-end (§13); closure is checked against cu_in − Σcu_out."),
  ]),
  P([
    new TextRun({ text: "Mineralogy-keyed lookups. ", bold: true }),
    new TextRun("The mineralogy string is the only ore descriptor that fans out to multiple stages: it sets flotation Rmax, POX/heap extraction ceilings, byproduct head grades, and concentrate As. This keeps the rest of the model ore-agnostic."),
  ]),
  P([
    new TextRun({ text: "Route choice. ", bold: true }),
    new TextRun("Topology validity rules (§12) are evaluated before each candidate flowsheet runs; invalid combos are dropped without a physics call. Of the three product routes (sulfide concentrate, concentrate leach to cathode, whole-ore heap to cathode) exactly one wins per scenario."),
  ]),
  P([
    new TextRun({ text: "TEA closure. ", bold: true }),
    new TextRun("Revenue uses payable Cu with smelter TC/RC and an As schedule; cathode routes skip TC/RC. Capex = Σ power-law equipment × Lang 5.0 + tailings + heap. Opex aggregates power, water, reagents, media, and O&M = 3.5% capex. Cashflow is a flat 25-yr annuity at 8%."),
  ]),
  P([
    new TextRun({ text: "Backward signal. ", bold: true }),
    new TextRun("Only the optimizer closes the loop. NPV is fed back to the DE search, which adjusts the 10 box-bounded operating variables (§11) and the topology bits. There is no analytic gradient — selection is by population search."),
  ]),
];

// Assumptions summary
const assumptions = [
  H1("4. Assumptions that shape the flow"),
  makeTable(
    [
      { header: "Area", width: 2200 },
      { header: "Choice", width: 4400 },
      { header: "Why", width: 2760 },
    ],
    [
      ["Scope", "Concentrator-only, ROM in, conc or cathode out", "Mine + G&A folded into ore $/t — keeps model size to one capex envelope"],
      ["Stage shape", "Pure functions Stream → Stream", "Composable, easy to enable/disable, no hidden state between stages"],
      ["Flotation bank", "One CSTR mixing stage at bank-total τ (not 1−(1−R)^N)", "Calibrates against disclosed plant recoveries better than the VBA cascade"],
      ["Cleaner circuit", "Wills recycle algebra, target g = 26% Cu", "Matches smelter-acceptable concentrate without iterating per cell"],
      ["Hydromet ceilings", "POX/heap caps by mineralogy", "Prevents chalcopyrite heap winning on bad kinetics"],
      ["Capex", "Power-law + Lang 5.0, no separate EPCM/contingency", "Lang already bundles installation; transparent single multiplier"],
      ["Cashflow", "Pre-tax flat annuity, 25 yr, 8%", "Strips tax/financing noise from route selection"],
      ["Optimizer", "DE, 10 box-bounded vars, no nonlinear constraints", "Robust on a non-smooth NPV surface; topology gates handle feasibility"],
    ]
  ),
];

// Why this structure
const why = [
  H1("5. Why this structure"),
  P([
    new TextRun({ text: "Separation of physics from economics. ", bold: true }),
    new TextRun("Stages output streams + kW; TEA maps streams + kW + capex to NPV. Either side can be re-fit (e.g., a new flotation model) without touching the other."),
  ]),
  P([
    new TextRun({ text: "Mass balance as the integrator. ", bold: true }),
    new TextRun("The Stream dataclass is the only contract between stages, so adding a new unit (e.g., dense-media separator) only needs a new Stream → Stream function plus a topology rule."),
  ]),
  P([
    new TextRun({ text: "Topology gates instead of penalty terms. ", bold: true }),
    new TextRun("Infeasible flowsheets are pruned before they cost a physics evaluation; the optimizer only sees the feasible NPV surface."),
  ]),
  P([
    new TextRun({ text: "Mineralogy as the single ore key. ", bold: true }),
    new TextRun("All ore-specific behavior (flotation max, POX/heap caps, byproducts, As) is keyed off one string so a project override propagates consistently."),
  ]),
];

// Appendix
const appendix = [
  new Paragraph({ children: [new PageBreak()] }),
  H1("Appendix A — Block diagram (ASCII)"),
  P("Reads top-to-bottom; branches show the three product routes; (*) marks always-on stages."),
  new Paragraph({
    spacing: { after: 40 },
    children: [new TextRun({
      text: [
"  ROM ore  (tph, g_Cu, mineralogy)",
"      |",
"  [Crusher*] --kW--+",
"      |            |",
"  [SAG + BM*] --kW-+",
"      |            |",
"  [Hydrocyclone]   |",
"      | OF         |",
"  [Rougher flot*] -+--> tails ---------------------+",
"      | conc       |                               |",
"      +--(A)-- sulfide route                       |",
"      +--(B)-- concentrate-leach route             |",
"                                                   |",
"  (C) heap-leach route: skip BM/cyclone/cleaner ---+",
"",
"  (A) [Regrind] -> [Cleaner] -> [Thickener] -> [Filter] -> Cu concentrate",
"  (B) [Regrind] -> [Cleaner] -> [POX leach] -> [Neut] -> [SX] -> [EW] -> Cathode",
"  (C) ROM -> [Heap pad] -> PLS -> [SX] -> [EW] -> Cathode",
"",
"  All routes -> [TEA] -> {Revenue, Opex, Capex, NPV, IRR}",
"  Optimizer (DE) <-- NPV -- adjusts topology bits + 10 operating vars",
      ].join("\n"),
      font: "Consolas",
      size: 16,
    })],
  }),
  H2("Appendix B — Stream contract"),
  P("Stream = (solids_tph, water_tph, F80_um, cu_grade, cu_aq_gpl, organic_tph, cu_org_gpl). Derived properties: solids_pct, cv (volume % solids @ ρs=2.7). Anything outside this tuple is stage-local and never crosses a stage boundary."),
];

// ---------- Assemble ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Calibri" },
        paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
      },
    },
    children: [
      title, subtitle, intro,
      ...structure,
      ...ioTable,
      ...flow,
      ...assumptions,
      ...why,
      ...appendix,
    ],
  }],
});

const out = path.join(__dirname, "INFORMATION_FLOW.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
