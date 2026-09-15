"""Designer-facing flowsheet recommendation report (derisking PS-4, v0 surface).

Runs the topology-selecting optimizer for a feed scenario, re-simulates the
winning flowsheet at its optimum, and writes ONE self-contained HTML file a
process designer can read end-to-end: recommended flowsheet diagram, equipment
counts and sizing, operating setpoints, stream table, economics, and a live
validation/limitations block (derisking PS-2 groundwork — the same M1/M2
harness numbers as benchmark_scorecard, computed fresh at report time).

CLI:
    python -m process_model.report --grade 0.008 --tph 2000
    python -m process_model.report --grade 0.008 --tph 2000 --quick
Options:
    --maxiter / --popsize  inner DE budget per topology (default 60x12)
    --quick                25x8 DE budget (previewing the report, not deciding)
    --no-validation        skip the live M1/M2 section (~1-2 min faster)
    -o / --out             output path (default process_model/reports/
                           recommendation_<grade>pct_<tph>tph.html)
"""
from __future__ import annotations

import argparse
import base64
import datetime
import io
import os

from .optimizer import Optimizer, DECISION_VARS, _apply
from .topology import topology_label
from .throughput import CuSulfideParams, simulate_cu_sulfide
from .tea import TEAParams, evaluate


# Display metadata for the DE decision variables (units + plain-English name).
_VAR_DISPLAY = {
    "sp_power":                ("Specific power", "kW/m3"),
    "sp_gas_rate":             ("Superficial gas rate (Jg)", "cm/s"),
    "frother_conc":            ("Frother concentration", "ppm"),
    "flot_residence_time_min": ("Rougher residence time", "min"),
    "air_fraction":            ("Cell air fraction (gas holdup)", "frac"),
    "slurry_fraction":         ("Pulp density (vol. solids/slurry)", "frac"),
    "target_flot_P80_um":      ("Flotation feed P80", "um"),
}

# Fixed (non-optimized) parameters a designer should still see.
_FIXED_PARAMS = [
    ("Contact angle (chalcopyrite-xanthate)", "55 deg",
     "mineral-surface property, not a design dial"),
    ("Circulating load (BM-cyclone)", "2.5",
     "industry standard; fixed, benefit not representable in DOE"),
    ("Rougher cell volume", "500 m3",
     "discrete equipment selection (TankCell e500 class)"),
]

_LIMITATIONS = [
    "TEA is calibrated for RELATIVE comparison between flowsheets, not "
    "absolute project valuation. Scope is concentrator-only: tailings-dam "
    "construction beyond the favorable-site minimum, EPCM/contingency, and "
    "owners' infrastructure are excluded (~$0.5B at 50 ktpd vs full-project "
    "benchmarks).",
    "Validated on Cu porphyry/chalcocite/oxide systems (10 disclosed plants). "
    "Known structural miss: concentrate-POX routes are under-selected (El "
    "Abra picks heap) — treat POX-vs-heap recommendations with caution.",
    "Recovery model is validated at the reference operating window; the "
    "scavenger bank has no separate capex line (cells fold into the rougher "
    "count), and collector dosing is not yet modeled (frother only).",
    "Ore is treated as purchased merchant feed; mining economics are out of "
    "scope.",
]


def _fmt(v, kind="num"):
    if v is None:
        return "n/a"
    if kind == "usd":
        return f"${v:,.0f}"
    if kind == "usdB":
        return f"${v/1e9:,.2f}B"
    if kind == "pct":
        return f"{v*100:.1f}%"
    return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"


def _table(headers, rows):
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def _flowsheet_png_b64(topo, sim) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .visualize import plot_flowsheet
    fig = plot_flowsheet(topology=topo, sim_result=sim)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _validation_section() -> str:
    from .benchmarks import run_m1_route_selection, run_m2_metallurgy
    m1 = run_m1_route_selection(verbose=False)
    m1_score = sum(1 for r in m1["rows"] if r["match"])
    m1_total = len(m1["rows"])
    m2 = run_m2_metallurgy(verbose=False)
    n = len(m2["rows"])
    mean_dr = sum(r["dR_pp"] for r in m2["rows"]) / n
    worst = max(m2["rows"], key=lambda r: r["dR_pp"])
    med_dg = m2["median_dG_pp"]
    m1_ok = "PASS" if m1_score >= 8 else "FAIL"
    m2_ok = "PASS" if mean_dr <= 5.0 else "FAIL"
    dg = f"{med_dg:.1f}pp" if med_dg is not None else "n/a"
    return f"""
    <div class="kpis">
      <div class="kpi {m1_ok.lower()}"><div class="big">{m1_score}/{m1_total}</div>
        route selection vs disclosed plants <span>({m1_ok}, threshold 8/10)</span></div>
      <div class="kpi {m2_ok.lower()}"><div class="big">{mean_dr:.1f}pp</div>
        mean Cu-recovery error over {n} disclosed plants <span>({m2_ok},
        threshold 5pp; worst {worst['plant']} {worst['dR_pp']:.1f}pp;
        median grade error {dg})</span></div>
    </div>
    <p class="note">Computed live from the benchmark harness at report time
    (same numbers as <code>benchmark_scorecard</code>).</p>"""


def build_report(feed_tph: float, feed_grade: float, maxiter: int = 60,
                 popsize: int = 12, out_path: str | None = None,
                 with_validation: bool = True, seed: int = 42) -> str:
    """Run the optimizer, re-simulate the winner, write the HTML report.

    Returns the output path."""
    opt = Optimizer(feed_tph=feed_tph, feed_grade=feed_grade)
    res = opt.run(maxiter=maxiter, popsize=popsize)
    ranked = sorted(res.results, key=lambda r: r.best_npv, reverse=True)
    win = ranked[0]

    # Re-simulate the winner at its optimum for streams/equipment/economics.
    p, t = _apply(win.best_x, CuSulfideParams(), TEAParams())
    sim = simulate_cu_sulfide(feed_tph, feed_grade, p, topology=win.topology)
    tea_res = evaluate(sim, t)
    bd = tea_res["breakdown"]
    bal = sim["balance"]
    flot = sim["flotation"]
    cleaner = sim.get("cleaner") or {}
    thick = sim.get("conc_thickener") or {}
    filt = sim.get("belt_filter") or {}

    # -- Section: leaderboard (top 5) --
    lb_rows = [(i + 1, r.label, _fmt(r.best_npv, "usdB"),
                f"{r.irr*100:.0f}%" if r.irr is not None else "n/a",
                _fmt(r.capex, "usdB"), _fmt(r.cu_recovery, "pct"),
                f"{r.conc_grade_pct:.1f}%")
               for i, r in enumerate(ranked[:5])]
    leaderboard = _table(
        ["#", "Flowsheet", "NPV", "IRR", "Capex", "Cu recovery", "Conc grade"],
        lb_rows)

    # -- Section: operating setpoints --
    sp_rows = []
    for i, (name, lo, hi) in enumerate(DECISION_VARS):
        label, unit = _VAR_DISPLAY.get(name, (name, ""))
        sp_rows.append((label, f"{float(win.best_x[i]):.2f} {unit}",
                        f"{lo:g}-{hi:g} {unit}"))
    setpoints = _table(["Operating variable (optimized)", "Value",
                        "Allowed range"], sp_rows)
    fixed = _table(["Fixed parameter", "Value", "Why fixed"], _FIXED_PARAMS)

    # -- Section: equipment & sizing --
    eq_rows = [
        ("Rougher flotation cells",
         f"{flot.get('num_cells', 0)} x {_fmt(flot.get('cell_volume_m3', 0))} m3 "
         f"({_fmt(flot.get('total_volume_m3', 0))} m3 bank)",
         f"{sim['power_kw'].get('flotation_kw', 0):,.0f} kW"),
        ("Hydrocyclones", f"{bd.get('n_cyclone', 0)} units "
         f"({bd.get('cyclone_feed_m3_per_h', 0):,.0f} m3/h feed)", ""),
        ("SAG mill", f"{bd.get('n_sag', 0)} unit",
         f"{sim['power_kw'].get('sag_kw', 0):,.0f} kW"),
        ("Ball mill", f"{bd.get('n_bm', 0)} unit",
         f"{sim['power_kw'].get('bm_kw', 0):,.0f} kW"),
        ("Concentrate thickener",
         f"{thick.get('diameter_m', bd.get('thickener_diameter_m', 0)):,.0f} m dia.", ""),
        ("Tails thickener",
         f"{bd.get('tails_thickener_diameter_m', 0):,.0f} m dia.", ""),
        ("Belt filter",
         f"{filt.get('area_m2', bd.get('filter_area_m2', 0)):,.0f} m2", ""),
    ]
    if cleaner.get("n_stages"):
        eq_rows.insert(1, ("Cleaner flotation",
                           f"{cleaner['n_stages']} stages, "
                           f"{cleaner.get('num_cells', 0)} x "
                           f"{_fmt(cleaner.get('cell_volume_m3', 0))} m3 cells", ""))
    equipment = _table(["Equipment", "Count / size", "Installed power"], eq_rows)

    # -- Section: streams --
    st_rows = []
    for name, s in sim["streams"].items():
        tot = s.solids_tph + s.water_tph
        pct = 100.0 * s.solids_tph / tot if tot else 0.0
        st_rows.append((name, f"{s.solids_tph:,.0f}", f"{s.water_tph:,.0f}",
                        f"{pct:.0f}%", f"{s.cu_grade*100:.2f}%",
                        f"{s.F80_um:,.0f}" if s.F80_um else ""))
    streams = _table(["Stream", "Solids tph", "Water tph", "% solids",
                      "Cu grade", "F80 um"], st_rows)

    # -- Section: economics --
    econ_head = _table(["NPV (25y, 8%)", "IRR", "Installed capex",
                        "Annual revenue (net)", "Annual opex", "Payback"], [(
        _fmt(tea_res["npv"], "usdB"),
        f"{tea_res['irr']*100:.0f}%" if tea_res["irr"] and
        tea_res["irr"] == tea_res["irr"] else "n/a",
        _fmt(tea_res["total_capex"], "usdB"),
        _fmt(tea_res["annual_revenue"], "usd"),
        _fmt(tea_res["annual_opex"], "usd"),
        f"{tea_res['payback_years']:.1f} yr"
        if tea_res["payback_years"] not in (None, float("inf")) else "n/a")])
    capex_rows = [(k.replace("_capex", "").replace("_", " "),
                   _fmt(v, "usd")) for k, v in bd.items()
                  if k.endswith("_capex") and v]
    opex_keys = ["power_cost", "water_cost", "reagent_cost", "media_cost",
                 "ore_purchase_cost", "om_opex", "hydromet_opex"]
    opex_rows = [(k.replace("_cost", "").replace("_opex", " O&M/other")
                  .replace("_", " "), _fmt(bd[k], "usd"))
                 for k in opex_keys if bd.get(k)]
    econ_detail = ("<div class='cols'><div><h3>Capex lines (installed)</h3>" +
                   _table(["Line", "USD"], capex_rows) +
                   "</div><div><h3>Opex lines (USD/yr)</h3>" +
                   _table(["Line", "USD/yr"], opex_rows) + "</div></div>")

    flowsheet_b64 = _flowsheet_png_b64(win.topology, sim)
    validation = _validation_section() if with_validation else \
        "<p class='note'>Live validation skipped (--no-validation).</p>"
    limitations = "".join(f"<li>{x}</li>" for x in _LIMITATIONS)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Flowsheet recommendation — {feed_grade*100:.2f}% Cu @ {feed_tph:,.0f} tph</title>
<style>
 body {{ font-family: Arial, Helvetica, sans-serif; margin: 2em auto;
        max-width: 1100px; color: #1a1a1a; }}
 h1 {{ font-size: 1.5em; }} h2 {{ font-size: 1.15em; margin-top: 1.6em;
        border-bottom: 2px solid #1F77B4; padding-bottom: 4px; }}
 table {{ border-collapse: collapse; margin: 0.6em 0; font-size: 0.92em; }}
 th, td {{ border: 1px solid #ccc; padding: 5px 10px; text-align: left; }}
 th {{ background: #eef3f8; }}
 .kpis {{ display: flex; gap: 1em; }}
 .kpi {{ border: 2px solid #2CA02C; border-radius: 8px; padding: 0.8em 1.2em;
        flex: 1; }} .kpi.fail {{ border-color: #D62728; }}
 .kpi .big {{ font-size: 1.8em; font-weight: bold; }}
 .kpi span {{ color: #555; font-size: 0.85em; }}
 .note {{ color: #555; font-size: 0.85em; }}
 .cols {{ display: flex; gap: 2em; }} .cols > div {{ flex: 1; }}
 img.flowsheet {{ max-width: 100%; border: 1px solid #ddd; }}
 .banner {{ background: #fff8e1; border: 1px solid #e0c060; padding: 0.7em 1em;
        border-radius: 6px; margin: 1em 0; font-size: 0.9em; }}
</style></head><body>
<h1>Flowsheet recommendation — {feed_grade*100:.2f}% Cu chalcopyrite @
{feed_tph:,.0f} tph</h1>
<p class="note">Generated {now} · physics-informed process model ·
DE budget {maxiter}x{popsize}, seed {seed} · topology enumerated over
{len(ranked)} archetypes</p>
<div class="banner"><b>Decision-support output.</b> Economics are for
relative flowsheet comparison, not absolute valuation — see Limitations.</div>

<h2>Recommended flowsheet</h2>
<p><b>{win.label}</b> — NPV {_fmt(win.best_npv, 'usdB')} ·
IRR {f"{win.irr*100:.0f}%" if win.irr is not None else "n/a"} ·
capex {_fmt(win.capex, 'usdB')} · Cu recovery {_fmt(win.cu_recovery, 'pct')} ·
concentrate grade {win.conc_grade_pct:.1f}% Cu
(overall recovery incl. dewatering {_fmt(bal['overall_cu_recovery'], 'pct')})</p>
<img class="flowsheet" src="data:image/png;base64,{flowsheet_b64}"
 alt="flowsheet diagram">

<h2>Alternatives considered</h2>{leaderboard}

<h2>Operating setpoints</h2>{setpoints}
<h3>Fixed parameters</h3>{fixed}

<h2>Equipment &amp; sizing</h2>{equipment}

<h2>Stream table</h2>{streams}

<h2>Economics</h2>{econ_head}{econ_detail}

<h2>Model validation (live)</h2>{validation}

<h2>Limitations</h2><ul>{limitations}</ul>
</body></html>"""

    if out_path is None:
        out_path = os.path.join(
            "process_model", "reports",
            f"recommendation_{feed_grade*100:.2f}pct_{feed_tph:.0f}tph.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out_path}")
    print(f"  winner: {win.label}")
    print(f"  NPV {_fmt(win.best_npv, 'usdB')} · capex {_fmt(win.capex, 'usdB')}"
          f" · R {_fmt(win.cu_recovery, 'pct')} · conc {win.conc_grade_pct:.1f}%")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grade", type=float, default=0.008,
                    help="Cu head grade as a fraction (default 0.008)")
    ap.add_argument("--tph", type=float, default=2000.0)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--quick", action="store_true",
                    help="25x8 DE budget for a fast preview")
    ap.add_argument("--no-validation", action="store_true",
                    help="skip the live M1/M2 validation section")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args(argv)
    maxiter = 25 if args.quick else args.maxiter
    popsize = 8 if args.quick else args.popsize
    build_report(args.tph, args.grade, maxiter=maxiter, popsize=popsize,
                 out_path=args.out, with_validation=not args.no_validation)


if __name__ == "__main__":
    main()
