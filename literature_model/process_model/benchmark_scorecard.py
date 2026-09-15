"""Validation scorecard — visual before/after of the success metrics.

Renders a single figure showing where the model's M1 (route selection)
and M2 (recovery agreement) benchmarks stood before the 2026-05-19 SX-EW
closed-circuit recycle closure versus now. Intended as a progress
artifact, not part of the model itself.

CLI: `python -m process_model.benchmark_scorecard`
  -> process_model/figures/benchmark_scorecard.png
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from .benchmarks import run_m1_route_selection, run_m2_metallurgy

# "Before" = documented reading immediately prior to the SX-EW recycle
# closure (PROGRESS.md, 2026-05-07d header / 2026-05-06b M2 reading).
# Per-plant ΔR captured from the pre-change benchmark run on 2026-05-19.
BEFORE_M1_SCORE = 7          # of 10
BEFORE_M2_MEDIAN_DR = 6.83   # percentage points
BEFORE_M2_DR = {
    "Cobre Panama": 2.19, "Escondida concentrator": 2.19,
    "Cerro Verde": 3.19, "El Abra POX": 9.26,
    "Sherritt Bagdad pilot": 10.26, "Morenci POX (Freeport)": 8.26,
    "Spence heap": 9.83, "Lomas Bayas heap": 6.83, "Mantoverde heap": 2.80,
}

_BEFORE = "#9AA7B4"   # muted grey-blue
_AFTER = "#2CA02C"    # green
_FAIL = "#D62728"

# --- Headline KPI figures ------------------------------------------------
# Both headline KPIs are computed LIVE from the benchmark harness so the tiles
# are reproducible and self-updating. The metallurgical error reported is the
# MEAN |Cu recovery error| across all disclosed M2 plants — deliberately the
# mean rather than the median, so single-plant outliers (e.g. Mantoverde heap)
# are NOT hidden. (A prior version hardcoded 5.8% here, a figure that could not
# be reproduced from the harness; that override has been removed.)
METALLURGICAL_ERROR_THRESHOLD_PCT = 5.0   # pass <= this (standing M2 bar, pp)


def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })


def render(out_path: str = "process_model/figures/benchmark_scorecard.png"):
    _style()
    m1 = run_m1_route_selection(verbose=False)
    m2 = run_m2_metallurgy(verbose=False)
    now_m1 = sum(1 for r in m1["rows"] if r["match"])
    now_m2_dr = m2["median_dR_pp"]
    now_dr = {r["plant"]: r["dR_pp"] for r in m2["rows"]}

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(15, 5.2),
        gridspec_kw={"width_ratios": [1, 1, 2.1]})
    fig.suptitle("Process model validation scorecard — "
                 "SX-EW closed-circuit recycle closure (2026-05-19)",
                 fontsize=13, fontweight="bold")

    # --- Panel A: M1 route-selection score ---------------------------
    bars = axA.bar(["Before", "Now"], [BEFORE_M1_SCORE, now_m1],
                   color=[_BEFORE, _AFTER], width=0.6)
    axA.axhline(8, color=_FAIL, linestyle="--", linewidth=1.3)
    axA.text(-0.45, 8.15, "pass = 8/10", color=_FAIL, fontsize=9,
             ha="left", va="bottom")
    axA.set_ylim(0, 10)
    axA.set_ylabel("projects routed correctly (of 10)")
    axA.set_title("M1 — route selection")
    for b, v, ok in zip(bars, [BEFORE_M1_SCORE, now_m1],
                        [BEFORE_M1_SCORE >= 8, now_m1 >= 8]):
        axA.text(b.get_x() + b.get_width() / 2, v + 0.15,
                 f"{v}/10\n{'PASS' if ok else 'FAIL'}",
                 ha="center", va="bottom", fontweight="bold",
                 color=_AFTER if ok else _FAIL)

    # --- Panel B: M2 median recovery error ---------------------------
    bars = axB.bar(["Before", "Now"], [BEFORE_M2_MEDIAN_DR, now_m2_dr],
                   color=[_BEFORE, _AFTER], width=0.6)
    axB.set_ylim(0, 8)
    axB.axhline(5, color=_FAIL, linestyle="--", linewidth=1.3)
    axB.text(-0.45, 5.15, "pass <= 5pp", color=_FAIL, fontsize=9,
             ha="left", va="bottom")
    axB.set_ylabel("median |Cu recovery error| (pp)")
    axB.set_title("M2 — recovery agreement (lower is better)")
    for b, v, ok in zip(bars, [BEFORE_M2_MEDIAN_DR, now_m2_dr],
                        [BEFORE_M2_MEDIAN_DR <= 5, now_m2_dr <= 5]):
        axB.text(b.get_x() + b.get_width() / 2, v + 0.15,
                 f"{v:.2f}pp\n{'PASS' if ok else 'FAIL'}",
                 ha="center", va="bottom", fontweight="bold",
                 color=_AFTER if ok else _FAIL)

    # --- Panel C: per-plant recovery error before/after --------------
    plants = [r["plant"] for r in m2["rows"]]
    before = [BEFORE_M2_DR.get(p, 0.0) for p in plants]
    after = [now_dr[p] for p in plants]
    y = range(len(plants))
    h = 0.38
    axC.barh([i + h / 2 for i in y], before, height=h,
             color=_BEFORE, label="Before")
    axC.barh([i - h / 2 for i in y], after, height=h,
             color=_AFTER, label="Now")
    axC.axvline(5, color=_FAIL, linestyle="--", linewidth=1.3)
    axC.text(5.1, len(plants) - 0.4, "5pp threshold", color=_FAIL,
             fontsize=9, va="top")
    axC.set_yticks(list(y))
    axC.set_yticklabels(plants)
    axC.invert_yaxis()
    axC.set_xlabel("|Cu recovery error| vs disclosed plant (pp)")
    axC.set_title("M2 — per-plant recovery error")
    axC.legend(loc="lower right", frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  M1: {BEFORE_M1_SCORE}/10 -> {now_m1}/10")
    print(f"  M2 median |dR|: {BEFORE_M2_MEDIAN_DR}pp -> {now_m2_dr:.2f}pp")


def _kpi_tile(ax, value_text, label, sub, ok):
    """Draw one big KPI tile on a borderless axis."""
    edge = _AFTER if ok else _FAIL
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # rounded background panel
    panel = plt.matplotlib.patches.FancyBboxPatch(
        (0.04, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=2.4, edgecolor=edge, facecolor="#F5FBF5" if ok else "#FCF3F3",
        mutation_aspect=1.0, transform=ax.transAxes)
    ax.add_patch(panel)
    ax.text(0.5, 0.62, value_text, ha="center", va="center",
            fontsize=58, fontweight="bold", color=edge, transform=ax.transAxes)
    ax.text(0.5, 0.34, label, ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333333",
            transform=ax.transAxes)
    ax.text(0.5, 0.20, sub, ha="center", va="center",
            fontsize=11, color="#555555", transform=ax.transAxes)
    ax.text(0.5, 0.115, "PASS" if ok else "FAIL", ha="center", va="center",
            fontsize=13, fontweight="bold", color=edge, transform=ax.transAxes)


def render_headline(out_path: str = "process_model/figures/headline_scorecard.png"):
    """Two-KPI headline, both computed LIVE: M1 route-selection score and the
    aggregate Cu recovery agreement error (mean |dR| across disclosed plants),
    each against its pass threshold."""
    _style()
    m1 = run_m1_route_selection(verbose=False)
    now_m1 = sum(1 for r in m1["rows"] if r["match"])
    total_m1 = len(m1["rows"])
    m1_ok = now_m1 >= 8

    # Live metallurgical error: mean |dR| over all M2 plants (conservative —
    # keeps outliers visible), with the worst plant called out for honesty.
    m2 = run_m2_metallurgy(verbose=False)
    n_plants = len(m2["rows"])
    met_err = sum(r["dR_pp"] for r in m2["rows"]) / n_plants
    worst = max(m2["rows"], key=lambda r: r["dR_pp"])
    med_dG = m2["median_dG_pp"]
    met_ok = met_err <= METALLURGICAL_ERROR_THRESHOLD_PCT

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.0))
    fig.suptitle("Process model validation — route selection & metallurgical agreement",
                 fontsize=14, fontweight="bold")

    _kpi_tile(axL, f"{now_m1}/{total_m1}", "M1 — route selection",
              "disclosed plants routed correctly  (pass ≥ 8/10)", m1_ok)
    dG_str = f"{med_dG:.1f}pp" if med_dG is not None else "n/a"
    _kpi_tile(axR, f"{met_err:.1f}pp",
              "Cu recovery agreement error",
              f"mean |recovery error| over {n_plants} disclosed plants  "
              f"(pass ≤ {METALLURGICAL_ERROR_THRESHOLD_PCT:.0f}pp)\n"
              f"worst: {worst['plant']} {worst['dR_pp']:.1f}pp · "
              f"median |grade err| {dG_str}", met_ok)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  M1: {now_m1}/{total_m1} ({'PASS' if m1_ok else 'FAIL'})")
    print(f"  metallurgical agreement error (mean |dR|): {met_err:.2f}pp "
          f"({'PASS' if met_ok else 'FAIL'}); worst {worst['plant']} "
          f"{worst['dR_pp']:.2f}pp")


if __name__ == "__main__":
    render()
    render_headline()
