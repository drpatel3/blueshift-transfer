"""Visualizations for the topology-selecting optimizer.

Four charts:
    1. plot_leaderboard         — NPV bar chart across topologies
    2. plot_capex_breakdown     — stacked bar of capex line items
    3. plot_grade_sensitivity   — NPV/IRR vs feed Cu grade sweep
    4. plot_operating_point     — where each continuous param landed within bounds

CLI: `python -m process_model.visualize` runs all four and saves PNGs to
`process_model/figures/`.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .optimizer import (
    Optimizer, OptimizationResult, TopologyResult,
    BOUNDS, NAMES, DECISION_VARS, _apply,
)
from .topology import Topology, all_topologies, topology_label
from .throughput import simulate_cu_sulfide, CuSulfideParams, REGRIND_MILL_PROPS


def _regrind_label(topo: "Topology") -> str:
    """Flowsheet label for the regrind mill, varies by mill type."""
    if not topo.regrind_enabled:
        return "Regrind\nmill"
    return REGRIND_MILL_PROPS.get(topo.regrind_mill_type,
                                   REGRIND_MILL_PROPS["ball"])["label"]
from .tea import TEAParams, evaluate


# Anthropic-ish flat palette — readable on white, print-safe
_PALETTE = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
            "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]


def _style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
    })


# ------------------------------------------------------------------ 1. leaderboard
def plot_leaderboard(opt_result: OptimizationResult,
                     save: Optional[str] = None,
                     show: bool = False) -> plt.Figure:
    """Horizontal bar chart of NPV by topology, sorted high-to-low."""
    _style()
    rows = sorted(opt_result.results, key=lambda r: r.best_npv)
    labels = [r.label for r in rows]
    npvs = np.array([r.best_npv / 1e9 for r in rows])
    irrs = [r.irr for r in rows]
    colors = ["#2CA02C" if v > 0 else "#D62728" for v in npvs]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels, npvs, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("NPV ($B, 25-yr DCF @ 8%)")
    ax.set_title(
        f"Topology leaderboard — {opt_result.feed_tph:.0f} tph @ "
        f"{opt_result.feed_grade*100:.2f}% Cu"
    )
    # Annotate IRR next to each bar
    for bar, irr, npv in zip(bars, irrs, npvs):
        x = bar.get_width()
        irr_s = f"IRR {irr*100:.0f}%" if irr is not None else "IRR n/a"
        ha = "left" if x >= 0 else "right"
        offset = 0.15 if x >= 0 else -0.15
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                irr_s, va="center", ha=ha, fontsize=8.5,
                color="dimgray")
    ax.margins(x=0.18)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ------------------------------------------------------------------ 2. capex breakdown
_CAPEX_LINES = [
    ("Crusher (gyratory)",        "crusher_capex",            "#1F77B4"),
    ("Crushing plant",            "crusher_plant_capex",      "#AEC7E8"),
    ("SAG mill",                  "sag_capex",                "#FF7F0E"),
    ("Ball mill",                 "bm_capex",                 "#FFBB78"),
    ("Processing section",        "processing_section_capex", "#2CA02C"),
    ("Tailings",                  "tailings_capex",           "#98DF8A"),
    ("Other",                     "other_capex",              "#7F7F7F"),
]


def plot_capex_breakdown(tea_result: dict,
                         topology_label_text: str = "",
                         save: Optional[str] = None,
                         show: bool = False) -> plt.Figure:
    """Stacked-bar capex breakdown by O'Hara line item."""
    _style()
    bd = tea_result["breakdown"]
    items = [(label, bd.get(key, 0.0) / 1e6, color)
             for label, key, color in _CAPEX_LINES]
    items_nz = [(l, v, c) for l, v, c in items if v > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = 0.0
    for label, val, color in items_nz:
        ax.bar([0], [val], bottom=[bottom], color=color, label=label,
               edgecolor="white", linewidth=0.6)
        # Inline labels for sizable slices
        if val > 5:
            ax.text(0, bottom + val / 2, f"{label}\n${val:.1f}M",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if val > 30 else "black")
        bottom += val
    ax.set_xticks([])
    ax.set_ylabel("Installed capex ($M, 2025 USD)")
    title = "Capex breakdown — O'Hara 1992 SME Ch. 6.3"
    if topology_label_text:
        title += f"\n{topology_label_text}"
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=9)
    ax.text(0, bottom + 8, f"Total: ${bottom:.1f}M",
            ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, bottom * 1.12)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ------------------------------------------------------------------ 3. grade sensitivity
def plot_grade_sensitivity(grades: Optional[list[float]] = None,
                           feed_tph: float = 2000.0,
                           topology: Optional[Topology] = None,
                           save: Optional[str] = None,
                           show: bool = False) -> plt.Figure:
    """NPV and IRR vs feed Cu grade, holding everything else fixed."""
    _style()
    grades = grades if grades is not None else [
        0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020,
    ]
    topo = topology or Topology()
    base = CuSulfideParams()
    econ = TEAParams()

    npvs, irrs, recs, grades_pct = [], [], [], []
    for g in grades:
        sim = simulate_cu_sulfide(feed_tph, g, base, topology=topo)
        res = evaluate(sim, econ)
        npvs.append(res["npv"] / 1e9)
        irrs.append(res["irr"] * 100 if res["irr"] == res["irr"] else np.nan)
        recs.append(sim["balance"]["overall_cu_recovery"] * 100)
        grades_pct.append(g * 100)

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    color_npv = _PALETTE[0]
    color_irr = _PALETTE[1]
    ax1.plot(grades_pct, npvs, "-o", color=color_npv, label="NPV",
             linewidth=2, markersize=6)
    ax1.set_xlabel("Feed Cu grade (%)")
    ax1.set_ylabel("NPV ($B)", color=color_npv)
    ax1.tick_params(axis="y", labelcolor=color_npv)
    ax1.axhline(0, color="black", linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(grades_pct, irrs, "-s", color=color_irr, label="IRR",
             linewidth=2, markersize=6)
    ax2.set_ylabel("IRR (%)", color=color_irr)
    ax2.tick_params(axis="y", labelcolor=color_irr)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    ax1.set_title(f"NPV / IRR vs feed grade — {feed_tph:.0f} tph, "
                  f"{topology_label(topo)}")
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ------------------------------------------------------------------ 4. operating-point bounds
def plot_operating_point(tr: TopologyResult,
                         save: Optional[str] = None,
                         show: bool = False) -> plt.Figure:
    """Show where each continuous decision variable landed within its bounds.

    Helps diagnose whether DE pushed against the bounds (signal that the
    bound itself is the binding constraint) or settled in the interior.
    """
    _style()
    n = len(NAMES)
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * n + 1.8))
    y = np.arange(n)
    for i, (name, lo, hi) in enumerate(DECISION_VARS):
        x = tr.best_x[i]
        norm = (x - lo) / (hi - lo) if hi > lo else 0.5
        # Bound bar (gray track)
        ax.barh(i, 1.0, left=0.0, color="#E5E5E5", edgecolor="#CCCCCC",
                height=0.55)
        # Optimum marker
        marker_color = "#D62728" if norm < 0.02 or norm > 0.98 else "#2CA02C"
        ax.scatter([norm], [i], s=90, color=marker_color, zorder=3,
                   edgecolor="white", linewidth=1.2)
        # Annotations: name on left, value+units on right
        ax.text(-0.04, i, name, ha="right", va="center", fontsize=9)
        ax.text(1.04, i, f"{x:.3f}", ha="left", va="center", fontsize=9,
                color="dimgray")
        ax.text(-0.04, i - 0.32, f"[{lo:.2f}]", ha="right", va="center",
                fontsize=7.5, color="gray")
        ax.text(1.04, i - 0.32, f"[{hi:.2f}]", ha="left", va="center",
                fontsize=7.5, color="gray")
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["min", "", "mid", "", "max"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()
    ax.set_title(
        f"Optimum within bounds — {tr.label}\n"
        f"NPV ${tr.best_npv/1e9:.2f}B, IRR "
        f"{tr.irr*100:.1f}%" if tr.irr is not None else
        f"NPV ${tr.best_npv/1e9:.2f}B"
    )
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    # Legend chip
    ax.scatter([], [], s=80, color="#2CA02C", label="interior optimum")
    ax.scatter([], [], s=80, color="#D62728", label="hits bound")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ------------------------------------------------------------------ 5. param sensitivity grid
def plot_param_sensitivity_grid(tr: TopologyResult,
                                topo: Topology,
                                feed_tph: float = 2000.0,
                                feed_grade: float = 0.008,
                                n_samples: int = 21,
                                save: Optional[str] = None,
                                show: bool = False) -> plt.Figure:
    """Small-multiples grid: hold all params at the optimum, sweep one,
    plot NPV. One subplot per decision variable."""
    _style()
    base_x = tr.best_x.copy()
    base_params = CuSulfideParams()
    tea_base = TEAParams()

    # 3 rows × 4 cols accommodates the 11 vars; hide trailing unused subplots
    n_vars = len(DECISION_VARS)
    ncols = 4
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows),
                             sharey=False)
    axes = axes.flatten()
    # Hide extra subplots beyond n_vars
    for j in range(n_vars, len(axes)):
        axes[j].axis("off")

    for i, (name, lo, hi) in enumerate(DECISION_VARS):
        ax = axes[i]
        sweep = np.linspace(lo, hi, n_samples)
        npvs = np.empty(n_samples)
        for k, v in enumerate(sweep):
            x = base_x.copy()
            x[i] = v
            p, t = _apply(x, base_params, tea_base)
            sim = simulate_cu_sulfide(feed_tph, feed_grade, p, topology=topo)
            res = evaluate(sim, t)
            npvs[k] = res["npv"] / 1e9

        ax.plot(sweep, npvs, "-", color=_PALETTE[0], linewidth=1.8)
        # Mark the optimum
        ax.axvline(base_x[i], color="#D62728", linewidth=1.2,
                   linestyle="--", alpha=0.7)
        ax.scatter([base_x[i]], [tr.best_npv / 1e9], s=60,
                   color="#D62728", zorder=3, edgecolor="white", linewidth=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(lo, hi)
        ax.tick_params(labelsize=8)
        ax.set_xlabel(f"[{lo:.2f}, {hi:.2f}]", fontsize=8, color="gray")
        if i % ncols == 0:
            ax.set_ylabel("NPV ($B)", fontsize=9)

    fig.suptitle(f"Single-variable sensitivity around optimum — {tr.label}",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ------------------------------------------------------------------ 6. DE convergence
def plot_de_convergence(opt: Optimizer, topo: Topology,
                        maxiter: int = 80, popsize: int = 12,
                        seed: int = 42,
                        save: Optional[str] = None,
                        show: bool = False) -> plt.Figure:
    """Run DE once with a callback that records the best NPV per generation,
    then plot NPV vs generation."""
    from scipy.optimize import differential_evolution
    from .optimizer import _objective

    history: list[tuple[int, float]] = []

    def cb(xk, convergence):
        # Evaluate the current candidate's NPV (cheap; one extra sim)
        f = _objective(xk, base=opt.base_params, tea_base=opt.tea_params,
                       topo=topo, feed_tph=opt.feed_tph,
                       feed_grade=opt.feed_grade)
        history.append((len(history) + 1, -f))
        return False

    differential_evolution(
        lambda x: _objective(x, base=opt.base_params, tea_base=opt.tea_params,
                             topo=topo, feed_tph=opt.feed_tph,
                             feed_grade=opt.feed_grade),
        BOUNDS,
        maxiter=maxiter, popsize=popsize, tol=0.01, atol=0.0,
        seed=seed, workers=1, polish=True, callback=cb,
    )

    if not history:
        history = [(1, 0.0)]
    gens = np.array([h[0] for h in history])
    npvs = np.array([h[1] for h in history]) / 1e9
    # Running best (cumulative max) so the plot shows monotonic improvement
    running_best = np.maximum.accumulate(npvs)

    _style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(gens, npvs, "o-", color=_PALETTE[0], markersize=4,
            linewidth=1.2, alpha=0.6, label="generation best")
    ax.plot(gens, running_best, "-", color="#D62728", linewidth=2.0,
            label="running best")
    ax.set_xlabel("DE generation")
    ax.set_ylabel("NPV ($B)")
    ax.set_title(f"Differential-evolution convergence — {topology_label(topo)}")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ------------------------------------------------------------------ 7. tornado
def plot_tornado(tr: TopologyResult, topo: Topology,
                 feed_tph: float = 2000.0, feed_grade: float = 0.008,
                 save: Optional[str] = None,
                 show: bool = False) -> plt.Figure:
    """For each decision variable: hold others at the optimum, push this
    one to its lo and hi bound, record NPV. Bars show downside (low->lo)
    in red and upside (low->hi) in green, sorted by total swing."""
    _style()
    base_x = tr.best_x.copy()
    base_npv = tr.best_npv / 1e9
    base_params = CuSulfideParams()
    tea_base = TEAParams()

    rows = []
    for i, (name, lo, hi) in enumerate(DECISION_VARS):
        x_lo = base_x.copy(); x_lo[i] = lo
        x_hi = base_x.copy(); x_hi[i] = hi
        p_lo, t_lo = _apply(x_lo, base_params, tea_base)
        p_hi, t_hi = _apply(x_hi, base_params, tea_base)
        sim_lo = simulate_cu_sulfide(feed_tph, feed_grade, p_lo, topology=topo)
        sim_hi = simulate_cu_sulfide(feed_tph, feed_grade, p_hi, topology=topo)
        npv_lo = evaluate(sim_lo, t_lo)["npv"] / 1e9
        npv_hi = evaluate(sim_hi, t_hi)["npv"] / 1e9
        rows.append((name, npv_lo, npv_hi, abs(npv_hi - npv_lo)))

    rows.sort(key=lambda r: r[3], reverse=False)  # smallest swing at bottom
    names = [r[0] for r in rows]
    lows = np.array([r[1] for r in rows])
    highs = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.5))
    y = np.arange(len(rows))
    # Down side (low bound) — bar from low_npv to baseline
    for i, (lo_v, hi_v) in enumerate(zip(lows, highs)):
        # Two bars centered at base_npv
        if lo_v < base_npv:
            ax.barh(i, lo_v - base_npv, left=base_npv,
                    color="#D62728", edgecolor="white", height=0.65)
        else:
            ax.barh(i, lo_v - base_npv, left=base_npv,
                    color="#9CCC9C", edgecolor="white", height=0.65)
        if hi_v > base_npv:
            ax.barh(i, hi_v - base_npv, left=base_npv,
                    color="#2CA02C", edgecolor="white", height=0.65)
        else:
            ax.barh(i, hi_v - base_npv, left=base_npv,
                    color="#F5A6A6", edgecolor="white", height=0.65)
    ax.axvline(base_npv, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("NPV ($B)")
    ax.set_title(f"Tornado — single-variable swing about the optimum  "
                 f"(baseline ${base_npv:.2f}B)")
    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color="#D62728", label="lower bound"),
        Patch(color="#2CA02C", label="upper bound"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=False, fontsize=9)
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ------------------------------------------------------------------ 8. end-to-end flowsheet
def plot_flowsheet(topology: Optional[Topology] = None,
                   sim_result: Optional[dict] = None,
                   save: Optional[str] = None,
                   show: bool = False) -> plt.Figure:
    """Full-route flowsheet visualisation covering both the sulfide
    concentrate and hydromet (leach/SX/EW) branches plus optional
    regrind/cleaner stages. Disabled stages are hidden so the active
    flowsheet shape is unambiguous; route boxes (sulfide vs hydromet)
    surround the active product side.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
    _style()
    topo = topology or Topology()
    fig, ax = plt.subplots(figsize=(17, 7))

    # Layout: three rows
    #   y = 4.0  comminution + flotation (always sulfide front-end)
    #   y = 4.0 → 2.5 dewatering-and-product (sulfide route)
    #   y = 2.5 → 1.0 hydromet branch (when leach enabled)
    #   y = 0.0  tails stream
    Y_TOP = 4.0
    Y_HYDRO = 1.5
    Y_TAILS = 0.0

    # All possible stages, with positions and on/off flags from topology.
    # Stages whose `on` is False are skipped entirely from the drawing.
    leach_route = topo.leach_enabled
    heap_route = topo.heap_leach_enabled
    any_hydromet = leach_route or heap_route
    # Heap-leach skips SAG/BM/cyclone/flotation entirely.
    sulfide_frontend = topo.flotation_enabled
    stages_all = [
        # key,            label,                x,    y,        on
        ("ore",            "Ore feed",          0.5,  Y_TOP,   True),
        ("crusher",        "Crusher",           1.8,  Y_TOP,   True),
        ("screen",         "Vibrating\nscreen", 3.0,  Y_TOP,   topo.screen_enabled and sulfide_frontend),
        ("sag",            "SAG mill",          4.2,  Y_TOP,   sulfide_frontend),
        ("bm",             "Ball mill",         5.4,  Y_TOP,   topo.ball_mill_enabled),
        ("cyclone",        "Hydrocyclone",      6.6,  Y_TOP,   topo.cyclone_enabled),
        ("flotation",      "Rougher\nflotation", 8.0, Y_TOP,   topo.flotation_enabled),
        # Scavenger bank sits on the rougher-tails line: it rescues Cu the
        # rougher missed before the stream reports to final tails. Always
        # present with the rougher (recovery already in the kinetics).
        ("scavenger",      "Scavenger\nflotation", 9.5, 2.0,    topo.flotation_enabled),
        ("regrind",        _regrind_label(topo), 9.6,  Y_TOP,   topo.regrind_enabled),
        ("cleaner",        f"Cleaner\nflotation\n({getattr(topo,'n_cleaner_stages',1)}-stage)", 11.2, Y_TOP,  topo.cleaner_enabled),
        # Sulfide product side — active only when NOT hydromet.
        ("thickener",      "Conc\nthickener",   13.0, Y_TOP,   topo.thickener_enabled and not any_hydromet),
        ("filter",         "Belt filter",       14.5, Y_TOP,   topo.filter_enabled and not any_hydromet),
        ("conc",           "Cu concentrate\n(to smelter)", 16.0, Y_TOP, not any_hydromet),
        # Concentrate-leach hydromet branch (POX/Albion on flotation conc)
        ("leach",          "Leach\n(POX/Albion)", 13.0, Y_HYDRO, topo.leach_enabled),
        # Heap-leach branch — pad replaces the entire sulfide front-end
        ("heap",           "Heap pad\n(whole-ore)", 4.5, Y_HYDRO, topo.heap_leach_enabled),
        # SX/EW/cathode positions depend on which hydromet branch is upstream
        ("sx",             "Solvent\nextraction",
         14.5 if leach_route else 8.0, Y_HYDRO, topo.sx_enabled),
        ("ew",             "Electro-\nwinning",
         15.7 if leach_route else 10.0, Y_HYDRO, topo.ew_enabled),
        ("cathode",        "Cathode\n(LME-grade)",
         16.9 if leach_route else 11.5, Y_HYDRO, topo.ew_enabled),
        # Tails (always present).
        ("tails",          "Final tails\n(to TSF)", 8.0, Y_TAILS, True),
    ]

    color_on = "#1F77B4"
    color_face_on = "#E5F0F9"
    color_hydro = "#9467BD"
    color_face_hydro = "#F2EBF7"

    box_w = 1.25
    box_h = 0.85

    pos = {}
    cleaner_n_stages = getattr(topo, "n_cleaner_stages", 1)
    for key, label, x, y, on in stages_all:
        if not on:
            continue
        # Multi-stage cleaner — render as N adjacent mini-boxes connected
        # by short forward arrows (and a recycle hint between stages) so
        # the stage count is visible at a glance, not just in the label.
        if key == "cleaner" and cleaner_n_stages > 1:
            sub_w = 0.42
            gap = 0.12
            total_w = cleaner_n_stages * sub_w + (cleaner_n_stages - 1) * gap
            x_left = x - total_w / 2
            for i in range(cleaner_n_stages):
                sx = x_left + i * (sub_w + gap) + sub_w / 2
                sub = FancyBboxPatch(
                    (sx - sub_w / 2, y - box_h / 2),
                    sub_w, box_h,
                    boxstyle="round,pad=0.04,rounding_size=0.08",
                    linewidth=1.2, edgecolor=color_on,
                    facecolor=color_face_on, alpha=0.95)
                ax.add_patch(sub)
                ax.text(sx, y + 0.05, f"C{i+1}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color_on)
                ax.text(sx, y - 0.18, "clean", ha="center", va="center",
                        fontsize=6.5, color="#444444")
                # Forward arrow to next stage
                if i < cleaner_n_stages - 1:
                    nxt_x = sx + sub_w / 2 + gap
                    arr = FancyArrowPatch(
                        (sx + sub_w / 2 + 0.02, y),
                        (nxt_x - 0.02, y),
                        arrowstyle="-|>", mutation_scale=8,
                        color=color_on, linewidth=1.0)
                    ax.add_patch(arr)
                # Stage-to-stage tails recycle (each cleaner's tails return
                # to the previous stage in series cleaner banks)
                if i > 0:
                    prev_x = sx - sub_w - gap
                    rec = FancyArrowPatch(
                        (sx - sub_w / 2 - 0.02, y - 0.25),
                        (prev_x + sub_w / 2 + 0.02, y - 0.25),
                        arrowstyle="-|>", mutation_scale=6,
                        color="#999999", linewidth=0.8,
                        connectionstyle="arc3,rad=-0.6")
                    ax.add_patch(rec)
            ax.text(x, y - box_h / 2 - 0.18,
                    f"{cleaner_n_stages}-stage cleaner bank",
                    ha="center", fontsize=8, color=color_on,
                    fontweight="bold")
            # Position used for arrows: centre of the bank.
            pos[key] = (x, y)
            continue
        # Pick palette: hydromet stages purple, sulfide blue, products black.
        if key in ("ore", "conc", "tails", "cathode"):
            face = "#FFFFFF"; edge = "black"; text_color = "black"
            fontweight = "bold"
        elif key in ("leach", "sx", "ew"):
            face = color_face_hydro; edge = color_hydro; text_color = "black"
            fontweight = "normal"
        elif key == "heap":
            # Heap pad — distinct red palette to set heap leach apart from
            # concentrate POX visually
            face = "#FDEDEC"; edge = "#C0392B"; text_color = "black"
            fontweight = "normal"
        else:
            face = color_face_on; edge = color_on; text_color = "black"
            fontweight = "normal"
        box = FancyBboxPatch((x - box_w / 2, y - box_h / 2),
                             box_w, box_h,
                             boxstyle="round,pad=0.06,rounding_size=0.12",
                             linewidth=1.4, edgecolor=edge,
                             facecolor=face, alpha=0.95)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, color=text_color, fontweight=fontweight)
        pos[key] = (x, y)

    # Route-frame backdrop: shaded rectangle around the active product side.
    if heap_route:
        # Heap-leach route — frame around the entire hydromet bottom row
        rect = Rectangle((3.7, Y_HYDRO - 0.7), 8.5, 1.4,
                         facecolor="#FDEDEC", edgecolor="#C0392B",
                         linewidth=1.0, linestyle="--", alpha=0.4, zorder=0)
        ax.add_patch(rect)
        ax.text(7.9, Y_HYDRO + 0.95, "WHOLE-ORE HEAP-LEACH ROUTE",
                ha="center", fontsize=9, color="#C0392B",
                fontweight="bold")
    elif leach_route:
        # Concentrate-leach hydromet — frame around leach/SX/EW/cathode
        rect = Rectangle((12.3, Y_HYDRO - 0.7), 5.2, 1.4,
                         facecolor="#F2EBF7", edgecolor="#9467BD",
                         linewidth=1.0, linestyle="--", alpha=0.4, zorder=0)
        ax.add_patch(rect)
        ax.text(14.9, Y_HYDRO + 0.95, "CONCENTRATE-LEACH HYDROMET ROUTE",
                ha="center", fontsize=9, color="#9467BD",
                fontweight="bold")
    else:
        # Sulfide route — frame around thickener/filter/conc
        rect = Rectangle((12.3, Y_TOP - 0.7), 5.2, 1.4,
                         facecolor="#E5F0F9", edgecolor="#1F77B4",
                         linewidth=1.0, linestyle="--", alpha=0.4, zorder=0)
        ax.add_patch(rect)
        ax.text(14.9, Y_TOP + 0.95, "SULFIDE ROUTE",
                ha="center", fontsize=9, color="#1F77B4",
                fontweight="bold")

    def _arrow(p_from, p_to, color="#333333", lw=1.5,
               connectionstyle="arc3,rad=0.0", label=None,
               label_dy=0.18):
        if p_from not in pos or p_to not in pos:
            return
        x0, y0 = pos[p_from]; x1, y1 = pos[p_to]
        dx, dy = x1 - x0, y1 - y0
        d = max(1e-6, (dx ** 2 + dy ** 2) ** 0.5)
        ux, uy = dx / d, dy / d
        margin = 0.65
        start = (x0 + ux * margin, y0 + uy * margin)
        end = (x1 - ux * margin, y1 - uy * margin)
        arr = FancyArrowPatch(start, end,
                              arrowstyle="-|>", mutation_scale=14,
                              color=color, linewidth=lw,
                              connectionstyle=connectionstyle)
        ax.add_patch(arr)
        if label:
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            ax.text(mx, my + label_dy, label, ha="center", fontsize=7.5,
                    color="#444444")

    # Build the active forward chain. Hop over disabled stages.
    if heap_route:
        # Heap-leach route: ore -> crusher -> heap pad -> SX -> EW -> cathode.
        # SAG/BM/cyclone/flotation/regrind/cleaner all bypassed.
        chain = ["ore", "crusher", "heap"]
        if topo.sx_enabled:
            chain.append("sx")
        if topo.ew_enabled:
            chain.extend(["ew", "cathode"])
    else:
        chain = ["ore", "crusher"]
        if topo.screen_enabled:
            chain.append("screen")
        chain.append("sag")
        if topo.ball_mill_enabled:
            chain.append("bm")
        if topo.cyclone_enabled and topo.ball_mill_enabled:
            chain.append("cyclone")
        chain.append("flotation")
        if topo.regrind_enabled:
            chain.append("regrind")
        if topo.cleaner_enabled:
            chain.append("cleaner")
        # End of common chain — branch to product side.
        if leach_route:
            # Concentrate (rougher or cleaner) feeds leach
            chain.append("leach")
            if topo.sx_enabled:
                chain.append("sx")
            if topo.ew_enabled:
                chain.extend(["ew", "cathode"])
        else:
            if topo.thickener_enabled:
                chain.append("thickener")
            if topo.filter_enabled:
                chain.append("filter")
            chain.append("conc")

    for a, b in zip(chain[:-1], chain[1:]):
        # Arrows between top row and hydro row drop diagonally; auto-routed
        if a in ("flotation", "regrind", "cleaner") and b == "leach":
            _arrow(a, b, color="#9467BD", lw=1.6,
                   connectionstyle="arc3,rad=0.18",
                   label="conc to leach", label_dy=0.20)
        elif a == "crusher" and b == "heap":
            # Drop primary-crushed ore down to the heap pad (heap is at y=1.5)
            _arrow(a, b, color="#C0392B", lw=1.6,
                   connectionstyle="arc3,rad=0.20",
                   label="primary crush", label_dy=0.20)
        else:
            heap_color = "#C0392B" if (a in ("heap",) or b in ("heap",)) else None
            hydro_color = "#9467BD" if (a in ("leach", "sx", "ew") or
                                         b in ("leach", "sx", "ew", "cathode")) \
                                    else None
            arrow_color = heap_color or hydro_color or "#333333"
            _arrow(a, b, color=arrow_color, lw=1.6)

    # Tails arrows — only drawn when sulfide-side flotation tails exist.
    # Rougher tails pass through the scavenger bank before reporting to final
    # tails; the scavenger concentrate recycles forward to the rougher feed
    # (Wills 7th ed. Ch. 12.4 rougher-scavenger circuit closure).
    if topo.flotation_enabled:
        _arrow("flotation", "scavenger", color="#777777", lw=1.2,
               connectionstyle="arc3,rad=-0.15", label="rougher tails")
        _arrow("scavenger", "tails", color="#777777", lw=1.2,
               connectionstyle="arc3,rad=-0.25", label="scavenger tails")
        _arrow("scavenger", "flotation", color="#2CA02C", lw=1.4,
               connectionstyle="arc3,rad=0.30", label="scav conc → rougher",
               label_dy=0.30)
        if topo.cleaner_enabled:
            # Cleaner tails recycle BACK to rougher feed (steady state).
            # Per Wills 7th ed. Ch. 12.4 cleaner/scavenger circuit closure:
            # cleaner tails return to rougher feed, NOT to final tails.
            # The N-stage cleaner bank has internal stage-to-stage recycles
            # too; depicting the bank as one closed loop captures the
            # steady-state material flow.
            n_stages = getattr(topo, "n_cleaner_stages", 1)
            recycle_label = (f"cleaner tails recycle → rougher"
                             f" ({n_stages}× internal pass)" if n_stages > 1
                             else "cleaner tails recycle → rougher")
            _arrow("cleaner", "flotation", color="#FF7F0E", lw=1.6,
                   connectionstyle="arc3,rad=0.55",
                   label=recycle_label, label_dy=0.55)
    elif heap_route:
        _arrow("heap", "tails", color="#999999", lw=1.0,
               connectionstyle="arc3,rad=-0.30", label="spent ore (residue)")

    # Cyclone recirculation (U/F → BM) — label with actual circulating load
    # so the loop's intensity is visible on the flowsheet.
    if topo.cyclone_enabled and topo.ball_mill_enabled:
        cl_val = None
        if sim_result is not None:
            streams = sim_result["streams"]
            if "cyc_uf" in streams and "ore" in streams:
                fresh = streams["ore"].solids_tph
                uf = streams["cyc_uf"].solids_tph
                if fresh > 0:
                    cl_val = uf / fresh
        cl_label = (f"U/F recycle (CL={cl_val:.1f}×)" if cl_val is not None
                    else "U/F recycle")
        _arrow("cyclone", "bm", color="#FF7F0E", lw=1.4,
               connectionstyle="arc3,rad=-0.5", label=cl_label)

    # Stream annotations: feed, final product, tails
    if sim_result is not None:
        streams = sim_result["streams"]
        # Feed
        if "ore" in streams and "ore" in pos:
            s = streams["ore"]
            ax.text(pos["ore"][0], pos["ore"][1] - 0.65,
                    f"{s.solids_tph:.0f} tph\n{s.cu_grade*100:.2f}% Cu",
                    ha="center", fontsize=8, color="#444444")
        # Tails
        tails_stream = streams.get("combined_tails",
                       streams.get("rougher_tails",
                       streams.get("heap_residue")))
        if tails_stream and "tails" in pos:
            ax.text(pos["tails"][0], pos["tails"][1] - 0.65,
                    f"{tails_stream.solids_tph:.0f} tph\n{tails_stream.cu_grade*100:.2f}% Cu",
                    ha="center", fontsize=8, color="#444444")
        # Scavenger recovery on rougher tails (fraction of the Cu the rougher
        # missed that the scavenger rescues).
        scav_R = sim_result.get("flotation", {}).get(
            "scavenger_recovery_on_tails")
        if scav_R is not None and "scavenger" in pos:
            ax.text(pos["scavenger"][0], pos["scavenger"][1] - 0.65,
                    f"rescues {scav_R*100:.0f}% of\nrougher-tails Cu",
                    ha="center", fontsize=8, color="#444444")
        # Heap-leach diagnostics: pad area + extraction
        heap_diag = sim_result.get("heap_leach", {})
        if heap_route and "heap" in pos and "X_Cu" in heap_diag:
            ax.text(pos["heap"][0], pos["heap"][1] - 0.65,
                    f"X_Cu = {heap_diag['X_Cu']*100:.0f}%\n"
                    f"pad {heap_diag['pad_area_m2']/1e4:.0f} ha",
                    ha="center", fontsize=8, color="#444444")
        # Sulfide product
        if "filter_cake" in streams and "conc" in pos:
            s = streams["filter_cake"]
            ax.text(pos["conc"][0], pos["conc"][1] - 0.65,
                    f"{s.solids_tph:.1f} tph\n{s.cu_grade*100:.1f}% Cu",
                    ha="center", fontsize=8, color="#444444")
        # Hydromet product (cathode) — applies to both leach and heap routes
        cathode_tph = sim_result.get("balance", {}).get("cu_in_cathode_tph", 0.0)
        if any_hydromet and "cathode" in pos:
            ax.text(pos["cathode"][0], pos["cathode"][1] - 0.65,
                    f"{cathode_tph:.2f} tph Cu\n(99.99% LME)",
                    ha="center", fontsize=8, color="#444444")

    # Title strip
    title = topology_label(topo)
    if sim_result is not None:
        bal = sim_result["balance"]
        title += (f"     R_Cu = {bal['overall_cu_recovery']*100:.1f}%"
                  f"     grade = {bal['conc_grade_pct']:.1f}%")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.4, 18.0)
    ax.set_ylim(-1.0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_capex_phase2bc(out_dir: str = "process_model/figures",
                        feed_tph: float = 2000.0, feed_grade: float = 0.008,
                        show: bool = False):
    """Before/after capex bar for the Phase 2B+2C rework, computed live on the
    default sulfide concentrator. 'Before' reverts the two new levers (Lang
    5.0, no balance-of-plant); 'after' is the current default. Annotated
    against the linear-scaled real-plant benchmark band ($1.4-2.4B for a
    50 ktpd Cu concentrator)."""
    _style()
    topo = Topology(regrind_enabled=True, cleaner_enabled=True,
                    n_cleaner_stages=2)
    sim = simulate_cu_sulfide(feed_tph, feed_grade, topology=topo)

    # 'Before' = Phase 2A state: single Lang 5.0, no balance-of-plant line.
    before = evaluate(sim, TEAParams(lang_factor_sulfide=5.0,
                                     balance_of_plant_per_tpd=0.0))["breakdown"]
    after = evaluate(sim, TEAParams())["breakdown"]

    def _stack(b):
        equip_installed = b["equipment_capex"] * b["lang_factor_applied"]
        return [equip_installed / 1e9,
                b["tailings_capex"] / 1e9,
                b["balance_of_plant_capex"] / 1e9]

    seg_labels = ["Equipment × Lang (installed)",
                  "Tailings storage (O'Hara)",
                  "Balance-of-plant (Phase 2B)"]
    seg_colors = ["#1F77B4", "#8C564B", "#2CA02C"]
    before_segs = _stack(before)
    after_segs = _stack(after)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    xs = [0, 1]
    bottoms = [0.0, 0.0]
    for i, (lab, col) in enumerate(zip(seg_labels, seg_colors)):
        vals = [before_segs[i], after_segs[i]]
        ax.bar(xs, vals, bottom=bottoms, width=0.55, color=col, label=lab,
               edgecolor="white", linewidth=0.6)
        bottoms = [bottoms[0] + vals[0], bottoms[1] + vals[1]]

    tot_before = sum(before_segs)
    tot_after = sum(after_segs)
    for x, tot, lang in zip(xs, [tot_before, tot_after],
                            [before["lang_factor_applied"],
                             after["lang_factor_applied"]]):
        ax.text(x, tot + 0.03, f"${tot:.2f}B\n(Lang {lang:.0f})",
                ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Real-plant benchmark band (linear-scaled to 50 ktpd, CAPEX_REWORK_PLAN).
    ax.axhspan(1.4, 2.4, color="#D62728", alpha=0.08, zorder=0)
    ax.axhline(1.4, color="#D62728", linestyle="--", linewidth=1.2)
    ax.axhline(2.4, color="#D62728", linestyle="--", linewidth=1.2)
    ax.text(1.46, 2.4, "real 50 ktpd Cu concentrator band  $1.4-2.4B\n"
                       "(Quellaveco / Cobre Panamá / Las Bambas, scaled)",
            ha="right", va="top", fontsize=8.5, color="#D62728")

    ax.set_xticks(xs)
    ax.set_xticklabels(["Before\n(Phase 2A)", "After\n(Phase 2B + 2C)"])
    ax.set_ylabel("Installed capex (USD billions)")
    ax.set_ylim(0, 2.7)
    ax.set_title(f"Capex realism — default sulfide concentrator "
                 f"@ {feed_tph:.0f} tph / {feed_grade*100:.1f}% Cu")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(0.5, -0.16,
            "Concentrator-only scope (no greenfield tailings-dam construction), "
            "so the model sits below the benchmark band by design.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555555")

    fig.tight_layout()
    save = os.path.join(out_dir, "10_capex_phase2bc.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"wrote {save}  (before ${tot_before:.2f}B -> after ${tot_after:.2f}B)")
    return fig


# ------------------------------------------------------------------ CLI battery
def run_all(out_dir: str = "process_model/figures",
            feed_tph: float = 2000.0, feed_grade: float = 0.008,
            maxiter: int = 80, popsize: int = 12) -> None:
    """Run optimizer once + render all eight plots into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"[1/4] Running optimizer at {feed_tph:.0f} tph @ "
          f"{feed_grade*100:.2f}% Cu  ...")
    opt = Optimizer(feed_tph=feed_tph, feed_grade=feed_grade)
    opt_res = opt.run(maxiter=maxiter, popsize=popsize, seed=42)
    print(f"      winner: {opt_res.winner.label}  "
          f"NPV ${opt_res.winner.best_npv/1e9:.2f}B")

    p1 = os.path.join(out_dir, "01_topology_leaderboard.png")
    plot_leaderboard(opt_res, save=p1)
    print(f"[2/4] {p1}")

    # Re-run TEA on the winning topology + best params for breakdown chart
    win = opt_res.winner
    p, t = _apply(win.best_x, CuSulfideParams(), TEAParams())
    sim = simulate_cu_sulfide(feed_tph, feed_grade, p, topology=win.topology)
    tea_res = evaluate(sim, t)
    p2 = os.path.join(out_dir, "02_capex_breakdown.png")
    plot_capex_breakdown(tea_res, topology_label_text=win.label, save=p2)
    print(f"      {p2}")

    p3 = os.path.join(out_dir, "03_grade_sensitivity.png")
    plot_grade_sensitivity(save=p3)
    print(f"[3/4] {p3}")

    p4 = os.path.join(out_dir, "04_operating_point.png")
    plot_operating_point(win, save=p4)
    print(f"[4/8] {p4}")

    # 5. single-variable sensitivity grid around the winner
    p5 = os.path.join(out_dir, "05_param_sensitivity.png")
    plot_param_sensitivity_grid(win, win.topology, feed_tph=feed_tph,
                                feed_grade=feed_grade, save=p5)
    print(f"[5/8] {p5}")

    # 6. DE convergence on the default topology
    p6 = os.path.join(out_dir, "06_de_convergence.png")
    plot_de_convergence(opt, Topology(), maxiter=maxiter,
                        popsize=popsize, save=p6)
    print(f"[6/8] {p6}")

    # 7. tornado on the winner
    p7 = os.path.join(out_dir, "07_tornado.png")
    plot_tornado(win, win.topology, feed_tph=feed_tph,
                 feed_grade=feed_grade, save=p7)
    print(f"[7/8] {p7}")

    # 8. end-to-end flowsheet with the winner's stream values
    p8 = os.path.join(out_dir, "08_flowsheet.png")
    plot_flowsheet(topology=win.topology, sim_result=sim, save=p8)
    print(f"[8/9] {p8}")

    # 9. flowsheet variant with stages disabled — shows how the diagram
    #    dims out when topology decisions remove stages
    alt_topo = Topology(thickener_enabled=False, filter_enabled=False,
                        cyclone_enabled=False)
    alt_p, alt_t = _apply(win.best_x, CuSulfideParams(), TEAParams())
    alt_sim = simulate_cu_sulfide(feed_tph, feed_grade, alt_p, topology=alt_topo)
    p9 = os.path.join(out_dir, "09_flowsheet_alt.png")
    plot_flowsheet(topology=alt_topo, sim_result=alt_sim, save=p9)
    print(f"[9/9] {p9}")
    print(f"\nMain figures written to {out_dir}/")


def run_grade_sweep(out_dir: str = "process_model/figures/grade_sweep",
                    feed_tph: float = 2000.0,
                    grades: Optional[list[float]] = None,
                    maxiter: int = 40, popsize: int = 10) -> None:
    """Re-run the optimizer at each grade and render an optimized flowsheet
    diagram per grade. Topology held at default (Topology()) since the
    leaderboard is grade-invariant in the current TEA structure (Option C
    caveat) — only the operating point varies, especially the grind P80
    via the recovery vs grinding-power tradeoff."""
    os.makedirs(out_dir, exist_ok=True)
    grades = grades if grades is not None else [
        0.002, 0.004, 0.006, 0.008, 0.010,
        0.012, 0.014, 0.016, 0.018, 0.020,
    ]
    topo = Topology()
    print(f"Re-running optimizer at each of {len(grades)} grades...")
    print(f"  topology held at: {topology_label(topo)}")
    print(f"  per-grade DE: maxiter={maxiter}, popsize={popsize}")
    print()
    print(f"{'grade':>6}  {'P80':>6}  {'R_Cu':>6}  {'grade_c':>7}  "
          f"{'NPV ($B)':>9}  {'IRR':>7}")
    print("-" * 60)

    summary = []
    for g in grades:
        opt = Optimizer(feed_tph=feed_tph, feed_grade=float(g))
        tr = opt.run_one(topo, maxiter=maxiter, popsize=popsize, seed=42)
        # Re-run sim at the optimum to get fresh stream values for the diagram
        p, t = _apply(tr.best_x, CuSulfideParams(), TEAParams())
        sim = simulate_cu_sulfide(feed_tph, float(g), p, topology=topo)
        fname = os.path.join(out_dir, f"flowsheet_{g*100:.1f}pct.png")
        fig = plot_flowsheet(topology=topo, sim_result=sim, save=fname)
        plt.close(fig)
        p80 = float(tr.best_x[9])
        irr_s = f"{tr.irr*100:5.1f}%" if tr.irr is not None else "  n/a"
        print(f"  {g*100:4.1f}%  {p80:5.0f}  {tr.cu_recovery*100:5.1f}%  "
              f"{tr.conc_grade_pct:6.1f}%  {tr.best_npv/1e9:>8.2f}  {irr_s}  "
              f"-> {os.path.basename(fname)}")
        summary.append((g, p80, tr.cu_recovery, tr.best_npv))
    print()
    print(f"Done: {len(grades)} PNGs in {out_dir}/")


def run_grade_sweep_route(out_dir: str = "process_model/figures/grade_sweep_route",
                          feed_tph: float = 2000.0,
                          grades: Optional[list[float]] = None,
                          maxiter: int = 40, popsize: int = 10) -> None:
    """At each grade, enumerate the curated archetypes (sulfide / hydromet /
    hybrid), run DE on each, and render the *winning* topology's flowsheet.
    Answers the question: does the optimizer pick leach for low-grade ore?"""
    os.makedirs(out_dir, exist_ok=True)
    grades = grades if grades is not None else [
        0.002, 0.004, 0.006, 0.008, 0.010,
        0.012, 0.014, 0.016, 0.018, 0.020,
    ]
    print(f"Per-grade archetype sweep at {len(grades)} grades...")
    print(f"  per-grade DE: maxiter={maxiter}, popsize={popsize}")
    print()
    print(f"{'grade':>6}  {'winner':<45s}  {'NPV ($B)':>9}  {'IRR':>7}  "
          f"{'capex ($M)':>10}")
    print("-" * 90)

    for g in grades:
        # Mineralogy by grade band: low-grade deposits in nature are
        # supergene chalcocite enrichment blankets (Spence, Mantoverde,
        # Lomas Bayas, Radomiro Tomic) — heap leach is industrial answer.
        # >=0.8% Cu deposits are primary chalcopyrite (Escondida, Cobre
        # Panama, Cerro Verde) — flotation to smelter is industrial answer.
        mineralogy = "chalcocite" if g < 0.008 else "chalcopyrite"
        base = CuSulfideParams(mineralogy=mineralogy)
        opt = Optimizer(feed_tph=feed_tph, feed_grade=float(g),
                        base_params=base)
        opt_res = opt.run(maxiter=maxiter, popsize=popsize, seed=42)
        win = opt_res.winner
        # Re-run sim at the winner's optimum for fresh stream values
        p, t = _apply(win.best_x, base, TEAParams())
        sim = simulate_cu_sulfide(feed_tph, float(g), p, topology=win.topology)
        fname = os.path.join(out_dir, f"flowsheet_{g*100:.1f}pct.png")
        fig = plot_flowsheet(topology=win.topology, sim_result=sim, save=fname)
        plt.close(fig)
        # Also dump leaderboard for this grade
        lb_fname = os.path.join(out_dir, f"leaderboard_{g*100:.1f}pct.png")
        fig_lb = plot_leaderboard(opt_res, save=lb_fname)
        plt.close(fig_lb)
        irr_s = f"{win.irr*100:6.1f}%" if win.irr is not None else "    n/a"
        print(f"  {g*100:4.1f}%  {win.label:<45s}  {win.best_npv/1e9:>8.2f}  "
              f"{irr_s}  {win.capex/1e6:>9.1f}")
    print()
    print(f"Done: flowsheet+leaderboard PNGs in {out_dir}/")


def run_route_comparison(out_dir: str = "process_model/figures/route_comparison",
                         feed_tph: float = 2000.0,
                         grades: Optional[list[float]] = None) -> None:
    """Head-to-head: sulfide (rougher+regrind+cleaner) vs hydromet (leach+SX+
    EW+neut) at default operating params (no DE). Plots NPV vs grade for both
    routes and marks the crossover."""
    os.makedirs(out_dir, exist_ok=True)
    grades = grades if grades is not None else [
        0.002, 0.004, 0.006, 0.008, 0.010,
        0.012, 0.014, 0.016, 0.018, 0.020,
    ]
    sulfide = Topology(regrind_enabled=True, cleaner_enabled=True)
    hydromet = Topology(filter_enabled=False,
                        regrind_enabled=True, cleaner_enabled=True,
                        leach_enabled=True, sx_enabled=True, ew_enabled=True,
                        neutralization_enabled=True)
    base = CuSulfideParams()
    econ = TEAParams()

    rows = []
    print(f"{'grade':>6}  "
          f"{'sulfide NPV':>11}  {'sulfide IRR':>11}  {'sulf cap $M':>11}  "
          f"{'hydromet NPV':>12}  {'hydro IRR':>9}  {'hydro cap $M':>12}  "
          f"{'winner':>9}")
    print("-" * 100)
    for g in grades:
        sim_s = simulate_cu_sulfide(feed_tph, g, base, topology=sulfide)
        sim_h = simulate_cu_sulfide(feed_tph, g, base, topology=hydromet)
        rs = evaluate(sim_s, econ); rh = evaluate(sim_h, econ)
        win = "sulfide" if rs["npv"] > rh["npv"] else "hydromet"
        rows.append({
            "grade": g,
            "sulf_npv": rs["npv"]/1e9, "sulf_irr": rs["irr"], "sulf_capex": rs["total_capex"]/1e6,
            "hydro_npv": rh["npv"]/1e9, "hydro_irr": rh["irr"], "hydro_capex": rh["total_capex"]/1e6,
            "winner": win,
        })
        s_irr = f"{rs['irr']*100:9.1f}%" if rs['irr']==rs['irr'] else "      n/a"
        h_irr = f"{rh['irr']*100:7.1f}%" if rh['irr']==rh['irr'] else "    n/a"
        print(f"  {g*100:4.1f}%  "
              f"{rs['npv']/1e9:>10.2f}  {s_irr}  {rs['total_capex']/1e6:>10.0f}  "
              f"{rh['npv']/1e9:>11.2f}  {h_irr}  {rh['total_capex']/1e6:>11.0f}  "
              f"{win:>9}")

    # ----- chart 1: NPV vs grade (both routes)
    _style()
    g_pct = np.array([r["grade"]*100 for r in rows])
    s_npv = np.array([r["sulf_npv"] for r in rows])
    h_npv = np.array([r["hydro_npv"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(g_pct, s_npv, "-o", color=_PALETTE[0], linewidth=2,
            markersize=7, label="Sulfide (rougher+regrind+cleaner)")
    ax.plot(g_pct, h_npv, "-s", color=_PALETTE[1], linewidth=2,
            markersize=7, label="Hydromet (flotation conc leach+SX+EW+neut)")
    ax.axhline(0, color="black", linewidth=0.5)
    # Mark crossover (linear interp) if there is one
    diff = s_npv - h_npv
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change):
        i = sign_change[0]
        x1, x2 = g_pct[i], g_pct[i+1]
        d1, d2 = diff[i], diff[i+1]
        x_cross = x1 - d1 * (x2 - x1) / (d2 - d1)
        y_cross = np.interp(x_cross, g_pct, s_npv)
        ax.scatter([x_cross], [y_cross], s=180, color="#D62728",
                   zorder=5, edgecolor="white", linewidth=1.5,
                   label=f"crossover ~ {x_cross:.2f}% Cu")
    ax.set_xlabel("Feed Cu grade (%)")
    ax.set_ylabel("NPV ($B, 25-yr DCF @ 8%)")
    ax.set_title(f"Route comparison at {feed_tph:.0f} tph — default operating params")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "01_npv_vs_grade.png")
    fig.savefig(p1); plt.close(fig)

    # ----- chart 2: capex side-by-side (constant across grades for hydromet,
    # rises slightly with grade for sulfide via conc-side sizing)
    s_cap = np.array([r["sulf_capex"] for r in rows])
    h_cap = np.array([r["hydro_capex"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(g_pct, s_cap, "-o", color=_PALETTE[0], linewidth=2, markersize=6,
            label="Sulfide capex")
    ax.plot(g_pct, h_cap, "-s", color=_PALETTE[1], linewidth=2, markersize=6,
            label="Hydromet capex")
    ax.set_xlabel("Feed Cu grade (%)")
    ax.set_ylabel("Total installed capex ($M)")
    ax.set_title("Total installed capex by route")
    ax.legend(frameon=False)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "02_capex_vs_grade.png")
    fig.savefig(p2); plt.close(fig)

    # ----- chart 2b: NPV (left axis) + IRR (right axis), both routes
    s_irr = np.array([r["sulf_irr"]*100 if r["sulf_irr"]==r["sulf_irr"] else np.nan
                      for r in rows])
    h_irr = np.array([r["hydro_irr"]*100 if r["hydro_irr"]==r["hydro_irr"] else np.nan
                      for r in rows])
    fig, ax_l = plt.subplots(figsize=(10, 5.5))
    # NPV — solid lines, left axis
    ax_l.plot(g_pct, s_npv, "-o", color=_PALETTE[0], linewidth=2.2,
              markersize=7, label="Sulfide NPV")
    ax_l.plot(g_pct, h_npv, "-s", color=_PALETTE[1], linewidth=2.2,
              markersize=7, label="Hydromet NPV")
    ax_l.axhline(0, color="black", linewidth=0.5)
    ax_l.set_xlabel("Feed Cu grade (%)")
    ax_l.set_ylabel("NPV ($B, 25-yr DCF @ 8%)")
    ax_l.tick_params(axis="y")

    # IRR — dashed lines, right axis
    ax_r = ax_l.twinx()
    ax_r.plot(g_pct, s_irr, "--o", color=_PALETTE[0], linewidth=1.6,
              markersize=5, alpha=0.65, markerfacecolor="white",
              label="Sulfide IRR")
    ax_r.plot(g_pct, h_irr, "--s", color=_PALETTE[1], linewidth=1.6,
              markersize=5, alpha=0.65, markerfacecolor="white",
              label="Hydromet IRR")
    ax_r.set_ylabel("IRR (%)")
    ax_r.grid(False)
    ax_r.spines["top"].set_visible(False)

    # Combined legend
    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
    ax_l.set_title(f"NPV (solid) & IRR (dashed) vs feed grade — "
                   f"{feed_tph:.0f} tph, default operating params")
    fig.tight_layout()
    p2b = os.path.join(out_dir, "02b_npv_irr_vs_grade.png")
    fig.savefig(p2b); plt.close(fig)

    # ----- chart 3: winner-by-grade strip
    fig, ax = plt.subplots(figsize=(9, 1.8))
    for r in rows:
        c = _PALETTE[0] if r["winner"] == "sulfide" else _PALETTE[1]
        ax.barh([0], [0.2], left=[r["grade"]*100 - 0.1], color=c,
                edgecolor="white", linewidth=1)
    ax.set_xlim(0, 2.1)
    ax.set_yticks([])
    ax.set_xlabel("Feed Cu grade (%)")
    ax.set_title("Winning route by grade  (blue = sulfide, orange = hydromet)")
    fig.tight_layout()
    p3 = os.path.join(out_dir, "03_winner_strip.png")
    fig.savefig(p3); plt.close(fig)

    print()
    print(f"Charts -> {out_dir}/")


if __name__ == "__main__":
    import sys
    if "--capex-phase2bc" in sys.argv:
        plot_capex_phase2bc()
    elif "--route-compare" in sys.argv:
        run_route_comparison()
    elif "--grade-sweep-route" in sys.argv:
        run_grade_sweep_route()
    elif "--grade-sweep" in sys.argv:
        run_grade_sweep()
    elif "--all" in sys.argv:
        run_all()
        run_grade_sweep()
    else:
        run_all()
