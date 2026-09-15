"""Generate the five INFORMATION_FLOW.docx supporting visuals.

Disposable helper — safe to delete after PNGs are produced.
Run from repo root:  python -m process_model.figures.info_flow._make_figs
"""
from __future__ import annotations
import os, sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.sankey import Sankey

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# 01 — Orchestration call graph                                                #
# --------------------------------------------------------------------------- #
def fig_callgraph() -> str:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    boxes = [
        # (x, y, w, h, label, color)
        (3.5, 9.0, 3.0, 0.7, "__main__.main()", "#E8E8E8"),
        (3.5, 8.0, 3.0, 0.7, "Optimizer.run()", "#CFE2F3"),
        (1.5, 6.7, 3.0, 0.9, "topology.all_topologies()\n+ _is_valid()", "#D9EAD3"),
        (5.5, 6.7, 3.0, 0.9, "differential_evolution\n(scipy)", "#D9EAD3"),
        (3.5, 5.3, 3.0, 0.7, "_objective(x)  →  −NPV", "#FFE599"),
        (3.5, 4.3, 3.0, 0.7, "_apply(x)  →  Params", "#FFE599"),
        (3.5, 3.0, 3.0, 0.9, "throughput.simulate_cu_sulfide()\n(stage-by-stage)", "#F4CCCC"),
        (0.3, 1.4, 4.6, 1.0,
         "flowsheet.run_crusher / run_sag / run_hydrocyclone\n"
         "/ run_flotation / run_thickener / run_filter",
         "#EAD1DC"),
        (5.1, 1.4, 4.6, 1.0,
         "_run_regrind_and_cleaner / _run_concentrate_hydromet\n"
         "/ _run_heap_leach_route / _build_mass_balance",
         "#EAD1DC"),
        (3.5, 0.1, 3.0, 0.7, "tea.evaluate(sim)  →  NPV, IRR", "#CFE2F3"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=color, ec="#444", linewidth=0.8))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=9, family="DejaVu Sans")

    arrows = [
        (5.0, 9.0, 5.0, 8.7),
        (4.0, 8.0, 3.0, 7.6),
        (6.0, 8.0, 7.0, 7.6),
        (3.0, 6.7, 4.5, 6.0),
        (7.0, 6.7, 5.5, 6.0),
        (5.0, 5.3, 5.0, 5.0),
        (5.0, 4.3, 5.0, 3.9),
        (4.5, 3.0, 2.6, 2.4),
        (5.5, 3.0, 7.4, 2.4),
        (2.6, 1.4, 5.0, 0.8),
        (7.4, 1.4, 5.0, 0.8),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

    ax.text(5.0, 9.9, "Process-model call graph", ha="center", fontsize=14,
            fontweight="bold")
    ax.text(5.0, -0.4,
            "DE feeds candidate x to _objective; _apply decodes x into Params; "
            "simulate runs stage-by-stage; tea.evaluate closes the loop with −NPV.",
            ha="center", fontsize=8, style="italic", color="#555")

    out = os.path.join(HERE, "01_orchestration_callgraph.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 02 — TEA waterfall: baseline vs optimizer winner                             #
# --------------------------------------------------------------------------- #
def _winner_breakdown() -> dict:
    """Run DE on the winner topology and return the full TEA dict."""
    from process_model.topology import Topology
    from process_model.optimizer import Optimizer, _apply
    from process_model.throughput import simulate_cu_sulfide
    from process_model.tea import evaluate

    winner = Topology(
        flotation_enabled=True, thickener_enabled=True, filter_enabled=True,
        ball_mill_enabled=True, cyclone_enabled=True, screen_enabled=True,
        regrind_enabled=True, cleaner_enabled=True,
        regrind_mill_type="tower", n_cleaner_stages=2,
    )
    opt = Optimizer(feed_tph=2000.0, feed_grade=0.008)
    tr = opt.run_one(winner, maxiter=30, popsize=10)
    p, t = _apply(tr.best_x, opt.base_params, opt.tea_params)
    sim = simulate_cu_sulfide(2000.0, 0.008, p, topology=winner)
    return evaluate(sim, t)


def _baseline_breakdown() -> dict:
    from process_model.throughput import simulate_cu_sulfide
    from process_model.tea import evaluate
    return evaluate(simulate_cu_sulfide(2000.0, 0.008))


def fig_tea_waterfall() -> str:
    base = _baseline_breakdown()
    win = _winner_breakdown()

    cats = ["Revenue\n(net, $M/yr)", "Opex\n($M/yr)", "FCF\n($M/yr)",
            "Capex\n(installed, $M)", "NPV\n($B)"]
    base_vals = [
        base["annual_revenue"]/1e6, base["annual_opex"]/1e6,
        base["annual_fcf"]/1e6, base["total_capex"]/1e6, base["npv"]/1e9,
    ]
    win_vals = [
        win["annual_revenue"]/1e6, win["annual_opex"]/1e6,
        win["annual_fcf"]/1e6, win["total_capex"]/1e6, win["npv"]/1e9,
    ]

    fig, axes = plt.subplots(1, 5, figsize=(13, 4.5))
    for ax, cat, b, w in zip(axes, cats, base_vals, win_vals):
        bars = ax.bar(["baseline", "winner"], [b, w],
                      color=["#c0504d", "#4f81bd"], edgecolor="#222")
        ax.set_title(cat, fontsize=10)
        ax.axhline(0, color="#444", lw=0.6)
        for bar, val in zip(bars, [b, w]):
            offset = (max(abs(b), abs(w)) or 1) * 0.04
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (offset if val >= 0 else -offset*1.5),
                    f"{val:,.0f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("TEA closure — unoptimised baseline vs DE winner "
                 "(2000 tph @ 0.8% Cu, chalcopyrite)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, "02_tea_waterfall.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)

    # Return numbers so we can paste a table into the doc
    return out, base, win


# --------------------------------------------------------------------------- #
# 03 — Cu mass-flow Sankey (Run 1 numbers, baseline)                           #
# --------------------------------------------------------------------------- #
def fig_cu_sankey() -> str:
    from process_model.throughput import simulate_cu_sulfide
    sim = simulate_cu_sulfide(2000.0, 0.008)
    s = sim["streams"]
    cu = {k: v.solids_tph * v.cu_grade for k, v in s.items()}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sankey = Sankey(ax=ax, scale=1.0/16.0, offset=0.25, head_angle=120,
                    format="%.2f t/h", unit=" Cu", gap=0.4)
    # Single-flow chain: ore -> rougher conc + rougher tails
    sankey.add(
        flows=[cu["ore"], -cu["rougher_conc"], -cu["rougher_tails"]],
        labels=["ROM ore", "rougher conc\n→ thickener/filter",
                "rougher tails\n→ TSF"],
        orientations=[0, 1, -1],
        pathlengths=[0.5, 0.6, 0.6],
        trunklength=1.5,
        facecolor="#c0a060",
    )
    sankey.finish()
    ax.set_title("Cu mass-flow trace — baseline sim (Cu_in = 16.0 t/h, "
                 "closure_err = 0.000)", fontsize=11)
    ax.axis("off")
    out = os.path.join(HERE, "03_cu_sankey.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 04 — DE saturation: where in the bound box does the winner sit?              #
# --------------------------------------------------------------------------- #
def fig_de_saturation() -> str:
    from process_model.optimizer import DECISION_VARS
    # Run 3 winner from scrollback (matches Topology used in step 2)
    winner = {
        "sp_power": 1.2000, "sp_gas_rate": 1.5000, "frother_conc": 50.0000,
        "flot_residence_time_min": 18.2740, "air_fraction": 0.2500,
        "slurry_fraction": 0.1500, "contact_angle": 55.0000,
        "circulating_load": 1.5000, "tailings_adverse_factor": 1.0000,
        "target_flot_P80_um": 200.0000,
    }
    names = [n for n, _, _ in DECISION_VARS]
    lo = [lo for _, lo, _ in DECISION_VARS]
    hi = [hi for _, _, hi in DECISION_VARS]
    frac = [(winner[n] - lo[i]) / (hi[i] - lo[i]) for i, n in enumerate(names)]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ys = list(range(len(names)))
    colors = ["#c0504d" if (f <= 0.02 or f >= 0.98) else "#4f81bd" for f in frac]
    ax.barh(ys, frac, color=colors, edgecolor="#222")
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#222", lw=0.6); ax.axvline(1, color="#222", lw=0.6)
    ax.set_xlim(-0.05, 1.10); ax.set_xlabel("Position in bound box (0 = low, 1 = high)")
    for y, (n, f) in enumerate(zip(names, frac)):
        ax.text(min(f, 1.0) + 0.02, y,
                f"{winner[n]:g}  ∈ [{lo[y]:g}, {hi[y]:g}]",
                va="center", fontsize=8)
    ax.set_title("DE winner — every variable pinned to a bound\n"
                 "(red = saturated; the bounds *are* the design constraints)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    out = os.path.join(HERE, "04_de_saturation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 05 — Block diagram (matplotlib render of the ASCII version)                  #
# --------------------------------------------------------------------------- #
def fig_block_diagram() -> str:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

    def box(x, y, w, h, text, color="#E8E8E8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=color, ec="#333", linewidth=0.9))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.1))
        if label:
            ax.text((x1+x2)/2 + 0.1, (y1+y2)/2, label, fontsize=8,
                    color="#333", style="italic")

    # ROM and comminution column (left)
    box(0.4, 8.0, 2.0, 0.7, "ROM ore", "#FFE599")
    box(0.4, 6.9, 2.0, 0.7, "Crusher *", "#D9EAD3")
    box(0.4, 5.8, 2.0, 0.7, "SAG + BM *", "#D9EAD3")
    box(0.4, 4.7, 2.0, 0.7, "Hydrocyclone", "#D9EAD3")
    box(0.4, 3.4, 2.0, 0.9, "Rougher\nflotation *", "#CFE2F3")

    # Sulfide route (A)
    box(3.5, 5.6, 2.6, 0.8, "Regrind\n+ Cleaner", "#CFE2F3")
    box(3.5, 4.2, 2.6, 0.8, "Thickener\n+ Filter", "#CFE2F3")
    box(3.5, 3.0, 2.6, 0.7, "Cu concentrate", "#FFD966")

    # Concentrate-leach (B)
    box(6.8, 5.6, 2.5, 0.8, "POX leach\n+ Neut.", "#F4CCCC")
    box(6.8, 4.2, 2.5, 0.8, "SX → EW", "#F4CCCC")
    box(6.8, 3.0, 2.5, 0.7, "Cathode (B)", "#FFD966")

    # Heap (C)
    box(9.7, 5.6, 2.0, 0.8, "Heap pad", "#EAD1DC")
    box(9.7, 4.2, 2.0, 0.8, "SX → EW", "#EAD1DC")
    box(9.7, 3.0, 2.0, 0.7, "Cathode (C)", "#FFD966")

    # TEA + DE
    box(3.5, 1.2, 5.0, 0.9, "TEA  →  Revenue · Opex · Capex · NPV · IRR",
        "#FFE599")
    box(8.7, 1.2, 3.0, 0.9, "Optimizer (DE)\nadjusts topology + 10 vars",
        "#CFE2F3")

    # Arrows down the left column
    for y1, y2 in [(8.0, 7.6), (6.9, 6.5), (5.8, 5.4), (4.7, 4.3)]:
        arrow(1.4, y1, 1.4, y2)

    # Rougher conc → A, B; tails → C? no — tails to TSF
    arrow(2.4, 4.0, 3.5, 6.0, "conc → A")
    arrow(2.4, 3.8, 6.8, 6.0, "conc → B")
    # heap-leach route bypasses BM/cyclone/cleaner; show as direct ROM branch
    arrow(2.4, 8.2, 9.7, 6.0, "ROM → C (heap)")

    arrow(4.8, 5.6, 4.8, 5.0)
    arrow(4.8, 4.2, 4.8, 3.7)
    arrow(8.05, 5.6, 8.05, 5.0)
    arrow(8.05, 4.2, 8.05, 3.7)
    arrow(10.7, 5.6, 10.7, 5.0)
    arrow(10.7, 4.2, 10.7, 3.7)

    # All routes → TEA
    arrow(4.8, 3.0, 5.0, 2.1)
    arrow(8.05, 3.0, 6.0, 2.1)
    arrow(10.7, 3.0, 7.0, 2.1)

    # TEA ↔ DE
    arrow(8.5, 1.65, 8.7, 1.65, "−NPV")
    arrow(8.7, 1.45, 8.5, 1.45)

    ax.text(6.0, 8.6, "Block diagram — three product routes share comminution; "
            "TEA + DE close the loop",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(6.0, 0.4, "(*) always-on; A = sulfide conc sale, "
            "B = concentrate-leach to cathode, C = whole-ore heap to cathode",
            ha="center", fontsize=8, style="italic", color="#555")

    out = os.path.join(HERE, "05_block_diagram.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    outs = []
    print("[1/5] callgraph ..."); outs.append(fig_callgraph())
    print("[2/5] tea waterfall (running DE for winner, ~30 s) ...")
    out2, base, win = fig_tea_waterfall(); outs.append(out2)
    print("[3/5] Cu sankey ...");        outs.append(fig_cu_sankey())
    print("[4/5] DE saturation ...");    outs.append(fig_de_saturation())
    print("[5/5] block diagram ...");    outs.append(fig_block_diagram())

    # Dump baseline vs winner TEA numbers for the doc table
    print("\n--- TEA numbers for §2c ---")
    for k in ("npv", "irr", "total_capex", "annual_revenue", "annual_opex",
              "annual_fcf"):
        b = base.get(k); w = win.get(k)
        print(f"  {k:18s}  baseline={b!r:>22}   winner={w!r:>22}")

    print("\nwrote:")
    for o in outs:
        print(" ", o)
