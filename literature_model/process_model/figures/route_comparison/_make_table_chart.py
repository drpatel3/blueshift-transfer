"""Hardcoded route-comparison chart from the user-provided table.

Input rows are the 6 grades from the side-by-side table:
    0.2, 0.6, 0.8, 1.0, 1.4, 2.0 % Cu
Left axis: NPV ($B). Right axis: IRR (%). Solid = NPV, dashed = IRR.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

ROWS = [
    # grade %, sulf NPV, sulf IRR (None=n/a), hydro NPV, hydro IRR
    (0.2,  -7.68, None,   -7.28, None),
    (0.4,  -4.98, None,   -4.43, None),
    (0.6,  -2.29, None,   -1.56, None),
    (0.8,   0.40, 0.162,   1.32, 0.198),
    (1.0,   3.09, 0.648,   4.21, 0.401),
    (1.2,   5.78, 1.129,   7.11, 0.571),
    (1.4,   8.48, 1.609,  10.01, 0.716),
    (1.6,  11.17, 2.086,  12.92, 0.842),
    (1.8,  13.86, 2.563,  15.83, 0.955),
    (2.0,  16.55, 3.037,  18.74, 1.055),
]

OUT_PATH = os.path.join(
    "process_model", "figures", "route_comparison",
    "table_npv_irr_vs_grade.png",
)

PALETTE = {"sulf": "#1F77B4", "hydro": "#FF7F0E"}


def main():
    g = np.array([r[0] for r in ROWS])
    s_npv = np.array([r[1] for r in ROWS])
    h_npv = np.array([r[3] for r in ROWS])
    s_irr = np.array([r[2]*100 if r[2] is not None else np.nan for r in ROWS])
    h_irr = np.array([r[4]*100 if r[4] is not None else np.nan for r in ROWS])

    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140,
                         "font.size": 10, "axes.spines.top": False,
                         "axes.grid": True, "grid.alpha": 0.25})

    fig, ax_l = plt.subplots(figsize=(10, 5.5))
    ax_l.plot(g, s_npv, "-o", color=PALETTE["sulf"], lw=2.2, ms=8,
              label="Sulfide NPV")
    ax_l.plot(g, h_npv, "-s", color=PALETTE["hydro"], lw=2.2, ms=8,
              label="Hydromet NPV")
    ax_l.set_xlabel("Feed Cu grade (%)")
    ax_l.set_xticks(np.arange(0.2, 2.01, 0.2))
    ax_l.set_ylabel("NPV ($B, 25-yr DCF @ 8%)")

    ax_r = ax_l.twinx()
    ax_r.plot(g, s_irr, "--o", color=PALETTE["sulf"], lw=1.6, ms=6,
              alpha=0.7, mfc="white", label="Sulfide IRR")
    ax_r.plot(g, h_irr, "--s", color=PALETTE["hydro"], lw=1.6, ms=6,
              alpha=0.7, mfc="white", label="Hydromet IRR")
    ax_r.set_ylabel("IRR (%)")
    ax_r.grid(False)
    ax_r.spines["top"].set_visible(False)

    # Align zero across left/right axes by matching the negative-fraction.
    npv_min = float(min(s_npv.min(), h_npv.min(), 0.0))
    npv_max = float(max(s_npv.max(), h_npv.max(), 0.0))
    pad = 0.05 * (npv_max - npv_min)
    npv_lo = npv_min - pad
    npv_hi = npv_max + pad
    ax_l.set_ylim(npv_lo, npv_hi)
    irr_max = float(np.nanmax(np.concatenate([s_irr, h_irr])))
    irr_max *= 1.05
    # Solve for irr_lo such that 0 sits at the same fraction on both axes:
    #   (0 - npv_lo) / (npv_hi - npv_lo) == (0 - irr_lo) / (irr_max - irr_lo)
    f = (0.0 - npv_lo) / (npv_hi - npv_lo)
    irr_lo = -f / (1.0 - f) * irr_max
    ax_r.set_ylim(irr_lo, irr_max)
    ax_l.axhline(0, color="black", lw=1.0, zorder=0)

    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
    ax_l.set_title("Sulfide vs Hydromet — NPV (solid) & IRR (dashed) vs grade")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
