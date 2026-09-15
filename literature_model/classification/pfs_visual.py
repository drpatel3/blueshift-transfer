"""PFS-style visualization comparing ground truth vs predicted flowsheets.

Reuses the equipment symbol rendering from mini/pfs_visual.py but operates
on the classification pipeline's V2 stage vocabulary.  Accepts per-doc
evaluation results (from llm_eval_entry.py) and renders side-by-side
PFS comparison diagrams with TP/FP/FN color coding.

Usage:
    python classification/pfs_visual.py eval_results.json --grade 0.50
    python classification/pfs_visual.py eval_results.json --doc DOC_ID
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon, FancyBboxPatch, Circle

# ---------------------------------------------------------------------------
# V2 stage -> equipment symbol mapping
# Maps each V2 stage to the closest canonical equipment symbol type
# ---------------------------------------------------------------------------
V2_TO_SYMBOL = {
    # Input / material handling
    "input": "input",
    "stockpile": "stockpile",
    "bin": "bin",
    "feeder": "conveyor",
    "conveyor": "conveyor",
    # Crushing
    "primary_crusher": "crusher",
    "secondary_crusher": "crusher",
    "hpgr": "crusher",
    # Grinding
    "sag_mill": "mill",
    "ball_mill": "mill",
    "regrind": "regrind",
    # Classification
    "screen": "screen",
    "cyclone": "cyclone",
    # Concentration
    "rougher_flotation": "flotation",
    "cleaner_flotation": "flotation",
    "gravity_concentration": "gravity",
    "magnetic_separation": "screen",      # closest visual analog
    # Hydrometallurgy
    "leach": "leach",
    "adsorption": "adsorption",
    "elution": "tank",
    "solvent_extraction": "solvent_extraction",
    "electrowinning": "electrowinning",
    "precipitation": "precipitation",
    "ion_exchange": "tank",
    "ccd": "thickener",
    # Thermal
    "kiln": "tank",
    "smelting": "tank",
    "carbon_regeneration": "tank",
    "agglomeration": "tank",
    # Solid-liquid separation
    "concentrate_thickener": "thickener",
    "tailings_thickener": "thickener",
    "filter": "filter",
    # Output / disposal
    "concentrate_product": "output",
    "tailing": "tailing",
    "solution_pond": "tailing",
    "water_treatment": "treatment",
    "ore_sorting": "ore_sorting",
    "tank": "tank",
}

# Canonical process order for V2 stages (left to right layout)
V2_CANONICAL_ORDER = [
    "input", "stockpile", "feeder", "conveyor",
    "primary_crusher", "secondary_crusher", "hpgr", "ore_sorting",
    "sag_mill", "ball_mill", "cyclone", "screen",
    "rougher_flotation", "regrind", "cleaner_flotation",
    "gravity_concentration", "magnetic_separation",
    "agglomeration", "leach", "adsorption", "elution", "carbon_regeneration",
    "solvent_extraction", "electrowinning", "precipitation",
    "ion_exchange", "ccd", "kiln", "smelting",
    "concentrate_thickener", "tailings_thickener", "filter",
    "bin", "tank", "solution_pond",
    "concentrate_product", "water_treatment", "tailing",
]

# ---------------------------------------------------------------------------
# Color scheme (same as mini/pfs_visual.py)
# ---------------------------------------------------------------------------
C_TP = "#1f4454"
C_FP = "#90CAF9"
C_FN = "#555555"
C_EDGE_TP = "#1f4454"
C_EDGE_FP = "#90CAF9"
C_EDGE_FN = "#555555"
C_EDGE_DEFAULT = "#757575"
C_BG = "#FAFAFA"

SYM_W = 1.6
SYM_H = 1.0


# ---------------------------------------------------------------------------
# Equipment symbol drawing functions (identical to mini/pfs_visual.py)
# ---------------------------------------------------------------------------

def _lighten(hex_color, amount=0.3):
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_input(ax, cx, cy, w, h, color):
    verts = [(cx - w/2, cy + h/2), (cx + w/2, cy + h/2),
             (cx + w/3, cy - h/2), (cx - w/3, cy - h/2)]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))

def _draw_crusher(ax, cx, cy, w, h, color):
    hw, hh = w/2, h/2
    ax.add_patch(FancyBboxPatch((cx - hw, cy - hh), w, h,
                                boxstyle="round,pad=0.05", fc=color, ec="black", lw=1.5))
    ax.plot([cx - w*0.3, cx], [cy + h*0.35, cy - h*0.2], color="black", lw=1.5)
    ax.plot([cx + w*0.3, cx], [cy + h*0.35, cy - h*0.2], color="black", lw=1.5)

def _draw_stockpile(ax, cx, cy, w, h, color):
    verts = [(cx - w/2, cy - h/2), (cx, cy + h/2), (cx + w/2, cy - h/2)]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))

def _draw_mill(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h*0.35), w, h*0.7,
                                boxstyle="round,pad=0.1", fc=color, ec="black", lw=1.5))
    ax.plot([cx - w/2 + 0.1, cx - w/2 + 0.1], [cy - h*0.3, cy + h*0.3], color="black", lw=1.2)
    ax.plot([cx + w/2 - 0.1, cx + w/2 - 0.1], [cy - h*0.3, cy + h*0.3], color="black", lw=1.2)

def _draw_cyclone(ax, cx, cy, w, h, color):
    verts = [(cx - w*0.35, cy + h/2), (cx + w*0.35, cy + h/2), (cx, cy - h/2)]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))
    ax.plot([cx, cx], [cy + h/2, cy + h*0.65], color="black", lw=1.2)

def _draw_screen(ax, cx, cy, w, h, color):
    tilt = 0.15
    verts = [(cx - w/2, cy - h/2 + tilt), (cx + w/2, cy - h/2 - tilt),
             (cx + w/2, cy + h/2 - tilt), (cx - w/2, cy + h/2 + tilt)]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))
    for frac in [0.33, 0.66]:
        y_off = tilt * (1 - 2*frac)
        ax.plot([cx - w*0.4, cx + w*0.4], [cy + y_off, cy + y_off - tilt*0.5],
                color="black", lw=0.8, ls="--")

def _draw_flotation(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color, ec="black", lw=1.5))
    for dx, dy in [(-0.2, 0.1), (0.15, -0.05), (0, 0.2), (0.25, 0.15)]:
        ax.add_patch(Circle((cx + dx, cy + dy), 0.06, fc="white", ec="black", lw=0.5, alpha=0.7))

def _draw_regrind(ax, cx, cy, w, h, color):
    rw, rh = w*0.85, h*0.65
    ax.add_patch(FancyBboxPatch((cx - rw/2, cy - rh/2), rw, rh,
                                boxstyle="round,pad=0.08", fc=color, ec="black", lw=1.5))
    ax.plot([cx - rw/2 + 0.08, cx - rw/2 + 0.08], [cy - rh*0.4, cy + rh*0.4], color="black", lw=1)

def _draw_thickener(ax, cx, cy, w, h, color):
    ell = mpatches.Ellipse((cx, cy), w*1.1, h*0.7, fc=color, ec="black", lw=1.5)
    ax.add_patch(ell)
    ax.plot([cx, cx + w*0.4], [cy, cy - h*0.1], color="black", lw=1)

def _draw_filter(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color, ec="black", lw=1.5))
    for i in range(3):
        x = cx - w/2 + (0.25 + i * 0.25) * w
        ax.plot([x, x], [cy - h*0.35, cy + h*0.35], color="black", lw=0.7, ls=":")

def _draw_leach(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color, ec="black", lw=1.5))
    for dx in [-0.25, 0, 0.25]:
        ax.plot([cx + dx, cx + dx], [cy + h*0.15, cy - h*0.15], color="black", lw=0.7, ls="--")

def _draw_adsorption(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="round,pad=0.05", fc=color, ec="black", lw=1.5))
    for dx, dy in [(-0.2, 0), (0.1, 0.15), (0.2, -0.1), (-0.05, -0.15)]:
        ax.add_patch(Circle((cx + dx, cy + dy), 0.05, fc="dimgray", ec="black", lw=0.4))

def _draw_solvent_extraction(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w/2, h,
                                boxstyle="square,pad=0", fc=color, ec="black", lw=1.5))
    lighter = _lighten(color, 0.3)
    ax.add_patch(FancyBboxPatch((cx, cy - h/2), w/2, h,
                                boxstyle="square,pad=0", fc=lighter, ec="black", lw=1.5))
    ax.plot([cx, cx], [cy - h/2, cy + h/2], color="black", lw=1.5)

def _draw_electrowinning(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color, ec="black", lw=1.5))
    for dx in [-0.3, -0.1, 0.1, 0.3]:
        ax.plot([cx + dx, cx + dx], [cy + h*0.3, cy - h*0.2], color="black", lw=1.2)

def _draw_precipitation(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color, ec="black", lw=1.5))
    for dx in [-0.25, -0.1, 0.05, 0.2, 0.3]:
        ax.plot(cx + dx, cy - h*0.3, "v", color="dimgray", ms=3)

def _draw_tank(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color, ec="black", lw=1.5))

def _draw_bin(ax, cx, cy, w, h, color):
    verts = [(cx - w*0.4, cy + h/2), (cx + w*0.4, cy + h/2),
             (cx + w*0.25, cy - h/2), (cx - w*0.25, cy - h/2)]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))

def _draw_tailing(ax, cx, cy, w, h, color):
    verts = [(cx - w*0.3, cy + h/2), (cx + w*0.3, cy + h/2),
             (cx + w/2, cy - h/2), (cx - w/2, cy - h/2)]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))
    xs = np.linspace(cx - w*0.25, cx + w*0.25, 20)
    ys = cy + h*0.15 + 0.04 * np.sin(xs * 8)
    ax.plot(xs, ys, color="steelblue", lw=1)

def _draw_treatment(ax, cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color, ec="black", lw=1.5))
    xs = np.linspace(cx - w*0.35, cx + w*0.35, 20)
    ys = cy + 0.04 * np.sin(xs * 10)
    ax.plot(xs, ys, color="steelblue", lw=1.2)

def _draw_output(ax, cx, cy, w, h, color):
    verts = [(cx - w*0.35, cy + h/2), (cx + w*0.2, cy + h/2),
             (cx + w/2, cy), (cx + w*0.2, cy - h/2), (cx - w*0.35, cy - h/2)]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))

def _draw_ore_sorting(ax, cx, cy, w, h, color):
    _draw_screen(ax, cx, cy, w, h, color)

def _draw_conveyor(ax, cx, cy, w, h, color):
    rh = h * 0.4
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - rh/2), w, rh,
                                boxstyle="square,pad=0", fc=color, ec="black", lw=1.5))
    for dx in [-w*0.3, 0, w*0.3]:
        ax.add_patch(Circle((cx + dx, cy), rh*0.35, fc="white", ec="black", lw=0.8))

def _draw_gravity(ax, cx, cy, w, h, color):
    verts = [(cx - w/2, cy + h*0.3), (cx + w/2, cy + h/2),
             (cx + w/2, cy - h/2), (cx - w/2, cy - h*0.3)]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))


SYMBOL_FUNCS = {
    "input": _draw_input, "crusher": _draw_crusher, "ore_sorting": _draw_ore_sorting,
    "stockpile": _draw_stockpile, "conveyor": _draw_conveyor, "mill": _draw_mill,
    "cyclone": _draw_cyclone, "screen": _draw_screen, "flotation": _draw_flotation,
    "regrind": _draw_regrind, "gravity": _draw_gravity, "leach": _draw_leach,
    "adsorption": _draw_adsorption, "solvent_extraction": _draw_solvent_extraction,
    "electrowinning": _draw_electrowinning, "thickener": _draw_thickener,
    "filter": _draw_filter, "precipitation": _draw_precipitation,
    "tank": _draw_tank, "bin": _draw_bin, "tailing": _draw_tailing,
    "treatment": _draw_treatment, "output": _draw_output,
}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _unified_x_positions(gt_stages, pred_stages):
    """Assign x-positions so matching stages share the same column."""
    all_stages = set(gt_stages) | set(pred_stages)
    canon_lookup = {s: i for i, s in enumerate(V2_CANONICAL_ORDER)}
    ordered = sorted(all_stages, key=lambda s: canon_lookup.get(s, 999))
    return {stage: i * 3.0 for i, stage in enumerate(ordered)}


def _row_positions(stages, x_positions, y_center):
    """Position nodes in a single horizontal row."""
    return {stage: (x_positions.get(stage, 0), y_center) for stage in stages}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_edges(ax, connections, pos, edge_colors):
    """Draw directed arrows for connections."""
    for src, dst in connections:
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ec = edge_colors.get((src, dst), C_EDGE_DEFAULT)
        if abs(y0 - y1) < 0.1:
            dx = SYM_W/2 * 0.9 if x1 > x0 else -SYM_W/2 * 0.9
            start, end = (x0 + dx, y0), (x1 - dx, y1)
            rad = 0.3
        else:
            dy_s = -SYM_H/2 * 0.9 if y0 > y1 else SYM_H/2 * 0.9
            dy_e = SYM_H/2 * 0.9 if y0 > y1 else -SYM_H/2 * 0.9
            start, end = (x0, y0 + dy_s), (x1, y1 + dy_e)
            rad = 0.1
        arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                lw=2.0, color=ec, connectionstyle=f"arc3,rad={rad}", zorder=1)
        ax.add_patch(arrow)


def _draw_nodes(ax, stages, pos, node_colors, label_side="below"):
    """Draw equipment symbols with labels."""
    for stage in stages:
        if stage not in pos:
            continue
        cx, cy = pos[stage]
        symbol = V2_TO_SYMBOL.get(stage, "tank")
        color = node_colors.get(stage, C_FP)
        draw_fn = SYMBOL_FUNCS.get(symbol, _draw_tank)
        draw_fn(ax, cx, cy, SYM_W, SYM_H, color)

        label = stage.replace("_", " ").title()
        if label_side == "above":
            ax.text(cx, cy + SYM_H/2 + 0.2, label, ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#333333", zorder=5)
        else:
            ax.text(cx, cy - SYM_H/2 - 0.2, label, ha="center", va="top",
                    fontsize=11, fontweight="bold", color="#333333", zorder=5)


def _draw_alignment_lines(ax, gt_pos, pred_pos, gt_stages, pred_stages):
    """Dashed vertical lines between matching stages."""
    for stage in set(gt_stages) & set(pred_stages):
        if stage in gt_pos and stage in pred_pos:
            px, py = pred_pos[stage]
            gx, gy = gt_pos[stage]
            ax.plot([px, gx], [py - SYM_H/2 - 0.5, gy + SYM_H/2 + 0.5],
                    color="#BDBDBD", lw=0.8, ls=":", zorder=0)


# ---------------------------------------------------------------------------
# Diff logic
# ---------------------------------------------------------------------------

def _parse_connections(conn_strings):
    """Parse 'src -> dst' strings into (src, dst) tuples."""
    pairs = []
    for c in conn_strings:
        if " -> " in c:
            src, dst = c.split(" -> ", 1)
            pairs.append((src.strip(), dst.strip()))
    return pairs


def compute_diff(gt_stages, pred_stages, gt_connections, pred_connections):
    """Color-code nodes and edges by TP/FP/FN status.

    Returns (node_colors_gt, node_colors_pred, edge_colors_gt, edge_colors_pred).
    """
    gt_set = set(gt_stages)
    pred_set = set(pred_stages)

    node_colors_gt = {}
    for s in gt_stages:
        node_colors_gt[s] = C_TP if s in pred_set else C_FN

    node_colors_pred = {}
    for s in pred_stages:
        node_colors_pred[s] = C_TP if s in gt_set else C_FP

    gt_edge_set = set(gt_connections)
    pred_edge_set = set(pred_connections)

    edge_colors_gt = {}
    for e in gt_connections:
        edge_colors_gt[e] = C_EDGE_TP if e in pred_edge_set else C_EDGE_FN

    edge_colors_pred = {}
    for e in pred_connections:
        edge_colors_pred[e] = C_EDGE_TP if e in gt_edge_set else C_EDGE_FP

    return node_colors_gt, node_colors_pred, edge_colors_gt, edge_colors_pred


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_comparison(doc_entry, output_file=None, subtitle=None):
    """Render a single doc's ground truth vs predicted flowsheet.

    Args:
        doc_entry: dict from llm_eval_entry per_doc results, with keys:
            - doc_id, ground_truth, final_predicted,
            - ground_truth_connections, final_connections,
            - correct, missed, false_positive, doc_f1
        output_file: optional Path for output PNG
        subtitle: optional subtitle text (e.g. grade info)
    """
    doc_id = doc_entry["doc_id"]
    gt_stages = [s for s in doc_entry["ground_truth"]
                 if s not in ("input", "output")]
    pred_stages = [s for s in doc_entry["final_predicted"]
                   if s not in ("input", "output")]

    gt_conns = _parse_connections(doc_entry.get("ground_truth_connections", []))
    pred_conns = _parse_connections(doc_entry.get("final_connections", []))

    if output_file is None:
        output_file = Path(f"pfs_{doc_id[:30]}.png")

    # Diff
    nc_gt, nc_pred, ec_gt, ec_pred = compute_diff(
        gt_stages, pred_stages, gt_conns, pred_conns)

    # Layout
    x_positions = _unified_x_positions(gt_stages, pred_stages)
    n_cols = max(len(x_positions), 1)
    row_gap = 6.0
    y_pred = row_gap / 2
    y_gt = -row_gap / 2

    pos_pred = _row_positions(pred_stages, x_positions, y_pred)
    pos_gt = _row_positions(gt_stages, x_positions, y_gt)

    # Draw
    fig_w = max(16, n_cols * 3.0)
    fig_h = max(8, row_gap + 6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(C_BG)

    _draw_alignment_lines(ax, pos_gt, pos_pred, gt_stages, pred_stages)

    # Row separator
    all_x = [p[0] for p in list(pos_pred.values()) + list(pos_gt.values())] or [0]
    x_min = min(all_x) - 2
    ax.axhline(y=0, xmin=0, xmax=1, color="#E0E0E0", lw=1.5, ls="-", zorder=0)

    ax.text(x_min - 1.0, y_pred, "PREDICTED", ha="right", va="center",
            fontsize=18, fontweight="bold", color="#1565C0", rotation=90)
    ax.text(x_min - 1.0, y_gt, "GROUND TRUTH", ha="right", va="center",
            fontsize=18, fontweight="bold", color="#2E7D32", rotation=90)

    _draw_edges(ax, pred_conns, pos_pred, ec_pred)
    _draw_edges(ax, gt_conns, pos_gt, ec_gt)
    _draw_nodes(ax, pred_stages, pos_pred, nc_pred, label_side="above")
    _draw_nodes(ax, gt_stages, pos_gt, nc_gt, label_side="below")

    ax.set_aspect("equal")
    ax.axis("off")

    all_pos = list(pos_pred.values()) + list(pos_gt.values())
    if all_pos:
        all_x_vals = [p[0] for p in all_pos]
        all_y_vals = [p[1] for p in all_pos]
        ax.set_xlim(min(all_x_vals) - 3.0, max(all_x_vals) + 3.0)
        ax.set_ylim(min(all_y_vals) - 2.5, max(all_y_vals) + 2.5)

    plt.tight_layout(rect=[0.05, 0.10, 1, 0.90])

    # Title
    fig.text(0.5, 0.96, "PFS Comparison — Predicted vs Ground Truth",
             ha="center", va="top", fontsize=20, fontweight="bold")
    subtitle_text = subtitle or f"Document: {doc_id}"
    fig.text(0.5, 0.92, subtitle_text,
             ha="center", va="top", fontsize=16, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_TP, edgecolor="black", label="True Positive"),
        mpatches.Patch(facecolor=C_FP, edgecolor="black", label="False Positive"),
        mpatches.Patch(facecolor=C_FN, edgecolor="black", label="False Negative"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=16, frameon=True, fancybox=True, bbox_to_anchor=(0.5, 0.05))

    # Stats
    gt_set = set(gt_stages)
    pred_set = set(pred_stages)
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    stats = (f"Stages — TP:{tp} FP:{fp} FN:{fn}  "
             f"P={prec:.0%} R={rec:.0%} F1={f1:.0%}")
    fig.text(0.5, 0.02, stats, ha="center", fontsize=16, style="italic", color="dimgray")

    plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PFS comparison saved to {output_file}")
    return output_file


def _extract_cu_grade(graph):
    """Extract Cu head grade from a graph's resource_estimate context nodes.

    Looks for resource_estimate sections where num_mean is in a plausible
    Cu grade range (0.01–10%).  Falls back to grade_cu_mean if available.
    """
    for node in graph.get("nodes", []):
        if node.get("type") != "context":
            continue
        feats = node.get("features", {})
        group = node.get("group", "")

        # Prefer explicit Cu grade feature
        cu_grade = feats.get("grade_cu_mean", 0)
        if cu_grade and 0.01 < cu_grade < 10.0:
            return cu_grade

        # resource_estimate num_mean is typically ore grade %
        if group == "resource_estimate":
            num_mean = feats.get("num_mean", 0)
            if 0.01 < num_mean < 10.0:
                return num_mean

    return None


def _load_graph_grades(eval_doc_ids):
    """Load Cu grades from S3 graphs for the given doc IDs.

    Returns dict {doc_id: cu_grade_pct}.
    """
    import boto3
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import SAGEMAKER_BUCKET

    s3 = boto3.client("s3")
    grades = {}
    for doc_id in eval_doc_ids:
        try:
            resp = s3.get_object(Bucket=SAGEMAKER_BUCKET,
                                 Key=f"graphs/{doc_id}.json")
            graph = json.loads(resp["Body"].read())
            grade = _extract_cu_grade(graph)
            if grade is not None:
                grades[doc_id] = grade
        except Exception:
            pass
    return grades


def find_nearest_grade_doc(per_doc, target_grade, grades_map):
    """Find the eval doc entry with the closest Cu grade to target.

    Args:
        per_doc: list of per-doc eval entries
        target_grade: target Cu% value
        grades_map: dict {doc_id: cu_grade} from _load_graph_grades

    Returns (entry, grade) or (None, None).
    """
    best_entry = None
    best_grade = None
    best_diff = float("inf")

    for entry in per_doc:
        grade = grades_map.get(entry["doc_id"])
        if grade is None:
            continue
        diff = abs(grade - target_grade)
        if diff < best_diff:
            best_diff = diff
            best_entry = entry
            best_grade = grade

    return best_entry, best_grade


def render_by_grade(eval_results_path, target_grade, output_file=None):
    """Find nearest-grade doc in eval results and render PFS comparison.

    Loads Cu grades directly from S3 graph resource_estimate nodes to
    get accurate values (not the economics num_mean fallback).

    Args:
        eval_results_path: path to llm_eval_results.json
        target_grade: target Cu head grade in percent (e.g. 0.50)
        output_file: optional output path (default: pfs_compare_{grade}.png)

    Returns:
        output file Path
    """
    with open(eval_results_path) as f:
        results = json.load(f)

    per_doc = results.get("per_doc", [])
    doc_ids = [d["doc_id"] for d in per_doc]

    print(f"Loading grades from S3 for {len(doc_ids)} eval docs...")
    grades_map = _load_graph_grades(doc_ids)
    print(f"Found Cu grades for {len(grades_map)}/{len(doc_ids)} docs")

    for doc_id, grade in sorted(grades_map.items(), key=lambda x: x[1]):
        print(f"  Cu={grade:.4f}%  {doc_id[:40]}")

    entry, actual_grade = find_nearest_grade_doc(per_doc, target_grade, grades_map)

    if entry is None:
        print("No documents with valid Cu grade found in eval results")
        return None

    doc_id = entry["doc_id"]
    params = entry.get("source_params", {})
    deposit = params.get("deposit_type", "")
    ore = params.get("ore_type", "")

    print(f"\nTarget grade: {target_grade:.2f}% Cu")
    print(f"Nearest doc:  {doc_id} (Cu={actual_grade:.4f}%)")
    if deposit:
        print(f"  Deposit: {deposit}")
    if ore:
        print(f"  Ore type: {ore}")
    print(f"  Ground truth: {sorted(entry.get('ground_truth', []))}")
    print(f"  Predicted:    {sorted(entry.get('final_predicted', []))}")

    if output_file is None:
        output_file = Path(eval_results_path).parent / f"pfs_compare_{target_grade:.2f}.png"

    return render_comparison(entry, output_file=output_file,
                             subtitle=f"Cu = {target_grade:.2f}%  |  Nearest: {doc_id} (Cu = {actual_grade:.4f}%)")


def render_all(eval_results_path, output_dir=None, max_docs=None):
    """Render PFS comparisons for all docs in an eval results file.

    Args:
        eval_results_path: path to llm_eval_results.json
        output_dir: directory for output PNGs (default: same dir as input)
        max_docs: limit number of docs to render
    """
    with open(eval_results_path) as f:
        results = json.load(f)

    per_doc = results.get("per_doc", [])
    if max_docs:
        per_doc = per_doc[:max_docs]

    if output_dir is None:
        output_dir = Path(eval_results_path).parent / "pfs_comparisons"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in per_doc:
        doc_id = entry["doc_id"]
        safe_name = doc_id[:40].replace("/", "_").replace("\\", "_")
        out_path = output_dir / f"pfs_{safe_name}.png"
        try:
            render_comparison(entry, output_file=out_path)
        except Exception as e:
            print(f"Failed to render {doc_id}: {e}")

    print(f"\nRendered {len(per_doc)} PFS comparisons to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PFS visualization of eval results")
    parser.add_argument("eval_file", help="Path to llm_eval_results.json")
    parser.add_argument("--grade", type=float,
                        help="Input Cu grade (%%) — finds nearest-grade doc and renders comparison")
    parser.add_argument("--doc", help="Render only this doc_id")
    parser.add_argument("--out", help="Output file or directory")
    parser.add_argument("--max", type=int, help="Max docs to render (for batch mode)")
    args = parser.parse_args()

    if args.grade is not None:
        out = Path(args.out) if args.out else None
        render_by_grade(args.eval_file, args.grade, output_file=out)
    elif args.doc:
        with open(args.eval_file) as f:
            results = json.load(f)
        entry = next((d for d in results["per_doc"] if d["doc_id"] == args.doc), None)
        if entry is None:
            print(f"Doc '{args.doc}' not found in results")
            sys.exit(1)
        out = Path(args.out) / f"pfs_{args.doc[:40]}.png" if args.out else None
        render_comparison(entry, output_file=out)
    else:
        render_all(args.eval_file, output_dir=args.out, max_docs=args.max)
