"""PFS-style process flowsheet visualization with comparison.

Renders digital twin predicted flowsheets using standard mineral processing
equipment symbols, and compares against the nearest-grade actual document
from results.json with color-coded alignment (green=match, red=extra,
orange=missed).

Usage:
    python mini/pfs_visual.py 0.50
"""

import json
import sys
from pathlib import Path
from collections import Counter, deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Polygon, FancyBboxPatch, Circle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from mini.normalize import normalize_stage, CANONICAL_STAGES
from mini.digital_twin import (
    load_training_data, build_feature_matrix, fit_model,
    postprocess, build_predicted_graph, select_primary_flowsheet,
    CANONICAL_ORDER, GRADE_MIN, GRADE_MAX, RESULTS_PATH, MINI_DIR,
)

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
C_TP = "#1f4454"          # Blueshift teal — true positive (in both)
C_FP = "#90CAF9"          # light blue — false positive (predicted only)
C_FN = "#555555"          # dark grey — false negative (actual only)
C_DEFAULT = "#90CAF9"     # light blue — no comparison
C_EDGE_TP = "#1f4454"
C_EDGE_FP = "#90CAF9"
C_EDGE_FN = "#555555"
C_EDGE_DEFAULT = "#757575"
C_BG = "#FAFAFA"

# Symbol dimensions
SYM_W = 1.6
SYM_H = 1.0


# ---------------------------------------------------------------------------
# Equipment symbol drawing functions
# Each draws at center (cx, cy) with given width/height and facecolor.
# ---------------------------------------------------------------------------

def _draw_input(ax, cx, cy, w, h, color):
    """Hopper / feed bin — downward trapezoid."""
    verts = [
        (cx - w/2, cy + h/2),
        (cx + w/2, cy + h/2),
        (cx + w/3, cy - h/2),
        (cx - w/3, cy - h/2),
    ]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))


def _draw_crusher(ax, cx, cy, w, h, color):
    """Jaw crusher — V-shape with housing."""
    # Outer housing
    hw, hh = w/2, h/2
    ax.add_patch(FancyBboxPatch((cx - hw, cy - hh), w, h,
                                boxstyle="round,pad=0.05", fc=color,
                                ec="black", lw=1.5))
    # V-jaw lines inside
    ax.plot([cx - w*0.3, cx], [cy + h*0.35, cy - h*0.2],
            color="black", lw=1.5)
    ax.plot([cx + w*0.3, cx], [cy + h*0.35, cy - h*0.2],
            color="black", lw=1.5)


def _draw_stockpile(ax, cx, cy, w, h, color):
    """Stockpile — triangle / mountain shape."""
    verts = [
        (cx - w/2, cy - h/2),
        (cx, cy + h/2),
        (cx + w/2, cy - h/2),
    ]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))


def _draw_mill(ax, cx, cy, w, h, color):
    """SAG/ball mill — horizontal cylinder."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h*0.35), w, h*0.7,
                                boxstyle="round,pad=0.1", fc=color,
                                ec="black", lw=1.5))
    # End caps
    ax.plot([cx - w/2 + 0.1, cx - w/2 + 0.1],
            [cy - h*0.3, cy + h*0.3], color="black", lw=1.2)
    ax.plot([cx + w/2 - 0.1, cx + w/2 - 0.1],
            [cy - h*0.3, cy + h*0.3], color="black", lw=1.2)


def _draw_cyclone(ax, cx, cy, w, h, color):
    """Hydrocyclone — inverted cone / triangle."""
    verts = [
        (cx - w*0.35, cy + h/2),
        (cx + w*0.35, cy + h/2),
        (cx, cy - h/2),
    ]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))
    # Vortex finder line at top
    ax.plot([cx, cx], [cy + h/2, cy + h*0.65], color="black", lw=1.2)


def _draw_screen(ax, cx, cy, w, h, color):
    """Vibrating screen — angled rectangle with mesh lines."""
    # Tilted rectangle
    tilt = 0.15
    verts = [
        (cx - w/2, cy - h/2 + tilt),
        (cx + w/2, cy - h/2 - tilt),
        (cx + w/2, cy + h/2 - tilt),
        (cx - w/2, cy + h/2 + tilt),
    ]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))
    # Mesh lines
    for frac in [0.33, 0.66]:
        y_off = tilt * (1 - 2*frac)
        ax.plot([cx - w*0.4, cx + w*0.4],
                [cy + y_off, cy + y_off - tilt*0.5],
                color="black", lw=0.8, ls="--")


def _draw_flotation(ax, cx, cy, w, h, color):
    """Flotation cell — rectangular tank with bubbles."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color,
                                ec="black", lw=1.5))
    # Bubbles
    for dx, dy in [(-0.2, 0.1), (0.15, -0.05), (0, 0.2), (0.25, 0.15)]:
        ax.add_patch(Circle((cx + dx, cy + dy), 0.06, fc="white",
                            ec="black", lw=0.5, alpha=0.7))


def _draw_regrind(ax, cx, cy, w, h, color):
    """Regrind mill — smaller cylinder."""
    rw, rh = w*0.85, h*0.65
    ax.add_patch(FancyBboxPatch((cx - rw/2, cy - rh/2), rw, rh,
                                boxstyle="round,pad=0.08", fc=color,
                                ec="black", lw=1.5))
    ax.plot([cx - rw/2 + 0.08, cx - rw/2 + 0.08],
            [cy - rh*0.4, cy + rh*0.4], color="black", lw=1)


def _draw_thickener(ax, cx, cy, w, h, color):
    """Thickener — wide ellipse (plan view)."""
    ell = mpatches.Ellipse((cx, cy), w*1.1, h*0.7, fc=color,
                           ec="black", lw=1.5)
    ax.add_patch(ell)
    # Rake arm
    ax.plot([cx, cx + w*0.4], [cy, cy - h*0.1], color="black", lw=1)


def _draw_filter(ax, cx, cy, w, h, color):
    """Filter press — rectangle with cross-hatch."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color,
                                ec="black", lw=1.5))
    # Hatch lines
    for i in range(3):
        frac = 0.25 + i * 0.25
        x = cx - w/2 + frac * w
        ax.plot([x, x], [cy - h*0.35, cy + h*0.35],
                color="black", lw=0.7, ls=":")


def _draw_leach(ax, cx, cy, w, h, color):
    """Leach tank/heap — rectangle with drip lines."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color,
                                ec="black", lw=1.5))
    # Drip lines
    for dx in [-0.25, 0, 0.25]:
        ax.plot([cx + dx, cx + dx], [cy + h*0.15, cy - h*0.15],
                color="black", lw=0.7, ls="--")


def _draw_adsorption(ax, cx, cy, w, h, color):
    """CIP/CIL adsorption — tank with carbon dots."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="round,pad=0.05", fc=color,
                                ec="black", lw=1.5))
    # Carbon granule dots
    for dx, dy in [(-0.2, 0), (0.1, 0.15), (0.2, -0.1), (-0.05, -0.15)]:
        ax.add_patch(Circle((cx + dx, cy + dy), 0.05, fc="dimgray",
                            ec="black", lw=0.4))


def _draw_solvent_extraction(ax, cx, cy, w, h, color):
    """Mixer-settler — split rectangle."""
    # Mixer (left half)
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w/2, h,
                                boxstyle="square,pad=0", fc=color,
                                ec="black", lw=1.5))
    # Settler (right half)
    lighter = _lighten(color, 0.3)
    ax.add_patch(FancyBboxPatch((cx, cy - h/2), w/2, h,
                                boxstyle="square,pad=0", fc=lighter,
                                ec="black", lw=1.5))
    # Divider
    ax.plot([cx, cx], [cy - h/2, cy + h/2], color="black", lw=1.5)


def _draw_electrowinning(ax, cx, cy, w, h, color):
    """Electrowinning — rectangle with electrode lines."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color,
                                ec="black", lw=1.5))
    # Electrode plates
    for dx in [-0.3, -0.1, 0.1, 0.3]:
        ax.plot([cx + dx, cx + dx], [cy + h*0.3, cy - h*0.2],
                color="black", lw=1.2)


def _draw_precipitation(ax, cx, cy, w, h, color):
    """Precipitation tank — rectangle with settling particles."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color,
                                ec="black", lw=1.5))
    # Settling particles at bottom
    for dx in [-0.25, -0.1, 0.05, 0.2, 0.3]:
        ax.plot(cx + dx, cy - h*0.3, "v", color="dimgray", ms=3)


def _draw_tank(ax, cx, cy, w, h, color):
    """Generic tank — simple rectangle."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color,
                                ec="black", lw=1.5))


def _draw_bin(ax, cx, cy, w, h, color):
    """Bin/hopper — small downward trapezoid."""
    verts = [
        (cx - w*0.4, cy + h/2),
        (cx + w*0.4, cy + h/2),
        (cx + w*0.25, cy - h/2),
        (cx - w*0.25, cy - h/2),
    ]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))


def _draw_tailing(ax, cx, cy, w, h, color):
    """Tailings dam — wide trapezoid."""
    verts = [
        (cx - w*0.3, cy + h/2),
        (cx + w*0.3, cy + h/2),
        (cx + w/2, cy - h/2),
        (cx - w/2, cy - h/2),
    ]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))
    # Water line
    xs = np.linspace(cx - w*0.25, cx + w*0.25, 20)
    ys = cy + h*0.15 + 0.04 * np.sin(xs * 8)
    ax.plot(xs, ys, color="steelblue", lw=1)


def _draw_treatment(ax, cx, cy, w, h, color):
    """Water treatment — tank with wavy line."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="square,pad=0", fc=color,
                                ec="black", lw=1.5))
    # Wavy water line
    xs = np.linspace(cx - w*0.35, cx + w*0.35, 20)
    ys = cy + 0.04 * np.sin(xs * 10)
    ax.plot(xs, ys, color="steelblue", lw=1.2)


def _draw_output(ax, cx, cy, w, h, color):
    """Output — right-pointing chevron / arrow."""
    verts = [
        (cx - w*0.35, cy + h/2),
        (cx + w*0.2, cy + h/2),
        (cx + w/2, cy),
        (cx + w*0.2, cy - h/2),
        (cx - w*0.35, cy - h/2),
    ]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))


def _draw_ore_sorting(ax, cx, cy, w, h, color):
    """Ore sorting / DMS — angled screen variant."""
    _draw_screen(ax, cx, cy, w, h, color)


def _draw_conveyor(ax, cx, cy, w, h, color):
    """Conveyor — narrow horizontal rectangle with rollers."""
    rh = h * 0.4
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - rh/2), w, rh,
                                boxstyle="square,pad=0", fc=color,
                                ec="black", lw=1.5))
    # Rollers
    for dx in [-w*0.3, 0, w*0.3]:
        ax.add_patch(Circle((cx + dx, cy), rh*0.35, fc="white",
                            ec="black", lw=0.8))


def _draw_gravity(ax, cx, cy, w, h, color):
    """Gravity concentration — shaking table shape."""
    verts = [
        (cx - w/2, cy + h*0.3),
        (cx + w/2, cy + h/2),
        (cx + w/2, cy - h/2),
        (cx - w/2, cy - h*0.3),
    ]
    ax.add_patch(Polygon(verts, closed=True, fc=color, ec="black", lw=1.5))


# Symbol dispatch table
SYMBOL_FUNCS = {
    "input": _draw_input,
    "crusher": _draw_crusher,
    "ore_sorting": _draw_ore_sorting,
    "stockpile": _draw_stockpile,
    "conveyor": _draw_conveyor,
    "mill": _draw_mill,
    "cyclone": _draw_cyclone,
    "screen": _draw_screen,
    "flotation": _draw_flotation,
    "regrind": _draw_regrind,
    "gravity": _draw_gravity,
    "leach": _draw_leach,
    "adsorption": _draw_adsorption,
    "solvent_extraction": _draw_solvent_extraction,
    "electrowinning": _draw_electrowinning,
    "thickener": _draw_thickener,
    "filter": _draw_filter,
    "precipitation": _draw_precipitation,
    "tank": _draw_tank,
    "bin": _draw_bin,
    "tailing": _draw_tailing,
    "treatment": _draw_treatment,
    "output": _draw_output,
}


def _lighten(hex_color, amount=0.3):
    """Lighten a hex color by blending toward white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _stage_base(node_id):
    """Extract base stage type from node ID (e.g. 'flotation_2' -> 'flotation')."""
    if node_id in ("INPUT", "OUTPUT"):
        return node_id.lower()
    # Strip trailing _N for multi-count stages
    parts = node_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return node_id


# ---------------------------------------------------------------------------
# Layout — unified column positions for vertical alignment
# ---------------------------------------------------------------------------

def _unified_x_positions(pred_G, actual_G):
    """Assign x-positions to each canonical stage type so that matching
    stages in both graphs share the same column.

    Returns dict {base_stage_type: x_position}.
    """
    # Collect all base stage types from both graphs, ordered by CANONICAL_ORDER
    all_bases = set()
    for n in pred_G.nodes:
        all_bases.add(_stage_base(n))
    for n in actual_G.nodes:
        all_bases.add(_stage_base(n))

    # Order by canonical sequence, unknowns at end
    canon_lookup = {s: i for i, s in enumerate(CANONICAL_ORDER)}
    canon_lookup["input"] = -1
    canon_lookup["output"] = len(CANONICAL_ORDER)
    ordered = sorted(all_bases, key=lambda s: canon_lookup.get(s, 999))

    x_positions = {}
    x_spacing = 3.0
    for i, stage in enumerate(ordered):
        x_positions[stage] = i * x_spacing
    return x_positions


def _row_layout(G, x_positions, y_center):
    """Position nodes in a single horizontal row at y_center,
    using the unified x_positions for vertical alignment.

    Multi-count nodes (e.g. flotation_1, flotation_2) get stacked
    vertically around y_center.
    """
    pos = {}
    # Group nodes by base stage type
    base_groups = {}
    for node in G.nodes:
        base = _stage_base(node)
        base_groups.setdefault(base, []).append(node)

    y_sub_offset = 1.3  # vertical offset between multi-count instances
    for base, nodes in base_groups.items():
        cx = x_positions.get(base, 0)
        if len(nodes) == 1:
            pos[nodes[0]] = (cx, y_center)
        else:
            # Stack vertically, centered on y_center
            nodes_sorted = sorted(nodes)
            for i, node in enumerate(nodes_sorted):
                y_off = (i - (len(nodes_sorted) - 1) / 2) * y_sub_offset
                pos[node] = (cx, y_center + y_off)
    return pos


# ---------------------------------------------------------------------------
# Drawing helpers for the stacked layout
# ---------------------------------------------------------------------------

def _draw_edges(ax, G, pos, edge_colors_map):
    """Draw edges with arrows."""
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ec = edge_colors_map.get((u, v), C_EDGE_DEFAULT)
        # Offset arrow start/end away from symbol center
        dy_start = -SYM_H/2 * 0.9 if y0 > y1 else SYM_H/2 * 0.9
        dy_end = SYM_H/2 * 0.9 if y0 > y1 else -SYM_H/2 * 0.9
        # For same-row edges, offset horizontally instead
        if abs(y0 - y1) < 0.1:
            dx = SYM_W/2 * 0.9 if x1 > x0 else -SYM_W/2 * 0.9
            start = (x0 + dx, y0)
            end = (x1 - dx, y1)
            rad = 0.3
        else:
            start = (x0, y0 + dy_start)
            end = (x1, y1 + dy_end)
            rad = 0.1
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=2.0,
            color=ec,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arrow)


def _draw_nodes(ax, G, pos, node_colors, label_side="below"):
    """Draw equipment symbols and labels."""
    for node in G.nodes:
        cx, cy = pos[node]
        base = _stage_base(node)
        color = node_colors.get(node, C_DEFAULT)
        draw_fn = SYMBOL_FUNCS.get(base, _draw_tank)
        draw_fn(ax, cx, cy, SYM_W, SYM_H, color)

        label = G.nodes[node].get("label", node)
        label = label.replace("_", " ").title()
        # Labels sit outside the symbol, so keep dark for readability
        # on the light background
        if label_side == "above":
            ax.text(cx, cy + SYM_H/2 + 0.2, label,
                    ha="center", va="bottom", fontsize=14,
                    fontweight="bold", color="#333333", zorder=5)
        else:
            ax.text(cx, cy - SYM_H/2 - 0.2, label,
                    ha="center", va="top", fontsize=14,
                    fontweight="bold", color="#333333", zorder=5)


def _draw_alignment_lines(ax, pred_pos, actual_pos, pred_G, actual_G):
    """Draw faint dashed vertical lines between matching stages."""
    pred_bases = {_stage_base(n): n for n in pred_G.nodes}
    actual_bases = {_stage_base(n): n for n in actual_G.nodes}
    for base in set(pred_bases) & set(actual_bases):
        px, py = pred_pos[pred_bases[base]]
        ax_, ay = actual_pos[actual_bases[base]]
        # They should share the same x, but draw the line regardless
        ax.plot([px, ax_], [py - SYM_H/2 - 0.5, ay + SYM_H/2 + 0.5],
                color="#BDBDBD", lw=0.8, ls=":", zorder=0)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def find_nearest_grade_doc(target_grade):
    """Find the document with the closest ore grade to target."""
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    best_doc = None
    best_diff = float("inf")
    for doc in data["documents"]:
        grade = doc.get("ore_grade_cu_pct")
        if grade is None:
            continue
        flowsheets = [fs for fs in doc.get("flowsheets", [])
                      if fs.get("stage_sequence")]
        if not flowsheets:
            continue
        diff = abs(grade - target_grade)
        if diff < best_diff:
            best_diff = diff
            best_doc = doc
    return best_doc


def build_actual_graph(doc):
    """Build a networkx DiGraph from an actual document's primary flowsheet."""
    import networkx as nx

    flowsheets = [fs for fs in doc.get("flowsheets", [])
                  if fs.get("stage_sequence")]
    primary = select_primary_flowsheet(flowsheets)

    # Normalize stages to canonical types
    canonical_set = set(CANONICAL_STAGES)
    stage_types = set()
    for s in primary.get("stage_sequence", []):
        ns = normalize_stage(s)
        if ns and ns in canonical_set:
            stage_types.add(ns)

    # Also collect from explicit stages list
    for stage in primary.get("stages", []):
        ns = normalize_stage(stage["id"])
        if ns and ns in canonical_set:
            stage_types.add(ns)

    G = nx.DiGraph()

    # Add INPUT/OUTPUT
    G.add_node("INPUT", label="INPUT")
    G.add_node("OUTPUT", label="OUTPUT")

    for st in stage_types:
        if st in ("input", "output"):
            continue
        G.add_node(st, label=st)

    # Stages that should never connect directly to OUTPUT (too early in process)
    _no_direct_output = {"crusher", "mill", "stockpile", "conveyor", "screen",
                         "cyclone", "bin", "tailing", "INPUT"}

    # Normalize and add connections
    for conn in primary.get("connections", []):
        parent = normalize_stage(conn["parent_id"])
        child = normalize_stage(conn["child_id"])
        if parent is None or child is None or parent == child:
            continue
        # Map input/output to INPUT/OUTPUT node names
        p_node = "INPUT" if parent == "input" else parent
        c_node = "OUTPUT" if child == "output" else child
        # Filter obviously invalid connections
        if c_node == "OUTPUT" and p_node in _no_direct_output:
            continue
        if c_node == "INPUT":
            continue
        if p_node in G and c_node in G:
            G.add_edge(p_node, c_node)

    # Connect orphans using canonical order
    ordered = ["INPUT"] + [s for s in CANONICAL_ORDER
                           if s not in ("input", "output") and s in G] + ["OUTPUT"]
    reachable = set()
    if "INPUT" in G:
        import networkx as nx2
        reachable = set(nx2.descendants(G, "INPUT")) | {"INPUT"}
    for idx, node in enumerate(ordered):
        if node in reachable:
            continue
        for prev_idx in range(idx - 1, -1, -1):
            prev = ordered[prev_idx]
            if prev in reachable:
                # Respect the same filter: don't connect blocked stages to OUTPUT
                if node == "OUTPUT" and prev in _no_direct_output:
                    continue
                G.add_edge(prev, node)
                reachable.add(node)
                break

    return G, primary.get("flowsheet_id", "unknown")


def compute_diff(pred_G, actual_G):
    """Compute stage and edge diffs between predicted and actual graphs.

    Returns (node_colors_pred, node_colors_actual,
             edge_colors_pred, edge_colors_actual).
    """
    pred_stages = {_stage_base(n) for n in pred_G.nodes}
    actual_stages = {_stage_base(n) for n in actual_G.nodes}

    # Node colors: TP (Blueshift), FP (light blue), FN (dark grey)
    node_colors_pred = {}
    for node in pred_G.nodes:
        base = _stage_base(node)
        if base in actual_stages:
            node_colors_pred[node] = C_TP   # true positive
        else:
            node_colors_pred[node] = C_FP   # false positive

    node_colors_actual = {}
    for node in actual_G.nodes:
        base = _stage_base(node)
        if base in pred_stages:
            node_colors_actual[node] = C_TP   # true positive
        else:
            node_colors_actual[node] = C_FN   # false negative

    # Edge comparison (by base stage types, not node IDs)
    pred_edges_base = {(_stage_base(u), _stage_base(v))
                       for u, v in pred_G.edges()}
    actual_edges_base = {(_stage_base(u), _stage_base(v))
                         for u, v in actual_G.edges()}

    edge_colors_pred = {}
    for u, v in pred_G.edges():
        base_edge = (_stage_base(u), _stage_base(v))
        if base_edge in actual_edges_base:
            edge_colors_pred[(u, v)] = C_EDGE_TP
        else:
            edge_colors_pred[(u, v)] = C_EDGE_FP

    edge_colors_actual = {}
    for u, v in actual_G.edges():
        base_edge = (_stage_base(u), _stage_base(v))
        if base_edge in pred_edges_base:
            edge_colors_actual[(u, v)] = C_EDGE_TP
        else:
            edge_colors_actual[(u, v)] = C_EDGE_FN

    return node_colors_pred, node_colors_actual, edge_colors_pred, edge_colors_actual


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_comparison(grade, output_file=None):
    """Predict flowsheet, find nearest actual, render stacked PFS
    with predicted on top, actual on bottom, vertically aligned."""
    import networkx as nx

    if output_file is None:
        output_file = MINI_DIR / f"pfs_compare_{grade:.2f}.png"

    # --- Predicted flowsheet ---
    training_data = load_training_data()
    X, Y, stage_types, edge_keys = build_feature_matrix(training_data)
    model = fit_model(X, Y)
    raw = model.predict(np.array([[grade]]))[0]
    predicted_stages, predicted_edges = postprocess(raw, stage_types, edge_keys)
    pred_G = build_predicted_graph(predicted_stages, predicted_edges)

    print(f"Predicted stages: {sorted(predicted_stages.keys())}")
    print(f"Predicted connections: {len(predicted_edges)}")

    # --- Actual flowsheet ---
    doc = find_nearest_grade_doc(grade)
    actual_grade = doc["ore_grade_cu_pct"]
    doc_id = doc["document_id"]
    actual_G, fs_id = build_actual_graph(doc)

    print(f"Nearest doc: {doc_id} (Cu={actual_grade:.4f}%)")
    print(f"Actual stages: {sorted(n for n in actual_G.nodes if n not in ('INPUT', 'OUTPUT'))}")
    print(f"Actual connections: {actual_G.number_of_edges()}")

    # --- Diff ---
    nc_pred, nc_actual, ec_pred, ec_actual = compute_diff(pred_G, actual_G)

    # --- Unified column layout ---
    x_positions = _unified_x_positions(pred_G, actual_G)
    n_cols = len(x_positions)
    row_gap = 6.0  # vertical gap between predicted and actual rows
    y_pred = row_gap / 2      # predicted row (top)
    y_actual = -row_gap / 2   # actual row (bottom)

    pos_pred = _row_layout(pred_G, x_positions, y_pred)
    pos_actual = _row_layout(actual_G, x_positions, y_actual)

    # --- Draw on single axes ---
    fig_w = max(16, n_cols * 3.0)
    fig_h = max(8, row_gap + 6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(C_BG)

    # Alignment lines first (behind everything)
    _draw_alignment_lines(ax, pos_pred, pos_actual, pred_G, actual_G)

    # Row separator
    all_x = [p[0] for p in list(pos_pred.values()) + list(pos_actual.values())]
    x_min, x_max = min(all_x) - 2, max(all_x) + 2
    ax.axhline(y=0, xmin=0, xmax=1, color="#E0E0E0", lw=1.5, ls="-", zorder=0)

    # Row labels
    ax.text(x_min - 1.0, y_pred, "PREDICTED", ha="right", va="center",
            fontsize=18, fontweight="bold", color="#1565C0", rotation=90)
    ax.text(x_min - 1.0, y_actual, "ACTUAL", ha="right", va="center",
            fontsize=18, fontweight="bold", color="#2E7D32", rotation=90)

    # Draw edges
    _draw_edges(ax, pred_G, pos_pred, ec_pred)
    _draw_edges(ax, actual_G, pos_actual, ec_actual)

    # Draw nodes — predicted labels above, actual labels below
    _draw_nodes(ax, pred_G, pos_pred, nc_pred, label_side="above")
    _draw_nodes(ax, actual_G, pos_actual, nc_actual, label_side="below")

    ax.set_aspect("equal")
    ax.axis("off")

    # Auto-scale
    all_y = [p[1] for p in list(pos_pred.values()) + list(pos_actual.values())]
    margin_x, margin_y = 3.0, 2.5
    ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
    ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)

    plt.tight_layout(rect=[0.05, 0.10, 1, 0.90])

    # --- All text via fig coordinates for consistent centering ---

    # Title
    fig.text(0.5, 0.96, "PFS Comparison — Predicted vs Actual",
             ha="center", va="top", fontsize=20, fontweight="bold")
    fig.text(0.5, 0.92, f"Cu = {grade:.2f}%  |  Nearest: {doc_id} (Cu = {actual_grade:.4f}%)",
             ha="center", va="top", fontsize=16, fontweight="bold")

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor=C_TP, edgecolor="black", label="True Positive"),
        mpatches.Patch(facecolor=C_FP, edgecolor="black", label="False Positive"),
        mpatches.Patch(facecolor=C_FN, edgecolor="black", label="False Negative"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=16, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 0.05))

    # --- F1 / Stats ---
    pred_stages_set = {_stage_base(n) for n in pred_G.nodes}
    actual_stages_set = {_stage_base(n) for n in actual_G.nodes}
    tp_stages = pred_stages_set & actual_stages_set
    fp_stages = pred_stages_set - actual_stages_set
    fn_stages = actual_stages_set - pred_stages_set

    def _f1(tp, fp, fn):
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec

    s_f1, s_prec, s_rec = _f1(len(tp_stages), len(fp_stages), len(fn_stages))

    stats = (f"Stages — TP:{len(tp_stages)} FP:{len(fp_stages)} FN:{len(fn_stages)}  "
             f"P={s_prec:.0%} R={s_rec:.0%} F1={s_f1:.0%}")
    fig.text(0.5, 0.02, stats, ha="center", fontsize=16, style="italic",
             color="dimgray")
    plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPFS comparison saved to {output_file}")
    return output_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mini/pfs_visual.py <grade>")
        print("  e.g. python mini/pfs_visual.py 0.50")
        sys.exit(1)
    grade = float(sys.argv[1])
    render_comparison(grade)
