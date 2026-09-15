"""Modern, native inline-SVG flowsheet for the KNN trace demo.

Reuses the existing predicted-flowsheet graph logic (pfs_render.build_knn_graph
+ KNN_ORDER, which mirror mini/pfs_visual's topology/orphan rules) but renders
it as crisp, theme-aware SVG instead of a matplotlib raster: rounded equipment
nodes with a minimal line glyph per stage, phase-accent colour, and smooth
feed-forward / dashed-recycle connectors. No scripts, no external assets.
"""
from __future__ import annotations

import html

from quantitative.pfs_render import build_knn_graph, _x_positions, KNN_ORDER
from mini.pfs_visual import _stage_base

# stage -> process phase (drives node accent colour)
_PHASES = {
    0: ["input", "feeder", "bin", "stockpile", "hopper"],
    1: ["crusher", "mill", "screen", "cyclone", "regrind", "gravity",
        "ore_sorting", "agglomeration"],
    2: ["flotation", "conditioner", "rougher", "cleaner", "scavenger"],
    3: ["thickener", "filter", "tank", "clarifier", "treatment"],
    4: ["leach", "adsorption", "elution", "solution_recovery", "electrowinning",
        "sx", "kiln", "precipitation", "product_recovery"],
    5: ["tailing", "tailings", "output", "concentrate", "product"],
}
_STAGE_PHASE = {s: p for p, ss in _PHASES.items() for s in ss}


def _phase(base: str) -> int:
    if base in ("INPUT", "input"):
        return 0
    if base in ("OUTPUT", "output"):
        return 5
    return _STAGE_PHASE.get(base, 3)


def _label(raw: str) -> str:
    s = raw.replace("\n", " ").replace("_", " ").strip()
    if s.upper() in ("INPUT", "OUTPUT"):
        return s.title()
    return s


# --- minimal equipment glyphs, drawn in a 26-wide box centred at (cx, cy) ------
# All use stroke="currentColor" (node inherits its phase colour) so they stay
# crisp and theme-aware. Kept deliberately simple and consistent.

def _g(base: str, cx: float, cy: float) -> str:
    r = 9.0
    x0, y0 = cx - r, cy - r
    x1, y1 = cx + r, cy + r
    b = _stage_base(base)
    if b in ("INPUT", "input", "bin", "feeder", "stockpile", "hopper"):
        return (f'<path d="M{x0:.0f},{y0:.0f} L{x1:.0f},{y0:.0f} '
                f'L{cx + r*0.55:.0f},{y1:.0f} L{cx - r*0.55:.0f},{y1:.0f} Z"/>')
    if b == "crusher":
        return (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{2*r:.0f}" '
                f'height="{2*r:.0f}" rx="2"/>'
                f'<path d="M{x0+3:.0f},{y0+2:.0f} L{cx:.0f},{y1-3:.0f} '
                f'L{x1-3:.0f},{y0+2:.0f}"/>')
    if b in ("mill", "regrind", "kiln"):
        return (f'<rect x="{x0:.0f}" y="{cy-r*0.7:.0f}" width="{2*r:.0f}" '
                f'height="{r*1.4:.0f}" rx="{r*0.7:.0f}"/>'
                f'<line x1="{cx-r*0.3:.0f}" y1="{cy-r*0.5:.0f}" '
                f'x2="{cx-r*0.3:.0f}" y2="{cy+r*0.5:.0f}"/>')
    if b in ("screen", "ore_sorting"):
        return (f'<rect x="{x0:.0f}" y="{cy-r*0.55:.0f}" width="{2*r:.0f}" '
                f'height="{r*1.1:.0f}" rx="2" transform="rotate(-8 {cx} {cy})"/>'
                f'<line x1="{x0+2:.0f}" y1="{cy:.0f}" x2="{x1-2:.0f}" '
                f'y2="{cy:.0f}" stroke-dasharray="2 2"/>')
    if b == "cyclone":
        return (f'<path d="M{x0:.0f},{y0:.0f} L{x1:.0f},{y0:.0f} '
                f'L{cx:.0f},{y1:.0f} Z"/>')
    if b in ("flotation", "rougher", "cleaner", "scavenger", "conditioner"):
        return (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{2*r:.0f}" '
                f'height="{2*r:.0f}" rx="2"/>'
                f'<circle cx="{cx-3:.0f}" cy="{cy-2:.0f}" r="1.6"/>'
                f'<circle cx="{cx+2:.0f}" cy="{cy-4:.0f}" r="1.6"/>'
                f'<circle cx="{cx+4:.0f}" cy="{cy+1:.0f}" r="1.6"/>')
    if b in ("thickener", "clarifier"):
        return (f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{r:.0f}" '
                f'ry="{r*0.7:.0f}"/>'
                f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{cx+r*0.7:.0f}" '
                f'y2="{cy+r*0.35:.0f}"/>')
    if b == "filter":
        return (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{2*r:.0f}" '
                f'height="{2*r:.0f}" rx="2"/>'
                f'<line x1="{cx-3:.0f}" y1="{y0+3:.0f}" x2="{cx-3:.0f}" '
                f'y2="{y1-3:.0f}"/><line x1="{cx+3:.0f}" y1="{y0+3:.0f}" '
                f'x2="{cx+3:.0f}" y2="{y1-3:.0f}"/>')
    if b in ("tank", "treatment", "precipitation"):
        return (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{2*r:.0f}" '
                f'height="{2*r:.0f}" rx="2"/>'
                f'<path d="M{x0+2:.0f},{cy:.0f} q{r*0.5:.0f},-4 {r:.0f},0 '
                f't{r:.0f},0" fill="none"/>')
    if b in ("leach",):
        return (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{2*r:.0f}" '
                f'height="{2*r:.0f}" rx="2"/>'
                f'<line x1="{cx-3:.0f}" y1="{cy-1:.0f}" x2="{cx-3:.0f}" '
                f'y2="{cy+4:.0f}" stroke-dasharray="1.5 1.5"/>'
                f'<line x1="{cx+3:.0f}" y1="{cy-1:.0f}" x2="{cx+3:.0f}" '
                f'y2="{cy+4:.0f}" stroke-dasharray="1.5 1.5"/>')
    if b in ("adsorption",):
        return (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{2*r:.0f}" '
                f'height="{2*r:.0f}" rx="3"/>'
                f'<circle cx="{cx-3:.0f}" cy="{cy:.0f}" r="1.3" fill="currentColor"/>'
                f'<circle cx="{cx+2:.0f}" cy="{cy-3:.0f}" r="1.3" fill="currentColor"/>'
                f'<circle cx="{cx+3:.0f}" cy="{cy+2:.0f}" r="1.3" fill="currentColor"/>')
    if b in ("solution_recovery", "sx"):
        return (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{2*r:.0f}" '
                f'height="{2*r:.0f}" rx="2"/>'
                f'<line x1="{cx:.0f}" y1="{y0:.0f}" x2="{cx:.0f}" '
                f'y2="{y1:.0f}"/>')
    if b in ("electrowinning",):
        return (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{2*r:.0f}" '
                f'height="{2*r:.0f}" rx="2"/>'
                + "".join(f'<line x1="{cx+dx:.0f}" y1="{y0+3:.0f}" '
                          f'x2="{cx+dx:.0f}" y2="{y1-3:.0f}"/>'
                          for dx in (-4, -1.3, 1.3, 4)))
    if b in ("gravity",):
        return (f'<path d="M{x0:.0f},{cy-r*0.5:.0f} L{x1:.0f},{y0:.0f} '
                f'L{x1:.0f},{y1:.0f} L{x0:.0f},{cy+r*0.5:.0f} Z"/>')
    if b in ("tailing", "tailings"):
        return (f'<path d="M{x0+r*0.5:.0f},{y0:.0f} L{x1-r*0.5:.0f},{y0:.0f} '
                f'L{x1:.0f},{y1:.0f} L{x0:.0f},{y1:.0f} Z"/>'
                f'<path d="M{x0+2:.0f},{cy:.0f} q{r*0.5:.0f},-3 {r:.0f},0 '
                f't{r:.0f},0" fill="none"/>')
    if b in ("OUTPUT", "output", "product_recovery", "concentrate", "product"):
        return (f'<path d="M{x0:.0f},{y0:.0f} L{cx+r*0.4:.0f},{y0:.0f} '
                f'L{x1:.0f},{cy:.0f} L{cx+r*0.4:.0f},{y1:.0f} '
                f'L{x0:.0f},{y1:.0f} Z"/>')
    # generic block
    return (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{2*r:.0f}" '
            f'height="{2*r:.0f}" rx="3"/>')


# --- layout + render ----------------------------------------------------------

NODE_W, NODE_H = 108, 84
COL_GAP, ROW_GAP = 158, 116
PAD = 26


def _layout(stage_counts: dict, edge_weights: dict):
    """Graph plus the column assignment every drawing decision keys off.

    Shared by render_svg and drawn_edges so the accounting table and the
    picture cannot disagree about which arrows exist or which way they run.
    """
    G = build_knn_graph(stage_counts, edge_weights)
    xpos = _x_positions(G)  # base -> float x (spacing 4.1 units)
    by_col: dict = {}
    for n in G.nodes:
        by_col.setdefault(xpos[_stage_base(n)], []).append(n)
    cols = sorted(by_col)
    return G, xpos, by_col, cols, {x: i for i, x in enumerate(cols)}


def drawn_edges(stage_counts: dict, edge_weights: dict) -> list:
    """Every arrow render_svg will actually draw, in draw order.

    The picture is not the caller's edge list. build_knn_graph drops self-loops
    and illegal terminations, and then reconnects any block the surviving edges
    left unreachable from INPUT by walking KNN_ORDER for the nearest upstream
    node. Those reconnection arrows are the renderer's topology rule, not the
    model's output, and on a sparse prediction they can outnumber the ones the
    model supplied — so each row carries `derived` and callers must say so
    rather than presenting the whole flowsheet as predicted.

    Returns dicts of src, dst (stage bases; INPUT/OUTPUT as "input"/"output"),
    kind ("feed-forward" or "recycle" — matching the drawn legend), weight (the
    caller's score for that pair, or None when derived) and derived.
    """
    G, xpos, _by_col, _cols, col_idx = _layout(stage_counts, edge_weights)
    out = []
    for i, (u, v) in enumerate(G.edges()):
        su, sv = _stage_base(u), _stage_base(v)
        cu, cv = col_idx[xpos[su]], col_idx[xpos[sv]]
        w = edge_weights.get((su, sv))
        attrs = G.edges[u, v]
        derived = bool(attrs.get("derived"))
        # The graph tag and the missing score must agree. They are derived
        # independently — build_knn_graph tags at insertion, the score lookup
        # asks the caller's edge dict — so a mismatch means one of the two
        # paths changed without the other, which is exactly the drift the
        # accounting exists to prevent.
        assert derived == (w is None), (
            f"provenance disagrees for {su}->{sv}: tagged derived={derived} "
            f"but weight={w!r}")
        out.append({
            "idx": i + 1,
            "src": su, "dst": sv,
            "kind": "feed-forward" if cu < cv else "recycle",
            "weight": w,
            "derived": derived,
            # The block that would have been unreachable from the feed had the
            # renderer not inserted this arrow. None for a real predicted edge.
            "derived_for": _stage_base(attrs["derived_for"]) if derived else None,
        })
    return out


def render_svg(stage_counts: dict, edge_weights: dict) -> str:
    G, xpos, by_col, cols, col_idx = _layout(stage_counts, edge_weights)
    max_rows = max((len(v) for v in by_col.values()), default=1)
    plot_h = max_rows * ROW_GAP

    coord: dict = {}
    for x, nodes in by_col.items():
        ci = col_idx[x]
        cx = PAD + NODE_W / 2 + ci * COL_GAP
        nodes = sorted(nodes, key=lambda n: (_stage_base(n) not in ("INPUT",), n))
        off = PAD + (plot_h - len(nodes) * ROW_GAP) / 2
        for j, n in enumerate(nodes):
            coord[n] = (cx, off + j * ROW_GAP + ROW_GAP / 2)

    scores = list(edge_weights.values()) or [1.0]
    smin, smax = min(scores), max(scores)

    # edges
    edge_svg, max_bottom, max_right = [], PAD + plot_h, PAD + NODE_W + (len(cols) - 1) * COL_GAP
    num_svg = []
    side_seen = 0
    for e_i, (u, v) in enumerate(G.edges()):
        (x1, y1), (x2, y2) = coord[u], coord[v]
        cu, cv = col_idx[xpos[_stage_base(u)]], col_idx[xpos[_stage_base(v)]]
        w = edge_weights.get((_stage_base(u), _stage_base(v)))
        t = (w - smin) / (smax - smin) if (w is not None and smax > smin) else 0.55
        sw = 1.5 + 1.6 * t
        if cu < cv:  # feed-forward
            sx, ex = x1 + NODE_W / 2, x2 - NODE_W / 2
            dx = (ex - sx) * 0.45
            d = f"M{sx:.0f},{y1:.0f} C{sx+dx:.0f},{y1:.0f} {ex-dx:.0f},{y2:.0f} {ex:.0f},{y2:.0f}"
            pts = ((sx, y1), (sx + dx, y1), (ex - dx, y2), (ex, y2))
            cls, mk = "fwd", "url(#ah)"
        elif cu == cv:  # same-column recycle -> side loop
            bulge = 34 + side_seen * 18
            side_seen += 1
            sx, ex = x1 + NODE_W / 2, x2 + NODE_W / 2
            d = f"M{sx:.0f},{y1:.0f} C{sx+bulge:.0f},{y1:.0f} {ex+bulge:.0f},{y2:.0f} {ex:.0f},{y2:.0f}"
            pts = ((sx, y1), (sx + bulge, y1), (ex + bulge, y2), (ex, y2))
            cls, mk = "rcy", "url(#ahr)"
            max_right = max(max_right, ex + bulge + 6)
        else:  # cross-column back-flow -> bottom bow
            bow = 46 + 14 * (cu - cv)
            sx, ex = x1, x2
            d = f"M{sx:.0f},{y1+NODE_H/2:.0f} C{sx:.0f},{y1+NODE_H/2+bow:.0f} {ex:.0f},{y2+NODE_H/2+bow:.0f} {ex:.0f},{y2+NODE_H/2:.0f}"
            pts = ((sx, y1 + NODE_H / 2), (sx, y1 + NODE_H / 2 + bow),
                   (ex, y2 + NODE_H / 2 + bow), (ex, y2 + NODE_H / 2))
            cls, mk = "rcy", "url(#ahr)"
            max_bottom = max(max_bottom, max(y1, y2) + NODE_H / 2 + bow + 6)
        edge_svg.append(f'<path class="e {cls}" d="{d}" stroke-width="{sw:.1f}" '
                        f'marker-end="{mk}" fill="none"/>')
        # Number every arrow so the connection table can point at it. The badge
        # sits at the curve's own midpoint (cubic Bezier at t=0.5), which keeps
        # it on its arrow even where several run between the same columns.
        mx = (pts[0][0] + 3 * pts[1][0] + 3 * pts[2][0] + pts[3][0]) / 8
        my = (pts[0][1] + 3 * pts[1][1] + 3 * pts[2][1] + pts[3][1]) / 8
        num_svg.append(
            f'<g class="enum"><circle cx="{mx:.0f}" cy="{my:.0f}" r="7.5"/>'
            f'<text x="{mx:.0f}" y="{my + 3.1:.0f}">{e_i + 1}</text></g>')

    # nodes
    node_svg = []
    for n, (cx, cy) in coord.items():
        ph = _phase(_stage_base(n))
        lbl = _label(G.nodes[n].get("label", n))
        gy = cy - 12
        # label wrap (<=2 lines)
        if len(lbl) > 11 and " " in lbl:
            parts = lbl.split(" ")
            l1 = parts[0]
            l2 = " ".join(parts[1:])
            text = (f'<text class="lbl" x="{cx:.0f}" y="{cy+18:.0f}">{html.escape(l1)}</text>'
                    f'<text class="lbl" x="{cx:.0f}" y="{cy+30:.0f}">{html.escape(l2)}</text>')
        else:
            text = f'<text class="lbl" x="{cx:.0f}" y="{cy+22:.0f}">{html.escape(lbl)}</text>'
        node_svg.append(
            f'<g class="n ph{ph}">'
            f'<rect class="card" x="{cx-NODE_W/2:.0f}" y="{cy-NODE_H/2:.0f}" '
            f'width="{NODE_W}" height="{NODE_H}" rx="14"/>'
            f'<rect class="accent" x="{cx-NODE_W/2:.0f}" y="{cy-NODE_H/2:.0f}" '
            f'width="{NODE_W}" height="4" rx="2"/>'
            f'<g class="gl">{_g(n, cx, gy)}</g>{text}</g>')

    width = max_right + PAD
    height = max_bottom + PAD
    defs = (
        '<defs>'
        '<marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto"><path class="ahp" d="M0,1 L9,5 L0,9"/></marker>'
        '<marker id="ahr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" '
        'markerHeight="6.5" orient="auto"><path class="ahrp" d="M0,1 L9,5 L0,9"/></marker>'
        '</defs>')
    return (f'<div class="fs-scroll"><svg class="flowsheet" '
            f'viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" '
            f'height="{height:.0f}" role="img" '
            f'aria-label="Designed process flowsheet">'
            f'{defs}{"".join(edge_svg)}{"".join(node_svg)}'
            f'{"".join(num_svg)}</svg></div>')
