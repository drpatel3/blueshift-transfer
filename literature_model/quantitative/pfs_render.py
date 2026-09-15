"""Render a KNN-predicted flowsheet using the existing PFS equipment-symbol
display (mini/pfs_visual), returned as a base64 PNG for embedding in the demo.

This reuses mini/pfs_visual's equipment symbols, row layout, and edge/node
drawing verbatim; it only supplies (a) a stage order wide enough for the KNN
model's vocabulary (mini's CANONICAL_ORDER predates the hydromet stages) and
(b) symbol aliases for the few tokens mini never drew. mini/ is not modified.

The graph builder mirrors mini.digital_twin.build_predicted_graph (INPUT/OUTPUT
framing, invalid-edge filter, orphan reconnection) so the flowsheet is drawn the
same way the existing display would draw it.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import networkx as nx
import matplotlib.pyplot as plt
from mini import pfs_visual as pv

# Process sequence covering the full KNN stage vocabulary, following mini's
# own CANONICAL_ORDER ordering and slotting the newer stages beside their kin.
KNN_ORDER = [
    "input", "feeder", "crusher", "ore_sorting", "stockpile", "screen",
    "mill", "cyclone", "gravity", "flotation", "regrind", "agglomeration",
    "leach", "adsorption", "elution", "solution_recovery", "electrowinning",
    "kiln",
    "thickener", "filter", "precipitation", "tank", "bin", "product_recovery",
    "tailing", "treatment", "output",
]

# Equipment symbols for tokens mini never drew (unknowns fall back to a plain
# tank in pv._draw_nodes; these pick a closer match where one exists).
pv.SYMBOL_FUNCS.setdefault("solution_recovery", pv._draw_solvent_extraction)
pv.SYMBOL_FUNCS.setdefault("feeder", pv._draw_bin)
pv.SYMBOL_FUNCS.setdefault("product_recovery", pv._draw_output)
pv.SYMBOL_FUNCS.setdefault("agglomeration", pv._draw_mill)

_NO_DIRECT_OUTPUT = {"crusher", "mill", "stockpile", "conveyor", "screen",
                     "cyclone", "bin", "feeder", "tailing", "INPUT"}


def build_knn_graph(stage_counts: dict, edge_weights: dict) -> "nx.DiGraph":
    """DiGraph from predicted stages/edges, mirroring build_predicted_graph but
    over the wider KNN_ORDER vocabulary.

    Note the vocabulary is a whitelist: a stage absent from KNN_ORDER is skipped
    entirely, so a caller drawing a different model's stage set must extend
    KNN_ORDER first or its blocks will be missing from the flowsheet while still
    appearing in the caller's own stage table."""
    G = nx.DiGraph()
    for stage in KNN_ORDER:
        if stage in ("input", "output") or stage not in stage_counts:
            continue
        count = stage_counts[stage]
        # Stack multi-word labels onto two lines so long names (e.g. "solution
        # recovery") don't collide with their neighbours' labels.
        lbl = stage.replace("_", "\n") if "_" in stage else stage
        if count <= 1:
            G.add_node(stage, label=lbl)
        else:
            for i in range(1, count + 1):
                G.add_node(f"{stage}_{i}", label=f"{lbl}\n({i})")
    G.add_node("INPUT", label="INPUT")
    G.add_node("OUTPUT", label="OUTPUT")

    for (parent, child), _w in sorted(edge_weights.items(), key=lambda x: -x[1]):
        if parent == child:
            continue
        p = "INPUT" if parent == "input" else parent
        c = "OUTPUT" if child == "output" else child
        p = p if p in G else f"{p}_1"
        c = c if c in G else f"{c}_1"
        if c == "OUTPUT" and p.rsplit("_", 1)[0] in _NO_DIRECT_OUTPUT:
            continue
        if c == "INPUT":
            continue
        if p in G and c in G:
            G.add_edge(p, c)

    # Reconnect anything unreachable from INPUT to its nearest upstream present
    # node, walking KNN_ORDER (same policy as the existing display).
    ordered = (["INPUT"]
               + [s for s in KNN_ORDER if s not in ("input", "output") and s in G]
               + ["OUTPUT"])
    reachable = set(nx.descendants(G, "INPUT")) | {"INPUT"}
    for idx, node in enumerate(ordered):
        if node in reachable:
            continue
        for j in range(idx - 1, -1, -1):
            prev = ordered[j]
            if prev in reachable:
                if node == "OUTPUT" and prev.rsplit("_", 1)[0] in _NO_DIRECT_OUTPUT:
                    continue
                # Tag the insertion so a caller can say *why* this arrow is on
                # the page: it is not a prediction, it exists because `node`
                # was left unreachable from the feed. Without the tag a
                # consumer can only infer "derived" from a missing score,
                # which cannot distinguish this from a scored edge of zero.
                # The walk can land on an edge that already exists: `reachable`
                # is updated node-by-node and never re-propagates to a node's
                # descendants, so a real predicted edge whose source only
                # became reachable earlier in this same loop is revisited here.
                # Re-adding it is a no-op, but it must NOT be tagged derived —
                # that would relabel a prediction as a drawing-rule artefact.
                if not G.has_edge(prev, node):
                    G.add_edge(prev, node, derived=True, derived_for=node)
                reachable.add(node)
                break
    # Drop OUTPUT if nothing legitimately reached it (KNN has no output stage).
    if G.in_degree("OUTPUT") == 0:
        G.remove_node("OUTPUT")
    return G


def _x_positions(G) -> dict:
    bases = {pv._stage_base(n) for n in G.nodes}
    lookup = {s: i for i, s in enumerate(KNN_ORDER)}
    lookup["input"] = -1
    lookup["output"] = len(KNN_ORDER)
    ordered = sorted(bases, key=lambda s: lookup.get(s, 999))
    return {s: i * 4.1 for i, s in enumerate(ordered)}


def render_pfs_b64(stage_counts: dict, edge_weights: dict,
                   title: str | None = None) -> str:
    """Return a base64 PNG of the predicted flowsheet drawn in the PFS style."""
    G = build_knn_graph(stage_counts, edge_weights)
    xpos = _x_positions(G)
    pos = pv._row_layout(G, xpos, 0.0)

    ncols = max(len(xpos), 1)
    fig, ax = plt.subplots(figsize=(max(9.0, ncols * 1.9), 4.0))
    pv._draw_edges(ax, G, pos, {})
    pv._draw_nodes(ax, G, pos, {n: pv.C_DEFAULT for n in G.nodes},
                   label_side="below")

    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    ax.set_xlim(min(xs) - 2.2, max(xs) + 2.2)
    ax.set_ylim(min(ys) - 2.4, max(ys) + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", color="#333333",
                     pad=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":  # quick smoke test against the frozen traces
    import json
    traces = json.loads(Path("quantitative/demo/traces.json").read_text())
    for ex in traces["examples"]:
        sc = {sv["stage"]: 1 for sv in ex["stage_votes"] if sv["selected"]}
        ew = {(e["src"], e["dst"]): e["score"]
              for e in ex["edge_votes"] if e["selected"]}
        b64 = render_pfs_b64(sc, ew, ex["query_doc_id"])
        out = Path("quantitative/demo") / f"_pfs_{ex['query_doc_id'][:12]}.png"
        out.write_bytes(base64.b64decode(b64))
        print(f"wrote {out} ({len(b64)//1024} KB b64)")
