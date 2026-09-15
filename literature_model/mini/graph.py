"""Aggregate connection graph across all flowsheets in mini/results.json.

Normalizes parent_id/child_id to canonical stage types, counts how many
times each directed edge occurs across all flowsheets, and draws the
graph with arrow width proportional to occurrence count.

Usage:
    python mini/graph.py
"""

import json
import sys
from pathlib import Path
from collections import Counter, deque

import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mini.normalize import normalize_stage

MINI_DIR = Path(__file__).resolve().parent
RESULTS_PATH = MINI_DIR / "results.json"


def load_edge_counts():
    """Load all connections from results.json, normalize, and count edges.

    Returns Counter of (parent_canonical, child_canonical) tuples.
    """
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    edge_counts = Counter()

    for doc in data["documents"]:
        for fs in doc.get("flowsheets", []):
            for conn in fs.get("connections", []):
                parent = normalize_stage(conn["parent_id"])
                child = normalize_stage(conn["child_id"])

                # Skip noise stages and self-loops
                if parent is None or child is None or parent == child:
                    continue

                edge_counts[(parent, child)] += 1

    return edge_counts


def build_graph(edge_counts):
    """Build a weighted directed graph from edge counts."""
    G = nx.DiGraph()

    for (parent, child), count in edge_counts.items():
        G.add_edge(parent, child, weight=count)

    return G


def draw_graph(G, output_file=None):
    """Draw the aggregate connection graph with weighted arrows."""
    if output_file is None:
        output_file = MINI_DIR / "connection_graph.png"

    if not G.nodes:
        print("Empty graph — nothing to draw.")
        return

    # BFS layer assignment from root nodes (no incoming edges)
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        roots = [max(G.nodes, key=lambda n: G.out_degree(n))]

    layer = {}
    visited = set()
    queue = deque([(r, 0) for r in roots])
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            # Update to deepest occurrence so downstream nodes stay below
            layer[node] = max(layer.get(node, 0), depth)
            continue
        visited.add(node)
        layer[node] = depth
        for child in G.successors(node):
            queue.append((child, depth + 1))

    # Catch any disconnected nodes
    for n in G.nodes:
        if n not in layer:
            layer[n] = 0

    # Group nodes by layer
    layers = {}
    for node, lyr in layer.items():
        layers.setdefault(lyr, []).append(node)

    # Position: y = layer (inverted so input is at top), x = spread within layer
    pos = {}
    max_layer = max(layers.keys()) if layers else 0
    for layer_idx, layer_nodes in layers.items():
        y = (max_layer - layer_idx) * 2.0
        for i, node in enumerate(sorted(layer_nodes)):
            x = (i - (len(layer_nodes) - 1) / 2) * 3.0
            pos[node] = (x, y)

    # Edge weights for line widths — log-scaled so dominant connections
    # don't overshadow rare but meaningful ones
    import numpy as np
    raw_weights = [G[u][v]["weight"] for u, v in G.edges()]
    log_weights = [np.log1p(w) for w in raw_weights]  # ln(1 + count)
    max_log = max(log_weights) if log_weights else 1
    min_width, max_width = 0.5, 6.0
    widths = [min_width + (lw / max_log) * (max_width - min_width)
              for lw in log_weights]

    # Edge colors: darker for higher weight (also log-scaled)
    alphas = [0.3 + 0.7 * (lw / max_log) for lw in log_weights]
    edge_colors = [(0.2, 0.2, 0.2, a) for a in alphas]

    _, ax = plt.subplots(figsize=(14, max(10, len(G.nodes) * 0.8)))

    nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_colors,
                           arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=1400, node_color="lightblue",
                           edgecolors="steelblue", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    # Edge labels: show count
    edge_labels = {(u, v): str(G[u][v]["weight"]) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=6, font_color="darkred", ax=ax)

    ax.set_title("Aggregate Process Flow — All Flowsheets\n"
                 f"({len(G.nodes)} stages, {len(G.edges())} unique connections)",
                 fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {output_file}")


def run():
    """Load, build, draw."""
    edge_counts = load_edge_counts()

    print(f"Total unique edges: {len(edge_counts)}")
    print(f"\nTop 20 most common connections:")
    for (parent, child), count in edge_counts.most_common(20):
        print(f"  {parent} -> {child}: {count}")

    G = build_graph(edge_counts)
    draw_graph(G)

    return edge_counts


if __name__ == "__main__":
    run()
