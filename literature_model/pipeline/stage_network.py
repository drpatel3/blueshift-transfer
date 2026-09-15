import json
import logging
import os
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from normalize_ids import normalize_id

logger = logging.getLogger(__name__)

def load_data(filepath):
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}


def load_data_from_db(conn=None):
    """Load flowsheet data from the database. Falls back to JSON if no connection."""
    if conn is None:
        return load_data("results.json")
    from database import get_all_flowsheets
    return get_all_flowsheets(conn)

def build_network(data):
    """
    Build a DIRECTED graph where edge weight = count of A->B occurrences.
    Uses connections data with parent_id/child_id (falls back to from_type/to_type).
    Normalizes stage IDs via process_output.normalize_id for cross-flowsheet matching.
    """
    G = nx.DiGraph()
    transition_counts = defaultdict(lambda: defaultdict(int))
    original_pairs = defaultdict(list)

    for flowsheet in data.values():
        for conn in flowsheet.get('connections', []):
            # Support new parent_id/child_id fields with fallback to old from_type/to_type
            parent_raw = conn.get('parent_id') or conn.get('from_type', '')
            child_raw = conn.get('child_id') or conn.get('to_type', '')

            if not parent_raw or not child_raw:
                continue

            parent = normalize_id(parent_raw)
            child = normalize_id(child_raw)

            if parent != child:
                transition_counts[parent][child] += 1
                original_pairs[(parent, child)].append((parent_raw, child_raw))

    for s1, targets in transition_counts.items():
        for s2, count in targets.items():
            G.add_edge(s1, s2, weight=count,
                       original_connections=original_pairs[(s1, s2)])

    return G

def visualize_network(G):
    plt.figure(figsize=(18, 14))

    # Use learned layout positions if checkpoint exists, else spring layout
    layout_checkpoint = os.path.join(os.path.dirname(__file__), "..", "classification", "layout_checkpoint.pt")
    if os.path.exists(layout_checkpoint):
        try:
            from classification.layout_gat import layout_flowsheet
            connections = [{"from": u, "to": v, "corpus_count": G[u][v].get("weight", 1)}
                           for u, v in G.edges()]
            pos = layout_flowsheet(layout_checkpoint, list(G.nodes()), connections)
            pos = {k: (v["x"], v["y"]) for k, v in pos.items() if k in G.nodes()}
            # Fill any missing nodes with spring layout
            if len(pos) < len(G.nodes()):
                missing = [n for n in G.nodes() if n not in pos]
                spring = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
                for n in missing:
                    pos[n] = spring[n]
        except Exception:
            logger.debug("Layout file parse failed, falling back to spring layout")
            pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    else:
        pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)

    # Edge width scales linearly with occurrence count
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    edge_widths = [0.5 + (w / max_weight) * 4 for w in weights]  # Scale 0.5 to 4.5

    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightgray", alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color="steelblue",
                           arrows=True, arrowsize=15, arrowstyle='->', connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold")

    plt.title("Process Stage Transitions\n(Line thickness = occurrence count)")
    plt.axis("off")
    plt.tight_layout()
    import config
    output_path = str(config.STAGE_NETWORK_OUTPUT)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved to {output_path}")

def print_stats(G):
    logger.info(f"\nNetwork: {G.number_of_nodes()} stages, {G.number_of_edges()} transitions")

    logger.info("\n=== Transitions by occurrence count ===")
    for u, v in sorted(G.edges(), key=lambda e: G[e[0]][e[1]]['weight'], reverse=True)[:20]:
        count = G[u][v]['weight']
        logger.info(f"  {count}x: '{u}' -> '{v}'")
        for orig_parent, orig_child in G[u][v].get('original_connections', []):
            logger.info(f"       {orig_parent} -> {orig_child}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import config
    data = load_data(str(config.RESULTS_PATH))
    G = build_network(data)
    print_stats(G)
    visualize_network(G)
