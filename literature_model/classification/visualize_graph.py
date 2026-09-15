"""Visualize document graphs from the graph-build pipeline output."""
import json
import sys
import random
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


GRAPHS_DIR = Path(__file__).parent / "graph_output" / "graphs"


def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load a single document graph JSON into a NetworkX DiGraph."""
    with open(graph_path, encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"], type=node["type"], group=node.get("group", ""))
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"],
                    type=edge["type"], weight=edge.get("weight", 1.0))
    return G


def visualize_single(graph_path: Path, save_path: Path = None):
    """Visualize a single document graph."""
    G = load_graph(graph_path)

    # Color by node type
    color_map = {"context": "#4A90D9", "stage": "#E74C3C"}
    node_colors = [color_map.get(G.nodes[n].get("type", ""), "#999") for n in G.nodes]
    node_sizes = [800 if G.nodes[n].get("type") == "stage" else 400 for n in G.nodes]

    # Edge style by type
    stage_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "stage_transition"]
    cooc_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "section_cooccurrence"]

    # Labels: strip ctx_ / stage_ prefix for readability
    labels = {}
    for n in G.nodes:
        label = n.replace("ctx_", "").replace("stage_", "")
        labels[n] = label

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    # Draw co-occurrence edges (light, thin)
    nx.draw_networkx_edges(G, pos, edgelist=cooc_edges, ax=ax,
                           edge_color="#CCCCCC", alpha=0.3, width=0.5,
                           connectionstyle="arc3,rad=0.05")
    # Draw stage transition edges (bold, red)
    nx.draw_networkx_edges(G, pos, edgelist=stage_edges, ax=ax,
                           edge_color="#E74C3C", alpha=0.8, width=2.0,
                           arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1")
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#4A90D9", label=f"Context section ({sum(1 for n in G.nodes if G.nodes[n].get('type')=='context')})"),
        mpatches.Patch(color="#E74C3C", label=f"Process stage ({sum(1 for n in G.nodes if G.nodes[n].get('type')=='stage')})"),
        plt.Line2D([0], [0], color="#CCCCCC", linewidth=1, label=f"Co-occurrence ({len(cooc_edges)})"),
        plt.Line2D([0], [0], color="#E74C3C", linewidth=2, label=f"Stage transition ({len(stage_edges)})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    doc_name = graph_path.stem[:16]
    ax.set_title(f"Document Graph: {doc_name}…\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges", fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def visualize_summary(graphs_dir: Path, save_path: Path = None):
    """Aggregate stats across all graphs and plot distributions."""
    with open(graphs_dir / "graphs_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Section coverage
    ax = axes[0, 0]
    sections = summary["section_coverage"]
    names = list(sections.keys())
    counts = list(sections.values())
    ax.barh(names, counts, color="#4A90D9")
    ax.set_xlabel("Documents with section")
    ax.set_title("Section Coverage Across 963 Documents")
    ax.invert_yaxis()

    # 2. Edge type breakdown
    ax = axes[0, 1]
    edge_types = summary["edge_type_counts"]
    ax.bar(edge_types.keys(), edge_types.values(), color=["#CCCCCC", "#E74C3C"])
    ax.set_ylabel("Count")
    ax.set_title(f"Edge Types (total: {summary['total_edges']:,})")
    for i, (k, v) in enumerate(edge_types.items()):
        ax.text(i, v + 500, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    # 3. Node type breakdown
    ax = axes[1, 0]
    node_data = {"Context": summary["total_context_nodes"], "Stage": summary["total_stage_nodes"]}
    ax.bar(node_data.keys(), node_data.values(), color=["#4A90D9", "#E74C3C"])
    ax.set_ylabel("Count")
    ax.set_title(f"Node Types (total: {summary['total_nodes']:,})")
    for i, (k, v) in enumerate(node_data.items()):
        ax.text(i, v + 100, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    # 4. Per-doc node/edge distribution (sample)
    ax = axes[1, 1]
    json_files = sorted(graphs_dir.glob("*.json"))
    json_files = [f for f in json_files if f.stem not in ("graphs_summary", "stage_vocab", "checkpoint")]
    sample = random.sample(json_files, min(200, len(json_files)))
    node_counts, edge_counts = [], []
    for fp in sample:
        with open(fp) as f:
            d = json.load(f)
        node_counts.append(len(d["nodes"]))
        edge_counts.append(len(d["edges"]))
    ax.scatter(node_counts, edge_counts, alpha=0.5, s=20, color="#4A90D9")
    ax.set_xlabel("Nodes per doc")
    ax.set_ylabel("Edges per doc")
    ax.set_title(f"Node vs Edge Count (n={len(sample)} sample)")

    # Top stats text
    fig.suptitle(
        f"Graph Pipeline Summary — {summary['total_docs']} docs, "
        f"{summary['docs_with_stages']} with stages, "
        f"{summary['stage_vocab_size']} unique stage types",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "summary"
    out_dir = Path(__file__).parent / "graph_output"

    if mode == "summary":
        visualize_summary(GRAPHS_DIR, out_dir / "summary.png")
    elif mode == "single":
        # Pick a graph with stages
        for fp in sorted(GRAPHS_DIR.glob("*.json")):
            if fp.stem in ("graphs_summary", "stage_vocab", "checkpoint"):
                continue
            with open(fp) as f:
                d = json.load(f)
            stages = [n for n in d["nodes"] if n["type"] == "stage"]
            if len(stages) >= 4:
                visualize_single(fp, out_dir / "single_graph.png")
                break
    elif mode == "both":
        visualize_summary(GRAPHS_DIR, out_dir / "summary.png")
        for fp in sorted(GRAPHS_DIR.glob("*.json")):
            if fp.stem in ("graphs_summary", "stage_vocab", "checkpoint"):
                continue
            with open(fp) as f:
                d = json.load(f)
            stages = [n for n in d["nodes"] if n["type"] == "stage"]
            if len(stages) >= 4:
                visualize_single(fp, out_dir / "single_graph.png")
                break
    else:
        # Treat as a specific file path
        p = Path(mode)
        if not p.exists():
            p = GRAPHS_DIR / mode
        visualize_single(p, out_dir / "single_graph.png")
