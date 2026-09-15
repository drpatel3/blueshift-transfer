"""Visualize current graph pipeline results from S3 checkpoint."""

import json
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

import os
import subprocess

PIPELINE_BUCKET = os.getenv("PIPELINE_BUCKET", "mineral-pipeline-pipeline")

def fetch_s3_json(key):
    result = subprocess.run(
        ["aws", "s3", "cp", f"s3://{PIPELINE_BUCKET}/graphs/{key}", "-"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)

def main():
    # 1. Load summary stats
    summary = fetch_s3_json("graphs_summary.json")
    stage_vocab = fetch_s3_json("stage_vocab.json")
    checkpoint = fetch_s3_json("checkpoint.json")

    total_docs = summary["total_docs"]
    completed = len(checkpoint)
    remaining = total_docs - completed

    print(f"Progress: {completed}/{total_docs} docs ({completed/total_docs*100:.1f}%)")
    print(f"Docs with stages: {summary['docs_with_stages']}")
    print(f"Docs context-only: {summary['docs_context_only']}")
    print(f"Total nodes: {summary['total_nodes']} ({summary['total_stage_nodes']} stage, {summary['total_context_nodes']} context)")
    print(f"Total edges: {summary['total_edges']}")
    print(f"Stage vocabulary: {summary['stage_vocab_size']} unique stages")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Graph Pipeline Progress: {completed}/{total_docs} docs ({completed/total_docs*100:.1f}%)",
                 fontsize=14, fontweight="bold")

    # Panel 1: Section coverage bar chart
    ax = axes[0, 0]
    sections = summary["section_coverage"]
    names = list(sections.keys())
    counts = list(sections.values())
    # Sort by count
    sorted_pairs = sorted(zip(names, counts), key=lambda x: x[1], reverse=True)
    names, counts = zip(*sorted_pairs) if sorted_pairs else ([], [])
    ax.barh(range(len(names)), counts, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title("Section Coverage")
    ax.invert_yaxis()

    # Panel 2: Edge type distribution
    ax = axes[0, 1]
    edge_types = summary["edge_type_counts"]
    labels = [k.replace("_", " ").title() for k in edge_types.keys()]
    values = list(edge_types.values())
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Edge Type Distribution")

    # Panel 3: Stage vocabulary (top 20)
    ax = axes[1, 0]
    sorted_stages = sorted(stage_vocab.items(), key=lambda x: x[1], reverse=False)
    top_stages = sorted_stages[-20:] if len(sorted_stages) > 20 else sorted_stages
    stage_names = [s[0].replace("_", " ").title() for s in top_stages]
    stage_ids = [s[1] for s in top_stages]
    ax.barh(range(len(stage_names)), stage_ids, color="#59a14f")
    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names, fontsize=8)
    ax.set_xlabel("Vocab ID")
    ax.set_title(f"Stage Vocabulary ({len(stage_vocab)} unique)")
    ax.invert_yaxis()

    # Panel 4: Summary stats text
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = (
        f"Total documents: {total_docs}\n"
        f"Completed: {completed} ({completed/total_docs*100:.1f}%)\n"
        f"Remaining: {remaining}\n"
        f"\n"
        f"Documents with stages: {summary['docs_with_stages']}\n"
        f"Documents context-only: {summary['docs_context_only']}\n"
        f"Documents no graph data: {completed - summary['docs_with_stages'] - summary['docs_context_only']}\n"
        f"\n"
        f"Total nodes: {summary['total_nodes']}\n"
        f"  Stage nodes: {summary['total_stage_nodes']}\n"
        f"  Context nodes: {summary['total_context_nodes']}\n"
        f"Total edges: {summary['total_edges']}\n"
        f"\n"
        f"Avg nodes/doc (with data): {summary['total_nodes'] / max(1, summary['docs_with_stages'] + summary['docs_context_only']):.1f}\n"
        f"Avg edges/doc (with data): {summary['total_edges'] / max(1, summary['docs_with_stages'] + summary['docs_context_only']):.1f}\n"
        f"Stage vocab size: {summary['stage_vocab_size']}"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Summary Statistics")

    plt.tight_layout()
    out_path = Path(__file__).parent / "graph_output" / "pipeline_progress.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
