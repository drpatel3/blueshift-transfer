"""Retroactively recompute stage_transition weights using log-1.8 mention counts.

Reads keyword counts (kw_*) from context node features, maps them to stage IDs
via the same SYNONYMS/keyword logic as graph_pipeline.py, then applies:
    weight = log(1 + parent_mentions, 1.8) + log(1 + child_mentions, 1.8)

Usage:
    python classification/reweight_transitions.py          # auto-pick richest graph
    python classification/reweight_transitions.py <hash>   # specific graph file
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from graph_pipeline import SYNONYMS, CORE_WORDS, EQUIPMENT_KEYWORDS

GRAPHS_DIR = Path(__file__).parent / "graph_output" / "graphs"
OUT_DIR = Path(__file__).parent / "graph_output"


def reweight_graph(graph: dict) -> dict:
    """Recompute stage_transition edge weights from keyword features."""
    stage_nodes = {n["stage_id"]: n for n in graph["nodes"] if n["type"] == "stage"}
    context_nodes = [n for n in graph["nodes"] if n["type"] == "context"]

    if not stage_nodes:
        return graph

    # Build keyword -> stage_id reverse lookup (same logic as graph_pipeline)
    keyword_to_stages = defaultdict(set)
    for norm_id in stage_nodes:
        id_tokens = set(norm_id.split("_"))
        for token in id_tokens:
            keyword_to_stages[token].add(norm_id)
        for syn, canonical in SYNONYMS.items():
            if canonical in id_tokens:
                keyword_to_stages[syn].add(norm_id)

    # Sum keyword mentions per stage from context node features
    stage_mentions = defaultdict(int)
    for ctx in context_nodes:
        feats = ctx.get("features", {})
        for feat_key, count in feats.items():
            if not feat_key.startswith("kw_") or count <= 0:
                continue
            keyword = feat_key[3:]  # strip "kw_"
            # Check direct keyword match to stages
            if keyword in keyword_to_stages:
                for sid in keyword_to_stages[keyword]:
                    stage_mentions[sid] += count
            # Also check individual tokens for multi-word keywords (e.g. "open_pit")
            for token in keyword.split("_"):
                if token in keyword_to_stages:
                    for sid in keyword_to_stages[token]:
                        stage_mentions[sid] += count

    # Reweight transition edges
    for edge in graph["edges"]:
        if edge["type"] != "stage_transition":
            continue
        parent = edge["source"].removeprefix("stg_")
        child = edge["target"].removeprefix("stg_")
        weight = (math.log(1 + stage_mentions.get(parent, 0), 1.8)
                  + math.log(1 + stage_mentions.get(child, 0), 1.8))
        edge["weight"] = round(weight, 4)

    return graph


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if target == "auto":
        best, best_count = None, 0
        for fp in sorted(GRAPHS_DIR.glob("*.json")):
            if fp.stem in ("graphs_summary", "stage_vocab", "checkpoint"):
                continue
            with open(fp) as f:
                d = json.load(f)
            if "edges" not in d:
                continue
            stages = sum(1 for n in d["nodes"] if n["type"] == "stage")
            if stages > best_count:
                best, best_count = fp, stages
        if not best:
            print("No graphs with stages found.")
            return
        graph_path = best
        print(f"Selected graph with {best_count} stages: {best.stem[:20]}...")
    else:
        graph_path = Path(target)
        if not graph_path.exists():
            graph_path = GRAPHS_DIR / target

    with open(graph_path) as f:
        graph = json.load(f)

    graph = reweight_graph(graph)

    # Print weight distribution
    transitions = [e for e in graph["edges"] if e["type"] == "stage_transition"]
    weights = [e["weight"] for e in transitions]
    if weights:
        print(f"\nStage transition weights ({len(transitions)} edges):")
        print(f"  min={min(weights):.2f}  max={max(weights):.2f}  "
              f"mean={sum(weights)/len(weights):.2f}  median={sorted(weights)[len(weights)//2]:.2f}")
        # Show per-stage mention counts
        stage_nodes = {n["stage_id"]: n for n in graph["nodes"] if n["type"] == "stage"}
        keyword_to_stages = defaultdict(set)
        for norm_id in stage_nodes:
            id_tokens = set(norm_id.split("_"))
            for token in id_tokens:
                keyword_to_stages[token].add(norm_id)
            for syn, canonical in SYNONYMS.items():
                if canonical in id_tokens:
                    keyword_to_stages[syn].add(norm_id)
        stage_mentions = defaultdict(int)
        for ctx in [n for n in graph["nodes"] if n["type"] == "context"]:
            feats = ctx.get("features", {})
            for feat_key, count in feats.items():
                if not feat_key.startswith("kw_") or count <= 0:
                    continue
                keyword = feat_key[3:]
                if keyword in keyword_to_stages:
                    for sid in keyword_to_stages[keyword]:
                        stage_mentions[sid] += count
                for token in keyword.split("_"):
                    if token in keyword_to_stages:
                        for sid in keyword_to_stages[token]:
                            stage_mentions[sid] += count
        print(f"\nStage mention counts:")
        for sid, count in sorted(stage_mentions.items(), key=lambda x: -x[1])[:15]:
            log_val = math.log(1 + count, 1.8)
            print(f"  {sid:30s}  mentions={count:4d}  log1.8={log_val:.2f}")

        # Bucketed distribution
        buckets = defaultdict(int)
        for w in weights:
            bucket = int(w)
            buckets[bucket] += 1
        print(f"\nWeight distribution (floor buckets):")
        for b in sorted(buckets):
            bar = "#" * buckets[b]
            print(f"  [{b:2d}-{b+1:2d})  {buckets[b]:3d}  {bar}")
    else:
        print("No stage transitions found.")

    # Save reweighted graph for visualization
    reweighted_path = OUT_DIR / "reweighted_graph.json"
    with open(reweighted_path, "w") as f:
        json.dump(graph, f, ensure_ascii=False)
    print(f"\nSaved reweighted graph: {reweighted_path}")

    # Generate D3 visualization
    from visualize_graph_interactive import build_interactive
    html_path = build_interactive(reweighted_path, OUT_DIR / "interactive_graph.html")
    print(f"Open in browser: {html_path}")


if __name__ == "__main__":
    main()
