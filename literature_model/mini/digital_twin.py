"""Digital twin prototype: predict process flowsheet from feed grade.

Given Cu% ore grade as input, predicts stage-type counts using degree-2
polynomial regression trained on mini/results.json, then renders the
predicted flowsheet as a directed graph.

Usage:
    python mini/digital_twin.py 0.8
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# Parent path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mini.normalize import normalize_stage

MINI_DIR = Path(__file__).resolve().parent
RESULTS_PATH = MINI_DIR / "results.json"

GRADE_MIN = 0.1
GRADE_MAX = 2.0

# Canonical mineral processing order — determines graph layout
CANONICAL_ORDER = [
    "input", "crusher", "ore_sorting", "stockpile", "mill", "cyclone",
    "screen", "flotation", "regrind", "leach", "adsorption",
    "thickener", "filter", "precipitation", "tank", "bin",
    "tailing", "treatment", "output",
]

# Every real copper plant has at least these
MANDATORY_STAGES = {"crusher", "mill", "flotation"}


# ---------------------------------------------------------------------------
# Data loading & feature engineering
# ---------------------------------------------------------------------------

def select_primary_flowsheet(flowsheets):
    """Pick the primary copper-processing flowsheet from a document's list.

    Prefers the longest stage_sequence that contains 'flotation'.
    """
    with_flotation = [
        fs for fs in flowsheets
        if "flotation" in fs.get("stage_sequence", [])
    ]
    candidates = with_flotation if with_flotation else flowsheets
    return max(candidates, key=lambda fs: len(fs.get("stage_sequence", [])))


def load_training_data():
    """Load mini/results.json -> list of (grade, stage_counts, edge_counts).

    Connections are normalized via normalize_stage() and filtered to only
    keep edges between known canonical stage types.
    """
    # Canonical types that normalize_stage can return (excluding None)
    from mini.normalize import CANONICAL_STAGES
    canonical_set = set(CANONICAL_STAGES)

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    training = []
    for doc in data["documents"]:
        grade = doc.get("ore_grade_cu_pct")
        if grade is None or not (GRADE_MIN <= grade <= GRADE_MAX):
            continue

        flowsheets = [
            fs for fs in doc.get("flowsheets", [])
            if fs.get("stage_sequence")
        ]
        if not flowsheets:
            continue

        primary = select_primary_flowsheet(flowsheets)
        normalized = [normalize_stage(s) for s in primary["stage_sequence"]]
        normalized = [s for s in normalized if s is not None]
        stage_counts = Counter(normalized)

        # Normalize connections — only keep edges between canonical types
        edge_counts = Counter()
        for conn in primary.get("connections", []):
            parent = normalize_stage(conn["parent_id"])
            child = normalize_stage(conn["child_id"])
            if (parent in canonical_set and child in canonical_set
                    and parent != child):
                edge_counts[(parent, child)] += 1

        training.append((grade, stage_counts, edge_counts))

    return training


# Minimum documents an edge must appear in to be kept as a feature
MIN_EDGE_SUPPORT = 3


def build_feature_matrix(training_data):
    """Convert training data into X and Y numpy arrays.

    Y columns = stage counts + connection counts (only edges appearing
    in >= MIN_EDGE_SUPPORT documents).
    Returns (X, Y, stage_types, edge_keys).
    """
    all_stages = set()
    edge_doc_freq = Counter()
    for _, stage_counts, edge_counts in training_data:
        all_stages.update(stage_counts.keys())
        for edge in edge_counts:
            edge_doc_freq[edge] += 1

    stage_types = sorted(all_stages)
    edge_keys = sorted(e for e, freq in edge_doc_freq.items()
                       if freq >= MIN_EDGE_SUPPORT)

    X = np.array([[grade] for grade, _, _ in training_data])
    Y = np.array([
        [stage_counts.get(st, 0) for st in stage_types]
        + [edge_counts.get(ek, 0) for ek in edge_keys]
        for _, stage_counts, edge_counts in training_data
    ])
    return X, Y, stage_types, edge_keys


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def fit_model(X, Y):
    """Fit degree-2 polynomial multi-output regression."""
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    )
    model.fit(X, Y)
    return model


def postprocess(raw_predictions, stage_types, edge_keys):
    """Split raw predictions into stage counts and edge weights.

    Stage counts: round, clip >= 0, enforce mandatory minimums.
    Edge weights: keep connections with predicted count >= 0.5
    (model predicts at least ~1 occurrence of this connection).
    """
    n_stages = len(stage_types)
    raw_stages = raw_predictions[:n_stages]
    raw_edges = raw_predictions[n_stages:]

    # Stage counts
    counts = np.round(raw_stages).astype(int)
    counts = np.clip(counts, 0, None)
    stage_result = dict(zip(stage_types, counts))
    for stage in MANDATORY_STAGES:
        if stage in stage_result:
            stage_result[stage] = max(stage_result[stage], 1)
    stage_result = {k: int(v) for k, v in stage_result.items() if v > 0}

    # Edge weights — threshold at 0.5 raw count
    edge_result = {}
    for ek, w in zip(edge_keys, raw_edges):
        if w >= 0.5:
            edge_result[ek] = float(w)

    return stage_result, edge_result


# ---------------------------------------------------------------------------
# Graph reconstruction & drawing
# ---------------------------------------------------------------------------

def build_predicted_graph(stage_counts, edge_weights):
    """Build a directed graph from predicted stages and connections.

    Uses predicted edge weights to determine which connections exist.
    Orphaned nodes fall back to canonical-order connectivity.
    """
    G = nx.DiGraph()

    # Add stage nodes
    present = set()
    for stage in CANONICAL_ORDER:
        if stage in ("input", "output"):
            continue
        if stage not in stage_counts:
            continue
        count = stage_counts[stage]
        if count == 1:
            G.add_node(stage, label=stage)
            present.add(stage)
        else:
            for i in range(1, count + 1):
                nid = f"{stage}_{i}"
                G.add_node(nid, label=f"{stage} ({i})")
                present.add(stage)

    # Always have INPUT and OUTPUT
    G.add_node("INPUT", label="INPUT")
    G.add_node("OUTPUT", label="OUTPUT")

    # Stages that should never connect directly to OUTPUT
    _no_direct_output = {"crusher", "mill", "stockpile", "conveyor", "screen",
                         "cyclone", "bin", "tailing", "INPUT"}

    # Add predicted connections (only between nodes that exist)
    for (parent, child), weight in sorted(edge_weights.items(),
                                          key=lambda x: -x[1]):
        p_label = "INPUT" if parent == "input" else parent
        c_label = "OUTPUT" if child == "output" else child

        # For multi-count stages, connect to the first instance
        p_node = p_label if p_label in G else f"{p_label}_1"
        c_node = c_label if c_label in G else f"{c_label}_1"

        # Filter invalid connections
        if c_node == "OUTPUT" and p_node.rsplit("_", 1)[0] in _no_direct_output:
            continue
        if c_node == "INPUT":
            continue

        if p_node in G and c_node in G:
            G.add_edge(p_node, c_node, weight=weight)

    # For multi-count stages, chain instances sequentially
    # (e.g. flotation_1 -> flotation_2 -> flotation_3)
    for stage in CANONICAL_ORDER:
        if stage not in stage_counts or stage_counts[stage] <= 1:
            continue
        for i in range(1, stage_counts[stage]):
            G.add_edge(f"{stage}_{i}", f"{stage}_{i+1}")

    # Ensure every node is reachable from INPUT.
    # Walk canonical order; any node not yet reachable gets connected
    # to its nearest upstream neighbor that IS reachable.
    ordered_nodes = ["INPUT"]
    for stage in CANONICAL_ORDER:
        if stage in ("input", "output") or stage not in stage_counts:
            continue
        count = stage_counts[stage]
        if count == 1:
            ordered_nodes.append(stage)
        else:
            ordered_nodes.extend(f"{stage}_{i}"
                                 for i in range(1, count + 1))
    ordered_nodes.append("OUTPUT")

    reachable = set(nx.descendants(G, "INPUT")) | {"INPUT"}
    for idx, node in enumerate(ordered_nodes):
        if node in reachable:
            continue
        # Find nearest upstream node that IS reachable from INPUT
        for prev_idx in range(idx - 1, -1, -1):
            prev = ordered_nodes[prev_idx]
            if prev in reachable:
                # Don't connect blocked stages to OUTPUT
                if node == "OUTPUT" and prev.rsplit("_", 1)[0] in _no_direct_output:
                    continue
                G.add_edge(prev, node)
                # Update reachable set to include this node and its descendants
                reachable.add(node)
                reachable.update(nx.descendants(G, node))
                break

    # OUTPUT must be terminal — re-parent any children as siblings
    output_children = list(G.successors("OUTPUT"))
    if output_children:
        output_parents = list(G.predecessors("OUTPUT"))
        for child in output_children:
            G.remove_edge("OUTPUT", child)
            for parent in output_parents:
                if parent != child:
                    G.add_edge(parent, child)

    # Ensure non-terminal nodes are not leaves (have at least one outgoing edge).
    # Only output, tailing, and bin are allowed to be leaf nodes.
    terminal_types = {"output", "tailing", "bin"}
    for idx, node in enumerate(ordered_nodes):
        base = node.rsplit("_", 1)[0].lower() if node not in ("INPUT", "OUTPUT") else node.lower()
        if base in terminal_types:
            continue
        if G.out_degree(node) == 0:
            # Connect to nearest downstream node in canonical order
            for next_idx in range(idx + 1, len(ordered_nodes)):
                nxt = ordered_nodes[next_idx]
                if nxt in G:
                    G.add_edge(node, nxt)
                    break

    return G


def draw_predicted_graph(G, grade, output_file=None):
    """Draw the predicted flowsheet with a top-down layered layout."""
    if output_file is None:
        output_file = MINI_DIR / f"predicted_{grade:.2f}.png"

    if not G.nodes:
        print("Empty graph — nothing to draw.")
        return

    # BFS layer assignment (adapted from test_refinement.draw_graph)
    from collections import deque

    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        roots = [list(G.nodes)[0]]

    layer = {}
    visited = set()
    queue = deque([(r, 0) for r in roots])
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            layer[node] = max(layer.get(node, 0), depth)
            continue
        visited.add(node)
        layer[node] = depth
        for child in G.successors(node):
            queue.append((child, depth + 1))
    for n in G.nodes:
        if n not in layer:
            layer[n] = 0

    # Push all leaf nodes to the same bottom layer
    max_layer = max(layer.values()) if layer else 0
    for n in G.nodes:
        if G.out_degree(n) == 0:
            layer[n] = max_layer

    # Group by layer
    layers = {}
    for node, lyr in layer.items():
        layers.setdefault(lyr, []).append(node)

    max_layer = max(layers.keys()) if layers else 0
    pos = {}
    for lyr, lyr_nodes in layers.items():
        y = max_layer - lyr
        for i, node in enumerate(lyr_nodes):
            x = (i - (len(lyr_nodes) - 1) / 2) * 3.0
            pos[node] = (x, y * 2.0)

    fig, ax = plt.subplots(figsize=(12, max(8, len(G.nodes) * 0.7)))
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=25,
                           arrowstyle="-|>", edge_color="gray",
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=20, min_target_margin=20,
                           ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="lightblue",
                           ax=ax)
    labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)

    ax.set_title(f"Predicted Flowsheet — Cu = {grade:.2f}%", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {output_file}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(test_size=0.25, random_state=42, min_support=2):
    """Train/test split evaluation with per-stage F1 scores.

    Treats each canonical stage type as a binary label (present/absent)
    and computes F1 across the test set.  Stages with test-set support
    below min_support are excluded from scoring.
    """
    training_data = load_training_data()
    if len(training_data) < 5:
        print(f"Not enough data to evaluate ({len(training_data)} samples)")
        return

    X, Y, stage_types, edge_keys = build_feature_matrix(training_data)
    n_stages = len(stage_types)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    print(f"Evaluate: {len(X_train)} train / {len(X_test)} test samples")

    model = fit_model(X_train, Y_train)
    print(f"  Training R²: {model.score(X_train, Y_train):.4f}")

    Y_pred = model.predict(X_test)

    # Binary presence: actual count > 0 vs predicted count >= 0.5
    actual_presence = (Y_test[:, :n_stages] > 0).astype(int)
    pred_presence = (np.round(Y_pred[:, :n_stages]).clip(0) > 0).astype(int)

    # Enforce mandatory stages in predictions (matches postprocess logic)
    for i, st in enumerate(stage_types):
        if st in MANDATORY_STAGES:
            pred_presence[:, i] = 1

    # Per-stage F1
    print(f"\n  {'Stage':<22} {'F1':>6}  {'Prec':>6}  {'Rec':>6}  Support")
    print("  " + "-" * 60)
    stage_f1s = []
    for i, st in enumerate(stage_types):
        a_col = actual_presence[:, i]
        p_col = pred_presence[:, i]
        support = int(a_col.sum())
        if support < min_support and p_col.sum() == 0:
            continue  # insufficient support — skip from scoring
        f1 = f1_score(a_col, p_col, zero_division=0)
        prec = f1_score(a_col, p_col, zero_division=0,
                        average='binary') if support else 0
        # manual precision/recall for clarity
        tp = int((a_col & p_col).sum())
        prec = tp / max(p_col.sum(), 1)
        rec = tp / max(support, 1)
        stage_f1s.append(f1)
        print(f"  {st:<22} {f1:>6.3f}  {prec:>6.3f}  {rec:>6.3f}  {support:>4}")

    # Overall micro and macro F1 (only scored columns)
    scored = [i for i, st in enumerate(stage_types)
              if actual_presence[:, i].sum() >= min_support
              or pred_presence[:, i].sum() > 0]
    if scored:
        a_scored = actual_presence[:, scored].ravel()
        p_scored = pred_presence[:, scored].ravel()
    else:
        a_scored = actual_presence.ravel()
        p_scored = pred_presence.ravel()
    micro = f1_score(a_scored, p_scored, zero_division=0)
    macro = np.mean(stage_f1s) if stage_f1s else 0
    tp_all = int((a_scored & p_scored).sum())
    precision = tp_all / max(p_scored.sum(), 1)
    recall = tp_all / max(a_scored.sum(), 1)
    print("  " + "-" * 60)
    print(f"  {'Precision':<22} {precision:>6.3f}")
    print(f"  {'Recall':<22} {recall:>6.3f}")
    print(f"  {'Micro F1':<22} {micro:>6.3f}")
    print(f"  {'Macro F1':<22} {macro:>6.3f}")

    return {"micro_f1": micro, "macro_f1": macro, "precision": precision,
            "recall": recall, "n_train": len(X_train),
            "n_test": len(X_test)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(grade):
    """Load data, train, predict, draw."""
    print(f"Digital Twin: predicting flowsheet for Cu = {grade:.2f}%")

    if not (GRADE_MIN <= grade <= GRADE_MAX):
        print(f"  Warning: grade {grade:.2f}% is outside training range "
              f"[{GRADE_MIN}, {GRADE_MAX}] — extrapolation may be unreliable")

    # Load & prepare
    training_data = load_training_data()
    print(f"  Training samples: {len(training_data)}")
    for g, stage_counts, log_edges in training_data:
        print(f"    Cu={g:.4f}%  stages: {dict(stage_counts)}  "
              f"edges: {len(log_edges)}")

    X, Y, stage_types, edge_keys = build_feature_matrix(training_data)
    print(f"  Stage types ({len(stage_types)}): {stage_types}")
    print(f"  Edge features: {len(edge_keys)}")

    # Fit
    model = fit_model(X, Y)
    r2 = model.score(X, Y)
    print(f"  Training R^2: {r2:.4f}")

    # Predict
    X_new = np.array([[grade]])
    raw = model.predict(X_new)[0]
    predicted_stages, predicted_edges = postprocess(raw, stage_types, edge_keys)
    print(f"\n  Predicted stage counts for Cu = {grade:.2f}%:")
    for stage, count in predicted_stages.items():
        print(f"    {stage}: {count}")
    print(f"  Predicted connections: {len(predicted_edges)}")
    for (p, c), w in sorted(predicted_edges.items(), key=lambda x: -x[1]):
        print(f"    {p} -> {c}  (weight: {w:.2f})")

    # Graph
    G = build_predicted_graph(predicted_stages, predicted_edges)
    draw_predicted_graph(G, grade)

    return predicted_stages


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate()
    else:
        grades = [.25, .5, .75, 1.0, 1.25, 1.5, 1.75]
        for input_grade in grades:
            run(input_grade)
