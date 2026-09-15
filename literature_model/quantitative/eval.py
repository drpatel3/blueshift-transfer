"""Evaluation harness for quantitative flowsheet predictor.

Computes per-stage F1, macro/micro F1, connection accuracy.
Compares against baseline models.
"""

from __future__ import annotations

import numpy as np

# sklearn is only needed by full-corpus eval (compute_metrics et al.).
# _reachability_f1 (used by the deployed web app) is pure Python, so we
# defer the import — keeps Lambda images sklearn-free.


BASELINES = {
    "Mini polynomial (Cu grade only)": 0.70,
    "XGBoost V1 (23 stages, 609 feat)": 0.76,
    "XGBoost V2 (37 stages, 609 feat)": 0.66,
}


def evaluate_stages(y_true: np.ndarray, y_pred: np.ndarray,
                    stage_vocab: list[str], min_support: int = 2) -> dict:
    """Per-stage and aggregate F1 metrics.

    Args:
        y_true: (N, S) binary ground truth
        y_pred: (N, S) binary predictions
        stage_vocab: list of stage names
        min_support: minimum test-set positives to include in scoring
    """
    per_stage = []
    scored_f1s = []

    for j, stage in enumerate(stage_vocab):
        support = int(y_true[:, j].sum())
        if support < min_support:
            per_stage.append({
                "stage": stage, "f1": None, "precision": None,
                "recall": None, "support": support, "scored": False,
            })
            continue

        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = float(f1_score(y_true[:, j], y_pred[:, j], zero_division=0))
        prec = float(precision_score(y_true[:, j], y_pred[:, j], zero_division=0))
        rec = float(recall_score(y_true[:, j], y_pred[:, j], zero_division=0))
        per_stage.append({
            "stage": stage, "f1": f1, "precision": prec,
            "recall": rec, "support": support, "scored": True,
        })
        scored_f1s.append(f1)

    # Aggregate metrics (only on scored stages)
    scored_mask = np.array([s["scored"] for s in per_stage])
    scored_indices = np.where(scored_mask)[0]

    if len(scored_indices) > 0:
        y_true_scored = y_true[:, scored_indices]
        y_pred_scored = y_pred[:, scored_indices]
        from sklearn.metrics import f1_score
        macro_f1 = float(np.mean(scored_f1s))
        micro_f1 = float(f1_score(y_true_scored.ravel(), y_pred_scored.ravel(), zero_division=0))
    else:
        macro_f1 = 0.0
        micro_f1 = 0.0

    return {
        "per_stage": per_stage,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "n_scored": len(scored_f1s),
        "n_total": len(stage_vocab),
    }


def evaluate_connections(pred_connections: list[set],
                         true_connections: list[set],
                         stage_vocab: list[str]) -> dict:
    """Per-doc connection F1 (edge-level) plus per-edge tallies.

    per_edge aggregates tp/fp/fn across all docs for each (src, dst) edge
    that appears in either predictions or ground truth, so we can see
    which edges KNN gets right and which it misses.
    """
    f1s, precs, recs = [], [], []
    per_edge_counts = {}  # (src, dst) -> {tp, fp, fn}

    def _bump(edge, key):
        if edge not in per_edge_counts:
            per_edge_counts[edge] = {"tp": 0, "fp": 0, "fn": 0}
        per_edge_counts[edge][key] += 1

    for pred_edges, true_edges in zip(pred_connections, true_connections):
        for edge in pred_edges & true_edges:
            _bump(edge, "tp")
        for edge in pred_edges - true_edges:
            _bump(edge, "fp")
        for edge in true_edges - pred_edges:
            _bump(edge, "fn")

        if not true_edges and not pred_edges:
            continue
        tp = len(pred_edges & true_edges)
        fp = len(pred_edges - true_edges)
        fn = len(true_edges - pred_edges)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    per_edge = {}
    for edge, c in per_edge_counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_edge[edge] = {
            "tp": tp, "fp": fp, "fn": fn,
            "f1": float(f1),
            "support": tp + fn,  # ground-truth occurrences
        }

    return {
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_precision": float(np.mean(precs)) if precs else 0.0,
        "mean_recall": float(np.mean(recs)) if recs else 0.0,
        "n_docs": len(f1s),
        "per_edge": per_edge,
    }


# ---------------------------------------------------------------------------
# Ordering evaluation
# ---------------------------------------------------------------------------

def _topological_sort(stages: set, edges: set) -> list:
    """Kahn's algorithm — returns ordered stage list from directed edges."""
    from collections import defaultdict, deque
    adj = defaultdict(list)
    in_deg = {s: 0 for s in stages}
    for src, dst in edges:
        if src in stages and dst in stages:
            adj[src].append(dst)
            in_deg[dst] = in_deg.get(dst, 0) + 1

    queue = deque(sorted(s for s in stages if in_deg.get(s, 0) == 0))
    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for neighbor in sorted(adj[node]):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    # Append remaining (cycles or disconnected)
    remaining = [s for s in sorted(stages) if s not in set(ordered)]
    ordered.extend(remaining)
    return ordered


def _kendall_tau(order_a: list, order_b: list) -> float:
    """Kendall's Tau between two orderings of the same items.

    Only considers items present in both lists. Returns value in [-1, 1]
    where 1 = identical ordering, -1 = reversed, 0 = no correlation.
    """
    common = [s for s in order_a if s in set(order_b)]
    if len(common) < 2:
        return 0.0

    # Build rank maps
    rank_a = {s: i for i, s in enumerate(order_a) if s in set(common)}
    rank_b = {s: i for i, s in enumerate(order_b) if s in set(common)}

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            s1, s2 = common[i], common[j]
            diff_a = rank_a[s1] - rank_a[s2]
            diff_b = rank_b[s1] - rank_b[s2]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1

    n_pairs = concordant + discordant
    if n_pairs == 0:
        return 0.0
    return (concordant - discordant) / n_pairs


def _lcs_ratio(seq_a: list, seq_b: list) -> float:
    """Longest common subsequence length / max(len(a), len(b)).

    Measures how much of the ground truth ordering is preserved
    in the predicted ordering. 1.0 = perfect subsequence match.
    """
    common_set = set(seq_a) & set(seq_b)
    a = [s for s in seq_a if s in common_set]
    b = [s for s in seq_b if s in common_set]
    if not a or not b:
        return 0.0

    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[n][m]
    return lcs_len / max(n, m)


def _reachability_f1(pred_edges: set, true_edges: set, stages: set) -> dict:
    """Reachability precision, recall, and F1 over the (a→…→b) pair set.

    Reachability is the right semantic for "do material flows match between two
    flowsheets?": it abstracts over edge granularity (collapsing intermediate
    stages doesn't change reachability) and captures the actual question
    "from stage a, can material reach stage b?".

    Recall  = |gt_reach_pairs ∩ pred_reach_pairs| / |gt_reach_pairs|
    Precision = |gt_reach_pairs ∩ pred_reach_pairs| / |pred_reach_pairs|
    F1 = harmonic mean

    The precision term stops a fully-connected graph from scoring 1.0 — that
    failure mode broke the previous pure-recall version.
    """
    from collections import defaultdict, deque

    def _build_reachable(edges, nodes):
        adj = defaultdict(set)
        for src, dst in edges:
            if src in nodes and dst in nodes:
                adj[src].add(dst)
        pairs = set()
        for start in nodes:
            visited = set()
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor in visited or neighbor == start:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)
            for dst in visited:
                pairs.add((start, dst))
        return pairs

    true_pairs = _build_reachable(true_edges, stages)
    pred_pairs = _build_reachable(pred_edges, stages)

    if not true_pairs and not pred_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    correct = len(true_pairs & pred_pairs)
    recall = correct / len(true_pairs) if true_pairs else 0.0
    precision = correct / len(pred_pairs) if pred_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_ordering(pred_connections: list[set],
                      true_connections: list[set],
                      pred_stages: list[set],
                      true_stages: list[set]) -> dict:
    """Evaluate predicted stage ordering quality.

    Computes per-doc:
    - Kendall's Tau on topological sort ordering
    - LCS ratio (longest common subsequence / max length)
    - Reachability accuracy (upstream/downstream relationships preserved)

    Args:
        pred_connections: list of sets of (src, dst) predicted edges per doc
        true_connections: list of sets of (src, dst) ground truth edges per doc
        pred_stages: list of sets of predicted stage names per doc
        true_stages: list of sets of ground truth stage names per doc
    """
    taus, lcs_ratios = [], []
    reach_f1s, reach_precs, reach_recs = [], [], []

    for pred_edges, true_edges, pred_stg, true_stg in zip(
            pred_connections, true_connections, pred_stages, true_stages):
        if not true_edges or not true_stg:
            continue

        # Common stages (intersection of predicted and true)
        common = pred_stg & true_stg
        if len(common) < 2:
            continue

        # Topological sort both
        true_order = _topological_sort(common, true_edges)
        pred_order = _topological_sort(common, pred_edges)

        tau = _kendall_tau(true_order, pred_order)
        lcs = _lcs_ratio(true_order, pred_order)
        reach = _reachability_f1(pred_edges, true_edges, common)

        taus.append(tau)
        lcs_ratios.append(lcs)
        reach_f1s.append(reach["f1"])
        reach_precs.append(reach["precision"])
        reach_recs.append(reach["recall"])

    return {
        "kendall_tau": float(np.mean(taus)) if taus else 0.0,
        "lcs_ratio": float(np.mean(lcs_ratios)) if lcs_ratios else 0.0,
        "reach_f1": float(np.mean(reach_f1s)) if reach_f1s else 0.0,
        "reach_precision": float(np.mean(reach_precs)) if reach_precs else 0.0,
        "reach_recall": float(np.mean(reach_recs)) if reach_recs else 0.0,
        "n_docs": len(taus),
    }


def print_report(stage_metrics: dict, conn_metrics: dict, model_name: str,
                 ordering_metrics: dict = None):
    """Print formatted evaluation report."""
    print(f"\n{'=' * 60}")
    print(f"  {model_name} — Validation Results")
    print(f"{'=' * 60}")

    # Per-stage table
    print(f"\n{'Stage':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Support':>8}")
    print("-" * 55)
    for s in sorted(stage_metrics["per_stage"], key=lambda x: -(x["f1"] or -1)):
        if s["scored"]:
            print(f"{s['stage']:<25} {s['f1']:>6.3f} {s['precision']:>6.3f} "
                  f"{s['recall']:>6.3f} {s['support']:>8}")
        else:
            print(f"{s['stage']:<25} {'—':>6} {'—':>6} {'—':>6} {s['support']:>8}")

    print("-" * 55)
    print(f"{'Macro F1':<25} {stage_metrics['macro_f1']:>6.3f}")
    print(f"{'Micro F1':<25} {stage_metrics['micro_f1']:>6.3f}")
    print(f"{'Scored stages':<25} {stage_metrics['n_scored']:>6}/{stage_metrics['n_total']}")

    # Connections
    if conn_metrics["n_docs"] > 0:
        print(f"\nConnections ({conn_metrics['n_docs']} docs):")
        print(f"  Mean F1:        {conn_metrics['mean_f1']:.3f}")
        print(f"  Mean Precision: {conn_metrics['mean_precision']:.3f}")
        print(f"  Mean Recall:    {conn_metrics['mean_recall']:.3f}")

        per_edge = conn_metrics.get("per_edge", {})
        if per_edge:
            scored = [(edge, m) for edge, m in per_edge.items() if m["support"] >= 2]
            if scored:
                print(f"\n  Top edges by support:")
                print(f"  {'Edge':<45} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Sup':>4}")
                print(f"  {'-' * 70}")
                top = sorted(scored, key=lambda x: -x[1]["support"])[:10]
                for edge, m in top:
                    label = f"{edge[0]} -> {edge[1]}"
                    print(f"  {label:<45} {m['f1']:>6.3f} {m['tp']:>4} "
                          f"{m['fp']:>4} {m['fn']:>4} {m['support']:>4}")
                worst = sorted(scored, key=lambda x: (x[1]["f1"], -x[1]["support"]))[:5]
                print(f"\n  Worst edges (lowest F1, support >= 2):")
                for edge, m in worst:
                    label = f"{edge[0]} -> {edge[1]}"
                    print(f"  {label:<45} {m['f1']:>6.3f} {m['tp']:>4} "
                          f"{m['fp']:>4} {m['fn']:>4} {m['support']:>4}")

    # Ordering metrics
    if ordering_metrics and ordering_metrics.get("n_docs", 0) > 0:
        print(f"\nFlow Structure Quality ({ordering_metrics['n_docs']} docs):")
        print(f"  Reachability F1:        {ordering_metrics['reach_f1']:.3f}  "
              f"(primary — directed flow correctness)")
        print(f"    precision:            {ordering_metrics['reach_precision']:.3f}")
        print(f"    recall:               {ordering_metrics['reach_recall']:.3f}")
        print(f"  Kendall's Tau:          {ordering_metrics['kendall_tau']:.3f}  "
              f"(local pairwise order)")
        print(f"  LCS Ratio:              {ordering_metrics['lcs_ratio']:.3f}  "
              f"(longest common subsequence)")

    # Baseline comparison
    print(f"\nBaseline Comparison:")
    print(f"  {'Model':<35} {'Macro F1':>8}")
    print(f"  {'-' * 45}")
    for name, f1 in BASELINES.items():
        delta = stage_metrics["macro_f1"] - f1
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<35} {f1:>8.2f}  ({sign}{delta:.2f})")
    print(f"  {'>>> ' + model_name:<35} {stage_metrics['macro_f1']:>8.2f}")
    print()
