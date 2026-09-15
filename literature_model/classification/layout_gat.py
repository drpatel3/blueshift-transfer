"""LayoutGAT — Graph Attention Network for process flowsheet layout optimization.

Produces (x, y) coordinates for stage nodes in flowsheet graphs. Trained on
pseudo-ground-truth layouts from networkx multipartite_layout, with physics-
inspired loss terms for edge crossing, spacing, flow direction, and branching.

Usage (SageMaker entry script):
    Hyperparameters: graphs-bucket, graphs-prefix, epochs, lr
"""

import argparse
import json
import logging
import math
import os
from collections import defaultdict

import boto3
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Comminution",
    "Concentration",
    "Gravity",
    "Hydrometallurgy",
    "Pyrometallurgy",
    "Material_Handling",
    "Auxiliary",
    "Specialized",
    "Standalone",
    "Other",
]

PROCESS_ORDER = {
    "Comminution": 1,
    "Concentration": 2,
    "Gravity": 2,
    "Hydrometallurgy": 3,
    "Pyrometallurgy": 4,
    "Material_Handling": 0,
    "Auxiliary": 5,
    "Specialized": 3,
    "Standalone": 5,
    "Other": 6,
}

SECTION_FEATURES = [
    "summary",
    "geology",
    "exploration",
    "drilling",
    "sample_analysis",
    "metallurgical_testing",
    "resource_estimate",
    "mining_method",
    "recovery_methods",
    "infrastructure",
    "economics",
]

# Maps every standard stage_id from TERMS.md to its category.
CATEGORY_MAP = {
    # Comminution
    "crusher": "Comminution",
    "mill": "Comminution",
    "regrind": "Comminution",
    "screen": "Comminution",
    "agglomeration": "Comminution",
    "cyclone": "Comminution",
    "input": "Comminution",
    # Concentration
    "flotation": "Concentration",
    "magnetic_separation": "Concentration",
    "thickener": "Concentration",
    "filter": "Concentration",
    "cell": "Concentration",
    # Gravity Separation
    "gravity": "Gravity",
    # Hydrometallurgy
    "leach": "Hydrometallurgy",
    "adsorption": "Hydrometallurgy",
    "precipitation": "Hydrometallurgy",
    "electrowinning": "Hydrometallurgy",
    "elution": "Hydrometallurgy",
    "solvent_extraction": "Hydrometallurgy",
    "merrill_crowe": "Hydrometallurgy",
    "ion_exchange": "Hydrometallurgy",
    # Pyrometallurgy
    "kiln": "Pyrometallurgy",
    "reactor": "Pyrometallurgy",
    # Material Handling
    "stockpile": "Material_Handling",
    "hopper": "Material_Handling",
    "feeder": "Material_Handling",
    "conveyor": "Material_Handling",
    "bin": "Material_Handling",
    # Auxiliary
    "tank": "Auxiliary",
    "pond": "Auxiliary",
    "tailing": "Auxiliary",
    "drying": "Auxiliary",
    "water_treatment": "Auxiliary",
    # Specialized Stages
    "ore_sorting": "Specialized",
    "copper_concentrate_dewatering": "Specialized",
    "copper_concentrate_storage": "Specialized",
    "zinc_concentrate_dewatering": "Specialized",
    "zinc_concentrate_storage": "Specialized",
    "sars": "Specialized",
    "mol": "Specialized",
    # Standalone Equipment
    "carbon_handling": "Standalone",
    "dore": "Standalone",
    "refinery": "Standalone",
    "detox": "Standalone",
    "water_treatment_plant": "Standalone",
    "diverter_gate": "Standalone",
    "hood": "Standalone",
    "super_sac": "Standalone",
    "copper_concentrate_shipping": "Standalone",
    "final_product_handling": "Standalone",
    "lead_oxide_product": "Standalone",
}

NODE_FEAT_DIM = 35
EDGE_FEAT_DIM = 3


# ---------------------------------------------------------------------------
# S3 data helpers
# ---------------------------------------------------------------------------

def _load_graphs_from_s3(graphs_bucket: str, graphs_prefix: str):
    """Load all graph JSONs from S3. Returns list of parsed dicts."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    graphs = []
    for page in paginator.paginate(Bucket=graphs_bucket, Prefix=graphs_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                try:
                    resp = s3.get_object(Bucket=graphs_bucket, Key=key)
                    graph = json.loads(resp["Body"].read())
                    graphs.append(graph)
                except Exception as e:
                    logger.warning(f"Failed to load {key}: {e}")
    logger.info(f"Loaded {len(graphs)} graphs from s3://{graphs_bucket}/{graphs_prefix}")
    return graphs


# ---------------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------------

def build_corpus_stats(graphs):
    """Compute per-stage statistics from pre-loaded graph dicts.

    Args:
        graphs: list of parsed graph dicts (from _load_graphs_from_s3)

    Returns dict[stage_id] -> {
        frequency, mean_order, mean_in_degree, mean_out_degree,
        num_distinct_parents, num_distinct_children,
        start_fraction, end_fraction,
        section_influences: {section -> weight}
    }
    """

    stats = defaultdict(lambda: {
        "orders": [],
        "in_degrees": [],
        "out_degrees": [],
        "parent_types": set(),
        "child_types": set(),
        "is_start": 0,
        "is_end": 0,
        "count": 0,
        "section_influences": defaultdict(float),
    })

    num_docs = len(graphs)

    for graph in graphs:
        nodes_by_id = {}
        stage_ids = []
        for node in graph.get("nodes", []):
            nodes_by_id[node["id"]] = node
            if node.get("type") == "stage":
                sid = node.get("stage_id", node["id"].replace("stg_", ""))
                stage_ids.append(sid)
                feats = node.get("features", {})
                s = stats[sid]
                s["count"] += 1
                s["orders"].append(feats.get("order_normalized", 0.5))
                s["in_degrees"].append(feats.get("in_degree", 0))
                s["out_degrees"].append(feats.get("out_degree", 0))
                if feats.get("in_degree", 0) == 0:
                    s["is_start"] += 1
                if feats.get("is_terminal", False):
                    s["is_end"] += 1

        # Collect parent/child types and section influences from edges
        for edge in graph.get("edges", []):
            etype = edge.get("type", "")
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if etype == "stage_transition":
                src_sid = src.replace("stg_", "")
                tgt_sid = tgt.replace("stg_", "")
                stats[tgt_sid]["parent_types"].add(src_sid)
                stats[src_sid]["child_types"].add(tgt_sid)
            elif etype == "context_influences_stage":
                # src is ctx_{group}, tgt is stg_{stage_id}
                group = src.replace("ctx_", "")
                tgt_sid = tgt.replace("stg_", "")
                weight = edge.get("weight", 1.0)
                stats[tgt_sid]["section_influences"][group] += weight

    # Aggregate into final stats dict
    result = {}
    for sid, s in stats.items():
        n = max(s["count"], 1)
        result[sid] = {
            "frequency": s["count"],
            "mean_order": float(np.mean(s["orders"])) if s["orders"] else 0.5,
            "mean_in_degree": float(np.mean(s["in_degrees"])) if s["in_degrees"] else 0.0,
            "mean_out_degree": float(np.mean(s["out_degrees"])) if s["out_degrees"] else 0.0,
            "num_distinct_parents": len(s["parent_types"]),
            "num_distinct_children": len(s["child_types"]),
            "start_fraction": s["is_start"] / n,
            "end_fraction": s["is_end"] / n,
            "section_influences": dict(s["section_influences"]),
        }

    logger.info(f"Corpus stats built for {len(result)} stage types across {num_docs} docs")
    return result


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def extract_stage_subgraphs(graph_data: dict):
    """Extract stage nodes and stage_transition edges from a graph JSON.

    Classifies forward/back edges via DFS from root nodes (in_degree == 0).

    Returns dict with:
        stages: list of stage_ids
        edges: list of (src_sid, dst_sid, weight)
        back_edges: set of (src_sid, dst_sid)
    """
    stages = []
    stage_set = set()
    for node in graph_data.get("nodes", []):
        if node.get("type") == "stage":
            sid = node.get("stage_id", node["id"].replace("stg_", ""))
            stages.append(sid)
            stage_set.add(sid)

    edges = []
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    for edge in graph_data.get("edges", []):
        if edge.get("type") == "stage_transition":
            src = edge["source"].replace("stg_", "")
            tgt = edge["target"].replace("stg_", "")
            if src in stage_set and tgt in stage_set:
                w = edge.get("weight", 1.0)
                edges.append((src, tgt, w))
                adj[src].append(tgt)
                in_degree[tgt] += 1

    # DFS to classify back edges
    roots = [s for s in stages if in_degree.get(s, 0) == 0]
    if not roots:
        roots = stages[:1]  # fallback: pick first node

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {s: WHITE for s in stages}
    back_edges = set()

    def dfs(u):
        color[u] = GRAY
        for v in adj.get(u, []):
            if color[v] == GRAY:
                back_edges.add((u, v))
            elif color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    for root in roots:
        if color[root] == WHITE:
            dfs(root)

    return {
        "stages": stages,
        "edges": edges,
        "back_edges": back_edges,
    }


# ---------------------------------------------------------------------------
# Reference layout generation
# ---------------------------------------------------------------------------

def generate_reference_layout(stages, edges):
    """Generate pseudo-ground-truth (x, y) using nx.multipartite_layout.

    Assigns each node a 'subset' = topological depth for multipartite_layout.
    Normalizes output coordinates to [0, 1] range.

    Returns dict[stage_id] -> (x, y).
    """
    G = nx.DiGraph()
    G.add_nodes_from(stages)
    for src, dst, _w in edges:
        G.add_edge(src, dst)

    # Compute topological depth via BFS from roots
    in_deg = dict(G.in_degree())
    roots = [n for n in stages if in_deg.get(n, 0) == 0]
    if not roots:
        roots = stages[:1]

    depth = {s: 0 for s in stages}
    visited = set()
    queue = list(roots)
    for r in roots:
        visited.add(r)

    while queue:
        u = queue.pop(0)
        for v in G.successors(u):
            new_depth = depth[u] + 1
            if new_depth > depth[v]:
                depth[v] = new_depth
            if v not in visited:
                visited.add(v)
                queue.append(v)

    # Set subset attribute for multipartite_layout
    for node in stages:
        G.nodes[node]["subset"] = depth[node]

    pos = nx.multipartite_layout(G, subset_key="subset")

    # Normalize to [0, 1]
    if not pos:
        return {}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    result = {}
    for sid, (x, y) in pos.items():
        result[sid] = ((x - x_min) / x_range, (y - y_min) / y_range)

    return result


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _get_category(stage_id: str) -> str:
    """Look up category for a stage_id, falling back to Other."""
    if stage_id in CATEGORY_MAP:
        return CATEGORY_MAP[stage_id]
    # Try base form (strip prefixes like copper_, zinc_, etc.)
    base = stage_id.split("_")[-1] if "_" in stage_id else stage_id
    return CATEGORY_MAP.get(base, "Other")


def build_layout_features(stages, edges, back_edges, corpus_stats):
    """Construct node features, edge index, and edge attributes as tensors.

    Returns:
        node_features: (N, 35) tensor
        edge_index: (2, E) long tensor
        edge_attr: (E, 3) tensor
    """
    N = len(stages)
    sid_to_idx = {s: i for i, s in enumerate(stages)}

    # Build adjacency for BFS
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    for src, dst, _w in edges:
        adj[src].append(dst)
        out_degree[src] += 1
        in_degree[dst] += 1

    # Topological depth/breadth via BFS from roots
    roots = [s for s in stages if in_degree.get(s, 0) == 0]
    if not roots:
        roots = stages[:1]

    depth = {s: 0 for s in stages}
    visited = set()
    queue = list(roots)
    for r in roots:
        visited.add(r)

    while queue:
        u = queue.pop(0)
        for v in adj.get(u, []):
            new_d = depth[u] + 1
            if new_d > depth[v]:
                depth[v] = new_d
            if v not in visited:
                visited.add(v)
                queue.append(v)

    # Breadth = number of nodes at each depth level
    depth_counts = defaultdict(int)
    for d in depth.values():
        depth_counts[d] += 1
    max_depth = max(depth.values()) if depth else 0

    # Build node features (N x 35)
    node_features = torch.zeros(N, NODE_FEAT_DIM, dtype=torch.float32)

    for i, sid in enumerate(stages):
        offset = 0

        # Category one-hot (10 dims)
        cat = _get_category(sid)
        cat_idx = CATEGORIES.index(cat) if cat in CATEGORIES else len(CATEGORIES) - 1
        node_features[i, cat_idx] = 1.0
        offset += 10

        # Topological features (6 dims)
        node_features[i, offset + 0] = float(in_degree.get(sid, 0))
        node_features[i, offset + 1] = float(out_degree.get(sid, 0))
        d = depth.get(sid, 0)
        node_features[i, offset + 2] = d / max(max_depth, 1)  # topological_depth normalized
        node_features[i, offset + 3] = depth_counts.get(d, 1) / max(N, 1)  # breadth normalized
        node_features[i, offset + 4] = 1.0 if in_degree.get(sid, 0) == 0 else 0.0  # is_root
        node_features[i, offset + 5] = 1.0 if out_degree.get(sid, 0) == 0 else 0.0  # is_terminal
        offset += 6

        # Corpus statistics (8 dims)
        cs = corpus_stats.get(sid, {})
        node_features[i, offset + 0] = cs.get("mean_order", 0.5)
        node_features[i, offset + 1] = math.log1p(cs.get("frequency", 0)) / 10.0  # log-scaled
        node_features[i, offset + 2] = cs.get("mean_in_degree", 0.0)
        node_features[i, offset + 3] = cs.get("mean_out_degree", 0.0)
        node_features[i, offset + 4] = cs.get("num_distinct_parents", 0) / 10.0
        node_features[i, offset + 5] = cs.get("num_distinct_children", 0) / 10.0
        node_features[i, offset + 6] = cs.get("start_fraction", 0.0)
        node_features[i, offset + 7] = cs.get("end_fraction", 0.0)
        offset += 8

        # Section association (11 dims)
        sec_inf = cs.get("section_influences", {})
        # Normalize section influences to sum to 1
        total_inf = sum(sec_inf.values()) if sec_inf else 1.0
        total_inf = max(total_inf, 1e-8)
        for j, sec in enumerate(SECTION_FEATURES):
            node_features[i, offset + j] = sec_inf.get(sec, 0.0) / total_inf
        offset += 11

    # Build edge_index (2 x E) and edge_attr (E x 3)
    E = len(edges)
    edge_index = torch.zeros(2, E, dtype=torch.long)
    edge_attr = torch.zeros(E, EDGE_FEAT_DIM, dtype=torch.float32)

    for ei, (src, dst, w) in enumerate(edges):
        edge_index[0, ei] = sid_to_idx[src]
        edge_index[1, ei] = sid_to_idx[dst]

        # Log-compressed weight
        edge_attr[ei, 0] = math.log1p(w)

        # Direction encoding
        is_back = (src, dst) in back_edges
        edge_attr[ei, 1] = -1.0 if is_back else 1.0

        # Edge type: back_edge=recycle=0.0, forward=flow=1.0
        # Heuristic: if src has multiple children going to same depth, parallel_split=0.5
        children_of_src = adj.get(src, [])
        if not is_back and len(children_of_src) > 1:
            sibling_depths = [depth.get(c, 0) for c in children_of_src]
            if sibling_depths.count(depth.get(dst, 0)) > 1:
                edge_attr[ei, 2] = 0.5  # parallel_split
            else:
                edge_attr[ei, 2] = 1.0  # flow
        elif is_back:
            edge_attr[ei, 2] = 0.0  # recycle
        else:
            edge_attr[ei, 2] = 1.0  # flow

    # Build category_ranks tensor (N,) for physics-informed losses
    category_ranks = torch.zeros(N, dtype=torch.float32)
    for i, sid in enumerate(stages):
        cat = _get_category(sid)
        category_ranks[i] = float(PROCESS_ORDER.get(cat, 6))

    return node_features, edge_index, edge_attr, category_ranks


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LayoutGAT(nn.Module):
    """Two-layer GAT producing (x, y) coordinates per node.

    Architecture:
        Input (35) -> GATConv(35, 64, heads=2, edge_dim=3) -> ELU
                   -> GATConv(128, 32, heads=2, edge_dim=3) -> ELU
                   -> Linear(64, 2) -> (x, y)
                   -> ConstraintProjection (enforce topo x-order, min spacing)
    """

    def __init__(self):
        super().__init__()
        self.gat1 = GATConv(NODE_FEAT_DIM, 64, heads=2, edge_dim=EDGE_FEAT_DIM)
        self.gat2 = GATConv(128, 32, heads=2, edge_dim=EDGE_FEAT_DIM)
        self.head = nn.Linear(64, 2)

    def forward(self, x, edge_index, edge_attr, topo_depths=None):
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        coords = self.head(x)

        # Constraint projection: enforce topological x-ordering and min spacing
        if topo_depths is not None:
            coords = _constraint_projection(coords, topo_depths)

        return coords


def _constraint_projection(coords, topo_depths, min_spacing=0.05):
    """Differentiable post-hoc projection enforcing layout constraints.

    - Soft-sort x-coordinates to respect topological depth ordering
    - Clamp coordinates to [0, 1] range
    - Enforce minimum spacing between nodes at the same depth layer
    """
    x, y = coords[:, 0], coords[:, 1]

    # Soft topological x-ordering: nudge x toward depth-proportional targets
    # This is a soft constraint (weighted residual) so gradients still flow
    max_depth = topo_depths.max()
    if max_depth > 0:
        target_x = topo_depths / max_depth
        # Blend: 80% model output + 20% depth-ordered target
        x = 0.8 * x + 0.2 * target_x

    # Clamp to [0, 1]
    x = torch.sigmoid(x)  # smooth clamp preserving gradients
    y = torch.sigmoid(y)

    # Enforce minimum y-spacing within same depth layer (differentiable push)
    unique_depths = topo_depths.unique()
    for d in unique_depths:
        mask = (topo_depths == d)
        if mask.sum() < 2:
            continue
        indices = mask.nonzero(as_tuple=True)[0]
        layer_y = y[indices]
        # Sort within layer and push apart if too close
        sorted_y, sort_idx = layer_y.sort()
        for k in range(1, len(sorted_y)):
            gap = sorted_y[k] - sorted_y[k - 1]
            if gap < min_spacing:
                push = (min_spacing - gap) / 2.0
                # Nudge via soft addition (keeps gradients)
                y = y.clone()
                y[indices[sort_idx[k]]] = y[indices[sort_idx[k]]] + push
                y[indices[sort_idx[k - 1]]] = y[indices[sort_idx[k - 1]]] - push

    return torch.stack([x, y], dim=1)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_layout_loss(pred_coords, ref_coords, edge_index, edge_types, topo_depths,
                        category_ranks=None):
    """Compute combined layout loss.

    Args:
        pred_coords: (N, 2) predicted coordinates
        ref_coords: (N, 2) reference coordinates from multipartite_layout
        edge_index: (2, E) edge indices
        edge_types: (E,) tensor, 1.0=forward, -1.0=back_edge
        topo_depths: (N,) tensor of normalized topological depths
        category_ranks: (N,) tensor of process order ranks (optional, for physics losses)

    Returns:
        total_loss: scalar
        loss_dict: dict of individual loss components
    """
    N = pred_coords.size(0)
    E = edge_index.size(1)

    # L_supervised: MSE to reference layout
    l_supervised = F.mse_loss(pred_coords, ref_coords)

    # L_flow: parent.x should be less than child.x for forward edges
    margin = 0.1
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    src_x = pred_coords[src_idx, 0]
    dst_x = pred_coords[dst_idx, 0]
    flow_penalty = F.relu(src_x - dst_x + margin)
    # Weight back-edges at 0.1x
    flow_weights = torch.where(edge_types > 0, torch.ones_like(edge_types), 0.1 * torch.ones_like(edge_types))
    l_flow = (flow_penalty * flow_weights).mean() if E > 0 else torch.tensor(0.0, device=pred_coords.device)

    # L_crossing: Vectorized soft crossing penalty via signed-area products
    l_crossing = torch.tensor(0.0, device=pred_coords.device)
    if E > 1:
        p1 = pred_coords[edge_index[0]]  # (E, 2)
        p2 = pred_coords[edge_index[1]]  # (E, 2)

        # Build upper-triangular pair indices
        ii, jj = torch.triu_indices(E, E, offset=1, device=pred_coords.device)

        # Mask out pairs sharing a node (vectorized)
        share = ((edge_index[0, ii] == edge_index[0, jj]) |
                 (edge_index[0, ii] == edge_index[1, jj]) |
                 (edge_index[1, ii] == edge_index[0, jj]) |
                 (edge_index[1, ii] == edge_index[1, jj]))
        valid = ~share
        if valid.any():
            vi, vj = ii[valid], jj[valid]
            a, b = p1[vi], p2[vi]  # (P, 2)
            c, d = p1[vj], p2[vj]  # (P, 2)

            # Signed area products (fully batched)
            d1 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            d2 = (b[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0])
            d3 = (d[:, 0] - c[:, 0]) * (a[:, 1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
            d4 = (d[:, 0] - c[:, 0]) * (b[:, 1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (b[:, 0] - c[:, 0])

            cross1 = torch.sigmoid(-d1 * d2 * 100.0)
            cross2 = torch.sigmoid(-d3 * d4 * 100.0)
            l_crossing = (cross1 * cross2).mean()

    # L_spacing: prevent overlap
    min_dist = 0.15
    if N > 1:
        # Pairwise distances
        diff = pred_coords.unsqueeze(0) - pred_coords.unsqueeze(1)  # (N, N, 2)
        dists = torch.norm(diff, dim=2)  # (N, N)
        # Mask diagonal
        mask = 1.0 - torch.eye(N, device=pred_coords.device)
        spacing_violations = F.relu(min_dist - dists) * mask
        l_spacing = spacing_violations.sum() / (N * (N - 1))
    else:
        l_spacing = torch.tensor(0.0, device=pred_coords.device)

    # L_branch: for multi-child parents, force parallel paths apart
    min_y_gap = 0.1
    l_branch = torch.tensor(0.0, device=pred_coords.device)
    if E > 0:
        # Group children by parent
        parent_children = defaultdict(list)
        for ei in range(E):
            if edge_types[ei] > 0:  # forward edges only
                parent = edge_index[0, ei].item()
                child = edge_index[1, ei].item()
                parent_children[parent].append(child)

        branch_penalties = []
        for parent, children in parent_children.items():
            if len(children) < 2:
                continue
            child_ys = pred_coords[children, 1]
            # All pairs of children
            for ci in range(len(children)):
                for cj in range(ci + 1, len(children)):
                    gap = torch.abs(child_ys[ci] - child_ys[cj])
                    branch_penalties.append(F.relu(min_y_gap - gap))

        if branch_penalties:
            l_branch = torch.stack(branch_penalties).mean()

    # --- Physics-informed losses (Phase 5) ---

    # L_process_order: penalize when category_rank(parent) > category_rank(child) for forward edges
    l_process_order = torch.tensor(0.0, device=pred_coords.device)
    if category_ranks is not None and E > 0:
        fwd_mask = edge_types > 0
        if fwd_mask.any():
            src_ranks = category_ranks[edge_index[0][fwd_mask]]
            dst_ranks = category_ranks[edge_index[1][fwd_mask]]
            # Penalize when parent rank > child rank (wrong process order)
            l_process_order = F.relu(src_ranks - dst_ranks).mean()

    # L_recycle: back-edges should route above or below the main flow band
    l_recycle = torch.tensor(0.0, device=pred_coords.device)
    if E > 0:
        back_mask = edge_types < 0
        if back_mask.any():
            # Main flow band: y range of forward-edge nodes
            fwd_mask = edge_types > 0
            if fwd_mask.any():
                fwd_nodes = torch.cat([edge_index[0][fwd_mask], edge_index[1][fwd_mask]]).unique()
                fwd_ys = pred_coords[fwd_nodes, 1]
                flow_center = fwd_ys.mean()
                flow_half = (fwd_ys.max() - fwd_ys.min()) / 2.0 + 0.05
                # Back-edge midpoints should be outside the flow band
                back_src_y = pred_coords[edge_index[0][back_mask], 1]
                back_dst_y = pred_coords[edge_index[1][back_mask], 1]
                back_mid_y = (back_src_y + back_dst_y) / 2.0
                dist_from_center = torch.abs(back_mid_y - flow_center)
                l_recycle = F.relu(flow_half - dist_from_center).mean()

    # L_mass_balance: early stages (low depth) get more vertical spacing
    l_mass_balance = torch.tensor(0.0, device=pred_coords.device)
    if N > 1:
        # Throughput proxy: exp(-depth), early stages have higher throughput
        throughput = torch.exp(-topo_depths * 3.0)  # scale factor for separation
        # For each pair, desired spacing scales with max throughput of the pair
        diff_y = pred_coords.unsqueeze(0) - pred_coords.unsqueeze(1)  # (N, N, 2)
        dists_y = torch.abs(diff_y[:, :, 1])  # (N, N) y-distances
        mask = 1.0 - torch.eye(N, device=pred_coords.device)
        # Desired min spacing proportional to throughput
        tp_matrix = torch.maximum(throughput.unsqueeze(0), throughput.unsqueeze(1))
        desired_spacing = 0.08 * tp_matrix  # base spacing * throughput factor
        mass_violations = F.relu(desired_spacing - dists_y) * mask
        l_mass_balance = mass_violations.sum() / max((mask.sum()).item(), 1.0)

    # Combined loss: base losses + physics-informed losses
    total = (l_supervised + 0.3 * l_flow + 0.2 * l_crossing + 0.2 * l_spacing + 0.3 * l_branch
             + 0.2 * l_process_order + 0.15 * l_recycle + 0.1 * l_mass_balance)

    loss_dict = {
        "supervised": l_supervised.item(),
        "flow": l_flow.item(),
        "crossing": l_crossing.item(),
        "spacing": l_spacing.item() if isinstance(l_spacing, torch.Tensor) else l_spacing,
        "branch": l_branch.item(),
        "process_order": l_process_order.item(),
        "recycle": l_recycle.item(),
        "mass_balance": l_mass_balance.item(),
        "total": total.item(),
    }

    return total, loss_dict


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_layout(graphs_bucket: str, graphs_prefix: str, epochs: int = 500,
                 lr: float = 1e-3, output_dir: str = "/opt/ml/model"):
    """Full training loop: load graphs, build features, train LayoutGAT.

    Saves best model checkpoint to output_dir.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Load all graphs from S3
    graphs = _load_graphs_from_s3(graphs_bucket, graphs_prefix)
    if not graphs:
        logger.error("No graphs found, aborting training.")
        return

    # Build corpus stats from already-loaded graphs (no second S3 download)
    logger.info("Building corpus statistics...")
    corpus_stats = build_corpus_stats(graphs)

    # Prepare training data: extract subgraphs, generate reference layouts, build features
    training_data = []
    for graph in graphs:
        sub = extract_stage_subgraphs(graph)
        if len(sub["stages"]) < 2:
            continue  # skip trivial graphs

        ref_layout = generate_reference_layout(sub["stages"], sub["edges"])
        if not ref_layout:
            continue

        node_features, edge_index, edge_attr, category_ranks = build_layout_features(
            sub["stages"], sub["edges"], sub["back_edges"], corpus_stats
        )

        # Build reference coords tensor (N x 2)
        ref_coords = torch.zeros(len(sub["stages"]), 2, dtype=torch.float32)
        for i, sid in enumerate(sub["stages"]):
            if sid in ref_layout:
                ref_coords[i, 0] = ref_layout[sid][0]
                ref_coords[i, 1] = ref_layout[sid][1]

        # Build edge_types tensor
        edge_types = torch.zeros(edge_index.size(1), dtype=torch.float32)
        for ei in range(edge_index.size(1)):
            edge_types[ei] = edge_attr[ei, 1]  # direction encoding from edge_attr

        # Build topo_depths tensor
        topo_depths = node_features[:, 12]  # offset 10 (category) + 2 (topo_depth)

        if edge_index.size(1) == 0:
            continue  # skip graphs with no edges

        training_data.append({
            "node_features": node_features.to(device),
            "edge_index": edge_index.to(device),
            "edge_attr": edge_attr.to(device),
            "ref_coords": ref_coords.to(device),
            "edge_types": edge_types.to(device),
            "topo_depths": topo_depths.to(device),
            "category_ranks": category_ranks.to(device),
        })

    logger.info(f"Prepared {len(training_data)} training graphs")
    if not training_data:
        logger.error("No valid training graphs, aborting.")
        return

    # Initialize model
    model = LayoutGAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_components = defaultdict(float)

        for data in training_data:
            optimizer.zero_grad()

            pred = model(data["node_features"], data["edge_index"], data["edge_attr"],
                         topo_depths=data["topo_depths"])
            loss, loss_dict = compute_layout_loss(
                pred, data["ref_coords"], data["edge_index"],
                data["edge_types"], data["topo_depths"],
                category_ranks=data["category_ranks"]
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_dict["total"]
            for k, v in loss_dict.items():
                epoch_components[k] += v

        avg_loss = epoch_loss / len(training_data)

        if (epoch + 1) % 10 == 0:
            comp_str = " | ".join(
                f"{k}={v / len(training_data):.4f}" for k, v in sorted(epoch_components.items())
            )
            logger.info(f"Epoch {epoch + 1}/{epochs} — avg_loss={avg_loss:.4f} — {comp_str}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best model
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "layout_gat.pt")
    torch.save({
        "model_state_dict": best_state,
        "corpus_stats": corpus_stats,
        "best_loss": best_loss,
    }, save_path)
    logger.info(f"Saved best model (loss={best_loss:.4f}) to {save_path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def layout_flowsheet(model_path, stages: list, connections: list,
                     corpus_stats: dict = None):
    """Run inference to produce (x, y) layout coordinates for a flowsheet.

    Args:
        model_path: path to saved layout_gat.pt checkpoint (or None for default)
        stages: list of stage_id strings
        connections: list of dicts with "from"/"to" keys, or (src, dst, weight) tuples
        corpus_stats: optional pre-loaded corpus stats; loaded from checkpoint if None

    Returns:
        dict[stage_id] -> {"x": float, "y": float}
    """
    # Resolve default checkpoint path
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "layout_checkpoint.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Layout checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if corpus_stats is None:
        corpus_stats = checkpoint.get("corpus_stats", {})

    model = LayoutGAT().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Normalize connections to (src, dst, weight) tuples
    norm_conns = []
    for c in connections:
        if isinstance(c, dict):
            src = c.get("from") or c.get("parent_id", "")
            dst = c.get("to") or c.get("child_id", "")
            w = c.get("corpus_count", c.get("confidence", 1))
            norm_conns.append((src, dst, w))
        else:
            norm_conns.append(c)

    # Build features
    sub = extract_stage_subgraphs({
        "nodes": [{"id": f"stg_{s}", "type": "stage", "stage_id": s} for s in stages],
        "edges": [{"source": f"stg_{src}", "target": f"stg_{dst}",
                    "type": "stage_transition", "weight": w}
                   for src, dst, w in norm_conns],
    })

    node_features, edge_index, edge_attr, _category_ranks = build_layout_features(
        sub["stages"], sub["edges"], sub["back_edges"], corpus_stats
    )

    # Build topo_depths for constraint projection
    topo_depths = node_features[:, 12]  # offset 10 (category) + 2 (topo_depth)

    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    topo_depths = topo_depths.to(device)

    with torch.no_grad():
        coords = model(node_features, edge_index, edge_attr,
                        topo_depths=topo_depths).cpu().numpy()

    result = {}
    for i, sid in enumerate(sub["stages"]):
        result[sid] = {"x": float(coords[i, 0]), "y": float(coords[i, 1])}

    return result


# ---------------------------------------------------------------------------
# SageMaker entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs-bucket", type=str, required=True)
    parser.add_argument("--graphs-prefix", type=str, default="graphs/")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()
    train_layout(args.graphs_bucket, args.graphs_prefix, args.epochs, args.lr, args.output_dir)
