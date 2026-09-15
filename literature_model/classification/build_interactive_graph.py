"""Build interactive HTML visualization of the cross-document similarity graph."""

import boto3
import json
import os
import numpy as np
import networkx as nx
from collections import Counter

from config import SAGEMAKER_BUCKET

GRAPHS_BUCKET = SAGEMAKER_BUCKET
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "graph_output", "cross_doc_interactive.html")


def main():
    s3 = boto3.client("s3")

    # Load cross-doc edges
    resp = s3.get_object(Bucket=GRAPHS_BUCKET, Key="graphs/cross_doc_edges.json")
    edges = json.loads(resp["Body"].read())

    # Load individual graphs for stage/section info
    print("Loading graphs...")
    paginator = s3.get_paginator("list_objects_v2")
    graphs = {}
    skip_suffixes = ("stage_vocab.json", "cross_doc_edges.json", "graphs_summary.json",
                     "checkpoint.json", "skip_list.json", "attempt_marker.json")
    for page in paginator.paginate(Bucket=GRAPHS_BUCKET, Prefix="graphs/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json") or key.endswith(skip_suffixes):
                continue
            try:
                resp2 = s3.get_object(Bucket=GRAPHS_BUCKET, Key=key)
                g = json.loads(resp2["Body"].read())
                if isinstance(g, dict) and g.get("document_id"):
                    graphs[g["document_id"]] = g
            except Exception:
                pass
    print(f"Loaded {len(graphs)} graphs")

    # Build networkx graph
    G = nx.Graph()
    for doc_id, data in edges.items():
        G.add_node(doc_id)
        for nb in data["neighbors"]:
            G.add_edge(doc_id, nb["doc_id"], weight=nb["overall_similarity"],
                       strategy=nb["strategy_similarity"])

    # Communities
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G, weight="weight", seed=42)
    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

    # Community top stages
    comm_top_stages = {}
    for i, comm in enumerate(communities):
        counter = Counter()
        for doc_id in comm:
            g = graphs.get(doc_id, {})
            for n in g.get("nodes", []):
                if n.get("type") == "stage" and n.get("id"):
                    counter[n["id"]] += 1
        top3 = [s[0].replace("stg_", "").replace("_", " ").title()
                for s in counter.most_common(3)]
        comm_top_stages[i] = ", ".join(top3) if top3 else "No stages"

    # Layout
    print("Computing layout...")
    pos = nx.spring_layout(G, k=0.3, iterations=80, weight="weight", seed=42)

    # Node info
    doc_info = {}
    for doc_id in G.nodes():
        g = graphs.get(doc_id, {})
        stages = []
        sections = []
        for n in g.get("nodes", []):
            if n.get("type") == "stage" and n.get("id"):
                stages.append(n["id"].replace("stg_", "").replace("_", " ").title())
            elif n.get("type") == "context" and n.get("group"):
                sections.append(n["group"].replace("_", " ").title())
        doc_info[doc_id] = {"stages": stages, "sections": sections}

    colors_hex = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                  "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]

    comm_sizes = sorted([(i, len(c)) for i, c in enumerate(communities)], key=lambda x: -x[1])

    # Build nodes JSON
    nodes_json = []
    for n in G.nodes():
        ci = node_community[n]
        info = doc_info[n]
        deg = G.degree(n)
        short_id = n[:50] + "..." if len(n) > 50 else n
        stages_str = ", ".join(info["stages"][:8]) or "None"
        sections_str = ", ".join(info["sections"][:10]) or "None"
        nb_data = edges.get(n, {}).get("neighbors", [])
        mean_sim = float(np.mean([nb["overall_similarity"] for nb in nb_data])) if nb_data else 0

        nodes_json.append({
            "id": n,
            "x": float(pos[n][0]),
            "y": float(pos[n][1]),
            "community": ci,
            "color": colors_hex[ci % len(colors_hex)],
            "degree": deg,
            "stages": stages_str,
            "sections": sections_str,
            "n_stages": len(info["stages"]),
            "mean_sim": round(mean_sim, 3),
            "short_id": short_id,
        })

    # Build edges JSON
    edges_json = []
    for u, v, d in G.edges(data=True):
        edges_json.append({
            "source": u,
            "target": v,
            "weight": round(d["weight"], 3),
            "strategy": round(d["strategy"], 3),
        })

    # Legend data
    legend_json = []
    for ci, size in comm_sizes:
        legend_json.append({
            "id": ci,
            "color": colors_hex[ci % len(colors_hex)],
            "label": f"C{ci} ({size} docs): {comm_top_stages[ci]}",
            "size": size,
        })

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_comms = len(communities)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cross-Document Similarity Graph</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a1a; color: #eee; font-family: 'Segoe UI', sans-serif; overflow: hidden; }}
  #container {{ display: flex; height: 100vh; }}
  canvas {{ flex: 1; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  #sidebar {{ width: 320px; background: #12122a; padding: 16px; overflow-y: auto; border-left: 1px solid #333; }}
  #sidebar h2 {{ font-size: 14px; color: #aaa; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}
  #sidebar h3 {{ font-size: 13px; color: #ccc; margin: 12px 0 4px; }}
  .legend-item {{ display: flex; align-items: center; padding: 3px 0; cursor: pointer; font-size: 11px; }}
  .legend-item:hover {{ background: #1a1a3a; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }}
  #tooltip {{ position: absolute; background: rgba(20,20,40,0.95); border: 1px solid #555; border-radius: 6px;
             padding: 10px 14px; font-size: 12px; pointer-events: none; display: none; max-width: 400px;
             line-height: 1.5; box-shadow: 0 4px 20px rgba(0,0,0,0.5); z-index: 100; }}
  #tooltip .label {{ color: #fff; font-weight: bold; font-size: 13px; margin-bottom: 4px; word-break: break-all; }}
  #tooltip .row {{ color: #bbb; }}
  #tooltip .row span {{ color: #fff; }}
  #stats {{ font-size: 12px; color: #888; margin-bottom: 12px; line-height: 1.6; }}
  #stats span {{ color: #ddd; }}
  #search {{ width: 100%; padding: 6px 10px; background: #1a1a3a; border: 1px solid #444; color: #eee;
            border-radius: 4px; font-size: 12px; margin-bottom: 12px; }}
  #search::placeholder {{ color: #666; }}
  #controls {{ font-size: 11px; color: #888; margin-bottom: 12px; line-height: 1.8; }}
  .highlight-info {{ background: #1a2a1a; border: 1px solid #3a5a3a; border-radius: 4px; padding: 8px;
                    margin-top: 8px; font-size: 11px; display: none; }}
</style>
</head>
<body>
<div id="container">
  <canvas id="graph"></canvas>
  <div id="sidebar">
    <h2>Cross-Document Similarity</h2>
    <div id="stats">
      <span>{n_nodes}</span> documents &middot;
      <span>{n_edges}</span> edges &middot;
      <span>{n_comms}</span> communities
    </div>
    <input type="text" id="search" placeholder="Search documents or stages..." />
    <div id="controls">
      Scroll to zoom &middot; Drag to pan &middot; Click node to highlight<br>
      Double-click to reset view &middot; Click legend to toggle
    </div>
    <h3>Communities</h3>
    <div id="legend"></div>
    <div id="highlight-info" class="highlight-info"></div>
  </div>
</div>
<div id="tooltip"></div>

<script>
const NODES = {json.dumps(nodes_json)};
const EDGES = {json.dumps(edges_json)};
const LEGEND = {json.dumps(legend_json)};

const canvas = document.getElementById("graph");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const legendDiv = document.getElementById("legend");
const searchInput = document.getElementById("search");
const highlightInfo = document.getElementById("highlight-info");

let W, H;
let offsetX = 0, offsetY = 0, scale = 1;
let dragging = false, dragStartX, dragStartY, dragOffsetX, dragOffsetY;
let hoveredNode = null, selectedNode = null;
let hiddenCommunities = new Set();

const nodeMap = {{}};
NODES.forEach(n => nodeMap[n.id] = n);
const adjList = {{}};
NODES.forEach(n => adjList[n.id] = []);
EDGES.forEach(e => {{
  adjList[e.source].push({{target: e.target, weight: e.weight, strategy: e.strategy}});
  adjList[e.target].push({{target: e.source, weight: e.weight, strategy: e.strategy}});
}});

function resize() {{
  W = canvas.parentElement.clientWidth - 320;
  H = canvas.parentElement.clientHeight;
  canvas.width = W * devicePixelRatio;
  canvas.height = H * devicePixelRatio;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}}

function worldToScreen(wx, wy) {{
  return [(wx * scale * Math.min(W, H) * 0.42) + W/2 + offsetX,
          (-wy * scale * Math.min(W, H) * 0.42) + H/2 + offsetY];
}}

function nodeRadius(n) {{
  return Math.max(2, 1.5 + n.degree * 0.12) * Math.min(scale, 3);
}}

function draw() {{
  ctx.clearRect(0, 0, W, H);
  const neighbors = selectedNode ? new Set(adjList[selectedNode].map(e => e.target)) : null;
  const searchTerm = searchInput.value.toLowerCase();
  const searchMatches = searchTerm ? new Set(NODES.filter(n =>
    n.id.toLowerCase().includes(searchTerm) || n.stages.toLowerCase().includes(searchTerm)
  ).map(n => n.id)) : null;

  // Edges
  for (const e of EDGES) {{
    const sn = nodeMap[e.source], tn = nodeMap[e.target];
    if (hiddenCommunities.has(sn.community) || hiddenCommunities.has(tn.community)) continue;
    let alpha = 0.03;
    let lw = 0.5;
    let color = "#555";
    if (selectedNode) {{
      if (e.source === selectedNode || e.target === selectedNode) {{
        alpha = 0.2 + e.weight * 0.6;
        lw = 1 + e.weight * 2;
        color = nodeMap[e.source === selectedNode ? e.target : e.source].color;
      }} else {{ alpha = 0.005; lw = 0.3; }}
    }} else if (searchMatches) {{
      alpha = (searchMatches.has(e.source) || searchMatches.has(e.target)) ? 0.15 : 0.005;
    }}
    const [x1, y1] = worldToScreen(sn.x, sn.y);
    const [x2, y2] = worldToScreen(tn.x, tn.y);
    if (x1 < -50 && x2 < -50 || x1 > W+50 && x2 > W+50 ||
        y1 < -50 && y2 < -50 || y1 > H+50 && y2 > H+50) continue;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }}

  // Nodes
  for (const n of NODES) {{
    if (hiddenCommunities.has(n.community)) continue;
    const [sx, sy] = worldToScreen(n.x, n.y);
    if (sx < -20 || sx > W+20 || sy < -20 || sy > H+20) continue;
    let alpha = 0.85, r = nodeRadius(n);
    if (selectedNode) {{
      if (n.id === selectedNode) {{ alpha = 1; r *= 1.8; }}
      else if (neighbors.has(n.id)) {{ alpha = 0.9; r *= 1.2; }}
      else alpha = 0.08;
    }} else if (searchMatches) {{
      alpha = searchMatches.has(n.id) ? 1 : 0.08;
      if (searchMatches.has(n.id)) r *= 1.5;
    }}
    if (n.id === hoveredNode) {{ alpha = 1; r *= 1.4; }}
    ctx.globalAlpha = alpha;
    ctx.fillStyle = n.color;
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = (n.id === selectedNode || n.id === hoveredNode) ? "#fff" : "#000";
    ctx.lineWidth = n.id === selectedNode ? 2 : 0.5;
    ctx.stroke();
  }}
  ctx.globalAlpha = 1;
  requestAnimationFrame(draw);
}}

// Legend
LEGEND.forEach(item => {{
  const div = document.createElement("div");
  div.className = "legend-item";
  div.innerHTML = '<div class="legend-dot" style="background:' + item.color + '"></div>' + item.label;
  div.onclick = () => {{
    if (hiddenCommunities.has(item.id)) hiddenCommunities.delete(item.id);
    else hiddenCommunities.add(item.id);
    div.style.opacity = hiddenCommunities.has(item.id) ? 0.3 : 1;
  }};
  legendDiv.appendChild(div);
}});

// Mouse interaction
function getMouseNode(mx, my) {{
  let best = null, bestDist = Infinity;
  for (const n of NODES) {{
    if (hiddenCommunities.has(n.community)) continue;
    const [sx, sy] = worldToScreen(n.x, n.y);
    const d = Math.hypot(sx - mx, sy - my);
    const r = nodeRadius(n) + 4;
    if (d < r && d < bestDist) {{ best = n; bestDist = d; }}
  }}
  return best;
}}

canvas.addEventListener("mousemove", e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (dragging) {{
    offsetX = dragOffsetX + (e.clientX - dragStartX);
    offsetY = dragOffsetY + (e.clientY - dragStartY);
    return;
  }}
  const n = getMouseNode(mx, my);
  hoveredNode = n ? n.id : null;
  canvas.style.cursor = n ? "pointer" : "grab";
  if (n) {{
    const nbs = adjList[n.id].sort((a,b) => b.weight - a.weight).slice(0, 5);
    const topNbs = nbs.map(e =>
      nodeMap[e.target].short_id + " (" + e.weight + ")").join("<br>");
    tooltip.innerHTML =
      '<div class="label">' + n.id + '</div>' +
      '<div class="row">Community: <span>' + n.community + '</span></div>' +
      '<div class="row">Degree: <span>' + n.degree + '</span> &middot; Mean sim: <span>' + n.mean_sim + '</span></div>' +
      '<div class="row">Stages (' + n.n_stages + '): <span>' + n.stages + '</span></div>' +
      '<div class="row">Sections: <span>' + n.sections + '</span></div>' +
      (topNbs ? '<div class="row" style="margin-top:6px">Top neighbors:<br><span style="font-size:10px">' + topNbs + '</span></div>' : '');
    tooltip.style.display = "block";
    tooltip.style.left = Math.min(e.clientX + 15, window.innerWidth - 420) + "px";
    tooltip.style.top = Math.min(e.clientY + 15, window.innerHeight - 250) + "px";
  }} else {{ tooltip.style.display = "none"; }}
}});

canvas.addEventListener("mousedown", e => {{
  dragging = true;
  dragStartX = e.clientX; dragStartY = e.clientY;
  dragOffsetX = offsetX; dragOffsetY = offsetY;
}});

canvas.addEventListener("mouseup", e => {{
  const moved = Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY);
  dragging = false;
  if (moved < 5) {{
    const rect = canvas.getBoundingClientRect();
    const n = getMouseNode(e.clientX - rect.left, e.clientY - rect.top);
    if (n) {{
      selectedNode = selectedNode === n.id ? null : n.id;
      if (selectedNode) {{
        const nbs = adjList[selectedNode];
        highlightInfo.style.display = "block";
        highlightInfo.innerHTML = "<b>" + n.short_id + "</b><br>" +
          nbs.length + " neighbors &middot; mean sim " + n.mean_sim + "<br>" +
          "Stages: " + n.stages;
      }} else {{ highlightInfo.style.display = "none"; }}
    }} else {{ selectedNode = null; highlightInfo.style.display = "none"; }}
  }}
}});

canvas.addEventListener("dblclick", () => {{
  offsetX = 0; offsetY = 0; scale = 1;
  selectedNode = null; highlightInfo.style.display = "none";
}});

canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const factor = e.deltaY > 0 ? 0.9 : 1.1;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  offsetX = mx - factor * (mx - offsetX);
  offsetY = my - factor * (my - offsetY);
  scale *= factor;
}}, {{passive: false}});

canvas.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});
searchInput.addEventListener("input", () => {{ selectedNode = null; highlightInfo.style.display = "none"; }});

window.addEventListener("resize", resize);
resize();
draw();
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
