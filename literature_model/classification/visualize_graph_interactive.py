"""Generate interactive HTML graph visualizations using D3.js."""
import json
import sys
from pathlib import Path

GRAPHS_DIR = Path(__file__).parent / "graph_output" / "graphs"
OUT_DIR = Path(__file__).parent / "graph_output"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Document Graph: {doc_name}</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; overflow: hidden; }
  svg { display: block; }
  .tooltip { position: absolute; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; font-size: 12px; pointer-events: none; opacity: 0; transition: opacity 0.15s; max-width: 350px; z-index: 10; }
  .tooltip .label { font-weight: 600; color: #f0f6fc; margin-bottom: 4px; }
  .tooltip .detail { color: #8b949e; }
  #controls { position: absolute; top: 12px; left: 12px; display: flex; gap: 8px; align-items: center; }
  #controls button { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; }
  #controls button:hover { background: #30363d; }
  #controls button.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  #info { position: absolute; top: 12px; right: 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; font-size: 12px; min-width: 180px; }
  #info h3 { color: #f0f6fc; margin-bottom: 6px; font-size: 13px; }
  #info .stat { display: flex; justify-content: space-between; margin: 2px 0; }
  #info .stat .val { color: #58a6ff; font-weight: 600; }
  .legend { position: absolute; bottom: 12px; left: 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; font-size: 12px; }
  .legend-item { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
  .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
  .legend-line { width: 20px; height: 0; border-top: 2px solid; }
</style>
</head>
<body>
<div id="controls">
  <button id="btn-all" class="active" onclick="filterEdges('all')">All Edges</button>
  <button id="btn-stage" onclick="filterEdges('stage_transition')">Stage Transitions</button>
  <button id="btn-cooc" onclick="filterEdges('section_cooccurrence')">Co-occurrence</button>
  <button id="btn-ctx" onclick="filterEdges('context_influences_stage')">Ctx→Stage</button>
  <button id="btn-reset" onclick="resetZoom()">Reset Zoom</button>
</div>
<div id="info">
  <h3>{doc_name_short}</h3>
  <div class="stat"><span>Nodes</span><span class="val">{n_nodes}</span></div>
  <div class="stat"><span>Context</span><span class="val">{n_context}</span></div>
  <div class="stat"><span>Stages</span><span class="val">{n_stages}</span></div>
  <div class="stat"><span>Edges</span><span class="val">{n_edges}</span></div>
  <div class="stat"><span>Transitions</span><span class="val">{n_transitions}</span></div>
  <div class="stat"><span>Co-occurrence</span><span class="val">{n_cooc}</span></div>
  <div class="stat"><span>Ctx→Stage</span><span class="val">{n_ctx_stage}</span></div>
</div>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#58a6ff"></div> Context section</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f85149"></div> Process stage</div>
  <div class="legend-item"><div class="legend-line" style="border-color:#f85149"></div> Stage transition (width = mention weight)</div>
  <div class="legend-item"><div class="legend-line" style="border-color:#8b949e"></div> Co-occurrence (width = similarity)</div>
  <div class="legend-item"><div class="legend-line" style="border-color:#3fb950"></div> Context → Stage (width = relevance)</div>
</div>
<div class="tooltip" id="tooltip"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const graphData = {graph_json};

const width = window.innerWidth;
const height = window.innerHeight;

const svg = d3.select("body").append("svg")
  .attr("width", width).attr("height", height);

const defs = svg.append("defs");
defs.append("marker").attr("id", "arrow-stage").attr("viewBox", "0 -5 10 10")
  .attr("refX", 22).attr("refY", 0).attr("markerWidth", 6).attr("markerHeight", 6)
  .attr("orient", "auto")
  .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#f85149");
defs.append("marker").attr("id", "arrow-ctx").attr("viewBox", "0 -5 10 10")
  .attr("refX", 22).attr("refY", 0).attr("markerWidth", 5).attr("markerHeight", 5)
  .attr("orient", "auto")
  .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#3fb950");

const edgeColor = d => d.type === "stage_transition" ? "#f85149" : d.type === "context_influences_stage" ? "#3fb950" : "#8b949e";
const maxTransW = graphData.edges.filter(e => e.type === "stage_transition").reduce((max, e) => Math.max(max, e.weight), 1);
const edgeWidth = d => {
  if (d.type === "stage_transition") return 1 + (d.weight / maxTransW) * 4; // 1-5px scaled by mention weight
  if (d.type === "context_influences_stage") return 1 + d.weight * 2;
  return 0.5 + d.weight * 2.5; // cooccurrence: 0.5-3.0 based on similarity
};
const edgeOpacity = d => {
  if (d.type === "stage_transition") return 0.4 + (d.weight / maxTransW) * 0.55; // 0.4-0.95 by weight
  if (d.type === "context_influences_stage") return 0.4 + d.weight * 0.5;
  return 0.15 + d.weight * 0.5; // cooccurrence: faint when low similarity
};
const edgeMarker = d => d.type === "stage_transition" ? "url(#arrow-stage)" : d.type === "context_influences_stage" ? "url(#arrow-ctx)" : "";

const g = svg.append("g");

const zoom = d3.zoom().scaleExtent([0.2, 5]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

const simulation = d3.forceSimulation(graphData.nodes)
  .force("link", d3.forceLink(graphData.edges).id(d => d.id).distance(d => d.type === "stage_transition" ? 120 : 180))
  .force("charge", d3.forceManyBody().strength(-400))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(30));

const link = g.append("g").selectAll("line")
  .data(graphData.edges).join("line")
  .attr("stroke", edgeColor)
  .attr("stroke-width", edgeWidth)
  .attr("stroke-opacity", edgeOpacity)
  .attr("marker-end", edgeMarker)
  .attr("class", d => "edge edge-" + d.type);

const node = g.append("g").selectAll("g")
  .data(graphData.nodes).join("g")
  .call(d3.drag()
    .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end", (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

node.append("circle")
  .attr("r", d => d.type === "stage" ? 14 : 10)
  .attr("fill", d => d.type === "stage" ? "#f85149" : "#58a6ff")
  .attr("stroke", "#0d1117").attr("stroke-width", 2)
  .style("cursor", "pointer");

node.append("text")
  .text(d => d.label)
  .attr("dx", 18).attr("dy", 4)
  .attr("font-size", "11px").attr("fill", "#c9d1d9")
  .attr("font-weight", d => d.type === "stage" ? "600" : "400");

const tooltip = d3.select("#tooltip");

node.on("mouseover", function(e, d) {
    d3.select(this).select("circle").attr("stroke", "#f0f6fc").attr("stroke-width", 3);
    // Highlight connected edges
    link.attr("stroke-opacity", l => (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.03);
    // Build tooltip
    let html = '<div class="label">' + d.label + '</div>';
    html += '<div class="detail">Type: ' + d.type + '</div>';
    if (d.group) html += '<div class="detail">Group: ' + d.group + '</div>';
    const conns = graphData.edges.filter(l => l.source.id === d.id || l.target.id === d.id);
    html += '<div class="detail">Connections: ' + conns.length + '</div>';
    const byType = {};
    conns.forEach(c => { byType[c.type] = (byType[c.type] || 0) + 1; });
    Object.entries(byType).forEach(([t, n]) => {
      html += '<div class="detail" style="margin-left:8px">' + t.replace(/_/g,' ') + ': ' + n + '</div>';
    });
    const weighted = conns.filter(c => c.weight > 0);
    if (weighted.length) {
      const ws = weighted.map(c => c.weight);
      html += '<div class="detail">Weights: ' + Math.min(...ws).toFixed(2) + ' – ' + Math.max(...ws).toFixed(2) + '</div>';
    }
    if (d.features) {
      const f = d.features;
      if (f.char_count) html += '<div class="detail">Text: ' + f.char_count.toLocaleString() + ' chars</div>';
      if (f.num_tables) html += '<div class="detail">Tables: ' + f.num_tables + '</div>';
      if (f.num_pages) html += '<div class="detail">Pages: ' + f.num_pages + '</div>';
      // Show top keywords
      const kws = Object.entries(f).filter(([k,v]) => k.startsWith("kw_") && v > 0).sort((a,b) => b[1]-a[1]).slice(0, 5);
      if (kws.length) html += '<div class="detail" style="margin-top:4px">Top keywords: ' + kws.map(([k,v]) => k.replace("kw_","") + " (" + v + ")").join(", ") + '</div>';
    }
    tooltip.html(html).style("opacity", 1).style("left", (e.pageX + 15) + "px").style("top", (e.pageY - 10) + "px");
  })
  .on("mousemove", function(e) { tooltip.style("left", (e.pageX + 15) + "px").style("top", (e.pageY - 10) + "px"); })
  .on("mouseout", function(e, d) {
    d3.select(this).select("circle").attr("stroke", "#0d1117").attr("stroke-width", 2);
    link.attr("stroke-opacity", edgeOpacity);
    tooltip.style("opacity", 0);
  });

simulation.on("tick", () => {
  link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  node.attr("transform", d => "translate(" + d.x + "," + d.y + ")");
});

// Edge filter buttons
window.filterEdges = function(type) {
  document.querySelectorAll("#controls button").forEach(b => b.classList.remove("active"));
  const btnMap = {"all":"btn-all","stage_transition":"btn-stage","section_cooccurrence":"btn-cooc","context_influences_stage":"btn-ctx"};
  document.getElementById(btnMap[type] || "btn-all").classList.add("active");
  if (type === "all") {
    link.style("display", null);
  } else {
    link.style("display", d => d.type === type ? null : "none");
  }
};

window.resetZoom = function() {
  svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
};
</script>
</body>
</html>"""


def build_interactive(graph_path: Path, out_path: Path):
    """Build an interactive HTML visualization for a single document graph."""
    with open(graph_path, encoding="utf-8") as f:
        data = json.load(f)

    # Prepare D3-friendly JSON (strip tfidf to keep file small)
    nodes = []
    for n in data["nodes"]:
        node = {"id": n["id"], "type": n["type"], "group": n.get("group", ""),
                "label": n["id"].replace("ctx_", "").replace("stage_", "")}
        if "features" in n:
            feats = {k: v for k, v in n["features"].items() if k != "tfidf"}
            node["features"] = feats
        nodes.append(node)

    node_ids = {n["id"] for n in nodes}
    edges = [{"source": e["source"], "target": e["target"],
              "type": e["type"], "weight": e.get("weight", 1.0)}
             for e in data["edges"]
             if e["source"] in node_ids and e["target"] in node_ids]

    graph_json = json.dumps({"nodes": nodes, "edges": edges})

    n_context = sum(1 for n in nodes if n["type"] == "context")
    n_stages = sum(1 for n in nodes if n["type"] == "stage")
    n_transitions = sum(1 for e in edges if e["type"] == "stage_transition")
    n_cooc = sum(1 for e in edges if e["type"] == "section_cooccurrence")
    n_ctx_stage = sum(1 for e in edges if e["type"] == "context_influences_stage")

    doc_name = graph_path.stem
    html = (HTML_TEMPLATE
        .replace("{doc_name}", doc_name)
        .replace("{doc_name_short}", doc_name[:20] + "…")
        .replace("{graph_json}", graph_json)
        .replace("{n_nodes}", str(len(nodes)))
        .replace("{n_context}", str(n_context))
        .replace("{n_stages}", str(n_stages))
        .replace("{n_edges}", str(len(edges)))
        .replace("{n_transitions}", str(n_transitions))
        .replace("{n_cooc}", str(n_cooc))
        .replace("{n_ctx_stage}", str(n_ctx_stage))
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if target == "auto":
        # Find a graph with the most stages
        best, best_count = None, 0
        for fp in sorted(GRAPHS_DIR.glob("*.json")):
            if fp.stem in ("graphs_summary", "stage_vocab", "checkpoint"):
                continue
            with open(fp) as f:
                d = json.load(f)
            stages = sum(1 for n in d["nodes"] if n["type"] == "stage")
            if stages > best_count:
                best, best_count = fp, stages
        if best:
            print(f"Selected doc with {best_count} stages: {best.stem[:20]}…")
            build_interactive(best, OUT_DIR / "interactive_graph.html")
    else:
        p = Path(target)
        if not p.exists():
            p = GRAPHS_DIR / target
        build_interactive(p, OUT_DIR / "interactive_graph.html")
