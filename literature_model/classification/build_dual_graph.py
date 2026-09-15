"""Build interactive dual-panel HTML: cross-doc graph + intra-doc detail view."""

import boto3
import json
import os
import numpy as np
import networkx as nx
from collections import Counter

from config import SAGEMAKER_BUCKET

GRAPHS_BUCKET = SAGEMAKER_BUCKET
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "graph_output", "dual_graph.html")


def main():
    s3 = boto3.client("s3")

    # Load cross-doc edges
    resp = s3.get_object(Bucket=GRAPHS_BUCKET, Key="graphs/cross_doc_edges.json")
    cross_edges = json.loads(resp["Body"].read())

    # Load all individual graphs
    print("Loading graphs...")
    paginator = s3.get_paginator("list_objects_v2")
    graphs = {}
    skip = ("stage_vocab.json", "cross_doc_edges.json", "graphs_summary.json",
            "checkpoint.json", "skip_list.json", "attempt_marker.json")
    for page in paginator.paginate(Bucket=GRAPHS_BUCKET, Prefix="graphs/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json") or key.endswith(skip):
                continue
            try:
                r = s3.get_object(Bucket=GRAPHS_BUCKET, Key=key)
                g = json.loads(r["Body"].read())
                if isinstance(g, dict) and g.get("document_id"):
                    graphs[g["document_id"]] = g
            except Exception:
                pass
    print(f"Loaded {len(graphs)} graphs")

    # Build cross-doc networkx graph
    G = nx.Graph()
    for doc_id, data in cross_edges.items():
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

    colors_hex = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                  "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]

    section_colors = {
        "property": "#636EFA", "history": "#EF553B", "climate": "#00CC96",
        "geology": "#AB63FA", "exploration": "#FFA15A", "drilling": "#19D3F3",
        "sample_analysis": "#FF6692", "data_verification": "#B6E880",
        "metallurgical_testing": "#FF97FF", "resource_estimate": "#FECB52",
        "mining_method": "#1f77b4", "recovery_methods": "#ff7f0e",
        "infrastructure": "#2ca02c", "economics": "#d62728",
        "environmental": "#9467bd", "summary": "#8c564b",
        "interpretation": "#e377c2", "market": "#bcbd22", "references": "#17becf",
    }

    # Build cross-doc nodes
    cross_nodes = []
    for n in G.nodes():
        ci = node_community[n]
        g = graphs.get(n, {})
        stages = [nd["id"].replace("stg_", "").replace("_", " ").title()
                  for nd in g.get("nodes", []) if nd.get("type") == "stage" and nd.get("id")]
        sections = [nd["group"].replace("_", " ").title()
                    for nd in g.get("nodes", []) if nd.get("type") == "context" and nd.get("group")]
        nb_data = cross_edges.get(n, {}).get("neighbors", [])
        mean_sim = float(np.mean([nb["overall_similarity"] for nb in nb_data])) if nb_data else 0

        cross_nodes.append({
            "id": n,
            "x": float(pos[n][0]),
            "y": float(pos[n][1]),
            "community": ci,
            "color": colors_hex[ci % len(colors_hex)],
            "degree": G.degree(n),
            "stages": ", ".join(stages[:8]) or "None",
            "sections": ", ".join(sections[:10]) or "None",
            "n_stages": len(stages),
            "n_sections": len(sections),
            "mean_sim": round(mean_sim, 3),
        })

    cross_edges_json = []
    for u, v, d in G.edges(data=True):
        cross_edges_json.append({
            "source": u, "target": v,
            "weight": round(d["weight"], 3),
            "strategy": round(d["strategy"], 3),
        })

    # Build per-document internal graphs (compact: only topology, not features)
    print("Building internal graphs...")
    internal_graphs = {}
    for doc_id, g in graphs.items():
        nodes = []
        for n in g.get("nodes", []):
            ntype = n.get("type", "")
            nid = n.get("id", "")
            if ntype == "context":
                group = n.get("group", "unknown")
                nodes.append({
                    "id": nid, "type": "context", "group": group,
                    "color": section_colors.get(group, "#888"),
                    "label": group.replace("_", " ").title(),
                })
            elif ntype == "stage":
                sid = n.get("stage_id", nid)
                nodes.append({
                    "id": nid, "type": "stage",
                    "color": "#FFD700",
                    "label": sid.replace("stg_", "").replace("_", " ").title()
                            if sid.startswith("stg_") else nid.replace("stg_", "").replace("_", " ").title(),
                })
        edges_int = []
        for e in g.get("edges", []):
            edges_int.append({
                "source": e.get("source", ""),
                "target": e.get("target", ""),
                "type": e.get("type", ""),
                "weight": round(float(e.get("weight", 0)), 3),
            })

        # Cross-doc neighbors for this doc
        nbs = []
        for nb in cross_edges.get(doc_id, {}).get("neighbors", []):
            nbs.append({
                "doc_id": nb["doc_id"],
                "overall": round(nb["overall_similarity"], 3),
                "strategy": round(nb["strategy_similarity"], 3),
                "sections": nb.get("section_similarities", {}),
                "community": node_community.get(nb["doc_id"], -1),
                "color": colors_hex[node_community.get(nb["doc_id"], 0) % len(colors_hex)],
            })

        internal_graphs[doc_id] = {"nodes": nodes, "edges": edges_int, "neighbors": nbs}

    # Legend
    comm_sizes = sorted([(i, len(c)) for i, c in enumerate(communities)], key=lambda x: -x[1])
    legend_json = []
    for ci, size in comm_sizes:
        legend_json.append({
            "id": ci,
            "color": colors_hex[ci % len(colors_hex)],
            "label": f"C{ci} ({size}): {comm_top_stages[ci]}",
        })

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_comms = len(communities)

    section_colors_json = json.dumps(section_colors)

    print("Writing HTML...")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Document Graph — Inter + Intra</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a1a; color:#eee; font-family:'Segoe UI',sans-serif; overflow:hidden; }}
#main {{ display:flex; height:100vh; }}

/* Left panel: cross-doc */
#left-panel {{ flex:1; display:flex; flex-direction:column; border-right:1px solid #333; }}
#left-header {{ padding:8px 12px; background:#12122a; font-size:12px; color:#aaa; display:flex; align-items:center; gap:12px; }}
#left-header h2 {{ font-size:13px; color:#ccc; white-space:nowrap; }}
#left-search {{ padding:4px 8px; background:#1a1a3a; border:1px solid #444; color:#eee; border-radius:3px; font-size:11px; width:180px; }}
#left-canvas {{ flex:1; cursor:grab; }}
#left-canvas:active {{ cursor:grabbing; }}

/* Right panel: detail */
#right-panel {{ width:50%; display:flex; flex-direction:column; }}
#right-header {{ padding:8px 12px; background:#12122a; font-size:12px; color:#aaa; min-height:36px; display:flex; align-items:center; gap:12px; }}
#right-header h2 {{ font-size:13px; color:#ccc; }}
#right-body {{ flex:1; display:flex; }}
#right-canvas {{ flex:1; cursor:grab; }}
#right-canvas:active {{ cursor:grabbing; }}
#right-sidebar {{ width:220px; background:#12122a; padding:10px; overflow-y:auto; border-left:1px solid #333; font-size:11px; }}
#right-sidebar h3 {{ font-size:11px; color:#999; margin:8px 0 4px; text-transform:uppercase; letter-spacing:0.5px; }}

/* Shared */
.legend-row {{ display:flex; align-items:center; padding:2px 0; cursor:pointer; }}
.legend-row:hover {{ background:#1a1a3a; }}
.legend-dot {{ width:8px; height:8px; border-radius:50%; margin-right:6px; flex-shrink:0; }}
.nb-row {{ padding:3px 0; cursor:pointer; border-bottom:1px solid #1a1a3a; }}
.nb-row:hover {{ background:#1a2a1a; }}
.nb-doc {{ color:#ccc; font-size:10px; word-break:break-all; }}
.nb-sim {{ color:#888; font-size:10px; }}

#tooltip {{ position:absolute; background:rgba(20,20,40,0.95); border:1px solid #555; border-radius:6px;
           padding:8px 12px; font-size:11px; pointer-events:none; display:none; max-width:380px;
           line-height:1.4; box-shadow:0 4px 20px rgba(0,0,0,0.5); z-index:100; }}
#tooltip .t-label {{ color:#fff; font-weight:bold; font-size:12px; margin-bottom:3px; word-break:break-all; }}
#tooltip .t-row {{ color:#bbb; }}
#tooltip .t-row span {{ color:#fff; }}

#placeholder {{ display:flex; align-items:center; justify-content:center; flex:1; color:#555; font-size:14px; }}

/* Edge type legend colors */
.edge-cooc {{ color:#4477aa; }}
.edge-influence {{ color:#ee6677; }}
.edge-transition {{ color:#228833; }}
.edge-crossdoc {{ color:#ccbb44; }}
</style>
</head>
<body>
<div id="main">
  <div id="left-panel">
    <div id="left-header">
      <h2>Inter-Document Similarity</h2>
      <input type="text" id="left-search" placeholder="Search docs/stages..." />
      <span style="color:#666">{n_nodes} docs &middot; {n_edges} edges &middot; {n_comms} communities</span>
    </div>
    <canvas id="left-canvas"></canvas>
  </div>
  <div id="right-panel">
    <div id="right-header">
      <h2 id="right-title">Intra-Document Graph</h2>
      <span id="right-subtitle" style="color:#666">Click a document to explore</span>
    </div>
    <div id="right-body">
      <div id="placeholder">Click a document node to view its internal graph</div>
      <canvas id="right-canvas" style="display:none"></canvas>
      <div id="right-sidebar" style="display:none">
        <h3>Legend</h3>
        <div id="edge-legend">
          <div style="padding:2px 0"><span class="edge-cooc">&#9644;</span> Section Co-occurrence</div>
          <div style="padding:2px 0"><span class="edge-influence">&#9644;</span> Context → Stage</div>
          <div style="padding:2px 0"><span class="edge-transition">&#9644;</span> Stage Transition</div>
          <div style="padding:2px 0"><span class="edge-crossdoc">&#9644;</span> Cross-Doc Neighbor</div>
        </div>
        <h3>Sections</h3>
        <div id="section-legend"></div>
        <h3>Stages</h3>
        <div id="stage-list"></div>
        <h3>Cross-Doc Neighbors</h3>
        <div id="neighbor-list"></div>
      </div>
    </div>
  </div>
</div>
<div id="tooltip"></div>

<script>
const CROSS_NODES = {json.dumps(cross_nodes)};
const CROSS_EDGES = {json.dumps(cross_edges_json)};
const LEGEND = {json.dumps(legend_json)};
const INTERNAL = {json.dumps(internal_graphs)};
const SECTION_COLORS = {section_colors_json};

// ===== LEFT PANEL: Cross-doc graph =====
const lCanvas = document.getElementById("left-canvas");
const lCtx = lCanvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const searchInput = document.getElementById("left-search");

let lW, lH, lOffX=0, lOffY=0, lScale=1;
let lDragging=false, lDragSX, lDragSY, lDragOX, lDragOY;
let lHovered=null, lSelected=null;
let hiddenComm = new Set();

const crossMap = {{}};
CROSS_NODES.forEach(n => crossMap[n.id] = n);
const crossAdj = {{}};
CROSS_NODES.forEach(n => crossAdj[n.id] = []);
CROSS_EDGES.forEach(e => {{
  crossAdj[e.source].push(e);
  crossAdj[e.target].push({{source:e.target, target:e.source, weight:e.weight, strategy:e.strategy}});
}});

function lResize() {{
  lW = lCanvas.parentElement.clientWidth;
  lH = lCanvas.parentElement.clientHeight - 36;
  lCanvas.width = lW * devicePixelRatio;
  lCanvas.height = lH * devicePixelRatio;
  lCanvas.style.width = lW + "px";
  lCanvas.style.height = lH + "px";
  lCtx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}}

function lW2S(wx,wy) {{
  const s = Math.min(lW,lH)*0.42;
  return [wx*lScale*s + lW/2 + lOffX, -wy*lScale*s + lH/2 + lOffY];
}}

function lNodeR(n) {{ return Math.max(2, 1.5+n.degree*0.12)*Math.min(lScale,3); }}

function lDraw() {{
  lCtx.clearRect(0,0,lW,lH);
  const nbs = lSelected ? new Set(crossAdj[lSelected].map(e=>e.target)) : null;
  const st = searchInput.value.toLowerCase();
  const sm = st ? new Set(CROSS_NODES.filter(n=>n.id.toLowerCase().includes(st)||n.stages.toLowerCase().includes(st)).map(n=>n.id)) : null;

  for (const e of CROSS_EDGES) {{
    const sn=crossMap[e.source], tn=crossMap[e.target];
    if (hiddenComm.has(sn.community)||hiddenComm.has(tn.community)) continue;
    let a=0.03, lw=0.5, c="#555";
    if (lSelected) {{
      if (e.source===lSelected||e.target===lSelected) {{a=0.2+e.weight*0.6; lw=1+e.weight*2; c=crossMap[e.source===lSelected?e.target:e.source].color;}}
      else {{a=0.005; lw=0.3;}}
    }} else if (sm) {{ a=(sm.has(e.source)||sm.has(e.target))?0.15:0.005; }}
    const [x1,y1]=lW2S(sn.x,sn.y), [x2,y2]=lW2S(tn.x,tn.y);
    if ((x1<-50&&x2<-50)||(x1>lW+50&&x2>lW+50)||(y1<-50&&y2<-50)||(y1>lH+50&&y2>lH+50)) continue;
    lCtx.globalAlpha=a; lCtx.strokeStyle=c; lCtx.lineWidth=lw;
    lCtx.beginPath(); lCtx.moveTo(x1,y1); lCtx.lineTo(x2,y2); lCtx.stroke();
  }}

  for (const n of CROSS_NODES) {{
    if (hiddenComm.has(n.community)) continue;
    const [sx,sy]=lW2S(n.x,n.y);
    if (sx<-20||sx>lW+20||sy<-20||sy>lH+20) continue;
    let a=0.85, r=lNodeR(n);
    if (lSelected) {{
      if (n.id===lSelected) {{a=1;r*=1.8;}} else if (nbs.has(n.id)) {{a=0.9;r*=1.2;}} else a=0.08;
    }} else if (sm) {{ a=sm.has(n.id)?1:0.08; if(sm.has(n.id))r*=1.5; }}
    if (n.id===lHovered) {{a=1;r*=1.4;}}
    lCtx.globalAlpha=a; lCtx.fillStyle=n.color;
    lCtx.beginPath(); lCtx.arc(sx,sy,r,0,Math.PI*2); lCtx.fill();
    lCtx.strokeStyle=(n.id===lSelected||n.id===lHovered)?"#fff":"#000";
    lCtx.lineWidth=n.id===lSelected?2:0.5; lCtx.stroke();
  }}
  lCtx.globalAlpha=1;
  requestAnimationFrame(lDraw);
}}

function lHitTest(mx,my) {{
  let best=null, bd=Infinity;
  for (const n of CROSS_NODES) {{
    if (hiddenComm.has(n.community)) continue;
    const [sx,sy]=lW2S(n.x,n.y);
    const d=Math.hypot(sx-mx,sy-my);
    if (d<lNodeR(n)+4 && d<bd) {{best=n; bd=d;}}
  }}
  return best;
}}

lCanvas.addEventListener("mousemove", e => {{
  const r=lCanvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  if (lDragging) {{ lOffX=lDragOX+(e.clientX-lDragSX); lOffY=lDragOY+(e.clientY-lDragSY); return; }}
  const n=lHitTest(mx,my);
  lHovered=n?n.id:null;
  lCanvas.style.cursor=n?"pointer":"grab";
  if (n) {{
    tooltip.innerHTML='<div class="t-label">'+n.id+'</div>'+
      '<div class="t-row">Community <span>'+n.community+'</span> &middot; Degree <span>'+n.degree+'</span></div>'+
      '<div class="t-row">Stages ('+n.n_stages+'): <span>'+n.stages+'</span></div>'+
      '<div class="t-row">Sections ('+n.n_sections+'): <span>'+n.sections+'</span></div>'+
      '<div class="t-row">Mean similarity: <span>'+n.mean_sim+'</span></div>';
    tooltip.style.display="block";
    tooltip.style.left=Math.min(e.clientX+12,window.innerWidth-400)+"px";
    tooltip.style.top=Math.min(e.clientY+12,window.innerHeight-180)+"px";
  }} else tooltip.style.display="none";
}});
lCanvas.addEventListener("mousedown", e => {{ lDragging=true; lDragSX=e.clientX; lDragSY=e.clientY; lDragOX=lOffX; lDragOY=lOffY; }});
lCanvas.addEventListener("mouseup", e => {{
  const moved=Math.hypot(e.clientX-lDragSX,e.clientY-lDragSY); lDragging=false;
  if (moved<5) {{
    const r=lCanvas.getBoundingClientRect(), n=lHitTest(e.clientX-r.left,e.clientY-r.top);
    if (n) {{ lSelected=n.id; showInternalGraph(n.id); }}
    else {{ lSelected=null; hideInternalGraph(); }}
  }}
}});
lCanvas.addEventListener("dblclick", () => {{ lOffX=0;lOffY=0;lScale=1;lSelected=null;hideInternalGraph(); }});
lCanvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const f=e.deltaY>0?0.9:1.1, r=lCanvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  lOffX=mx-f*(mx-lOffX); lOffY=my-f*(my-lOffY); lScale*=f;
}}, {{passive:false}});
lCanvas.addEventListener("mouseleave", () => {{ tooltip.style.display="none"; }});

// ===== RIGHT PANEL: Intra-doc graph =====
const rCanvas = document.getElementById("right-canvas");
const rCtx = rCanvas.getContext("2d");
let rW, rH, rOffX=0, rOffY=0, rScale=1;
let rDragging=false, rDragSX, rDragSY, rDragOX, rDragOY;
let rHovered=null;
let rNodes=[], rEdges=[], rNodeMap={{}};
let rActive=false;

function rResize() {{
  if (!rActive) return;
  const parent = rCanvas.parentElement;
  rW = parent.clientWidth - 220;
  rH = parent.clientHeight;
  rCanvas.width = rW * devicePixelRatio;
  rCanvas.height = rH * devicePixelRatio;
  rCanvas.style.width = rW + "px";
  rCanvas.style.height = rH + "px";
  rCtx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}}

function rW2S(wx,wy) {{
  const s=Math.min(rW,rH)*0.38;
  return [wx*rScale*s + rW/2 + rOffX, -wy*rScale*s + rH/2 + rOffY];
}}

const EDGE_COLORS = {{
  "section_cooccurrence": "#4477aa",
  "context_influences_stage": "#ee6677",
  "stage_transition": "#228833",
  "cross_doc": "#ccbb44",
}};

function rDraw() {{
  if (!rActive) {{ requestAnimationFrame(rDraw); return; }}
  rCtx.clearRect(0,0,rW,rH);

  // Edges — line width and opacity proportional to weight
  // Weight ranges: cooccurrence 0.15-1.0, influence 0.01-1.0, transition 1-24, cross_doc 0.3-1.0
  for (const e of rEdges) {{
    const sn=rNodeMap[e.source], tn=rNodeMap[e.target];
    if (!sn||!tn) continue;
    const [x1,y1]=rW2S(sn.x,sn.y), [x2,y2]=rW2S(tn.x,tn.y);
    const w = e.weight;
    let a, lw;
    if (e.type==="section_cooccurrence") {{
      // w: 0.15-1.0 → lw: 0.5-4, alpha: 0.06-0.35
      const t = Math.max(0, (w - 0.15) / 0.85);
      lw = 0.5 + t * 3.5;
      a = 0.06 + t * 0.29;
    }} else if (e.type==="context_influences_stage") {{
      // w: 0.01-1.0 → lw: 0.5-5, alpha: 0.08-0.6
      const t = Math.max(0, (w - 0.01) / 0.99);
      lw = 0.5 + t * 4.5;
      a = 0.08 + t * 0.52;
    }} else if (e.type==="stage_transition") {{
      // w: 1-24 → lw: 1.5-8, alpha: 0.3-0.85
      const t = Math.max(0, (w - 1) / 23);
      lw = 1.5 + t * 6.5;
      a = 0.3 + t * 0.55;
    }} else {{
      // cross_doc: w 0.3-1.0 → lw: 0.8-3, alpha: 0.15-0.5
      const t = Math.max(0, (w - 0.3) / 0.7);
      lw = 0.8 + t * 2.2;
      a = 0.15 + t * 0.35;
    }}
    if (rHovered) {{
      if (e.source===rHovered||e.target===rHovered) {{ a=Math.min(a*2.5,0.95); lw*=1.4; }}
      else {{ a*=0.15; }}
    }}
    rCtx.globalAlpha=a;
    rCtx.strokeStyle=EDGE_COLORS[e.type]||"#555";
    rCtx.lineWidth=lw;
    rCtx.beginPath(); rCtx.moveTo(x1,y1); rCtx.lineTo(x2,y2); rCtx.stroke();
    // Arrow for stage transitions
    if (e.type==="stage_transition") {{
      const dx=x2-x1, dy=y2-y1, len=Math.hypot(dx,dy);
      if (len>10) {{
        const ux=dx/len, uy=dy/len;
        const arrowSize = 3 + lw;
        const ax=x2-ux*(arrowSize+2), ay=y2-uy*(arrowSize+2);
        rCtx.fillStyle="#228833";
        rCtx.globalAlpha=Math.min(a*1.5,0.95);
        rCtx.beginPath();
        rCtx.moveTo(x2,y2);
        rCtx.lineTo(ax-uy*arrowSize*0.6, ay+ux*arrowSize*0.6);
        rCtx.lineTo(ax+uy*arrowSize*0.6, ay-ux*arrowSize*0.6);
        rCtx.fill();
      }}
    }}
  }}

  // Nodes
  for (const n of rNodes) {{
    const [sx,sy]=rW2S(n.x,n.y);
    if (sx<-30||sx>rW+30||sy<-30||sy>rH+30) continue;
    let r = n.type==="stage" ? 10 : (n.type==="neighbor" ? 6 : 8);
    r *= Math.min(rScale, 2.5);
    let a = 0.9;
    if (rHovered && n.id!==rHovered) a=0.4;
    if (n.id===rHovered) {{ r*=1.3; a=1; }}

    rCtx.globalAlpha=a;

    if (n.type==="stage") {{
      // Diamond shape for stages
      rCtx.fillStyle=n.color;
      rCtx.beginPath();
      rCtx.moveTo(sx, sy-r); rCtx.lineTo(sx+r, sy);
      rCtx.lineTo(sx, sy+r); rCtx.lineTo(sx-r, sy);
      rCtx.closePath(); rCtx.fill();
      rCtx.strokeStyle="#000"; rCtx.lineWidth=1; rCtx.stroke();
    }} else if (n.type==="neighbor") {{
      // Small circle for neighbors
      rCtx.fillStyle=n.color;
      rCtx.beginPath(); rCtx.arc(sx,sy,r,0,Math.PI*2); rCtx.fill();
      rCtx.strokeStyle="#555"; rCtx.lineWidth=0.5; rCtx.stroke();
    }} else {{
      // Circle for context
      rCtx.fillStyle=n.color;
      rCtx.beginPath(); rCtx.arc(sx,sy,r,0,Math.PI*2); rCtx.fill();
      rCtx.strokeStyle="#000"; rCtx.lineWidth=1; rCtx.stroke();
    }}

    // Label if zoomed in enough
    if (rScale > 0.6) {{
      rCtx.globalAlpha = n.id===rHovered ? 1 : 0.7;
      rCtx.fillStyle="#fff";
      rCtx.font=(n.type==="neighbor"?"9":"10")+"px 'Segoe UI'";
      rCtx.textAlign="center";
      rCtx.fillText(n.label, sx, sy+r+12);
    }}
  }}
  rCtx.globalAlpha=1;
  requestAnimationFrame(rDraw);
}}

function rHitTest(mx,my) {{
  let best=null, bd=Infinity;
  for (const n of rNodes) {{
    const [sx,sy]=rW2S(n.x,n.y);
    const d=Math.hypot(sx-mx,sy-my);
    const r=(n.type==="stage"?10:8)*Math.min(rScale,2.5)+4;
    if (d<r&&d<bd) {{best=n;bd=d;}}
  }}
  return best;
}}

rCanvas.addEventListener("mousemove", e => {{
  const r=rCanvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  if (rDragging) {{ rOffX=rDragOX+(e.clientX-rDragSX); rOffY=rDragOY+(e.clientY-rDragSY); return; }}
  const n=rHitTest(mx,my);
  rHovered=n?n.id:null;
  rCanvas.style.cursor=n?"pointer":"grab";
  if (n) {{
    let html='<div class="t-label">'+n.label+'</div><div class="t-row">Type: <span>'+n.type+'</span></div>';
    if (n.type==="neighbor") html+='<div class="t-row">Similarity: <span>'+n.sim+'</span></div>';
    if (n.group) html+='<div class="t-row">Section: <span>'+n.group+'</span></div>';
    // Show connected edges with weights
    const connEdges = rEdges.filter(e => e.source===n.id || e.target===n.id);
    if (connEdges.length > 0) {{
      const byType = {{}};
      connEdges.forEach(e => {{
        if (!byType[e.type]) byType[e.type] = [];
        const other = e.source===n.id ? e.target : e.source;
        const otherNode = rNodeMap[other];
        byType[e.type].push({{label: otherNode ? otherNode.label : other, w: e.weight}});
      }});
      html += '<div style="margin-top:4px;border-top:1px solid #444;padding-top:4px">';
      for (const [etype, conns] of Object.entries(byType)) {{
        const ename = etype.replace(/_/g," ");
        const sorted = conns.sort((a,b) => b.w - a.w).slice(0, 5);
        html += '<div class="t-row" style="margin-top:2px"><span style="color:'+(EDGE_COLORS[etype]||"#888")+'">'+ename+'</span></div>';
        sorted.forEach(c => {{
          html += '<div class="t-row" style="font-size:10px;padding-left:8px">'+c.label+': <span>'+c.w+'</span></div>';
        }});
        if (conns.length > 5) html += '<div class="t-row" style="font-size:10px;padding-left:8px;color:#666">+'+(conns.length-5)+' more</div>';
      }}
      html += '</div>';
    }}
    tooltip.innerHTML=html;
    tooltip.style.display="block";
    tooltip.style.left=Math.min(e.clientX+12,window.innerWidth-380)+"px";
    tooltip.style.top=Math.min(e.clientY+12,window.innerHeight-120)+"px";
  }} else tooltip.style.display="none";
}});
rCanvas.addEventListener("mousedown", e => {{ rDragging=true; rDragSX=e.clientX; rDragSY=e.clientY; rDragOX=rOffX; rDragOY=rOffY; }});
rCanvas.addEventListener("mouseup", e => {{
  rDragging=false;
  const moved=Math.hypot(e.clientX-rDragSX,e.clientY-rDragSY);
  if (moved<5) {{
    const r=rCanvas.getBoundingClientRect(), n=rHitTest(e.clientX-r.left,e.clientY-r.top);
    if (n && n.type==="neighbor") {{
      // Navigate to that document
      lSelected=n.doc_id;
      showInternalGraph(n.doc_id);
    }}
  }}
}});
rCanvas.addEventListener("dblclick", () => {{ rOffX=0;rOffY=0;rScale=1; }});
rCanvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const f=e.deltaY>0?0.9:1.1, r=rCanvas.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  rOffX=mx-f*(mx-rOffX); rOffY=my-f*(my-rOffY); rScale*=f;
}}, {{passive:false}});
rCanvas.addEventListener("mouseleave", () => {{ tooltip.style.display="none"; }});

function showInternalGraph(docId) {{
  const data = INTERNAL[docId];
  if (!data) return;
  rActive = true;
  rOffX=0; rOffY=0; rScale=1; rHovered=null;

  document.getElementById("placeholder").style.display="none";
  rCanvas.style.display="block";
  document.getElementById("right-sidebar").style.display="block";
  document.getElementById("right-title").textContent = "Intra-Document Graph";
  document.getElementById("right-subtitle").textContent = docId.substring(0,60) + (docId.length>60?"...":"");

  // Build internal graph with force layout
  const iG = {{}};
  rNodes = [];
  rEdges = [];
  rNodeMap = {{}};

  // Add context + stage nodes
  data.nodes.forEach(n => {{
    rNodes.push({{...n, x:0, y:0}});
    rNodeMap[n.id] = rNodes[rNodes.length-1];
  }});

  // Add neighbor nodes (ring around)
  const nbs = data.neighbors.slice(0, 12);
  nbs.forEach((nb, i) => {{
    const angle = (2*Math.PI*i)/nbs.length;
    const id = "nb_"+nb.doc_id;
    const shortLabel = nb.doc_id.substring(0,25)+"...";
    rNodes.push({{
      id: id, doc_id: nb.doc_id, type:"neighbor", color:nb.color,
      label: shortLabel, sim: nb.overall, x: Math.cos(angle)*1.8, y: Math.sin(angle)*1.8,
    }});
    rNodeMap[id] = rNodes[rNodes.length-1];
    // Add cross-doc edge to center (connect to first context node if exists)
    const centerNode = data.nodes.length > 0 ? data.nodes[0].id : null;
    if (centerNode) {{
      rEdges.push({{source:centerNode, target:id, type:"cross_doc", weight:nb.overall}});
    }}
  }});

  // Add internal edges
  data.edges.forEach(e => {{
    if (rNodeMap[e.source] && rNodeMap[e.target])
      rEdges.push(e);
  }});

  // Simple force layout for internal nodes
  const contextNodes = rNodes.filter(n=>n.type==="context");
  const stageNodes = rNodes.filter(n=>n.type==="stage");
  // Place context in a circle, stages in inner circle
  contextNodes.forEach((n,i) => {{
    const a = (2*Math.PI*i)/Math.max(contextNodes.length,1);
    n.x = Math.cos(a)*0.8; n.y = Math.sin(a)*0.8;
  }});
  stageNodes.forEach((n,i) => {{
    const a = (2*Math.PI*i)/Math.max(stageNodes.length,1) + 0.3;
    n.x = Math.cos(a)*0.35; n.y = Math.sin(a)*0.35;
  }});

  // Quick force iterations
  for (let iter=0; iter<100; iter++) {{
    // Repulsion between all non-neighbor nodes
    for (let i=0;i<rNodes.length;i++) {{
      if (rNodes[i].type==="neighbor") continue;
      for (let j=i+1;j<rNodes.length;j++) {{
        if (rNodes[j].type==="neighbor") continue;
        let dx=rNodes[j].x-rNodes[i].x, dy=rNodes[j].y-rNodes[i].y;
        let d=Math.max(Math.hypot(dx,dy),0.01);
        let f=0.002/(d*d);
        rNodes[i].x-=dx*f; rNodes[i].y-=dy*f;
        rNodes[j].x+=dx*f; rNodes[j].y+=dy*f;
      }}
    }}
    // Attraction along edges
    for (const e of rEdges) {{
      const a=rNodeMap[e.source], b=rNodeMap[e.target];
      if (!a||!b||a.type==="neighbor"||b.type==="neighbor") continue;
      let dx=b.x-a.x, dy=b.y-a.y;
      let d=Math.hypot(dx,dy);
      let ideal = e.type==="stage_transition" ? 0.2 : 0.4;
      let f=(d-ideal)*0.01;
      a.x+=dx*f; a.y+=dy*f;
      b.x-=dx*f; b.y-=dy*f;
    }}
  }}

  // Populate sidebar
  const secLegend = document.getElementById("section-legend");
  secLegend.innerHTML = "";
  contextNodes.forEach(n => {{
    secLegend.innerHTML += '<div class="legend-row"><div class="legend-dot" style="background:'+n.color+'"></div>'+n.label+'</div>';
  }});

  const stageList = document.getElementById("stage-list");
  stageList.innerHTML = "";
  if (stageNodes.length === 0) stageList.innerHTML = '<div style="color:#555">No stages extracted</div>';
  stageNodes.forEach(n => {{
    stageList.innerHTML += '<div class="legend-row"><div class="legend-dot" style="background:#FFD700;transform:rotate(45deg);border-radius:1px"></div>'+n.label+'</div>';
  }});

  const nbList = document.getElementById("neighbor-list");
  nbList.innerHTML = "";
  nbs.forEach(nb => {{
    const div = document.createElement("div");
    div.className = "nb-row";
    div.innerHTML = '<div class="nb-doc" style="color:'+nb.color+'">'+nb.doc_id.substring(0,50)+'</div><div class="nb-sim">sim: '+nb.overall+' &middot; strategy: '+nb.strategy+'</div>';
    div.onclick = () => {{ lSelected=nb.doc_id; showInternalGraph(nb.doc_id); }};
    nbList.appendChild(div);
  }});

  rResize();
}}

function hideInternalGraph() {{
  rActive = false;
  document.getElementById("placeholder").style.display="flex";
  rCanvas.style.display="none";
  document.getElementById("right-sidebar").style.display="none";
  document.getElementById("right-subtitle").textContent="Click a document to explore";
}}

window.addEventListener("resize", () => {{ lResize(); rResize(); }});
searchInput.addEventListener("input", () => {{ lSelected=null; hideInternalGraph(); }});

lResize(); lDraw(); rDraw();
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
