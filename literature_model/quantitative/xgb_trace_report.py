"""Render frozen XGBoost V2 traces (from xgb_trace_export) to a self-contained
HTML dossier.

Where the KNN dossier answered "which real projects voted for this block", this
one answers the question XGBoost can actually be held to:

  what did it predict      the flowsheet, drawn from the stages it selected
  how sure was it          probability against that stage's own threshold,
                           which varies from 0.65 to 0.90 across stages
  what drove it            the NI 43-101 sections feeding that classifier
  was it right             every block marked against the project's disclosed
                           flowsheet, including the ones the model missed
  how often is it right    each stage's held-out F1 and support, so a reader
                           can discount the blocks with thin evidence

Reuses the KNN dossier's stylesheet and flowsheet renderer unchanged.

Usage:
    python -m quantitative.xgb_trace_report \
        --traces quantitative/demo/xgb_traces.json \
        --out quantitative/demo/xgb_trace_demo.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantitative import flowsheet_svg
from quantitative.xgb_trace_export import EDGE_SCORE_MIN
from quantitative.trace_report import (
    _STYLE, _esc, _meter, _pretty, _provenance_line, _undrawn_reason,
    STATE_INSERTED, STATE_UNDRAWN)

# Blocks whose own held-out F1 is below this are flagged as thin evidence
# rather than presented at face value.
WEAK_F1 = 0.75

# The influence matrix keys sections by the short group names in
# enrich_graphs.SECTION_GROUPS, which are shorthand for NI 43-101 items and
# read misleadingly on their own — "climate" is Item 5, whose subject is site
# access, power, water and local resources, not weather. Showing the raw key
# would tell a reader the climate section chose their leach circuit.
SECTION_LABELS = {
    "climate": "site access &amp; local resources (5)",
    "geology": "geology &amp; deposit type (7&ndash;8)",
    "metallurgical_testing": "metallurgical testing (13)",
    "mining_method": "mining method (16)",
    "recovery_methods": "recovery methods (17)",
    "infrastructure": "project infrastructure (18)",
    "economics": "economics (21&ndash;22)",
    # Also reachable through the cross-document match chips, which report
    # similarity across every section group, not just the seven the influence
    # matrix covers.
    "summary": "summary (2&ndash;3)",
    "property": "property description (4)",
    "history": "history (6)",
    "exploration": "exploration (9)",
    "drilling": "drilling (10)",
    "sample_analysis": "sample preparation (11)",
    "data_verification": "data verification (12)",
    "resource_estimate": "resource estimate (14&ndash;15)",
    "market": "market studies (19)",
    "environmental": "environmental &amp; permitting (20)",
    "interpretation": "interpretation (23)",
    "references": "references (24&ndash;27)",
}


def _graph_inputs(ex: dict) -> tuple:
    """What the flowsheet is drawn from: the selected stages and the corpus
    transitions between them that cleared the relative-support cut."""
    return ({r["stage"]: 1 for r in ex["stages"] if r["selected"]},
            {(e["src"], e["dst"]): e["score"] for e in ex["edges"]})


# XGBoost has no vote to reject a link, so it cannot reach the KNN dossier's
# "reinstated" or "below cut" states from the frozen export — the transitions
# that failed the cut are not written out. What it does share is the pair that
# matters most: an arrow nothing supports, and a supported transition the
# renderer refused to draw.
STATE_CORPUS = "corpus"


def _connection_rows(ex: dict) -> list:
    """Each arrow on the flowsheet, paired with the corpus transition behind
    it. An arrow with no transition is the renderer's orphan-reconnection rule
    filling a gap the corpus edges left — not evidence of any kind."""
    seen = {(e["src"], e["dst"]): e for e in ex["edges"]}
    rows = flowsheet_svg.drawn_edges(*_graph_inputs(ex))
    for r in rows:
        r["edge"] = seen.get((r["src"], r["dst"]))
        r["state"] = STATE_INSERTED if r["derived"] else STATE_CORPUS
    return rows


def _considered_rows(ex: dict, drawn: list) -> list:
    """Corpus transitions between two predicted blocks that the flowsheet does
    not show. They cleared the support cut and were then dropped by a renderer
    rule the corpus statistics know nothing about, so the picture is quietly
    narrower than the evidence behind it."""
    shown = {(r["src"], r["dst"]) for r in drawn}
    out = []
    for e in ex["edges"]:
        pair = (e["src"], e["dst"])
        if pair in shown:
            continue
        out.append({
            "src": e["src"], "dst": e["dst"], "edge": e, "weight": e["score"],
            "derived": False, "derived_for": None, "idx": None,
            "kind": "recycle" if e["src"] == e["dst"] else None,
            "state": STATE_UNDRAWN, "reason": _undrawn_reason(*pair),
        })
    out.sort(key=lambda r: -r["weight"])
    return out


def _hero(ex: dict) -> str:
    svg = flowsheet_svg.render_svg(*_graph_inputs(ex))
    legend = (
        '<div class="legend">'
        '<span class="lg"><span class="ln fwd"></span>feed-forward</span>'
        '<span class="lg"><span class="ln rcy"></span>recycle</span>'
        '<span class="lg-sep"></span>'
        '<span class="lg"><i class="dot ph1"></i>comminution</span>'
        '<span class="lg"><i class="dot ph2"></i>flotation</span>'
        '<span class="lg"><i class="dot ph3"></i>dewatering</span>'
        '<span class="lg"><i class="dot ph4"></i>hydromet</span>'
        '<span class="lg"><i class="dot ph5"></i>product / tails</span>'
        '</div>')
    return f'<div class="hero-card">{svg}{legend}</div>'


def _influence_card(r: dict, scale: float) -> str:
    matches = "".join(
        f'<span class="chip chip-sec">'
        f'{SECTION_LABELS.get(m["section"], _esc(_pretty(m["section"])))}'
        f'<span class="w">{m["similarity"]:.2f}</span></span>'
        for m in r["matches_on"])
    return (
        f'<div class="analog">'
        f'<div class="analog-head">'
        f'<span class="analog-name">{_esc(r["label"])}</span>'
        f'<span class="chip chip-study st-{_esc(r["study_type"].lower())}" '
        f'title="{_esc(r["study_type_label"] or "")}">{_esc(r["study_type"])}'
        f'</span></div>'
        f'{_provenance_line({k: r.get(k) for k in ("year", "year_source")}, show_grade=False)}'
        f'<div class="analog-metrics"><span class="metric">'
        f'<span class="k">similarity</span>'
        f'<span class="v">{r["similarity"]:.2f}</span></span></div>'
        f'{_meter(r["similarity"], 0, True, scale)}'
        f'<div class="chips">{matches}</div></div>')


def _influences(ex: dict) -> str:
    rows, total = ex.get("influences") or [], ex.get("influences_of") or 0
    if not rows:
        return (
            f'<h3 class="sec">Reports behind this prediction'
            f'<span class="sec-note">none of the {total} projects this '
            f'prediction was compared against state an FS or PFS grade</span>'
            f'</h3>')
    scale = max(r["similarity"] for r in rows)
    cards = "".join(_influence_card(r, scale) for r in rows)
    return (
        f'<h3 class="sec">Reports behind this prediction'
        f'<span class="sec-note">the FS/PFS-grade projects among the '
        f'{total} the model measured this one against &mdash; '
        f'{len(rows)} of {total} &middot; closer similarity, stronger '
        f'influence</span></h3>'
        f'<div class="analogs">{cards}</div>'
        f'<p class="note">Half of the model\'s inputs '
        f'(959 of 1918 features) are this project\'s numbers <i>minus</i> the '
        f'weighted average of those {total} projects, so they shape every '
        f'block above. They are ranked by overall similarity; the chips show '
        f'which report sections matched most closely.</p>')


def _scorecard(ex: dict) -> str:
    a = ex["accuracy"]
    return (
        f'<div class="score">'
        f'<span class="sc sc-ok"><b>{a["correct"]}</b> matched the disclosed '
        f'flowsheet</span>'
        f'<span class="sc sc-extra"><b>{a["extra"]}</b> predicted but not '
        f'disclosed</span>'
        f'<span class="sc sc-miss"><b>{a["missed"]}</b> disclosed but '
        f'missed</span>'
        f'<span class="sc sc-f1">doc F1 <b>{a["f1"]:.2f}</b></span></div>')


def _section_chips(sections: list) -> str:
    if not sections:
        return '<span class="chip chip-more">&mdash;</span>'
    return "".join(
        f'<span class="chip chip-sec" title="relative influence '
        f'{s["weight"]:.2f} of the strongest section for this stage '
        f'(NI 43-101 item in brackets)">'
        f'{SECTION_LABELS.get(s["section"], _esc(_pretty(s["section"])))}</span>'
        for s in sections)


def _reliability(row: dict) -> str:
    f1, sup = row.get("val_f1"), row.get("val_support")
    if f1 is None:
        return '<span class="chip chip-more">no val support</span>'
    cls = "rel-weak" if f1 < WEAK_F1 else "rel-ok"
    return (f'<span class="chip {cls}" title="held-out F1 for this stage '
            f'across {sup} validation documents">{f1:.2f}'
            f'<span class="w">n={sup}</span></span>')


def _stage_row(row: dict, scale: float) -> str:
    prob = row["probability"]
    if row["selected"] and row["in_ground_truth"]:
        badge, state = "badge-sel", "confirmed"
        title = "the model selected this block and the project disclosed it"
    elif row["selected"]:
        badge, state = "badge-extra", "not disclosed"
        title = ("the model selected this block but the project's disclosed "
                 "flowsheet does not list it — an error, or a stage the "
                 "disclosure omitted")
    else:
        badge, state = "badge-miss", "missed"
        title = ("the project disclosed this block and the model did not "
                 "return it")

    if prob is None:
        meter = '<span class="noprob" title="the checkpoint stores only the ' \
                'stages the model returned, so no probability exists for ' \
                'this one">not returned</span>'
        val = "&mdash;"
    else:
        meter = _meter(prob, row["threshold"], row["selected"], scale)
        val = f'{prob:.2f}'

    return (
        f'<div class="row">'
        f'<div class="row-name">{_esc(_pretty(row["stage"]))}</div>'
        f'<div class="row-meter">{meter}</div>'
        f'<div class="row-val">{val}</div>'
        f'<div class="row-cut" title="this stage\'s own decision threshold">'
        f'cut {row["threshold"]:.2f}</div>'
        f'<div class="row-state"><span class="badge {badge}" '
        f'title="{_esc(title)}">{state}</span></div>'
        f'<div class="row-rel">{_reliability(row)}</div>'
        f'<div class="row-support">{_section_chips(row["sections"])}</div>'
        f'</div>')


CORPUS_DOCS = 179  # disclosed flowsheets behind transition_counts.json


def _ceiling(src: str, dst: str) -> str:
    """The statement every XGBoost connection has to carry.

    The interaction invites a per-project question the model cannot answer:
    connectivity is a function of the predicted stage set alone, so two
    projects with the same blocks get byte-identical arrows. Saying it once in
    a footnote is not enough when a reader is opening links one at a time.
    """
    return (f'<p class="ev-say ev-ceil">This arrow is <b>not a prediction about '
            f'this project</b>. It is how often the training corpus runs '
            f'<b>{_esc(src)}</b> into <b>{_esc(dst)}</b> between two blocks the '
            f'model did select. Any project with the same stage set gets the '
            f'same arrow &mdash; the checkpoint has no learned edge model.</p>')


def _edge_body(r: dict) -> str:
    src, dst = _pretty(r["src"]), _pretty(r["dst"])
    stranded = _pretty(r["derived_for"]) if r.get("derived_for") else "a block"

    if r["state"] == STATE_INSERTED:
        return (f'<p class="ev-say ev-warn">No corpus transition supports this '
                f'link. The drawing rule added it because <b>{_esc(stranded)}</b> '
                f'would otherwise sit unreachable from the feed. <b>It is not a '
                f'model output and not evidence of any kind</b> &mdash; it is a '
                f'drawing convention that keeps the picture connected.</p>')

    edge = r.get("edge") or {}
    sup = edge.get("support")
    counted = (f'<p class="ev-say"><b>{sup}</b> of roughly {CORPUS_DOCS} '
               f'disclosed flowsheets in the training corpus run '
               f'<b>{_esc(src)}</b> into <b>{_esc(dst)}</b>.</p>'
               if sup is not None else '')

    if r["state"] == STATE_UNDRAWN:
        why = {
            "self-loop": (
                "the renderer drops self-loops. In the source reports this "
                "pattern is almost always two different units of the same type "
                "in series &mdash; a rougher feeding a scavenger, or one "
                "cleaner stage feeding the next &mdash; collapsed into a single "
                "stage name by the shared vocabulary. Roughly one raw "
                "connection in five hundred is a true self-loop; the rest is "
                "real circuit detail the vocabulary cannot express"),
            "into the feed": (
                "nothing may flow back into the feed node, which represents "
                "run-of-mine ore entering the plant"),
            "cannot terminate": (
                "this block may not discharge straight to product; the "
                "renderer requires a dewatering or recovery step first"),
        }.get(r["reason"], "the renderer did not draw it")
        return (counted
                + f'<p class="ev-say ev-warn">The corpus supports this '
                  f'transition and the flowsheet above does not show it: '
                  f'{why}.</p>'
                + _ceiling(src, dst))

    return counted + _ceiling(src, dst)


_STATE_BADGE = {
    STATE_CORPUS: ("badge-sel", "in corpus",
                   "cleared the relative-support cut among transitions out of "
                   "this block"),
    STATE_INSERTED: ("badge-derived", "drawing rule",
                     "not evidence - the renderer inserted this link to reach "
                     "a block the corpus edges left unreachable"),
    STATE_UNDRAWN: ("badge-undrawn", "supported, not drawn",
                    "the corpus supports this transition but the renderer "
                    "refused to draw it"),
}


def _edge_row(r: dict, scale: float) -> str:
    """One connection as an expandable row, matching the KNN dossier so the two
    read the same way. <details> is native and script-free."""
    cls, label, tip = _STATE_BADGE[r["state"]]
    num = (f'<span class="rn" title="arrow {r["idx"]} on the flowsheet">'
           f'{r["idx"]}</span>' if r.get("idx") else
           '<span class="rn rn-off" title="not on the flowsheet">&mdash;</span>')
    name = (f'{num}{_esc(_pretty(r["src"]))} <span class="arr">&rarr;</span> '
            f'{_esc(_pretty(r["dst"]))}')
    kind = ('<span class="chip chip-kind" title="returns to an earlier block">'
            'recycle</span>' if r.get("kind") == "recycle" else '')
    edge = r.get("edge") or {}
    sup = edge.get("support")
    sup_chip = (f'<span class="chip chip-sec" title="corpus flowsheets '
                f'containing this transition">{sup} reports</span>'
                if sup is not None else '')

    if r["state"] == STATE_INSERTED:
        meter = ('<span class="noprob" title="no corpus transition supports '
                 'this link">added to reach a stranded block</span>')
        val = "&mdash;"
    else:
        meter = _meter(r["weight"], EDGE_SCORE_MIN,
                       r["state"] == STATE_CORPUS, scale)
        val = f'{r["weight"]:.2f}'

    return (
        f'<details class="row-d">'
        f'<summary class="row">'
        f'<div class="row-name">{name}</div>'
        f'<div class="row-meter">{meter}</div>'
        f'<div class="row-val">{val}</div>'
        f'<div class="row-state"><span class="badge {cls}" '
        f'title="{_esc(tip)}">{label}</span></div>'
        f'<div class="row-support">{kind}{sup_chip}</div>'
        f'</summary>'
        f'<div class="ebody">{_edge_body(r)}</div>'
        f'</details>')


def _edge_table(rows: list, scale: float, head: str) -> str:
    body = "".join(_edge_row(r, scale) for r in rows)
    return (
        f'<div class="rows rows-head"><div class="row-name">{head}</div>'
        f'<div class="row-meter">relative strength</div>'
        f'<div class="row-val"></div><div class="row-state"></div>'
        f'<div class="row-support">corpus support</div></div>'
        f'<div class="rows rows-edge">{body}</div>')


def _connections(ex: dict, rows: list) -> str:
    """The arrow-by-arrow account of the flowsheet above.

    The arrows are the weakest claim on the page — corpus frequency between
    predicted stages, not a learned edge model — and a good share of them are
    not even that. Saying so per arrow is the only way a reader can tell the
    two apart, since the picture cannot.
    """
    if not rows:
        return ""
    considered = _considered_rows(ex, rows)
    weights = [r["weight"] for r in rows + considered if r.get("weight")]
    scale = max(weights + [EDGE_SCORE_MIN * 1.2])
    derived = sum(1 for r in rows if r["derived"])
    note = (f'{derived} of {len(rows)} added by the drawing rule, not '
            f'supported' if derived else 'every arrow has corpus support')
    out = (
        f'<h3 class="sec">Connections'
        f'<span class="sec-note">every arrow on the flowsheet above, numbered '
        f'to match it &mdash; strength relative to the busiest transition out '
        f'of that block, against the {EDGE_SCORE_MIN:g} cut &middot; {note} '
        f'&middot; open a row for the evidence behind it</span></h3>'
        f'{_edge_table(rows, scale, "connection")}')
    if considered:
        out += (
            f'<h3 class="sec">Supported but not drawn'
            f'<span class="sec-note">{len(considered)} corpus transition'
            f'{"s" if len(considered) != 1 else ""} between two predicted '
            f'blocks that the renderer refused to draw</span></h3>'
            f'{_edge_table(considered, scale, "link")}')
    out += (
        f'<p class="note">These arrows are not a prediction. They are how '
        f'often the training corpus runs one predicted stage into the next, '
        f'shared by every project with the same stage set &mdash; the '
        f'checkpoint has no learned edge model. The rows marked '
        f'<i>drawing rule</i> are weaker still: nothing proposed them, and the '
        f'renderer inserted them only so no selected block sits unreachable '
        f'from the feed.</p>')
    return out


def _panel(ex: dict, idx: int) -> str:
    rows = ex["stages"]
    scale = max([r["probability"] for r in rows if r["probability"]] + [1.0])
    selected = sum(1 for r in rows if r["selected"])
    body = "".join(_stage_row(r, scale) for r in rows)
    edges = _connection_rows(ex)
    hero = _hero(ex)
    return (
        f'<section class="panel" id="panel-{idx}">'
        f'<div class="panel-head">'
        f'<div class="panel-summary">{selected} stages selected &middot; '
        f'{len(edges)} connections drawn</div></div>'
        f'{_scorecard(ex)}'
        f'{hero}'
        f'{_influences(ex)}'
        f'<h3 class="sec">Stage decisions'
        f'<span class="sec-note">probability against each stage\'s own cut '
        f'&middot; whether the disclosure agreed &middot; that stage\'s '
        f'held-out F1 &middot; the report sections driving it</span></h3>'
        f'<div class="rows rows-head"><div class="row-name">stage</div>'
        f'<div class="row-meter">probability</div><div class="row-val"></div>'
        f'<div class="row-cut"></div><div class="row-state"></div>'
        f'<div class="row-rel">stage F1</div>'
        f'<div class="row-support">driving sections</div></div>'
        f'<div class="rows">{body}</div>'
        f'{_connections(ex, edges)}'
        f'</section>')


def render(traces: dict) -> str:
    examples, model = traces["examples"], traces["model"]
    radios = "".join(
        f'<input type="radio" name="ex" id="tab-{i}" class="tabradio"'
        f'{" checked" if i == 0 else ""}>' for i in range(len(examples)))
    tabs = "".join(
        f'<label for="tab-{i}" class="tab"><span class="tab-n">0{i + 1}</span>'
        f'{_esc(ex["label"])}</label>' for i, ex in enumerate(examples))
    panels = "".join(_panel(ex, i) for i, ex in enumerate(examples))
    return f"""{_STYLE}{_EXTRA_STYLE}
<div class="wrap">
  <header class="masthead">
    <div class="eyebrow">{_esc(model['name'])} &middot; decision trace</div>
    <h1>How the model designs a flowsheet</h1>
    <p class="lede">The synthesizer reads a project's technical report and
    decides, stage by stage, which process blocks the plant needs. Below is
    what it produced for real projects &mdash; every block with the confidence
    behind it, the report sections that drove it, its track record on unseen
    projects, and whether the project's own disclosed flowsheet agreed.
    Nothing is re-scored; this is the model's own output, drawn.</p>
  </header>
  {radios}
  <nav class="tabs">{tabs}</nav>
  <main class="panels">{panels}</main>
</div>"""


_EXTRA_STYLE = """<style>
.row,.rows-head{grid-template-columns:130px 1fr 40px 62px 104px 92px minmax(130px,1fr)}
.row-cut{font-family:var(--mono); font-size:11px; color:var(--ink-2); cursor:help}
.badge-extra{background:rgba(245,158,11,.14); color:#b45309}
.badge-miss{background:rgba(225,29,72,.12); color:#e11d48}
.chip-sec{background:var(--panel-2); color:var(--ink-2); border:1px solid var(--line)}
.chip-sec .w{opacity:.6; font-size:10px; margin-left:5px; font-family:var(--mono)}
.note{font-size:12.5px; line-height:1.6; color:var(--ink-2); max-width:78ch;
  margin:14px 0 0}
.rel-ok{background:var(--sel-soft); color:var(--sel); font-family:var(--mono);
  font-size:11px; display:inline-flex; gap:5px; align-items:center}
.rel-weak{background:rgba(225,29,72,.10); color:#e11d48; font-family:var(--mono);
  font-size:11px; display:inline-flex; gap:5px; align-items:center}
.rel-ok .w,.rel-weak .w{opacity:.65; font-size:10px}
.score{display:flex; flex-wrap:wrap; gap:8px; margin:0 0 18px}
.sc{font-size:12.5px; padding:5px 11px; border-radius:8px; background:var(--panel-2);
  color:var(--ink-2); border:1px solid var(--line)}
.sc b{font-family:var(--mono); color:var(--ink)}
.sc-ok{background:var(--sel-soft); border-color:transparent; color:var(--accent-2)}
.sc-ok b{color:var(--accent-2)}
.sc-extra{background:rgba(245,158,11,.12); border-color:transparent; color:#b45309}
.sc-extra b{color:#b45309}
.sc-miss{background:rgba(225,29,72,.10); border-color:transparent; color:#e11d48}
.sc-miss b{color:#e11d48}
@media (max-width:820px){
  .row,.rows-head,.rows-edge .row{grid-template-columns:1fr 1fr; gap:6px 10px}
  .rows-head{display:none}
  .row-meter,.row-rel,.row-support{grid-column:1 / -1}
}
</style>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", default="quantitative/demo/xgb_traces.json")
    ap.add_argument("--out", default="quantitative/demo/xgb_trace_demo.html")
    args = ap.parse_args()
    traces = json.loads(Path(args.traces).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(traces), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB, "
          f"{len(traces['examples'])} examples)")


if __name__ == "__main__":
    main()
