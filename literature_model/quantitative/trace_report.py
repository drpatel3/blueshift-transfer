"""Render frozen KNN decision traces (from trace_export.py) to a self-contained
HTML flowsheet dossier for stakeholders.

The hero of each project is the *designed flowsheet*, drawn with the project's
existing PFS equipment-symbol display (mini/pfs_visual, via quantitative.
pfs_render) and embedded as a base64 PNG. Below it, the analog projects and
per-stage / per-connection votes show *why* each block and link was chosen.

Self-contained and script-free (a CSS-only tab control switches examples). Every
number is a straight re-derivation of the model's own prediction; this view
changes nothing. Analogs are pseudonymized (Analog A..E).

Usage:
    python -m quantitative.trace_report \
        --traces quantitative/demo/traces.json \
        --out quantitative/demo/knn_trace_demo.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from quantitative import flowsheet_svg, pfs_render

STAGE_THRESHOLD = 0.5
EDGE_THRESHOLD = 0.3
BINARY = (0.0, 1.0)


def _esc(s) -> str:
    return html.escape(str(s))


def _pretty(feature: str) -> str:
    s = feature.replace("_", " ")
    for prefix in ("ore ", "dep ", "deposit ", "mine "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip()


def _fmt(v: float) -> str:
    return f"{v:g}"


# ---------------------------------------------------------------- hero flowsheet

def _grade_of(ex: dict):
    for fm in ex["neighbors"][0]["feature_match"]:
        if fm["feature"] == "cu_grade":
            return fm["query_value"]
    return None


def _graph_inputs(ex: dict) -> tuple:
    """What the flowsheet is drawn from: the selected stages and the edge
    votes that cleared the cut. One source for the picture and the tables."""
    return ({sv["stage"]: 1 for sv in ex["stage_votes"] if sv["selected"]},
            {(e["src"], e["dst"]): e["score"]
             for e in ex["edge_votes"] if e["selected"]})


# A connection is in exactly one of five states. The table used to know two
# of them, which is why three quarters of the model's connection reasoning had
# no surface: a link the vote rejected and the renderer then reinstated read
# identically to a link nothing ever proposed, and a link the vote *selected*
# and the renderer refused to draw appeared nowhere at all.
STATE_VOTED = "voted"          # A - cleared the cut, drawn
STATE_INSERTED = "inserted"    # B - drawing rule, never proposed
STATE_REINSTATED = "reinstated"  # C - rejected by the vote, drawn anyway
STATE_UNDRAWN = "undrawn"      # D - cleared the cut, renderer refused
STATE_REJECTED = "rejected"    # E - scored below the cut

DRAWN_STATES = (STATE_VOTED, STATE_INSERTED, STATE_REINSTATED)


def _undrawn_reason(src: str, dst: str) -> str:
    """Why the renderer refused a link the vote had selected.

    Mirrors the filters in pfs_render.build_knn_graph. Returning the rule by
    name matters: "not drawn" alone reads as an oversight, when in fact the
    renderer applied a deliberate topology rule the vote knows nothing about.
    """
    if src == dst:
        return "self-loop"
    if dst == "input":
        return "into the feed"
    if dst == "output" and src in pfs_render._NO_DIRECT_OUTPUT:
        return "cannot terminate"
    return "not drawn"


def _connection_rows(ex: dict) -> list:
    """Each arrow on the flowsheet, paired with the vote behind it and its
    state. A drawn arrow with no vote came from the renderer's
    orphan-reconnection rule; a drawn arrow whose vote was *rejected* is the
    renderer overriding the model, which is a different claim entirely.
    """
    votes = {(e["src"], e["dst"]): e for e in ex["edge_votes"]}
    rows = flowsheet_svg.drawn_edges(*_graph_inputs(ex))
    for r in rows:
        vote = votes.get((r["src"], r["dst"]))
        r["vote"] = vote
        if not r["derived"]:
            r["state"] = STATE_VOTED
        elif vote is None:
            r["state"] = STATE_INSERTED
        else:
            r["state"] = STATE_REINSTATED
    return rows


def _considered_rows(ex: dict, drawn: list) -> list:
    """Links the model scored that never reached the picture.

    Two kinds, and they are opposites: the vote selected it and the renderer
    refused (D), or the vote turned it down (E). Both are design alternatives a
    process engineer would want to see, and neither has ever been rendered.
    """
    shown = {(r["src"], r["dst"]) for r in drawn}
    out = []
    for e in ex["edge_votes"]:
        pair = (e["src"], e["dst"])
        if pair in shown:
            continue
        out.append({
            "src": e["src"], "dst": e["dst"], "vote": e, "weight": e["score"],
            "derived": False, "derived_for": None, "idx": None,
            "kind": "recycle" if e["src"] == e["dst"] else None,
            "state": STATE_UNDRAWN if e["selected"] else STATE_REJECTED,
            "reason": _undrawn_reason(*pair) if e["selected"] else None,
        })
    out.sort(key=lambda r: (r["state"] != STATE_UNDRAWN, -r["weight"]))
    return out


def _hero(ex: dict) -> str:
    svg = flowsheet_svg.render_svg(*_graph_inputs(ex))
    legend = (
        '<div class="legend">'
        '<span class="lg"><span class="ln fwd"></span>feed-forward</span>'
        '<span class="lg"><span class="ln rcy"></span>recycle</span>'
        '<span class="lg-sep"></span>'
        '<span class="lg"><i class="dot ph0"></i>feed</span>'
        '<span class="lg"><i class="dot ph1"></i>comminution</span>'
        '<span class="lg"><i class="dot ph2"></i>flotation</span>'
        '<span class="lg"><i class="dot ph3"></i>dewatering</span>'
        '<span class="lg"><i class="dot ph4"></i>hydromet</span>'
        '<span class="lg"><i class="dot ph5"></i>product / tails</span>'
        '</div>')
    return (
        f'<div class="hero-card">{svg}{legend}'
        f'<div class="hero-cap">The flowsheet the model designed &mdash; each '
        f'block is a process stage it selected; arrows are its voted '
        f'connections (dashed = recycle), plus the links the drawing rule '
        f'adds to reach a block the votes left stranded. Every arrow is '
        f'itemised below. Same topology as the project\'s PFS '
        f'display, drawn natively.</div></div>')


# ---------------------------------------------------------------- evidence

def _agree_chips(feature_match: list) -> str:
    agree = [fm for fm in feature_match if fm["contribution"] < 1e-9]
    chips, absent = [], 0
    for fm in agree:
        q = fm["query_value"]
        if q in BINARY and fm["neighbor_value"] in BINARY:
            if q == 1.0:
                chips.append(_pretty(fm["feature"]))
            else:
                absent += 1
        else:
            chips.append(f"{_pretty(fm['feature'])} {_fmt(q)}")
    shown = chips[:5]
    out = "".join(f'<span class="chip chip-agree">{_esc(c)}</span>' for c in shown)
    extra = len(chips) - len(shown)
    if extra > 0:
        out += f'<span class="chip chip-more">+{extra}</span>'
    return out or '<span class="chip chip-more">—</span>'


def _provenance_line(prov: dict, show_grade: bool = True) -> str:
    """Vintage of one document — how old the disclosure behind its vote is.

    `show_grade=False` renders the year alone, for callers that already display
    the study grade elsewhere on the same card; without it those callers would
    get the "not stated" chip beside a grade they have just shown.
    """
    if not prov:
        return ""
    year = prov.get("year")
    year_txt = str(year) if year else "year n/a"
    if year and prov.get("year_source") == "filing":
        title = "SEDAR filing year (no effective date stated on the report)"
    elif year:
        title = "effective date stated on the report"
    else:
        title = "no date recovered from the document"
    study = prov.get("study_type")
    if not show_grade:
        chip = ""
    elif study:
        chip = (f'<span class="chip chip-study st-{_esc(study.lower())}" '
                f'title="{_esc(prov.get("study_type_label") or study)}'
                f' — stated in the report\'s own title">{_esc(study)}</span>')
    elif prov.get("grade_stated"):
        # States a grade, just not one in the displayed set. Must not be called
        # "not stated" — that would read as missing documentation when the
        # truth is a study grade below FS/PFS.
        chip = ('<span class="chip chip-study st-other" title="this report '
                'states a study grade outside the FS/PFS pair shown here">'
                'other grade</span>')
    else:
        # Render the absence explicitly. A missing chip is indistinguishable
        # from a missing feature, and it would also let a reader assume the
        # study grade is simply unknown to us rather than unstated by the report.
        chip = ('<span class="chip chip-study st-none" title="the report does '
                'not state a study grade in its title or summary">not stated</span>')
    return (f'<div class="prov">'
            f'<span class="prov-year" title="{_esc(title)}">{_esc(year_txt)}</span>'
            f'{chip}</div>')


def _provenance_summary(ex: dict) -> str:
    """Vintage span across the k analogs."""
    years = sorted(p["year"] for p in
                   (nb.get("provenance") or {} for nb in ex["neighbors"])
                   if p.get("year"))
    if not years:
        return "analog years n/a"
    span = (f"{years[0]}" if years[0] == years[-1]
            else f"{years[0]}&ndash;{years[-1]}")
    return f"analogs {span}"


def _meter(value: float, threshold: float, selected: bool, scale: float) -> str:
    pct = max(0.0, min(1.0, value / scale)) * 100
    cls = "fill-sel" if selected else "fill-cut"
    thresh = ""
    if threshold > 0:
        tpct = max(0.0, min(1.0, threshold / scale)) * 100
        thresh = (f'<div class="meter-thresh" style="left:{tpct:.1f}%" '
                  f'title="decision threshold {threshold:g}"></div>')
    return (f'<div class="meter"><div class="meter-fill {cls}" '
            f'style="width:{pct:.1f}%"></div>{thresh}</div>')


def _support_chips(supporters: list) -> str:
    return "".join(
        f'<span class="chip chip-sup">{_esc(s["doc_id"])}'
        f'<span class="w">{s["similarity_weight"]:.2f}</span></span>'
        for s in supporters)


def _analog_card(nb: dict, weight_scale: float) -> str:
    diverge = sorted((fm for fm in nb["feature_match"] if fm["contribution"] >= 1e-9),
                     key=lambda f: -f["contribution"])[:1]
    differ = ""
    if diverge:
        fm = diverge[0]
        q, n = fm["query_value"], fm["neighbor_value"]
        if q in BINARY and n in BINARY:
            txt = f"{_pretty(fm['feature'])}: {'yes' if q == 1 else 'no'} vs {'yes' if n == 1 else 'no'}"
        else:
            txt = f"{_pretty(fm['feature'])}: {q:.2f} vs {n:.2f}"
        differ = f'<div class="differ">differs on {_esc(txt)}</div>'
    return (
        f'<div class="analog">'
        f'<div class="analog-head"><span class="analog-name">{_esc(nb["doc_id"])}</span>'
        f'<span class="rank">#{nb["rank"] + 1}</span></div>'
        f'{_provenance_line(nb.get("provenance"))}'
        f'<div class="analog-metrics">'
        f'<span class="metric"><span class="k">distance</span>'
        f'<span class="v">{nb["distance"]:.2f}</span></span>'
        f'<span class="metric"><span class="k">vote</span>'
        f'<span class="v">{nb["similarity_weight"]:.2f}</span></span></div>'
        f'{_meter(nb["similarity_weight"], 0, True, weight_scale)}'
        f'<div class="chips">{_agree_chips(nb["feature_match"])}</div>{differ}</div>')


def _stage_row(sv: dict, scale: float) -> str:
    badge = "badge-sel" if sv["selected"] else "badge-cut"
    state = "selected" if sv["selected"] else "below cut"
    return (
        f'<div class="row">'
        f'<div class="row-name">{_esc(_pretty(sv["stage"]))}</div>'
        f'<div class="row-meter">{_meter(sv["probability"], STAGE_THRESHOLD, sv["selected"], scale)}</div>'
        f'<div class="row-val">{sv["probability"]:.2f}</div>'
        f'<div class="row-state"><span class="badge {badge}">{state}</span></div>'
        f'<div class="row-support">{_support_chips(sv["supporting_neighbors"])}</div>'
        f'</div>')


def _neighbor_index(ex: dict) -> dict:
    """doc_id -> the full neighbor record.

    Supporters are exported as {doc_id, similarity_weight} while the same
    analog carries its rank, distance, year and study grade in `neighbors`.
    Joining them costs nothing and is the difference between "Analog A" and
    "Analog A, the closest match, a 2014 feasibility study".
    """
    return {nb["doc_id"]: nb for nb in ex["neighbors"]}


def _rich_supporter(sup: dict, idx: dict) -> str:
    nb = idx.get(sup["doc_id"]) or {}
    prov = nb.get("provenance") or {}
    bits = []
    if nb.get("rank") is not None:
        bits.append(f'#{nb["rank"] + 1}')
    if nb.get("distance") is not None:
        bits.append(f'dist {nb["distance"]:.2f}')
    if prov.get("year"):
        bits.append(str(prov["year"]))
    if prov.get("study_type"):
        bits.append(_esc(prov["study_type"]))
    meta = " &middot; ".join(bits)
    return (
        f'<span class="sup-rich" title="weight is this analog&rsquo;s share of '
        f'the vote for this connection">'
        f'<b>{_esc(sup["doc_id"])}</b>'
        f'<span class="sup-meta">{meta}</span>'
        f'<span class="sup-w">{sup["similarity_weight"]:.2f}</span></span>')


def _supporters_block(r: dict, ex: dict, heading: str) -> str:
    sups = (r.get("vote") or {}).get("supporting_neighbors") or []
    if not sups:
        return ""
    idx = _neighbor_index(ex)
    chips = "".join(_rich_supporter(s, idx) for s in sups)
    return (f'<div class="ev-h">{heading} '
            f'<span class="ev-n">{len(sups)} of {ex["k"]}</span></div>'
            f'<div class="sup-list">{chips}</div>')


def _score_of(r: dict):
    """The score to meter. A reinstated link has no entry in the drawn edge
    weights - it was filtered out before the graph was built - so its score
    lives on the vote that rejected it."""
    if r.get("weight") is not None:
        return r["weight"]
    vote = r.get("vote") or {}
    return vote.get("score")


def _recycle_note(r: dict) -> str:
    if r.get("kind") != "recycle":
        return ""
    return ('<p class="ev-say ev-rcy">Drawn as a recycle: material returns to '
            'a block earlier in the circuit.</p>')


def _edge_body(r: dict, ex: dict) -> str:
    """The evidence behind one connection, in the language a process designer
    would use to check it."""
    st, src, dst = r["state"], _pretty(r["src"]), _pretty(r["dst"])
    stranded = _pretty(r["derived_for"]) if r.get("derived_for") else "a block"

    if st == STATE_VOTED:
        say = (f'<p class="ev-say">Analog projects whose own disclosed '
               f'flowsheet runs <b>{_esc(src)}</b> into <b>{_esc(dst)}</b> '
               f'carry a combined vote of <b>{r["weight"]:.2f}</b>, which '
               f'clears the {EDGE_THRESHOLD:g} cut. Weight is similarity, so a '
               f'closer project counts for more.</p>')
        return say + _recycle_note(r) + _supporters_block(
            r, ex, "Analogs that wire it this way")

    if st == STATE_INSERTED:
        return (f'<p class="ev-say ev-warn">No analog proposed this link. The '
                f'drawing rule added it because <b>{_esc(stranded)}</b> would '
                f'otherwise sit unreachable from the feed, and a flowsheet with '
                f'a disconnected block is not a flowsheet. <b>This arrow is not '
                f'a model output</b> &mdash; treat the sequence as a drawing '
                f'convention, not a recommendation.</p>')

    if st == STATE_REINSTATED:
        sups = (r.get("vote") or {}).get("supporting_neighbors") or []
        n = len(sups)
        say = (f'<p class="ev-say ev-warn">{n} analog'
               f'{"s" if n != 1 else ""} did propose this link, but their '
               f'combined weight was <b>{r["vote"]["score"]:.3f}</b> against '
               f'the {EDGE_THRESHOLD:g} cut, so the vote turned it down. The '
               f'drawing rule then added the same link anyway, because '
               f'<b>{_esc(stranded)}</b> was otherwise unreachable from the '
               f'feed. <b>The arrow you see is the drawing rule&rsquo;s, not '
               f'the model&rsquo;s</b> &mdash; though the analogs it overrode '
               f'agreed with it.</p>')
        return say + _supporters_block(r, ex, "Analogs that proposed it anyway")

    if st == STATE_UNDRAWN:
        why = {
            "self-loop": (
                "the renderer drops self-loops. In the source reports this "
                "pattern is almost always two different units of the same type "
                "in series &mdash; a rougher feeding a scavenger, or one "
                "cleaner stage feeding the next &mdash; collapsed into a single "
                "stage name by the shared vocabulary. Roughly one raw "
                "connection in five hundred is a true self-loop; the rest of "
                "this signal is real circuit detail the vocabulary cannot "
                "express"),
            "into the feed": (
                "nothing may flow back into the feed node, which represents "
                "run-of-mine ore entering the plant"),
            "cannot terminate": (
                "this block may not discharge straight to product; the "
                "renderer requires a dewatering or recovery step first"),
        }.get(r["reason"], "the renderer did not draw it")
        say = (f'<p class="ev-say ev-warn">The vote <b>selected</b> this link '
               f'at <b>{r["weight"]:.2f}</b> and the flowsheet above does not '
               f'show it: {why}.</p>'
               f'<p class="ev-say">The picture and the vote disagree here. The '
               f'vote is the model&rsquo;s output; the omission is the '
               f'renderer&rsquo;s rule.</p>')
        return say + _supporters_block(r, ex, "Analogs that wire it this way")

    say = (f'<p class="ev-say">Considered and turned down. Analogs backing '
           f'<b>{_esc(src)}</b> into <b>{_esc(dst)}</b> carry a combined vote '
           f'of <b>{r["weight"]:.2f}</b>, short of the {EDGE_THRESHOLD:g} cut, '
           f'so the link was not selected.</p>')
    return say + _supporters_block(r, ex, "Analogs that wire it this way")


_STATE_BADGE = {
    STATE_VOTED: ("badge-sel", "voted",
                  "analog vote share cleared the connection cut"),
    STATE_INSERTED: ("badge-derived", "drawing rule",
                     "not a model output - the renderer inserted this link to "
                     "reach a block the votes left unreachable"),
    STATE_REINSTATED: ("badge-reinst", "rule overrode vote",
                       "the vote rejected this link and the renderer drew it "
                       "anyway to reach an otherwise stranded block"),
    STATE_UNDRAWN: ("badge-undrawn", "selected, not drawn",
                    "the vote selected this link but the renderer refused to "
                    "draw it"),
    STATE_REJECTED: ("badge-cut", "below cut",
                     "scored under the connection cut, so not selected"),
}


def _edge_row(r: dict, ex: dict, scale: float) -> str:
    """One connection as an expandable row. <details> keeps the page readable
    while every link stays one click from its full evidence, and needs no
    script and no generated CSS."""
    cls, label, tip = _STATE_BADGE[r["state"]]
    num = (f'<span class="rn" title="arrow {r["idx"]} on the flowsheet">'
           f'{r["idx"]}</span>' if r.get("idx") else
           '<span class="rn rn-off" title="not on the flowsheet">&mdash;</span>')
    name = (f'{num}{_esc(_pretty(r["src"]))} <span class="arr">&rarr;</span> '
            f'{_esc(_pretty(r["dst"]))}')
    kind = ('<span class="chip chip-kind" title="returns to an earlier block">'
            'recycle</span>' if r.get("kind") == "recycle" else '')

    if r["state"] == STATE_INSERTED:
        meter = ('<span class="noprob" title="no analog proposed this link">'
                 'added to reach a stranded block</span>')
        val = "&mdash;"
    else:
        # A rejected or overridden link is metered against the same cut, but
        # never in the selected colour - the fill states plainly whether the
        # model chose it.
        chose = r["state"] in (STATE_VOTED, STATE_UNDRAWN)
        score = _score_of(r)
        meter = _meter(score, EDGE_THRESHOLD, chose, scale)
        val = f'{score:.2f}'

    return (
        f'<details class="row-d">'
        f'<summary class="row">'
        f'<div class="row-name">{name}</div>'
        f'<div class="row-meter">{meter}</div>'
        f'<div class="row-val">{val}</div>'
        f'<div class="row-state"><span class="badge {cls}" '
        f'title="{_esc(tip)}">{label}</span></div>'
        f'<div class="row-support">{kind}</div>'
        f'</summary>'
        f'<div class="ebody">{_edge_body(r, ex)}</div>'
        f'</details>')


def _edge_table(rows: list, ex: dict, scale: float, head: str) -> str:
    body = "".join(_edge_row(r, ex, scale) for r in rows)
    return (
        f'<div class="rows rows-head"><div class="row-name">{head}</div>'
        f'<div class="row-meter">vote share</div><div class="row-val"></div>'
        f'<div class="row-state"></div><div class="row-support"></div></div>'
        f'<div class="rows rows-edge">{body}</div>')


def _connections(ex: dict, rows: list) -> str:
    """The arrow-by-arrow account of the flowsheet, plus the links that never
    reached it. Without the second table the page shows only the connections
    that survived, which reads as though nothing else was considered."""
    if not rows:
        return ""
    considered = _considered_rows(ex, rows)
    weights = [w for w in (_score_of(r) for r in rows + considered) if w]
    scale = max(weights + [EDGE_THRESHOLD * 1.2])
    rule = sum(1 for r in rows if r["derived"])
    note = (f' &middot; {rule} of {len(rows)} added by the drawing rule, not '
            f'predicted' if rule else
            ' &middot; every arrow carries an analog vote')
    out = (
        f'<h3 class="sec">Connection decisions'
        f'<span class="sec-note">every arrow on the flowsheet above, numbered '
        f'to match it &mdash; vote share vs the {EDGE_THRESHOLD:g} cut{note} '
        f'&middot; open a row for the evidence behind it</span></h3>'
        f'{_edge_table(rows, ex, scale, "connection")}')
    if considered:
        undrawn = sum(1 for r in considered if r["state"] == STATE_UNDRAWN)
        extra = (f' &middot; {undrawn} of them the model selected and the '
                 f'renderer refused to draw' if undrawn else '')
        out += (
            f'<h3 class="sec">Considered and not drawn'
            f'<span class="sec-note">{len(considered)} further links the model '
            f'scored for this project{extra}</span></h3>'
            f'{_edge_table(considered, ex, scale, "link")}')
    return out


def _route_summary(ex: dict, n_drawn: int) -> str:
    sel = [sv["stage"] for sv in ex["stage_votes"] if sv["selected"]]
    return (f'{len(sel)} stages &middot; {n_drawn} connections drawn &middot; '
            f'from k={ex["k"]} analog projects &middot; {_provenance_summary(ex)}')


def _panel(ex: dict, idx: int) -> str:
    max_w = max((nb["similarity_weight"] for nb in ex["neighbors"]), default=1.0)
    max_p = max((sv["probability"] for sv in ex["stage_votes"]), default=1.0)
    stage_scale = max(max_p, STAGE_THRESHOLD * 1.2)

    edges = _connection_rows(ex)
    analogs = "".join(_analog_card(nb, max_w) for nb in ex["neighbors"])
    stages = "".join(_stage_row(sv, stage_scale) for sv in ex["stage_votes"])

    return (
        f'<section class="panel" id="panel-{idx}">'
        f'<div class="panel-head">'
        f'<div class="panel-summary">{_route_summary(ex, len(edges))}</div>'
        f'<div class="panel-prov"><span class="k">source report</span>'
        f'{_provenance_line(ex.get("provenance"))}</div></div>'
        f'{_hero(ex)}'
        f'<h3 class="sec">Nearest analog projects'
        f'<span class="sec-note">the real operations this design was built from '
        f'&mdash; shorter distance, heavier vote &middot; each carries the year '
        f'of its disclosure, and its study grade where the report states '
        f'one</span></h3>'
        f'<div class="analogs">{analogs}</div>'
        f'<h3 class="sec">Stage decisions'
        f'<span class="sec-note">share of analog vote vs the {STAGE_THRESHOLD:g} '
        f'cut &middot; which analogs backed each block</span></h3>'
        f'<div class="rows rows-head"><div class="row-name">stage</div>'
        f'<div class="row-meter">vote share</div><div class="row-val"></div>'
        f'<div class="row-state"></div><div class="row-support">'
        f'supporting analogs (weight)</div></div>'
        f'<div class="rows">{stages}</div>'
        f'{_connections(ex, edges)}'
        f'</section>')


def render(traces: dict) -> str:
    examples = traces["examples"]
    radios = "".join(
        f'<input type="radio" name="ex" id="tab-{i}" class="tabradio"'
        f'{" checked" if i == 0 else ""}>' for i in range(len(examples)))
    tabs = "".join(
        f'<label for="tab-{i}" class="tab">'
        f'<span class="tab-n">0{i + 1}</span>{_esc(ex["query_doc_id"])}</label>'
        for i, ex in enumerate(examples))
    panels = "".join(_panel(ex, i) for i, ex in enumerate(examples))
    return f"""{_STYLE}
<div class="wrap">
  <header class="masthead">
    <div class="eyebrow">Flowsheet synthesizer &middot; decision trace</div>
    <h1>How the model designs a flowsheet</h1>
    <p class="lede">For a real deposit, the synthesizer finds its nearest analogs
    among disclosed operations and lets them vote on every process block and
    connection. Below is the flowsheet it produced &mdash; and, underneath,
    the analog projects and votes that built it. Nothing is re-scored; this is
    the model's own arithmetic, drawn.</p>
  </header>
  {radios}
  <nav class="tabs">{tabs}</nav>
  <main class="panels">{panels}</main>
  <footer class="foot">
    <span>Flowsheet uses the project's PFS topology, drawn natively. Arrows are
    the model's voted connections plus, where those left a selected block
    unreachable from the feed, a link the drawing rule inserted &mdash; the
    connection table on each panel says which is which.</span>
    <span>Year is each report's stated effective date, or its SEDAR filing year
    where the report states none &mdash; hover a year to see which.
    <b>FS</b> feasibility study &middot; <b>PFS</b> pre-feasibility study,
    shown only where the report names that grade in its own title or summary.
    <b>other grade</b> means the report states a grade below that pair;
    <b>not stated</b> means it names none at all, which is not evidence that
    the study was less rigorous.</span>
    <span>Analogs pseudonymized (A&ndash;E) from disclosed NI&nbsp;43-101
    operations &middot; k=5 &middot; stage cut {STAGE_THRESHOLD:g} &middot;
    connection cut {EDGE_THRESHOLD:g}.</span>
    <span>All values re-derive the model's prediction exactly; this view changes
    no decision.</span>
  </footer>
</div>"""


_STYLE = """<style>
:root{
  --bg:#f8fafc; --panel:#ffffff; --panel-2:#f1f5f9; --ink:#0f172a;
  --ink-2:#64748b; --line:#e2e8f0; --accent:#12bfa5; --accent-2:#0d9488;
  --accent-soft:rgba(18,191,165,.10); --sel:#12bfa5; --sel-soft:rgba(18,191,165,.13);
  --cut:#94a3b8; --cut-soft:rgba(148,163,184,.14); --track:#eef2f6;
  --shadow:0 1px 2px rgba(15,23,42,.04),0 4px 14px rgba(15,23,42,.06);
  --node-bg:#ffffff; --node-bd:#e2e8f0; --edge:#94a3b8; --edge-rcy:#f59e0b;
  --ph0:#64748b; --ph1:#3b82f6; --ph2:#f59e0b; --ph3:#06b6d4; --ph4:#8b5cf6; --ph5:#e11d48;
}
*{box-sizing:border-box}
.wrap{
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code","Segoe UI Mono",Menlo,Consolas,monospace;
  font-family:var(--sans); color:var(--ink); background:var(--bg);
  max-width:1120px; margin:0 auto; padding:46px 30px 60px;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}
.masthead{margin-bottom:8px}
.eyebrow{font-size:12px; letter-spacing:.15em; text-transform:uppercase;
  color:var(--accent); font-weight:650; margin-bottom:12px}
h1{font-size:clamp(26px,3.6vw,36px); line-height:1.08; margin:0 0 14px;
  letter-spacing:-.02em; text-wrap:balance; font-weight:680}
.lede{font-size:16px; line-height:1.6; color:var(--ink-2); max-width:66ch; margin:0}
.tabradio{position:absolute; opacity:0; pointer-events:none}
.tabs{display:flex; flex-wrap:wrap; gap:6px; margin:30px 0 0;
  background:var(--panel-2); border:1px solid var(--line); border-radius:13px; padding:6px}
.tab{flex:1 1 200px; display:flex; align-items:center; gap:9px;
  padding:11px 15px; border-radius:9px; font-size:14px; font-weight:570;
  color:var(--ink-2); cursor:pointer; white-space:nowrap;
  transition:background .16s,color .16s}
.tab-n{font-family:var(--mono); font-size:11px; opacity:.6; font-weight:600}
.tab:hover{color:var(--ink)}
.panel{display:none; margin-top:24px}
#tab-0:checked~.tabs label[for="tab-0"],#tab-1:checked~.tabs label[for="tab-1"],
#tab-2:checked~.tabs label[for="tab-2"],#tab-3:checked~.tabs label[for="tab-3"],
#tab-4:checked~.tabs label[for="tab-4"]{background:var(--panel); color:var(--ink);
  box-shadow:var(--shadow); font-weight:640}
#tab-0:checked~.tabs label[for="tab-0"] .tab-n,#tab-1:checked~.tabs label[for="tab-1"] .tab-n,
#tab-2:checked~.tabs label[for="tab-2"] .tab-n,#tab-3:checked~.tabs label[for="tab-3"] .tab-n,
#tab-4:checked~.tabs label[for="tab-4"] .tab-n{color:var(--accent); opacity:1}
#tab-0:checked~.panels #panel-0,#tab-1:checked~.panels #panel-1,
#tab-2:checked~.panels #panel-2,#tab-3:checked~.panels #panel-3,
#tab-4:checked~.panels #panel-4{display:block}
.tabradio:focus-visible~.tabs label.tab{outline:2px solid var(--accent-2); outline-offset:2px}
.panel-head{display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px;
  margin-bottom:18px}
.panel-summary{font-family:var(--mono); font-size:12.5px; color:var(--ink-2);
  padding:9px 14px; background:var(--accent-soft); border-radius:8px;
  display:inline-block; letter-spacing:.01em}
.panel-prov{display:flex; align-items:center; gap:8px}
.panel-prov .k{font-size:10px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--ink-2)}
.panel-prov .prov{margin-top:0}
/* Hero flowsheet (native SVG) */
.hero-card{background:var(--panel); border:1px solid var(--line); border-radius:18px;
  padding:22px 22px 18px; box-shadow:var(--shadow); position:relative; overflow:hidden}
.hero-card::before{content:""; position:absolute; inset:0 0 auto 0; height:120px;
  background:radial-gradient(120% 100% at 15% 0%,var(--accent-soft),transparent 70%);
  pointer-events:none}
.fs-scroll{overflow-x:auto; overflow-y:hidden; position:relative}
.flowsheet{display:block; margin:0 auto; max-width:100%; height:auto}
.n .card{fill:var(--node-bg); stroke:var(--node-bd); stroke-width:1.2;
  filter:drop-shadow(0 2px 4px rgba(15,20,28,.10))}
.n .gl{stroke:currentColor; stroke-width:1.7; fill:none;
  stroke-linecap:round; stroke-linejoin:round}
.n .lbl{fill:var(--ink); font-family:var(--sans); font-size:12.5px;
  font-weight:600; text-anchor:middle; letter-spacing:-.01em}
.e.fwd{stroke:var(--edge)} .e.rcy{stroke:var(--edge-rcy); stroke-dasharray:5 4}
.ahp{fill:var(--edge)} .ahrp{fill:var(--edge-rcy)}
.enum circle{fill:var(--ink); opacity:.86}
.enum text{fill:var(--panel); font-family:var(--mono); font-size:9.5px;
  font-weight:600; text-anchor:middle}
.n.ph0{color:var(--ph0)} .n.ph1{color:var(--ph1)} .n.ph2{color:var(--ph2)}
.n.ph3{color:var(--ph3)} .n.ph4{color:var(--ph4)} .n.ph5{color:var(--ph5)}
.n.ph0 .accent{fill:var(--ph0)} .n.ph1 .accent{fill:var(--ph1)}
.n.ph2 .accent{fill:var(--ph2)} .n.ph3 .accent{fill:var(--ph3)}
.n.ph4 .accent{fill:var(--ph4)} .n.ph5 .accent{fill:var(--ph5)}
.legend{display:flex; flex-wrap:wrap; align-items:center; gap:8px 15px;
  margin-top:16px; font-size:12px; color:var(--ink-2)}
.legend .lg{display:inline-flex; align-items:center; gap:6px}
.legend .dot{width:9px; height:9px; border-radius:3px; display:inline-block}
.legend .ln{width:18px; height:0; display:inline-block}
.legend .ln.fwd{border-top:2px solid var(--edge)}
.legend .ln.rcy{border-top:2px dashed var(--edge-rcy)}
.legend .dot.ph0{background:var(--ph0)} .legend .dot.ph1{background:var(--ph1)}
.legend .dot.ph2{background:var(--ph2)} .legend .dot.ph3{background:var(--ph3)}
.legend .dot.ph4{background:var(--ph4)} .legend .dot.ph5{background:var(--ph5)}
.lg-sep{width:1px; height:13px; background:var(--line)}
.hero-cap{font-size:12.5px; color:var(--ink-2); line-height:1.55; margin-top:14px;
  padding-top:13px; border-top:1px solid var(--line); max-width:80ch}
/* Evidence */
.sec{font-size:15px; font-weight:660; margin:34px 0 15px; letter-spacing:-.01em;
  display:flex; flex-wrap:wrap; align-items:baseline; gap:10px;
  padding-bottom:10px; border-bottom:1px solid var(--line)}
.sec-note{font-size:12.5px; font-weight:440; color:var(--ink-2); letter-spacing:0}
.analogs{display:grid; grid-template-columns:repeat(auto-fit,minmax(196px,1fr)); gap:12px}
.analog{background:var(--panel); border:1px solid var(--line); border-radius:13px;
  padding:15px; box-shadow:var(--shadow); display:flex; flex-direction:column; gap:10px}
.analog-head{display:flex; align-items:center; justify-content:space-between; gap:8px}
.analog-name{font-weight:640; font-size:14px}
.rank{font-family:var(--mono); font-size:11px; color:var(--ink-2);
  background:var(--panel-2); border:1px solid var(--line); border-radius:6px; padding:1px 6px}
.analog-metrics{display:flex; gap:18px}
.metric{display:flex; flex-direction:column; gap:1px}
.metric .k{font-size:10px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-2)}
.metric .v{font-family:var(--mono); font-size:17px; font-weight:600; font-variant-numeric:tabular-nums}
.prov{display:flex; align-items:center; gap:8px; margin-top:-4px}
.prov-year{font-family:var(--mono); font-size:12.5px; font-weight:620;
  color:var(--ink); font-variant-numeric:tabular-nums; cursor:help}
.chip-study{font-size:10.5px; font-weight:650; letter-spacing:.05em;
  padding:1px 7px; border:1px solid currentColor; background:transparent; cursor:help}
.st-fs{color:var(--accent-2)} .st-pfs{color:var(--ph1)}
.st-none{color:var(--cut); border-style:dashed; font-weight:560; letter-spacing:.02em}
.st-other{color:var(--ph2); border-style:dashed; font-weight:560; letter-spacing:.02em}
.chips{display:flex; flex-wrap:wrap; gap:5px}
.chip{font-size:11.5px; padding:2px 8px; border-radius:20px; line-height:1.5; white-space:nowrap}
.chip-agree{background:var(--accent-soft); color:var(--accent)}
.chip-more{background:var(--panel-2); color:var(--ink-2); border:1px solid var(--line)}
.chip-sup{background:var(--sel-soft); color:var(--sel); font-family:var(--mono);
  font-size:11px; display:inline-flex; gap:5px; align-items:center}
.chip-sup .w{opacity:.7; font-size:10px}
.differ{font-size:11.5px; color:var(--ink-2); line-height:1.45}
.rows{display:flex; flex-direction:column; gap:2px}
.row,.rows-head,.row-d>summary.row{display:grid; align-items:center; gap:12px;
  grid-template-columns:150px 1fr 44px 88px minmax(150px,1.3fr);
  padding:8px 10px; border-radius:8px}
.row:hover{background:var(--panel-2)}
.rows-head{font-size:11px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--ink-2); padding-bottom:4px}
.rows-head:hover{background:none}
.row-name{font-weight:560; font-size:13.5px; word-break:break-word}
.row-val{font-family:var(--mono); font-size:13px; text-align:right;
  font-variant-numeric:tabular-nums; color:var(--ink-2)}
.meter{position:relative; height:9px; background:var(--track); border-radius:6px; overflow:hidden}
.meter-fill{position:absolute; top:0; left:0; height:100%; border-radius:6px}
.fill-sel{background:var(--sel)} .fill-cut{background:var(--cut)}
.meter-thresh{position:absolute; top:-2px; width:2px; height:13px;
  background:var(--ink); opacity:.5; border-radius:2px}
.badge{font-size:11px; font-weight:600; padding:2px 9px; border-radius:20px; white-space:nowrap}
.badge-sel{background:var(--sel-soft); color:var(--sel)}
.badge-cut{background:var(--cut-soft); color:var(--cut)}
.badge-derived{background:var(--panel-2); color:var(--ink-2);
  border:1px dashed var(--line)}
.badge-reinst{background:rgba(245,158,11,.16); color:#b45309}
.badge-undrawn{background:rgba(99,102,241,.13); color:#4f46e5}
/* Expandable evidence. details/summary is native, script-free and keyboard
   accessible, so a reader can open any connection without the page carrying a
   pre-rendered card for every one of them. */
.row-d{border-radius:8px}
.row-d+.row-d{margin-top:2px}
.row-d>summary{cursor:pointer; list-style:none}
.row-d>summary::-webkit-details-marker{display:none}
.row-d>summary:hover{background:var(--panel-2)}
.row-d[open]>summary{background:var(--panel-2)}
.row-d>summary:focus-visible{outline:2px solid var(--accent-2); outline-offset:1px}
.rn{display:inline-flex; align-items:center; justify-content:center;
  min-width:17px; height:17px; margin-right:7px; padding:0 4px; border-radius:5px;
  background:var(--ink); color:var(--panel); font-family:var(--mono);
  font-size:10px; font-weight:600; vertical-align:1px}
.rn-off{background:transparent; color:var(--cut); border:1px dashed var(--line)}
.ebody{padding:12px 14px 15px 40px; border-left:2px solid var(--accent-2);
  margin:2px 0 6px 10px; background:var(--panel-2); border-radius:0 8px 8px 0}
.ev-say{font-size:12.5px; line-height:1.62; color:var(--ink-2); margin:0 0 9px;
  max-width:82ch}
.ev-say b{color:var(--ink); font-weight:600}
.ev-warn{border-left:3px solid #f59e0b; padding-left:10px}
.ev-rcy{color:var(--ink-2); font-style:italic}
.ev-h{font-size:10.5px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--ink-2); margin:12px 0 7px; font-weight:600}
.ev-n{font-family:var(--mono); text-transform:none; letter-spacing:0; opacity:.7}
.sup-list{display:flex; flex-direction:column; gap:4px; max-width:60ch}
.sup-rich{display:flex; align-items:baseline; gap:9px; padding:5px 9px;
  border-radius:7px; background:var(--panel); border:1px solid var(--line);
  font-size:12px}
.sup-rich b{font-weight:600; color:var(--ink)}
.sup-meta{color:var(--ink-2); font-size:11px; font-family:var(--mono);
  flex:1; min-width:0}
.sup-w{font-family:var(--mono); font-size:11.5px; color:var(--sel);
  font-weight:600}
.row-support{display:flex; flex-wrap:wrap; gap:4px}
.rows-edge .row{grid-template-columns:210px 1fr 44px 108px minmax(120px,1fr)}
.arr{color:var(--ink-2); font-weight:400; padding:0 2px}
.chip-kind{background:rgba(245,158,11,.12); color:#b45309; font-size:10.5px}
.noprob{font-size:11.5px; color:var(--cut); font-style:italic; cursor:help}
.foot{margin-top:46px; padding-top:18px; border-top:1px solid var(--line);
  display:flex; flex-direction:column; gap:5px; font-size:12px; color:var(--ink-2); line-height:1.5}
@media (max-width:720px){
  .row,.rows-head,.rows-edge .row{grid-template-columns:1fr 1fr; gap:6px 10px}
  .rows-head{display:none}
  .row-meter{grid-column:1 / -1; order:5}
  .row-support{grid-column:1 / -1; order:6}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", default="quantitative/demo/traces.json")
    ap.add_argument("--out", default="quantitative/demo/knn_trace_demo.html")
    args = ap.parse_args()
    traces = json.loads(Path(args.traces).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(traces), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB, "
          f"{len(traces['examples'])} examples)")


if __name__ == "__main__":
    main()
