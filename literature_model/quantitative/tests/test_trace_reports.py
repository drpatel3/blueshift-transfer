"""The dossiers must not claim more design than the model produced.

Both trace reports draw a flowsheet, count its connections in the panel header
and itemise them in a table. Those three numbers came apart twice before: the
XGBoost header once reported candidate transitions from the export rather than
arrows on the page, and the KNN header reported edge votes rather than arrows.
The renderer's orphan-reconnection rule is what drives them apart — it inserts
links the model never proposed, and on a sparse prediction those outnumber the
predicted ones. These tests pin the invariant against the frozen demo traces.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from quantitative import flowsheet_svg, trace_report, xgb_trace_report

DEMO = Path(__file__).resolve().parents[1] / "demo"


def _load(name):
    path = DEMO / name
    if not path.exists():  # frozen traces are the fixture; skip if absent
        pytest.skip(f"{path} not present")
    return json.loads(path.read_text())["examples"]


KNN = [(ex, trace_report) for ex in _load("traces.json")]
XGB = [(ex, xgb_trace_report) for ex in _load("xgb_traces.json")]
ALL = KNN + XGB
IDS = ([f"knn-{i}" for i in range(len(KNN))]
       + [f"xgb-{i}" for i in range(len(XGB))])


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_every_drawn_arrow_is_itemised(ex, mod):
    """One table row per arrow on the page — no arrow goes unaccounted for."""
    svg = flowsheet_svg.render_svg(*mod._graph_inputs(ex))
    rows = mod._connection_rows(ex)
    assert len(rows) == svg.count('class="e ')


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_panel_header_counts_arrows_not_candidates(ex, mod):
    """The header number is the picture's, not the export's edge list."""
    panel = mod._panel(ex, 0)
    stated = int(re.search(r"(\d+) connections drawn", panel).group(1))
    assert stated == panel.count('class="e ') == len(mod._connection_rows(ex))


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_reconnection_arrows_are_flagged_not_scored(ex, mod):
    """An arrow the drawing rule inserted carries no score and says so."""
    rows = mod._connection_rows(ex)
    derived = [r for r in rows if r["derived"]]
    assert all(r["weight"] is None for r in derived)
    assert all(r["weight"] is not None for r in rows if not r["derived"])
    # A derived arrow is badged either "drawing rule" (nothing proposed it) or
    # "rule overrode vote" (the model scored it and said no). Both are the
    # renderer's, so together they must account for every derived arrow.
    panel = mod._panel(ex, 0)
    badges = (panel.count(">drawing rule</span>")
              + panel.count(">rule overrode vote</span>"))
    assert badges == len(derived)


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_derived_share_is_stated_when_present(ex, mod):
    """A reader is told how much of the flowsheet the model did not design."""
    rows = mod._connection_rows(ex)
    derived = sum(1 for r in rows if r["derived"])
    section = mod._connections(ex, rows)
    if derived:
        assert f"{derived} of {len(rows)} added by the drawing rule" in section
    else:
        assert "drawing rule" not in section

# --------------------------------------------------------------------------
# Connection states. A link is in exactly one of five states; the table used to
# know two, which is why a link the vote rejected and the renderer reinstated
# read identically to one nothing ever proposed, and a link the vote selected
# and the renderer dropped appeared nowhere at all.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_states_are_exhaustive_and_disjoint(ex, mod):
    """Every scored link lands in exactly one state, and the drawn table is
    exactly the states that are on the picture."""
    drawn = mod._connection_rows(ex)
    considered = mod._considered_rows(ex, drawn)
    assert all(r.get("state") for r in drawn + considered)
    drawn_pairs = {(r["src"], r["dst"]) for r in drawn}
    cons_pairs = {(r["src"], r["dst"]) for r in considered}
    assert not (drawn_pairs & cons_pairs), "a link is both drawn and not drawn"
    # Nothing in the considered table claims a flowsheet number.
    assert all(r["idx"] is None for r in considered)
    assert all(r["idx"] is not None for r in drawn)


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_arrow_numbers_match_the_picture(ex, mod):
    """The row number and the badge on the arrow must be the same number, or
    the cross-reference silently misleads."""
    svg = flowsheet_svg.render_svg(*mod._graph_inputs(ex))
    badges = [int(m) for m in re.findall(
        r'<g class="enum">.*?<text[^>]*>(\d+)</text>', svg)]
    rows = [r["idx"] for r in mod._connection_rows(ex)]
    assert badges == rows == list(range(1, len(rows) + 1))


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_inserted_links_name_the_stranded_block(ex, mod):
    """A drawing-rule arrow must say which block it was inserted to reach.
    Without the name it is an excuse rather than an explanation."""
    for r in mod._connection_rows(ex):
        if r["state"] != trace_report.STATE_INSERTED:
            continue
        assert r["derived_for"], f'{r["src"]}->{r["dst"]} has no stranded block'
        body = mod._edge_body(r) if mod is xgb_trace_report else mod._edge_body(r, ex)
        assert trace_report._pretty(r["derived_for"]) in body
        assert "not a model output" in body or "not evidence of any kind" in body


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_links_the_model_did_not_choose_carry_no_vote_meter(ex, mod):
    """States where the renderer, not the model, put the arrow there must never
    render the selected-colour fill that means 'the model chose this'."""
    rows = mod._connection_rows(ex)
    scale = 1.0
    for r in rows:
        if r["state"] != trace_report.STATE_INSERTED:
            continue
        html = (mod._edge_row(r, scale) if mod is xgb_trace_report
                else mod._edge_row(r, ex, scale))
        assert "fill-sel" not in html
        # the bar itself, not the row-meter cell that always wraps it
        assert 'class="meter"' not in html


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_supported_links_the_renderer_dropped_are_surfaced(ex, mod):
    """A link the model backed and the picture omits is a contradiction; it has
    to appear somewhere on the page with its reason."""
    drawn = mod._connection_rows(ex)
    considered = mod._considered_rows(ex, drawn)
    undrawn = [r for r in considered if r["state"] == trace_report.STATE_UNDRAWN]
    if not undrawn:
        pytest.skip("no selected-but-undrawn links in this example")
    section = mod._connections(ex, drawn)
    for r in undrawn:
        assert r["reason"], f'{r["src"]}->{r["dst"]} undrawn with no reason'
        assert trace_report._pretty(r["src"]) in section


@pytest.mark.parametrize("ex,mod", ALL, ids=IDS)
def test_every_link_is_expandable(ex, mod):
    """One <details> per link, drawn or not — the whole point is that a
    designer can open any of them."""
    drawn = mod._connection_rows(ex)
    section = mod._connections(ex, drawn)
    n = len(drawn) + len(mod._considered_rows(ex, drawn))
    assert section.count("<details class=\"row-d\">") == n
    assert section.count("</details>") == n


# --- KNN-specific ----------------------------------------------------------


@pytest.mark.parametrize("ex", [e for e, _ in KNN],
                         ids=[f"knn-{i}" for i in range(len(KNN))])
def test_reinstated_links_are_not_presented_as_voted(ex):
    """The renderer overriding a rejection is the sharpest finding on the page.
    It must never be badged as though the model chose it."""
    for r in trace_report._connection_rows(ex):
        if r["state"] != trace_report.STATE_REINSTATED:
            continue
        html = trace_report._edge_row(r, ex, 1.0)
        assert ">voted</span>" not in html
        assert ">rule overrode vote</span>" in html
        body = trace_report._edge_body(r, ex)
        assert "turned it down" in body and "unreachable" in body


def test_the_known_reinstatement_cases_are_pinned():
    """crusher->mill (Cu 1.60%) and filter->tank (Cu 0.59%) are the two links
    the vote rejected and the drawing rule drew anyway. If a change makes them
    read as predictions again, that is the regression this suite exists for."""
    found = {}
    for ex, _ in KNN:
        for r in trace_report._connection_rows(ex):
            if r["state"] == trace_report.STATE_REINSTATED:
                found[(r["src"], r["dst"])] = trace_report._score_of(r)
    assert ("crusher", "mill") in found
    assert ("filter", "tank") in found
    assert found[("crusher", "mill")] == pytest.approx(0.181, abs=5e-4)


def test_the_selected_but_undrawn_self_loop_is_pinned():
    """input->input clears the cut at 0.82 and the flowsheet cannot show it.
    Pinned because it is the clearest case of the picture and the vote
    disagreeing."""
    scores = []
    for ex, _ in KNN:
        drawn = trace_report._connection_rows(ex)
        for r in trace_report._considered_rows(ex, drawn):
            if (r["src"], r["dst"]) == ("input", "input"):
                assert r["state"] == trace_report.STATE_UNDRAWN
                assert r["reason"] == "self-loop"
                scores.append(r["weight"])
    assert max(scores) == pytest.approx(0.821, abs=5e-4)


@pytest.mark.parametrize("ex", [e for e, _ in KNN],
                         ids=[f"knn-{i}" for i in range(len(KNN))])
def test_supporters_resolve_to_named_analogs(ex):
    """Every supporter must join to a neighbor record, or the enriched chip
    silently degrades to a bare id."""
    idx = trace_report._neighbor_index(ex)
    for r in trace_report._connection_rows(ex) + trace_report._considered_rows(
            ex, trace_report._connection_rows(ex)):
        for sup in (r.get("vote") or {}).get("supporting_neighbors") or []:
            assert sup["doc_id"] in idx
            chip = trace_report._rich_supporter(sup, idx)
            assert "sup-meta" in chip and sup["doc_id"] in chip


# --- XGBoost-specific ------------------------------------------------------


@pytest.mark.parametrize("ex", [e for e, _ in XGB],
                         ids=[f"xgb-{i}" for i in range(len(XGB))])
def test_every_xgb_edge_body_states_the_ceiling(ex):
    """XGBoost connectivity is a function of the stage set alone. Opening links
    one at a time invites a per-project question the model cannot answer, so
    every body must say so — not just a footnote once per panel."""
    rows = xgb_trace_report._connection_rows(ex)
    considered = xgb_trace_report._considered_rows(ex, rows)
    for r in rows + considered:
        body = xgb_trace_report._edge_body(r)
        if r["state"] == trace_report.STATE_INSERTED:
            assert "not evidence of any kind" in body
        else:
            assert "not a prediction about this project" in body
            assert "no learned edge model" in body


# --- the hard constraint ---------------------------------------------------


@pytest.mark.parametrize("mod,name", [(trace_report, "traces.json"),
                                      (xgb_trace_report, "xgb_traces.json")])
def test_dossiers_stay_script_free(mod, name):
    """Self-contained and script-free is the stated design property
    (trace_report.py docstring, flowsheet_svg.py docstring). Nothing pinned it
    until now, so a single convenience <script> could have removed it
    silently."""
    traces = json.loads((DEMO / name).read_text())
    html = mod.render(traces)
    assert "<script" not in html.lower()
    assert 'href="http' not in html and "src=" not in html
    assert not re.search(r'\son[a-z]+\s*=', html)
