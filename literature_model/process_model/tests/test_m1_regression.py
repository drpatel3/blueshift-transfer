"""M1 route-selection regression.

Pins the current (project -> predicted route) map for all 10 M1 plants. The
model is allowed to be wrong vs disclosed (today: 9/10, El Abra Phelps Dodge
predicts heap when actual is POX) but the *predictions* must not silently
change without a deliberate test update.

Baseline captured 2026-06-23.
"""
import pytest

from process_model.benchmarks import run_m1_route_selection


# Pinned current predictions, NOT the disclosed routes. Update intentionally
# when the model improves (e.g. when El Abra flips to POX).
PINNED_PREDICTIONS = {
    "Cobre Panama (FQM)":           "sulfide",
    "Cerro Verde (Freeport)":       "sulfide",
    "Quebrada Blanca QB2 (Teck)":   "sulfide",
    "Cobre Antamina (Glencore)":    "sulfide",
    "BHP Spence":                   "heap",
    "Lomas Bayas (Glencore)":       "heap",
    "Mantoverde (Capstone)":        "heap",
    "Codelco Radomiro Tomic":       "heap",
    "Phelps Dodge El Abra":         "heap",     # disclosed POX; known mismatch
    "Olympic Dam (BHP)":            "sulfide",
}

EXPECTED_MATCH_COUNT = 9   # 10/10 disclosed; El Abra is the one miss


@pytest.fixture(scope="module")
def m1():
    return run_m1_route_selection(verbose=False)


@pytest.mark.parametrize("project,expected_predicted",
                         list(PINNED_PREDICTIONS.items()))
def test_route_prediction_pinned(m1, project, expected_predicted):
    row = next((r for r in m1["rows"] if r["project"] == project), None)
    assert row is not None, f"project {project!r} missing from M1 output"
    assert row["predicted"] == expected_predicted, (
        f"{project}: predicted {row['predicted']!r}, pinned "
        f"{expected_predicted!r}. Either the model changed (update the pin "
        f"intentionally) or a regression slipped in.")


def test_overall_match_count_held(m1):
    assert m1["matches"] == EXPECTED_MATCH_COUNT, (
        f"M1 score changed: {m1['matches']}/{m1['total']} "
        f"(pinned {EXPECTED_MATCH_COUNT}/{m1['total']}). Update "
        f"EXPECTED_MATCH_COUNT if this is an intentional improvement.")
