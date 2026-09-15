"""M2 metallurgy regression.

Two gates per plant:
  1. Regression: model R_Cu within +/-0.5pp of the captured baseline. Catches
     any drift in the recovery cascade.
  2. M2 metric: model R_Cu within +/-5pp of the *disclosed* plant recovery.
     This is the M2 success threshold; Mantoverde currently runs at 4.74pp so
     the margin is tight.

Baseline captured 2026-06-23.
"""
import pytest

from process_model.benchmarks import run_m2_metallurgy


# (plant_name, baseline_R_model_frac)
PINNED_BASELINE_R = {
    "Cobre Panama":             0.8246,
    "Escondida concentrator":   0.8246,
    "Cerro Verde":              0.8246,
    "El Abra POX":              0.9469,
    "Sherritt Bagdad pilot":    0.9469,
    "Morenci POX (Freeport)":   0.9469,
    "Spence heap":              0.7177,
    "Lomas Bayas heap":         0.7177,
    "Mantoverde heap":          0.7974,
}

BASELINE_TOL_PP = 0.5   # +/-0.5pp on the regression assertion
M2_METRIC_TOL_PP = 5.0  # +/-5pp on the disclosed-recovery assertion


@pytest.fixture(scope="module")
def m2():
    return run_m2_metallurgy(verbose=False)


@pytest.mark.parametrize("plant,baseline_R", list(PINNED_BASELINE_R.items()))
def test_R_model_matches_baseline(m2, plant, baseline_R):
    row = next((r for r in m2["rows"] if r["plant"] == plant), None)
    assert row is not None, f"plant {plant!r} missing from M2 output"
    drift_pp = abs(row["R_model"] - baseline_R) * 100.0
    assert drift_pp <= BASELINE_TOL_PP, (
        f"{plant}: R_model = {row['R_model']*100:.2f}% drifted "
        f"{drift_pp:.2f}pp from baseline {baseline_R*100:.2f}% "
        f"(threshold {BASELINE_TOL_PP}pp). Update PINNED_BASELINE_R if "
        f"intentional.")


@pytest.mark.parametrize("plant", list(PINNED_BASELINE_R.keys()))
def test_R_model_within_M2_threshold_of_disclosed(m2, plant):
    row = next((r for r in m2["rows"] if r["plant"] == plant), None)
    assert row is not None
    assert row["dR_pp"] <= M2_METRIC_TOL_PP, (
        f"{plant}: |R_model - R_plant| = {row['dR_pp']:.2f}pp exceeds the "
        f"M2 threshold of {M2_METRIC_TOL_PP}pp")


def test_median_dR_under_threshold(m2):
    assert m2["pass_R"], (
        f"M2 median |dR| = {m2['median_dR_pp']:.2f}pp exceeds 5pp threshold")
