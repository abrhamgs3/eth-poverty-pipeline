import numpy as np
import pandas as pd

from eth_poverty_pipeline.annotation.annotator import annotate_batch, build_household_profile
from eth_poverty_pipeline.annotation.clients import MockLLMClient


def test_build_household_profile_mentions_key_covariates():
    row = pd.Series({
        "n_children": 2, "dist_market": 45.0, "srtm": 1900.0,
        "rain_annual_anomaly": -120.0, "drought": 1, "flood": 0,
    })

    note = build_household_profile(row)

    assert "2 child(ren)" in note
    assert "45 km" in note
    assert "1900m" in note
    assert "below" in note  # negative rain anomaly
    assert "drought" in note.lower()


def test_build_household_profile_never_leaks_consumption_or_true_label():
    # The note is generated from covariates only - if it ever included
    # nom_totcons_aeq or D_true directly, the annotation task would be
    # circular (the "classifier" could just read off the answer).
    row = pd.Series({
        "n_children": 1, "dist_market": 10.0, "srtm": 1000.0,
        "rain_annual_anomaly": 0.0, "drought": 0, "flood": 0,
        "nom_totcons_aeq": 99999.0, "D_true": 1,
    })

    note = build_household_profile(row)

    assert "99999" not in note
    assert "poor" not in note.lower()


def test_annotate_batch_returns_one_row_per_input_with_expected_columns():
    df = pd.DataFrame({
        "n_children": [1, 2], "dist_market": [10.0, 90.0], "srtm": [1000.0, 2000.0],
        "rain_annual_anomaly": [0.0, -50.0], "drought": [0, 1], "flood": [0, 0],
        "D_true": [0, 1],
    })
    client = MockLLMClient(accuracy=1.0, rng=np.random.default_rng(0))

    result = annotate_batch(df, client)

    assert len(result) == 2
    assert set(result["predicted_poverty_status"]) <= {"poor", "non_poor"}
    # accuracy=1.0 -> the mock never flips the label
    assert result.loc[0, "predicted_poverty_status"] == "non_poor"
    assert result.loc[1, "predicted_poverty_status"] == "poor"
