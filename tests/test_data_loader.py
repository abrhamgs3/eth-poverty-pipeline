import numpy as np
import pandas as pd

from eth_poverty_pipeline.data_extraction.loader import _assign_relative_poverty, load_poverty_panel


def test_relative_poverty_is_assigned_within_each_wave_separately():
    # Wave 1 consumption: [10, 20, 30, 40] -> bottom third (<20) is poor.
    # Wave 2 consumption: [100, 200, 300, 400] -> bottom third (<200) is poor.
    # A wave-1 household with consumption 40 must not be marked poor just
    # because it's below wave 2's (much larger) values.
    df = pd.DataFrame({
        "wave": [1, 1, 1, 1, 2, 2, 2, 2],
        "nom_totcons_aeq": [10, 20, 30, 40, 100, 200, 300, 400],
    })

    out = _assign_relative_poverty(df, poverty_quantile=1 / 3)

    assert out.loc[out["wave"] == 1, "D_true"].tolist() == [1, 0, 0, 0]
    assert out.loc[out["wave"] == 2, "D_true"].tolist() == [1, 0, 0, 0]


def test_falls_back_to_synthetic_panel_when_real_path_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.parquet")

    df = load_poverty_panel(missing_path, poverty_quantile=1 / 3, rng=np.random.default_rng(0))

    assert len(df) > 0
    assert set(df["D_true"].unique()) <= {0, 1}
    assert df["nom_totcons_aeq"].notna().all()
    assert df["mean_haz"].notna().all()
