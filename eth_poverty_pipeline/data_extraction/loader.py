"""Stage 0: data extraction - the real Ethiopia ESS/ERSS household panel.

Reads the household-wave panel already built by the sibling ``../DHS``
portfolio project (``DHS/src/ess_panel.py``): real World Bank Ethiopia
Socioeconomic Survey (LSMS-ISA) data, waves 2011/2013/2015, with real
consumption aggregates and real child-health (HAZ) outcomes.

That data is never bundled here. It's public-use but requires a free World
Bank account (see ``DHS/docs/wb_lsms_data_checklist.md``), and the DHS repo
itself gitignores it per those terms - this pipeline follows the same
convention: read it from disk if the sibling repo has it, otherwise fall
back to a synthetic panel with a matching schema and roughly matching
distributions, so the pipeline always runs end-to-end.

Poverty is defined *relatively*: a household-wave is "poor" if its
``nom_totcons_aeq`` (nominal total consumption per adult equivalent) falls
in the bottom ``poverty_quantile`` of its own wave. This is nominal birr,
not inflation-adjusted across 2011/2013/2015, so a fixed cross-wave birr
threshold would not be valid - within-wave relative poverty is the
methodologically defensible choice here, not an official Ethiopian
poverty line.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "household_id", "wave", "year", "mean_haz", "n_children",
    "rain_annual_anomaly", "rain_wetq_anomaly", "srtm", "dist_market",
    "drought", "flood", "any_weather_shock", "nom_totcons_aeq",
]


def _assign_relative_poverty(df: pd.DataFrame, poverty_quantile: float) -> pd.DataFrame:
    df = df.copy()
    threshold_by_wave = df.groupby("wave")["nom_totcons_aeq"].transform(
        lambda s: s.quantile(poverty_quantile)
    )
    df["D_true"] = (df["nom_totcons_aeq"] < threshold_by_wave).astype(int)
    return df


def _generate_synthetic_panel(rng: np.random.Generator, n: int = 2000) -> pd.DataFrame:
    """A synthetic panel with the real panel's columns and roughly its
    distributional shape (calibrated by eye against household_wave_panel_i's
    summary statistics) - not a claim about the real Ethiopian population.
    """
    waves = rng.choice([1, 2, 3], size=n)
    year_map = {1: 2011, 2: 2013, 3: 2015}

    df = pd.DataFrame({
        "household_id": [f"synthetic_{i:06d}" for i in range(n)],
        "wave": waves,
        "year": [year_map[w] for w in waves],
        "n_children": rng.poisson(1.8, n) + 1,
        "rain_annual_anomaly": rng.normal(0, 220, n),
        "rain_wetq_anomaly": rng.normal(0, 100, n),
        "srtm": np.clip(rng.normal(1800, 700, n), 500, 3400),
        "dist_market": np.clip(rng.normal(80, 50, n), 1, 300),
        "nom_totcons_aeq": rng.lognormal(mean=8.2, sigma=0.9, size=n),
    })
    df["drought"] = rng.binomial(1, 0.12, n)
    df["flood"] = rng.binomial(1, 0.05, n)
    df["any_weather_shock"] = np.maximum(df["drought"], df["flood"])

    # Poorer households (lower consumption) tend toward worse child health -
    # a real, well-documented gradient - plus noise, clipped to the WHO
    # flagging range used by the real pipeline.
    log_cons_z = (np.log(df["nom_totcons_aeq"]) - 8.2) / 0.9
    df["mean_haz"] = np.clip(-1.6 + 0.35 * log_cons_z + rng.normal(0, 1.1, n), -6, 6)

    return df


def load_poverty_panel(
    real_data_path: str,
    poverty_quantile: float = 1.0 / 3.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Load the real panel if available, else a synthetic one. Either way,
    returns rows with a non-null ``nom_totcons_aeq`` and ``mean_haz``, plus
    the derived binary ``D_true`` poverty indicator.
    """
    rng = rng if rng is not None else np.random.default_rng()

    if os.path.exists(real_data_path):
        logger.info("Loading real ESS panel from %s", real_data_path)
        df = pd.read_parquet(real_data_path)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Real panel is missing expected columns: {sorted(missing)}")
    else:
        logger.warning(
            "Real ESS panel not found at %s (see DHS/docs/wb_lsms_data_checklist.md "
            "for access). Using a synthetic panel instead.", real_data_path,
        )
        df = _generate_synthetic_panel(rng)

    df = df.dropna(subset=["nom_totcons_aeq", "mean_haz"]).reset_index(drop=True)
    df = _assign_relative_poverty(df, poverty_quantile)
    logger.info("Loaded %d household-wave rows.", len(df))
    return df
