"""Stage 1: generate a household field note, then annotate it.

``build_household_profile`` turns the structured covariates already in the
panel (household size, distance to the nearest market, elevation, this
year's rainfall anomaly, reported weather shocks) into a short natural-
language note - the kind of thing a community health extension worker
might jot down on a home visit, in lieu of administering the full,
expensive consumption module. It deliberately excludes ``nom_totcons_aeq``
itself (using it would make the annotation task circular) and is
therefore only a *weak, imperfect* proxy for poverty status, by design -
that's what makes the following pipeline stages worth running at all.

This is a plausible construction for demonstrating the annotation-bias
methodology, not a claim that these particular notes were collected in the
real ESS survey - they weren't; the real survey has no free-text field.
"""

from __future__ import annotations

import logging

import pandas as pd

from .clients import LLMClient

logger = logging.getLogger(__name__)


def build_household_profile(row: pd.Series) -> str:
    n_children = int(row["n_children"])
    dist_market = row["dist_market"]
    elevation = row["srtm"]
    rain_anomaly = row["rain_annual_anomaly"]

    rain_desc = (
        f"about {abs(rain_anomaly):.0f}mm {'above' if rain_anomaly >= 0 else 'below'} "
        "the long-run average"
    )
    shock_bits = []
    if row.get("drought"):
        shock_bits.append("the household reported crop losses from drought this year")
    if row.get("flood"):
        shock_bits.append("the household reported flood damage this year")
    shock_desc = "; ".join(shock_bits) if shock_bits else "no weather shocks were reported this year"

    return (
        f"Home visit note: household has {n_children} child(ren) under five. "
        f"It is located roughly {dist_market:.0f} km from the nearest market, "
        f"at an elevation of about {elevation:.0f}m. Annual rainfall this year was "
        f"{rain_desc}. {shock_desc.capitalize()}."
    )


def annotate_batch(
    df: pd.DataFrame,
    client: LLMClient,
    gold_col: str = "D_true",
) -> pd.DataFrame:
    """Build a field note per row, annotate it, and return the results."""
    logger.info("Annotating %d household-wave notes...", len(df))
    records = []
    for idx, row in df.iterrows():
        note = build_household_profile(row)
        true_label = "poor" if (gold_col in row and row[gold_col] == 1) else "non_poor"
        annotation = client.annotate(note, true_label=true_label)
        records.append({
            "id": idx,
            "note": note,
            "predicted_poverty_status": annotation.poverty_status,
            "confidence": annotation.confidence,
            "explanation": annotation.explanation,
        })
    logger.info("Annotation complete.")
    return pd.DataFrame(records)
