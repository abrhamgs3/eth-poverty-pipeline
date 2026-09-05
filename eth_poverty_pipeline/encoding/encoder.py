"""Stage 2: encode the LLM's predicted poverty status into the binary
``D_pred`` regressor, directly comparable to ``D_true``.
"""

from __future__ import annotations

import pandas as pd


def encode_predicted_poverty(df: pd.DataFrame, pred_col: str = "predicted_poverty_status") -> pd.DataFrame:
    encoded = df.copy()
    encoded["D_pred"] = (encoded[pred_col] == "poor").astype(int)
    return encoded
