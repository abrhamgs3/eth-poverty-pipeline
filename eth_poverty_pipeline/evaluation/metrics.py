"""Stage 3: evaluate the LLM poverty screen against the true (consumption-
based) poverty indicator on a held-out validation split.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

MIN_RELIABLE_VALIDATION_SIZE = 30


def evaluate_poverty_classification(df: pd.DataFrame, true_col: str = "D_true", pred_col: str = "D_pred") -> Dict[str, Any]:
    n = len(df)
    if n < MIN_RELIABLE_VALIDATION_SIZE:
        warnings.warn(
            f"Evaluating against only {n} validation examples. TPR/TNR estimates "
            f"below ~{MIN_RELIABLE_VALIDATION_SIZE} examples carry wide sampling "
            "uncertainty; treat downstream bias-correction results as illustrative.",
            stacklevel=2,
        )

    y_true = df[true_col].astype(int)
    y_pred = df[pred_col].astype(int)

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "n": n,
        "Accuracy": acc,
        "Cohen_Kappa": kappa,
        "F1_Score": f1,
        "True_Positive_Rate_TPR": tpr,
        "True_Negative_Rate_TNR": tnr,
        "False_Positive_Rate_FPR": 1.0 - tnr,
        "False_Negative_Rate_FNR": 1.0 - tpr,
        "Confusion_Matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
    }
