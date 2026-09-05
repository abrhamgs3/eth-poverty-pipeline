"""Stage 4: Rogan-Gladen / method-of-moments bias correction.

When the binary poverty indicator D is observed only through a classifier
with known sensitivity (TPR) and specificity (TNR), OLS of an outcome on
the noisy label attenuates the coefficient toward zero. This corrects it:

    q_corrected = (q_observed - FPR) / (TPR + TNR - 1)      # Rogan-Gladen
    beta_corrected = Cov(D_noisy, Y) / ((TPR + TNR - 1) * Var(D_corrected))

Standard errors come from a nonparametric bootstrap over the study sample.

Note on this application specifically: unlike a pure simulation, there is
no independently *known* true beta_1 here - ``D_true`` is itself a real,
measured quantity (a consumption-based poverty indicator), not a ground
truth planted by the analyst. The "True" regression is the best available
benchmark, not an oracle; both it and the correction carry real sampling
uncertainty. See the README's Limitations section.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _validate_skill(tpr: float, tnr: float) -> float:
    denom = tpr + tnr - 1.0
    if denom <= 1e-4:
        raise ValueError(
            f"Classifier skill (tpr + tnr - 1 = {denom:.4f}) must be positive and "
            f"bounded away from zero for the correction to be identified "
            f"(got tpr={tpr}, tnr={tnr})."
        )
    return denom


def correct_beta(x_noisy: np.ndarray, y: np.ndarray, tpr: float, tnr: float) -> Tuple[float, float]:
    """Return (corrected_prevalence, corrected_beta_1)."""
    denom = _validate_skill(tpr, tnr)
    fpr = 1.0 - tnr

    q_observed = np.mean(x_noisy)
    q_corrected = np.clip((q_observed - fpr) / denom, 1e-5, 1 - 1e-5)
    var_x_corrected = q_corrected * (1.0 - q_corrected)

    cov_noisy_y = np.cov(x_noisy, y)[0, 1]
    beta_1_corrected = cov_noisy_y / (denom * var_x_corrected)

    return float(q_corrected), float(beta_1_corrected)


def run_bootstrap_inference(
    df: pd.DataFrame,
    y_col: str,
    x_noisy_col: str,
    tpr: float,
    tnr: float,
    n_boot: int = 300,
    rng: np.random.Generator | None = None,
) -> Dict[str, Any]:
    rng = rng if rng is not None else np.random.default_rng()
    y_arr = df[y_col].to_numpy()
    x_arr = df[x_noisy_col].to_numpy()

    q_corr, beta_corr = correct_beta(x_arr, y_arr, tpr, tnr)
    beta_0_corr = float(np.mean(y_arr) - beta_corr * q_corr)

    n = len(df)
    boot_beta_0 = np.empty(n_boot)
    boot_beta_1 = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        q_b, beta_1_b = correct_beta(x_arr[idx], y_arr[idx], tpr, tnr)
        boot_beta_1[i] = beta_1_b
        boot_beta_0[i] = np.mean(y_arr[idx]) - beta_1_b * q_b

    return {
        "beta_0": {
            "estimate": beta_0_corr,
            "se": float(np.std(boot_beta_0)),
            "ci": [float(np.percentile(boot_beta_0, 2.5)), float(np.percentile(boot_beta_0, 97.5))],
        },
        "beta_1": {
            "estimate": beta_corr,
            "se": float(np.std(boot_beta_1)),
            "ci": [float(np.percentile(boot_beta_1, 2.5)), float(np.percentile(boot_beta_1, 97.5))],
        },
    }
