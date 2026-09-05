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

Reliability warning: TPR and TNR are themselves *estimates* from a finite
validation sample, and the correction's denominator (TPR+TNR-1) is exactly
what estimation noise can push close to zero even when the underlying
classifier is genuinely accurate. A companion manuscript
(``paper/manuscript.tex``, Section 5.2) demonstrates this directly: 20
independent n=38 validation splits of the *same* 80%-accurate mock
classifier produced 19 sane corrected estimates and one that diverged by
five orders of magnitude, because that split's estimated TPR+TNR-1 (0.114)
happened to land near zero. ``MIN_RELIABLE_SKILL`` below is the paper's
provisional threshold for flagging that regime - it is a warning, not a
hard failure, and calibrated to one accuracy level and one sample size
(see the manuscript's Limitations).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

MIN_RELIABLE_SKILL = 0.2


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
    min_reliable_skill: float = MIN_RELIABLE_SKILL,
) -> Dict[str, Any]:
    """``min_reliable_skill`` gates a one-time warning (not a failure) when
    TPR+TNR-1 is positive but small enough that the correction's point
    estimate should not be trusted - see the module docstring. Checked once
    here against the point-estimate TPR/TNR, not per bootstrap draw: TPR
    and TNR are held fixed across the resampling loop below (only the
    study-sample rows are resampled), so the skill estimate - and whether
    it's in the unreliable regime - doesn't vary draw to draw.
    """
    denom = tpr + tnr - 1.0
    if 0 < denom < min_reliable_skill:
        warnings.warn(
            f"Estimated classifier skill (TPR+TNR-1={denom:.3f}) is below the "
            f"min_reliable_skill threshold ({min_reliable_skill}). The correction's "
            f"denominator amplifies validation-sample noise as it approaches zero - "
            f"point estimates in this regime can diverge by orders of magnitude even "
            f"when the underlying classifier is genuinely accurate (see "
            f"paper/manuscript.tex, Section 5.2, for a demonstrated 5-order-of-"
            f"magnitude divergence at skill=0.114). Treat this corrected estimate as "
            f"unreliable; a larger validation sample is the fix, not a different "
            f"correction formula.",
            stacklevel=2,
        )

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
