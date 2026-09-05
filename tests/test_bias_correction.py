import warnings

import numpy as np
import pandas as pd
import pytest

from eth_poverty_pipeline.econometrics.bias_correction import correct_beta, run_bootstrap_inference


def test_rogan_gladen_prevalence_recovery():
    rng = np.random.default_rng(42)
    tpr, tnr = 0.85, 0.90
    x_noisy = rng.binomial(1, 0.325, 20000)  # q_true=0.30, FPR=0.10 -> q_observed=0.325

    q_corr, _ = correct_beta(x_noisy, np.zeros(len(x_noisy)), tpr, tnr)

    assert abs(q_corr - 0.30) < 0.02


def test_covariance_scaling_beta_recovery_with_negative_effect():
    # HAZ-style negative effect - the real use case here, unlike the two
    # sibling repos where the demo effect happened to be positive.
    rng = np.random.default_rng(42)
    n = 10000
    tpr, tnr = 0.80, 0.80
    true_beta_1 = -0.4

    D_true = rng.binomial(1, 0.35, n)
    Y = -1.5 + true_beta_1 * D_true + rng.normal(0, 1.0, n)

    flip_prob = np.where(D_true == 1, 1 - tpr, 1 - tnr)
    correct = rng.random(n) >= flip_prob
    D_noisy = np.where(correct, D_true, 1 - D_true)

    _, beta_1_corrected = correct_beta(D_noisy, Y, tpr, tnr)

    assert abs(beta_1_corrected - true_beta_1) < 0.15


def test_random_classification_threshold_raises_value_error():
    with pytest.raises(ValueError):
        correct_beta(np.array([0, 1, 0, 1]), np.array([1.0, 2.0, 1.0, 2.0]), tpr=0.5, tnr=0.5)


def test_worse_than_random_classification_raises_value_error():
    with pytest.raises(ValueError):
        correct_beta(np.array([0, 1, 0, 1]), np.array([1.0, 2.0, 1.0, 2.0]), tpr=0.3, tnr=0.3)


def test_bootstrap_inference_returns_valid_structure():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "haz": rng.normal(-1.5, 1.2, 200),
        "D_pred": rng.binomial(1, 0.3, 200),
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # tpr=tnr=0.8 -> skill=0.6, well above default threshold
        results = run_bootstrap_inference(
            df, y_col="haz", x_noisy_col="D_pred", tpr=0.8, tnr=0.8, n_boot=50, rng=rng,
        )

    for key in ("beta_0", "beta_1"):
        assert key in results
        assert set(results[key]) == {"estimate", "se", "ci"}
        assert len(results[key]["ci"]) == 2


def test_low_skill_estimate_warns_but_still_returns_a_result():
    # tpr=0.55, tnr=0.55 -> skill=0.10: positive (won't raise) but below the
    # default 0.2 reliability threshold - this is the regime documented in
    # paper/manuscript.tex Section 5.2, where the correction can diverge by
    # orders of magnitude from validation-sample noise alone.
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "haz": rng.normal(-1.5, 1.2, 100),
        "D_pred": rng.binomial(1, 0.3, 100),
    })

    with pytest.warns(UserWarning, match="below the min_reliable_skill threshold"):
        results = run_bootstrap_inference(
            df, y_col="haz", x_noisy_col="D_pred", tpr=0.55, tnr=0.55, n_boot=20, rng=rng,
        )

    assert "estimate" in results["beta_1"]  # still computes - a warning, not a failure


def test_min_reliable_skill_threshold_is_configurable():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "haz": rng.normal(-1.5, 1.2, 100),
        "D_pred": rng.binomial(1, 0.3, 100),
    })

    # skill=0.10 is below the default (0.2) but above a lowered threshold.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_bootstrap_inference(
            df, y_col="haz", x_noisy_col="D_pred", tpr=0.55, tnr=0.55,
            n_boot=20, rng=rng, min_reliable_skill=0.05,
        )
