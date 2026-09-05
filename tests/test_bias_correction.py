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

    results = run_bootstrap_inference(
        df, y_col="haz", x_noisy_col="D_pred", tpr=0.8, tnr=0.8, n_boot=50, rng=rng,
    )

    for key in ("beta_0", "beta_1"):
        assert key in results
        assert set(results[key]) == {"estimate", "se", "ci"}
        assert len(results[key]["ci"]) == 2
