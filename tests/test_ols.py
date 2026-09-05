import numpy as np
import pytest

from eth_poverty_pipeline.econometrics.ols import SimpleOLS


def test_recovers_exact_coefficients_with_no_noise():
    x = np.arange(50, dtype=float)
    y = 5.0 + 3.0 * x

    result = SimpleOLS(y, x, add_constant=True).fit()

    np.testing.assert_allclose(result.beta, [5.0, 3.0], atol=1e-8)
    np.testing.assert_allclose(result.se, [0.0, 0.0], atol=1e-8)
    assert result.r_squared == pytest.approx(1.0)


def test_recovers_approximate_coefficients_with_noise():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    y = 5.0 + 3.0 * x + rng.normal(0, 0.5, 500)

    result = SimpleOLS(y, x, add_constant=True).fit()

    assert result.beta[0] == pytest.approx(5.0, abs=0.1)
    assert result.beta[1] == pytest.approx(3.0, abs=0.1)
    assert result.r_squared > 0.8


def test_singular_matrix_falls_back_without_raising_and_se_has_no_nan():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 200)
    y = 5.0 + 3.0 * x + rng.normal(0, 0.5, 200)
    collinear_x = np.column_stack((x, x * 2.0))

    result = SimpleOLS(y, collinear_x, add_constant=True).fit()

    assert result.beta.shape == (3,)
    assert np.all(np.isfinite(result.se))
    assert np.all(np.isfinite(result.t_stats))


def test_clips_negative_variance_and_warns_when_it_occurs():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fake_pinv = np.array([[1.0, 0.0], [0.0, -0.5]])
    original_pinv = np.linalg.pinv
    original_inv = np.linalg.inv

    def _boom(*args, **kwargs):
        raise np.linalg.LinAlgError("forced for test")

    try:
        np.linalg.inv = _boom
        np.linalg.pinv = lambda *a, **k: fake_pinv
        with pytest.warns(UserWarning, match="non-positive variance"):
            result = SimpleOLS(y, x, add_constant=True).fit()
    finally:
        np.linalg.inv = original_inv
        np.linalg.pinv = original_pinv

    assert result.se[1] == 0.0
    assert np.isfinite(result.t_stats[1])


def test_insufficient_sample_raises_value_error():
    y_tiny = np.array([1.0, 2.0])
    x_tiny = np.array([1.0, 3.0])

    with pytest.raises(ValueError):
        SimpleOLS(y_tiny, x_tiny, add_constant=True).fit()
