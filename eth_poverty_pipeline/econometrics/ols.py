"""Dependency-free OLS solver.

Plain NumPy linear algebra - no ``statsmodels``, so it has no compiled
C-extension dependency that locked-down Windows environments (AppLocker /
WDAC policies) can block. p-values use a normal approximation (via
``math.erf``) rather than the exact t-distribution, needing no ``scipy``
dependency, accurate for the sample sizes this pipeline targets.
"""

from __future__ import annotations

import logging
import warnings
from math import erf

import numpy as np

logger = logging.getLogger(__name__)


class SimpleOLS:
    """Ordinary least squares via the normal equations."""

    def __init__(self, y: np.ndarray, x: np.ndarray, add_constant: bool = True):
        self.y = np.asarray(y, dtype=float).flatten()
        x = np.asarray(x, dtype=float)
        self.x = x.reshape(-1, 1) if x.ndim == 1 else x
        if add_constant:
            self.x = np.column_stack((np.ones(len(self.y)), self.x))

        self.n_samples, self.n_features = self.x.shape
        self.beta = None
        self.residuals = None
        self.se = None
        self.t_stats = None
        self.p_values = None
        self.r_squared = None

    def fit(self) -> "SimpleOLS":
        xtx = self.x.T @ self.x
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            logger.warning("Singular design matrix; falling back to pseudoinverse (SVD).")
            xtx_inv = np.linalg.pinv(xtx)

        self.beta = xtx_inv @ self.x.T @ self.y
        self.residuals = self.y - self.x @ self.beta

        dof = self.n_samples - self.n_features
        if dof <= 0:
            raise ValueError("Degrees of freedom <= 0: insufficient sample size.")

        sigma_sq = float(self.residuals @ self.residuals) / dof
        cov_beta_diag = sigma_sq * np.diag(xtx_inv)

        # Under near-perfect multicollinearity the pseudoinverse can yield a
        # `cov_beta` that isn't positive semi-definite, producing negative
        # "variances." Clip rather than let sqrt silently emit NaN, and
        # surface it - a clipped coefficient's SE/t-stat/p-value is not
        # trustworthy and callers should treat the design as near-singular.
        n_negative = int(np.sum(cov_beta_diag < 0))
        if n_negative:
            warnings.warn(
                f"{n_negative} coefficient(s) have a non-positive variance estimate "
                "(near-singular design matrix) - their SE/t-stat/p-value are unreliable.",
                stacklevel=2,
            )
        self.se = np.sqrt(np.clip(cov_beta_diag, 0, None))

        self.t_stats = np.divide(
            self.beta, self.se, out=np.zeros_like(self.beta), where=self.se > 0
        )
        self.p_values = 2 * (1.0 - np.vectorize(self._standard_normal_cdf)(np.abs(self.t_stats)))

        y_mean = np.mean(self.y)
        tss = np.sum((self.y - y_mean) ** 2)
        rss = np.sum(self.residuals**2)
        self.r_squared = 1.0 - (rss / tss) if tss > 0 else 0.0

        return self

    @staticmethod
    def _standard_normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
