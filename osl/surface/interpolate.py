"""Monotone cubic-spline-in-variance fallback when SVI calibration fails.

Interpolates total variance against log-moneyness with a shape-preserving PCHIP
spline, extending flat beyond the last observed strikes. This is a numerical
fallback only and carries no arbitrage guarantees — callers should label it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import PchipInterpolator

FloatArray = npt.NDArray[np.float64]


@dataclass
class VarianceSpline:
    """PCHIP interpolation of total variance w(k)."""

    k: FloatArray
    w: FloatArray
    _spline: PchipInterpolator

    @classmethod
    def fit(cls, k: npt.ArrayLike, total_variance: npt.ArrayLike) -> VarianceSpline:
        k_ = np.asarray(k, dtype=np.float64)
        w_ = np.asarray(total_variance, dtype=np.float64)
        order = np.argsort(k_)
        k_s, w_s = k_[order], w_[order]
        # De-duplicate identical knots (PCHIP requires strictly increasing x).
        unique_mask = np.concatenate(([True], np.diff(k_s) > 0))
        k_s, w_s = k_s[unique_mask], w_s[unique_mask]
        if k_s.size < 2:
            raise ValueError("variance spline needs >= 2 distinct knots")
        return cls(k=k_s, w=w_s, _spline=PchipInterpolator(k_s, w_s, extrapolate=False))

    def total_variance(self, k: npt.ArrayLike) -> FloatArray:
        k_ = np.clip(np.asarray(k, dtype=np.float64), self.k[0], self.k[-1])
        return np.asarray(self._spline(k_), dtype=np.float64)

    def iv(self, k: npt.ArrayLike, T: float) -> FloatArray:
        w = np.maximum(self.total_variance(k), 0.0)
        return np.asarray(np.sqrt(w / T), dtype=np.float64)
