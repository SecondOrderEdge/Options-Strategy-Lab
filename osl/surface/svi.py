"""Raw SVI parameterization, calibration, and no-arbitrage checks.

Raw SVI (Gatheral) total variance in log-moneyness ``k``::

    w(k) = a + b ( rho (k - m) + sqrt((k - m)^2 + sigma^2) )

Calibrated per expiry by Levenberg-Marquardt (``scipy.least_squares``) with
vega weights. Butterfly arbitrage is checked via Gatheral's g(k) function
(Gatheral & Jacquier 2014); calendar arbitrage via monotonicity of total
variance in T.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

FloatArray = npt.NDArray[np.float64]

# Parameter bounds: (a, b, rho, m, sigma).
_LOWER = np.array([-5.0, 0.0, -0.999, -2.0, 1e-4])
_UPPER = np.array([5.0, 10.0, 0.999, 2.0, 5.0])


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters for a single expiry."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: npt.ArrayLike) -> FloatArray:
        k_ = np.asarray(k, dtype=np.float64)
        x = k_ - self.m
        return np.asarray(
            self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma * self.sigma)),
            dtype=np.float64,
        )

    def d_total_variance(self, k: npt.ArrayLike) -> FloatArray:
        k_ = np.asarray(k, dtype=np.float64)
        x = k_ - self.m
        r = np.sqrt(x * x + self.sigma * self.sigma)
        return np.asarray(self.b * (self.rho + x / r), dtype=np.float64)

    def d2_total_variance(self, k: npt.ArrayLike) -> FloatArray:
        k_ = np.asarray(k, dtype=np.float64)
        x = k_ - self.m
        r = np.sqrt(x * x + self.sigma * self.sigma)
        return np.asarray(self.b * self.sigma * self.sigma / (r * r * r), dtype=np.float64)

    def iv(self, k: npt.ArrayLike, T: float) -> FloatArray:
        w = np.maximum(self.total_variance(k), 0.0)
        return np.asarray(np.sqrt(w / T), dtype=np.float64)

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


@dataclass(frozen=True)
class SVIFit:
    """Result of an SVI calibration."""

    params: SVIParams
    rmse_vol: float  # weighted RMSE in vol points (decimal)
    success: bool
    n_points: int
    butterfly_free: bool


def _g(params: SVIParams, k: FloatArray) -> FloatArray:
    """Gatheral's g(k); butterfly-arb-free iff g(k) >= 0 for all k (and w>0)."""
    w = params.total_variance(k)
    wp = params.d_total_variance(k)
    wpp = params.d2_total_variance(k)
    term1 = (1.0 - k * wp / (2.0 * w)) ** 2
    term2 = (wp * wp / 4.0) * (1.0 / w + 0.25)
    return np.asarray(term1 - term2 + wpp / 2.0, dtype=np.float64)


def butterfly_min_g(params: SVIParams, k_grid: npt.ArrayLike | None = None) -> float:
    """Minimum of g(k) over a grid; negative implies butterfly arbitrage."""
    grid = (
        np.asarray(k_grid, dtype=np.float64) if k_grid is not None else np.linspace(-1.5, 1.5, 601)
    )
    w = params.total_variance(grid)
    if np.any(w <= 0):
        return float("-inf")
    return float(np.min(_g(params, grid)))


def is_butterfly_arbitrage_free(
    params: SVIParams, k_grid: npt.ArrayLike | None = None, *, tol: float = 1e-8
) -> bool:
    """True if no butterfly arbitrage (density nonneg) on the grid."""
    return butterfly_min_g(params, k_grid) >= -tol


def calendar_arbitrage_free(
    fits: Sequence[tuple[float, SVIParams]],
    k_grid: npt.ArrayLike | None = None,
    *,
    tol: float = 1e-8,
) -> bool:
    """True if total variance is non-decreasing in T across the given expiries.

    ``fits`` is a sequence of ``(T, params)`` pairs; they are sorted by T.
    """
    grid = (
        np.asarray(k_grid, dtype=np.float64) if k_grid is not None else np.linspace(-1.0, 1.0, 201)
    )
    ordered = sorted(fits, key=lambda p: p[0])
    prev: FloatArray | None = None
    for _T, params in ordered:
        w = params.total_variance(grid)
        if prev is not None and np.any(w - prev < -tol):
            return False
        prev = w
    return True


def _initial_guess(k: FloatArray, w: FloatArray) -> FloatArray:
    return np.array([max(float(np.min(w)) * 0.5, 1e-4), 0.1, -0.3, float(np.median(k)), 0.1])


def fit_svi(
    k: npt.ArrayLike,
    total_variance: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
    T: float | None = None,
) -> SVIFit:
    """Calibrate raw SVI to ``(k, w)`` points with optional weights.

    ``T`` (if given) is used to report RMSE in vol points; otherwise RMSE is in
    total-variance units. Requires at least 5 points.
    """
    k_ = np.asarray(k, dtype=np.float64)
    w_ = np.asarray(total_variance, dtype=np.float64)
    if k_.size < 5:
        raise ValueError(f"SVI needs >= 5 points, got {k_.size}")
    wgt = np.ones_like(k_) if weights is None else np.asarray(weights, dtype=np.float64)
    sqrt_wgt = np.sqrt(np.maximum(wgt, 0.0))

    def residuals(theta: FloatArray) -> FloatArray:
        a, b, rho, m, sigma = theta
        x = k_ - m
        model = a + b * (rho * x + np.sqrt(x * x + sigma * sigma))
        return np.asarray(sqrt_wgt * (model - w_), dtype=np.float64)

    result = least_squares(
        residuals,
        _initial_guess(k_, w_),
        bounds=(_LOWER, _UPPER),
        method="trf",
        max_nfev=10_000,
    )
    params = SVIParams(*(float(v) for v in result.x))

    model_w = params.total_variance(k_)
    if T is not None:
        iv_model = np.sqrt(np.maximum(model_w, 0.0) / T)
        iv_mkt = np.sqrt(np.maximum(w_, 0.0) / T)
        rmse = float(np.sqrt(np.average((iv_model - iv_mkt) ** 2, weights=wgt)))
    else:
        rmse = float(np.sqrt(np.average((model_w - w_) ** 2, weights=wgt)))

    return SVIFit(
        params=params,
        rmse_vol=rmse,
        success=bool(result.success),
        n_points=int(k_.size),
        butterfly_free=is_butterfly_arbitrage_free(params),
    )


def make_iv_of_strike(
    params: SVIParams, *, forward: float, T: float
) -> Callable[[FloatArray], FloatArray]:
    """Return a callable K -> IV from SVI params (k = ln(K/forward))."""

    def iv_of_strike(strikes: FloatArray) -> FloatArray:
        k = np.log(np.asarray(strikes, dtype=np.float64) / forward)
        return params.iv(k, T)

    return iv_of_strike
