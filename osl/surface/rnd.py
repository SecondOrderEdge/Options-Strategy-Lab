"""Risk-neutral density via Breeden-Litzenberger.

The RND is the discounted second derivative of the call price w.r.t. strike::

    f^Q(K) = e^{rT} * d^2C/dK^2     (Breeden & Litzenberger 1978)

Given a smile expressed as a strike->IV callable, we price calls on a dense
strike grid, take the numerical second derivative, discount, and clip tiny
negatives. For an arbitrage-free smile the result is a proper density
(integrates to 1, mean equals the forward).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from osl.pricing.bsm import bsm_price

FloatArray = npt.NDArray[np.float64]


def risk_neutral_density(
    iv_of_strike: Callable[[FloatArray], FloatArray],
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    T: float,
    k_min: float = -1.5,
    k_max: float = 1.5,
    n: int = 600,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(strikes, density)`` on a log-moneyness grid around the forward."""
    forward = spot * np.exp((rate - dividend_yield) * T)
    grid_k = np.linspace(k_min, k_max, n)
    strikes = forward * np.exp(grid_k)

    iv = np.asarray(iv_of_strike(strikes), dtype=np.float64)
    calls = bsm_price(spot, strikes, T, rate, dividend_yield, iv, "c")

    d2c = np.gradient(np.gradient(calls, strikes), strikes)
    density = np.exp(rate * T) * d2c
    density = np.clip(density, 0.0, None)
    return np.asarray(strikes, dtype=np.float64), np.asarray(density, dtype=np.float64)


def normalize(strikes: FloatArray, density: FloatArray) -> FloatArray:
    """Rescale a density so it integrates to 1 over the strike grid."""
    area = float(np.trapz(density, strikes))
    if area <= 0:
        return density
    return np.asarray(density / area, dtype=np.float64)
