"""Implied-volatility inversion.

Default method is Jaeckel's "Let's Be Rational" (via ``vollib``), with a robust
Brent-bisection fallback that inverts :func:`osl.pricing.bsm.bsm_price`
directly. Both honour a continuous dividend yield ``q``.
"""

from __future__ import annotations

from typing import Literal

from scipy.optimize import brentq

from osl.pricing.bsm import bsm_price

Method = Literal["jaeckel", "bisect"]

_SIGMA_LO = 1e-6
_SIGMA_HI = 10.0


def _bisect_iv(price: float, S: float, K: float, T: float, r: float, q: float, flag: str) -> float:
    def objective(sigma: float) -> float:
        return float(bsm_price(S, K, T, r, q, sigma, flag).item()) - price

    return float(brentq(objective, _SIGMA_LO, _SIGMA_HI, xtol=1e-12, rtol=1e-12))


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    right: str,
    *,
    method: Method = "jaeckel",
) -> float:
    """Return the BSM implied volatility for a single option price.

    Parameters mirror :func:`osl.pricing.bsm.bsm_price`. ``right`` is
    ``"c"``/``"p"`` (case-insensitive). Raises ``ValueError`` if ``price``
    violates no-arbitrage bounds for the bisection method.
    """
    flag = "c" if str(right).lower().startswith("c") else "p"

    if method == "jaeckel":
        from vollib.black_scholes_merton.implied_volatility import implied_volatility

        return float(implied_volatility(price, S, K, T, r, q, flag))
    if method == "bisect":
        return _bisect_iv(price, S, K, T, r, q, flag)
    raise ValueError(f"unknown IV method: {method!r}")
