"""Implied-volatility inversion.

Default method is Jaeckel's "Let's Be Rational" (via ``vollib``), with a robust
Brent-bisection fallback that inverts :func:`osl.pricing.bsm.bsm_price`
directly. Both honour a continuous dividend yield ``q``.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Literal

from scipy.optimize import brentq

from osl.pricing.bsm import bsm_price

Method = Literal["jaeckel", "bisect"]

_SIGMA_LO = 1e-6
_SIGMA_HI = 10.0


def _ensure_testcapi() -> None:
    """Provide a minimal ``_testcapi`` stand-in if the interpreter lacks it.

    ``py_lets_be_rational`` (pulled in by vollib for Jaeckel IV) imports
    ``DBL_MIN``/``DBL_MAX`` from CPython's internal ``_testcapi`` module, which
    is absent from python-build-standalone interpreters (e.g. those installed by
    ``uv python``). Inject those two float limits so Jaeckel IV works on every
    CPython build. No-op when the real ``_testcapi`` is present.
    """
    if "_testcapi" in sys.modules:
        return
    try:
        importlib.import_module("_testcapi")
        return
    except ImportError:
        shim = types.ModuleType("_testcapi")
        shim.__dict__.update({"DBL_MIN": sys.float_info.min, "DBL_MAX": sys.float_info.max})
        sys.modules["_testcapi"] = shim


_ensure_testcapi()


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
