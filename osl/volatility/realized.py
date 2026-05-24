"""Realized-volatility estimators on daily OHLC bars.

All estimators return an **annualized** volatility (decimal) as a rolling
``pd.Series`` aligned to the input index, using 252 trading days per year by
default. Implemented: close-to-close, Parkinson, Garman-Klass,
Rogers-Satchell, and Yang-Zhang (the default for daily ETF bars).

Input DataFrames use lowercase ``open``/``high``/``low``/``close`` columns
(the convention emitted by the provider ``get_history`` methods).
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

ANNUALIZATION = 252
Method = Literal["cc", "parkinson", "gk", "rs", "yz"]

_FOUR_LN2 = 4.0 * float(np.log(2.0))


def _log(s: pd.Series) -> pd.Series:
    """Elementwise log preserving the Series type (numpy ufunc loses it in stubs)."""
    return cast(pd.Series, np.log(s))


def _ann(variance: pd.Series, annualization: int) -> pd.Series:
    return cast(pd.Series, np.sqrt(annualization * variance))


def close_to_close(
    df: pd.DataFrame, window: int = 20, *, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Classic close-to-close estimator (sample std of log returns)."""
    log_ret = _log(df["close"]).diff()
    var = log_ret.rolling(window).var(ddof=1)
    return _ann(var, annualization)


def parkinson(
    df: pd.DataFrame, window: int = 20, *, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Parkinson high-low range estimator."""
    hl = _log(df["high"] / df["low"]) ** 2
    var = hl.rolling(window).mean() / _FOUR_LN2
    return _ann(var, annualization)


def garman_klass(
    df: pd.DataFrame, window: int = 20, *, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Garman-Klass OHLC estimator."""
    hl = _log(df["high"] / df["low"]) ** 2
    co = _log(df["close"] / df["open"]) ** 2
    term = 0.5 * hl - (2.0 * float(np.log(2.0)) - 1.0) * co
    var = term.rolling(window).mean()
    return _ann(var, annualization)


def rogers_satchell(
    df: pd.DataFrame, window: int = 20, *, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Rogers-Satchell estimator (drift-independent)."""
    term = _log(df["high"] / df["close"]) * _log(df["high"] / df["open"]) + _log(
        df["low"] / df["close"]
    ) * _log(df["low"] / df["open"])
    var = term.rolling(window).mean()
    return _ann(var, annualization)


def yang_zhang(
    df: pd.DataFrame, window: int = 20, *, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Yang-Zhang estimator: overnight + k·open-to-close + (1-k)·Rogers-Satchell."""
    overnight = _log(df["open"] / df["close"].shift(1))
    open_to_close = _log(df["close"] / df["open"])

    sigma_o = overnight.rolling(window).var(ddof=1)
    sigma_c = open_to_close.rolling(window).var(ddof=1)

    rs_term = _log(df["high"] / df["close"]) * _log(df["high"] / df["open"]) + _log(
        df["low"] / df["close"]
    ) * _log(df["low"] / df["open"])
    sigma_rs = rs_term.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = sigma_o + k * sigma_c + (1.0 - k) * sigma_rs
    return _ann(var, annualization)


_ESTIMATORS = {
    "cc": close_to_close,
    "parkinson": parkinson,
    "gk": garman_klass,
    "rs": rogers_satchell,
    "yz": yang_zhang,
}


def realized_vol(
    df: pd.DataFrame, method: Method = "yz", window: int = 20, *, annualization: int = ANNUALIZATION
) -> pd.Series:
    """Dispatch to a named estimator; default Yang-Zhang."""
    try:
        estimator = _ESTIMATORS[method]
    except KeyError:
        raise ValueError(f"unknown realized-vol method: {method!r}") from None
    return estimator(df, window, annualization=annualization)
