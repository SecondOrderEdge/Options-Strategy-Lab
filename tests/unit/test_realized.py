"""Realized-vol estimators: closed-form checks on constructed series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osl.volatility.realized import (
    close_to_close,
    parkinson,
    realized_vol,
    yang_zhang,
)

ANN = 252


def _flat_intraday_prices(returns: np.ndarray) -> pd.DataFrame:
    """OHLC where O=H=L=C each day (no intraday range); close follows returns."""
    close = 100.0 * np.exp(np.cumsum(returns))
    close = np.insert(close, 0, 100.0)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close})


def test_close_to_close_matches_numpy_std():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, size=300)
    df = _flat_intraday_prices(returns)
    n = 60
    cc = close_to_close(df, window=n).iloc[-1]

    log_ret = np.log(df["close"]).diff().dropna().to_numpy()[-n:]
    expected = np.sqrt(ANN) * np.std(log_ret, ddof=1)
    assert cc == pytest.approx(expected, rel=1e-12)


def test_yang_zhang_reduces_to_overnight_when_no_intraday():
    # With O=H=L=C, GK/Parkinson/RS terms vanish and open-to-close var is 0,
    # so YZ collapses to the overnight (== close-to-close) variance.
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.012, size=300)
    df = _flat_intraday_prices(returns)
    n = 60
    yz = yang_zhang(df, window=n).iloc[-1]
    cc = close_to_close(df, window=n).iloc[-1]
    assert yz == pytest.approx(cc, rel=1e-10)


def test_parkinson_constant_range():
    # Constant H/L ratio => Parkinson vol is constant and analytically known.
    n = 20
    ratio = 1.02
    close = np.full(60, 100.0)
    high = close * np.sqrt(ratio)
    low = close / np.sqrt(ratio)
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
    pk = parkinson(df, window=n).iloc[-1]
    ln_hl_sq = np.log(high[0] / low[0]) ** 2
    expected = np.sqrt(ANN * ln_hl_sq / (4 * np.log(2)))
    assert pk == pytest.approx(expected, rel=1e-12)


def test_estimators_nonnegative_and_dispatch():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0005, 0.02, size=200)
    close = 100 * np.exp(np.cumsum(returns))
    noise = rng.uniform(0.0, 0.01, size=close.size)
    df = pd.DataFrame(
        {
            "open": close * (1 - noise / 2),
            "high": close * (1 + noise),
            "low": close * (1 - noise),
            "close": close,
        }
    )
    for method in ("cc", "parkinson", "gk", "rs", "yz"):
        rv = realized_vol(df, method=method, window=20).dropna()  # type: ignore[arg-type]
        assert (rv >= 0).all()
        assert not rv.empty


def test_unknown_method():
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})
    with pytest.raises(ValueError, match="unknown realized-vol method"):
        realized_vol(df, method="garch", window=5)  # type: ignore[arg-type]
