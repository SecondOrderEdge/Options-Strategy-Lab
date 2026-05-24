"""yfinance chain parser: normalizes to canonical schema; greeks are NaN in M0."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from osl.data.base import CANONICAL_COLUMNS, validate_chain
from osl.data.yfinance_provider import parse_yf_chain


def _calls() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contractSymbol": "SPY260619C00500000",
                "strike": 500.0,
                "lastPrice": 12.3,
                "bid": 12.0,
                "ask": 12.6,
                "volume": 1500,
                "openInterest": 8000,
                "impliedVolatility": 0.185,
                "inTheMoney": True,
                "contractSize": "REGULAR",
            },
            {
                "contractSymbol": "SPY260619C00520000",
                "strike": 520.0,
                "lastPrice": 3.1,
                "bid": 0.0,  # no quote
                "ask": 0.0,
                "volume": float("nan"),
                "openInterest": 200,
                "impliedVolatility": 0.21,
                "inTheMoney": False,
                "contractSize": "REGULAR",
            },
        ]
    )


def _puts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contractSymbol": "SPY260619P00500000",
                "strike": 500.0,
                "lastPrice": 9.8,
                "bid": 9.5,
                "ask": 10.1,
                "volume": 900,
                "openInterest": 5000,
                "impliedVolatility": 0.2,
                "inTheMoney": False,
                "contractSize": "REGULAR",
            }
        ]
    )


def test_parse_conforms_to_schema() -> None:
    df = parse_yf_chain(
        _calls(),
        _puts(),
        underlying="SPY",
        expiration=date(2026, 6, 19),
        snapshot_id="snap-yf",
        asof=pd.Timestamp("2026-05-24 20:00:00", tz="UTC"),
    )
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    validate_chain(df)
    assert len(df) == 3


def test_greeks_are_nan() -> None:
    df = parse_yf_chain(
        _calls(),
        _puts(),
        underlying="SPY",
        expiration=date(2026, 6, 19),
        snapshot_id="snap-yf",
        asof=pd.Timestamp("2026-05-24 20:00:00", tz="UTC"),
    )
    for greek in ("delta", "gamma", "theta", "vega", "rho"):
        assert df[greek].isna().all()


def test_zero_quote_becomes_nan() -> None:
    df = parse_yf_chain(
        _calls(),
        _puts(),
        underlying="SPY",
        expiration=date(2026, 6, 19),
        snapshot_id="snap-yf",
        asof=pd.Timestamp("2026-05-24 20:00:00", tz="UTC"),
    )
    row = df.loc[df["option_symbol"] == "SPY260619C00520000"].iloc[0]
    assert math.isnan(row["bid"])
    assert math.isnan(row["ask"])


def test_iv_is_decimal_and_provider_tagged() -> None:
    df = parse_yf_chain(
        _calls(),
        _puts(),
        underlying="SPY",
        expiration=date(2026, 6, 19),
        snapshot_id="snap-yf",
        asof=pd.Timestamp("2026-05-24 20:00:00", tz="UTC"),
    )
    assert (df["provider"] == "yfinance").all()
    call_500 = df.loc[df["option_symbol"] == "SPY260619C00500000"].iloc[0]
    assert call_500["iv"] == 0.185
