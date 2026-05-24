"""Schwab chain parser: unit normalization, sentinels, and price handling."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from osl.data.schwab import parse_chain


@pytest.fixture
def parsed(schwab_payload: dict[str, Any]) -> pd.DataFrame:
    return parse_chain(schwab_payload, snapshot_id="snap-xyz")


def _row(df: pd.DataFrame, symbol: str) -> pd.Series:
    return df.loc[df["option_symbol"] == symbol].iloc[0]


def test_row_count(parsed: pd.DataFrame) -> None:
    assert len(parsed) == 4
    assert set(parsed["right"]) == {"C", "P"}


def test_iv_percent_to_decimal(parsed: pd.DataFrame) -> None:
    # 190 call has volatility=25.5 (percent) -> 0.255 decimal.
    call_190 = _row(parsed, "AAPL  260619C00190000")
    assert call_190["iv"] == pytest.approx(0.255)


def test_theta_is_per_day_passthrough(parsed: pd.DataFrame) -> None:
    # Schwab reports per-day theta; adapter passes it through unchanged.
    call_185 = _row(parsed, "AAPL  260619C00185000")
    assert call_185["theta"] == pytest.approx(-0.045)


def test_zero_bid_zero_size_is_nan(parsed: pd.DataFrame) -> None:
    call_190 = _row(parsed, "AAPL  260619C00190000")
    assert math.isnan(call_190["bid"])
    assert math.isnan(call_190["ask"])
    assert math.isnan(call_190["mid"])


def test_greek_sentinel_becomes_nan(parsed: pd.DataFrame) -> None:
    # 185 put has delta=-999.0 sentinel -> NaN.
    put_185 = _row(parsed, "AAPL  260619P00185000")
    assert math.isnan(put_185["delta"])
    # Other greeks remain populated.
    assert put_185["gamma"] == pytest.approx(0.03)


def test_mid_computed_from_bid_ask(parsed: pd.DataFrame) -> None:
    call_185 = _row(parsed, "AAPL  260619C00185000")
    assert call_185["mid"] == pytest.approx((6.2 + 6.4) / 2)


def test_expiration_is_eastern_four_pm(parsed: pd.DataFrame) -> None:
    exp = parsed["expiration"].iloc[0]
    assert str(exp.tz) == "US/Eastern"
    assert exp.hour == 16
    assert exp.date().isoformat() == "2026-06-19"


def test_snapshot_id_and_provider_propagate(parsed: pd.DataFrame) -> None:
    assert (parsed["snapshot_id"] == "snap-xyz").all()
    assert (parsed["provider"] == "schwab").all()


def test_flags_parsed(parsed: pd.DataFrame) -> None:
    call_185 = _row(parsed, "AAPL  260619C00185000")
    assert bool(call_185["in_the_money"]) is True
    assert bool(call_185["is_nonstandard"]) is False
