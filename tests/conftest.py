"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from osl.data.base import CANONICAL_COLUMNS, coerce_chain
from osl.pricing.bsm import bsm_price

FIXTURES_DIR = Path(__file__).parent / "data"
_BASE_DATE = date(2026, 6, 1)


def build_synthetic_chain(
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    dte_list: Sequence[int] = (30,),
    strikes: Sequence[float],
    iv: float = 0.20,
    iv_fn: Callable[[float, int], float] | None = None,
    open_interest: int = 1000,
    volume: int = 250,
    spread: float = 0.05,
    zero_bid: Sequence[tuple[int, float, str]] = (),
) -> tuple[pd.DataFrame, dict[int, date]]:
    """Build a canonical chain priced via BSM, plus a {dte: expiration date} map.

    ``zero_bid`` lists (dte, strike, right) contracts whose bid is forced to 0.
    """
    exp_for_dte = {dte: _BASE_DATE + timedelta(days=dte) for dte in dte_list}
    zb = set(zero_bid)
    rows: list[dict[str, Any]] = []
    quote_time = pd.Timestamp("2026-06-01 20:00:00", tz="UTC")

    for dte in dte_list:
        T = dte / 365.0
        exp_ts = pd.Timestamp(f"{exp_for_dte[dte].isoformat()} 16:00:00", tz="US/Eastern")
        for strike in strikes:
            sigma = iv_fn(strike, dte) if iv_fn is not None else iv
            for right, flag in (("C", "c"), ("P", "p")):
                mid = float(bsm_price(spot, strike, T, rate, dividend_yield, sigma, flag).item())
                bid = max(mid - spread / 2, 0.0)
                ask = mid + spread / 2
                if (dte, strike, right) in zb:
                    bid = 0.0
                rows.append(
                    {
                        "underlying": "TST",
                        "option_symbol": f"TST{dte}{right}{int(strike * 1000):08d}",
                        "expiration": exp_ts,
                        "dte": dte,
                        "right": right,
                        "strike": float(strike),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "last": mid,
                        "mark": mid,
                        "bid_size": 10,
                        "ask_size": 10,
                        "volume": volume,
                        "open_interest": open_interest,
                        "iv": sigma,
                        "delta": np.nan,
                        "gamma": np.nan,
                        "theta": np.nan,
                        "vega": np.nan,
                        "rho": np.nan,
                        "theo": np.nan,
                        "multiplier": 100,
                        "exchange": "OPR",
                        "quote_time": quote_time,
                        "is_nonstandard": False,
                        "in_the_money": False,
                        "provider": "schwab",
                        "snapshot_id": "synthetic",
                    }
                )
    chain = coerce_chain(pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS)))
    return chain, exp_for_dte


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def schwab_payload() -> dict[str, Any]:
    """The golden Schwab `chains` response for AAPL."""
    data: dict[str, Any] = json.loads((FIXTURES_DIR / "schwab_chain_AAPL.json").read_text())
    return data
