"""Tradier and MarketData.app chain parsers + provider wiring (offline)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from osl.data.base import validate_chain
from osl.data.marketdata import MarketDataProvider
from osl.data.marketdata import parse_chain as parse_marketdata
from osl.data.tradier import TradierProvider
from osl.data.tradier import parse_chain as parse_tradier


class FakeResponse:
    def __init__(self, body: Any) -> None:
        self._body = body
        self.status_code = 200

    def json(self) -> Any:
        return self._body

    @property
    def text(self) -> str:
        return str(self._body)


# --- Tradier ---
def _tradier_options() -> list[dict[str, Any]]:
    out = []
    for strike in (95.0, 100.0, 105.0):
        for opt_type in ("call", "put"):
            out.append(
                {
                    "symbol": f"SPY260619{opt_type[0].upper()}{int(strike)}",
                    "underlying": "SPY",
                    "option_type": opt_type,
                    "strike": strike,
                    "bid": 1.0,
                    "ask": 1.2,
                    "last": 1.1,
                    "bidsize": 10,
                    "asksize": 12,
                    "volume": 100,
                    "open_interest": 500,
                    "contract_size": 100,
                    "exch": "Z",
                    "expiration_date": "2026-06-19",
                    "greeks": {
                        "delta": 0.5,
                        "gamma": 0.03,
                        "theta": -0.02,
                        "vega": 0.1,
                        "rho": 0.05,
                        "mid_iv": 0.2,
                    },
                }
            )
    return out


def test_tradier_parse_conforms():
    df = parse_tradier(_tradier_options(), "snap", asof=date(2026, 5, 24))
    validate_chain(df)
    assert len(df) == 6
    assert (df["provider"] == "tradier").all()
    row = df[df["right"] == "C"].iloc[0]
    assert row["iv"] == pytest.approx(0.2)
    assert row["delta"] == pytest.approx(0.5)
    assert row["dte"] == 26


def test_tradier_provider_via_fake_http():
    def http_get(url: str, *, headers: Any, params: Any) -> FakeResponse:
        if "expirations" in url:
            return FakeResponse({"expirations": {"date": ["2026-06-19"]}})
        return FakeResponse({"options": {"option": _tradier_options()}})

    provider = TradierProvider("tok", http_get=http_get)
    chain = provider.get_option_chain("SPY")
    validate_chain(chain)
    assert len(chain) == 6
    assert provider.healthcheck()["ok"] is True


# --- MarketData.app ---
def _marketdata_payload() -> dict[str, Any]:
    exp = 1781913600  # 2026-06-19 epoch (approx)
    return {
        "s": "ok",
        "optionSymbol": ["SPY260619C100", "SPY260619P100"],
        "underlying": ["SPY", "SPY"],
        "expiration": [exp, exp],
        "side": ["call", "put"],
        "strike": [100.0, 100.0],
        "dte": [26, 26],
        "bid": [1.0, 0.9],
        "bidSize": [10, 8],
        "ask": [1.2, 1.1],
        "askSize": [12, 9],
        "last": [1.1, 1.0],
        "openInterest": [500, 400],
        "volume": [100, 90],
        "inTheMoney": [False, False],
        "iv": [0.2, 0.21],
        "delta": [0.5, -0.5],
        "gamma": [0.03, 0.03],
        "theta": [-0.02, -0.02],
        "vega": [0.1, 0.1],
        "rho": [0.05, -0.05],
    }


def test_marketdata_parse_conforms():
    df = parse_marketdata(_marketdata_payload(), "snap")
    validate_chain(df)
    assert len(df) == 2
    assert (df["provider"] == "marketdata").all()
    assert set(df["right"]) == {"C", "P"}
    assert df["dte"].iloc[0] == 26


def test_marketdata_provider_via_fake_http():
    def http_get(url: str, *, headers: Any, params: Any) -> FakeResponse:
        return FakeResponse(_marketdata_payload())

    provider = MarketDataProvider("tok", http_get=http_get)
    chain = provider.get_option_chain("SPY")
    validate_chain(chain)
    assert len(chain) == 2
