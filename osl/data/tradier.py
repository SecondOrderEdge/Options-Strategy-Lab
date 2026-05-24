"""Tradier market-data adapter (read-only chains).

Normalizes Tradier's ``markets/options/chains`` response (with greeks) to the
canonical chain schema. HTTP is injectable so the parser is unit-tested offline.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol

import numpy as np
import pandas as pd

from osl.data.base import CANONICAL_COLUMNS, UnderlyingQuote, coerce_chain, empty_chain
from osl.logging import get_logger
from osl.utils.time import now_utc

log = get_logger(__name__)
BASE_URL = "https://api.tradier.com/v1"


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...

    @property
    def text(self) -> str: ...


class HttpGetter(Protocol):
    def __call__(
        self, url: str, *, headers: Mapping[str, str], params: Mapping[str, str]
    ) -> HttpResponse: ...


def _num(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_chain(
    options: list[Mapping[str, Any]], snapshot_id: str, *, asof: date | None = None
) -> pd.DataFrame:
    """Parse a Tradier option list into the canonical chain schema."""
    today = asof or now_utc().date()
    rows: list[dict[str, Any]] = []
    for c in options:
        if c.get("option_type") not in ("call", "put"):
            continue
        right = "C" if c["option_type"] == "call" else "P"
        greeks = c.get("greeks") or {}
        bid, ask = _num(c.get("bid")), _num(c.get("ask"))
        mid = (bid + ask) / 2.0 if not (np.isnan(bid) or np.isnan(ask)) else float("nan")
        exp = pd.Timestamp(f"{c['expiration_date']} 16:00:00", tz="US/Eastern")
        dte = max((exp.date() - today).days, 0)
        rows.append(
            {
                "underlying": str(c.get("underlying", c.get("root_symbol", ""))),
                "option_symbol": str(c.get("symbol", "")),
                "expiration": exp,
                "dte": dte,
                "right": right,
                "strike": _num(c.get("strike")),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "last": _num(c.get("last")),
                "mark": mid,
                "bid_size": _int(c.get("bidsize")),
                "ask_size": _int(c.get("asksize")),
                "volume": _int(c.get("volume")),
                "open_interest": _int(c.get("open_interest")),
                "iv": _num(greeks.get("mid_iv")),
                "delta": _num(greeks.get("delta")),
                "gamma": _num(greeks.get("gamma")),
                "theta": _num(greeks.get("theta")),
                "vega": _num(greeks.get("vega")),
                "rho": _num(greeks.get("rho")),
                "theo": float("nan"),
                "multiplier": _int(c.get("contract_size"), default=100),
                "exchange": str(c.get("exch", "")),
                "quote_time": now_utc(),
                "is_nonstandard": _int(c.get("contract_size"), default=100) != 100,
                "in_the_money": False,  # Tradier omits ITM at the contract level
                "provider": "tradier",
                "snapshot_id": snapshot_id,
            }
        )
    if not rows:
        return empty_chain()
    return coerce_chain(pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS)))


class TradierProvider:
    """Tradier implementation of the DataProvider interface (chains, read-only)."""

    name = "tradier"

    def __init__(self, token: str, *, http_get: HttpGetter | None = None) -> None:
        self._token = token
        self._http_get = http_get

    def _get(self, path: str, params: Mapping[str, str]) -> Any:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if self._http_get is None:  # pragma: no cover - live path
            import requests

            return requests.get(
                f"{BASE_URL}{path}", headers=headers, params=dict(params), timeout=20
            ).json()
        return self._http_get(f"{BASE_URL}{path}", headers=headers, params=params).json()

    def get_expirations(self, symbol: str) -> list[date]:
        payload = self._get("/markets/options/expirations", {"symbol": symbol})
        dates = (payload.get("expirations") or {}).get("date", [])
        return [pd.Timestamp(d).date() for d in dates]

    def get_option_chain(
        self,
        symbol: str,
        *,
        expirations: list[date] | None = None,
        strike_range: tuple[float, float] | None = None,
        min_oi: int = 0,
    ) -> pd.DataFrame:
        import uuid

        exps = expirations or self.get_expirations(symbol)
        frames = []
        for exp in exps:
            payload = self._get(
                "/markets/options/chains",
                {"symbol": symbol, "expiration": exp.isoformat(), "greeks": "true"},
            )
            options = (payload.get("options") or {}).get("option", []) or []
            frames.append(parse_chain(options, str(uuid.uuid4())))
        if not frames:
            return empty_chain()
        df = pd.concat(frames, ignore_index=True)
        if strike_range is not None:
            low, high = strike_range
            df = df[(df["strike"] >= low) & (df["strike"] <= high)]
        if min_oi > 0:
            df = df[df["open_interest"] >= min_oi]
        return df.reset_index(drop=True)

    def get_underlying(self, symbol: str) -> UnderlyingQuote:
        payload = self._get("/markets/quotes", {"symbols": symbol})
        q = (payload.get("quotes") or {}).get("quote", {})
        return UnderlyingQuote(
            symbol=symbol,
            last=_num(q.get("last")),
            bid=_num(q.get("bid")),
            ask=_num(q.get("ask")),
            mark=_num(q.get("last")),
            volume=_int(q.get("volume")),
            quote_time=now_utc(),
            is_delayed=False,
        )

    def get_history(self, symbol: str, *, start: date, end: date, bar: str = "1d") -> pd.DataFrame:
        raise NotImplementedError("Tradier history not implemented (read-only chains adapter)")

    def healthcheck(self) -> dict[str, object]:
        return {"provider": self.name, "ok": bool(self._token)}
