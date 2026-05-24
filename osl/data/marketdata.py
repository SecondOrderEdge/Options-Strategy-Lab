"""MarketData.app adapter (read-only chains).

Normalizes MarketData.app's columnar options-chain response to the canonical
schema. HTTP is injectable so the parser is unit-tested offline.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol

import numpy as np
import pandas as pd

from osl.data.base import CANONICAL_COLUMNS, UnderlyingQuote, coerce_chain, empty_chain
from osl.logging import get_logger
from osl.utils.time import epoch_ms_to_utc, now_utc

log = get_logger(__name__)
BASE_URL = "https://api.marketdata.app/v1"


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...

    @property
    def text(self) -> str: ...


class HttpGetter(Protocol):
    def __call__(
        self, url: str, *, headers: Mapping[str, str], params: Mapping[str, str]
    ) -> HttpResponse: ...


def _col(payload: Mapping[str, Any], key: str, n: int) -> list[Any]:
    val = payload.get(key)
    return list(val) if isinstance(val, list) else [None] * n


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


def parse_chain(payload: Mapping[str, Any], snapshot_id: str) -> pd.DataFrame:
    """Parse a MarketData.app columnar chain response into the canonical schema."""
    symbols = payload.get("optionSymbol") or []
    n = len(symbols)
    if n == 0:
        return empty_chain()

    cols = {
        k: _col(payload, k, n)
        for k in (
            "underlying",
            "expiration",
            "side",
            "strike",
            "dte",
            "bid",
            "bidSize",
            "ask",
            "askSize",
            "last",
            "openInterest",
            "volume",
            "inTheMoney",
            "iv",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
        )
    }
    rows: list[dict[str, Any]] = []
    for i in range(n):
        side = cols["side"][i]
        right = "C" if side == "call" else "P"
        exp_epoch = cols["expiration"][i]
        exp_date = epoch_ms_to_utc(int(exp_epoch) * 1000).date() if exp_epoch else now_utc().date()
        exp_ts = pd.Timestamp(f"{exp_date.isoformat()} 16:00:00", tz="US/Eastern")
        bid, ask = _num(cols["bid"][i]), _num(cols["ask"][i])
        mid = (bid + ask) / 2.0 if not (np.isnan(bid) or np.isnan(ask)) else float("nan")
        rows.append(
            {
                "underlying": str(cols["underlying"][i] or ""),
                "option_symbol": str(symbols[i]),
                "expiration": exp_ts,
                "dte": _int(cols["dte"][i]),
                "right": right,
                "strike": _num(cols["strike"][i]),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "last": _num(cols["last"][i]),
                "mark": mid,
                "bid_size": _int(cols["bidSize"][i]),
                "ask_size": _int(cols["askSize"][i]),
                "volume": _int(cols["volume"][i]),
                "open_interest": _int(cols["openInterest"][i]),
                "iv": _num(cols["iv"][i]),
                "delta": _num(cols["delta"][i]),
                "gamma": _num(cols["gamma"][i]),
                "theta": _num(cols["theta"][i]),
                "vega": _num(cols["vega"][i]),
                "rho": _num(cols["rho"][i]),
                "theo": float("nan"),
                "multiplier": 100,
                "exchange": "",
                "quote_time": now_utc(),
                "is_nonstandard": False,
                "in_the_money": bool(cols["inTheMoney"][i]),
                "provider": "marketdata",
                "snapshot_id": snapshot_id,
            }
        )
    return coerce_chain(pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS)))


class MarketDataProvider:
    """MarketData.app implementation of the DataProvider interface (chains)."""

    name = "marketdata"

    def __init__(self, token: str, *, http_get: HttpGetter | None = None) -> None:
        self._token = token
        self._http_get = http_get

    def _get(self, path: str, params: Mapping[str, str]) -> Any:
        headers = {"Authorization": f"Token {self._token}", "Accept": "application/json"}
        if self._http_get is None:  # pragma: no cover - live path
            import requests

            return requests.get(
                f"{BASE_URL}{path}", headers=headers, params=dict(params), timeout=20
            ).json()
        return self._http_get(f"{BASE_URL}{path}", headers=headers, params=params).json()

    def get_option_chain(
        self,
        symbol: str,
        *,
        expirations: list[date] | None = None,
        strike_range: tuple[float, float] | None = None,
        min_oi: int = 0,
    ) -> pd.DataFrame:
        import uuid

        payload = self._get(f"/options/chain/{symbol}/", {})
        df = parse_chain(payload, str(uuid.uuid4()))
        if expirations is not None:
            wanted = {pd.Timestamp(e).date() for e in expirations}
            df = df[df["expiration"].dt.date.isin(wanted)]
        if strike_range is not None:
            low, high = strike_range
            df = df[(df["strike"] >= low) & (df["strike"] <= high)]
        if min_oi > 0:
            df = df[df["open_interest"] >= min_oi]
        return df.reset_index(drop=True)

    def get_expirations(self, symbol: str) -> list[date]:
        payload = self._get(f"/options/expirations/{symbol}/", {})
        return [pd.Timestamp(d).date() for d in (payload.get("expirations") or [])]

    def get_underlying(self, symbol: str) -> UnderlyingQuote:
        payload = self._get(f"/stocks/quotes/{symbol}/", {})
        last = _num((payload.get("last") or [float("nan")])[0])
        return UnderlyingQuote(
            symbol=symbol,
            last=last,
            bid=_num((payload.get("bid") or [float("nan")])[0]),
            ask=_num((payload.get("ask") or [float("nan")])[0]),
            mark=last,
            volume=_int((payload.get("volume") or [0])[0]),
            quote_time=now_utc(),
            is_delayed=False,
        )

    def get_history(self, symbol: str, *, start: date, end: date, bar: str = "1d") -> pd.DataFrame:
        raise NotImplementedError("MarketData history not implemented (read-only chains adapter)")

    def healthcheck(self) -> dict[str, object]:
        return {"provider": self.name, "ok": bool(self._token)}
