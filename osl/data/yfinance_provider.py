"""yfinance adapter (fallback provider).

Normalizes Yahoo option chains to the canonical schema. yfinance does **not**
supply greeks, mark, or an exchange, so those canonical columns are emitted as
NaN/empty in M0 — greeks are computed downstream by ``osl.pricing.bsm`` (M1).
The chain parser is split out so it can be unit-tested with synthetic
DataFrames without any network access.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from osl.data.base import CANONICAL_COLUMNS, UnderlyingQuote, coerce_chain, empty_chain
from osl.logging import get_logger
from osl.utils.time import now_utc

if TYPE_CHECKING:
    import yfinance as yf

log = get_logger(__name__)
EASTERN_TZ = "US/Eastern"


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _i(value: Any) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_side(
    df: pd.DataFrame,
    right: str,
    *,
    underlying: str,
    expiration: pd.Timestamp,
    dte: int,
    snapshot_id: str,
    quote_time: pd.Timestamp,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, r in df.iterrows():
        bid = _f(r.get("bid"))
        ask = _f(r.get("ask"))
        bid = float("nan") if bid == 0.0 else bid
        ask = float("nan") if ask == 0.0 else ask
        mid = (bid + ask) / 2.0 if not (np.isnan(bid) or np.isnan(ask)) else float("nan")
        iv = _f(r.get("impliedVolatility"))
        if not np.isnan(iv) and iv <= 0.0:
            iv = float("nan")
        contract_size = str(r.get("contractSize", "REGULAR"))
        rows.append(
            {
                "underlying": underlying,
                "option_symbol": str(r.get("contractSymbol", "")),
                "expiration": expiration,
                "dte": dte,
                "right": right,
                "strike": _f(r.get("strike")),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "last": _f(r.get("lastPrice")),
                "mark": float("nan"),  # yfinance has no mark
                "bid_size": 0,  # yfinance does not expose sizes
                "ask_size": 0,
                "volume": _i(r.get("volume")),
                "open_interest": _i(r.get("openInterest")),
                "iv": iv,  # yfinance IV is already decimal
                "delta": float("nan"),  # greeks computed downstream (M1)
                "gamma": float("nan"),
                "theta": float("nan"),
                "vega": float("nan"),
                "rho": float("nan"),
                "theo": float("nan"),
                "multiplier": 100,  # standard equity; non-standard flagged separately
                "exchange": "",
                "quote_time": quote_time,
                "is_nonstandard": contract_size.upper() != "REGULAR",
                "in_the_money": bool(r.get("inTheMoney", False)),
                "provider": "yfinance",
                "snapshot_id": snapshot_id,
            }
        )
    return rows


def parse_yf_chain(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    *,
    underlying: str,
    expiration: date,
    snapshot_id: str,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Parse yfinance ``(calls, puts)`` frames into the canonical chain schema."""
    quote_time = asof if asof is not None else pd.Timestamp(now_utc())
    if quote_time.tzinfo is None:
        quote_time = quote_time.tz_localize("UTC")
    else:
        quote_time = quote_time.tz_convert("UTC")

    exp_ts = pd.Timestamp(f"{expiration.isoformat()} 16:00:00", tz=EASTERN_TZ)
    dte = max((expiration - quote_time.tz_convert(EASTERN_TZ).date()).days, 0)

    rows = _parse_side(
        calls,
        "C",
        underlying=underlying,
        expiration=exp_ts,
        dte=dte,
        snapshot_id=snapshot_id,
        quote_time=quote_time,
    )
    rows += _parse_side(
        puts,
        "P",
        underlying=underlying,
        expiration=exp_ts,
        dte=dte,
        snapshot_id=snapshot_id,
        quote_time=quote_time,
    )
    if not rows:
        return empty_chain()
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    return coerce_chain(df)


class YFinanceProvider:
    """yfinance implementation of :class:`DataProvider`."""

    name = "yfinance"

    def __init__(self, *, ticker_factory: type[yf.Ticker] | None = None) -> None:
        self._ticker_factory = ticker_factory

    def _ticker(self, symbol: str) -> yf.Ticker:
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        import yfinance as yf

        return yf.Ticker(symbol)

    def get_expirations(self, symbol: str) -> list[date]:
        options = self._ticker(symbol).options
        return [pd.Timestamp(s).date() for s in options]

    def get_underlying(self, symbol: str) -> UnderlyingQuote:
        t = self._ticker(symbol)
        info = t.fast_info
        last = _f(getattr(info, "last_price", float("nan")))
        return UnderlyingQuote(
            symbol=symbol,
            last=last,
            bid=_f(getattr(info, "bid", float("nan"))),
            ask=_f(getattr(info, "ask", float("nan"))),
            mark=last,
            volume=_i(getattr(info, "last_volume", 0)),
            quote_time=now_utc(),
            is_delayed=True,  # Yahoo is delayed; surface honestly
        )

    def get_history(self, symbol: str, *, start: date, end: date, bar: str = "1d") -> pd.DataFrame:
        interval = {"1d": "1d", "1w": "1wk"}.get(bar, "1d")
        hist = self._ticker(symbol).history(start=start, end=end, interval=interval)
        hist = hist.rename(columns=str.lower)
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in hist.columns]
        result: pd.DataFrame = hist[cols]
        return result

    def get_option_chain(
        self,
        symbol: str,
        *,
        expirations: list[date] | None = None,
        strike_range: tuple[float, float] | None = None,
        min_oi: int = 0,
    ) -> pd.DataFrame:
        import uuid

        t = self._ticker(symbol)
        all_exps = [pd.Timestamp(s).date() for s in t.options]
        wanted = set(expirations) if expirations is not None else set(all_exps)
        frames: list[pd.DataFrame] = []
        for exp in all_exps:
            if exp not in wanted:
                continue
            chain = t.option_chain(exp.isoformat())
            frames.append(
                parse_yf_chain(
                    chain.calls,
                    chain.puts,
                    underlying=symbol,
                    expiration=exp,
                    snapshot_id=str(uuid.uuid4()),
                )
            )
        if not frames:
            return empty_chain()
        df = pd.concat(frames, ignore_index=True)
        if strike_range is not None:
            low, high = strike_range
            df = df[(df["strike"] >= low) & (df["strike"] <= high)]
        if min_oi > 0:
            df = df[df["open_interest"] >= min_oi]
        return df.reset_index(drop=True)

    def healthcheck(self) -> dict[str, object]:
        try:
            import yfinance  # noqa: F401

            return {"provider": self.name, "ok": True}
        except ImportError as exc:
            return {"provider": self.name, "ok": False, "error": str(exc)}
