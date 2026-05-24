"""Schwab Trader API adapter (primary provider).

Implements the :class:`~osl.data.base.DataProvider` interface against Schwab's
``marketdata/v1`` endpoints. Designed for offline testability: the OAuth token
exchange, the HTTP transport, the clock, and the RNG are all injectable, so
token-refresh logic, rate-limit backoff, and the chain parser can be unit
tested with golden fixtures and mocked endpoints (no live credentials needed).

Schwab facts encoded here (see blueprint §3.3):

* Access tokens are valid 30 minutes; refresh tokens 7 days.
* ``volatility`` is reported in **percent** (divide by 100).
* ``theta`` is **per day**; greeks use ``-999.0`` as a not-available sentinel.
* ``*ExpDateMap`` are nested ``{"YYYY-MM-DD:DTE": {"<strike>": [contract]}}``.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from osl.data.base import CANONICAL_COLUMNS, UnderlyingQuote, coerce_chain
from osl.logging import get_logger
from osl.utils.time import epoch_ms_to_utc, now_utc

log = get_logger(__name__)

BASE_URL = "https://api.schwabapi.com/marketdata/v1"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

ACCESS_TTL_SECONDS = 1800  # 30 minutes
ACCESS_REFRESH_MARGIN_SECONDS = 300  # refresh ~5 min early (=> at 25 min)
REFRESH_TTL_SECONDS = 7 * 86_400  # 7 days
NA_SENTINEL = -999.0
EASTERN_TZ = "US/Eastern"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class SchwabError(Exception):
    """Base error for the Schwab adapter."""


class SchwabAuthError(SchwabError):
    """OAuth / token-related failure."""


class RefreshTokenExpired(SchwabAuthError):
    """The refresh token is older than 7 days and re-authentication is required."""


class SchwabAPIError(SchwabError):
    """A non-success HTTP response that backoff could not resolve."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


# ---------------------------------------------------------------------------
# HTTP / token transport protocols (injectable)
# ---------------------------------------------------------------------------
class HttpResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...

    @property
    def text(self) -> str: ...


class HttpGetter(Protocol):
    def __call__(
        self, url: str, *, headers: Mapping[str, str], params: Mapping[str, str]
    ) -> HttpResponse: ...


TokenPoster = Callable[[Mapping[str, str]], dict[str, Any]]
"""Callable taking OAuth form data and returning the parsed token JSON."""

Clock = Callable[[], datetime]


# ---------------------------------------------------------------------------
# Token model + auth
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SchwabToken:
    """Persisted OAuth token state with issuance timestamps (epoch seconds)."""

    access_token: str
    refresh_token: str
    access_issued_at: float
    refresh_issued_at: float
    expires_in: int = ACCESS_TTL_SECONDS

    def to_json(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "access_issued_at": self.access_issued_at,
            "refresh_issued_at": self.refresh_issued_at,
            "expires_in": self.expires_in,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SchwabToken:
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            access_issued_at=float(data["access_issued_at"]),
            refresh_issued_at=float(data["refresh_issued_at"]),
            expires_in=int(data.get("expires_in", ACCESS_TTL_SECONDS)),
        )


def _default_token_poster(app_key: str, app_secret: str) -> TokenPoster:
    def poster(form: Mapping[str, str]) -> dict[str, Any]:
        import requests

        resp = requests.post(
            TOKEN_URL,
            data=dict(form),
            auth=(app_key, app_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if resp.status_code != 200:
            raise SchwabAuthError(f"token endpoint returned HTTP {resp.status_code}: {resp.text}")
        result: dict[str, Any] = resp.json()
        return result

    return poster


class SchwabAuth:
    """Manages the Schwab OAuth token lifecycle.

    Proactively refreshes the access token ~5 minutes before its 30-minute
    expiry and refuses to proceed once the refresh token's 7-day life is
    exhausted (raising :class:`RefreshTokenExpired`).
    """

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        token_path: str | Path,
        *,
        token_poster: TokenPoster | None = None,
        clock: Clock = now_utc,
    ) -> None:
        self._app_key = app_key
        self._app_secret = app_secret
        self._token_path = Path(token_path)
        self._token_poster = token_poster or _default_token_poster(app_key, app_secret)
        self._clock = clock
        self._token: SchwabToken | None = None

    # -- persistence --
    def load(self) -> SchwabToken | None:
        if self._token is not None:
            return self._token
        if self._token_path.exists():
            data = json.loads(self._token_path.read_text())
            self._token = SchwabToken.from_json(data)
        return self._token

    def save(self) -> None:
        if self._token is None:
            return
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        self._token_path.write_text(json.dumps(self._token.to_json(), indent=2))

    def set_token(self, token: SchwabToken) -> None:
        """Directly install a token (used in tests and after initial auth)."""
        self._token = token

    # -- lifecycle checks --
    def _now(self) -> float:
        return self._clock().timestamp()

    def _needs_access_refresh(self, token: SchwabToken, now: float) -> bool:
        age = now - token.access_issued_at
        return age >= (token.expires_in - ACCESS_REFRESH_MARGIN_SECONDS)

    def _refresh_expired(self, token: SchwabToken, now: float) -> bool:
        return (now - token.refresh_issued_at) >= REFRESH_TTL_SECONDS

    def refresh(self) -> SchwabToken:
        """Exchange the refresh token for a new access token."""
        token = self.load()
        if token is None:
            raise SchwabAuthError("no token available; complete the OAuth flow first")
        now = self._now()
        if self._refresh_expired(token, now):
            raise RefreshTokenExpired("refresh token older than 7 days; re-authenticate")

        payload = self._token_poster(
            {"grant_type": "refresh_token", "refresh_token": token.refresh_token}
        )
        new_token = replace(
            token,
            access_token=str(payload["access_token"]),
            access_issued_at=now,
            # Schwab may rotate the refresh token; keep the new one if present.
            refresh_token=str(payload.get("refresh_token", token.refresh_token)),
            expires_in=int(payload.get("expires_in", ACCESS_TTL_SECONDS)),
        )
        self._token = new_token
        self.save()
        log.info("schwab_access_token_refreshed")
        return new_token

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing proactively if needed."""
        token = self.load()
        if token is None:
            raise SchwabAuthError("no token available; complete the OAuth flow first")
        now = self._now()
        if self._refresh_expired(token, now):
            raise RefreshTokenExpired("refresh token older than 7 days; re-authenticate")
        if self._needs_access_refresh(token, now):
            token = self.refresh()
        return token.access_token


# ---------------------------------------------------------------------------
# Rate-limit backoff
# ---------------------------------------------------------------------------
def request_with_backoff(
    send: Callable[[], HttpResponse],
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    retry_statuses: tuple[int, ...] = (429, 503),
    jitter: float = 0.5,
    sleep_fn: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> HttpResponse:
    """Send a request, retrying retryable statuses with exponential backoff + jitter.

    ``send`` is a zero-arg callable performing one HTTP attempt. Sleeps grow as
    ``base_delay * 2**attempt`` (capped at ``max_delay``) plus uniform jitter in
    ``[0, jitter * delay]``. Raises :class:`SchwabAPIError` if all retries are
    exhausted on a retryable status.
    """
    rng = rng or random.Random()
    last: HttpResponse | None = None
    for attempt in range(max_retries + 1):
        resp = send()
        last = resp
        if resp.status_code not in retry_statuses:
            return resp
        if attempt == max_retries:
            break
        delay = min(max_delay, base_delay * (2.0**attempt))
        delay += rng.uniform(0.0, jitter * delay)
        log.warning("schwab_rate_limited", status=resp.status_code, attempt=attempt, delay=delay)
        sleep_fn(delay)
    assert last is not None
    raise SchwabAPIError(last.status_code, "rate limit not resolved after retries")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _num(value: Any) -> float:
    """Convert a Schwab numeric to float, mapping None/sentinel to NaN."""
    if value is None:
        return float("nan")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if f == NA_SENTINEL:
        return float("nan")
    return f


def _price(value: Any, size: Any) -> float:
    """A bid/ask price; treat 0.0 with zero size as 'no quote' (NaN)."""
    f = _num(value)
    if f == 0.0 and _int(size) == 0:
        return float("nan")
    return f


def _int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _expiration_timestamp(date_str: str) -> pd.Timestamp:
    """Build the 4:00 PM ET expiry timestamp from a ``YYYY-MM-DD`` string."""
    return pd.Timestamp(f"{date_str} 16:00:00", tz=EASTERN_TZ)


def parse_chain(payload: Mapping[str, Any], snapshot_id: str) -> pd.DataFrame:
    """Parse a Schwab ``chains`` payload into the canonical chain DataFrame."""
    underlying = str(payload.get("symbol", ""))
    rows: list[dict[str, Any]] = []

    for map_key, exp_map in (("callExpDateMap", "C"), ("putExpDateMap", "P")):
        right = exp_map
        for exp_date_key, strike_map in payload.get(map_key, {}).items():
            date_str = exp_date_key.split(":")[0]
            expiration = _expiration_timestamp(date_str)
            for _strike_str, contracts in strike_map.items():
                for c in contracts:
                    bid = _price(c.get("bid"), c.get("bidSize"))
                    ask = _price(c.get("ask"), c.get("askSize"))
                    mid = (
                        (bid + ask) / 2.0 if not (np.isnan(bid) or np.isnan(ask)) else float("nan")
                    )

                    iv = _num(c.get("volatility")) / 100.0
                    if not np.isnan(iv) and iv <= 0.0:
                        iv = float("nan")

                    qt_ms = _int(c.get("quoteTimeInLong")) or _int(c.get("tradeTimeInLong"))
                    quote_time = epoch_ms_to_utc(qt_ms) if qt_ms else now_utc()

                    rows.append(
                        {
                            "underlying": underlying,
                            "option_symbol": str(c.get("symbol", "")),
                            "expiration": expiration,
                            "dte": _int(c.get("daysToExpiration")),
                            "right": right,
                            "strike": _num(c.get("strikePrice")),
                            "bid": bid,
                            "ask": ask,
                            "mid": mid,
                            "last": _num(c.get("last")),
                            "mark": _num(c.get("mark")),
                            "bid_size": _int(c.get("bidSize")),
                            "ask_size": _int(c.get("askSize")),
                            "volume": _int(c.get("totalVolume")),
                            "open_interest": _int(c.get("openInterest")),
                            "iv": iv,
                            "delta": _num(c.get("delta")),
                            "gamma": _num(c.get("gamma")),
                            "theta": _num(c.get("theta")),  # already per-day
                            "vega": _num(c.get("vega")),
                            "rho": _num(c.get("rho")),
                            "theo": _num(c.get("theoreticalOptionValue")),
                            "multiplier": _int(c.get("multiplier"), default=100),
                            "exchange": str(c.get("exchangeName", "")),
                            "quote_time": quote_time,
                            "is_nonstandard": bool(c.get("nonStandard", False)),
                            "in_the_money": bool(c.get("inTheMoney", False)),
                            "provider": "schwab",
                            "snapshot_id": snapshot_id,
                        }
                    )

    if not rows:
        from osl.data.base import empty_chain

        return empty_chain()
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    return coerce_chain(df)


def parse_underlying(payload: Mapping[str, Any], symbol: str) -> UnderlyingQuote:
    """Parse a Schwab quotes payload into an :class:`UnderlyingQuote`."""
    entry = payload.get(symbol, payload)
    quote = entry.get("quote", entry)
    qt_ms = _int(quote.get("quoteTimeInLong")) or _int(quote.get("tradeTimeInLong"))
    return UnderlyingQuote(
        symbol=symbol,
        last=_num(quote.get("lastPrice", quote.get("last"))),
        bid=_num(quote.get("bidPrice", quote.get("bid"))),
        ask=_num(quote.get("askPrice", quote.get("ask"))),
        mark=_num(quote.get("mark")),
        volume=_int(quote.get("totalVolume")),
        quote_time=epoch_ms_to_utc(qt_ms) if qt_ms else now_utc(),
        is_delayed=bool(entry.get("realtime") is False or quote.get("delayed", False)),
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class SchwabProvider:
    """Schwab Trader API implementation of :class:`DataProvider`."""

    name = "schwab"

    def __init__(
        self,
        auth: SchwabAuth,
        *,
        http_get: HttpGetter | None = None,
        raw_root: str | Path | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._auth = auth
        self._http_get = http_get or _default_http_getter()
        self._raw_root = Path(raw_root) if raw_root is not None else None
        self._sleep_fn = sleep_fn
        self._rng = rng

    # -- internal request --
    def _get(self, path: str, params: Mapping[str, str]) -> HttpResponse:
        token = self._auth.get_access_token()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        url = f"{BASE_URL}{path}"

        def send() -> HttpResponse:
            return self._http_get(url, headers=headers, params=params)

        resp = request_with_backoff(send, sleep_fn=self._sleep_fn, rng=self._rng)
        if resp.status_code != 200:
            raise SchwabAPIError(resp.status_code, resp.text)
        return resp

    def _tee_raw(self, symbol: str, snapshot_id: str, payload: Any) -> None:
        if self._raw_root is None:
            return
        try:
            import zstandard as zstd

            out_dir = self._raw_root / "schwab" / symbol
            out_dir.mkdir(parents=True, exist_ok=True)
            blob = json.dumps(payload).encode("utf-8")
            compressed = zstd.ZstdCompressor().compress(blob)
            (out_dir / f"{snapshot_id}.json.zst").write_bytes(compressed)
        except Exception as exc:  # teeing must never break a fetch
            log.warning("schwab_raw_tee_failed", error=str(exc))

    # -- DataProvider interface --
    def get_option_chain(
        self,
        symbol: str,
        *,
        expirations: list[date] | None = None,
        strike_range: tuple[float, float] | None = None,
        min_oi: int = 0,
    ) -> pd.DataFrame:
        import uuid

        params = {
            "symbol": symbol,
            "contractType": "ALL",
            "strategy": "SINGLE",
            "includeUnderlyingQuote": "true",
            "range": "ALL",
        }
        resp = self._get("/chains", params)
        payload = resp.json()
        snapshot_id = str(uuid.uuid4())
        self._tee_raw(symbol, snapshot_id, payload)

        if payload.get("isDelayed"):
            log.warning("schwab_chain_delayed", symbol=symbol)

        df = parse_chain(payload, snapshot_id)
        if expirations is not None:
            wanted = {pd.Timestamp(e).date() for e in expirations}
            df = df[df["expiration"].dt.date.isin(wanted)]
        if strike_range is not None:
            low, high = strike_range
            df = df[(df["strike"] >= low) & (df["strike"] <= high)]
        if min_oi > 0:
            df = df[df["open_interest"] >= min_oi]
        return df.reset_index(drop=True)

    def get_underlying(self, symbol: str) -> UnderlyingQuote:
        resp = self._get("/quotes", {"symbols": symbol})
        return parse_underlying(resp.json(), symbol)

    def get_expirations(self, symbol: str) -> list[date]:
        resp = self._get("/expirationchain", {"symbol": symbol})
        payload = resp.json()
        out: list[date] = []
        for item in payload.get("expirationList", []):
            ds = item.get("expirationDate")
            if ds:
                out.append(pd.Timestamp(ds).date())
        return sorted(set(out))

    def get_history(self, symbol: str, *, start: date, end: date, bar: str = "1d") -> pd.DataFrame:
        freq = {"1d": ("daily", 1), "1w": ("weekly", 1)}.get(bar, ("daily", 1))
        params = {
            "symbol": symbol,
            "periodType": "year",
            "frequencyType": freq[0],
            "frequency": str(freq[1]),
            "startDate": str(int(pd.Timestamp(start).timestamp() * 1000)),
            "endDate": str(int(pd.Timestamp(end).timestamp() * 1000)),
        }
        resp = self._get("/pricehistory", params)
        payload = resp.json()
        candles = payload.get("candles", [])
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(candles)
        df["ts"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("ts")[["open", "high", "low", "close", "volume"]]
        return df

    def healthcheck(self) -> dict[str, object]:
        status: dict[str, object] = {"provider": self.name}
        try:
            token = self._auth.load()
            status["has_token"] = token is not None
            if token is not None:
                now = self._auth._now()  # internal status read
                status["refresh_expired"] = self._auth._refresh_expired(token, now)
            status["ok"] = token is not None
        except SchwabError as exc:
            status["ok"] = False
            status["error"] = str(exc)
        return status


def _default_http_getter() -> HttpGetter:
    def getter(url: str, *, headers: Mapping[str, str], params: Mapping[str, str]) -> HttpResponse:
        import requests

        return requests.get(url, headers=dict(headers), params=dict(params), timeout=20)

    return getter
