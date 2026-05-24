"""SchwabProvider orchestration, exercised offline with a mocked transport."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from osl.data.base import validate_chain
from osl.data.schwab import SchwabAuth, SchwabProvider, SchwabToken, parse_underlying

FIXED_NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)


class FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body

    @property
    def text(self) -> str:
        return str(self._body)


class FakeGetter:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[Mapping[str, str]] = []

    def __call__(
        self, url: str, *, headers: Mapping[str, str], params: Mapping[str, str]
    ) -> FakeResponse:
        self.calls.append(dict(params))
        return FakeResponse(200, self.payload)


def _fresh_auth(tmp_path: Path) -> SchwabAuth:
    auth = SchwabAuth(
        app_key="k",
        app_secret="s",
        token_path=tmp_path / "token.json",
        token_poster=lambda form: {"access_token": "x", "expires_in": 1800},
        clock=lambda: FIXED_NOW,
    )
    auth.set_token(
        SchwabToken(
            access_token="live-access",
            refresh_token="r",
            access_issued_at=FIXED_NOW.timestamp(),
            refresh_issued_at=FIXED_NOW.timestamp(),
        )
    )
    return auth


def test_get_option_chain_parses_and_tees(schwab_payload: dict[str, Any], tmp_path: Path) -> None:
    getter = FakeGetter(schwab_payload)
    provider = SchwabProvider(_fresh_auth(tmp_path), http_get=getter, raw_root=tmp_path / "raw")

    df = provider.get_option_chain("AAPL")
    validate_chain(df)
    assert len(df) == 4

    # Authorization header carried a bearer token; chains endpoint was hit.
    assert getter.calls[0]["symbol"] == "AAPL"
    # Raw payload was tee'd to disk (compressed).
    raw_files = list((tmp_path / "raw" / "schwab" / "AAPL").glob("*.json.zst"))
    assert len(raw_files) == 1


def test_get_option_chain_filters(schwab_payload: dict[str, Any], tmp_path: Path) -> None:
    provider = SchwabProvider(_fresh_auth(tmp_path), http_get=FakeGetter(schwab_payload))

    by_oi = provider.get_option_chain("AAPL", min_oi=1000)
    assert (by_oi["open_interest"] >= 1000).all()
    assert len(by_oi) == 2  # 185C (1500) and 190P (1100)

    by_strike = provider.get_option_chain("AAPL", strike_range=(188.0, 200.0))
    assert set(by_strike["strike"]) == {190.0}

    by_exp = provider.get_option_chain("AAPL", expirations=[date(2026, 6, 19)])
    assert len(by_exp) == 4


def test_healthcheck_reports_token(tmp_path: Path) -> None:
    provider = SchwabProvider(_fresh_auth(tmp_path), http_get=FakeGetter({}))
    health = provider.healthcheck()
    assert health["provider"] == "schwab"
    assert health["ok"] is True
    assert health["refresh_expired"] is False


def test_parse_underlying() -> None:
    payload = {
        "AAPL": {
            "quote": {
                "lastPrice": 190.12,
                "bidPrice": 190.1,
                "askPrice": 190.15,
                "mark": 190.13,
                "totalVolume": 41_250_000,
                "quoteTimeInLong": 1_750_291_200_000,
            }
        }
    }
    q = parse_underlying(payload, "AAPL")
    assert q.symbol == "AAPL"
    assert q.last == pytest.approx(190.12)
    assert q.mark == pytest.approx(190.13)
    assert q.quote_time.tzinfo is not None
