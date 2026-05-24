"""Schwab OAuth token lifecycle: proactive 30-min refresh, 7-day expiry, persistence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from osl.data.schwab import (
    REFRESH_TTL_SECONDS,
    RefreshTokenExpired,
    SchwabAuth,
    SchwabToken,
)

FIXED_NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class RecordingPoster:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, str]] = []

    def __call__(self, form: Mapping[str, str]) -> dict[str, Any]:
        self.calls.append(form)
        return {"access_token": "new-access", "expires_in": 1800}


def _make_auth(
    tmp_path: Path, clock: Clock, poster: RecordingPoster, token: SchwabToken
) -> SchwabAuth:
    auth = SchwabAuth(
        app_key="key",
        app_secret="secret",
        token_path=tmp_path / "token.json",
        token_poster=poster,
        clock=clock,
    )
    auth.set_token(token)
    return auth


def test_proactive_refresh_at_25_minutes(tmp_path: Path) -> None:
    clock = Clock(FIXED_NOW)
    poster = RecordingPoster()
    token = SchwabToken(
        access_token="old-access",
        refresh_token="refresh",
        access_issued_at=(FIXED_NOW - timedelta(minutes=26)).timestamp(),
        refresh_issued_at=(FIXED_NOW - timedelta(days=1)).timestamp(),
    )
    auth = _make_auth(tmp_path, clock, poster, token)

    access = auth.get_access_token()
    assert access == "new-access"
    assert len(poster.calls) == 1
    assert poster.calls[0]["grant_type"] == "refresh_token"


def test_fresh_access_token_not_refreshed(tmp_path: Path) -> None:
    clock = Clock(FIXED_NOW)
    poster = RecordingPoster()
    token = SchwabToken(
        access_token="still-good",
        refresh_token="refresh",
        access_issued_at=(FIXED_NOW - timedelta(minutes=10)).timestamp(),
        refresh_issued_at=(FIXED_NOW - timedelta(days=1)).timestamp(),
    )
    auth = _make_auth(tmp_path, clock, poster, token)

    assert auth.get_access_token() == "still-good"
    assert poster.calls == []


def test_refresh_token_expired_raises(tmp_path: Path) -> None:
    clock = Clock(FIXED_NOW)
    poster = RecordingPoster()
    token = SchwabToken(
        access_token="old",
        refresh_token="refresh",
        access_issued_at=(FIXED_NOW - timedelta(minutes=26)).timestamp(),
        refresh_issued_at=(FIXED_NOW - timedelta(seconds=REFRESH_TTL_SECONDS + 1)).timestamp(),
    )
    auth = _make_auth(tmp_path, clock, poster, token)

    with pytest.raises(RefreshTokenExpired):
        auth.get_access_token()
    assert poster.calls == []


def test_token_persistence_roundtrip(tmp_path: Path) -> None:
    clock = Clock(FIXED_NOW)
    poster = RecordingPoster()
    token = SchwabToken(
        access_token="a",
        refresh_token="r",
        access_issued_at=(FIXED_NOW - timedelta(minutes=26)).timestamp(),
        refresh_issued_at=(FIXED_NOW - timedelta(days=1)).timestamp(),
    )
    auth = _make_auth(tmp_path, clock, poster, token)
    auth.get_access_token()  # triggers refresh + save

    reloaded = SchwabAuth(
        app_key="key",
        app_secret="secret",
        token_path=tmp_path / "token.json",
        token_poster=poster,
        clock=clock,
    )
    loaded = reloaded.load()
    assert loaded is not None
    assert loaded.access_token == "new-access"
    assert loaded.refresh_token == "r"
