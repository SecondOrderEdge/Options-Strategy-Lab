"""Provider factory + freshness badge classification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from osl.config import Settings
from osl.data.base import DataProvider
from osl.data.providers import Freshness, build_provider, freshness_badge
from osl.data.schwab import SchwabProvider
from osl.data.yfinance_provider import YFinanceProvider

NOW = datetime(2026, 5, 24, 16, 0, 0, tzinfo=UTC)


def test_build_provider_returns_concrete_types() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert isinstance(build_provider("schwab", settings), SchwabProvider)
    assert isinstance(build_provider("yfinance", settings), YFinanceProvider)


def test_providers_satisfy_protocol() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert isinstance(build_provider("schwab", settings), DataProvider)
    assert isinstance(build_provider("yfinance", settings), DataProvider)


def test_build_provider_unknown() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("ibkr", settings)


def test_live_schwab_is_green() -> None:
    badge = freshness_badge(
        "schwab",
        is_delayed=False,
        quote_time=NOW - timedelta(minutes=1),
        now=NOW,
    )
    assert badge is Freshness.GREEN


def test_delayed_schwab_is_amber() -> None:
    badge = freshness_badge(
        "schwab",
        is_delayed=True,
        quote_time=NOW - timedelta(minutes=2),
        now=NOW,
    )
    assert badge is Freshness.AMBER


def test_yfinance_is_amber() -> None:
    badge = freshness_badge(
        "yfinance",
        is_delayed=False,
        quote_time=NOW - timedelta(minutes=2),
        now=NOW,
    )
    assert badge is Freshness.AMBER


def test_stale_is_red_regardless_of_provider() -> None:
    badge = freshness_badge(
        "schwab",
        is_delayed=False,
        quote_time=NOW - timedelta(minutes=30),
        now=NOW,
    )
    assert badge is Freshness.RED
