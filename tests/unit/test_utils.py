"""Utilities: day-count, epoch conversion, rate curves, trading calendar."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from osl.utils import calendars, rates
from osl.utils.time import epoch_ms_to_utc, year_fraction


def test_year_fraction_act365f() -> None:
    assert year_fraction(date(2026, 1, 1), date(2027, 1, 1)) == pytest.approx(365 / 365)
    assert year_fraction(date(2026, 1, 1), date(2026, 1, 31)) == pytest.approx(30 / 365)


def test_year_fraction_datetimes() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, 12, tzinfo=UTC)
    assert year_fraction(start, end) == pytest.approx(0.5 / 365)


def test_epoch_ms_to_utc() -> None:
    dt = epoch_ms_to_utc(1_750_291_200_000)
    assert dt.tzinfo is not None
    assert dt.year == 2025


def test_flat_curve() -> None:
    curve = rates.flat_curve(0.045)
    assert curve.rate(0.1) == pytest.approx(0.045)
    assert curve.rate(5.0) == pytest.approx(0.045)


def test_from_fred_with_injected_fetcher() -> None:
    fake = {"DGS1MO": 5.30, "DGS3MO": 5.25, "DGS6MO": 5.10, "DGS1": 4.90, "DGS2": 4.50}

    def fetch(series_id: str, _api_key: str) -> float:
        return fake[series_id]

    curve = rates.from_fred("dummy", fetch=fetch)
    # 1-month tenor (~0.083y) close to the 5.30% short rate.
    assert curve.rate(1 / 12) == pytest.approx(0.053)
    assert curve.rate(2.0) == pytest.approx(0.045)


def test_from_fred_skips_missing() -> None:
    def fetch(series_id: str, _api_key: str) -> float:
        if series_id == "DGS1MO":
            raise ValueError("missing")
        return 4.0

    curve = rates.from_fred("dummy", fetch=fetch)
    assert curve.rate(1.0) == pytest.approx(0.04)


def test_trading_days_excludes_weekends() -> None:
    # 2026-05-23 is a Saturday; the prior Friday and following Monday are sessions.
    days = calendars.trading_days(date(2026, 5, 22), date(2026, 5, 26))
    assert date(2026, 5, 22) in days  # Friday
    assert date(2026, 5, 23) not in days  # Saturday
    assert date(2026, 5, 24) not in days  # Sunday
    # 2026-05-25 is Memorial Day (market holiday).
    assert date(2026, 5, 25) not in days
