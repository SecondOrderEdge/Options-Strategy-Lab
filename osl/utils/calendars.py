"""NYSE trading-calendar helpers backed by ``pandas_market_calendars`` (XNYS)."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas_market_calendars as mcal

XNYS = "XNYS"


@lru_cache(maxsize=4)
def _calendar(name: str = XNYS) -> mcal.MarketCalendar:
    import pandas_market_calendars as mcal

    return mcal.get_calendar(name)


def trading_days(start: date, end: date, *, calendar: str = XNYS) -> list[date]:
    """Return the list of exchange trading days in ``[start, end]`` inclusive."""
    sched = _calendar(calendar).schedule(start_date=start, end_date=end)
    return [ts.date() for ts in sched.index]


def is_trading_day(day: date, *, calendar: str = XNYS) -> bool:
    """Return ``True`` if ``day`` is an exchange trading day."""
    return day in set(trading_days(day, day, calendar=calendar))


def num_trading_days(start: date, end: date, *, calendar: str = XNYS) -> int:
    """Count exchange trading days in ``[start, end]`` inclusive."""
    return len(trading_days(start, end, calendar=calendar))
