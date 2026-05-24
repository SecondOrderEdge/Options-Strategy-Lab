"""Time helpers and the ACT/365F day-count convention.

OSL standardizes on ACT/365 Fixed for option time-to-expiry: year fraction is
calendar days divided by 365, regardless of leap years. All "now" timestamps
are timezone-aware UTC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

DAYS_PER_YEAR: float = 365.0
EASTERN_TZ = "US/Eastern"


def now_utc() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(tz=UTC)


def to_utc(dt: datetime) -> datetime:
    """Return ``dt`` as a UTC-aware datetime.

    Naive datetimes are assumed to already be in UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def year_fraction(start: date | datetime, end: date | datetime) -> float:
    """ACT/365F year fraction between two dates.

    Uses whole-day differences; for intraday precision pass datetimes and the
    fractional seconds are included.
    """
    if isinstance(start, datetime) and isinstance(end, datetime):
        delta_seconds = (to_utc(end) - to_utc(start)).total_seconds()
        return delta_seconds / (DAYS_PER_YEAR * 86_400.0)
    s = start.date() if isinstance(start, datetime) else start
    e = end.date() if isinstance(end, datetime) else end
    return (e - s).days / DAYS_PER_YEAR


def epoch_ms_to_utc(epoch_ms: int | float) -> datetime:
    """Convert a millisecond Unix epoch (Schwab's format) to UTC datetime."""
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)
