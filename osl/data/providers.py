"""Provider construction + a data-freshness badge helper.

A full health-aware ``ProviderRouter`` is M0-broad; the first PR ships a small
factory plus the freshness logic the Streamlit badge needs (GREEN = live,
AMBER = delayed/yfinance, RED = stale beyond the threshold).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from osl.config import Settings
from osl.data.base import DataProvider
from osl.data.schwab import SchwabAuth, SchwabProvider
from osl.data.yfinance_provider import YFinanceProvider
from osl.utils.time import now_utc, to_utc


class Freshness(StrEnum):
    GREEN = "GREEN"  # live, real-time
    AMBER = "AMBER"  # delayed feed or yfinance fallback
    RED = "RED"  # stale beyond threshold


def build_schwab(settings: Settings) -> SchwabProvider:
    """Construct a Schwab provider that loads its token from the configured path."""
    auth = SchwabAuth(
        app_key=settings.schwab_app_key.get_secret_value(),
        app_secret=settings.schwab_app_secret.get_secret_value(),
        token_path=settings.schwab_token_path,
    )
    return SchwabProvider(auth, raw_root=settings.raw_data_root)


def build_yfinance(_: Settings | None = None) -> YFinanceProvider:
    return YFinanceProvider()


def build_provider(name: str, settings: Settings) -> DataProvider:
    """Return a provider by name (``"schwab"`` or ``"yfinance"``)."""
    if name == "schwab":
        return build_schwab(settings)
    if name == "yfinance":
        return build_yfinance(settings)
    raise ValueError(f"unknown provider: {name!r}")


def freshness_badge(
    provider: str,
    *,
    is_delayed: bool,
    quote_time: datetime,
    now: datetime | None = None,
    stale_after_minutes: int = 15,
) -> Freshness:
    """Classify data freshness for the UI badge.

    Staleness always wins: a quote older than the threshold is RED regardless of
    provider. Otherwise yfinance / delayed Schwab is AMBER, live Schwab GREEN.
    """
    now = now or now_utc()
    age_minutes = (to_utc(now) - to_utc(quote_time)).total_seconds() / 60.0
    if age_minutes > stale_after_minutes:
        return Freshness.RED
    if provider != "schwab" or is_delayed:
        return Freshness.AMBER
    return Freshness.GREEN
