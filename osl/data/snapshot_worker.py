"""Snapshot worker: capture option chains into the Parquet store.

Pulls each watchlist symbol's chain + underlying through a provider and persists
a Hive-partitioned snapshot (see :mod:`osl.data.snapshots`). Run it once per
scheduled time (``--once``, suited to a GitHub Action cron) or as a long-lived
daemon (``--daemon``, via APScheduler at 09:35 / 12:30 / 15:55 ET).

Snapshots are the only way to accumulate the chain history the backtester needs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime

from osl.config import Settings, get_settings
from osl.data.base import DataProvider
from osl.data.providers import build_provider
from osl.data.snapshots import SnapshotContext, context_from_quote, write_snapshot
from osl.logging import get_logger
from osl.utils.calendars import is_trading_day
from osl.utils.time import now_utc

log = get_logger(__name__)

SNAPSHOT_TIMES_ET = ("09:35", "12:30", "15:55")


@dataclass(frozen=True)
class SnapshotReport:
    symbol: str
    ok: bool
    rows: int
    path: str | None = None
    error: str | None = None


def capture_symbol(
    provider: DataProvider,
    symbol: str,
    *,
    root: str,
    rate: float,
    dividend_yield: float = 0.0,
    snapshot_time: datetime | None = None,
) -> SnapshotReport:
    """Capture one symbol's chain + underlying to a snapshot partition."""
    ts = snapshot_time or now_utc()
    try:
        chain = provider.get_option_chain(symbol)
        underlying = provider.get_underlying(symbol)
        context: SnapshotContext = context_from_quote(
            underlying, risk_free_rate=rate, dividend_yield=dividend_yield
        )
        out = write_snapshot(chain, root=root, context=context, snapshot_time=ts)
        return SnapshotReport(symbol=symbol, ok=True, rows=len(chain), path=str(out))
    except Exception as exc:  # one bad symbol must not abort the run
        log.warning("snapshot_capture_failed", symbol=symbol, error=str(exc))
        return SnapshotReport(symbol=symbol, ok=False, rows=0, error=str(exc))


def capture_watchlist(
    provider: DataProvider,
    symbols: list[str],
    *,
    root: str,
    rate: float,
    dividend_yield: float = 0.0,
    snapshot_time: datetime | None = None,
    skip_non_trading: bool = True,
) -> list[SnapshotReport]:
    """Capture every symbol; optionally skip non-trading days."""
    ts = snapshot_time or now_utc()
    if skip_non_trading and not is_trading_day(ts.date()):
        log.info("snapshot_skip_non_trading_day", date=str(ts.date()))
        return []
    return [
        capture_symbol(
            provider, s, root=root, rate=rate, dividend_yield=dividend_yield, snapshot_time=ts
        )
        for s in symbols
    ]


def run_capture(
    settings: Settings | None = None,
    *,
    provider_name: str | None = None,
    symbols: list[str] | None = None,
    skip_non_trading: bool = True,
) -> list[SnapshotReport]:
    """Capture the configured watchlist using the configured provider + rate."""
    cfg = settings or get_settings()
    provider = build_provider(provider_name or cfg.default_provider, cfg)
    reports = capture_watchlist(
        provider,
        symbols or cfg.watchlist,
        root=str(cfg.snapshot_root),
        rate=cfg.risk_free_flat_rate,
        skip_non_trading=skip_non_trading,
    )
    ok = sum(r.ok for r in reports)
    log.info("snapshot_run_complete", captured=ok, total=len(reports))
    return reports


def _run_daemon(cfg: Settings) -> None:  # pragma: no cover - long-running loop
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="US/Eastern")
    for hhmm in SNAPSHOT_TIMES_ET:
        hour, minute = (int(x) for x in hhmm.split(":"))
        scheduler.add_job(
            lambda: run_capture(cfg),
            CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute, timezone="US/Eastern"),
            name=f"snapshot-{hhmm}",
        )
    log.info("snapshot_daemon_started", times=list(SNAPSHOT_TIMES_ET))
    scheduler.start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Options Strategy Lab snapshot worker")
    parser.add_argument("--daemon", action="store_true", help="run scheduled (APScheduler)")
    parser.add_argument("--provider", default=None, help="override default provider")
    parser.add_argument("--symbols", default=None, help="comma-separated symbol override")
    parser.add_argument("--force", action="store_true", help="capture even on non-trading days")
    args = parser.parse_args(argv)

    cfg = get_settings()
    if args.daemon:
        _run_daemon(cfg)
        return 0

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    )
    reports = run_capture(
        cfg, provider_name=args.provider, symbols=symbols, skip_non_trading=not args.force
    )
    for r in reports:
        status = "ok" if r.ok else f"FAILED ({r.error})"
        log.info("snapshot_symbol_report", symbol=r.symbol, rows=r.rows, status=status)
    return 0 if all(r.ok for r in reports) or not reports else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
