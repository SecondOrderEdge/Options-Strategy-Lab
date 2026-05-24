"""Load backtest days from the Parquet snapshot store.

Reads the Hive-partitioned snapshots written by ``osl.data.snapshots`` and
returns one :class:`~osl.backtest.engine.BTDay` per snapshot date (the latest
snapshot of each day), ready to feed :func:`~osl.backtest.engine.run_backtest`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from osl.backtest.engine import BTDay
from osl.data.snapshots import CHAIN_FILE, read_chain, read_context


def _parse_snapshot_time(dirname: str) -> datetime:
    # "snapshot_time=20260524T160500Z" -> aware UTC datetime
    stamp = dirname.split("=", 1)[1]
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def load_snapshot_days(root: str | Path, symbol: str) -> list[BTDay]:
    """Return BTDays for ``symbol`` from the snapshot store (latest per date)."""
    base = Path(root) / f"underlying={symbol}"
    if not base.exists():
        return []

    by_date: dict[str, Path] = {}
    asof_by_date: dict[str, datetime] = {}
    for date_dir in sorted(base.glob("snapshot_date=*")):
        for time_dir in sorted(date_dir.glob("snapshot_time=*")):
            if not (time_dir / CHAIN_FILE).exists():
                continue
            date_key = date_dir.name.split("=", 1)[1]
            by_date[date_key] = time_dir  # later sorted dir wins (latest of the day)
            asof_by_date[date_key] = _parse_snapshot_time(time_dir.name)

    days: list[BTDay] = []
    for date_key in sorted(by_date):
        partition = by_date[date_key]
        chain = read_chain(partition)
        ctx = read_context(partition)
        spot = ctx.spot if ctx is not None else float("nan")
        days.append(
            BTDay(
                date=datetime.strptime(date_key, "%Y-%m-%d").date(),
                asof=asof_by_date[date_key],
                spot=spot,
                chain=chain,
            )
        )
    return days
