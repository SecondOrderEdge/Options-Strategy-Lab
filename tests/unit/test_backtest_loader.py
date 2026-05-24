"""Snapshot-store loader round-trips into backtest days."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from osl.backtest.config import BacktestConfig
from osl.backtest.engine import run_backtest
from osl.backtest.loader import load_snapshot_days
from osl.backtest.rules import ShortPut16Delta
from osl.data.snapshots import SnapshotContext, write_snapshot
from tests.conftest import build_snapshot_history


def test_loader_reads_written_snapshots(tmp_path: Path):
    days = build_snapshot_history(n_days=10, seed=1)
    for d in days:
        ctx = SnapshotContext(
            underlying="TST",
            spot=d.spot,
            risk_free_rate=0.0,
            dividend_yield=0.0,
            snapshot_time=datetime.combine(d.date, datetime.min.time(), tzinfo=UTC).replace(
                hour=20
            ),
        )
        write_snapshot(d.chain, root=tmp_path, context=ctx, snapshot_time=ctx.snapshot_time)

    loaded = load_snapshot_days(tmp_path, "TST")
    assert len(loaded) == 10
    assert [d.date for d in loaded] == [d.date for d in days]
    assert loaded[0].spot == days[0].spot
    # The loaded days run through the engine end to end.
    res = run_backtest(loaded, ShortPut16Delta(), BacktestConfig(fill_model="mid"))
    assert len(res.equity) == 10


def test_loader_missing_symbol_returns_empty(tmp_path: Path):
    assert load_snapshot_days(tmp_path, "NOPE") == []
