"""Snapshot worker captures chains to the store and tolerates per-symbol errors."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from osl.backtest.loader import load_snapshot_days
from osl.data.base import UnderlyingQuote
from osl.data.snapshot_worker import capture_symbol, capture_watchlist
from osl.utils.time import now_utc
from tests.conftest import build_synthetic_chain

SNAP_TS = pd.Timestamp("2026-06-01 20:00:00", tz="UTC").to_pydatetime()


class FakeProvider:
    name = "fake"

    def __init__(self, chain: pd.DataFrame, *, fail: set[str] | None = None) -> None:
        self._chain = chain
        self._fail = fail or set()

    def get_option_chain(self, symbol: str, **_: object) -> pd.DataFrame:
        if symbol in self._fail:
            raise RuntimeError("provider boom")
        return self._chain

    def get_underlying(self, symbol: str) -> UnderlyingQuote:
        return UnderlyingQuote(
            symbol=symbol,
            last=100.0,
            bid=99.9,
            ask=100.1,
            mark=100.0,
            volume=1000,
            quote_time=now_utc(),
            is_delayed=False,
        )

    def get_history(self, symbol, *, start, end, bar="1d"):  # pragma: no cover - unused
        raise NotImplementedError

    def get_expirations(self, symbol):  # pragma: no cover - unused
        raise NotImplementedError

    def healthcheck(self):  # pragma: no cover - unused
        return {"provider": self.name, "ok": True}


def _chain() -> pd.DataFrame:
    chain, _ = build_synthetic_chain(spot=100.0, rate=0.02, strikes=[90, 95, 100, 105, 110])
    return chain


def test_capture_symbol_writes_snapshot(tmp_path: Path):
    provider = FakeProvider(_chain())
    report = capture_symbol(provider, "TST", root=str(tmp_path), rate=0.02, snapshot_time=SNAP_TS)
    assert report.ok
    assert report.rows == 10
    assert report.path is not None and Path(report.path).exists()


def test_capture_watchlist_partial_failure(tmp_path: Path):
    provider = FakeProvider(_chain(), fail={"BAD"})
    reports = capture_watchlist(
        provider,
        ["TST", "BAD"],
        root=str(tmp_path),
        rate=0.02,
        snapshot_time=SNAP_TS,
        skip_non_trading=False,
    )
    by_symbol = {r.symbol: r for r in reports}
    assert by_symbol["TST"].ok
    assert not by_symbol["BAD"].ok
    assert "boom" in (by_symbol["BAD"].error or "")


def test_captured_snapshots_load_for_backtest(tmp_path: Path):
    provider = FakeProvider(_chain())
    capture_symbol(provider, "TST", root=str(tmp_path), rate=0.02, snapshot_time=SNAP_TS)
    days = load_snapshot_days(tmp_path, "TST")
    assert len(days) == 1
    assert days[0].spot == 100.0
    assert len(days[0].chain) == 10


def test_skip_non_trading_day(tmp_path: Path):
    provider = FakeProvider(_chain())
    # 2026-05-23 is a Saturday -> no capture.
    saturday = pd.Timestamp("2026-05-23 20:00:00", tz="UTC").to_pydatetime()
    assert saturday.date() == date(2026, 5, 23)
    reports = capture_watchlist(
        provider, ["TST"], root=str(tmp_path), rate=0.02, snapshot_time=saturday
    )
    assert reports == []
