"""Backtest engine: no-lookahead guard, fill sensitivity, determinism, trades."""

from __future__ import annotations

import numpy as np
import pytest

from osl.backtest.config import BacktestConfig
from osl.backtest.engine import LookaheadError, run_backtest
from osl.backtest.rules import IronCondor1Sigma, LongStraddle, ShortPut16Delta
from tests.conftest import build_snapshot_history


def test_no_lookahead_guard():
    days = build_snapshot_history(n_days=60, lookahead_on=20, seed=1)
    with pytest.raises(LookaheadError):
        run_backtest(days, ShortPut16Delta(), BacktestConfig())


def test_clean_history_has_no_lookahead():
    days = build_snapshot_history(n_days=60, seed=1)
    res = run_backtest(days, ShortPut16Delta(), BacktestConfig(fill_model="mid"))
    assert len(res.equity) == 60  # ran to completion


def test_fill_model_sensitivity_credit_strategy():
    days = build_snapshot_history(n_days=150, vol=0.3, spread=0.20, seed=3)
    system = IronCondor1Sigma()
    mid = run_backtest(days, system, BacktestConfig(fill_model="mid", spread_penalty=0.25))
    worst = run_backtest(days, system, BacktestConfig(fill_model="worst", spread_penalty=0.25))
    assert worst.stats["final_equity"] <= mid.stats["final_equity"]
    assert worst.trades  # the system actually traded


def test_replay_is_deterministic():
    days = build_snapshot_history(n_days=120, seed=5)
    cfg = BacktestConfig(fill_model="mid_penalty")
    a = run_backtest(days, LongStraddle(), cfg)
    b = run_backtest(days, LongStraddle(), cfg)
    np.testing.assert_array_equal(a.equity.to_numpy(), b.equity.to_numpy())
    assert [t.pnl for t in a.trades] == [t.pnl for t in b.trades]


def test_short_put_system_trades_and_settles():
    days = build_snapshot_history(n_days=150, vol=0.25, seed=2)
    res = run_backtest(days, ShortPut16Delta(), BacktestConfig(fill_model="mid"))
    assert res.trades
    # Closes happen at the 21-DTE rule or expiry.
    assert all(t.reason in {"21 DTE", "expiry"} for t in res.trades)
    assert "sharpe" in res.stats


def test_capital_scaling_changes_size():
    days = build_snapshot_history(n_days=80, seed=4)
    small = run_backtest(days, ShortPut16Delta(), BacktestConfig(capital=50_000))
    big = run_backtest(days, ShortPut16Delta(), BacktestConfig(capital=500_000))
    # More capital -> larger positions -> larger absolute P&L swings.
    assert big.equity.std() > small.equity.std()
