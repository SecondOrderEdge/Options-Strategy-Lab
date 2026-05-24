"""Snapshot-driven backtester: engine, rules, fills, portfolio, and metrics."""

from __future__ import annotations

from osl.backtest.config import BacktestConfig
from osl.backtest.engine import BacktestResult, BTDay, LookaheadError, System, run_backtest
from osl.backtest.loader import load_snapshot_days
from osl.backtest.metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    sortino_ratio,
)
from osl.backtest.portfolio import BTLeg, BTPosition, ClosedTrade, OpenOrder
from osl.backtest.rules import IronCondor1Sigma, LongStraddle, ShortPut16Delta

__all__ = [
    "BTDay",
    "BTLeg",
    "BTPosition",
    "BacktestConfig",
    "BacktestResult",
    "ClosedTrade",
    "IronCondor1Sigma",
    "LongStraddle",
    "LookaheadError",
    "OpenOrder",
    "ShortPut16Delta",
    "System",
    "deflated_sharpe_ratio",
    "load_snapshot_days",
    "max_drawdown",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "run_backtest",
    "sharpe_ratio",
    "sortino_ratio",
]
