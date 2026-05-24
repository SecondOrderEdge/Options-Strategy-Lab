"""Snapshot-driven backtest event loop (single concurrent position per system).

Each trading day the engine: (1) refuses any chain containing data stamped
after the decision time (no look-ahead), (2) settles expired positions at
intrinsic value, (3) applies the system's exit rule, (4) applies its entry rule
(sizing against capital), and (5) records mark-to-market equity. Given the same
data and seed it replays bit-for-bit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

import numpy as np
import pandas as pd

from osl.backtest.config import BacktestConfig
from osl.backtest.metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    trade_stats,
)
from osl.backtest.portfolio import (
    BTPosition,
    ClosedTrade,
    OpenOrder,
    entry_cashflow,
    exit_cashflow,
    position_value_mid,
    settlement_cashflow,
)
from osl.utils.time import to_utc


class LookaheadError(Exception):
    """Raised when a snapshot exposes data stamped after the decision time."""


@dataclass(frozen=True)
class BTDay:
    """One trading day's decision context."""

    date: date
    asof: datetime  # decision timestamp (UTC); chain data must predate it
    spot: float
    chain: pd.DataFrame


class System(Protocol):
    """An entry+exit system the engine drives."""

    name: str

    def enter(self, day: BTDay, capital: float, config: BacktestConfig) -> OpenOrder | None: ...

    def should_exit(self, day: BTDay, position: BTPosition) -> str | None: ...


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: list[ClosedTrade]
    config: BacktestConfig
    stats: dict[str, float] = field(default_factory=dict)


def _assert_no_lookahead(day: BTDay) -> None:
    qt = day.chain["quote_time"]
    if qt.empty:
        return
    latest = qt.max()
    if latest > pd.Timestamp(to_utc(day.asof)):
        raise LookaheadError(
            f"chain for {day.date} contains data at {latest} after decision time {day.asof}"
        )


def run_backtest(days: Sequence[BTDay], system: System, config: BacktestConfig) -> BacktestResult:
    """Run ``system`` over ``days`` and return equity, trades, and stats."""
    ordered = sorted(days, key=lambda d: d.date)
    cash = config.capital
    position: BTPosition | None = None
    eq_dates: list[date] = []
    eq_vals: list[float] = []
    trades: list[ClosedTrade] = []

    for day in ordered:
        _assert_no_lookahead(day)

        # 1. Expiry settlement.
        if position is not None and any(leg.expiration <= day.date for leg in position.legs):
            settle = settlement_cashflow(day.spot, position)
            cash += settle
            trades.append(
                ClosedTrade(
                    position.label,
                    position.entry_date,
                    day.date,
                    position.entry_cash + settle,
                    "expiry",
                )
            )
            position = None

        # 2. Exit rule.
        if position is not None:
            reason = system.should_exit(day, position)
            if reason:
                proceeds = exit_cashflow(day.chain, position, config)
                cash += proceeds
                trades.append(
                    ClosedTrade(
                        position.label,
                        position.entry_date,
                        day.date,
                        position.entry_cash + proceeds,
                        reason,
                    )
                )
                position = None

        # 3. Entry rule.
        if position is None:
            order = system.enter(day, cash, config)
            if order is not None and order.legs:
                entry_cash = entry_cashflow(day.chain, order.legs, config)
                position = BTPosition(
                    label=order.label,
                    legs=order.legs,
                    entry_date=day.date,
                    entry_cash=entry_cash,
                    risk_per_unit=order.risk_per_unit,
                )
                cash += entry_cash

        # 4. Mark to market.
        mv = position_value_mid(day.chain, position) if position is not None else 0.0
        eq_dates.append(day.date)
        eq_vals.append(cash + mv)

    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates), name="equity")
    returns = equity.pct_change().dropna()
    stats = _summarize(equity, returns, trades)
    return BacktestResult(equity=equity, returns=returns, trades=trades, config=config, stats=stats)


def _summarize(
    equity: pd.Series, returns: pd.Series, trades: list[ClosedTrade]
) -> dict[str, float]:
    r = returns.to_numpy(dtype=float)
    eq = equity.to_numpy(dtype=float)
    ts = trade_stats(np.array([t.pnl for t in trades], dtype=float))
    return {
        "final_equity": float(eq[-1]) if eq.size else 0.0,
        "total_return": float(eq[-1] / eq[0] - 1.0) if eq.size else 0.0,
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "calmar": calmar_ratio(eq, r),
        "max_drawdown": max_drawdown(eq),
        "n_trades": float(ts.n_trades),
        "win_rate": ts.win_rate,
        "profit_factor": ts.profit_factor,
        "expectancy": ts.expectancy,
    }
