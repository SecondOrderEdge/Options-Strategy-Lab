"""Backtest discipline metrics: PSR, DSR, PBO, and standard ratios."""

from __future__ import annotations

import numpy as np
import pytest

from osl.backtest.metrics import (
    deflated_sharpe_ratio,
    max_drawdown,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    sortino_ratio,
    trade_stats,
)


def test_sharpe_and_sortino_signs():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    assert sharpe_ratio(r) > 0
    assert sortino_ratio(r) > 0


def test_max_drawdown():
    equity = np.array([100.0, 110.0, 90.0, 95.0, 120.0])
    # Peak 110 -> trough 90 = -18.18%.
    assert max_drawdown(equity) == pytest.approx((90 - 110) / 110)


def test_psr_increases_with_track_record():
    rng = np.random.default_rng(1)
    r = rng.normal(0.05, 1.0, 2000)
    sr = per_period_sharpe(r)
    short = probabilistic_sharpe_ratio(sr, 100, skew=0.0, kurt=3.0)
    long = probabilistic_sharpe_ratio(sr, 2000, skew=0.0, kurt=3.0)
    assert 0.0 <= short <= 1.0
    assert long >= short  # more data -> more confident the SR > 0


def test_dsr_le_psr_when_multiple_trials():
    sr = 0.10  # per-period
    n = 1000
    psr0 = probabilistic_sharpe_ratio(sr, n, skew=0.0, kurt=3.0, benchmark_sr=0.0)
    dsr = deflated_sharpe_ratio(sr, n, skew=0.0, kurt=3.0, sr_variance=0.01, n_trials=50)
    assert dsr <= psr0  # deflation against many trials lowers the probability


def test_pbo_random_strategies_near_half():
    rng = np.random.default_rng(7)
    # 20 skill-less strategies over 240 periods -> PBO ~ 0.5.
    returns = rng.normal(0.0, 0.01, size=(240, 20))
    pbo = probability_of_backtest_overfitting(returns, n_splits=10)
    assert 0.3 <= pbo <= 0.7


def test_trade_stats():
    pnls = np.array([100.0, -50.0, 200.0, -25.0])
    ts = trade_stats(pnls)
    assert ts.n_trades == 4
    assert ts.win_rate == pytest.approx(0.5)
    assert ts.profit_factor == pytest.approx(300 / 75)
    assert ts.expectancy == pytest.approx(56.25)
