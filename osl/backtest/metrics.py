"""Backtest performance and overfitting statistics.

Standard ratios (Sharpe, Sortino, Calmar, max drawdown, VaR/ES, win/profit
factor) plus the discipline metrics the blueprint mandates:

* **PSR** — Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).
* **DSR** — Deflated Sharpe Ratio (Bailey & López de Prado 2014): PSR against a
  benchmark inflated by the number of trials.
* **PBO** — Probability of Backtest Overfitting via combinatorially symmetric
  cross-validation (Bailey, Borwein, López de Prado, Zhu 2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr
from scipy.stats import norm

FloatArray = npt.NDArray[np.float64]

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Standard ratios
# ---------------------------------------------------------------------------
def _per_period_sharpe(returns: FloatArray) -> float:
    if returns.size < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd == 0:
        return 0.0
    return float(np.mean(returns)) / sd


def sharpe_ratio(returns: FloatArray, *, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized Sharpe ratio of per-period returns (excess assumed)."""
    return _per_period_sharpe(returns) * math.sqrt(periods_per_year)


def sortino_ratio(returns: FloatArray, *, periods_per_year: int = TRADING_DAYS) -> float:
    if returns.size < 2:
        return 0.0
    downside = returns[returns < 0]
    dd = float(np.sqrt(np.mean(downside**2))) if downside.size else 0.0
    if dd == 0:
        return math.inf if np.mean(returns) > 0 else 0.0
    return float(np.mean(returns)) / dd * math.sqrt(periods_per_year)


def max_drawdown(equity: FloatArray) -> float:
    """Maximum drawdown as a negative fraction of the running peak."""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def calmar_ratio(
    equity: FloatArray, returns: FloatArray, *, periods_per_year: int = TRADING_DAYS
) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    cagr = float(np.mean(returns)) * periods_per_year
    return cagr / mdd


def value_at_risk(returns: FloatArray, *, alpha: float = 0.05) -> float:
    """VaR at level alpha (a negative number for losses)."""
    if returns.size == 0:
        return 0.0
    return float(np.quantile(returns, alpha))


def expected_shortfall(returns: FloatArray, *, alpha: float = 0.05) -> float:
    if returns.size == 0:
        return 0.0
    cutoff = np.quantile(returns, alpha)
    tail = returns[returns <= cutoff]
    return float(np.mean(tail)) if tail.size else 0.0


@dataclass(frozen=True)
class TradeStats:
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float


def trade_stats(pnls: FloatArray) -> TradeStats:
    if pnls.size == 0:
        return TradeStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    return TradeStats(
        n_trades=int(pnls.size),
        win_rate=float(wins.size / pnls.size),
        avg_win=float(wins.mean()) if wins.size else 0.0,
        avg_loss=float(losses.mean()) if losses.size else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else math.inf,
        expectancy=float(pnls.mean()),
    )


# ---------------------------------------------------------------------------
# PSR / DSR
# ---------------------------------------------------------------------------
def probabilistic_sharpe_ratio(
    sr: float, n: int, *, skew: float, kurt: float, benchmark_sr: float = 0.0
) -> float:
    """PSR: P(true per-period SR > benchmark) given the estimate's moments.

    ``sr`` and ``benchmark_sr`` are **per-period** Sharpe ratios; ``kurt`` is
    Pearson kurtosis (3 for a normal distribution).
    """
    if n < 2:
        return float("nan")
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - benchmark_sr) * math.sqrt(n - 1) / denom
    return float(ndtr(z))


def deflated_benchmark_sr(sr_variance: float, n_trials: int) -> float:
    """Expected maximum per-period SR across ``n_trials`` independent trials."""
    if n_trials < 1 or sr_variance <= 0:
        return 0.0
    if n_trials == 1:
        return 0.0
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(sr_variance) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(
    sr: float, n: int, *, skew: float, kurt: float, sr_variance: float, n_trials: int
) -> float:
    """DSR: PSR against the trial-inflated benchmark Sharpe."""
    benchmark = deflated_benchmark_sr(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(sr, n, skew=skew, kurt=kurt, benchmark_sr=benchmark)


def per_period_sharpe(returns: FloatArray) -> float:
    """Public accessor for the per-period Sharpe used by PSR/DSR."""
    return _per_period_sharpe(returns)


# ---------------------------------------------------------------------------
# PBO via CSCV
# ---------------------------------------------------------------------------
def probability_of_backtest_overfitting(returns_matrix: FloatArray, *, n_splits: int = 16) -> float:
    """PBO via combinatorially symmetric cross-validation.

    ``returns_matrix`` has shape (T periods, N strategies). The rows are split
    into ``n_splits`` contiguous blocks; over every balanced train/test split,
    the in-sample-best strategy's out-of-sample rank gives a logit, and PBO is
    the fraction of splits where it lands below the OOS median. ~0.5 under the
    null (no skill).
    """
    m = np.asarray(returns_matrix, dtype=np.float64)
    t, n = m.shape
    if n < 2 or t < n_splits:
        raise ValueError("need >= 2 strategies and at least n_splits periods")
    s = n_splits - (n_splits % 2)  # force even
    block_ids = np.array_split(np.arange(t), s)

    def _sharpe_cols(rows: FloatArray) -> FloatArray:
        mean = rows.mean(axis=0)
        sd = rows.std(axis=0, ddof=1)
        sd = np.where(sd == 0, np.nan, sd)
        return np.asarray(mean / sd, dtype=np.float64)

    logits: list[float] = []
    for train in combinations(range(s), s // 2):
        train_set = set(train)
        train_rows = np.concatenate([block_ids[i] for i in range(s) if i in train_set])
        test_rows = np.concatenate([block_ids[i] for i in range(s) if i not in train_set])
        is_perf = _sharpe_cols(m[train_rows])
        oos_perf = _sharpe_cols(m[test_rows])
        best = int(np.nanargmax(is_perf))
        # Relative rank of the IS-best strategy out-of-sample (1 = worst .. N = best).
        order = np.argsort(np.argsort(np.nan_to_num(oos_perf, nan=-np.inf)))
        rank = int(order[best]) + 1
        omega = rank / (n + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(math.log(omega / (1.0 - omega)))

    arr = np.array(logits)
    return float(np.mean(arr <= 0.0))
