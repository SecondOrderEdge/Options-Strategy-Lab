"""Strategy-level analytics: payoff, POP, EV, tail risk, and aggregated greeks.

Hard rule (blueprint §4.5): POP is never reported without its probability
measure and model, and always alongside an expected-value and tail-loss
companion. A high POP does not imply positive EV.

Probability-measure labels:
  * ``pop_delta``       — risk-neutral, delta proxy (overstates POP)
  * ``pop_rn``          — risk-neutral lognormal, closed form (N(d2)-type)
  * ``pop_mc``          — risk-neutral GBM Monte Carlo
  * ``pop_empirical``   — real-world, bootstrap of historical returns
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import ndtr

from osl.pricing.bsm import bsm_greeks, bsm_price
from osl.pricing.monte_carlo import gbm_terminal
from osl.strategy.legs import OptionType, Strategy

FloatArray = npt.NDArray[np.float64]

GREEK_NAMES = ("delta", "gamma", "theta", "vega", "rho", "vanna", "charm", "vomma")


# ---------------------------------------------------------------------------
# Payoff geometry
# ---------------------------------------------------------------------------
def upside_slope(strategy: Strategy) -> float:
    """dP&L/dS as S -> infinity (calls + stock; puts vanish). <0 => unbounded loss."""
    slope = float(strategy.stock_quantity)
    for leg in strategy.legs:
        if leg.option_type is OptionType.CALL:
            slope += leg.sign * leg.quantity * leg.multiplier
    return slope


@dataclass(frozen=True)
class PayoffProfile:
    breakevens: tuple[float, ...]
    max_profit: float
    max_loss: float
    loss_unbounded: bool
    profit_unbounded: bool


def _interp_zero(x0: float, y0: float, x1: float, y1: float) -> float:
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def payoff_profile(strategy: Strategy, *, grid_points: int = 4000) -> PayoffProfile:
    """Breakevens and bounded/unbounded max profit & loss at the primary expiry."""
    strikes = list(strategy.strikes)
    up = upside_slope(strategy)

    if strategy.all_same_expiry:
        far = (max(strikes) if strikes else strategy.spot) * 3 + 100.0
        pts = np.array(sorted({0.0, *strikes, far}))
        vals = strategy.pnl(pts)
    else:
        hi = (max(strikes) if strikes else strategy.spot) * 3 + 100.0
        pts = np.linspace(1e-6, hi, grid_points)
        vals = strategy.pnl(pts)

    breakevens: list[float] = []
    for i in range(len(pts) - 1):
        y0, y1 = float(vals[i]), float(vals[i + 1])
        if y0 == 0.0:
            breakevens.append(float(pts[i]))
        elif y0 * y1 < 0:
            breakevens.append(_interp_zero(float(pts[i]), y0, float(pts[i + 1]), y1))

    profit_unbounded = up > 0
    loss_unbounded = up < 0
    max_profit = math.inf if profit_unbounded else float(np.max(vals))
    max_loss = -math.inf if loss_unbounded else float(np.min(vals))

    return PayoffProfile(
        breakevens=tuple(round(b, 6) for b in sorted(set(breakevens))),
        max_profit=max_profit,
        max_loss=max_loss,
        loss_unbounded=loss_unbounded,
        profit_unbounded=profit_unbounded,
    )


def _profit_intervals(strategy: Strategy, profile: PayoffProfile) -> list[tuple[float, float]]:
    edges = [0.0, *profile.breakevens, math.inf]
    intervals: list[tuple[float, float]] = []
    for lo, hi in itertools.pairwise(edges):
        test = (lo + hi) / 2 if math.isfinite(hi) else lo * 1.5 + 1.0
        if float(strategy.pnl(np.array([test]))[0]) > 0:
            intervals.append((lo, hi))
    return intervals


# ---------------------------------------------------------------------------
# Reference volatility for the RN distribution
# ---------------------------------------------------------------------------
def reference_sigma(strategy: Strategy) -> float:
    """Vega-weighted average leg IV (the RN sigma for POP/EV)."""
    sigmas: list[float] = []
    weights: list[float] = []
    for leg in strategy.legs:
        if not math.isfinite(leg.iv) or leg.iv <= 0:
            continue
        vega = abs(
            float(
                bsm_greeks(
                    strategy.spot,
                    leg.strike,
                    leg.T,
                    strategy.rate,
                    strategy.dividend_yield,
                    leg.iv,
                    leg.right,
                )["vega"].item()
            )
        )
        sigmas.append(leg.iv)
        weights.append(vega * leg.quantity + 1e-9)
    if not sigmas:
        return float("nan")
    return float(np.average(sigmas, weights=weights))


# ---------------------------------------------------------------------------
# Probability of profit
# ---------------------------------------------------------------------------
def _rn_cdf_le(x: float, forward: float, sig_t: float, *, delta_proxy: bool) -> float:
    """P(S_T <= x) under RN lognormal; delta_proxy flips the +/-0.5 sigma^2 term."""
    if x <= 0:
        return 0.0
    if math.isinf(x):
        return 1.0
    shift = -0.5 if delta_proxy else 0.5
    d = (math.log(x / forward) + shift * sig_t * sig_t) / sig_t
    return float(ndtr(d))


def _pop_lognormal(
    strategy: Strategy, profile: PayoffProfile, sigma: float, *, delta_proxy: bool
) -> float:
    if not math.isfinite(sigma) or sigma <= 0:
        return float("nan")
    T = strategy.primary_T
    forward = strategy.spot * math.exp((strategy.rate - strategy.dividend_yield) * T)
    sig_t = sigma * math.sqrt(T)
    total = 0.0
    for lo, hi in _profit_intervals(strategy, profile):
        total += _rn_cdf_le(hi, forward, sig_t, delta_proxy=delta_proxy) - _rn_cdf_le(
            lo, forward, sig_t, delta_proxy=delta_proxy
        )
    return total


def pop_monte_carlo(strategy: Strategy, sigma: float, *, n: int = 50_000, seed: int = 42) -> float:
    if not math.isfinite(sigma) or sigma <= 0:
        return float("nan")
    s_t = gbm_terminal(
        strategy.spot,
        strategy.rate,
        strategy.dividend_yield,
        sigma,
        strategy.primary_T,
        n=n,
        seed=seed,
    )
    return float(np.mean(strategy.pnl(s_t) > 0))


def pop_empirical(
    strategy: Strategy, log_returns: pd.Series, *, n: int = 50_000, seed: int = 42
) -> float:
    """Bootstrap POP under historical daily log returns (real-world measure)."""
    rng = np.random.default_rng(seed)
    rets = log_returns.dropna().to_numpy(dtype=np.float64)
    if rets.size == 0:
        return float("nan")
    horizon = max(strategy.primary_dte, 1)
    draws = rng.choice(rets, size=(n, horizon), replace=True).sum(axis=1)
    s_t = strategy.spot * np.exp(draws)
    return float(np.mean(strategy.pnl(s_t) > 0))


# ---------------------------------------------------------------------------
# Expected value
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EVResult:
    value: float
    ci_low: float
    ci_high: float
    method: str


def ev_closed_form(strategy: Strategy) -> float:
    """Edge vs fair value: sum of leg (theo - premium), each at its own IV."""
    total = 0.0
    for leg in strategy.legs:
        theo = float(
            bsm_price(
                strategy.spot,
                leg.strike,
                leg.T,
                strategy.rate,
                strategy.dividend_yield,
                leg.iv,
                leg.right,
            ).item()
        )
        total += leg.sign * leg.quantity * leg.multiplier * (theo - leg.premium)
    total += strategy.stock_quantity * (strategy.spot - strategy.stock_entry)
    return total


def ev_monte_carlo(
    strategy: Strategy, sigma: float, *, n: int = 50_000, seed: int = 42
) -> EVResult:
    T = strategy.primary_T
    s_t = gbm_terminal(
        strategy.spot, strategy.rate, strategy.dividend_yield, sigma, T, n=n, seed=seed
    )
    disc = math.exp(-strategy.rate * T)
    pnl = disc * strategy.terminal_value(s_t) - strategy.net_debit
    mean = float(np.mean(pnl))
    stderr = float(np.std(pnl, ddof=1) / math.sqrt(pnl.size))
    return EVResult(mean, mean - 1.96 * stderr, mean + 1.96 * stderr, "rn_mc")


def ev_empirical(
    strategy: Strategy, log_returns: pd.Series, *, n: int = 50_000, seed: int = 42
) -> EVResult:
    rng = np.random.default_rng(seed)
    rets = log_returns.dropna().to_numpy(dtype=np.float64)
    horizon = max(strategy.primary_dte, 1)
    draws = rng.choice(rets, size=(n, horizon), replace=True).sum(axis=1)
    s_t = strategy.spot * np.exp(draws)
    pnl = strategy.terminal_value(s_t) - strategy.net_debit
    mean = float(np.mean(pnl))
    stderr = float(np.std(pnl, ddof=1) / math.sqrt(pnl.size))
    return EVResult(mean, mean - 1.96 * stderr, mean + 1.96 * stderr, "empirical")


def expected_shortfall(
    strategy: Strategy, sigma: float, *, alpha: float = 0.05, n: int = 50_000, seed: int = 42
) -> float:
    """Mean loss in the worst ``alpha`` tail (positive number), RN GBM."""
    s_t = gbm_terminal(
        strategy.spot,
        strategy.rate,
        strategy.dividend_yield,
        sigma,
        strategy.primary_T,
        n=n,
        seed=seed,
    )
    pnl = strategy.pnl(s_t)
    cutoff = float(np.quantile(pnl, alpha))
    tail = pnl[pnl <= cutoff]
    if tail.size == 0:
        return float("nan")
    return float(-np.mean(tail))


# ---------------------------------------------------------------------------
# Greeks & convexity
# ---------------------------------------------------------------------------
def aggregate_greeks(strategy: Strategy) -> dict[str, float]:
    totals = dict.fromkeys(GREEK_NAMES, 0.0)
    for leg in strategy.legs:
        g = bsm_greeks(
            strategy.spot,
            leg.strike,
            leg.T,
            strategy.rate,
            strategy.dividend_yield,
            leg.iv,
            leg.right,
        )
        scale = leg.sign * leg.quantity * leg.multiplier
        for name in GREEK_NAMES:
            totals[name] += scale * float(g[name].item())
    totals["delta"] += strategy.stock_quantity  # stock delta = 1 share
    return totals


def convexity_proxy(
    strategy: Strategy, gamma: float, *, portfolio_value: float | None = None
) -> float:
    pv = portfolio_value if portfolio_value is not None else max(abs(strategy.net_debit), 1.0)
    return gamma * strategy.spot**2 / pv


def return_on_risk(profile: PayoffProfile) -> float:
    if profile.loss_unbounded:
        return 0.0
    if profile.max_loss >= 0:
        return math.inf
    return profile.max_profit / abs(profile.max_loss)


# ---------------------------------------------------------------------------
# Full summary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyMetrics:
    net_debit: float
    is_credit: bool
    max_profit: float
    max_loss: float
    loss_unbounded: bool
    profit_unbounded: bool
    breakevens: tuple[float, ...]
    return_on_risk: float
    reference_sigma: float
    pop_delta: float
    pop_rn: float
    pop_mc: float
    pop_empirical: float | None
    ev_closed: float
    ev_mc: EVResult
    ev_empirical: EVResult | None
    expected_shortfall: float
    greeks: dict[str, float] = field(default_factory=dict)
    convexity: float = 0.0


def compute_metrics(
    strategy: Strategy,
    *,
    sigma: float | None = None,
    n_mc: int = 20_000,
    seed: int = 42,
    log_returns: pd.Series | None = None,
    alpha: float = 0.05,
    portfolio_value: float | None = None,
) -> StrategyMetrics:
    """Compute the full metric bundle for a strategy."""
    profile = payoff_profile(strategy)
    sig = sigma if sigma is not None else reference_sigma(strategy)
    greeks = aggregate_greeks(strategy)

    pop_emp = (
        pop_empirical(strategy, log_returns, n=n_mc, seed=seed) if log_returns is not None else None
    )
    ev_emp = (
        ev_empirical(strategy, log_returns, n=n_mc, seed=seed) if log_returns is not None else None
    )

    return StrategyMetrics(
        net_debit=strategy.net_debit,
        is_credit=strategy.is_credit,
        max_profit=profile.max_profit,
        max_loss=profile.max_loss,
        loss_unbounded=profile.loss_unbounded,
        profit_unbounded=profile.profit_unbounded,
        breakevens=profile.breakevens,
        return_on_risk=return_on_risk(profile),
        reference_sigma=sig,
        pop_delta=_pop_lognormal(strategy, profile, sig, delta_proxy=True),
        pop_rn=_pop_lognormal(strategy, profile, sig, delta_proxy=False),
        pop_mc=pop_monte_carlo(strategy, sig, n=n_mc, seed=seed),
        pop_empirical=pop_emp,
        ev_closed=ev_closed_form(strategy),
        ev_mc=ev_monte_carlo(strategy, sig, n=n_mc, seed=seed),
        ev_empirical=ev_emp,
        expected_shortfall=expected_shortfall(strategy, sig, alpha=alpha, n=n_mc, seed=seed),
        greeks=greeks,
        convexity=convexity_proxy(strategy, greeks["gamma"], portfolio_value=portfolio_value),
    )
