"""Objective functions and ranking for strategy candidates.

A candidate bundles a :class:`~osl.strategy.legs.Strategy` with its computed
metrics and liquidity. Objectives map a candidate to a score; ``rank`` sorts
descending (best first), deterministically and stably.

The blueprint's hard rule is encoded by ``pop_capped``: maximizing POP must never
silently select an unbounded-risk position, so those are pushed to the bottom.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from osl.strategy.legs import Strategy
from osl.strategy.liquidity import LiquidityReport
from osl.strategy.metrics import StrategyMetrics


@dataclass(frozen=True)
class Candidate:
    strategy: Strategy
    metrics: StrategyMetrics
    liquidity: LiquidityReport


def _risk(m: StrategyMetrics) -> float:
    return math.inf if m.loss_unbounded else abs(m.max_loss)


def _pop_capped(c: Candidate) -> float:
    return -math.inf if c.metrics.loss_unbounded else c.metrics.pop_rn


def _ev(c: Candidate) -> float:
    return c.metrics.ev_mc.value


def _ev_per_risk(c: Candidate) -> float:
    risk = _risk(c.metrics)
    return -math.inf if math.isinf(risk) or risk == 0 else c.metrics.ev_mc.value / risk


def _theta_per_risk(c: Candidate) -> float:
    risk = _risk(c.metrics)
    return -math.inf if math.isinf(risk) or risk == 0 else c.metrics.greeks["theta"] / risk


def _convexity_per_dollar(c: Candidate) -> float:
    return c.metrics.convexity


def _vega_long(c: Candidate) -> float:
    return c.metrics.greeks["vega"]


def _vega_short(c: Candidate) -> float:
    return -c.metrics.greeks["vega"]


def _liquidity_adj_ev(c: Candidate) -> float:
    return c.metrics.ev_mc.value * c.liquidity.score


def _regime_aware(c: Candidate) -> float:
    return _ev_per_risk(c) * c.liquidity.score


OBJECTIVES: dict[str, Callable[[Candidate], float]] = {
    "pop_capped": _pop_capped,
    "ev": _ev,
    "ev_per_risk": _ev_per_risk,
    "theta_per_risk": _theta_per_risk,
    "convexity_per_dollar": _convexity_per_dollar,
    "vega_long": _vega_long,
    "vega_short": _vega_short,
    "liquidity_adj_ev": _liquidity_adj_ev,
    "regime_aware": _regime_aware,
}


def score(candidate: Candidate, objective: str) -> float:
    try:
        fn = OBJECTIVES[objective]
    except KeyError:
        raise ValueError(f"unknown objective: {objective!r}") from None
    value = fn(candidate)
    return value if math.isfinite(value) else (-1e18 if value < 0 else 1e18)


def rank(candidates: list[Candidate], objective: str) -> list[Candidate]:
    """Return candidates sorted best-first for the objective (stable)."""
    return sorted(candidates, key=lambda c: score(c, objective), reverse=True)
