"""Strategy engine: legs, builders, metrics, liquidity, enumeration, ranking."""

from __future__ import annotations

from osl.strategy.enumerate import enumerate_candidates
from osl.strategy.legs import Leg, OptionType, Side, Strategy
from osl.strategy.liquidity import LiquidityReport, strategy_liquidity
from osl.strategy.metrics import (
    EVResult,
    PayoffProfile,
    StrategyMetrics,
    aggregate_greeks,
    compute_metrics,
    payoff_profile,
)
from osl.strategy.optimizer import OBJECTIVES, Candidate, rank, score
from osl.strategy.strategies import ChainContext

__all__ = [
    "OBJECTIVES",
    "Candidate",
    "ChainContext",
    "EVResult",
    "Leg",
    "LiquidityReport",
    "OptionType",
    "PayoffProfile",
    "Side",
    "Strategy",
    "StrategyMetrics",
    "aggregate_greeks",
    "compute_metrics",
    "enumerate_candidates",
    "payoff_profile",
    "rank",
    "score",
    "strategy_liquidity",
]
