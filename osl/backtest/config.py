"""Backtest configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FillModel = Literal["mid", "mid_penalty", "worst", "spread_frac"]


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters controlling a single-system backtest."""

    capital: float = 100_000.0
    fill_model: FillModel = "mid_penalty"
    spread_penalty: float = 0.25  # fraction of the half-spread paid on entry/exit
    commission_per_contract: float = 0.65
    slippage_bp: float = 0.0
    max_position_pct: float = 0.05  # capital fraction risked per position
    avoid_earnings: bool = True
    seed: int = 42
