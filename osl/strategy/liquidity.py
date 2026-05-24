"""Liquidity scoring and slippage estimation for option strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from osl.strategy.legs import Strategy

OI_REFERENCE = 500.0
VOLUME_REFERENCE = 100.0
SPREAD_CAP = 0.10  # spread% at/above which the spread factor is 0


@dataclass(frozen=True)
class LiquidityReport:
    score: float  # 0 (illiquid) .. 1 (liquid)
    min_open_interest: int
    min_volume: int
    max_spread_pct: float
    slippage: float  # estimated $ cost to cross half the spread on every leg
    has_zero_bid: bool


def _contract_row(chain: pd.DataFrame, expiration: object, strike: float, right: str) -> pd.Series:
    mask = (
        (chain["expiration"].dt.date == expiration)
        & (np.isclose(chain["strike"].to_numpy(dtype=float), strike))
        & (chain["right"] == right)
    )
    sub = chain[mask]
    if sub.empty:
        raise KeyError(f"contract not found: {right} {strike} {expiration}")
    return cast(pd.Series, sub.iloc[0])


def strategy_liquidity(chain: pd.DataFrame, strategy: Strategy) -> LiquidityReport:
    """Aggregate per-leg liquidity into a single 0..1 score plus a slippage estimate."""
    min_oi = math.inf
    min_vol = math.inf
    max_spread_pct = 0.0
    slippage = 0.0
    has_zero_bid = False

    for leg in strategy.legs:
        row = _contract_row(chain, leg.expiration, leg.strike, leg.right)
        bid = float(row["bid"])
        ask = float(row["ask"])
        mid = float(row["mid"])
        oi = int(row["open_interest"])
        vol = int(row["volume"])

        if not np.isfinite(bid) or bid <= 0:
            has_zero_bid = True
        if np.isfinite(bid) and np.isfinite(ask) and mid > 0:
            spread_pct = (ask - bid) / mid
            max_spread_pct = max(max_spread_pct, spread_pct)
            slippage += 0.5 * (ask - bid) * leg.quantity * leg.multiplier

        min_oi = min(min_oi, oi)
        min_vol = min(min_vol, vol)

    min_oi_i = 0 if math.isinf(min_oi) else int(min_oi)
    min_vol_i = 0 if math.isinf(min_vol) else int(min_vol)

    oi_factor = min(min_oi_i / OI_REFERENCE, 1.0)
    vol_factor = min(min_vol_i / VOLUME_REFERENCE, 1.0)
    spread_factor = max(0.0, 1.0 - min(max_spread_pct / SPREAD_CAP, 1.0))
    atm = math.exp(-abs(math.log((strategy.strikes[0] or strategy.spot) / strategy.spot)) / 0.25)

    score = 0.4 * spread_factor + 0.3 * oi_factor + 0.2 * vol_factor + 0.1 * atm
    if has_zero_bid:
        score *= 0.25  # heavily penalize untradeable legs

    return LiquidityReport(
        score=float(np.clip(score, 0.0, 1.0)),
        min_open_interest=min_oi_i,
        min_volume=min_vol_i,
        max_spread_pct=float(max_spread_pct),
        slippage=float(slippage),
        has_zero_bid=has_zero_bid,
    )
