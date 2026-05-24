"""Execution-price (fill) models.

``mid`` crosses at the NBBO mid (optimistic); ``worst`` pays the ask on buys
and takes the bid on sells (pessimistic); ``mid_penalty`` pays a fraction of the
half-spread in the adverse direction (the default); ``spread_frac`` scales that
penalty by illiquidity.
"""

from __future__ import annotations

from osl.backtest.config import FillModel


def execution_price(
    *,
    is_buy: bool,
    bid: float,
    ask: float,
    model: FillModel,
    spread_penalty: float = 0.25,
    liquidity_score: float = 1.0,
) -> float:
    """Return the price paid (buy) or received (sell) under the fill model."""
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0
    direction = 1.0 if is_buy else -1.0

    if model == "mid":
        return mid
    if model == "worst":
        return ask if is_buy else bid
    if model == "mid_penalty":
        return mid + direction * half * spread_penalty
    if model == "spread_frac":
        frac = spread_penalty + (1.0 - min(max(liquidity_score, 0.0), 1.0)) * (1.0 - spread_penalty)
        return mid + direction * half * frac
    raise ValueError(f"unknown fill model: {model!r}")
