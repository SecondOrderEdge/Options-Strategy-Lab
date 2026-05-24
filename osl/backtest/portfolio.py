"""Backtest position model and cash-flow / mark-to-market helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from osl.backtest.config import BacktestConfig
from osl.backtest.fills import execution_price


@dataclass(frozen=True)
class BTLeg:
    right: str  # "C" or "P"
    strike: float
    expiration: date
    side: int  # +1 long, -1 short
    quantity: int
    multiplier: int = 100


@dataclass(frozen=True)
class BTPosition:
    label: str
    legs: tuple[BTLeg, ...]
    entry_date: date
    entry_cash: float  # signed net cash at open (negative = debit paid)
    risk_per_unit: float


@dataclass(frozen=True)
class OpenOrder:
    """A proposed position from an entry rule (legs + sizing), pre-fill."""

    label: str
    legs: tuple[BTLeg, ...]
    risk_per_unit: float


@dataclass(frozen=True)
class ClosedTrade:
    label: str
    entry_date: date
    exit_date: date
    pnl: float
    reason: str


def _row(chain: pd.DataFrame, leg: BTLeg) -> pd.Series:
    mask = (
        (chain["expiration"].dt.date == leg.expiration)
        & (np.isclose(chain["strike"].to_numpy(dtype=float), leg.strike))
        & (chain["right"] == leg.right)
    )
    sub = chain[mask]
    if sub.empty:
        raise KeyError(f"leg not in chain: {leg.right} {leg.strike} {leg.expiration}")
    return cast(pd.Series, sub.iloc[0])


def position_value_mid(chain: pd.DataFrame, position: BTPosition) -> float:
    """Liquidation value at mid (what you'd receive to close; negative = owe)."""
    total = 0.0
    for leg in position.legs:
        mid = float(_row(chain, leg)["mid"])
        total += leg.side * leg.quantity * leg.multiplier * mid
    return total


def _commission(legs: tuple[BTLeg, ...], config: BacktestConfig) -> float:
    contracts = sum(leg.quantity for leg in legs)
    return config.commission_per_contract * contracts


def entry_cashflow(chain: pd.DataFrame, legs: tuple[BTLeg, ...], config: BacktestConfig) -> float:
    """Signed cash at open: negative for a net debit, positive for a net credit."""
    cash = 0.0
    for leg in legs:
        row = _row(chain, leg)
        is_buy = leg.side > 0  # opening a long leg is a buy
        price = execution_price(
            is_buy=is_buy,
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            model=config.fill_model,
            spread_penalty=config.spread_penalty,
        )
        cash += (-1.0 if is_buy else 1.0) * price * leg.quantity * leg.multiplier
    return cash - _commission(legs, config)


def exit_cashflow(chain: pd.DataFrame, position: BTPosition, config: BacktestConfig) -> float:
    """Signed cash to close: sell longs (receive), buy back shorts (pay)."""
    cash = 0.0
    for leg in position.legs:
        row = _row(chain, leg)
        is_buy = leg.side < 0  # closing a short leg is a buy-to-close
        price = execution_price(
            is_buy=is_buy,
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            model=config.fill_model,
            spread_penalty=config.spread_penalty,
        )
        cash += (1.0 if leg.side > 0 else -1.0) * price * leg.quantity * leg.multiplier
    return cash - _commission(position.legs, config)


def settlement_cashflow(spot: float, position: BTPosition) -> float:
    """Cash at expiry: legs settle to intrinsic value against the underlying spot."""
    cash = 0.0
    for leg in position.legs:
        intrinsic = max(spot - leg.strike, 0.0) if leg.right == "C" else max(leg.strike - spot, 0.0)
        cash += leg.side * leg.quantity * leg.multiplier * intrinsic
    return cash
