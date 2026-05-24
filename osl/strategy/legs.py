"""Core option-leg and multi-leg strategy data structures.

A :class:`Leg` is one option contract with a side (long/short) and quantity. A
:class:`Strategy` is a named bundle of legs over one underlying, optionally with
a stock position (for covered calls, collars, protective puts). Sign convention:
``net_debit`` is positive when the position costs money to open (debit) and
negative for a credit; P&L at expiry is ``terminal_value(S) - net_debit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum, StrEnum

import numpy as np
import numpy.typing as npt

from osl.pricing.bsm import bsm_price
from osl.utils.time import DAYS_PER_YEAR

FloatArray = npt.NDArray[np.float64]


class OptionType(StrEnum):
    CALL = "C"
    PUT = "P"


class Side(IntEnum):
    LONG = 1
    SHORT = -1


@dataclass(frozen=True)
class Leg:
    """A single option contract within a strategy."""

    option_type: OptionType
    side: Side
    strike: float
    expiration: date
    dte: int
    premium: float  # per-share entry price (mid)
    iv: float
    quantity: int = 1
    multiplier: int = 100

    @property
    def sign(self) -> int:
        return int(self.side.value)

    @property
    def right(self) -> str:
        return self.option_type.value

    @property
    def T(self) -> float:
        return self.dte / DAYS_PER_YEAR

    @property
    def entry_cost(self) -> float:
        """Dollar cash outflow to open (positive = debit paid)."""
        return self.sign * self.premium * self.quantity * self.multiplier

    def intrinsic(self, spot: FloatArray) -> FloatArray:
        if self.option_type is OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)

    def terminal_value(self, spot: FloatArray) -> FloatArray:
        """Position value at this leg's own expiry (intrinsic, signed)."""
        return self.sign * self.intrinsic(spot) * self.quantity * self.multiplier

    def value_at(self, spot: FloatArray, t_remaining: float, r: float, q: float) -> FloatArray:
        """BSM mark of the (still-open) leg with ``t_remaining`` years left."""
        price = bsm_price(spot, self.strike, t_remaining, r, q, self.iv, self.right)
        return self.sign * price * self.quantity * self.multiplier


@dataclass(frozen=True)
class Strategy:
    """A named multi-leg position over one underlying."""

    name: str
    underlying: str
    legs: tuple[Leg, ...]
    spot: float
    rate: float
    dividend_yield: float = 0.0
    stock_quantity: int = 0  # signed shares
    stock_entry: float = 0.0
    meta: dict[str, float] = field(default_factory=dict)

    @property
    def primary_dte(self) -> int:
        return min(leg.dte for leg in self.legs)

    @property
    def primary_T(self) -> float:
        return self.primary_dte / DAYS_PER_YEAR

    @property
    def all_same_expiry(self) -> bool:
        return len({leg.dte for leg in self.legs}) == 1

    @property
    def strikes(self) -> tuple[float, ...]:
        return tuple(sorted({leg.strike for leg in self.legs}))

    @property
    def net_debit(self) -> float:
        """Positive = net debit paid; negative = net credit received."""
        return sum(leg.entry_cost for leg in self.legs) + self.stock_quantity * self.stock_entry

    @property
    def is_credit(self) -> bool:
        return self.net_debit < 0

    def terminal_value(self, spot: FloatArray) -> FloatArray:
        """Position value at the primary (nearest) expiry.

        Legs expiring at the primary date contribute intrinsic value; legs
        expiring later are repriced via BSM with the remaining time (constant
        IV). Stock contributes ``shares * S``.
        """
        spot = np.asarray(spot, dtype=np.float64)
        total = np.zeros_like(spot)
        primary = self.primary_dte
        for leg in self.legs:
            if leg.dte == primary:
                total = total + leg.terminal_value(spot)
            else:
                t_rem = (leg.dte - primary) / DAYS_PER_YEAR
                total = total + leg.value_at(spot, t_rem, self.rate, self.dividend_yield)
        if self.stock_quantity:
            total = total + self.stock_quantity * spot
        return np.asarray(total, dtype=np.float64)

    def pnl(self, spot: FloatArray) -> FloatArray:
        """P&L at the primary expiry across terminal spot prices."""
        return np.asarray(self.terminal_value(spot) - self.net_debit, dtype=np.float64)
