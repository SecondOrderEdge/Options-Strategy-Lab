"""Builders for the standard option strategies.

Each builder takes a :class:`ChainContext` (canonical chain + spot/rate/div) plus
explicit strike/expiration parameters and returns a :class:`Strategy`. IV is the
provider's when sane, otherwise recomputed from the mid price.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, cast

import numpy as np
import pandas as pd

from osl.pricing.iv import implied_vol
from osl.strategy.legs import Leg, OptionType, Side, Strategy
from osl.utils.time import DAYS_PER_YEAR

IVSource = Literal["provider", "recompute", "auto"]


@dataclass
class ChainContext:
    """A chain plus the market context needed to build strategies."""

    chain: pd.DataFrame
    underlying: str
    spot: float
    rate: float
    dividend_yield: float = 0.0
    iv_source: IVSource = "auto"

    def _row(self, expiration: date, strike: float, right: str) -> pd.Series:
        c = self.chain
        mask = (
            (c["expiration"].dt.date == expiration)
            & (np.isclose(c["strike"].to_numpy(dtype=float), strike))
            & (c["right"] == right)
        )
        sub = c[mask]
        if sub.empty:
            raise KeyError(f"no {right} contract at strike {strike} exp {expiration}")
        return cast(pd.Series, sub.iloc[0])

    def leg(
        self,
        expiration: date,
        strike: float,
        option_type: OptionType,
        side: Side,
        *,
        quantity: int = 1,
    ) -> Leg:
        right = option_type.value
        row = self._row(expiration, strike, right)
        premium = float(row["mid"])
        if not np.isfinite(premium) or premium <= 0:
            premium = float(row["last"])
        dte = int(row["dte"])
        T = dte / DAYS_PER_YEAR
        flag = "c" if option_type is OptionType.CALL else "p"

        provider_iv = float(row["iv"])
        if self.iv_source == "provider":
            iv = provider_iv
        elif self.iv_source == "recompute":
            iv = self._solve_iv(premium, strike, T, flag)
        else:
            iv = (
                provider_iv
                if np.isfinite(provider_iv) and provider_iv > 0
                else self._solve_iv(premium, strike, T, flag)
            )
        return Leg(
            option_type=option_type,
            side=side,
            strike=float(strike),
            expiration=expiration,
            dte=dte,
            premium=premium,
            iv=float(iv),
            quantity=quantity,
        )

    def _solve_iv(self, premium: float, strike: float, T: float, flag: str) -> float:
        if not np.isfinite(premium) or premium <= 0:
            return float("nan")
        try:
            return implied_vol(premium, self.spot, strike, T, self.rate, self.dividend_yield, flag)
        except Exception:
            return float("nan")

    def _strategy(self, name: str, legs: list[Leg], **extra: float) -> Strategy:
        stock_quantity = int(extra.pop("stock_quantity", 0))
        stock_entry = float(extra.pop("stock_entry", 0.0))
        return Strategy(
            name=name,
            underlying=self.underlying,
            legs=tuple(legs),
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            stock_quantity=stock_quantity,
            stock_entry=stock_entry,
        )


C = OptionType.CALL
P = OptionType.PUT
LONG = Side.LONG
SHORT = Side.SHORT


# --- single leg ---
def long_call(ctx: ChainContext, *, expiration: date, strike: float, qty: int = 1) -> Strategy:
    return ctx._strategy("Long Call", [ctx.leg(expiration, strike, C, LONG, quantity=qty)])


def long_put(ctx: ChainContext, *, expiration: date, strike: float, qty: int = 1) -> Strategy:
    return ctx._strategy("Long Put", [ctx.leg(expiration, strike, P, LONG, quantity=qty)])


def cash_secured_put(
    ctx: ChainContext, *, expiration: date, strike: float, qty: int = 1
) -> Strategy:
    return ctx._strategy("Cash-Secured Put", [ctx.leg(expiration, strike, P, SHORT, quantity=qty)])


# --- vertical spreads ---
def bull_call_spread(
    ctx: ChainContext, *, expiration: date, long_strike: float, short_strike: float, qty: int = 1
) -> Strategy:
    return ctx._strategy(
        "Bull Call Spread",
        [
            ctx.leg(expiration, long_strike, C, LONG, quantity=qty),
            ctx.leg(expiration, short_strike, C, SHORT, quantity=qty),
        ],
    )


def bear_put_spread(
    ctx: ChainContext, *, expiration: date, long_strike: float, short_strike: float, qty: int = 1
) -> Strategy:
    return ctx._strategy(
        "Bear Put Spread",
        [
            ctx.leg(expiration, long_strike, P, LONG, quantity=qty),
            ctx.leg(expiration, short_strike, P, SHORT, quantity=qty),
        ],
    )


def short_put_spread(
    ctx: ChainContext, *, expiration: date, short_strike: float, long_strike: float, qty: int = 1
) -> Strategy:
    """Bull put spread (credit): short higher put, long lower put."""
    return ctx._strategy(
        "Short Put Spread",
        [
            ctx.leg(expiration, short_strike, P, SHORT, quantity=qty),
            ctx.leg(expiration, long_strike, P, LONG, quantity=qty),
        ],
    )


def short_call_spread(
    ctx: ChainContext, *, expiration: date, short_strike: float, long_strike: float, qty: int = 1
) -> Strategy:
    """Bear call spread (credit): short lower call, long higher call."""
    return ctx._strategy(
        "Short Call Spread",
        [
            ctx.leg(expiration, short_strike, C, SHORT, quantity=qty),
            ctx.leg(expiration, long_strike, C, LONG, quantity=qty),
        ],
    )


# --- stock combinations ---
def covered_call(ctx: ChainContext, *, expiration: date, strike: float, qty: int = 1) -> Strategy:
    return ctx._strategy(
        "Covered Call",
        [ctx.leg(expiration, strike, C, SHORT, quantity=qty)],
        stock_quantity=qty * 100,
        stock_entry=ctx.spot,
    )


def protective_put(ctx: ChainContext, *, expiration: date, strike: float, qty: int = 1) -> Strategy:
    return ctx._strategy(
        "Protective Put",
        [ctx.leg(expiration, strike, P, LONG, quantity=qty)],
        stock_quantity=qty * 100,
        stock_entry=ctx.spot,
    )


def collar(
    ctx: ChainContext, *, expiration: date, put_strike: float, call_strike: float, qty: int = 1
) -> Strategy:
    return ctx._strategy(
        "Collar",
        [
            ctx.leg(expiration, put_strike, P, LONG, quantity=qty),
            ctx.leg(expiration, call_strike, C, SHORT, quantity=qty),
        ],
        stock_quantity=qty * 100,
        stock_entry=ctx.spot,
    )


# --- straddles / strangles ---
def long_straddle(ctx: ChainContext, *, expiration: date, strike: float, qty: int = 1) -> Strategy:
    return ctx._strategy(
        "Long Straddle",
        [
            ctx.leg(expiration, strike, C, LONG, quantity=qty),
            ctx.leg(expiration, strike, P, LONG, quantity=qty),
        ],
    )


def short_straddle(ctx: ChainContext, *, expiration: date, strike: float, qty: int = 1) -> Strategy:
    return ctx._strategy(
        "Short Straddle",
        [
            ctx.leg(expiration, strike, C, SHORT, quantity=qty),
            ctx.leg(expiration, strike, P, SHORT, quantity=qty),
        ],
    )


def long_strangle(
    ctx: ChainContext, *, expiration: date, put_strike: float, call_strike: float, qty: int = 1
) -> Strategy:
    return ctx._strategy(
        "Long Strangle",
        [
            ctx.leg(expiration, call_strike, C, LONG, quantity=qty),
            ctx.leg(expiration, put_strike, P, LONG, quantity=qty),
        ],
    )


def short_strangle(
    ctx: ChainContext, *, expiration: date, put_strike: float, call_strike: float, qty: int = 1
) -> Strategy:
    return ctx._strategy(
        "Short Strangle",
        [
            ctx.leg(expiration, call_strike, C, SHORT, quantity=qty),
            ctx.leg(expiration, put_strike, P, SHORT, quantity=qty),
        ],
    )


# --- four-leg ---
def iron_condor(
    ctx: ChainContext,
    *,
    expiration: date,
    long_put_strike: float,
    short_put_strike: float,
    short_call_strike: float,
    long_call_strike: float,
    qty: int = 1,
) -> Strategy:
    return ctx._strategy(
        "Iron Condor",
        [
            ctx.leg(expiration, long_put_strike, P, LONG, quantity=qty),
            ctx.leg(expiration, short_put_strike, P, SHORT, quantity=qty),
            ctx.leg(expiration, short_call_strike, C, SHORT, quantity=qty),
            ctx.leg(expiration, long_call_strike, C, LONG, quantity=qty),
        ],
    )


def iron_fly(
    ctx: ChainContext,
    *,
    expiration: date,
    long_put_strike: float,
    center_strike: float,
    long_call_strike: float,
    qty: int = 1,
) -> Strategy:
    return ctx._strategy(
        "Iron Fly",
        [
            ctx.leg(expiration, long_put_strike, P, LONG, quantity=qty),
            ctx.leg(expiration, center_strike, P, SHORT, quantity=qty),
            ctx.leg(expiration, center_strike, C, SHORT, quantity=qty),
            ctx.leg(expiration, long_call_strike, C, LONG, quantity=qty),
        ],
    )


def broken_wing_butterfly(
    ctx: ChainContext,
    *,
    expiration: date,
    lower_strike: float,
    center_strike: float,
    upper_strike: float,
    option_type: OptionType = C,
    qty: int = 1,
) -> Strategy:
    """Asymmetric butterfly: long 1 lower, short 2 center, long 1 upper (wings unequal)."""
    return ctx._strategy(
        "Broken-Wing Butterfly",
        [
            ctx.leg(expiration, lower_strike, option_type, LONG, quantity=qty),
            ctx.leg(expiration, center_strike, option_type, SHORT, quantity=2 * qty),
            ctx.leg(expiration, upper_strike, option_type, LONG, quantity=qty),
        ],
    )


def ratio_spread(
    ctx: ChainContext,
    *,
    expiration: date,
    long_strike: float,
    short_strike: float,
    ratio: int = 2,
    option_type: OptionType = C,
    qty: int = 1,
) -> Strategy:
    """Long 1, short ``ratio`` further-OTM (unbounded risk on the short side)."""
    return ctx._strategy(
        "Ratio Spread",
        [
            ctx.leg(expiration, long_strike, option_type, LONG, quantity=qty),
            ctx.leg(expiration, short_strike, option_type, SHORT, quantity=ratio * qty),
        ],
    )


def jade_lizard(
    ctx: ChainContext,
    *,
    expiration: date,
    short_put_strike: float,
    short_call_strike: float,
    long_call_strike: float,
    qty: int = 1,
) -> Strategy:
    """Short put + short call spread, sized so total credit removes upside risk."""
    return ctx._strategy(
        "Jade Lizard",
        [
            ctx.leg(expiration, short_put_strike, P, SHORT, quantity=qty),
            ctx.leg(expiration, short_call_strike, C, SHORT, quantity=qty),
            ctx.leg(expiration, long_call_strike, C, LONG, quantity=qty),
        ],
    )


# --- multi-expiry ---
def calendar(
    ctx: ChainContext,
    *,
    near_expiration: date,
    far_expiration: date,
    strike: float,
    option_type: OptionType = C,
    qty: int = 1,
) -> Strategy:
    return ctx._strategy(
        "Calendar",
        [
            ctx.leg(near_expiration, strike, option_type, SHORT, quantity=qty),
            ctx.leg(far_expiration, strike, option_type, LONG, quantity=qty),
        ],
    )


def diagonal(
    ctx: ChainContext,
    *,
    near_expiration: date,
    far_expiration: date,
    short_strike: float,
    long_strike: float,
    option_type: OptionType = C,
    qty: int = 1,
) -> Strategy:
    return ctx._strategy(
        "Diagonal",
        [
            ctx.leg(near_expiration, short_strike, option_type, SHORT, quantity=qty),
            ctx.leg(far_expiration, long_strike, option_type, LONG, quantity=qty),
        ],
    )


def pmcc(
    ctx: ChainContext,
    *,
    near_expiration: date,
    far_expiration: date,
    long_strike: float,
    short_strike: float,
    qty: int = 1,
) -> Strategy:
    """Poor man's covered call: long deep-ITM far call + short near OTM call."""
    return ctx._strategy(
        "PMCC",
        [
            ctx.leg(far_expiration, long_strike, C, LONG, quantity=qty),
            ctx.leg(near_expiration, short_strike, C, SHORT, quantity=qty),
        ],
    )
