"""Candidate enumeration over strikes and expiries for a directional view.

Generates a bounded set of strategies appropriate to the requested ``view``
(bullish / bearish / neutral / volatile). Contracts with a zero bid are excluded
by default (they cannot be sold), satisfying the liquidity-filter rule.
"""

from __future__ import annotations

import contextlib
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from osl.strategy import strategies as S
from osl.strategy.legs import Strategy
from osl.strategy.strategies import ChainContext

View = Literal["bullish", "bearish", "neutral", "volatile", "all"]


def _usable_strikes(
    chain: pd.DataFrame, expiration: date, right: str, *, allow_zero_bid: bool
) -> list[float]:
    mask = (chain["expiration"].dt.date == expiration) & (chain["right"] == right)
    sub = chain[mask]
    if not allow_zero_bid:
        sub = sub[sub["bid"].fillna(0.0) > 0.0]
    return sorted(float(s) for s in sub["strike"].unique())


def _nearest_index(strikes: list[float], target: float) -> int:
    return int(np.argmin(np.abs(np.array(strikes) - target)))


def enumerate_candidates(
    ctx: ChainContext,
    *,
    view: View = "all",
    dte_range: tuple[int, int] = (20, 60),
    max_strikes_per_leg: int = 3,
    allow_zero_bid: bool = False,
) -> list[Strategy]:
    """Enumerate strategy candidates within the filters for the given view."""
    chain = ctx.chain
    spot = ctx.spot
    dte_min, dte_max = dte_range
    expirations = sorted(
        d
        for d in chain["expiration"].dt.date.unique()
        if dte_min <= int(chain.loc[chain["expiration"].dt.date == d, "dte"].iloc[0]) <= dte_max
    )

    out: list[Strategy] = []
    want = {view} if view != "all" else {"bullish", "bearish", "neutral", "volatile"}

    for exp in expirations:
        calls = _usable_strikes(chain, exp, "C", allow_zero_bid=allow_zero_bid)
        puts = _usable_strikes(chain, exp, "P", allow_zero_bid=allow_zero_bid)
        if len(calls) < 3 or len(puts) < 3:
            continue
        c_atm = _nearest_index(calls, spot)
        p_atm = _nearest_index(puts, spot)
        step = max(1, max_strikes_per_leg)

        def _safe(fn: object) -> None:
            with contextlib.suppress(KeyError, ValueError):
                out.append(fn())  # type: ignore[operator]

        if "bullish" in want:
            for off in range(0, max_strikes_per_leg):
                ci = min(c_atm + off, len(calls) - 1)
                _safe(lambda exp=exp, k=calls[ci]: S.long_call(ctx, expiration=exp, strike=k))
            if c_atm + step < len(calls):
                _safe(
                    lambda exp=exp, lo=calls[c_atm], hi=calls[c_atm + step]: S.bull_call_spread(
                        ctx, expiration=exp, long_strike=lo, short_strike=hi
                    )
                )
            if p_atm - step >= 0:
                _safe(
                    lambda exp=exp, sh=puts[p_atm], lo=puts[p_atm - step]: S.short_put_spread(
                        ctx, expiration=exp, short_strike=sh, long_strike=lo
                    )
                )

        if "bearish" in want:
            for off in range(0, max_strikes_per_leg):
                pi = max(p_atm - off, 0)
                _safe(lambda exp=exp, k=puts[pi]: S.long_put(ctx, expiration=exp, strike=k))
            if c_atm + step < len(calls):
                _safe(
                    lambda exp=exp, sh=calls[c_atm], hi=calls[c_atm + step]: S.short_call_spread(
                        ctx, expiration=exp, short_strike=sh, long_strike=hi
                    )
                )

        if "neutral" in want and p_atm - step >= 0 and c_atm + step < len(calls):
            lp = puts[max(p_atm - 2 * step, 0)]
            sp = puts[p_atm - step]
            sc = calls[c_atm + step]
            lc = calls[min(c_atm + 2 * step, len(calls) - 1)]
            _safe(
                lambda exp=exp, lp=lp, sp=sp, sc=sc, lc=lc: S.iron_condor(
                    ctx,
                    expiration=exp,
                    long_put_strike=lp,
                    short_put_strike=sp,
                    short_call_strike=sc,
                    long_call_strike=lc,
                )
            )
            _safe(
                lambda exp=exp, ps=sp, cs=sc: S.short_strangle(
                    ctx, expiration=exp, put_strike=ps, call_strike=cs
                )
            )

        if "volatile" in want:
            _safe(lambda exp=exp, k=calls[c_atm]: S.long_straddle(ctx, expiration=exp, strike=k))
            if p_atm - step >= 0 and c_atm + step < len(calls):
                _safe(
                    lambda exp=exp, ps=puts[p_atm - step], cs=calls[c_atm + step]: S.long_strangle(
                        ctx, expiration=exp, put_strike=ps, call_strike=cs
                    )
                )

    return out
