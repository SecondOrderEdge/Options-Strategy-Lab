"""Mark-to-market valuation of a strategy under spot / time / volatility shocks.

Unlike :meth:`Strategy.pnl` (terminal, intrinsic at the primary expiry), this
module values the position at an arbitrary point *before* expiry: each leg is
repriced by BSM with its remaining time and a shocked volatility. At
``days_forward = primary_dte`` and ``vol_shock = 0`` it reduces to the terminal
payoff; at ``days_forward = 0`` and ``vol_shock = 0`` to the entry value.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from osl.pricing.bsm import bsm_price
from osl.strategy.legs import Strategy
from osl.utils.time import DAYS_PER_YEAR

FloatArray = npt.NDArray[np.float64]
_MIN_T = 1e-9
_MIN_SIGMA = 1e-9


def value_at(
    strategy: Strategy,
    spot: npt.ArrayLike,
    *,
    days_forward: float = 0.0,
    vol_shock: float = 0.0,
) -> FloatArray:
    """Position value at ``spot``, ``days_forward`` days from now, IV shifted by ``vol_shock``."""
    spot_arr = np.asarray(spot, dtype=np.float64)
    total = np.zeros_like(spot_arr)
    dt = days_forward / DAYS_PER_YEAR

    for leg in strategy.legs:
        t_rem = leg.T - dt
        scale = leg.sign * leg.quantity * leg.multiplier
        if t_rem <= _MIN_T:
            total = total + scale * leg.intrinsic(spot_arr)
        else:
            sigma = max(leg.iv + vol_shock, _MIN_SIGMA)
            price = bsm_price(
                spot_arr,
                leg.strike,
                t_rem,
                strategy.rate,
                strategy.dividend_yield,
                sigma,
                leg.right,
            )
            total = total + scale * price

    if strategy.stock_quantity:
        total = total + strategy.stock_quantity * spot_arr
    return np.asarray(total, dtype=np.float64)


def pnl_at(
    strategy: Strategy,
    spot: npt.ArrayLike,
    *,
    days_forward: float = 0.0,
    vol_shock: float = 0.0,
) -> FloatArray:
    """P&L = mark-to-market value minus the net debit paid to open."""
    return np.asarray(
        value_at(strategy, spot, days_forward=days_forward, vol_shock=vol_shock)
        - strategy.net_debit,
        dtype=np.float64,
    )


def payoff_curve(
    strategy: Strategy,
    *,
    n: int = 201,
    lo_mult: float = 0.6,
    hi_mult: float = 1.4,
    days_forward: float | None = None,
    vol_shock: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(spots, pnl)`` across a spot grid (terminal payoff by default)."""
    df = float(strategy.primary_dte) if days_forward is None else days_forward
    strikes = list(strategy.strikes)
    lo = min([strategy.spot * lo_mult, *strikes])
    hi = max([strategy.spot * hi_mult, *strikes])
    spots = np.linspace(lo, hi, n)
    pnl = pnl_at(strategy, spots, days_forward=df, vol_shock=vol_shock)
    return spots, pnl
