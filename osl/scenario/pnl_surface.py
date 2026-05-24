"""P&L surface over (spot, time-decay) at a fixed volatility shock."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from osl.scenario.payoff import pnl_at
from osl.strategy.legs import Strategy

FloatArray = npt.NDArray[np.float64]


def pnl_grid(
    strategy: Strategy,
    *,
    spot_shocks: Sequence[float] = tuple(np.linspace(-0.3, 0.3, 31)),
    days_forward: Sequence[float] | None = None,
    vol_shock: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(spots, days, Z)`` where ``Z[i, j]`` is P&L at day i, spot j.

    ``spot_shocks`` are fractional moves from the current spot; ``days_forward``
    defaults to an even split from now to the primary expiry.
    """
    spots = strategy.spot * (1.0 + np.asarray(spot_shocks, dtype=np.float64))
    if days_forward is None:
        days = np.linspace(0.0, float(strategy.primary_dte), 13)
    else:
        days = np.asarray(days_forward, dtype=np.float64)

    z = np.empty((days.size, spots.size), dtype=np.float64)
    for i, d in enumerate(days):
        z[i, :] = pnl_at(strategy, spots, days_forward=float(d), vol_shock=vol_shock)
    return np.asarray(spots, dtype=np.float64), np.asarray(days, dtype=np.float64), z
