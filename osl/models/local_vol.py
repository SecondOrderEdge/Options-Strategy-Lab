"""Dupire local volatility from an implied-vol surface (QuantLib, read-only).

Builds a Black variance surface from a grid of implied vols and queries the
Dupire local-vol surface on the same grid. Cells where the (numerically
delicate) Dupire computation fails are returned as NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_EVAL = (15, 5, 2026)


def dupire_local_vol(
    expiries_years: Sequence[float],
    strikes: Sequence[float],
    iv_grid: npt.ArrayLike,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
) -> FloatArray:
    """Return local vol on the (strike x expiry) grid.

    ``iv_grid`` has shape ``(len(strikes), len(expiries_years))``.
    """
    import QuantLib as ql

    iv = np.asarray(iv_grid, dtype=np.float64)
    day, month, year = _EVAL
    today = ql.Date(day, month, year)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    exp_dates = [today + round(t * 365) for t in expiries_years]
    vol_matrix = ql.Matrix(len(strikes), len(expiries_years))
    for i in range(len(strikes)):
        for j in range(len(expiries_years)):
            vol_matrix[i][j] = float(iv[i, j])

    bvs = ql.BlackVarianceSurface(today, calendar, exp_dates, list(strikes), vol_matrix, dc)
    bvs.enableExtrapolation()
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_yield, dc))
    local = ql.LocalVolSurface(ql.BlackVolTermStructureHandle(bvs), r_ts, q_ts, spot_h)
    local.enableExtrapolation()

    out = np.full_like(iv, np.nan)
    for j, t in enumerate(expiries_years):
        t_query = max(min(t * 0.9, t - 1e-3), 1e-3)
        for i, k in enumerate(strikes):
            try:
                out[i, j] = float(local.localVol(t_query, float(k), True))
            except RuntimeError:
                out[i, j] = np.nan
    return out
