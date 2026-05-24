"""IV rank, IV percentile, and volatility cones.

IV **rank** locates today's IV within its trailing min/max band; IV
**percentile** is the fraction of trailing days with IV at or below today's.
They answer different questions and the blueprint mandates showing both.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from osl.volatility.realized import ANNUALIZATION, close_to_close

DEFAULT_LOOKBACK = 252
DEFAULT_WINDOWS: tuple[int, ...] = (10, 20, 30, 60, 90, 180)
DEFAULT_PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)


def iv_rank(series: pd.Series, lookback: int = DEFAULT_LOOKBACK) -> float:
    """(IV_today − min) / (max − min) over the trailing ``lookback`` window."""
    window = series.dropna().iloc[-lookback:]
    if window.empty:
        return float("nan")
    lo, hi, last = window.min(), window.max(), window.iloc[-1]
    if hi == lo:
        return float("nan")
    return float((last - lo) / (hi - lo))


def iv_percentile(series: pd.Series, lookback: int = DEFAULT_LOOKBACK) -> float:
    """Fraction of the trailing ``lookback`` window at or below today's IV."""
    window = series.dropna().iloc[-lookback:]
    if window.empty:
        return float("nan")
    last = window.iloc[-1]
    return float((window <= last).mean())


def vol_cone(
    close: pd.Series,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    percentiles: Sequence[int] = DEFAULT_PERCENTILES,
    annualization: int = ANNUALIZATION,
) -> pd.DataFrame:
    """Realized-vol percentile cone across rolling windows.

    Returns a DataFrame indexed by window length with one column per requested
    percentile (e.g. ``p10``…``p90``) plus ``current`` (the most recent rolling
    realized vol at that window).
    """
    df = pd.DataFrame({"close": close})
    rows: list[dict[str, float]] = []
    index: list[int] = []
    for w in windows:
        rv = close_to_close(df, window=w, annualization=annualization).dropna()
        if rv.empty:
            continue
        row = {f"p{p}": float(np.percentile(rv, p)) for p in percentiles}
        row["current"] = float(rv.iloc[-1])
        rows.append(row)
        index.append(w)
    out = pd.DataFrame(rows, index=pd.Index(index, name="window"))
    return out
