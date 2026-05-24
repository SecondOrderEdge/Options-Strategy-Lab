"""Put-call parity diagnostics.

European parity with continuous dividend yield::

    C - P = S * e^{-qT} - K * e^{-rT}

Deviations beyond a tolerance flag stale quotes, mispriced wings, or bad
borrow/dividend assumptions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from osl.utils.time import DAYS_PER_YEAR

FloatArray = npt.NDArray[np.float64]


def parity_gap(
    call: npt.ArrayLike,
    put: npt.ArrayLike,
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    q: npt.ArrayLike,
) -> FloatArray:
    """Return ``(C - P) - (S e^{-qT} - K e^{-rT})`` (zero under exact parity)."""
    call_, put_, S_, K_, T_, r_, q_ = (
        np.asarray(x, dtype=np.float64) for x in (call, put, S, K, T, r, q)
    )
    forward_value = S_ * np.exp(-q_ * T_) - K_ * np.exp(-r_ * T_)
    return np.asarray((call_ - put_) - forward_value, dtype=np.float64)


def put_call_parity_violations(
    chain: pd.DataFrame,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    price_col: str = "mid",
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Pair calls and puts by (expiration, strike) and flag parity breaches.

    Returns one row per strike present on both sides, with the parity ``gap``
    and a ``violation`` flag where ``|gap| > tolerance``. Rows with a missing
    price on either side are dropped.
    """
    calls = chain[chain["right"] == "C"]
    puts = chain[chain["right"] == "P"]
    merged = calls.merge(
        puts,
        on=["expiration", "strike"],
        suffixes=("_call", "_put"),
        how="inner",
    )
    merged = merged.dropna(subset=[f"{price_col}_call", f"{price_col}_put"])
    if merged.empty:
        return pd.DataFrame(
            columns=["expiration", "strike", "T", "call", "put", "gap", "violation"]
        )

    T = merged["dte_call"].to_numpy(dtype=np.float64) / DAYS_PER_YEAR
    gap = parity_gap(
        merged[f"{price_col}_call"].to_numpy(dtype=np.float64),
        merged[f"{price_col}_put"].to_numpy(dtype=np.float64),
        spot,
        merged["strike"].to_numpy(dtype=np.float64),
        T,
        rate,
        dividend_yield,
    )
    out = pd.DataFrame(
        {
            "expiration": merged["expiration"].to_numpy(),
            "strike": merged["strike"].to_numpy(dtype=np.float64),
            "T": T,
            "call": merged[f"{price_col}_call"].to_numpy(dtype=np.float64),
            "put": merged[f"{price_col}_put"].to_numpy(dtype=np.float64),
            "gap": gap,
            "violation": np.abs(gap) > tolerance,
        }
    )
    return out.sort_values(["expiration", "strike"]).reset_index(drop=True)
