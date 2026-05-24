"""Turn a canonical option chain into per-expiry smiles ready for SVI.

For each expiry we build OTM-side points (calls above the forward, puts below)
in log-moneyness ``k = ln(K/F)`` with total variance ``w = iv^2 T`` and vega
weights. IV is taken from the provider when sane, otherwise recomputed from the
mid price (Jaeckel with bisection fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from osl.pricing.bsm import bsm_greeks
from osl.pricing.iv import implied_vol
from osl.utils.time import DAYS_PER_YEAR

FloatArray = npt.NDArray[np.float64]
IVSource = Literal["provider", "recompute", "auto"]


@dataclass(frozen=True)
class Smile:
    """A single-expiry implied-volatility smile in log-moneyness."""

    expiration: date
    T: float
    forward: float
    strikes: FloatArray
    k: FloatArray
    iv: FloatArray
    total_variance: FloatArray
    weights: FloatArray

    def __len__(self) -> int:
        return int(self.strikes.size)


def _solve_iv(
    price: float, spot: float, K: float, T: float, r: float, q: float, right: str
) -> float:
    try:
        return implied_vol(price, spot, K, T, r, q, right, method="jaeckel")
    except Exception:
        try:
            return implied_vol(price, spot, K, T, r, q, right, method="bisect")
        except Exception:
            return float("nan")


def prepare_smile(
    chain: pd.DataFrame,
    expiration: date,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    iv_source: IVSource = "auto",
    price_col: str = "mid",
) -> Smile:
    """Build a :class:`Smile` for one expiry from the canonical chain."""
    sub = chain[chain["expiration"].dt.date == expiration]
    if sub.empty:
        raise ValueError(f"no contracts for expiration {expiration}")

    T = float(sub["dte"].iloc[0]) / DAYS_PER_YEAR
    if T <= 0:
        raise ValueError("expiration already at/!past expiry (T<=0)")
    forward = spot * np.exp((rate - dividend_yield) * T)

    strikes: list[float] = []
    ivs: list[float] = []
    weights: list[float] = []
    for strike, grp in sub.groupby("strike"):
        K = float(strike)
        right = "C" if forward <= K else "P"
        row = grp[grp["right"] == right]
        if row.empty:
            continue
        price = float(row[price_col].iloc[0])
        if not np.isfinite(price) or price <= 0:
            continue

        provider_iv = float(row["iv"].iloc[0])
        flag = "c" if right == "C" else "p"
        if iv_source == "provider":
            iv = provider_iv
        elif iv_source == "recompute":
            iv = _solve_iv(price, spot, K, T, rate, dividend_yield, flag)
        else:  # auto
            iv = (
                provider_iv
                if np.isfinite(provider_iv) and provider_iv > 0
                else _solve_iv(price, spot, K, T, rate, dividend_yield, flag)
            )
        if not np.isfinite(iv) or iv <= 0:
            continue

        vega = float(bsm_greeks(spot, K, T, rate, dividend_yield, iv, flag)["vega"].item())
        strikes.append(K)
        ivs.append(iv)
        weights.append(max(vega, 1e-8))

    if not strikes:
        raise ValueError(f"no usable quotes for expiration {expiration}")

    strike_arr = np.array(strikes, dtype=np.float64)
    order = np.argsort(strike_arr)
    strike_arr = strike_arr[order]
    iv_arr = np.array(ivs, dtype=np.float64)[order]
    weight_arr = np.array(weights, dtype=np.float64)[order]
    k_arr = np.log(strike_arr / forward)
    w_arr = iv_arr * iv_arr * T

    return Smile(
        expiration=expiration,
        T=T,
        forward=float(forward),
        strikes=strike_arr,
        k=k_arr,
        iv=iv_arr,
        total_variance=w_arr,
        weights=weight_arr,
    )


def prepare_smiles(
    chain: pd.DataFrame,
    *,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    iv_source: IVSource = "auto",
    min_points: int = 5,
) -> list[Smile]:
    """Build smiles for every expiry with at least ``min_points`` usable quotes."""
    smiles: list[Smile] = []
    for expiration in sorted(chain["expiration"].dt.date.unique()):
        try:
            smile = prepare_smile(
                chain,
                expiration,
                spot=spot,
                rate=rate,
                dividend_yield=dividend_yield,
                iv_source=iv_source,
            )
        except ValueError:
            continue
        if len(smile) >= min_points:
            smiles.append(smile)
    return smiles
