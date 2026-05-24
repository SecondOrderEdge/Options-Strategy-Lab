"""Synthetic snapshot-history generator for demos and examples.

Produces a list of :class:`~osl.backtest.engine.BTDay` over a GBM underlying
path with a fixed strike grid priced via BSM at a flat IV. This is **synthetic
demonstration data**, not market data — use it to exercise the engine and the
Backtester page before real snapshots have accumulated.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from osl.backtest.engine import BTDay
from osl.data.base import CANONICAL_COLUMNS, coerce_chain
from osl.pricing.bsm import bsm_price


def synthetic_history(
    symbol: str = "DEMO",
    *,
    n_days: int = 180,
    start_spot: float = 100.0,
    rate: float = 0.0,
    vol: float = 0.25,
    drift: float = 0.0,
    seed: int = 0,
    expiry_every: int = 14,
    n_future_expiries: int = 6,
    strike_step: float = 5.0,
    n_strikes: int = 25,
    spread: float = 0.10,
    lookahead_on: int | None = None,
) -> list[BTDay]:
    """Build synthetic backtest days. ``lookahead_on`` stamps one day's chain ahead."""
    rng = np.random.default_rng(seed)
    base = date(2026, 1, 5)
    trading_days = [base + timedelta(days=i) for i in range(n_days)]

    dt = 1 / 365.0
    shocks = rng.normal((drift - 0.5 * vol**2) * dt, vol * np.sqrt(dt), size=n_days)
    spots = start_spot * np.exp(np.cumsum(shocks))

    strikes = start_spot + strike_step * np.arange(-(n_strikes // 2), n_strikes // 2 + 1)
    strikes = strikes[strikes > 0]

    n_exp = n_days // expiry_every + n_future_expiries + 2
    all_expiries = [base + timedelta(days=expiry_every * (k + 1)) for k in range(n_exp)]

    days: list[BTDay] = []
    for i, today in enumerate(trading_days):
        spot = float(spots[i])
        asof = datetime.combine(today, datetime.min.time(), tzinfo=UTC).replace(hour=20)
        qt = asof + timedelta(days=1) if lookahead_on == i else asof
        future = [e for e in all_expiries if (e - today).days > 0][:n_future_expiries]

        rows: list[dict[str, Any]] = []
        for exp in future:
            dte = (exp - today).days
            T = dte / 365.0
            exp_ts = pd.Timestamp(f"{exp.isoformat()} 16:00:00", tz="US/Eastern")
            for right, flag in (("C", "c"), ("P", "p")):
                mids = bsm_price(spot, strikes, T, rate, 0.0, vol, flag)
                for k, mid in zip(strikes, mids, strict=True):
                    mid_f = float(mid)
                    rows.append(
                        {
                            "underlying": symbol,
                            "option_symbol": f"{symbol}{dte}{right}{int(k * 1000):08d}",
                            "expiration": exp_ts,
                            "dte": dte,
                            "right": right,
                            "strike": float(k),
                            "bid": max(mid_f - spread / 2, 0.0),
                            "ask": mid_f + spread / 2,
                            "mid": mid_f,
                            "last": mid_f,
                            "mark": mid_f,
                            "bid_size": 10,
                            "ask_size": 10,
                            "volume": 200,
                            "open_interest": 1000,
                            "iv": vol,
                            "delta": np.nan,
                            "gamma": np.nan,
                            "theta": np.nan,
                            "vega": np.nan,
                            "rho": np.nan,
                            "theo": np.nan,
                            "multiplier": 100,
                            "exchange": "OPR",
                            "quote_time": qt,
                            "is_nonstandard": False,
                            "in_the_money": False,
                            "provider": "schwab",
                            "snapshot_id": f"snap-{i}",
                        }
                    )
        chain = coerce_chain(pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS)))
        days.append(BTDay(date=today, asof=asof, spot=spot, chain=chain))
    return days
