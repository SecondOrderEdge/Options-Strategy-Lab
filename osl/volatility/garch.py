"""GARCH(1,1) / GJR-GARCH volatility forecasting via the ``arch`` package.

Fits to daily log returns and reports persistence plus an annualized
volatility forecast over a horizon. Returns are scaled to percent internally
(arch's numerically-stable convention) and converted back to decimals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model

Model = Literal["garch", "gjr"]
Dist = Literal["normal", "t"]


@dataclass(frozen=True)
class GarchForecast:
    model: str
    dist: str
    params: dict[str, float]
    persistence: float
    horizon_days: int
    annualized_vol: float  # decimal, forecast over the horizon
    conditional_vol: float  # decimal, last in-sample annualized vol
    converged: bool


def fit_garch(
    log_returns: pd.Series,
    *,
    model: Model = "garch",
    dist: Dist = "normal",
    horizon: int = 21,
    annualization: int = 252,
) -> GarchForecast:
    """Fit GARCH/GJR to daily log returns and forecast volatility."""
    rets = log_returns.dropna().to_numpy(dtype=np.float64) * 100.0  # percent
    o = 1 if model == "gjr" else 0
    am = arch_model(rets, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)
    res = am.fit(disp="off")

    params = {str(k): float(v) for k, v in res.params.items()}
    alpha = params.get("alpha[1]", 0.0)
    beta = params.get("beta[1]", 0.0)
    gamma = params.get("gamma[1]", 0.0)
    persistence = alpha + beta + 0.5 * gamma

    fc = res.forecast(horizon=horizon, reindex=False)
    daily_var_pct2 = np.asarray(fc.variance.values[-1], dtype=np.float64)  # percent^2 per day
    annualized_vol = float(np.sqrt(daily_var_pct2.mean()) / 100.0 * np.sqrt(annualization))

    cond_vol_pct = float(res.conditional_volatility[-1])
    conditional_vol = cond_vol_pct / 100.0 * float(np.sqrt(annualization))

    return GarchForecast(
        model=model,
        dist=dist,
        params=params,
        persistence=persistence,
        horizon_days=horizon,
        annualized_vol=annualized_vol,
        conditional_vol=conditional_vol,
        converged=bool(getattr(res, "convergence_flag", 0) == 0),
    )
