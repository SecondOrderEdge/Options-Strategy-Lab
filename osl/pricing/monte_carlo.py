"""Geometric Brownian Motion Monte Carlo for terminal-distribution sampling.

Used by the strategy engine for risk-neutral POP/EV. Antithetic variates reduce
variance; the generator is seeded for reproducibility.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def gbm_terminal(
    spot: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    *,
    n: int = 50_000,
    seed: int = 42,
    antithetic: bool = True,
) -> FloatArray:
    """Sample terminal prices ``S_T`` under risk-neutral GBM.

    ``S_T = F * exp(-0.5 sigma^2 T + sigma sqrt(T) Z)`` with forward
    ``F = spot * exp((r - q) T)``, so ``E_Q[S_T] = F``.
    """
    rng = np.random.default_rng(seed)
    forward = spot * np.exp((r - q) * T)
    drift = -0.5 * sigma * sigma * T
    vol = sigma * np.sqrt(T)

    if antithetic:
        half = (n + 1) // 2
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])[:n]
    else:
        z = rng.standard_normal(n)

    return np.asarray(forward * np.exp(drift + vol * z), dtype=np.float64)
