"""Merton (1976) jump-diffusion European pricing.

A Poisson series of Black-Scholes prices with jump-adjusted drift and variance.
As the jump intensity ``lambda`` -> 0 the price collapses to Black-Scholes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from osl.pricing.bsm import bsm_price


@dataclass(frozen=True)
class MertonParams:
    jump_intensity: float  # lambda: expected jumps per year
    jump_mean: float  # mu_J: mean of log jump size
    jump_std: float  # delta: std of log jump size


def merton_price(
    spot: float,
    strike: float,
    T: float,
    rate: float,
    dividend_yield: float,
    sigma: float,
    right: str,
    params: MertonParams,
    *,
    n_terms: int = 60,
) -> float:
    """Merton jump-diffusion European option price (closed-form Poisson series)."""
    lam, mu_j, delta = params.jump_intensity, params.jump_mean, params.jump_std
    if lam <= 0:
        return float(bsm_price(spot, strike, T, rate, dividend_yield, sigma, right).item())

    k = math.exp(mu_j + 0.5 * delta * delta) - 1.0  # expected relative jump
    lam_prime = lam * (1.0 + k)
    total = 0.0
    for n in range(n_terms):
        weight = math.exp(-lam_prime * T) * (lam_prime * T) ** n / math.factorial(n)
        sigma_n = math.sqrt(sigma * sigma + n * delta * delta / T)
        rate_n = rate - lam * k + n * math.log(1.0 + k) / T
        price_n = float(bsm_price(spot, strike, T, rate_n, dividend_yield, sigma_n, right).item())
        total += weight * price_n
        if weight < 1e-12 and n > lam_prime * T + 5:
            break
    return total
