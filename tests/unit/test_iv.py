"""Implied volatility: Jaeckel vs bisection agreement and round-trip accuracy."""

from __future__ import annotations

import pytest

from osl.pricing.bsm import bsm_price
from osl.pricing.iv import implied_vol

CASES = [
    (100.0, 100.0, 0.5, 0.03, 0.01, 0.20, "c"),
    (100.0, 90.0, 1.0, 0.05, 0.02, 0.35, "p"),
    (100.0, 120.0, 0.25, 0.01, 0.00, 0.15, "c"),
    (250.0, 240.0, 0.1, 0.02, 0.015, 0.40, "p"),
]


@pytest.mark.parametrize("case", CASES)
def test_roundtrip_jaeckel(case):
    S, K, T, r, q, sigma, right = case
    price = float(bsm_price(S, K, T, r, q, sigma, right).item())
    recovered = implied_vol(price, S, K, T, r, q, right, method="jaeckel")
    assert recovered == pytest.approx(sigma, abs=1e-8)


@pytest.mark.parametrize("case", CASES)
def test_jaeckel_vs_bisect(case):
    S, K, T, r, q, sigma, right = case
    price = float(bsm_price(S, K, T, r, q, sigma, right).item())
    j = implied_vol(price, S, K, T, r, q, right, method="jaeckel")
    b = implied_vol(price, S, K, T, r, q, right, method="bisect")
    assert j == pytest.approx(b, abs=1e-8)


def test_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown IV method"):
        implied_vol(1.0, 100.0, 100.0, 0.5, 0.03, 0.0, "c", method="newton")  # type: ignore[arg-type]
