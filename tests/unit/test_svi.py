"""SVI calibration, no-arbitrage checks, and fit quality on a synthetic smile."""

from __future__ import annotations

import numpy as np
import pytest

from osl.surface.svi import (
    SVIParams,
    butterfly_min_g,
    calendar_arbitrage_free,
    fit_svi,
    is_butterfly_arbitrage_free,
)

# A well-behaved, arbitrage-free equity-like smile.
GOOD = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)


def test_total_variance_and_derivatives_consistent():
    k = np.linspace(-0.5, 0.5, 50)
    h = 1e-5
    wp_fd = (GOOD.total_variance(k + h) - GOOD.total_variance(k - h)) / (2 * h)
    wpp_fd = (
        GOOD.total_variance(k + h) - 2 * GOOD.total_variance(k) + GOOD.total_variance(k - h)
    ) / (h * h)
    np.testing.assert_allclose(GOOD.d_total_variance(k), wp_fd, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(GOOD.d2_total_variance(k), wpp_fd, rtol=1e-3, atol=1e-4)


def test_good_params_are_butterfly_free():
    assert is_butterfly_arbitrage_free(GOOD)
    assert butterfly_min_g(GOOD) >= 0.0


def test_butterfly_arbitrage_detected():
    # Large b with extreme rho creates negative density (butterfly arbitrage).
    bad = SVIParams(a=0.001, b=2.5, rho=-0.95, m=0.0, sigma=0.02)
    assert butterfly_min_g(bad) < 0.0
    assert not is_butterfly_arbitrage_free(bad)


def test_random_sweep_classification_is_consistent():
    rng = np.random.default_rng(0)
    grid = np.linspace(-1.0, 1.0, 401)
    for _ in range(50):
        params = SVIParams(
            a=float(rng.uniform(0.0, 0.1)),
            b=float(rng.uniform(0.0, 1.5)),
            rho=float(rng.uniform(-0.95, 0.95)),
            m=float(rng.uniform(-0.2, 0.2)),
            sigma=float(rng.uniform(0.02, 0.5)),
        )
        free = is_butterfly_arbitrage_free(params, grid)
        # If declared free, density (g) must be nonneg on the grid.
        if free:
            assert butterfly_min_g(params, grid) >= -1e-8


def test_svi_fit_quality_recovers_synthetic_smile():
    T = 0.25
    forward = 100.0
    strikes = np.linspace(70, 130, 21)
    k = np.log(strikes / forward)
    w_true = GOOD.total_variance(k)
    iv_true = np.sqrt(w_true / T)
    # Add tiny noise (0.1 vol pt) to emulate quote noise.
    rng = np.random.default_rng(42)
    iv_noisy = iv_true + rng.normal(0.0, 0.001, size=iv_true.size)
    w_noisy = iv_noisy**2 * T

    fit = fit_svi(k, w_noisy, T=T)
    assert fit.success
    assert fit.rmse_vol <= 0.005  # well under the 0.5 vol-pt (=0.005) bar
    assert fit.butterfly_free


def test_calendar_arbitrage_free_monotone():
    near = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
    far = SVIParams(a=0.05, b=0.1, rho=-0.3, m=0.0, sigma=0.1)  # higher total variance
    assert calendar_arbitrage_free([(0.25, near), (1.0, far)])
    # Reversed (far expiry has lower variance) violates calendar arbitrage.
    assert not calendar_arbitrage_free([(0.25, far), (1.0, near)])


def test_fit_requires_five_points():
    with pytest.raises(ValueError, match=">= 5 points"):
        fit_svi([0.0, 0.1], [0.04, 0.04])
