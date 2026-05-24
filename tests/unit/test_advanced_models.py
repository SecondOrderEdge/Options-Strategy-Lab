"""Advanced models: Heston/BSM limit + calibration, Merton jumps, Dupire, PCA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osl.models.heston import HestonParams, calibrate_heston, heston_price
from osl.models.jump_diffusion import MertonParams, merton_price
from osl.models.local_vol import dupire_local_vol
from osl.pricing.bsm import bsm_price
from osl.surface.pca import surface_pca

QuantLib = pytest.importorskip("QuantLib")

SPOT, RATE, DIV = 100.0, 0.03, 0.0


# --- Heston ---
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("right", ["c", "p"])
def test_heston_recovers_bsm(strike, right):
    sigma = 0.2
    T = 0.5
    params = HestonParams(v0=sigma**2, kappa=1.0, theta=sigma**2, sigma=1e-5, rho=0.0)
    h = heston_price(SPOT, strike, T, RATE, DIV, right, params)
    # Heston prices at an integer-day maturity; compare BSM at the same T.
    t_eff = round(T * 365) / 365
    b = float(bsm_price(SPOT, strike, t_eff, RATE, DIV, sigma, right).item())
    assert h == pytest.approx(b, abs=1e-3)


def test_heston_calibration_recovers_known_surface():
    # Generate market vols from a known Heston model, then calibrate back.
    true = HestonParams(v0=0.04, kappa=1.5, theta=0.05, sigma=0.4, rho=-0.6)
    quotes = []
    for T in (0.25, 0.5, 1.0):
        for k in (90.0, 100.0, 110.0):
            price = heston_price(SPOT, k, T, RATE, DIV, "c", true)
            # implied vol of that Heston price
            from osl.pricing.iv import implied_vol

            iv = implied_vol(price, SPOT, k, T, RATE, DIV, "c", method="bisect")
            quotes.append((T, k, iv))
    cal = calibrate_heston(quotes, spot=SPOT, rate=RATE, dividend_yield=DIV)
    assert cal.rmse_vol < 0.01  # fits the surface to under 1 vol point


# --- Merton jumps ---
def test_merton_reduces_to_bsm_without_jumps():
    params = MertonParams(jump_intensity=0.0, jump_mean=0.0, jump_std=0.0)
    m = merton_price(SPOT, 100.0, 0.5, RATE, DIV, 0.2, "c", params)
    b = float(bsm_price(SPOT, 100.0, 0.5, RATE, DIV, 0.2, "c").item())
    assert m == pytest.approx(b, abs=1e-10)


def test_merton_jumps_fatten_tails():
    # Negative-mean jumps raise OTM put value vs pure diffusion.
    params = MertonParams(jump_intensity=1.0, jump_mean=-0.1, jump_std=0.15)
    otm_put = merton_price(SPOT, 80.0, 0.5, RATE, DIV, 0.2, "p", params)
    bsm_put = float(bsm_price(SPOT, 80.0, 0.5, RATE, DIV, 0.2, "p").item())
    assert otm_put > bsm_put


# --- Dupire local vol ---
def test_dupire_flat_surface_recovers_flat_vol():
    expiries = [0.25, 0.5, 1.0]
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    sigma = 0.2
    iv_grid = np.full((len(strikes), len(expiries)), sigma)
    lv = dupire_local_vol(expiries, strikes, iv_grid, spot=SPOT, rate=RATE, dividend_yield=DIV)
    finite = lv[np.isfinite(lv)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, sigma, atol=0.02)


# --- PCA ---
def test_pca_three_factors_explain_most_variance():
    # Synthetic surface history driven by 3 latent factors + small noise.
    rng = np.random.default_rng(0)
    n_days, n_points = 80, 12
    loadings = rng.normal(size=(3, n_points))
    factors = rng.normal(size=(n_days, 3)) * np.array([1.0, 0.5, 0.25])
    surface = factors @ loadings + rng.normal(0, 0.002, size=(n_days, n_points))
    panel = pd.DataFrame(surface, columns=[f"p{i}" for i in range(n_points)])

    pca = surface_pca(panel, n_factors=3, on="changes")
    assert pca.explained_variance_ratio[:3].sum() >= 0.90
    assert pca.factor_names == ("level", "slope", "curvature")
    assert pca.eigenvectors.shape == (n_points, 3)


def test_pca_levels_basis_runs():
    rng = np.random.default_rng(1)
    panel = pd.DataFrame(rng.normal(0.2, 0.02, size=(60, 8)))
    pca = surface_pca(panel, n_factors=3, on="levels")
    assert abs(pca.explained_variance_ratio.sum() - 1.0) < 1e-6
