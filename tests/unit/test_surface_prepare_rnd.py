"""Smile preparation from a chain and Breeden-Litzenberger RND correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osl.data.base import coerce_chain
from osl.pricing.bsm import bsm_price
from osl.surface.prepare import prepare_smile, prepare_smiles
from osl.surface.rnd import normalize, risk_neutral_density
from osl.surface.svi import SVIParams, fit_svi, make_iv_of_strike

SPOT = 100.0
RATE = 0.03
DIV = 0.0
DTE = 91
T = DTE / 365.0


def _synthetic_chain(sigma_fn) -> pd.DataFrame:
    """Build a canonical chain priced off a (strike->sigma) function."""
    forward = SPOT * np.exp((RATE - DIV) * T)
    strikes = np.arange(70, 131, 5.0)
    rows = []
    exp = pd.Timestamp("2026-08-21 16:00:00", tz="US/Eastern")
    qt = pd.Timestamp("2026-05-22 20:00:00", tz="UTC")
    for K in strikes:
        sigma = float(sigma_fn(K))
        for right, flag in (("C", "c"), ("P", "p")):
            mid = float(bsm_price(SPOT, K, T, RATE, DIV, sigma, flag).item())
            rows.append(
                {
                    "underlying": "TST",
                    "option_symbol": f"TST{int(K)}{right}",
                    "expiration": exp,
                    "dte": DTE,
                    "right": right,
                    "strike": float(K),
                    "bid": mid - 0.05,
                    "ask": mid + 0.05,
                    "mid": mid,
                    "last": mid,
                    "mark": mid,
                    "bid_size": 10,
                    "ask_size": 10,
                    "volume": 100,
                    "open_interest": 500,
                    "iv": np.nan,  # force recompute path
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
                    "snapshot_id": "snap",
                }
            )
    _ = forward
    return coerce_chain(pd.DataFrame(rows))


def test_prepare_smile_recovers_flat_iv():
    chain = _synthetic_chain(lambda _K: 0.22)
    smile = prepare_smile(chain, pd.Timestamp("2026-08-21").date(), spot=SPOT, rate=RATE)
    assert len(smile) >= 5
    np.testing.assert_allclose(smile.iv, 0.22, atol=1e-4)
    # Total variance consistent with iv^2 * T.
    np.testing.assert_allclose(smile.total_variance, smile.iv**2 * smile.T, rtol=1e-10)


def test_prepare_smile_recovers_skew():
    # Linear skew in strike: lower strikes have higher IV.
    def skew(K: float) -> float:
        return 0.30 - 0.0010 * (K - SPOT)

    chain = _synthetic_chain(skew)
    smile = prepare_smile(chain, pd.Timestamp("2026-08-21").date(), spot=SPOT, rate=RATE)
    expected = np.array([skew(k) for k in smile.strikes])
    np.testing.assert_allclose(smile.iv, expected, atol=2e-3)


def test_prepare_smiles_filters_min_points():
    chain = _synthetic_chain(lambda _K: 0.2)
    smiles = prepare_smiles(chain, spot=SPOT, rate=RATE, min_points=5)
    assert len(smiles) == 1
    smiles_strict = prepare_smiles(chain, spot=SPOT, rate=RATE, min_points=100)
    assert smiles_strict == []


def test_svi_fit_on_prepared_smile():
    def skew(K: float) -> float:
        return 0.30 - 0.0010 * (K - SPOT)

    chain = _synthetic_chain(skew)
    smile = prepare_smile(chain, pd.Timestamp("2026-08-21").date(), spot=SPOT, rate=RATE)
    fit = fit_svi(smile.k, smile.total_variance, weights=smile.weights, T=smile.T)
    assert fit.success
    assert fit.rmse_vol < 0.01


def test_rnd_flat_iv_integrates_to_one_and_mean_is_forward():
    sigma = 0.25
    iv_fn = lambda K: np.full_like(np.asarray(K, dtype=float), sigma)  # noqa: E731
    strikes, density = risk_neutral_density(
        iv_fn, spot=SPOT, rate=RATE, dividend_yield=DIV, T=T, k_min=-2.0, k_max=2.0, n=2000
    )
    area = np.trapz(density, strikes)
    assert area == pytest.approx(1.0, abs=0.02)

    dens = normalize(strikes, density)
    mean = np.trapz(strikes * dens, strikes)
    forward = SPOT * np.exp((RATE - DIV) * T)
    assert mean == pytest.approx(forward, rel=0.01)
    assert (density >= 0).all()


def test_rnd_from_svi_params():
    params = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)
    forward = SPOT * np.exp((RATE - DIV) * T)
    iv_fn = make_iv_of_strike(params, forward=forward, T=T)
    strikes, density = risk_neutral_density(
        iv_fn, spot=SPOT, rate=RATE, T=T, k_min=-1.5, k_max=1.5, n=1500
    )
    dens = normalize(strikes, density)
    area = np.trapz(dens, strikes)
    assert area == pytest.approx(1.0, abs=1e-6)
    assert (density >= 0).all()
