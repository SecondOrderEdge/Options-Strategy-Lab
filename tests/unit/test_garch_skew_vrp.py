"""GARCH forecasting, 25Δ skew, VRP, and P-measure / blended POP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osl.strategy import strategies as S
from osl.strategy.metrics import blended_pop, compute_metrics, pop_garch
from osl.strategy.strategies import ChainContext
from osl.surface.prepare import prepare_smile
from osl.volatility.garch import fit_garch
from osl.volatility.skew import delta_skew_25
from osl.volatility.vrp import variance_risk_premium, vol_point_premium
from tests.conftest import build_synthetic_chain

SPOT = 100.0
RATE = 0.02
STRIKES = [80, 85, 90, 95, 100, 105, 110, 115, 120]


def _garch_returns(n: int = 1500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    sigma = np.empty(n)
    eps = np.empty(n)
    sigma[0] = 0.01
    omega, alpha, beta = 1e-6, 0.08, 0.90  # persistence 0.98 < 1
    for t in range(1, n):
        sigma[t] = np.sqrt(omega + alpha * eps[t - 1] ** 2 + beta * sigma[t - 1] ** 2)
        eps[t] = sigma[t] * rng.standard_normal()
    return pd.Series(eps)


def test_garch_persistence_below_one():
    res = fit_garch(_garch_returns(), model="garch", dist="normal")
    assert res.persistence < 1.0
    assert res.annualized_vol > 0
    assert res.conditional_vol > 0


def test_gjr_t_fits_and_is_stationary():
    res = fit_garch(_garch_returns(seed=1), model="gjr", dist="t")
    assert res.model == "gjr"
    assert res.persistence < 1.0
    assert "gamma[1]" in res.params


def test_delta_skew_detects_put_skew():
    # Lower strikes carry higher IV -> positive 25Δ risk reversal.
    def skew(strike: float, _dte: int) -> float:
        return 0.30 - 0.0015 * (strike - SPOT)

    chain, exps = build_synthetic_chain(spot=SPOT, rate=RATE, strikes=STRIKES, iv_fn=skew)
    smile = prepare_smile(chain, exps[30], spot=SPOT, rate=RATE)
    ds = delta_skew_25(smile)
    assert ds.risk_reversal_25 > 0  # put skew
    assert ds.atm_iv == pytest.approx(0.30, abs=0.01)


def test_delta_skew_flat_smile_zero_rr():
    chain, exps = build_synthetic_chain(spot=SPOT, rate=RATE, strikes=STRIKES, iv=0.25)
    smile = prepare_smile(chain, exps[30], spot=SPOT, rate=RATE)
    ds = delta_skew_25(smile)
    assert ds.risk_reversal_25 == pytest.approx(0.0, abs=1e-6)
    assert ds.butterfly_25 == pytest.approx(0.0, abs=1e-6)


def test_vrp_signs():
    assert variance_risk_premium(0.25, 0.18) == pytest.approx(0.25**2 - 0.18**2)
    assert vol_point_premium(0.25, 0.18) == pytest.approx(0.07)
    assert variance_risk_premium(0.10, 0.20) < 0  # implied below realized


def test_pop_garch_and_blended():
    chain, exps = build_synthetic_chain(spot=SPOT, rate=RATE, strikes=STRIKES, iv=0.25)
    ctx = ChainContext(chain, "TST", SPOT, RATE)
    lc = S.long_call(ctx, expiration=exps[30], strike=100)
    m = compute_metrics(lc, n_mc=20_000)

    p_pop = pop_garch(lc, 0.25, n=20_000, seed=3)
    assert 0.0 < p_pop < 1.0
    blended = blended_pop(m.pop_rn, p_pop, rn_weight=0.5)
    assert min(m.pop_rn, p_pop) - 1e-9 <= blended <= max(m.pop_rn, p_pop) + 1e-9
    # Weight extremes recover the components.
    assert blended_pop(m.pop_rn, p_pop, rn_weight=1.0) == pytest.approx(m.pop_rn)
    assert blended_pop(m.pop_rn, p_pop, rn_weight=0.0) == pytest.approx(p_pop)
