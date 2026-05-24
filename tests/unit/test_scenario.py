"""Scenario analysis: entry P&L, time monotonicity, and vega-shock isolation."""

from __future__ import annotations

import numpy as np
import pytest

from osl.scenario.payoff import payoff_curve, pnl_at, value_at
from osl.scenario.pnl_surface import pnl_grid
from osl.scenario.stress import stress_scenarios
from osl.strategy import strategies as S
from osl.strategy.metrics import aggregate_greeks, reference_sigma
from osl.strategy.strategies import ChainContext
from tests.conftest import build_synthetic_chain

SPOT = 100.0
RATE = 0.02
STRIKES = [80, 85, 90, 95, 100, 105, 110, 115, 120]


def _ctx(**kw):
    chain, exps = build_synthetic_chain(spot=SPOT, rate=RATE, strikes=STRIKES, **kw)
    return ChainContext(chain, "TST", SPOT, RATE), exps


def test_pnl_at_t0_zero_for_credit_at_mid():
    ctx, exps = _ctx(dte_list=[30], iv=0.3)
    spread = S.short_put_spread(ctx, expiration=exps[30], short_strike=95, long_strike=90)
    assert spread.is_credit
    pnl0 = float(pnl_at(spread, np.array([SPOT]), days_forward=0.0, vol_shock=0.0)[0])
    assert pnl0 == pytest.approx(0.0, abs=1e-6)


def test_pnl_at_t0_zero_for_debit_at_mid():
    ctx, exps = _ctx(dte_list=[30], iv=0.25)
    bcs = S.bull_call_spread(ctx, expiration=exps[30], long_strike=100, short_strike=110)
    pnl0 = float(pnl_at(bcs, np.array([SPOT]))[0])
    assert pnl0 == pytest.approx(0.0, abs=1e-6)


def test_pnl_surface_monotone_in_time_for_positive_theta():
    ctx, exps = _ctx(dte_list=[45], iv=0.3)
    # Iron condor: positive theta, neutral -> P&L rises with time at the ATM spot.
    ic = S.iron_condor(
        ctx,
        expiration=exps[45],
        long_put_strike=85,
        short_put_strike=95,
        short_call_strike=105,
        long_call_strike=115,
    )
    assert aggregate_greeks(ic)["theta"] > 0
    early = float(pnl_at(ic, np.array([SPOT]), days_forward=0.0)[0])
    later = float(pnl_at(ic, np.array([SPOT]), days_forward=20.0)[0])
    assert later > early


def test_stress_vega_shock_isolation():
    # A pure IV shock at fixed spot/time changes P&L by ~ aggregate vega * dvol.
    ctx, exps = _ctx(dte_list=[30], iv=0.25)
    straddle = S.long_straddle(ctx, expiration=exps[30], strike=100)
    vega = aggregate_greeks(straddle)["vega"]  # per 1 vol point (0.01)
    dvol = 0.01  # one vol point
    base = float(pnl_at(straddle, np.array([SPOT]), vol_shock=0.0)[0])
    shocked = float(pnl_at(straddle, np.array([SPOT]), vol_shock=dvol)[0])
    assert (shocked - base) == pytest.approx(vega, rel=0.02)


def test_payoff_curve_matches_terminal_pnl():
    ctx, exps = _ctx(dte_list=[30], iv=0.25)
    bcs = S.bull_call_spread(ctx, expiration=exps[30], long_strike=100, short_strike=110)
    spots, pnl = payoff_curve(bcs, n=50)
    # Terminal payoff curve equals the intrinsic-based Strategy.pnl.
    np.testing.assert_allclose(pnl, bcs.pnl(spots), atol=1e-6)


def test_value_at_now_equals_net_debit_for_fair_chain():
    ctx, exps = _ctx(dte_list=[30], iv=0.25)
    ic = S.iron_condor(
        ctx,
        expiration=exps[30],
        long_put_strike=85,
        short_put_strike=95,
        short_call_strike=105,
        long_call_strike=115,
    )
    v = float(value_at(ic, np.array([SPOT]))[0])
    assert v == pytest.approx(ic.net_debit, abs=1e-6)


def test_pnl_grid_shape_and_axes():
    ctx, exps = _ctx(dte_list=[30], iv=0.25)
    lc = S.long_call(ctx, expiration=exps[30], strike=100)
    spots, days, z = pnl_grid(lc, spot_shocks=np.linspace(-0.2, 0.2, 11), days_forward=[0, 15, 30])
    assert z.shape == (3, 11)
    assert spots[5] == pytest.approx(SPOT)  # middle shock is 0%
    # At expiry (day 30) deep ITM, long call P&L is large and positive.
    assert z[2, -1] > 0


def test_stress_scenarios_cover_sigma_moves():
    ctx, exps = _ctx(dte_list=[30], iv=0.3)
    spread = S.short_put_spread(ctx, expiration=exps[30], short_strike=95, long_strike=90)
    sigma = reference_sigma(spread)
    results = stress_scenarios(spread, sigma=sigma, days_forward=30.0)
    labels = {r.label for r in results}
    assert {"-2 sigma", "-1 sigma", "flat", "+1 sigma", "+2 sigma"} <= labels
    flat = next(r for r in results if r.label == "flat")
    down2 = next(r for r in results if r.label == "-2 sigma")
    # A short put spread loses on a large down move, wins when flat (credit kept).
    assert flat.pnl > down2.pnl
