"""Strategy metrics: breakevens, unbounded risk, POP agreement, EV vs POP."""

from __future__ import annotations

import math
from datetime import date

import pytest

from osl.pricing.bsm import bsm_price
from osl.strategy import strategies as S
from osl.strategy.legs import Leg, OptionType, Side, Strategy
from osl.strategy.metrics import (
    aggregate_greeks,
    compute_metrics,
    ev_closed_form,
    payoff_profile,
    pop_monte_carlo,
)
from osl.strategy.strategies import ChainContext
from tests.conftest import build_synthetic_chain

SPOT = 100.0
RATE = 0.02


def _ctx(**kw):
    chain, exps = build_synthetic_chain(
        spot=SPOT,
        rate=RATE,
        strikes=[80, 85, 90, 95, 100, 105, 110, 115, 120],
        **kw,
    )
    return ChainContext(chain, "TST", SPOT, RATE), exps


def test_iron_condor_breakevens_closed_form():
    ctx, exps = _ctx(dte_list=[30], iv=0.25)
    exp = exps[30]
    ic = S.iron_condor(
        ctx,
        expiration=exp,
        long_put_strike=90,
        short_put_strike=95,
        short_call_strike=105,
        long_call_strike=110,
    )
    profile = payoff_profile(ic)
    credit_per_share = -ic.net_debit / 100.0
    assert credit_per_share > 0  # it's a credit structure
    expected_low = 95 - credit_per_share
    expected_high = 105 + credit_per_share
    assert len(profile.breakevens) == 2
    assert profile.breakevens[0] == pytest.approx(expected_low, abs=1e-3)
    assert profile.breakevens[1] == pytest.approx(expected_high, abs=1e-3)
    # Bounded structure.
    assert not profile.loss_unbounded
    assert math.isfinite(profile.max_loss)


def test_bull_call_spread_max_profit_loss():
    ctx, exps = _ctx(dte_list=[30], iv=0.22)
    exp = exps[30]
    bcs = S.bull_call_spread(ctx, expiration=exp, long_strike=100, short_strike=110)
    profile = payoff_profile(bcs)
    debit = bcs.net_debit  # positive
    assert debit > 0
    width = (110 - 100) * 100
    assert profile.max_profit == pytest.approx(width - debit, abs=1e-2)
    assert profile.max_loss == pytest.approx(-debit, abs=1e-2)
    assert profile.breakevens[0] == pytest.approx(100 + debit / 100, abs=1e-3)


def test_short_strangle_uncovered_loss_unbounded():
    ctx, exps = _ctx(dte_list=[30], iv=0.3)
    exp = exps[30]
    strangle = S.short_strangle(ctx, expiration=exp, put_strike=90, call_strike=110)
    profile = payoff_profile(strangle)
    assert profile.loss_unbounded is True
    assert profile.max_loss == -math.inf
    # The engine must not present a finite "max loss" for this.


def test_pop_methods_agree_on_long_call():
    ctx, exps = _ctx(dte_list=[14], iv=0.18)
    exp = exps[14]
    lc = S.long_call(ctx, expiration=exp, strike=100)
    m = compute_metrics(lc, n_mc=50_000, seed=7)
    assert m.pop_rn == pytest.approx(m.pop_mc, abs=0.02)
    assert m.pop_rn == pytest.approx(m.pop_delta, abs=0.02)
    # All three are sensible probabilities.
    for p in (m.pop_rn, m.pop_mc, m.pop_delta):
        assert 0.0 < p < 1.0


def test_high_pop_can_have_negative_ev():
    # Deep-OTM short put spread sold BELOW fair value -> POP high, EV negative.
    dte = 30
    T = dte / 365.0
    exp = date(2026, 7, 1)
    short_k, long_k, sigma = 80.0, 75.0, 0.5
    fair_short = float(bsm_price(SPOT, short_k, T, RATE, 0.0, sigma, "p").item())
    fair_long = float(bsm_price(SPOT, long_k, T, RATE, 0.0, sigma, "p").item())
    legs = (
        Leg(OptionType.PUT, Side.SHORT, short_k, exp, dte, premium=fair_short * 0.5, iv=sigma),
        Leg(OptionType.PUT, Side.LONG, long_k, exp, dte, premium=fair_long, iv=sigma),
    )
    spread = Strategy("Short Put Spread", "TST", legs, SPOT, RATE)
    m = compute_metrics(spread, n_mc=50_000, seed=11)
    assert m.pop_rn > 0.90
    assert ev_closed_form(spread) < 0
    assert m.ev_mc.value < 0
    # Tail loss companion is surfaced and positive.
    assert m.expected_shortfall > 0


def test_greeks_aggregation_signs():
    ctx, exps = _ctx(dte_list=[30], iv=0.25)
    exp = exps[30]
    lc = S.long_call(ctx, expiration=exp, strike=100)
    g = aggregate_greeks(lc)
    assert g["delta"] > 0  # long call positive delta
    assert g["gamma"] > 0
    assert g["vega"] > 0
    assert g["theta"] < 0  # long option bleeds theta

    sc = S.cash_secured_put(ctx, expiration=exp, strike=95)
    gp = aggregate_greeks(sc)
    assert gp["delta"] > 0  # short put positive delta
    assert gp["theta"] > 0  # short option collects theta


def test_covered_call_has_stock_and_capped_upside():
    ctx, exps = _ctx(dte_list=[30], iv=0.25)
    exp = exps[30]
    cc = S.covered_call(ctx, expiration=exp, strike=105)
    assert cc.stock_quantity == 100
    profile = payoff_profile(cc)
    assert not profile.profit_unbounded  # short call caps the upside
    g = aggregate_greeks(cc)
    # Delta = +100 shares minus the short call's delta exposure -> in (0, 100).
    assert 0 < g["delta"] < 100


def test_pop_mc_matches_closed_form_iron_condor():
    ctx, exps = _ctx(dte_list=[30], iv=0.25)
    exp = exps[30]
    ic = S.iron_condor(
        ctx,
        expiration=exp,
        long_put_strike=85,
        short_put_strike=90,
        short_call_strike=110,
        long_call_strike=115,
    )
    profile = payoff_profile(ic)
    sigma = 0.25
    from osl.strategy.metrics import _pop_lognormal

    pop_cf = _pop_lognormal(ic, profile, sigma, delta_proxy=False)
    pop_mc = pop_monte_carlo(ic, sigma, n=80_000, seed=3)
    assert pop_cf == pytest.approx(pop_mc, abs=0.01)
    assert 0.0 < pop_cf < 1.0
