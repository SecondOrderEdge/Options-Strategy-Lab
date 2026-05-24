"""Leg/Strategy mechanics and strategy builders."""

from __future__ import annotations

import numpy as np
import pytest

from osl.strategy import strategies as S
from osl.strategy.legs import OptionType, Side, Strategy
from osl.strategy.strategies import ChainContext
from tests.conftest import build_synthetic_chain

SPOT = 100.0
RATE = 0.02


def _ctx(dte_list=(30,), **kw):
    chain, exps = build_synthetic_chain(
        spot=SPOT,
        rate=RATE,
        dte_list=dte_list,
        strikes=[80, 85, 90, 95, 100, 105, 110, 115, 120],
        **kw,
    )
    return ChainContext(chain, "TST", SPOT, RATE), exps


def test_leg_entry_cost_and_signs():
    ctx, exps = _ctx()
    exp = exps[30]
    long_leg = ctx.leg(exp, 100, OptionType.CALL, Side.LONG)
    short_leg = ctx.leg(exp, 100, OptionType.CALL, Side.SHORT)
    assert long_leg.entry_cost > 0  # debit
    assert short_leg.entry_cost < 0  # credit
    assert long_leg.entry_cost == pytest.approx(-short_leg.entry_cost)
    assert long_leg.sign == 1
    assert short_leg.sign == -1


def test_strategy_net_debit_and_pnl():
    ctx, exps = _ctx()
    exp = exps[30]
    lc = S.long_call(ctx, expiration=exp, strike=100)
    assert lc.net_debit > 0
    assert not lc.is_credit
    # Deep ITM at expiry: pnl ~ (S-K)*100 - debit.
    pnl = lc.pnl(np.array([150.0]))[0]
    assert pnl == pytest.approx(50 * 100 - lc.net_debit, abs=1e-6)
    # Far OTM: lose the debit.
    assert lc.pnl(np.array([50.0]))[0] == pytest.approx(-lc.net_debit, abs=1e-6)


def test_credit_spread_is_credit():
    ctx, exps = _ctx(iv=0.3)
    exp = exps[30]
    sps = S.short_put_spread(ctx, expiration=exp, short_strike=95, long_strike=90)
    assert sps.is_credit
    assert sps.net_debit < 0


def test_all_builders_construct():
    ctx, exps = _ctx(dte_list=[30, 60], iv=0.25)
    near, far = exps[30], exps[60]
    built: list[Strategy] = [
        S.long_call(ctx, expiration=near, strike=100),
        S.long_put(ctx, expiration=near, strike=100),
        S.cash_secured_put(ctx, expiration=near, strike=95),
        S.bull_call_spread(ctx, expiration=near, long_strike=100, short_strike=110),
        S.bear_put_spread(ctx, expiration=near, long_strike=100, short_strike=90),
        S.short_put_spread(ctx, expiration=near, short_strike=95, long_strike=90),
        S.short_call_spread(ctx, expiration=near, short_strike=105, long_strike=110),
        S.covered_call(ctx, expiration=near, strike=105),
        S.protective_put(ctx, expiration=near, strike=95),
        S.collar(ctx, expiration=near, put_strike=95, call_strike=105),
        S.long_straddle(ctx, expiration=near, strike=100),
        S.short_straddle(ctx, expiration=near, strike=100),
        S.long_strangle(ctx, expiration=near, put_strike=95, call_strike=105),
        S.short_strangle(ctx, expiration=near, put_strike=95, call_strike=105),
        S.iron_condor(
            ctx,
            expiration=near,
            long_put_strike=85,
            short_put_strike=90,
            short_call_strike=110,
            long_call_strike=115,
        ),
        S.iron_fly(
            ctx, expiration=near, long_put_strike=90, center_strike=100, long_call_strike=110
        ),
        S.broken_wing_butterfly(
            ctx, expiration=near, lower_strike=95, center_strike=105, upper_strike=115
        ),
        S.ratio_spread(ctx, expiration=near, long_strike=100, short_strike=110, ratio=2),
        S.jade_lizard(
            ctx, expiration=near, short_put_strike=90, short_call_strike=105, long_call_strike=110
        ),
        S.calendar(ctx, near_expiration=near, far_expiration=far, strike=100),
        S.diagonal(
            ctx, near_expiration=near, far_expiration=far, short_strike=105, long_strike=100
        ),
        S.pmcc(ctx, near_expiration=near, far_expiration=far, long_strike=85, short_strike=110),
    ]
    assert len(built) == 22
    for strat in built:
        assert len(strat.legs) >= 1
        assert strat.underlying == "TST"


def test_calendar_is_multi_expiry():
    ctx, exps = _ctx(dte_list=[30, 60], iv=0.25)
    cal = S.calendar(ctx, near_expiration=exps[30], far_expiration=exps[60], strike=100)
    assert not cal.all_same_expiry
    assert cal.primary_dte == 30
    # A calendar is long vega (more time value in the far leg).
    from osl.strategy.metrics import aggregate_greeks

    assert aggregate_greeks(cal)["vega"] > 0


def test_ratio_spread_quantities():
    ctx, exps = _ctx(iv=0.25)
    rs = S.ratio_spread(ctx, expiration=exps[30], long_strike=100, short_strike=110, ratio=3)
    short_leg = next(leg for leg in rs.legs if leg.side is Side.SHORT)
    assert short_leg.quantity == 3
