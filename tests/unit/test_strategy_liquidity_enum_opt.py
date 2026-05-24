"""Liquidity scoring, candidate enumeration, and optimizer ranking."""

from __future__ import annotations

import pytest

from osl.strategy import strategies as S
from osl.strategy.enumerate import enumerate_candidates
from osl.strategy.liquidity import strategy_liquidity
from osl.strategy.metrics import compute_metrics
from osl.strategy.optimizer import OBJECTIVES, Candidate, rank, score
from osl.strategy.strategies import ChainContext
from tests.conftest import build_synthetic_chain

SPOT = 100.0
RATE = 0.02
STRIKES = [80, 85, 90, 95, 100, 105, 110, 115, 120]


def _ctx(**kw):
    chain, exps = build_synthetic_chain(spot=SPOT, rate=RATE, strikes=STRIKES, **kw)
    return ChainContext(chain, "TST", SPOT, RATE), exps, chain


# --- liquidity ---
def test_liquidity_score_bounds_and_slippage():
    ctx, exps, chain = _ctx(dte_list=[30], iv=0.25, spread=0.10)
    bcs = S.bull_call_spread(ctx, expiration=exps[30], long_strike=100, short_strike=110)
    rep = strategy_liquidity(chain, bcs)
    assert 0.0 <= rep.score <= 1.0
    assert not rep.has_zero_bid
    # Slippage = half-spread on both legs.
    assert rep.slippage == pytest.approx(0.5 * 0.10 * 100 * 2, abs=1e-6)


def test_liquidity_penalizes_zero_bid():
    ctx, exps, chain = _ctx(dte_list=[30], iv=0.25, zero_bid=[(30, 110.0, "C")])
    bcs = S.bull_call_spread(ctx, expiration=exps[30], long_strike=100, short_strike=110)
    rep = strategy_liquidity(chain, bcs)
    assert rep.has_zero_bid is True
    assert rep.score < 0.5


# --- enumerate ---
def test_enumerate_excludes_zero_bid_by_default():
    # Force the 110 call (a natural short-call-spread/IC leg) to zero bid.
    ctx, exps, chain = _ctx(dte_list=[30], iv=0.25, zero_bid=[(30, 110.0, "C")])
    candidates = enumerate_candidates(ctx, view="all", dte_range=(20, 40), max_strikes_per_leg=2)
    assert candidates  # some candidates generated
    # No leg uses the zero-bid 110 call.
    for strat in candidates:
        for leg in strat.legs:
            assert not (leg.strike == 110.0 and leg.right == "C")
    # When explicitly allowed, the 110 call can appear.
    allowed = enumerate_candidates(
        ctx, view="all", dte_range=(20, 40), max_strikes_per_leg=2, allow_zero_bid=True
    )
    used_110c = any(leg.strike == 110.0 and leg.right == "C" for s in allowed for leg in s.legs)
    assert used_110c


def test_enumerate_views_differ():
    ctx, exps, chain = _ctx(dte_list=[30], iv=0.25)
    bullish = enumerate_candidates(ctx, view="bullish", dte_range=(20, 40))
    neutral = enumerate_candidates(ctx, view="neutral", dte_range=(20, 40))
    assert {s.name for s in bullish} != {s.name for s in neutral}
    assert all(s.name in {"Long Call", "Bull Call Spread", "Short Put Spread"} for s in bullish)


# --- optimizer ---
def _candidate(strategy, chain) -> Candidate:
    return Candidate(
        strategy=strategy,
        metrics=compute_metrics(strategy, n_mc=8000, seed=1),
        liquidity=strategy_liquidity(chain, strategy),
    )


def test_rank_is_deterministic_and_objective_dependent():
    ctx, exps, chain = _ctx(dte_list=[30], iv=0.3)
    exp = exps[30]
    candidates = [
        _candidate(S.long_straddle(ctx, expiration=exp, strike=100), chain),
        _candidate(
            S.iron_condor(
                ctx,
                expiration=exp,
                long_put_strike=85,
                short_put_strike=90,
                short_call_strike=110,
                long_call_strike=115,
            ),
            chain,
        ),
        _candidate(S.short_strangle(ctx, expiration=exp, put_strike=90, call_strike=110), chain),
        _candidate(
            S.bull_call_spread(ctx, expiration=exp, long_strike=100, short_strike=110), chain
        ),
    ]
    by_pop = [c.strategy.name for c in rank(candidates, "pop_capped")]
    by_ev = [c.strategy.name for c in rank(candidates, "ev")]
    # Deterministic: re-ranking gives the same order.
    assert by_pop == [c.strategy.name for c in rank(candidates, "pop_capped")]
    # Objective changes the ordering.
    assert by_pop != by_ev
    # pop_capped pushes the unbounded-risk short strangle to the bottom.
    assert by_pop[-1] == "Short Strangle"


def test_unknown_objective_raises():
    ctx, exps, chain = _ctx(dte_list=[30], iv=0.25)
    cand = _candidate(S.long_call(ctx, expiration=exps[30], strike=100), chain)
    with pytest.raises(ValueError, match="unknown objective"):
        score(cand, "sharpe")


def test_every_objective_scores_finite():
    ctx, exps, chain = _ctx(dte_list=[30], iv=0.25)
    exp = exps[30]
    cands = [
        _candidate(
            S.iron_condor(
                ctx,
                expiration=exp,
                long_put_strike=85,
                short_put_strike=90,
                short_call_strike=110,
                long_call_strike=115,
            ),
            chain,
        ),
        _candidate(S.long_straddle(ctx, expiration=exp, strike=100), chain),
        _candidate(S.short_strangle(ctx, expiration=exp, put_strike=90, call_strike=110), chain),
    ]
    for objective in OBJECTIVES:
        for cand in cands:
            value = score(cand, objective)
            assert isinstance(value, float)
        # Ranking by each objective returns all candidates, best-first.
        assert len(rank(cands, objective)) == len(cands)
