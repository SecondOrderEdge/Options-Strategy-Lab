"""BSM pricing & greeks: QuantLib cross-check and finite-difference validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from osl.pricing.bsm import bsm_greeks, bsm_price

QuantLib = pytest.importorskip("QuantLib")

# A grid of (S, K, T, r, q, sigma) covering ITM/ATM/OTM, short/long dated.
GRID = [
    (100.0, 100.0, 0.5, 0.03, 0.01, 0.20),
    (100.0, 90.0, 1.0, 0.05, 0.02, 0.35),
    (100.0, 120.0, 0.25, 0.01, 0.00, 0.15),
    (50.0, 55.0, 2.0, 0.04, 0.03, 0.45),
    (250.0, 240.0, 0.1, 0.02, 0.015, 0.25),
]


def _ql_price(S, K, T, r, q, sigma, right):
    ql = QuantLib
    today = ql.Date(15, 5, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    option_type = ql.Option.Call if right == "c" else ql.Option.Put

    maturity = today + round(T * 365)
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
    process = ql.BlackScholesMertonProcess(spot, q_ts, r_ts, vol_ts)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.NPV()


@pytest.mark.parametrize("params", GRID)
@pytest.mark.parametrize("right", ["c", "p"])
def test_bsm_vs_quantlib(params, right):
    S, K, T, r, q, sigma = params
    ours = float(bsm_price(S, K, T, r, q, sigma, right).item())
    # QuantLib's day-count rounds T to whole days; price with the same rounded T.
    T_days = round(T * 365) / 365.0
    ours_rounded = float(bsm_price(S, K, T_days, r, q, sigma, right).item())
    theirs = _ql_price(S, K, T, r, q, sigma, right)
    assert ours_rounded == pytest.approx(theirs, abs=1e-8)
    assert ours == pytest.approx(theirs, abs=5e-2)  # sanity even without rounding


@pytest.mark.parametrize("params", GRID)
@pytest.mark.parametrize("right", ["c", "p"])
def test_greeks_finite_diff(params, right):
    S, K, T, r, q, sigma = params

    def px(**kw):
        args = {"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma, "right": right}
        args.update(kw)
        return float(bsm_price(**args).item())

    def delta_at(**kw):
        args = {"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma, "right": right}
        args.update(kw)
        return float(bsm_greeks(**args)["delta"].item())

    def vega_raw_at(**kw):
        args = {"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma, "right": right}
        args.update(kw)
        return float(bsm_greeks(**args)["vega"].item()) / 0.01

    g = bsm_greeks(S, K, T, r, q, sigma, right)
    h = 1e-4

    delta_fd = (px(S=S + h) - px(S=S - h)) / (2 * h)
    # gamma = d(delta)/dS via first-order FD of analytic delta (stable; a direct
    # second difference of price suffers catastrophic cancellation at h=1e-4).
    gamma_fd = (delta_at(S=S + h) - delta_at(S=S - h)) / (2 * h)
    vega_fd = (px(sigma=sigma + h) - px(sigma=sigma - h)) / (2 * h) * 0.01
    theta_fd = -(px(T=T + h) - px(T=T - h)) / (2 * h) / 365.0
    rho_fd = (px(r=r + h) - px(r=r - h)) / (2 * h)
    vanna_fd = (delta_at(sigma=sigma + h) - delta_at(sigma=sigma - h)) / (2 * h)
    charm_fd = -(delta_at(T=T + h) - delta_at(T=T - h)) / (2 * h) / 365.0
    vomma_fd = (vega_raw_at(sigma=sigma + h) - vega_raw_at(sigma=sigma - h)) / (2 * h)

    assert g["delta"].item() == pytest.approx(delta_fd, rel=1e-5, abs=1e-7)
    assert g["gamma"].item() == pytest.approx(gamma_fd, rel=1e-5, abs=1e-7)
    assert g["vega"].item() == pytest.approx(vega_fd, rel=1e-5, abs=1e-9)
    assert g["theta"].item() == pytest.approx(theta_fd, rel=1e-5, abs=1e-9)
    assert g["rho"].item() == pytest.approx(rho_fd, rel=1e-5, abs=1e-7)
    assert g["vanna"].item() == pytest.approx(vanna_fd, rel=1e-4, abs=1e-6)
    assert g["charm"].item() == pytest.approx(charm_fd, rel=1e-4, abs=1e-9)
    assert g["vomma"].item() == pytest.approx(vomma_fd, rel=1e-4, abs=1e-6)


def test_vectorized_broadcasting():
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bsm_price(100.0, strikes, 0.5, 0.03, 0.01, 0.2, "c")
    assert prices.shape == (3,)
    # Call price is monotonically decreasing in strike.
    assert prices[0] > prices[1] > prices[2]


def test_put_call_share_gamma_vega():
    args = (100.0, 105.0, 0.5, 0.03, 0.01, 0.25)
    gc = bsm_greeks(*args, "c")
    gp = bsm_greeks(*args, "p")
    assert gc["gamma"].item() == pytest.approx(gp["gamma"].item())
    assert gc["vega"].item() == pytest.approx(gp["vega"].item())


def test_intrinsic_at_expiry():
    assert bsm_price(110.0, 100.0, 0.0, 0.03, 0.0, 0.2, "c").item() == pytest.approx(10.0)
    assert bsm_price(90.0, 100.0, 0.0, 0.03, 0.0, 0.2, "p").item() == pytest.approx(10.0)
    assert math.isclose(bsm_price(90.0, 100.0, 0.0, 0.03, 0.0, 0.2, "c").item(), 0.0)
