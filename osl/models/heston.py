"""Heston stochastic-volatility pricing and calibration via QuantLib.

Heston dynamics: ``dS = (r-q)S dt + sqrt(v) S dW1``,
``dv = kappa(theta - v) dt + sigma sqrt(v) dW2``, ``dW1 dW2 = rho dt``.

As the vol-of-vol ``sigma`` -> 0 and ``rho`` -> 0 (with ``v0 = theta = sigma_BS^2``)
Heston collapses to Black-Scholes, which the tests verify.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from osl.utils.time import DAYS_PER_YEAR

_EVAL_YEAR = 2026
_EVAL_MONTH = 5
_EVAL_DAY = 15

# Calibration parameter bounds, in QuantLib's HestonModel parameter order
# (theta, kappa, sigma, rho, v0). Bounding theta and v0 away from zero is what
# stops the optimiser collapsing to the degenerate theta=0 corner.
_BOUNDS_LO = (1e-4, 1e-2, 1e-3, -0.999, 1e-4)
_BOUNDS_HI = (1.0, 50.0, 5.0, 0.999, 1.0)


@dataclass(frozen=True)
class HestonParams:
    v0: float  # initial variance
    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    sigma: float  # vol of vol
    rho: float  # spot/vol correlation


@dataclass(frozen=True)
class HestonCalibration:
    params: HestonParams
    rmse_vol: float


def _ql_context(spot: float, rate: float, dividend_yield: float):  # type: ignore[no-untyped-def]
    import QuantLib as ql

    today = ql.Date(_EVAL_DAY, _EVAL_MONTH, _EVAL_YEAR)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, dividend_yield, dc))
    return ql, today, dc, spot_h, r_ts, q_ts


def heston_price(
    spot: float,
    strike: float,
    T: float,
    rate: float,
    dividend_yield: float,
    right: str,
    params: HestonParams,
) -> float:
    """Price a European option under Heston via QuantLib's analytic engine."""
    ql, today, _dc, spot_h, r_ts, q_ts = _ql_context(spot, rate, dividend_yield)
    process = ql.HestonProcess(
        r_ts, q_ts, spot_h, params.v0, params.kappa, params.theta, params.sigma, params.rho
    )
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
    option_type = ql.Option.Call if str(right).lower().startswith("c") else ql.Option.Put
    maturity = today + round(T * DAYS_PER_YEAR)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(option_type, strike), ql.EuropeanExercise(maturity)
    )
    option.setPricingEngine(engine)
    return float(option.NPV())


def _seed_guesses(
    quotes: Sequence[tuple[float, float, float]],
    spot: float,
    rate: float,
    dividend_yield: float,
) -> list[HestonParams]:
    """Data-driven Heston starting points for a multi-start calibration.

    Levenberg-Marquardt is local and Heston's objective is riddled with poor
    local minima, so the starting point matters more than the optimiser. Seed
    ``v0`` from the at-the-money variance and ``theta`` from the median variance,
    then fan out over a few (kappa, sigma, rho) regimes and keep the best fit.
    """
    atm_var = quotes[0][2] ** 2
    best_moneyness = float("inf")
    variances: list[float] = []
    for t_years, strike, vol in quotes:
        forward = spot * math.exp((rate - dividend_yield) * t_years)
        moneyness = abs(math.log(strike / forward))
        variances.append(vol**2)
        if moneyness < best_moneyness:
            best_moneyness, atm_var = moneyness, vol**2
    theta0 = statistics.median(variances)
    return [
        HestonParams(v0=atm_var, kappa=2.0, theta=theta0, sigma=0.5, rho=-0.5),
        HestonParams(v0=atm_var, kappa=5.0, theta=theta0, sigma=1.0, rho=-0.7),
        HestonParams(v0=atm_var, kappa=1.0, theta=theta0, sigma=0.3, rho=-0.3),
    ]


def _calibrate_once(  # type: ignore[no-untyped-def]
    ql,
    spot_h,
    r_ts,
    q_ts,
    quotes: Sequence[tuple[float, float, float]],
    spot: float,
    guess: HestonParams,
) -> tuple[HestonParams, float]:
    """Run one bounded Levenberg-Marquardt fit from ``guess``; return (params, vol RMSE)."""
    process = ql.HestonProcess(
        r_ts, q_ts, spot_h, guess.v0, guess.kappa, guess.theta, guess.sigma, guess.rho
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    helpers = []
    for t_years, strike, vol in quotes:
        period = ql.Period(round(t_years * DAYS_PER_YEAR), ql.Days)
        helper = ql.HestonModelHelper(
            period,
            ql.NullCalendar(),
            spot,
            strike,
            ql.QuoteHandle(ql.SimpleQuote(vol)),
            r_ts,
            q_ts,
            # Minimise implied-vol error (not price error) so the wings — where
            # prices are tiny but vols large — are not effectively ignored.
            ql.BlackCalibrationHelper.ImpliedVolError,
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)

    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    constraint = ql.NonhomogeneousBoundaryConstraint(ql.Array(_BOUNDS_LO), ql.Array(_BOUNDS_HI))
    model.calibrate(helpers, lm, ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8), constraint)

    theta, kappa, sigma, rho, v0 = model.params()
    params = HestonParams(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)

    sq = 0.0
    for helper, (_t, _k, vol) in zip(helpers, quotes, strict=True):
        try:
            iv = helper.impliedVolatility(helper.modelValue(), 1e-6, 500, 1e-4, 5.0)
        except RuntimeError:
            iv = vol
        sq += (iv - vol) ** 2
    rmse = (sq / len(quotes)) ** 0.5
    return params, rmse


def calibrate_heston(
    quotes: Sequence[tuple[float, float, float]],
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    initial: HestonParams | None = None,
) -> HestonCalibration:
    """Calibrate Heston to ``(T_years, strike, implied_vol)`` quotes.

    Multi-start bounded Levenberg-Marquardt over QuantLib Heston helpers,
    minimising implied-vol error. Reports the root-mean-square implied-vol error
    of the best fit. Pass ``initial`` to pin a single starting point.
    """
    if not quotes:
        raise ValueError("calibrate_heston requires at least one quote")
    ql, _today, _dc, spot_h, r_ts, q_ts = _ql_context(spot, rate, dividend_yield)
    guesses = (
        [initial] if initial is not None else _seed_guesses(quotes, spot, rate, dividend_yield)
    )

    best_params: HestonParams | None = None
    best_rmse = float("inf")
    for guess in guesses:
        try:
            params, rmse = _calibrate_once(ql, spot_h, r_ts, q_ts, quotes, spot, guess)
        except RuntimeError:
            continue  # this start failed to converge; try the next
        if rmse < best_rmse:
            best_params, best_rmse = params, rmse

    if best_params is None:
        raise RuntimeError("Heston calibration failed from every starting point")
    return HestonCalibration(params=best_params, rmse_vol=best_rmse)
