"""Heston stochastic-volatility pricing and calibration via QuantLib.

Heston dynamics: ``dS = (r-q)S dt + sqrt(v) S dW1``,
``dv = kappa(theta - v) dt + sigma sqrt(v) dW2``, ``dW1 dW2 = rho dt``.

As the vol-of-vol ``sigma`` -> 0 and ``rho`` -> 0 (with ``v0 = theta = sigma_BS^2``)
Heston collapses to Black-Scholes, which the tests verify.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from osl.utils.time import DAYS_PER_YEAR

_EVAL_YEAR = 2026
_EVAL_MONTH = 5
_EVAL_DAY = 15


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


def calibrate_heston(
    quotes: Sequence[tuple[float, float, float]],
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    initial: HestonParams | None = None,
) -> HestonCalibration:
    """Calibrate Heston to ``(T_years, strike, implied_vol)`` quotes.

    Uses Levenberg-Marquardt over QuantLib Heston helpers and reports the
    root-mean-square implied-vol error of the fit.
    """
    ql, _today, _dc, spot_h, r_ts, q_ts = _ql_context(spot, rate, dividend_yield)
    guess = initial or HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
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
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)

    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    model.calibrate(helpers, lm, ql.EndCriteria(500, 50, 1e-8, 1e-8, 1e-8))

    theta, kappa, sigma, rho, v0 = model.params()
    params = HestonParams(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)

    # RMSE in vol via each helper's implied-vol error.
    sq = 0.0
    for helper, (_t, _k, vol) in zip(helpers, quotes, strict=True):
        model_price = helper.modelValue()
        try:
            iv = helper.impliedVolatility(model_price, 1e-6, 200, 1e-4, 5.0)
        except RuntimeError:
            iv = vol
        sq += (iv - vol) ** 2
    rmse = (sq / len(quotes)) ** 0.5 if quotes else 0.0
    return HestonCalibration(params=params, rmse_vol=rmse)
