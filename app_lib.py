"""Shared Streamlit helpers used by the multipage app.

Keeps pages thin: provider selection, cached data loaders, the freshness
badge, and rate assumptions live here so each page is only ``osl.*`` calls plus
viz. This module is the *only* place pages touch Streamlit caching.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
import streamlit as st

from osl.config import Settings, get_settings
from osl.data.providers import Freshness, build_provider, freshness_badge
from osl.strategy.enumerate import View, enumerate_candidates
from osl.strategy.liquidity import strategy_liquidity
from osl.strategy.metrics import compute_metrics
from osl.strategy.optimizer import Candidate, rank
from osl.strategy.strategies import ChainContext
from osl.surface.prepare import prepare_smiles
from osl.volatility.garch import GarchForecast, fit_garch
from osl.volatility.ranks import iv_percentile, iv_rank
from osl.volatility.realized import realized_vol
from osl.volatility.skew import delta_skew_25

DISCLAIMER = "Research and education only — not investment advice."
BADGE_COLOR = {Freshness.GREEN: "green", Freshness.AMBER: "orange", Freshness.RED: "red"}


def settings() -> Settings:
    return get_settings()


def sidebar_controls(default_symbol: str = "SPY") -> tuple[str, str]:
    """Render the shared sidebar; return (symbol, provider_name)."""
    cfg = get_settings()
    with st.sidebar:
        st.header("Data")
        symbol = st.text_input("Symbol", value=default_symbol).strip().upper()
        provider = st.radio(
            "Provider",
            options=["schwab", "yfinance"],
            index=0 if cfg.default_provider == "schwab" else 1,
        )
    return symbol, provider


def rate_assumptions() -> tuple[float, float]:
    """Return (risk_free_rate, dividend_yield) used across analytics.

    M1 uses the configured flat risk-free rate and a zero dividend yield; FRED
    curves and per-name dividends arrive in later milestones.
    """
    return get_settings().risk_free_flat_rate, 0.0


@st.cache_data(ttl=60, show_spinner=False)
def load_chain(provider_name: str, symbol: str) -> pd.DataFrame:
    provider = build_provider(provider_name, get_settings())
    return provider.get_option_chain(symbol)


@st.cache_data(ttl=15, show_spinner=False)
def load_underlying_dict(provider_name: str, symbol: str) -> dict[str, object]:
    provider = build_provider(provider_name, get_settings())
    q = provider.get_underlying(symbol)
    return {
        "symbol": q.symbol,
        "last": q.last,
        "mark": q.mark,
        "quote_time": q.quote_time,
        "is_delayed": q.is_delayed,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(provider_name: str, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
    provider = build_provider(provider_name, get_settings())
    end = date.today()
    start = end - timedelta(days=lookback_days)
    return provider.get_history(symbol, start=start, end=end)


def render_badge(provider_name: str, *, is_delayed: bool, quote_time: pd.Timestamp) -> None:
    badge = freshness_badge(provider_name, is_delayed=is_delayed, quote_time=quote_time)
    color = BADGE_COLOR[badge]
    st.markdown(
        f":{color}[**{badge.value}**] — {provider_name} @ "
        f"{pd.Timestamp(quote_time):%Y-%m-%d %H:%M:%S %Z}"
    )


@st.cache_data(ttl=120, show_spinner="Enumerating and scoring candidates…")
def build_candidates(
    provider_name: str,
    symbol: str,
    view: str,
    dte_low: int,
    dte_high: int,
    max_strikes: int,
    n_mc: int = 6000,
) -> list[Candidate]:
    """Enumerate strategies for the view and attach metrics + liquidity."""
    chain = load_chain(provider_name, symbol)
    under = load_underlying_dict(provider_name, symbol)
    spot = float(under["mark"] or under["last"])
    rate, div = rate_assumptions()
    ctx = ChainContext(chain, symbol, spot, rate, div)
    strategies = enumerate_candidates(
        ctx,
        view=cast(View, view),
        dte_range=(dte_low, dte_high),
        max_strikes_per_leg=max_strikes,
    )
    out: list[Candidate] = []
    for strat in strategies:
        out.append(
            Candidate(
                strategy=strat,
                metrics=compute_metrics(strat, n_mc=n_mc),
                liquidity=strategy_liquidity(chain, strat),
            )
        )
    return out


def candidates_table(candidates: list[Candidate]) -> pd.DataFrame:
    """Flatten candidates into a display DataFrame (one row per strategy)."""
    rows = []
    for c in candidates:
        m = c.metrics
        rows.append(
            {
                "strategy": c.strategy.name,
                "expiry": min(leg.expiration for leg in c.strategy.legs).isoformat(),
                "net": round(m.net_debit, 2),
                "credit?": m.is_credit,
                "POP (RN)": round(m.pop_rn, 3),
                "EV": round(m.ev_mc.value, 2),
                "max_profit": "∞" if m.profit_unbounded else round(m.max_profit, 2),
                "max_loss": "∞ (uncovered)" if m.loss_unbounded else round(m.max_loss, 2),
                "ES(95%)": round(m.expected_shortfall, 2),
                "ROR": round(m.return_on_risk, 3),
                "theta/day": round(m.greeks["theta"], 2),
                "vega": round(m.greeks["vega"], 2),
                "liquidity": round(c.liquidity.score, 2),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def load_log_returns(provider_name: str, symbol: str) -> pd.Series:
    hist = load_history(provider_name, symbol)
    return cast(pd.Series, np.log(hist["close"]).diff().dropna())


@st.cache_data(ttl=3600, show_spinner="Fitting GARCH…")
def garch_forecast(
    provider_name: str, symbol: str, model: str, dist: str, horizon: int
) -> GarchForecast:
    return fit_garch(
        load_log_returns(provider_name, symbol),
        model=cast("Any", model),
        dist=cast("Any", dist),
        horizon=horizon,
    )


@st.cache_data(ttl=300, show_spinner=False)
def screen_symbol(provider_name: str, symbol: str, objective: str) -> dict[str, object]:
    """Per-name screener row: IVR/IVP proxy, 25Δ RR, VRP, and the top strategy."""
    chain = load_chain(provider_name, symbol)
    under = load_underlying_dict(provider_name, symbol)
    spot = float(under["mark"] or under["last"])
    rate, div = rate_assumptions()

    hist = load_history(provider_name, symbol)
    rv30_series = realized_vol(hist, method="yz", window=30).dropna()
    rv30 = float(rv30_series.iloc[-1]) if not rv30_series.empty else float("nan")
    rv20 = realized_vol(hist, method="yz", window=20).dropna()

    smiles = prepare_smiles(chain, spot=spot, rate=rate, dividend_yield=div)
    iv30 = float("nan")
    rr25 = float("nan")
    if smiles:
        nearest = min(smiles, key=lambda s: abs(s.T - 30 / 365))
        ds = delta_skew_25(nearest)
        iv30, rr25 = ds.atm_iv, ds.risk_reversal_25

    ctx = ChainContext(chain, symbol, spot, rate, div)
    candidates = [
        Candidate(strat, compute_metrics(strat, n_mc=4000), strategy_liquidity(chain, strat))
        for strat in enumerate_candidates(ctx, view="all", dte_range=(20, 60))
    ]
    best = rank(candidates, objective)[0] if candidates else None

    return {
        "symbol": symbol,
        "spot": round(spot, 2),
        "IV30": None if np.isnan(iv30) else round(iv30, 4),
        "RV30": None if np.isnan(rv30) else round(rv30, 4),
        "IV-RV": None if (np.isnan(iv30) or np.isnan(rv30)) else round(iv30 - rv30, 4),
        "25dRR": None if np.isnan(rr25) else round(rr25, 4),
        "RVrank": None if rv20.empty else round(iv_rank(rv20), 3),
        "RVpct": None if rv20.empty else round(iv_percentile(rv20), 3),
        "top_strategy": best.strategy.name if best else None,
        "top_POP": round(best.metrics.pop_rn, 3) if best else None,
        "top_EV": round(best.metrics.ev_mc.value, 2) if best else None,
        "liquidity": round(best.liquidity.score, 2) if best else None,
    }


def page_header(title: str) -> None:
    st.title(title)
    st.caption(DISCLAIMER)


def page_footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
