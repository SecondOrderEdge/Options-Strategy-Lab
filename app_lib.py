"""Shared Streamlit helpers used by the multipage app.

Keeps pages thin: provider selection, cached data loaders, the freshness
badge, and rate assumptions live here so each page is only ``osl.*`` calls plus
viz. This module is the *only* place pages touch Streamlit caching.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import cast

import pandas as pd
import streamlit as st

from osl.config import Settings, get_settings
from osl.data.providers import Freshness, build_provider, freshness_badge
from osl.strategy.enumerate import View, enumerate_candidates
from osl.strategy.liquidity import strategy_liquidity
from osl.strategy.metrics import compute_metrics
from osl.strategy.optimizer import Candidate
from osl.strategy.strategies import ChainContext

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


def page_header(title: str) -> None:
    st.title(title)
    st.caption(DISCLAIMER)


def page_footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
