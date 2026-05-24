"""Options Strategy Lab — Streamlit entrypoint (M0).

A single page that fetches an underlying quote + option chain through the
selected provider and renders it with a data-freshness badge. Per the
blueprint, this page contains no business logic: it only calls ``osl.*`` and
displays results.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from osl.config import get_settings
from osl.data.providers import Freshness, build_provider, freshness_badge

DISCLAIMER = "Research and education only — not investment advice."
BADGE_COLOR = {Freshness.GREEN: "green", Freshness.AMBER: "orange", Freshness.RED: "red"}

st.set_page_config(page_title="Options Strategy Lab", layout="wide")
settings = get_settings()

st.title("Options Strategy Lab")
st.caption(DISCLAIMER)

with st.sidebar:
    st.header("Data")
    symbol = st.text_input("Symbol", value="SPY").strip().upper()
    provider_name = st.radio(
        "Provider",
        options=["schwab", "yfinance"],
        index=0 if settings.default_provider == "schwab" else 1,
    )
    fetch = st.button("Fetch chain", type="primary")


def _render_badge(provider: str, *, is_delayed: bool, quote_time: pd.Timestamp) -> None:
    badge = freshness_badge(provider, is_delayed=is_delayed, quote_time=quote_time)
    color = BADGE_COLOR[badge]
    st.markdown(f":{color}[**{badge.value}**] — {provider} @ {quote_time:%Y-%m-%d %H:%M:%S %Z}")


if fetch and symbol:
    try:
        provider = build_provider(provider_name, settings)
        underlying = provider.get_underlying(symbol)
        chain = provider.get_option_chain(symbol)
    except Exception as exc:  # surface any provider failure in the UI
        st.error(f"Could not load data for {symbol} via {provider_name}: {exc}")
        st.info(
            "Schwab requires OAuth credentials and a saved token; yfinance requires "
            "network access. Configure `.env` (see `.env.example`) and complete the "
            "Schwab OAuth flow, or switch providers."
        )
    else:
        col_price, col_badge = st.columns([1, 2])
        col_price.metric(f"{symbol} last", f"{underlying.last:,.2f}")
        with col_badge:
            _render_badge(
                provider_name,
                is_delayed=underlying.is_delayed,
                quote_time=pd.Timestamp(underlying.quote_time),
            )

        st.subheader(f"Option chain — {len(chain):,} contracts")
        if chain.empty:
            st.warning("No contracts returned.")
        else:
            expirations = sorted(chain["expiration"].dt.date.unique())
            chosen = st.multiselect("Expirations", expirations, default=expirations[:1])
            view = chain[chain["expiration"].dt.date.isin(chosen)] if chosen else chain
            st.dataframe(view, use_container_width=True, hide_index=True)
else:
    st.info("Enter a symbol and click **Fetch chain** to begin.")

st.divider()
st.caption(DISCLAIMER)
