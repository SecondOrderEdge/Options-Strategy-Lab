"""Ticker Overview: spot, price history, IV30, IV rank/percentile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_lib import (
    load_chain,
    load_history,
    load_underlying_dict,
    page_footer,
    page_header,
    rate_assumptions,
    render_badge,
    render_chart,
    sidebar_controls,
)
from osl.surface.prepare import prepare_smiles
from osl.viz.charts import price_history_chart

page_header("Ticker Overview")
symbol, provider = sidebar_controls()

if not symbol:
    st.info("Enter a symbol in the sidebar.")
    page_footer()
    st.stop()

try:
    under = load_underlying_dict(provider, symbol)
    history = load_history(provider, symbol)
    chain = load_chain(provider, symbol)
except Exception as exc:  # surface provider/credential failures in the UI
    st.error(f"Could not load {symbol} via {provider}: {exc}")
    page_footer()
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric(
    f"{symbol} last",
    f"{under['last']:,.2f}",
    help="Last traded price of the underlying.",
)

rate, div = rate_assumptions()
smiles = prepare_smiles(
    chain, spot=float(under["mark"] or under["last"]), rate=rate, dividend_yield=div
)

# IV30: ATM IV of the smile whose T is closest to 30 calendar days.
iv30 = np.nan
if smiles:
    target_T = 30 / 365
    nearest = min(smiles, key=lambda s: abs(s.T - target_T))
    atm_idx = int(np.argmin(np.abs(nearest.k)))
    iv30 = float(nearest.iv[atm_idx])
col2.metric(
    "IV30 (ATM, nearest expiry)",
    "n/a" if np.isnan(iv30) else f"{iv30:.1%}",
    help="At-the-money implied vol of the expiry nearest 30 days — the market's "
    "expected annualized volatility over roughly the next month.",
)

with col3:
    render_badge(
        provider, is_delayed=bool(under["is_delayed"]), quote_time=pd.Timestamp(under["quote_time"])
    )

if not history.empty:
    render_chart(price_history_chart(history, title=f"{symbol} close"))
else:
    st.warning("No price history returned.")

with st.expander("Model assumptions"):
    st.markdown(
        f"- Risk-free rate: flat {rate:.2%} (M1). Dividend yield: {div:.2%}.\n"
        "- IV30 is the ATM IV of the nearest expiry, not a constant-maturity interpolation.\n"
        "- See **Assumptions & Disclaimers** for full caveats."
    )

page_footer()
