"""Options Chain: filterable chain with computed greeks and liquidity flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_lib import (
    column_help,
    load_chain,
    load_underlying_dict,
    page_footer,
    page_header,
    rate_assumptions,
    render_badge,
    sidebar_controls,
)
from osl.pricing.bsm import bsm_greeks
from osl.utils.time import DAYS_PER_YEAR

page_header("Options Chain")
symbol, provider = sidebar_controls()

if not symbol:
    st.info("Enter a symbol in the sidebar.")
    page_footer()
    st.stop()

try:
    under = load_underlying_dict(provider, symbol)
    chain = load_chain(provider, symbol).copy()
except Exception as exc:
    st.error(f"Could not load {symbol} via {provider}: {exc}")
    page_footer()
    st.stop()

render_badge(
    provider, is_delayed=bool(under["is_delayed"]), quote_time=pd.Timestamp(under["quote_time"])
)

spot = float(under["mark"] or under["last"])
rate, div = rate_assumptions()

# Compute greeks where the provider omitted them (e.g. yfinance), from mid IV.
T = chain["dte"].to_numpy(dtype=float) / DAYS_PER_YEAR
flags = np.where(chain["right"].to_numpy() == "C", "c", "p")
iv = chain["iv"].to_numpy(dtype=float)
have_iv = np.isfinite(iv) & (iv > 0)
if chain["delta"].isna().any():
    greeks = bsm_greeks(
        spot,
        chain["strike"].to_numpy(dtype=float),
        T,
        rate,
        div,
        np.where(have_iv, iv, np.nan),
        flags,
    )
    for name in ("delta", "gamma", "theta", "vega", "rho"):
        chain[name] = chain[name].where(
            chain[name].notna(), pd.Series(greeks[name], index=chain.index)
        )

# Liquidity flags.
spread = chain["ask"] - chain["bid"]
chain["spread_pct"] = (spread / chain["mid"]).where(chain["mid"] > 0, np.nan)
chain["zero_bid"] = chain["bid"].isna() | (chain["bid"] <= 0)
chain["wide_spread"] = chain["spread_pct"] > 0.10

expirations = sorted(chain["expiration"].dt.date.unique())
chosen = st.multiselect("Expirations", expirations, default=expirations[:1])
right = st.radio("Right", ["Both", "Calls", "Puts"], horizontal=True)

view = chain[chain["expiration"].dt.date.isin(chosen)] if chosen else chain
if right == "Calls":
    view = view[view["right"] == "C"]
elif right == "Puts":
    view = view[view["right"] == "P"]

cols = [
    "expiration",
    "right",
    "strike",
    "bid",
    "ask",
    "mid",
    "volume",
    "open_interest",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "spread_pct",
    "zero_bid",
    "wide_spread",
    "is_nonstandard",
]
st.subheader(f"{len(view):,} contracts")
st.caption(
    "Hover any column header for what it means. Greeks: delta ≈ chance ITM, "
    "theta = daily decay, vega = sensitivity to a 1-point IV move."
)
st.dataframe(
    view[cols],
    use_container_width=True,
    hide_index=True,
    column_config=column_help(cols),
)

n_flagged = int(view["zero_bid"].sum() + view["wide_spread"].sum())
if n_flagged:
    st.warning(f"{n_flagged} rows flagged for zero-bid or wide spread (>10%).")

with st.expander("Model assumptions"):
    st.markdown(
        "- Greeks shown are provider-reported when available, otherwise computed "
        f"via Black-Scholes-Merton at the row IV (flat r={rate:.2%}, q={div:.2%}).\n"
        "- theta is per day; vega per 1 vol point.\n"
        "- Liquidity flags: zero-bid, or bid/ask spread > 10% of mid."
    )

page_footer()
