"""Vol Diagnostics: skew, term structure, volatility cone, IV vs realized."""

from __future__ import annotations

import contextlib

import numpy as np
import streamlit as st

from app_lib import (
    load_chain,
    load_history,
    load_underlying_dict,
    page_footer,
    page_header,
    rate_assumptions,
    render_chart,
    sidebar_controls,
)
from osl.surface.prepare import prepare_smiles
from osl.surface.svi import fit_svi
from osl.viz.charts import skew_chart, term_structure_chart, vol_cone_chart
from osl.volatility.ranks import iv_percentile, iv_rank, vol_cone
from osl.volatility.realized import realized_vol

page_header("Vol Diagnostics")
symbol, provider = sidebar_controls()

if not symbol:
    st.info("Enter a symbol in the sidebar.")
    page_footer()
    st.stop()

try:
    under = load_underlying_dict(provider, symbol)
    chain = load_chain(provider, symbol)
    history = load_history(provider, symbol)
except Exception as exc:
    st.error(f"Could not load {symbol} via {provider}: {exc}")
    page_footer()
    st.stop()

spot = float(under["mark"] or under["last"])
rate, div = rate_assumptions()
smiles = prepare_smiles(chain, spot=spot, rate=rate, dividend_yield=div)

tab_skew, tab_term, tab_cone, tab_ivrv = st.tabs(["Skew", "Term structure", "Vol cone", "IV vs RV"])

with tab_skew:
    if smiles:
        fits = {}
        for s in smiles:
            with contextlib.suppress(ValueError):
                fits[str(s.expiration)] = fit_svi(
                    s.k, s.total_variance, weights=s.weights, T=s.T
                ).params
        render_chart(skew_chart(smiles, fits))
    else:
        st.warning("No smiles available.")

with tab_term:
    if smiles:
        tenors, atm = [], []
        for s in smiles:
            tenors.append(s.T)
            atm.append(float(s.iv[int(np.argmin(np.abs(s.k)))]))
        render_chart(term_structure_chart(tenors, atm))
        shape = "contango" if len(atm) > 1 and atm[-1] > atm[0] else "backwardation"
        st.caption(f"Term-structure shape: {shape} (RN, model-implied ATM IV).")
    else:
        st.warning("No smiles available.")

with tab_cone:
    if not history.empty:
        cone = vol_cone(history["close"])
        render_chart(vol_cone_chart(cone))
        st.dataframe(cone, use_container_width=True)
    else:
        st.warning("No price history for the cone.")

with tab_ivrv:
    if not history.empty:
        rv30 = realized_vol(history, method="yz", window=30).dropna()
        rv_now = float(rv30.iloc[-1]) if not rv30.empty else float("nan")
        iv30 = np.nan
        if smiles:
            nearest = min(smiles, key=lambda s: abs(s.T - 30 / 365))
            iv30 = float(nearest.iv[int(np.argmin(np.abs(nearest.k)))])
        c1, c2, c3 = st.columns(3)
        c1.metric("Realized vol (30D, YZ)", "n/a" if np.isnan(rv_now) else f"{rv_now:.1%}")
        c2.metric("Implied vol (30D ATM)", "n/a" if np.isnan(iv30) else f"{iv30:.1%}")
        if not (np.isnan(rv_now) or np.isnan(iv30)):
            c3.metric("IV − RV (VRP proxy)", f"{(iv30 - rv_now) * 100:.1f} vol pts")
    else:
        st.warning("No price history for IV/RV.")

# IV rank/percentile from the realized-vol history as a stand-in series until a
# multi-day IV snapshot history accumulates (M5).
if not history.empty:
    rv_series = realized_vol(history, method="yz", window=20).dropna()
    if not rv_series.empty:
        st.subheader("Realized-vol rank (proxy until IV history accumulates)")
        c1, c2 = st.columns(2)
        c1.metric("RV rank", f"{iv_rank(rv_series):.0%}")
        c2.metric("RV percentile", f"{iv_percentile(rv_series):.0%}")

with st.expander("Model assumptions"):
    st.markdown(
        "- Skew/term use raw SVI ATM IV (risk-neutral, model-implied).\n"
        "- Vol cone and realized vol use Yang-Zhang on daily bars (252/yr).\n"
        "- IV−RV is a variance-risk-premium *proxy*, not a variance-swap replication.\n"
        "- IV rank/percentile here use realized vol as a stand-in; a true IV-history "
        "rank arrives once daily snapshots accumulate (M5)."
    )

page_footer()
