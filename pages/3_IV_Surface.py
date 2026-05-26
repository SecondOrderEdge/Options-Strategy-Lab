"""IV Surface: SVI fit per expiry, 3D surface / heatmap, arbitrage checks."""

from __future__ import annotations

import contextlib

import pandas as pd
import streamlit as st

from app_lib import (
    column_help,
    load_chain,
    load_underlying_dict,
    page_footer,
    page_header,
    rate_assumptions,
    render_chart,
    sidebar_controls,
    skew_reading,
)
from osl.surface.prepare import prepare_smiles
from osl.surface.svi import SVIParams, calendar_arbitrage_free, fit_svi
from osl.viz.charts import iv_heatmap, iv_surface_3d, skew_chart
from osl.volatility.skew import delta_skew_25

page_header("IV Surface")
symbol, provider = sidebar_controls()

if not symbol:
    st.info("Enter a symbol in the sidebar.")
    page_footer()
    st.stop()

try:
    under = load_underlying_dict(provider, symbol)
    chain = load_chain(provider, symbol)
except Exception as exc:
    st.error(f"Could not load {symbol} via {provider}: {exc}")
    page_footer()
    st.stop()

spot = float(under["mark"] or under["last"])
rate, div = rate_assumptions()
smiles = prepare_smiles(chain, spot=spot, rate=rate, dividend_yield=div)

if not smiles:
    st.warning("Not enough liquid quotes to build a surface.")
    page_footer()
    st.stop()

fits: dict[str, SVIParams] = {}
rows = []
for smile in smiles:
    label = str(smile.expiration)
    try:
        fit = fit_svi(smile.k, smile.total_variance, weights=smile.weights, T=smile.T)
    except ValueError:
        continue
    fits[label] = fit.params
    rows.append(
        {
            "expiration": label,
            "T": round(smile.T, 4),
            "n": fit.n_points,
            "rmse_vol_pts": round(fit.rmse_vol * 100, 3),
            "butterfly_free": fit.butterfly_free,
            "a": round(fit.params.a, 4),
            "b": round(fit.params.b, 4),
            "rho": round(fit.params.rho, 4),
            "m": round(fit.params.m, 4),
            "sigma": round(fit.params.sigma, 4),
        }
    )

view = st.radio("View", ["Skew (2D)", "Surface (3D)", "Heatmap"], horizontal=True)
if view == "Skew (2D)":
    render_chart(skew_chart(smiles, fits))
    near30 = min(smiles, key=lambda s: abs(s.T - 30 / 365))
    with contextlib.suppress(Exception):  # 25Δ may not be spanned on a thin smile
        st.caption(
            "Reading this for trades: " + skew_reading(delta_skew_25(near30).risk_reversal_25)
        )
elif view == "Surface (3D)":
    render_chart(iv_surface_3d(smiles))
else:
    render_chart(iv_heatmap(smiles))

st.subheader("SVI parameters per expiry")
st.caption(
    "Each expiry's smile is summarised by 5 raw-SVI numbers (hover the headers for "
    "each): **a** = level (height), **b** = wing steepness, **ρ** = skew/tilt "
    "(negative = downside-heavy), **m** = where the smile bottoms, **σ** = how rounded "
    "the bottom is. **rmse_vol_pts** is the fit error; **butterfly_free** flags arbitrage."
)
params_df = pd.DataFrame(rows)
st.dataframe(
    params_df,
    use_container_width=True,
    hide_index=True,
    column_config=column_help(params_df.columns),
)

cal_free = calendar_arbitrage_free(
    [(s.T, fits[str(s.expiration)]) for s in smiles if str(s.expiration) in fits]
)
if not params_df.empty and not params_df["butterfly_free"].all():
    st.warning("Butterfly arbitrage detected on one or more expiries — wings may be unreliable.")
if not cal_free:
    st.warning("Calendar arbitrage detected: total variance is not monotonic across expiries.")
if not params_df.empty and params_df["butterfly_free"].all() and cal_free:
    st.success("No butterfly or calendar arbitrage detected on the fitted surface.")

with st.expander("Model assumptions"):
    st.markdown(
        "- Surface: raw SVI per expiry (Gatheral), vega-weighted least squares.\n"
        "- No-arb checks: Gatheral g(k) butterfly test and calendar monotonicity "
        "(Gatheral & Jacquier 2014).\n"
        f"- Forward uses flat r={rate:.2%}, q={div:.2%}. Wings beyond liquid strikes are extrapolated."
    )

page_footer()
