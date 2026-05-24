"""Advanced Models (experimental): Heston, Merton jumps, Dupire, surface PCA.

Gated behind OSL_ENABLE_EXPERIMENTAL. These are research models with real
calibration cost and identification caveats — read the assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_lib import (
    load_chain,
    load_underlying_dict,
    page_footer,
    page_header,
    rate_assumptions,
    settings,
    sidebar_controls,
)
from osl.models.heston import calibrate_heston, heston_price
from osl.models.jump_diffusion import MertonParams, merton_price
from osl.models.local_vol import dupire_local_vol
from osl.pricing.bsm import bsm_price
from osl.pricing.iv import implied_vol
from osl.surface.pca import surface_pca
from osl.surface.prepare import prepare_smiles
from osl.surface.svi import fit_svi, make_iv_of_strike
from osl.viz.charts import heatmap_chart

page_header("Advanced Models (experimental)")

if not settings().enable_experimental:
    st.warning(
        "Experimental models are gated. Set `OSL_ENABLE_EXPERIMENTAL=true` to enable "
        "Heston, Merton jump-diffusion, Dupire local vol, and surface PCA."
    )
    page_footer()
    st.stop()

symbol, provider = sidebar_controls()
go = st.sidebar.button("Load", type="primary")

if not (go and symbol):
    st.caption("Set a symbol and click **Load**.")
    page_footer()
    st.stop()

try:
    under = load_underlying_dict(provider, symbol)
    chain = load_chain(provider, symbol)
except Exception as exc:  # surface provider/credential failures
    st.error(f"Could not load {symbol} via {provider}: {exc}")
    page_footer()
    st.stop()

spot = float(under["mark"] or under["last"])
rate, div = rate_assumptions()
smiles = prepare_smiles(chain, spot=spot, rate=rate, dividend_yield=div)[:4]
if not smiles:
    st.warning("Not enough liquid quotes to fit advanced models.")
    page_footer()
    st.stop()

tab_h, tab_m, tab_lv, tab_pca = st.tabs(
    ["Heston", "Merton jumps", "Dupire local vol", "Surface PCA"]
)

with tab_h:
    quotes = [
        (s.T, float(k), float(iv)) for s in smiles for k, iv in zip(s.strikes, s.iv, strict=True)
    ]
    try:
        cal = calibrate_heston(quotes, spot=spot, rate=rate, dividend_yield=div)
    except Exception as exc:  # calibration can fail to converge
        st.error(f"Heston calibration failed: {exc}")
    else:
        p = cal.params
        st.write(
            {
                "v0": round(p.v0, 4),
                "kappa": round(p.kappa, 4),
                "theta": round(p.theta, 4),
                "sigma(vol-of-vol)": round(p.sigma, 4),
                "rho": round(p.rho, 4),
                "RMSE (vol pts)": round(cal.rmse_vol * 100, 3),
            }
        )
        near = smiles[0]
        model_iv = []
        for k in near.strikes:
            price = heston_price(spot, float(k), near.T, rate, div, "c", p)
            try:
                model_iv.append(
                    implied_vol(price, spot, float(k), near.T, rate, div, "c", method="bisect")
                )
            except Exception:
                model_iv.append(np.nan)
        st.line_chart(
            pd.DataFrame({"market": near.iv, "Heston": model_iv}, index=near.strikes.round(1))
        )

with tab_m:
    c1, c2, c3 = st.columns(3)
    lam = c1.slider("Jump intensity λ", 0.0, 3.0, 1.0)
    mu_j = c2.slider("Mean log-jump μ", -0.3, 0.1, -0.1)
    delta = c3.slider("Jump vol δ", 0.0, 0.4, 0.15)
    params = MertonParams(jump_intensity=lam, jump_mean=mu_j, jump_std=delta)
    near = smiles[0]
    atm_iv = float(near.iv[int(np.argmin(np.abs(near.k)))])
    strikes = near.strikes
    merton_p = [
        merton_price(spot, float(k), near.T, rate, div, atm_iv, "p", params) for k in strikes
    ]
    bsm_p = [
        float(bsm_price(spot, float(k), near.T, rate, div, atm_iv, "p").item()) for k in strikes
    ]
    st.line_chart(pd.DataFrame({"Merton put": merton_p, "BSM put": bsm_p}, index=strikes.round(1)))
    st.caption("Jumps fatten the left tail, lifting OTM put values above pure diffusion.")

with tab_lv:
    grid_strikes = np.linspace(0.8 * spot, 1.2 * spot, 11)
    expiries = [s.T for s in smiles]
    iv_grid = np.empty((grid_strikes.size, len(smiles)))
    for j, s in enumerate(smiles):
        fit = fit_svi(s.k, s.total_variance, weights=s.weights, T=s.T)
        iv_fn = make_iv_of_strike(fit.params, forward=s.forward, T=s.T)
        iv_grid[:, j] = iv_fn(grid_strikes)
    lv = dupire_local_vol(
        expiries, grid_strikes.tolist(), iv_grid, spot=spot, rate=rate, dividend_yield=div
    )
    st.plotly_chart(
        heatmap_chart(
            lv,
            x=[round(t, 3) for t in expiries],
            y=grid_strikes.round(1).tolist(),
            x_title="T (years)",
            y_title="Strike",
            title="Dupire local volatility",
            colorbar_title="local vol",
        ),
        use_container_width=True,
    )

with tab_pca:
    st.info(
        "Surface PCA needs an accumulated multi-day snapshot history. Showing a "
        "**synthetic 3-factor demo** until snapshots accumulate."
    )
    rng = np.random.default_rng(0)
    n_days, n_points = 80, 12
    loadings = rng.normal(size=(3, n_points))
    factors = rng.normal(size=(n_days, 3)) * np.array([1.0, 0.5, 0.25])
    panel = pd.DataFrame(
        factors @ loadings + rng.normal(0, 0.002, size=(n_days, n_points)),
        columns=[f"p{i}" for i in range(n_points)],
    )
    pca = surface_pca(panel, n_factors=3, on="changes")
    st.bar_chart(
        pd.DataFrame(
            {"explained variance": pca.explained_variance_ratio[:6]},
            index=[f"PC{i + 1}" for i in range(6)],
        )
    )
    st.caption(
        f"Top 3 factors ({', '.join(pca.factor_names)}) explain "
        f"{pca.explained_variance_ratio[:3].sum():.1%} of surface-change variance."
    )

with st.expander("Model assumptions"):
    st.markdown(
        "- **Heston** is calibrated by Levenberg-Marquardt to the OTM smile; "
        "parameters can be weakly identified on sparse chains.\n"
        "- **Merton** jumps are illustrative; intensity/size are hard to identify "
        "from a single snapshot.\n"
        "- **Dupire local vol** is numerically delicate in the wings (cells may be "
        "blank where it fails).\n"
        "- **Surface PCA** here is a synthetic demo; real factor structure needs "
        "accumulated snapshots (Cont & da Fonseca 2002)."
    )

page_footer()
