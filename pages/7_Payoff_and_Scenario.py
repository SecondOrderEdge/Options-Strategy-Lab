"""Payoff & Scenario: payoff curves, the P&L surface, and stress scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_lib import build_candidates, page_footer, page_header, render_chart, sidebar_controls
from osl.scenario.payoff import payoff_curve
from osl.scenario.pnl_surface import pnl_grid
from osl.scenario.stress import stress_scenarios
from osl.viz.charts import payoff_chart, pnl_surface_chart

page_header("Payoff & Scenario")
symbol, provider = sidebar_controls()

with st.sidebar:
    view = st.selectbox("View", ["all", "bullish", "bearish", "neutral", "volatile"], index=0)
    dte_low, dte_high = st.slider("DTE range", 1, 180, (20, 60))
    max_strikes = st.slider("Max strikes per leg", 1, 5, 3)
    go = st.button("Build candidates", type="primary")

if not (go and symbol):
    st.caption("Set a symbol and filters, then click **Build candidates** and pick one.")
    page_footer()
    st.stop()

try:
    candidates = build_candidates(provider, symbol, view, dte_low, dte_high, max_strikes)
except Exception as exc:  # surface provider/credential failures
    st.error(f"Could not build candidates for {symbol} via {provider}: {exc}")
    page_footer()
    st.stop()

if not candidates:
    st.warning("No candidates met the filters.")
    page_footer()
    st.stop()


def _label(i: int) -> str:
    c = candidates[i]
    m = c.metrics
    exp = min(leg.expiration for leg in c.strategy.legs).isoformat()
    return f"{c.strategy.name} | {exp} | POP {m.pop_rn:.0%} | EV {m.ev_mc.value:,.0f}"


idx = st.selectbox("Strategy", range(len(candidates)), format_func=_label)
chosen = candidates[idx]
strategy = chosen.strategy

shock_days = st.slider("Days forward (for the dashed curve)", 0, max(strategy.primary_dte, 1), 0)
vol_shock_pts = st.slider("IV shock (vol points)", -20, 20, 0)
vol_shock = vol_shock_pts / 100.0

# Payoff: at expiry (solid) and at the chosen time/IV shock (dashed overlay via a 2nd chart).
spots, pnl_exp = payoff_curve(strategy)
fig = payoff_chart(
    spots,
    pnl_exp,
    current_spot=strategy.spot,
    breakevens=chosen.metrics.breakevens,
    title=f"{strategy.name} — payoff at expiry",
)
if shock_days > 0 or vol_shock != 0:
    _, pnl_now = payoff_curve(strategy, days_forward=float(shock_days), vol_shock=vol_shock)
    fig.add_scatter(
        x=spots,
        y=pnl_now,
        mode="lines",
        line={"dash": "dash", "color": "#9d6bb0"},
        name=f"+{shock_days}d, {vol_shock_pts:+d} vol pts",
    )
render_chart(fig)

st.subheader("P&L surface (spot × time)")
sg, dg, z = pnl_grid(strategy, spot_shocks=np.linspace(-0.3, 0.3, 31), vol_shock=vol_shock)
render_chart(pnl_surface_chart(sg, dg, z))

st.subheader("Stress scenarios")
iv_crush = st.slider("Earnings IV crush (vol points)", -40, 0, -20) / 100.0
results = stress_scenarios(
    strategy, sigma=chosen.metrics.reference_sigma, days_forward=0.0, earnings_iv_crush=iv_crush
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "scenario": r.label,
                "spot": round(r.spot, 2),
                "IV shock (pts)": round(r.vol_shock * 100, 1),
                "P&L": round(r.pnl, 2),
            }
            for r in results
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Model assumptions"):
    st.markdown(
        "- Payoff at expiry uses intrinsic value; the dashed curve and the surface "
        "reprice each leg via BSM at the remaining time with the IV shock applied.\n"
        "- σ-moves use the strategy's vega-weighted reference IV over its primary "
        "horizon. Stress shocks are **instantaneous** (constant t) unless noted.\n"
        "- Scenarios hold IV flat across spot except where an explicit IV shock is set; "
        "real skew dynamics under a spot move are not modeled here."
    )

page_footer()
