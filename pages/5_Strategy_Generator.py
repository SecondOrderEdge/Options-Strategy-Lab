"""Strategy Generator: top-N candidates per objective for a directional view."""

from __future__ import annotations

import streamlit as st

from app_lib import (
    build_candidates,
    candidates_table,
    column_help,
    page_footer,
    page_header,
    render_chart,
    sidebar_controls,
)
from osl.strategy.optimizer import rank
from osl.viz.charts import radar_chart

page_header("Strategy Generator")
symbol, provider = sidebar_controls()

with st.sidebar:
    view = st.selectbox("View", ["bullish", "bearish", "neutral", "volatile", "all"], index=4)
    dte_low, dte_high = st.slider("DTE range", 1, 180, (20, 60))
    max_strikes = st.slider("Max strikes per leg", 1, 5, 3)
    top_n = st.slider("Top N per objective", 1, 10, 3)
    go = st.button("Generate", type="primary")

st.info(
    "POP is **not** low risk. Every strategy below shows EV and tail loss (ES) "
    "beside POP; a high-POP credit trade can still have negative EV."
)

if not (go and symbol):
    st.caption("Set a symbol and filters, then click **Generate**.")
    page_footer()
    st.stop()

try:
    candidates = build_candidates(provider, symbol, view, dte_low, dte_high, max_strikes)
except Exception as exc:  # surface provider/credential failures
    st.error(f"Could not generate candidates for {symbol} via {provider}: {exc}")
    page_footer()
    st.stop()

if not candidates:
    st.warning("No candidates met the filters (check liquidity / DTE range).")
    page_footer()
    st.stop()

OBJECTIVE_LABELS = {
    "ev_per_risk": "EV per $ risk",
    "pop_capped": "POP (capped risk)",
    "theta_per_risk": "Theta per $ risk",
    "liquidity_adj_ev": "Liquidity-adjusted EV",
}

st.caption(
    "Hover any column header for what it means. Read **POP** next to **EV** and "
    "**max_loss**: a high probability of profit can still carry negative EV or "
    "unbounded loss."
)
for obj, label in OBJECTIVE_LABELS.items():
    st.subheader(label)
    top = rank(candidates, obj)[:top_n]
    table = candidates_table(top)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config=column_help(table.columns),
    )

# Radar for the single best EV-per-risk candidate.
best = rank(candidates, "ev_per_risk")[0]
m = best.metrics
norm = [
    max(min(m.pop_rn, 1.0), 0.0),
    (
        max(min((m.ev_mc.value / max(abs(m.max_loss), 1.0)) + 0.5, 1.0), 0.0)
        if not m.loss_unbounded
        else 0.0
    ),
    max(min(abs(m.greeks["theta"]) / 50.0, 1.0), 0.0),
    max(min(abs(m.greeks["vega"]) / 100.0, 1.0), 0.0),
    max(min(abs(m.convexity), 1.0), 0.0),
    best.liquidity.score,
]
st.subheader(f"Profile — {best.strategy.name}")
render_chart(
    radar_chart(
        ["POP", "EV/risk", "Theta", "Vega", "Convexity", "Liquidity"], norm, name=best.strategy.name
    ),
)
st.caption(
    "Each spoke is scaled 0-1 (further out = stronger): POP = chance of profit, "
    "EV/risk = edge per dollar risked, Theta = decay collected, Vega = volatility "
    "exposure, Convexity = payoff curvature, Liquidity = tradability."
)

with st.expander("Model assumptions"):
    st.markdown(
        "- POP (RN) is risk-neutral lognormal at the vega-weighted IV; EV is RN "
        "Monte Carlo (edge vs fair value). Both ignore real-world drift.\n"
        "- Uncovered short options show max loss as ∞ and are never ranked top by "
        "the capped-risk objective.\n"
        "- Liquidity blends spread, open interest, volume, and ATM distance."
    )

page_footer()
