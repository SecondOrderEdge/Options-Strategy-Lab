"""Strategy Optimizer: sortable ranking with EV/POP-vs-risk frontiers."""

from __future__ import annotations

import math

import streamlit as st

from app_lib import (
    build_candidates,
    candidates_table,
    page_footer,
    page_header,
    render_chart,
    sidebar_controls,
)
from osl.strategy.optimizer import OBJECTIVES, rank
from osl.viz.charts import scatter_chart

page_header("Strategy Optimizer")
symbol, provider = sidebar_controls()

with st.sidebar:
    view = st.selectbox("View", ["all", "bullish", "bearish", "neutral", "volatile"], index=0)
    dte_low, dte_high = st.slider("DTE range", 1, 180, (20, 60))
    max_strikes = st.slider("Max strikes per leg", 1, 5, 3)
    objective = st.selectbox("Objective", list(OBJECTIVES), index=2)
    max_loss_cap = st.number_input("Max loss cap ($, 0 = none)", min_value=0, value=0, step=100)
    go = st.button("Optimize", type="primary")

if not (go and symbol):
    st.caption("Set a symbol, objective, and filters, then click **Optimize**.")
    page_footer()
    st.stop()

try:
    candidates = build_candidates(provider, symbol, view, dte_low, dte_high, max_strikes)
except Exception as exc:
    st.error(f"Could not optimize for {symbol} via {provider}: {exc}")
    page_footer()
    st.stop()

if max_loss_cap > 0:
    candidates = [
        c
        for c in candidates
        if not c.metrics.loss_unbounded and abs(c.metrics.max_loss) <= max_loss_cap
    ]

if not candidates:
    st.warning("No candidates met the filters.")
    page_footer()
    st.stop()

ranked = rank(candidates, objective)
st.subheader(f"Ranking by {objective} ({len(ranked)} candidates)")
st.dataframe(candidates_table(ranked), use_container_width=True, hide_index=True)

# Frontiers (bounded-risk candidates only, so axes are finite).
bounded = [c for c in ranked if not c.metrics.loss_unbounded and math.isfinite(c.metrics.max_loss)]
if bounded:
    labels = [c.strategy.name for c in bounded]
    risk = [abs(c.metrics.max_loss) for c in bounded]
    ev = [c.metrics.ev_mc.value for c in bounded]
    pop = [c.metrics.pop_rn for c in bounded]

    col1, col2 = st.columns(2)
    with col1:
        render_chart(
            scatter_chart(
                risk,
                ev,
                labels=labels,
                x_title="Max loss ($)",
                y_title="EV ($)",
                title="EV frontier",
            ),
        )
    with col2:
        render_chart(
            scatter_chart(
                risk,
                pop,
                labels=labels,
                x_title="Max loss ($)",
                y_title="POP (RN)",
                title="POP vs max loss",
            ),
        )

st.caption(
    f"{len([c for c in ranked if c.metrics.loss_unbounded])} uncovered (unbounded-risk) "
    "candidates are excluded from the frontiers and ranked last by capped objectives."
)

with st.expander("Model assumptions"):
    st.markdown(
        "- Ranking objectives are deterministic functions of the metrics; switching "
        "the objective re-orders the table.\n"
        "- `pop_capped` never ranks an unbounded-loss position first.\n"
        "- EV is RN Monte Carlo with a 95% confidence band (see the Probability Lab "
        "in a later milestone for real-world EV)."
    )

page_footer()
