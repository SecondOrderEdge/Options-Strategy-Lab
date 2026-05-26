"""Options Playbook: a reproducible, signed per-name HTML/PDF report."""

from __future__ import annotations

import streamlit as st

from app_lib import armed_button, build_playbook_data, page_footer, page_header, sidebar_controls
from osl.report.playbook import build_playbook_html, render_pdf, report_digest

page_header("Options Playbook")
symbol, provider = sidebar_controls()

with st.sidebar:
    view = st.selectbox("View", ["all", "bullish", "bearish", "neutral", "volatile"], index=0)
    dte_low, dte_high = st.slider("DTE range", 1, 180, (20, 60))
    armed = armed_button("Build report", key="playbook", signature=symbol, type="primary")

if not (armed and symbol):
    st.caption("Set a symbol and click **Build report** to generate the playbook.")
    page_footer()
    st.stop()

try:
    data = build_playbook_data(provider, symbol, view=view, dte_low=dte_low, dte_high=dte_high)
except Exception as exc:  # surface provider/credential failures
    st.error(f"Could not build the playbook for {symbol}: {exc}")
    page_footer()
    st.stop()

html = build_playbook_html(data)
st.success(f"Report digest `{report_digest(data)}` — reproducible for identical inputs.")
st.caption(
    "A one-page, shareable summary for this name: current IV level/rank, the surface "
    "no-arbitrage note, and the top strategy per objective with POP, EV, tail loss and "
    "liquidity. The report carries its own glossary and a content **digest** — the same "
    "inputs always produce the same digest, so a report can be verified and reproduced."
)

st.download_button(
    "Download HTML report",
    data=html,
    file_name=f"playbook_{symbol}.html",
    mime="text/html",
)

try:
    pdf = render_pdf(html)
except Exception as exc:  # weasyprint optional / system deps
    st.info(f"PDF export needs the `report` extra (weasyprint): {exc}")
else:
    st.download_button(
        "Download PDF report",
        data=pdf,
        file_name=f"playbook_{symbol}.pdf",
        mime="application/pdf",
    )

st.components.v1.html(html, height=600, scrolling=True)

page_footer()
