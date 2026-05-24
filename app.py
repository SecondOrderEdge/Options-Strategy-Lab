"""Options Strategy Lab — Streamlit entrypoint (multipage home).

Run with ``streamlit run app.py``. Analytical pages live in ``pages/`` and are
thin shells over ``osl.*`` plus ``osl.viz``; shared UI lives in ``app_lib``.
"""

from __future__ import annotations

import streamlit as st

from app_lib import DISCLAIMER, require_login

st.set_page_config(page_title="Options Strategy Lab", layout="wide")
require_login()

st.title("Options Strategy Lab")
st.caption(DISCLAIMER)

st.markdown("""
**Analytical honesty over visual flash.** Every probability, expected value, and
ranking is labelled with its model assumptions and probability measure
(risk-neutral vs real-world).

Use the sidebar to navigate:

- **Ticker Overview** — spot, returns, IV30, IV rank/percentile.
- **Options Chain** — filterable chain with computed greeks and liquidity flags.
- **IV Surface** — SVI fit per expiry, 3D surface/heatmap, arbitrage checks.
- **Vol Diagnostics** — skew, term structure, volatility cone, IV vs realized.
- **Strategy Generator / Optimizer** — ranked candidates per objective.
- **Payoff & Scenario / Probability Lab** — P&L surfaces and POP by measure.
- **Backtester / Watchlist Screener** — snapshot-driven backtests and screening.
- **Advanced Models** — Heston, jumps, local vol, PCA (experimental).
- **Playbook** — a reproducible, signed per-name report.
- **Assumptions & Disclaimers** — what every model does and does not claim.
""")

st.info(
    "Pick a page from the sidebar to begin. Configure data access in `.env` (see `.env.example`)."
)

st.divider()
st.caption(DISCLAIMER)
