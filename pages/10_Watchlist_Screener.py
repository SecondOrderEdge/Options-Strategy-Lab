"""Watchlist Screener: per-name IVR/VRP/skew and the top strategy."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_lib import armed_button, column_help, page_footer, page_header, screen_symbol
from osl.config import get_settings
from osl.strategy.optimizer import OBJECTIVES

page_header("Watchlist Screener")

cfg = get_settings()
with st.sidebar:
    st.header("Universe")
    raw = st.text_area("Symbols (comma/space separated)", value="SPY QQQ IWM")
    provider = st.radio(
        "Provider",
        options=["schwab", "yfinance"],
        index=0 if cfg.default_provider == "schwab" else 1,
    )
    objective = st.selectbox("Top-strategy objective", list(OBJECTIVES), index=2)
    symbols = sorted({s.strip().upper() for s in raw.replace(",", " ").split() if s.strip()})
    armed = armed_button(
        "Screen",
        key="watchlist_screener",
        signature=(tuple(symbols), objective, provider),
        type="primary",
    )

if not (armed and symbols):
    st.caption("Enter symbols and click **Screen**.")
    page_footer()
    st.stop()

rows: list[dict[str, object]] = []
errors: list[str] = []
progress = st.progress(0.0)
for i, sym in enumerate(symbols, start=1):
    try:
        rows.append(screen_symbol(provider, sym, objective))
    except Exception as exc:  # one bad symbol shouldn't sink the screen
        errors.append(f"{sym}: {exc}")
    progress.progress(i / len(symbols))
progress.empty()

if rows:
    screen_df = pd.DataFrame(rows)
    st.caption(
        "Hover any column header for what it means. **IV-RV** > 0 = options rich "
        "vs realized; **25dRR** < 0 = downside put skew."
    )
    st.dataframe(
        screen_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_help(screen_df.columns),
    )
else:
    st.warning("No symbols could be screened.")

if errors:
    with st.expander(f"{len(errors)} symbol(s) failed"):
        for e in errors:
            st.text(e)

with st.expander("Model assumptions & warnings"):
    st.markdown(
        "- **IV30/RV30** are ATM IV (from the ~30-DTE SVI smile) and 30-day "
        "Yang-Zhang realized vol; **IV−RV** is a variance-risk-premium proxy.\n"
        "- **25dRR** is the 25Δ risk reversal (positive = downside put skew).\n"
        "- **RVrank/RVpct** use realized vol as a stand-in until a daily IV "
        "history accumulates (M5).\n"
        "- The top strategy is the best candidate by the chosen objective; check "
        "its liquidity before trading. Capped objectives never surface uncovered risk."
    )

page_footer()
