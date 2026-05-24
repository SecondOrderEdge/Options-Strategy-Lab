"""Backtester: snapshot-driven reference systems with DSR alongside Sharpe."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_lib import (
    BACKTEST_SYSTEMS,
    column_help,
    page_footer,
    page_header,
    render_chart,
    run_system_backtest,
    sidebar_controls,
)
from osl.backtest.metrics import (
    deflated_sharpe_ratio,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)
from osl.viz.charts import drawdown_chart, equity_curve_chart

page_header("Backtester")
symbol, provider = sidebar_controls()

with st.sidebar:
    system_name = st.selectbox("System", list(BACKTEST_SYSTEMS))
    fill_model = st.selectbox("Fill model", ["mid", "mid_penalty", "worst", "spread_frac"], index=1)
    spread_penalty = st.slider("Spread penalty", 0.0, 1.0, 0.25)
    commission = st.number_input("Commission / contract", min_value=0.0, value=0.65, step=0.05)
    capital = st.number_input("Capital", min_value=1000, value=100_000, step=10_000)
    n_trials = st.number_input("Parameter trials tried (for DSR)", min_value=1, value=20, step=1)
    use_demo = st.checkbox("Use synthetic demo data", value=True)
    go = st.button("Run backtest", type="primary")

if use_demo:
    st.warning(
        "**Synthetic demo data** — GBM path priced at flat IV, not market data. "
        "Real backtests need accumulated chain snapshots (run the snapshot worker)."
    )

if not (go and symbol):
    st.caption("Set a symbol and system, then click **Run backtest**.")
    page_footer()
    st.stop()

try:
    result = run_system_backtest(
        provider,
        symbol,
        system_name,
        fill_model=fill_model,
        spread_penalty=spread_penalty,
        commission=commission,
        capital=float(capital),
        use_demo=use_demo,
    )
except Exception as exc:  # surface provider/data failures
    st.error(f"Backtest failed for {symbol}: {exc}")
    page_footer()
    st.stop()

if result.equity.empty:
    st.warning(
        "No snapshot history found for this symbol. Enable **synthetic demo data** "
        "or accumulate snapshots first."
    )
    page_footer()
    st.stop()

s = result.stats
returns = result.returns.to_numpy(dtype=float)
sr = per_period_sharpe(returns)
skew = float(pd.Series(returns).skew()) if returns.size > 2 else 0.0
kurt = float(pd.Series(returns).kurt()) + 3.0 if returns.size > 3 else 3.0  # Pearson
psr = probabilistic_sharpe_ratio(sr, returns.size, skew=skew, kurt=kurt)
# Deflate against the number of parameter trials, using the SR-estimator variance
# as an approximate trial dispersion.
sr_var = max((1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / max(returns.size - 1, 1), 1e-9)
dsr = deflated_sharpe_ratio(
    sr, returns.size, skew=skew, kurt=kurt, sr_variance=sr_var, n_trials=int(n_trials)
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final equity", f"{s['final_equity']:,.0f}", help="Ending account value.")
c2.metric("Total return", f"{s['total_return']:.1%}", help="Final equity vs starting capital.")
c3.metric(
    "Sharpe (ann.)",
    f"{s['sharpe']:.2f}",
    help="Annualized return per unit of total volatility (>1 is good). Easily inflated "
    "by curve-fitting — read it next to DSR.",
)
c4.metric(
    "Max drawdown",
    f"{s['max_drawdown']:.1%}",
    help="Largest peak-to-trough equity decline — the worst losing stretch.",
)

c5, c6, c7, c8 = st.columns(4)
c5.metric(
    "Sortino",
    f"{s['sortino']:.2f}",
    help="Like Sharpe but penalizes only downside volatility.",
)
c6.metric(
    "PSR (vs 0)",
    f"{psr:.1%}",
    help="Probabilistic Sharpe: confidence the true Sharpe exceeds 0, given sample "
    "size, skew and fat tails.",
)
c7.metric(
    "DSR",
    f"{dsr:.1%}",
    help=f"Deflated Sharpe: PSR after deflating for the {int(n_trials)} strategy "
    "variants tried — guards against picking a lucky backtest.",
)
c8.metric("Trades", f"{int(s['n_trades'])}", help="Number of closed trades in the run.")
if dsr > psr:  # guard: DSR must not exceed PSR
    st.caption("note: DSR ≤ PSR by construction when trials > 1.")
st.caption(f"Win rate {s['win_rate']:.0%} · profit factor {s['profit_factor']:.2f}")

render_chart(equity_curve_chart(result.equity))
render_chart(drawdown_chart(result.equity))

st.subheader("Trades")
if result.trades:
    trades_df = pd.DataFrame(
        [
            {
                "entry": t.entry_date.isoformat(),
                "exit": t.exit_date.isoformat(),
                "reason": t.reason,
                "P&L": round(t.pnl, 2),
            }
            for t in result.trades
        ]
    )
    st.dataframe(
        trades_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_help(trades_df.columns),
    )

with st.expander("Model assumptions & bias controls"):
    st.markdown(
        "- **No look-ahead**: each day's decisions use only data stamped at or "
        "before the snapshot time; the engine raises if a chain contains future data.\n"
        "- **Fill model** governs slippage: `mid` (optimistic) → `worst` "
        "(pay ask / take bid). The default `mid_penalty` pays a fraction of the "
        "half-spread; switching to `worst` monotonically worsens a credit system.\n"
        "- **DSR is shown beside Sharpe** and deflates for the number of trials "
        "(Bailey & López de Prado 2014); never read Sharpe alone for a swept strategy.\n"
        "- One concurrent position per system; commissions per contract; "
        "European cash settlement at expiry. Survivorship is N/A for a single name."
    )

page_footer()
