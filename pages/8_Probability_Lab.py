"""Probability Lab: POP across measures, EV ± CI, and the risk-neutral density."""

from __future__ import annotations

import numpy as np
import streamlit as st

from app_lib import (
    build_candidates,
    column_help,
    garch_forecast,
    load_chain,
    load_log_returns,
    page_footer,
    page_header,
    rate_assumptions,
    render_chart,
    settings,
    sidebar_controls,
)
from osl.strategy.metrics import blended_pop, ev_empirical, pop_empirical, pop_garch
from osl.surface.prepare import prepare_smiles
from osl.surface.rnd import normalize, risk_neutral_density
from osl.surface.svi import fit_svi, make_iv_of_strike
from osl.viz.charts import rnd_chart

page_header("Probability Lab")
symbol, provider = sidebar_controls()

with st.sidebar:
    view = st.selectbox("View", ["all", "bullish", "bearish", "neutral", "volatile"], index=0)
    dte_low, dte_high = st.slider("DTE range", 1, 180, (20, 60))
    max_strikes = st.slider("Max strikes per leg", 1, 5, 3)
    go = st.button("Analyze", type="primary")

experimental = settings().enable_experimental

if not (go and symbol):
    st.caption("Set a symbol and filters, then click **Analyze**.")
    page_footer()
    st.stop()

try:
    candidates = build_candidates(provider, symbol, view, dte_low, dte_high, max_strikes)
    log_returns = load_log_returns(provider, symbol)
    chain = load_chain(provider, symbol)
except Exception as exc:  # surface provider/credential failures
    st.error(f"Could not analyze {symbol} via {provider}: {exc}")
    page_footer()
    st.stop()

if not candidates:
    st.warning("No candidates met the filters.")
    page_footer()
    st.stop()


def _label(i: int) -> str:
    c = candidates[i]
    exp = min(leg.expiration for leg in c.strategy.legs).isoformat()
    return f"{c.strategy.name} | {exp} | POP {c.metrics.pop_rn:.0%}"


idx = st.selectbox("Strategy", range(len(candidates)), format_func=_label)
chosen = candidates[idx]
strat = chosen.strategy
m = chosen.metrics

gf = garch_forecast(provider, symbol, "garch", "normal", strat.primary_dte)
p_pop = pop_garch(strat, gf.annualized_vol, n=20_000)
emp_pop = pop_empirical(strat, log_returns, n=20_000)
emp_ev = ev_empirical(strat, log_returns, n=20_000)

st.subheader("Probability of profit — by measure")
rows = [
    ("Delta proxy (RN, approx)", m.pop_delta),
    ("RN lognormal, closed form N(d2)", m.pop_rn),
    ("RN Monte Carlo (GBM)", m.pop_mc),
    ("P-measure (GARCH σ-forecast)", p_pop),
    ("Historical bootstrap (P)", emp_pop),
]
if experimental:
    rows.append(("Blended (RN prior, P likelihood)", blended_pop(m.pop_rn, p_pop)))
st.dataframe(
    {"measure": [r[0] for r in rows], "POP": [round(r[1], 3) for r in rows]},
    use_container_width=True,
    hide_index=True,
    column_config=column_help(["measure", "POP"]),
)
st.caption(
    "How to read: **RN** (risk-neutral) rows are what the *market* charges; **P** "
    "(real-world) rows estimate the *actual* odds from history/GARCH. They legitimately "
    "differ — the gap is roughly the volatility risk premium. The delta proxy overstates POP."
)

c1, c2, c3 = st.columns(3)
c1.metric(
    "EV (RN MC)",
    f"{m.ev_mc.value:,.0f}",
    help=f"Risk-neutral expected P&L (edge vs fair value), Monte Carlo. "
    f"95% CI [{m.ev_mc.ci_low:,.0f}, {m.ev_mc.ci_high:,.0f}].",
)
c2.metric(
    "EV (historical)",
    f"{emp_ev.value:,.0f}",
    help=f"Expected P&L under the real-world return distribution (historical bootstrap). "
    f"95% CI [{emp_ev.ci_low:,.0f}, {emp_ev.ci_high:,.0f}].",
)
c3.metric(
    "Expected shortfall (95%)",
    f"{m.expected_shortfall:,.0f}",
    help="Average loss in the worst 5% of outcomes — a tail-risk companion to EV.",
)
st.caption(f"GARCH(1,1) annualized vol forecast over {gf.horizon_days}d: {gf.annualized_vol:.1%}")

st.subheader("Risk-neutral density")
rate, div = rate_assumptions()
T = strat.primary_T
forward = strat.spot * np.exp((rate - div) * T)
primary_exp = min(leg.expiration for leg in strat.legs)
smiles = prepare_smiles(chain, spot=strat.spot, rate=rate, dividend_yield=div)
smile = next((s for s in smiles if s.expiration == primary_exp), None)

if smile is not None and len(smile) >= 5:
    fit = fit_svi(smile.k, smile.total_variance, weights=smile.weights, T=smile.T)
    strikes, dens = risk_neutral_density(
        make_iv_of_strike(fit.params, forward=forward, T=T),
        spot=strat.spot,
        rate=rate,
        dividend_yield=div,
        T=T,
        k_min=-1.5,
        k_max=1.5,
        n=800,
    )
    atm_iv = float(smile.iv[int(np.argmin(np.abs(smile.k)))])
    ln_strikes, ln_dens = risk_neutral_density(
        lambda k: np.full_like(np.asarray(k, float), atm_iv),
        spot=strat.spot,
        rate=rate,
        dividend_yield=div,
        T=T,
        k_min=-1.5,
        k_max=1.5,
        n=800,
    )
    render_chart(
        rnd_chart(strikes, normalize(strikes, dens), lognormal=normalize(ln_strikes, ln_dens)),
    )
else:
    st.info("Not enough liquid strikes to build a market-implied RND for this expiry.")

with st.expander("Model assumptions"):
    st.markdown(
        "- Each POP row states its **measure** (RN vs P) and model. RN POP is "
        "market/lognormal; the GARCH and bootstrap rows are real-world.\n"
        "- The delta proxy overstates POP; never read it as a true probability.\n"
        "- The blended POP is experimental (enable via `OSL_ENABLE_EXPERIMENTAL`).\n"
        "- The RND is Breeden-Litzenberger on the arbitrage-checked SVI smile; the "
        "lognormal overlay uses flat ATM IV."
    )

page_footer()
