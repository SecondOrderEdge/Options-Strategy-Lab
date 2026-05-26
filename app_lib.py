"""Shared Streamlit helpers used by the multipage app.

Keeps pages thin: provider selection, cached data loaders, the freshness
badge, and rate assumptions live here so each page is only ``osl.*`` calls plus
viz. This module is the *only* place pages touch Streamlit caching.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
import streamlit as st

from osl.auth import verify_password
from osl.backtest.config import BacktestConfig, FillModel
from osl.backtest.demo import synthetic_history
from osl.backtest.engine import BacktestResult, run_backtest
from osl.backtest.loader import load_snapshot_days
from osl.backtest.rules import IronCondor1Sigma, LongStraddle, ShortPut16Delta
from osl.config import Settings, get_settings
from osl.data.providers import Freshness, build_provider, freshness_badge
from osl.report.playbook import LegRow, PlaybookData, StrategyRow
from osl.strategy.enumerate import View, enumerate_candidates
from osl.strategy.liquidity import strategy_liquidity
from osl.strategy.metrics import compute_metrics
from osl.strategy.optimizer import Candidate, rank
from osl.strategy.strategies import ChainContext
from osl.surface.prepare import prepare_smiles
from osl.utils.time import now_utc
from osl.volatility.garch import GarchForecast, fit_garch
from osl.volatility.ranks import iv_percentile, iv_rank
from osl.volatility.realized import realized_vol
from osl.volatility.skew import delta_skew_25

DISCLAIMER = "Research and education only — not investment advice."
BADGE_PILL = {Freshness.GREEN: "green", Freshness.AMBER: "amber", Freshness.RED: "red"}

# Dark "terminal" aesthetic. The Streamlit [theme] in .streamlit/config.toml sets
# the palette; this stylesheet layers on the things native theming can't express:
# the serif/mono font pairing, card tiles, uppercase letter-spaced labels,
# gold-outline buttons, styled tabs, a dark sidebar, and the freshness pills.
_THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Spectral:wght@400;500;600&display=swap');

:root {
  --osl-gold: #c9a24b;
  --osl-teal: #3fb6a8;
  --osl-red:  #c0564b;
  --osl-ink:  #0b0c0e;
  --osl-card: #14161b;
  --osl-line: #2a2d34;
  --osl-muted:#8b8e97;
  --osl-text: #e7e3d8;
}

.stApp { background: var(--osl-ink); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1280px; }

h1, h2, h3, h4,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {
  font-family: 'Spectral', Georgia, 'Times New Roman', serif !important;
  font-weight: 500;
  letter-spacing: 0.005em;
  color: var(--osl-text);
}
h1 { font-size: 2.1rem; }

[data-testid="stCaptionContainer"], .stCaption, small {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  color: var(--osl-muted) !important;
  letter-spacing: 0.02em;
}

[data-testid="stWidgetLabel"] label,
[data-testid="stMetricLabel"] {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  font-size: 0.72rem !important;
  color: var(--osl-muted) !important;
}

[data-testid="stMetric"] {
  background: var(--osl-card);
  border: 1px solid var(--osl-line);
  border-radius: 6px;
  padding: 1rem 1.1rem;
}
[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  color: var(--osl-gold);
  font-weight: 600;
}

[data-baseweb="tab-list"] { border-bottom: 1px solid var(--osl-line); gap: 1.5rem; }
[data-baseweb="tab"] {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 0.74rem;
  color: var(--osl-muted);
  background: transparent;
}
[data-baseweb="tab"][aria-selected="true"] { color: var(--osl-gold); }
[data-baseweb="tab-highlight"] { background-color: var(--osl-gold) !important; }

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  background: transparent;
  border: 1px solid var(--osl-gold);
  color: var(--osl-gold);
  border-radius: 4px;
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.78rem;
  font-weight: 500;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  background: var(--osl-gold);
  color: var(--osl-ink);
  border-color: var(--osl-gold);
}

[data-testid="stSidebar"] {
  background: #0e1014;
  border-right: 1px solid var(--osl-line);
}

[data-testid="stExpander"],
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--osl-line) !important;
  border-radius: 6px;
  background: rgba(255,255,255,0.012);
}

hr { border-color: var(--osl-line); }

.osl-pill {
  display: inline-block;
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 0.12rem 0.55rem;
  border-radius: 3px;
  border: 1px solid currentColor;
}
.osl-pill--green { color: var(--osl-teal); }
.osl-pill--amber { color: var(--osl-gold); }
.osl-pill--red   { color: var(--osl-red); }
.osl-badge-meta {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  color: var(--osl-muted);
  font-size: 0.78rem;
  margin-left: 0.4rem;
}
"""


def inject_theme() -> None:
    """Inject the dark/gold stylesheet. Call once per page run (idempotent per run)."""
    st.markdown(f"<style>{_THEME_CSS}</style>", unsafe_allow_html=True)


def render_chart(fig: Any, **kwargs: Any) -> None:
    """Render a Plotly figure using its own dark template instead of Streamlit's.

    Passing ``theme=None`` lets the template defined in ``osl.viz.charts`` (dark
    background, gold/teal palette) drive the look; Streamlit's default
    ``theme="streamlit"`` would otherwise override it.
    """
    kwargs.setdefault("use_container_width", True)
    st.plotly_chart(fig, theme=None, **kwargs)


def armed_button(
    label: str,
    *,
    key: str,
    signature: object,
    **button_kwargs: Any,
) -> bool:
    """A "latched" action button that survives subsequent slider interactions.

    A plain ``st.button`` returns ``True`` only on the run immediately after
    the click, so any later slider/select tick reruns the script with the
    button ``False`` — the page's ``if not go: st.stop()`` gate then fires and
    the outputs disappear until you click again. This wraps the button to
    latch the click in ``st.session_state`` under ``key``. The page stays
    armed across reruns until the ``signature`` changes (e.g. you edit the
    symbol), at which point the gate re-arms and demands another click.
    """
    clicked = st.button(label, **button_kwargs)
    state_key = f"_armed::{key}"
    if clicked:
        st.session_state[state_key] = signature
    return st.session_state.get(state_key) == signature


# Plain-English tooltips for dataframe columns across the app, keyed by column
# name. Shared keys (vega, liquidity, expiration) carry the same meaning wherever
# they appear, so one table is enough.
COLUMN_HELP: dict[str, str] = {
    # --- strategy candidates ---
    "strategy": "Strategy structure (e.g. vertical, iron condor, straddle).",
    "expiry": "Expiration date of the (nearest) leg.",
    "net": "Net cash to open: positive = debit paid, negative = credit received.",
    "credit?": "True if the trade collects net premium at entry.",
    "POP (RN)": "Risk-neutral probability of finishing profitable (lognormal at the fitted "
    "IV). A market-implied odd, not a real-world forecast.",
    "EV": "Risk-neutral expected P&L from Monte Carlo — edge vs fair value, not a "
    "real-world profit forecast.",
    "max_profit": "Largest possible profit at expiry; ∞ if unbounded.",
    "max_loss": "Largest possible loss at expiry; ∞ for uncovered (naked) risk.",
    "ES(95%)": "Expected shortfall: average loss in the worst 5% of outcomes (tail risk).",
    "ROR": "Return on risk: expected value divided by capital at risk.",
    "theta/day": "Time decay per calendar day, in dollars (positive = collects decay).",
    "vega": "P&L for a 1 vol-point rise in implied vol (positive = long volatility).",
    "liquidity": "0-1 blend of bid/ask spread, open interest, volume and ATM distance "
    "(higher = more tradable).",
    # --- options chain ---
    "expiration": "Option expiration date.",
    "right": "C = call, P = put.",
    "strike": "Strike price.",
    "bid": "Best bid (what buyers will pay).",
    "ask": "Best ask (what sellers want).",
    "mid": "Midpoint of bid and ask.",
    "volume": "Contracts traded today.",
    "open_interest": "Open contracts outstanding — a depth/liquidity gauge.",
    "iv": "Implied volatility (annualized) backed out from the option's price.",
    "delta": "∂price/∂spot — roughly the chance of finishing ITM and the hedge ratio.",
    "gamma": "∂delta/∂spot — how fast delta moves as spot moves.",
    "theta": "Time decay per day, in dollars per share.",
    "spread_pct": "Bid/ask spread as a fraction of mid (lower = tighter, cheaper to trade).",
    "zero_bid": "True if there is no bid — you can't sell it.",
    "wide_spread": "True if the spread exceeds 10% of mid.",
    "is_nonstandard": "True for adjusted/non-standard contracts (e.g. post-split deliverable).",
    # --- SVI surface fit ---
    "T": "Time to expiry, in years.",
    "n": "Number of liquid quotes used in the fit.",
    "rmse_vol_pts": "Fit error in vol points (model IV vs market IV).",
    "butterfly_free": "True if the smile has no static (butterfly) arbitrage.",
    "a": "SVI level — overall height of the total-variance curve.",
    "b": "SVI angle — steepness of the two wings.",
    "rho": "SVI rotation — skew/asymmetry (negative = downside-heavy).",
    "m": "SVI shift — horizontal position of the smile's minimum (in log-moneyness).",
    "sigma": "SVI smoothness — how rounded the ATM bottom of the smile is.",
    # --- probability-of-profit table ---
    "measure": "Probability measure & model: RN = market-implied, P = real-world.",
    "POP": "Probability of profit under that measure.",
    # --- watchlist screener ---
    "symbol": "Ticker.",
    "spot": "Underlying price.",
    "IV30": "ATM implied vol at ~30 DTE (annualized).",
    "RV30": "30-day realized (historical) volatility, Yang-Zhang estimator.",
    "IV-RV": "IV30 minus RV30 — variance-risk-premium proxy (positive = options look rich).",
    "25dRR": "25-delta risk reversal (call IV − put IV); negative = downside put skew.",
    "RVrank": "Where current realized vol sits in its 1-year range (0-1).",
    "RVpct": "Percentile of current realized vol over the past year.",
    "top_strategy": "Best candidate for this name by the chosen objective.",
    "top_POP": "Risk-neutral POP of the top strategy.",
    "top_EV": "RN expected value of the top strategy.",
    # --- backtest trades ---
    "entry": "Trade entry date.",
    "exit": "Trade exit date.",
    "reason": "Why the position closed (expiry, stop, target, etc.).",
    "P&L": "Realized profit/loss for the trade, in dollars.",
}


def column_help(columns: object) -> dict[str, Any]:
    """Build a Streamlit ``column_config`` of header tooltips for known columns.

    Pass anything iterable of column names (a DataFrame's ``.columns`` or a list).
    Unknown columns are left untouched.
    """
    return {
        str(c): st.column_config.Column(help=COLUMN_HELP[str(c)])
        for c in cast("Any", columns)
        if str(c) in COLUMN_HELP
    }


def skew_reading(rr25: float) -> str:
    """One-line read of the 25-delta risk reversal (the skew sign)."""
    rr_pts = rr25 * 100
    if rr25 < -0.005:
        return (
            f"Downside-heavy skew (25-delta risk reversal {rr_pts:+.1f} pts): OTM puts are bid "
            "versus calls — the richness that put credit spreads, put ratios and risk "
            "reversals are built to harvest."
        )
    if rr25 > 0.005:
        return (
            f"Upside-heavy skew (25-delta risk reversal {rr_pts:+.1f} pts, atypical for "
            "equities): calls are richer than puts; the mirror-image structures apply."
        )
    return (
        f"Balanced skew (25-delta risk reversal {rr_pts:+.1f} pts): little directional vol premium."
    )


def trade_reading(
    *,
    iv30: float,
    rv30: float,
    rv_rank: float | None = None,
    rr25: float | None = None,
    term_shape: str | None = None,
) -> str:
    """Plain-English "how to read this for trades" framing from live vol signals.

    Educational regime framing only: it names the structures a regime is
    *consistent with* (premium-selling when vol is rich, long-gamma when cheap,
    skew-harvesting structures for the skew sign), never a recommendation to
    enter a trade. Returns a markdown bullet list.
    """
    lines: list[str] = []
    if not (np.isnan(iv30) or np.isnan(rv30)):
        vrp = (iv30 - rv30) * 100
        rank_txt = (
            f", realized vol in the {rv_rank:.0%} percentile of its year"
            if rv_rank is not None and not np.isnan(rv_rank)
            else ""
        )
        if vrp > 0.5:
            lines.append(
                f"**Premium looks rich** — IV30 {iv30:.0%} is above realized {rv30:.0%} "
                f"(+{vrp:.1f} vol pts{rank_txt}). This regime favors *defined-risk premium "
                "selling* (credit spreads, iron condors): you collect the richness with a "
                "capped loss."
            )
        elif vrp < -0.5:
            lines.append(
                f"**Premium looks cheap** — IV30 {iv30:.0%} is below realized {rv30:.0%} "
                f"({vrp:.1f} vol pts{rank_txt}). This regime favors *premium buying / long "
                "gamma* (debit spreads, calendars, long straddles): you pay theta but own "
                "convexity."
            )
        else:
            lines.append(
                f"**Vol looks fair** — IV30 {iv30:.0%} is in line with realized {rv30:.0%}; "
                "no strong premium edge, so let direction and skew drive the structure."
            )
    if rr25 is not None and not np.isnan(rr25):
        lines.append(skew_reading(rr25))
    if term_shape == "contango":
        lines.append(
            "**Term structure in contango** (longer-dated IV higher) — a typical calm regime; "
            "calendars and diagonals lean on the cheaper near-dated leg."
        )
    elif term_shape == "backwardation":
        lines.append(
            "**Term structure in backwardation** (near-dated IV higher) — often an event or "
            "stress signal; near-dated premium is elevated."
        )
    return "\n".join(f"- {ln}" for ln in lines)


def _load_streamlit_secrets_into_env() -> None:
    """Bridge Streamlit Cloud secrets into env vars so Settings (OSL_*) reads them.

    Streamlit Cloud exposes config via ``st.secrets``; pydantic-settings reads
    environment variables. Copying OSL_-prefixed secrets across (without
    overriding an already-set env var) makes both deployment models work.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:  # no secrets configured (e.g. local run) — fine
        return
    for key, value in secrets.items():
        if isinstance(key, str) and isinstance(value, str):
            os.environ.setdefault(key, value)


_load_streamlit_secrets_into_env()


def settings() -> Settings:
    return get_settings()


def sidebar_controls(default_symbol: str = "SPY") -> tuple[str, str]:
    """Render the shared sidebar; return (symbol, provider_name).

    The symbol and provider are stored in ``st.session_state`` under stable
    keys, so a ticker set on one page carries across the whole app for the
    session. Seed defaults only on first use (don't clobber the user's choice).
    """
    cfg = get_settings()
    st.session_state.setdefault("symbol", default_symbol)
    st.session_state.setdefault("provider", cfg.default_provider)
    with st.sidebar:
        st.header("Data")
        symbol = st.text_input("Symbol", key="symbol").strip().upper()
        provider = st.radio("Provider", options=["schwab", "yfinance"], key="provider")
    return symbol, provider


def rate_assumptions() -> tuple[float, float]:
    """Return (risk_free_rate, dividend_yield) used across analytics.

    M1 uses the configured flat risk-free rate and a zero dividend yield; FRED
    curves and per-name dividends arrive in later milestones.
    """
    return get_settings().risk_free_flat_rate, 0.0


@st.cache_data(ttl=60, show_spinner=False)
def load_chain(provider_name: str, symbol: str) -> pd.DataFrame:
    provider = build_provider(provider_name, get_settings())
    return provider.get_option_chain(symbol)


@st.cache_data(ttl=15, show_spinner=False)
def load_underlying_dict(provider_name: str, symbol: str) -> dict[str, object]:
    provider = build_provider(provider_name, get_settings())
    q = provider.get_underlying(symbol)
    return {
        "symbol": q.symbol,
        "last": q.last,
        "mark": q.mark,
        "quote_time": q.quote_time,
        "is_delayed": q.is_delayed,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(provider_name: str, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
    provider = build_provider(provider_name, get_settings())
    end = date.today()
    start = end - timedelta(days=lookback_days)
    return provider.get_history(symbol, start=start, end=end)


def render_badge(provider_name: str, *, is_delayed: bool, quote_time: pd.Timestamp) -> None:
    badge = freshness_badge(provider_name, is_delayed=is_delayed, quote_time=quote_time)
    cls = BADGE_PILL[badge]
    ts = f"{pd.Timestamp(quote_time):%Y-%m-%d %H:%M:%S %Z}"
    st.markdown(
        f'<span class="osl-pill osl-pill--{cls}">{badge.value}</span>'
        f'<span class="osl-badge-meta">{provider_name} @ {ts}</span>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=120, show_spinner="Enumerating and scoring candidates…")
def build_candidates(
    provider_name: str,
    symbol: str,
    view: str,
    dte_low: int,
    dte_high: int,
    max_strikes: int,
    n_mc: int = 6000,
) -> list[Candidate]:
    """Enumerate strategies for the view and attach metrics + liquidity."""
    chain = load_chain(provider_name, symbol)
    under = load_underlying_dict(provider_name, symbol)
    spot = float(under["mark"] or under["last"])
    rate, div = rate_assumptions()
    ctx = ChainContext(chain, symbol, spot, rate, div)
    strategies = enumerate_candidates(
        ctx,
        view=cast(View, view),
        dte_range=(dte_low, dte_high),
        max_strikes_per_leg=max_strikes,
    )
    out: list[Candidate] = []
    for strat in strategies:
        out.append(
            Candidate(
                strategy=strat,
                metrics=compute_metrics(strat, n_mc=n_mc),
                liquidity=strategy_liquidity(chain, strat),
            )
        )
    return out


def candidates_table(candidates: list[Candidate]) -> pd.DataFrame:
    """Flatten candidates into a display DataFrame (one row per strategy)."""
    rows = []
    for c in candidates:
        m = c.metrics
        rows.append(
            {
                "strategy": c.strategy.name,
                "expiry": min(leg.expiration for leg in c.strategy.legs).isoformat(),
                "net": round(m.net_debit, 2),
                "credit?": m.is_credit,
                "POP (RN)": round(m.pop_rn, 3),
                "EV": round(m.ev_mc.value, 2),
                "max_profit": "∞" if m.profit_unbounded else round(m.max_profit, 2),
                "max_loss": "∞ (uncovered)" if m.loss_unbounded else round(m.max_loss, 2),
                "ES(95%)": round(m.expected_shortfall, 2),
                "ROR": round(m.return_on_risk, 3),
                "theta/day": round(m.greeks["theta"], 2),
                "vega": round(m.greeks["vega"], 2),
                "liquidity": round(c.liquidity.score, 2),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def load_log_returns(provider_name: str, symbol: str) -> pd.Series:
    hist = load_history(provider_name, symbol)
    return cast(pd.Series, np.log(hist["close"]).diff().dropna())


@st.cache_data(ttl=3600, show_spinner="Fitting GARCH…")
def garch_forecast(
    provider_name: str, symbol: str, model: str, dist: str, horizon: int
) -> GarchForecast:
    return fit_garch(
        load_log_returns(provider_name, symbol),
        model=cast("Any", model),
        dist=cast("Any", dist),
        horizon=horizon,
    )


@st.cache_data(ttl=300, show_spinner=False)
def screen_symbol(provider_name: str, symbol: str, objective: str) -> dict[str, object]:
    """Per-name screener row: IVR/IVP proxy, 25Δ RR, VRP, and the top strategy."""
    chain = load_chain(provider_name, symbol)
    under = load_underlying_dict(provider_name, symbol)
    spot = float(under["mark"] or under["last"])
    rate, div = rate_assumptions()

    hist = load_history(provider_name, symbol)
    rv30_series = realized_vol(hist, method="yz", window=30).dropna()
    rv30 = float(rv30_series.iloc[-1]) if not rv30_series.empty else float("nan")
    rv20 = realized_vol(hist, method="yz", window=20).dropna()

    smiles = prepare_smiles(chain, spot=spot, rate=rate, dividend_yield=div)
    iv30 = float("nan")
    rr25 = float("nan")
    if smiles:
        nearest = min(smiles, key=lambda s: abs(s.T - 30 / 365))
        ds = delta_skew_25(nearest)
        iv30, rr25 = ds.atm_iv, ds.risk_reversal_25

    ctx = ChainContext(chain, symbol, spot, rate, div)
    candidates = [
        Candidate(strat, compute_metrics(strat, n_mc=4000), strategy_liquidity(chain, strat))
        for strat in enumerate_candidates(ctx, view="all", dte_range=(20, 60))
    ]
    best = rank(candidates, objective)[0] if candidates else None

    return {
        "symbol": symbol,
        "spot": round(spot, 2),
        "IV30": None if np.isnan(iv30) else round(iv30, 4),
        "RV30": None if np.isnan(rv30) else round(rv30, 4),
        "IV-RV": None if (np.isnan(iv30) or np.isnan(rv30)) else round(iv30 - rv30, 4),
        "25dRR": None if np.isnan(rr25) else round(rr25, 4),
        "RVrank": None if rv20.empty else round(iv_rank(rv20), 3),
        "RVpct": None if rv20.empty else round(iv_percentile(rv20), 3),
        "top_strategy": best.strategy.name if best else None,
        "top_POP": round(best.metrics.pop_rn, 3) if best else None,
        "top_EV": round(best.metrics.ev_mc.value, 2) if best else None,
        "liquidity": round(best.liquidity.score, 2) if best else None,
    }


BACKTEST_SYSTEMS = {
    "45D 16-delta Short Put": ShortPut16Delta,
    "45D 1-sigma Iron Condor": IronCondor1Sigma,
    "30D ATM Long Straddle": LongStraddle,
}


@st.cache_data(ttl=300, show_spinner="Running backtest…")
def run_system_backtest(
    provider_name: str,
    symbol: str,
    system_name: str,
    *,
    fill_model: str,
    spread_penalty: float,
    commission: float,
    capital: float,
    use_demo: bool,
) -> BacktestResult:
    """Run a reference system over store snapshots (or synthetic demo data)."""
    if use_demo:
        days = synthetic_history(symbol or "DEMO", n_days=240, seed=7)
    else:
        days = load_snapshot_days(get_settings().snapshot_root, symbol)
    config = BacktestConfig(
        capital=capital,
        fill_model=cast(FillModel, fill_model),
        spread_penalty=spread_penalty,
        commission_per_contract=commission,
    )
    return run_backtest(days, BACKTEST_SYSTEMS[system_name](), config)


def require_login() -> None:
    """Single-user password gate. No-op when OSL_APP_PASSWORD_HASH is unset."""
    cfg = get_settings()
    if cfg.app_password_hash is None:
        return
    if st.session_state.get("_authed"):
        return
    pw = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if verify_password(pw, cfg.app_password_hash.get_secret_value()):
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    if not st.session_state.get("_authed"):
        st.stop()


_PLAYBOOK_OBJECTIVES = ("ev_per_risk", "pop_capped", "theta_per_risk")


def build_playbook_data(
    provider_name: str,
    symbol: str,
    *,
    view: str = "all",
    dte_low: int = 20,
    dte_high: int = 60,
    max_strikes: int = 3,
) -> PlaybookData:
    """Assemble the per-name playbook report data from live analytics."""
    candidates = build_candidates(provider_name, symbol, view, dte_low, dte_high, max_strikes)
    under = load_underlying_dict(provider_name, symbol)
    chain = load_chain(provider_name, symbol)
    history = load_history(provider_name, symbol)
    spot = float(under["mark"] or under["last"])
    rate, div = rate_assumptions()

    smiles = prepare_smiles(chain, spot=spot, rate=rate, dividend_yield=div)
    iv30 = float("nan")
    if smiles:
        nearest = min(smiles, key=lambda s: abs(s.T - 30 / 365))
        iv30 = float(nearest.iv[int(np.argmin(np.abs(nearest.k)))])
    rv20 = (
        realized_vol(history, method="yz", window=20).dropna()
        if not history.empty
        else pd.Series(dtype=float)
    )
    ivr = iv_rank(rv20) if not rv20.empty else float("nan")
    ivp = iv_percentile(rv20) if not rv20.empty else float("nan")

    rows: list[StrategyRow] = []
    for obj in _PLAYBOOK_OBJECTIVES:
        if not candidates:
            break
        best = rank(candidates, obj)[0]
        m = best.metrics
        legs = tuple(
            LegRow(
                side="Sell" if leg.sign < 0 else "Buy",
                quantity=int(leg.quantity),
                right=leg.right,
                strike=float(leg.strike),
                expiry=leg.expiration.isoformat(),
                premium=float(leg.premium),
                iv=float(leg.iv),
            )
            for leg in best.strategy.legs
        )
        rows.append(
            StrategyRow(
                name=best.strategy.name,
                objective=obj,
                expiry=min(leg.expiration for leg in best.strategy.legs).isoformat(),
                pop_rn=m.pop_rn,
                ev=m.ev_mc.value,
                expected_shortfall=m.expected_shortfall,
                max_loss=m.max_loss,
                loss_unbounded=m.loss_unbounded,
                liquidity=best.liquidity.score,
                breakevens=m.breakevens,
                net_debit=float(best.strategy.net_debit),
                legs=legs,
            )
        )

    return PlaybookData(
        symbol=symbol,
        spot=spot,
        as_of=now_utc(),
        iv30=iv30,
        iv_rank=ivr,
        iv_percentile=ivp,
        surface_note=f"{len(smiles)} expiries fit; "
        + ("no butterfly arbitrage flagged." if smiles else "no liquid smile."),
        strategies=rows,
        assumptions=[
            "POP is risk-neutral lognormal; the delta proxy overstates it.",
            "EV is risk-neutral Monte Carlo (edge vs fair value), shown with a tail-loss (ES) companion.",
            "IV rank/percentile use realized vol as a stand-in until IV history accumulates.",
            "Liquidity blends spread, open interest, volume, and ATM distance.",
            "Research and education only — not investment advice.",
        ],
    )


def page_header(title: str) -> None:
    inject_theme()
    st.title(title)
    st.caption(DISCLAIMER)


def page_footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
