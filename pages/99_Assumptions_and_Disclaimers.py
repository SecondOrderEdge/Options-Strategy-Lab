"""Assumptions & Disclaimers — what every model does and does not claim."""

from __future__ import annotations

import streamlit as st

from app_lib import page_header

page_header("Assumptions & Disclaimers")

st.markdown("""
### Scope

Options Strategy Lab is a **research and education** tool. Nothing here is
investment advice. Every output is traceable to a model and a data snapshot.

### Pricing & greeks

- **Black-Scholes-Merton** (European, continuous dividend yield) underlies all
  greeks and IV inversion. Assumptions: constant volatility, no jumps,
  lognormal returns — violated near earnings and in the wings.
- **theta** is per calendar day; **vega** per 1 vol point; **rho** per 1.00 rate.
- Implied volatility uses Jaeckel's "Let's Be Rational" with a bisection
  fallback.

### Volatility surface

- **Raw SVI** per expiry (Gatheral), vega-weighted least squares.
- No-arbitrage is checked two ways: **butterfly** via Gatheral's g(k) and
  **calendar** via monotonic total variance (Gatheral & Jacquier 2014). Flagged
  violations mean the fit — especially the wings — should not be trusted.
- A monotone variance spline is the fallback when SVI fails; it carries **no**
  arbitrage guarantee.

### Probability (risk-neutral vs real-world)

- The **risk-neutral density** (Breeden-Litzenberger) is implied by option
  prices, not a forecast of realized outcomes.
- ATM IV term structure and skew are risk-neutral, model-implied quantities.
- Realized-vol estimators (Yang-Zhang default) describe the **past**, under a
  stationarity assumption that does not hold across regime changes.

### Realized volatility

- Estimators: close-to-close, Parkinson, Garman-Klass, Rogers-Satchell,
  Yang-Zhang. Annualized at 252 trading days.
- IV−RV shown on Vol Diagnostics is a **variance-risk-premium proxy**, not a
  variance-swap replication.

### Data

- Schwab (primary) and yfinance (fallback). yfinance is delayed and supplies no
  greeks; those are computed locally. A freshness badge (GREEN/AMBER/RED) shows
  data age and source on each page.
- Risk-free rate is a flat configured value in M1; FRED curves and per-name
  dividends arrive later.

### Not yet built (later milestones)

Strategy generation/optimization, payoff & scenario analysis, the probability
lab, GARCH forecasts, PCA/Heston, and the snapshot-driven backtester.
""")
