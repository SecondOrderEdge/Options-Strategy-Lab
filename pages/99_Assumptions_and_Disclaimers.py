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

### Current limitations

These features ship today but with caveats worth knowing:

- **Experimental models** (Heston, Merton jumps, Dupire local vol, surface PCA)
  are gated behind `OSL_ENABLE_EXPERIMENTAL` and can be weakly identified on
  sparse chains — read each tab's notes.
- **Surface PCA** shows a synthetic 3-factor demo until a multi-day IV-snapshot
  history accumulates.
- **IV rank / percentile** use realized vol as a stand-in until a daily IV
  history accumulates.
- **Backtests** use synthetic GBM demo data unless real chain snapshots have
  been captured (run the snapshot worker); results are illustrative until then.
- **Risk-free rate** is a flat configured value; FRED Treasury curves and
  per-name dividend yields arrive later.
""")

st.markdown("""
### Glossary

Plain-English definitions of the terms used across the app. Hover the column
headers and metric labels on each page for the same explanations in context.

**Volatility**

- **IV (implied volatility)** — the annualized volatility an option's market price implies under Black-Scholes.
- **IV30** — ATM IV of the expiry nearest 30 days; the market's expected vol over roughly one month.
- **RV (realized volatility)** — how much the underlying actually moved historically (Yang-Zhang by default).
- **IV rank / percentile** — where current vol sits in its 1-year range (rank), or the share of the year it was lower (percentile).
- **IV minus RV (VRP proxy)** — variance-risk-premium proxy; positive means options are priced above realized movement.
- **Term structure (contango / backwardation)** — ATM IV rising / falling as expiry lengthens.
- **25-delta risk reversal** — 25-delta call IV minus put IV; negative = downside put skew.

**Probability & expected value**

- **POP** — probability of profit.
- **Risk-neutral (RN) vs real-world (P)** — RN is market-implied (what options charge); P estimates actual odds from history/GARCH. The gap is roughly the volatility risk premium.
- **EV (expected value)** — expected P&L. RN EV is edge vs fair value; historical EV uses the real-world return distribution.
- **Expected shortfall (ES, 95%)** — average loss in the worst 5% of outcomes (tail risk).
- **Risk-neutral density (RND)** — the distribution of future prices implied by option prices (Breeden-Litzenberger).

**Strategy metrics**

- **Max loss / max profit** — worst / best outcome at expiry; infinity = uncovered (unbounded) risk.
- **Breakeven** — underlying price where P&L is zero at expiry.
- **ROR (return on risk)** — expected value divided by capital at risk.
- **Liquidity score (0-1)** — blend of bid/ask spread, open interest, volume and ATM distance.
- **Credit vs debit** — net premium collected vs paid to open.

**Greeks**

- **Delta** — sensitivity to spot; roughly the chance of finishing in the money.
- **Gamma** — how fast delta changes as spot moves.
- **Theta** — time decay per calendar day.
- **Vega** — P&L for a 1 vol-point change in IV.
- **Rho** — sensitivity to a 1.00 change in the interest rate.

**Surface (SVI)**

- **SVI a / b / rho / m / sigma** — raw-SVI smile parameters: level, wing steepness, skew/tilt, location of the minimum, and ATM curvature.
- **Butterfly / calendar arbitrage** — static no-arbitrage checks; a flagged fit (especially the wings) should not be trusted.

**Advanced models**

- **Heston v0 / kappa / theta / sigma / rho** — current variance, mean-reversion speed, long-run variance, vol-of-vol, and spot/vol correlation.
- **Feller condition (2·kappa·theta vs sigma squared)** — when satisfied, variance stays strictly positive; a violation signals a stressed fit.
- **Merton lambda / mu / delta** — jump intensity per year, average jump size, and jump-size dispersion.

**Backtest statistics**

- **Sharpe / Sortino** — return per unit of total / downside volatility.
- **PSR (probabilistic Sharpe)** — confidence the true Sharpe exceeds zero, adjusting for sample size, skew and fat tails.
- **DSR (deflated Sharpe)** — PSR deflated for the number of strategy variants tried; guards against a lucky backtest.
- **Max drawdown** — the largest peak-to-trough decline in equity.
""")
