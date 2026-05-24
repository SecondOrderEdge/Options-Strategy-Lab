# Options Strategy Lab

A strictly layered Python library and Streamlit UI for options analytics, built
around one principle: **analytical honesty over visual flash**. Every metric
carries its model assumptions, probability measure (risk-neutral vs real-world),
and data provenance.

> Research and education only — not investment advice. All models have explicit
> assumptions; read the assumptions before interpreting any output.

## Status: M7 — Institutional polish

The credibility milestone:

- **`osl.report`** — a reproducible per-name **Options Playbook** as HTML (and
  PDF via the optional `report` extra). Each report carries a provenance footer:
  git revision, UTC timestamp, and a content **digest** stable across runs for
  identical inputs. Page 12 offers HTML/PDF downloads.
- **`osl.data.tradier` / `osl.data.marketdata`** — read-only chain adapters to
  the canonical schema (offline-tested parsers; injectable HTTP). Selectable via
  `build_provider` and `OSL_TRADIER_TOKEN` / `OSL_MARKETDATA_TOKEN`.
- **`osl.auth`** — single-user password gate (`OSL_APP_PASSWORD_HASH`), enforced
  by `require_login()` in the app.
- **`osl.bench`** + `.github/workflows/nightly.yml` — a benchmark suite and a
  nightly full-test + benchmark CI job.

### M6 — Advanced models (gated)

Experimental research models behind `OSL_ENABLE_EXPERIMENTAL`:

- **`osl.models.heston`** — QuantLib Heston pricing + Levenberg-Marquardt
  calibration to a smile (collapses to BSM as vol-of-vol -> 0).
- **`osl.models.jump_diffusion`** — Merton jump-diffusion closed form
  (collapses to BSM as the jump intensity -> 0; fattens tails otherwise).
- **`osl.models.local_vol`** — Dupire local-vol surface via QuantLib (read-only).
- **`osl.surface.pca`** — PCA of accumulated surface changes into level / slope /
  curvature factors (Cont & da Fonseca 2002).
- **Page 11** — Advanced Models (gated): Heston fit, Merton overlay, Dupire
  heatmap, and a surface-PCA demo.

QuantLib is now a core dependency.

### Snapshot worker

Captures the chain history the backtester runs on:

- **`osl.data.snapshot_worker`** — pulls each watchlist symbol's chain +
  underlying through a provider and writes a Hive-partitioned Parquet snapshot.
  Per-symbol failures are isolated; non-trading days are skipped.
- **CLI** — `python -m osl.data.snapshot_worker --once` (for a cron / GitHub
  Action) or `--daemon` (APScheduler at 09:35 / 12:30 / 15:55 ET).
- **`OSL_WATCHLIST`** setting (comma-separated) selects the symbols.
- **`.github/workflows/snapshot.yml`** — a cron template (UTC times inside RTH
  for both EST/EDT) that captures and uploads the store as an artifact.

### M5 — Backtester

Adds a snapshot-driven backtester with overfitting-aware statistics:

- **`osl.backtest.engine`** — single-position-per-system event loop that refuses
  look-ahead (chains stamped after the decision time), settles at expiry, marks
  to market, and replays deterministically.
- **`osl.backtest.rules`** — entry/exit systems incl. three references: 45-DTE
  16Δ short put (close 21 DTE), 45-DTE 1σ iron condor (50% credit or 21 DTE),
  30-DTE ATM long straddle (100% gain or 14 DTE).
- **`osl.backtest.fills`** — mid / mid_penalty / worst / spread_frac fill models.
- **`osl.backtest.metrics`** — Sharpe, Sortino, Calmar, max drawdown, VaR/ES,
  trade stats, **PSR**, **DSR** (deflated for the number of trials), and **PBO**
  via combinatorially symmetric cross-validation.
- **`osl.backtest.loader` / `osl.backtest.demo`** — read the Parquet snapshot
  store, or generate synthetic demo history.
- **Page 9** — Backtester: equity/drawdown curves, trades, and **DSR shown
  beside Sharpe**.

### M4 — Probability Lab & Watchlist Screener

Adds real-world probability tooling and a multi-name screener:

- **`osl.volatility.garch`** — GARCH(1,1) / GJR-GARCH (normal or Student-t) via
  `arch`, reporting persistence and an annualized vol forecast.
- **`osl.volatility.skew` / `osl.volatility.vrp`** — 25Δ risk reversal &
  butterfly from a smile, and a variance-risk-premium proxy.
- **`osl.strategy.metrics`** — P-measure POP from a GARCH σ-forecast and a
  (gated, experimental) Bayesian-blended POP alongside the RN measures.
- **Pages 8 & 10** — Probability Lab (POP by measure, EV ± CI, market-implied
  RND vs lognormal) and the Watchlist Screener (per-name IV/RV, 25Δ RR, VRP,
  top strategy per objective).

### M3 — Payoff & scenario analysis

Adds intermediate-time valuation and scenario tooling on top of the M2 engine:

- **`osl.scenario.payoff`** — mark-to-market `value_at` / `pnl_at` that reprice
  each leg via BSM at the remaining time with an IV shock (reduces to the
  terminal payoff at expiry and to the entry value now), plus `payoff_curve`.
- **`osl.scenario.pnl_surface`** — `pnl_grid` over (spot shock, days-forward) at
  a fixed IV shock.
- **`osl.scenario.stress`** — scripted ±1σ/±2σ and earnings-move (with IV crush)
  scenarios from the strategy's reference IV.
- **Page 7** — Payoff & Scenario: payoff curve (with a time/IV-shocked overlay),
  3D P&L surface, and a stress-scenario table.

### M2 — Strategy engine & optimizer

Adds the multi-leg strategy engine on top of M1's pricing/vol/surface library:

- **Legs & strategies** (`osl.strategy`) — `Leg`/`Strategy` dataclasses and 22
  builders (long call/put, verticals, covered call, CSP, collar, protective
  put, straddles/strangles, iron condor/fly, broken-wing butterfly, ratio
  spread, jade lizard, calendar, diagonal, PMCC).
- **Metrics** — breakevens, bounded/**unbounded** max P/L, POP under four
  labelled measures (delta proxy, RN closed-form, RN Monte Carlo, empirical
  bootstrap), EV (closed-form edge + RN MC with 95% CI + empirical), expected
  shortfall, return-on-risk, aggregated greeks, and a convexity proxy. POP is
  never shown without an EV and tail-loss companion.
- **Liquidity** — per-strategy score (spread, OI, volume, ATM distance) and a
  slippage estimate; zero-bid legs are penalized.
- **Enumeration & optimizer** — `enumerate_candidates` (view-aware, excludes
  zero-bid legs) and a ranking optimizer with nine objectives; `pop_capped`
  never ranks an unbounded-risk position first.
- **Pages 5–6** — Strategy Generator (top-N per objective, radar) and Strategy
  Optimizer (sortable ranking, EV/POP-vs-risk frontiers).

### M1 — Per-name volatility diagnostics

Builds on the M0 data foundation with the pricing, volatility, and surface
library plus the first analytical UI pages:

- **Pricing** (`osl.pricing`) — vectorized Black-Scholes-Merton price and greeks
  (Δ, Γ, ν, Θ, ρ, vanna, charm, vomma; theta/day, vega/vol-point), cross-checked
  against QuantLib and finite differences; implied vol via Jaeckel (`vollib`)
  with a Brent fallback; put-call parity diagnostics.
- **Volatility** (`osl.volatility`) — realized-vol estimators (close-to-close,
  Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang), IV rank/percentile,
  and volatility cones.
- **Surface** (`osl.surface`) — smile preparation, raw SVI calibration with
  Gatheral butterfly + calendar no-arbitrage checks, a monotone variance-spline
  fallback, and Breeden-Litzenberger risk-neutral density.
- **Viz + pages** (`osl.viz`, `pages/`) — Plotly chart builders and four
  Streamlit pages (Ticker Overview, Options Chain, IV Surface, Vol Diagnostics)
  plus an Assumptions & Disclaimers page. Pages are thin shells over `osl.*`.

### M0 — Data Foundation

The data layer and project scaffold:

- **Configuration & logging** — `osl.config` (Pydantic settings, `OSL_`-prefixed
  env vars, secret-safe) and `osl.logging` (structlog).
- **Provider-agnostic interface** — `osl.data.base` defines the `DataProvider`
  protocol, immutable quote dataclasses, and the **canonical option-chain
  DataFrame schema** validated with `pandera`. All providers normalize to it.
- **Schwab adapter** (`osl.data.schwab`) — OAuth token lifecycle (proactive
  30-minute access-token refresh, 7-day refresh-token expiry), HTTP 429
  exponential backoff with jitter, the `chains` response parser (IV %→decimal,
  per-day theta, `-999` greek sentinels, zero-bid handling), and raw-payload
  teeing. All HTTP/token/clock/RNG dependencies are injectable for offline tests.
- **yfinance adapter** (`osl.data.yfinance_provider`) — fallback provider;
  greeks are emitted as `NaN` (computed downstream in M1).
- **Caching** (`osl.data.cache`) — disk-backed memoization via `diskcache`.
- **Snapshot store** (`osl.data.snapshots`) — Hive-partitioned Parquet chain
  history with byte-stable round-trips.
- **Utilities** — ACT/365F day-count, NYSE trading calendar, FRED/flat
  risk-free curves.
- **Streamlit app** (`app.py`) — a single page showing an option chain with a
  provider toggle and a data-freshness badge (GREEN/AMBER/RED).

Later milestones (pricing/greeks, vol diagnostics, strategy engine, payoff and
scenario analysis, probability lab, backtester) build on this foundation.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"        # core + dev tooling
# optional integrations:
uv pip install -e ".[data,ui]"    # yfinance + streamlit/plotly
```

Copy `.env.example` to `.env` and fill in credentials (Schwab app key/secret,
optional FRED key). Schwab requires a one-time browser-loopback OAuth flow to
mint the first token.

## Develop

```bash
uv run ruff check osl tests app.py app_lib.py pages   # lint
uv run black --check osl tests app.py app_lib.py pages # format
uv run mypy osl                                        # strict type-check (library core)
uv run pytest                                          # tests
```

Install the git hooks with `uv run pre-commit install`.

## Run the app

```bash
uv run streamlit run app.py   # requires the [ui] extra
```

## Layout

```
osl/
├── config.py, logging.py
├── data/         # base contracts, schwab/yfinance adapters, cache, snapshots
├── pricing/      # BSM price + greeks, IV inversion, put-call parity
├── volatility/   # realized-vol estimators, IV rank/percentile, cones
├── surface/      # smile prep, SVI fit + no-arb checks, variance spline, RND
├── viz/          # Plotly chart builders (no Streamlit)
└── utils/        # calendars, rates, time/day-count
app.py            # Streamlit entrypoint (multipage home)
app_lib.py        # shared Streamlit helpers (provider, caching, badge)
pages/            # thin analytical pages (Overview, Chain, Surface, Vol Diag)
tests/            # unit tests + golden Schwab fixture
```
