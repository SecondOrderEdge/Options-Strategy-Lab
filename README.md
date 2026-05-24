# Options Strategy Lab

A strictly layered Python library and Streamlit UI for options analytics, built
around one principle: **analytical honesty over visual flash**. Every metric
carries its model assumptions, probability measure (risk-neutral vs real-world),
and data provenance.

> Research and education only — not investment advice. All models have explicit
> assumptions; read the assumptions before interpreting any output.

## Status: M0 — Data Foundation

This is the first milestone: the data layer and project scaffold. It ships:

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
uv run ruff check osl tests        # lint
uv run black --check osl tests     # format
uv run mypy osl                    # strict type-check (library core)
uv run pytest                      # tests
```

Install the git hooks with `uv run pre-commit install`.

## Run the app

```bash
uv run streamlit run app.py
```

## Layout

```
osl/
├── config.py, logging.py
├── data/        # base contracts, schwab/yfinance adapters, cache, snapshots
└── utils/       # calendars, rates, time/day-count
app.py           # Streamlit entrypoint (thin UI; no business logic)
tests/           # unit tests + golden Schwab fixture
```
