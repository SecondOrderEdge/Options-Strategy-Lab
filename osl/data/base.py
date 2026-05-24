"""Provider-agnostic data contracts.

This module defines:

* immutable quote dataclasses (:class:`OptionQuote`, :class:`UnderlyingQuote`,
  :class:`OHLCVBar`),
* the :class:`DataProvider` ``Protocol`` every adapter implements,
* the **canonical option-chain DataFrame schema** that all providers normalize
  to, validated with ``pandera``.

The canonical schema is the single source of truth: Schwab and yfinance
adapters each emit a DataFrame conforming to it, so downstream code never sees
provider-specific quirks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

RIGHT_VALUES = ("C", "P")
PROVIDER_VALUES = ("schwab", "yfinance")


# ---------------------------------------------------------------------------
# Immutable quote records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptionQuote:
    """A single option contract quote (provider-normalized)."""

    symbol: str  # OCC option symbol
    underlying: str
    expiration: date
    strike: float
    right: str  # "C" or "P"
    bid: float
    ask: float
    mid: float
    last: float | None
    mark: float | None
    volume: int
    open_interest: int
    iv: float | None  # decimal (0.23 == 23%)
    delta: float | None
    gamma: float | None
    theta: float | None  # per day
    vega: float | None  # per 1 vol point (0.01)
    rho: float | None
    multiplier: int  # 100 for standard equity
    exchange: str
    quote_time: datetime  # UTC
    is_nonstandard: bool
    in_the_money: bool


@dataclass(frozen=True)
class UnderlyingQuote:
    """A snapshot of the underlying instrument."""

    symbol: str
    last: float
    bid: float
    ask: float
    mark: float
    volume: int
    quote_time: datetime
    is_delayed: bool


@dataclass(frozen=True)
class OHLCVBar:
    """A single OHLCV bar."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


# ---------------------------------------------------------------------------
# Canonical chain schema
# ---------------------------------------------------------------------------
# Column -> target pandas dtype. Order here is the canonical column order.
CANONICAL_DTYPES: dict[str, str] = {
    "underlying": "string",
    "option_symbol": "string",
    "expiration": "datetime64[ns, US/Eastern]",
    "dte": "int16",
    "right": "category",
    "strike": "float64",
    "bid": "float64",
    "ask": "float64",
    "mid": "float64",
    "last": "float64",
    "mark": "float64",
    "bid_size": "int32",
    "ask_size": "int32",
    "volume": "int32",
    "open_interest": "int32",
    "iv": "float64",
    "delta": "float64",
    "gamma": "float64",
    "theta": "float64",
    "vega": "float64",
    "rho": "float64",
    "theo": "float64",
    "multiplier": "int16",
    "exchange": "string",
    "quote_time": "datetime64[ns, UTC]",
    "is_nonstandard": "bool",
    "in_the_money": "bool",
    "provider": "category",
    "snapshot_id": "string",
}

CANONICAL_COLUMNS: tuple[str, ...] = tuple(CANONICAL_DTYPES.keys())

# Numeric columns that are allowed to be NaN (provider may omit greeks/IV/prices).
_NULLABLE_FLOATS = (
    "bid",
    "ask",
    "mid",
    "last",
    "mark",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "theo",
)


def chain_schema() -> DataFrameSchema:
    """Return the pandera schema for the canonical chain DataFrame."""
    columns: dict[str, Column] = {
        "underlying": Column("string", nullable=False),
        "option_symbol": Column("string", nullable=False),
        "expiration": Column("datetime64[ns, US/Eastern]", nullable=False),
        "dte": Column("int16", Check.ge(0), nullable=False),
        "right": Column("category", Check.isin(RIGHT_VALUES), nullable=False),
        "strike": Column("float64", Check.gt(0), nullable=False),
        "bid_size": Column("int32", Check.ge(0), nullable=False),
        "ask_size": Column("int32", Check.ge(0), nullable=False),
        "volume": Column("int32", Check.ge(0), nullable=False),
        "open_interest": Column("int32", Check.ge(0), nullable=False),
        "multiplier": Column("int16", Check.gt(0), nullable=False),
        "exchange": Column("string", nullable=True),
        "quote_time": Column("datetime64[ns, UTC]", nullable=False),
        "is_nonstandard": Column("bool", nullable=False),
        "in_the_money": Column("bool", nullable=False),
        "provider": Column("category", Check.isin(PROVIDER_VALUES), nullable=False),
        "snapshot_id": Column("string", nullable=False),
    }
    for name in _NULLABLE_FLOATS:
        columns[name] = Column("float64", nullable=True)

    return DataFrameSchema(
        columns=columns,
        strict=True,
        ordered=False,
        coerce=False,
        name="canonical_option_chain",
    )


def empty_chain() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical columns and dtypes."""
    df = pd.DataFrame({col: pd.Series(dtype="object") for col in CANONICAL_COLUMNS})
    return coerce_chain(df)


def coerce_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``df`` to canonical column order and dtypes.

    Missing columns raise ``KeyError`` — adapters are responsible for supplying
    every canonical column. Extra columns are dropped.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"chain DataFrame missing canonical columns: {missing}")

    out = df.loc[:, list(CANONICAL_COLUMNS)].copy()
    # Integer columns cannot hold NaN; widen to int64 first. Keeping them
    # NaN-free before coercion is the adapter's responsibility.
    for col, dtype in CANONICAL_DTYPES.items():
        if dtype.startswith("int"):
            out[col] = out[col].astype("int64")
    coerced: pd.DataFrame = out.astype(CANONICAL_DTYPES)
    return coerced


def validate_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``df`` against the canonical schema, returning it on success.

    Raises :class:`pandera.errors.SchemaError` on the first violation.
    """
    schema = chain_schema()
    validated: pd.DataFrame = schema.validate(df, lazy=False)
    return validated


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class DataProvider(Protocol):
    """The interface every market-data adapter implements."""

    name: str

    def get_underlying(self, symbol: str) -> UnderlyingQuote: ...

    def get_history(
        self, symbol: str, *, start: date, end: date, bar: str = "1d"
    ) -> pd.DataFrame: ...

    def get_option_chain(
        self,
        symbol: str,
        *,
        expirations: list[date] | None = None,
        strike_range: tuple[float, float] | None = None,
        min_oi: int = 0,
    ) -> pd.DataFrame: ...

    def get_expirations(self, symbol: str) -> list[date]: ...

    def healthcheck(self) -> dict[str, object]: ...


__all__ = [
    "CANONICAL_COLUMNS",
    "CANONICAL_DTYPES",
    "PROVIDER_VALUES",
    "RIGHT_VALUES",
    "DataProvider",
    "OHLCVBar",
    "OptionQuote",
    "UnderlyingQuote",
    "chain_schema",
    "coerce_chain",
    "empty_chain",
    "validate_chain",
]
