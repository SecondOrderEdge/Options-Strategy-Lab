"""Hive-partitioned Parquet snapshot store for chain history.

Layout (readable by ``pyarrow.dataset``)::

    {root}/underlying=SPY/snapshot_date=2026-05-24/snapshot_time=20260524T160500Z/
        chain.parquet
        _underlying.parquet

``chain.parquet`` holds the canonical chain DataFrame; ``_underlying.parquet``
holds the spot/rate/div context at snapshot time. Partition values use UTC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from osl.data.base import UnderlyingQuote, coerce_chain
from osl.logging import get_logger
from osl.utils.time import now_utc, to_utc

log = get_logger(__name__)

CHAIN_FILE = "chain.parquet"
UNDERLYING_FILE = "_underlying.parquet"


@dataclass(frozen=True)
class SnapshotContext:
    """Spot/rate/dividend context persisted alongside a chain snapshot."""

    underlying: str
    spot: float
    risk_free_rate: float
    dividend_yield: float
    snapshot_time: datetime


def _fmt_time(ts: datetime) -> str:
    return to_utc(ts).strftime("%Y%m%dT%H%M%SZ")


def snapshot_dir(root: str | Path, underlying: str, snapshot_time: datetime) -> Path:
    """Return the partition directory for a snapshot (not created)."""
    ts = to_utc(snapshot_time)
    return (
        Path(root)
        / f"underlying={underlying}"
        / f"snapshot_date={ts.date().isoformat()}"
        / f"snapshot_time={_fmt_time(ts)}"
    )


def write_snapshot(
    chain: pd.DataFrame,
    *,
    root: str | Path,
    context: SnapshotContext | None = None,
    snapshot_time: datetime | None = None,
) -> Path:
    """Write ``chain`` (and optional ``context``) to a Hive partition.

    Returns the partition directory. The chain is coerced to canonical dtypes
    before writing so re-reads are byte-stable.
    """
    if chain.empty and context is None:
        raise ValueError("refusing to write an empty snapshot with no context")

    underlying = context.underlying if context is not None else str(chain["underlying"].iloc[0])
    ts = snapshot_time or (context.snapshot_time if context else now_utc())

    out_dir = snapshot_dir(root, underlying, ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    coerce_chain(chain).to_parquet(out_dir / CHAIN_FILE, engine="pyarrow", index=False)

    if context is not None:
        ctx = asdict(context)
        ctx["snapshot_time"] = to_utc(context.snapshot_time)
        pd.DataFrame([ctx]).to_parquet(out_dir / UNDERLYING_FILE, engine="pyarrow", index=False)

    log.info("snapshot_written", underlying=underlying, path=str(out_dir), rows=len(chain))
    return out_dir


def read_chain(partition_dir: str | Path) -> pd.DataFrame:
    """Read the chain DataFrame from a snapshot partition directory."""
    df: pd.DataFrame = pd.read_parquet(Path(partition_dir) / CHAIN_FILE, engine="pyarrow")
    return df


def read_context(partition_dir: str | Path) -> SnapshotContext | None:
    """Read the underlying context from a snapshot partition, if present."""
    path = Path(partition_dir) / UNDERLYING_FILE
    if not path.exists():
        return None
    row = pd.read_parquet(path, engine="pyarrow").iloc[0]
    return SnapshotContext(
        underlying=str(row["underlying"]),
        spot=float(row["spot"]),
        risk_free_rate=float(row["risk_free_rate"]),
        dividend_yield=float(row["dividend_yield"]),
        snapshot_time=pd.Timestamp(row["snapshot_time"]).to_pydatetime(),
    )


def context_from_quote(
    quote: UnderlyingQuote, *, risk_free_rate: float, dividend_yield: float
) -> SnapshotContext:
    """Build a :class:`SnapshotContext` from an underlying quote + rates."""
    return SnapshotContext(
        underlying=quote.symbol,
        spot=quote.mark if quote.mark == quote.mark else quote.last,  # NaN-safe pick
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        snapshot_time=quote.quote_time,
    )
