"""Snapshot store: Parquet roundtrip is byte-stable; context persists."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from osl.data.schwab import parse_chain
from osl.data.snapshots import (
    SnapshotContext,
    read_chain,
    read_context,
    snapshot_dir,
    write_snapshot,
)

SNAP_TIME = datetime(2026, 5, 24, 16, 5, 0, tzinfo=UTC)


def test_chain_roundtrip_identical(schwab_payload: dict[str, Any], tmp_path: Path) -> None:
    chain = parse_chain(schwab_payload, snapshot_id="snap-1")
    out_dir = write_snapshot(chain, root=tmp_path, snapshot_time=SNAP_TIME)
    reloaded = read_chain(out_dir)
    pd.testing.assert_frame_equal(chain, reloaded)


def test_partition_layout(schwab_payload: dict[str, Any], tmp_path: Path) -> None:
    chain = parse_chain(schwab_payload, snapshot_id="snap-1")
    out_dir = write_snapshot(chain, root=tmp_path, snapshot_time=SNAP_TIME)
    expected = snapshot_dir(tmp_path, "AAPL", SNAP_TIME)
    assert out_dir == expected
    assert out_dir.name == "snapshot_time=20260524T160500Z"
    assert out_dir.parent.name == "snapshot_date=2026-05-24"
    assert out_dir.parent.parent.name == "underlying=AAPL"
    assert (out_dir / "chain.parquet").exists()


def test_context_roundtrip(schwab_payload: dict[str, Any], tmp_path: Path) -> None:
    chain = parse_chain(schwab_payload, snapshot_id="snap-1")
    ctx = SnapshotContext(
        underlying="AAPL",
        spot=190.12,
        risk_free_rate=0.045,
        dividend_yield=0.005,
        snapshot_time=SNAP_TIME,
    )
    out_dir = write_snapshot(chain, root=tmp_path, context=ctx, snapshot_time=SNAP_TIME)
    loaded = read_context(out_dir)
    assert loaded is not None
    assert loaded.underlying == "AAPL"
    assert loaded.spot == pytest.approx(190.12)
    assert loaded.risk_free_rate == pytest.approx(0.045)


def test_refuses_empty_snapshot_without_context(tmp_path: Path) -> None:
    from osl.data.base import empty_chain

    with pytest.raises(ValueError, match="empty snapshot"):
        write_snapshot(empty_chain(), root=tmp_path, snapshot_time=SNAP_TIME)
