"""Canonical chain schema: conformance, coercion, and validation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from pandera.errors import SchemaError

from osl.data.base import (
    CANONICAL_COLUMNS,
    CANONICAL_DTYPES,
    coerce_chain,
    empty_chain,
    validate_chain,
)
from osl.data.schwab import parse_chain


def test_empty_chain_conforms() -> None:
    df = empty_chain()
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    validated = validate_chain(df)
    assert validated.empty


def test_parsed_chain_conforms(schwab_payload: dict[str, Any]) -> None:
    df = parse_chain(schwab_payload, snapshot_id="snap-1")
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    validate_chain(df)  # raises on violation


def test_dtypes_match_canonical(schwab_payload: dict[str, Any]) -> None:
    df = parse_chain(schwab_payload, snapshot_id="snap-1")
    for col, dtype in CANONICAL_DTYPES.items():
        assert str(df[col].dtype) == dtype, f"{col}: {df[col].dtype} != {dtype}"


def test_coerce_chain_rejects_missing_columns() -> None:
    df = pd.DataFrame({"underlying": ["SPY"]})
    with pytest.raises(KeyError, match="missing canonical columns"):
        coerce_chain(df)


def test_validate_rejects_bad_right(schwab_payload: dict[str, Any]) -> None:
    df = parse_chain(schwab_payload, snapshot_id="snap-1")
    df["right"] = df["right"].cat.add_categories(["X"])
    df.loc[df.index[0], "right"] = "X"
    with pytest.raises(SchemaError):
        validate_chain(df)
