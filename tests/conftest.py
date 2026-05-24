"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "data"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def schwab_payload() -> dict[str, Any]:
    """The golden Schwab `chains` response for AAPL."""
    data: dict[str, Any] = json.loads((FIXTURES_DIR / "schwab_chain_AAPL.json").read_text())
    return data
