"""Settings load defaults and OSL_-prefixed env overrides without leaking secrets."""

from __future__ import annotations

import pytest

from osl.config import Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.default_provider == "schwab"
    assert s.risk_free_curve_source == "flat"
    assert s.risk_free_flat_rate == pytest.approx(0.045)
    assert s.iv_lookback_days == 252
    assert s.enable_experimental is False
    assert s.raw_data_root == s.data_root / "raw"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSL_DEFAULT_PROVIDER", "yfinance")
    monkeypatch.setenv("OSL_RISK_FREE_FLAT_RATE", "0.052")
    monkeypatch.setenv("OSL_ENABLE_EXPERIMENTAL", "true")
    monkeypatch.setenv("OSL_SCHWAB_APP_KEY", "super-secret-key")

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.default_provider == "yfinance"
    assert s.risk_free_flat_rate == pytest.approx(0.052)
    assert s.enable_experimental is True
    # Secret is retrievable but never rendered in repr/str.
    assert s.schwab_app_key.get_secret_value() == "super-secret-key"
    assert "super-secret-key" not in repr(s)


def test_iv_lookback_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, iv_lookback_days=0)  # type: ignore[call-arg]
