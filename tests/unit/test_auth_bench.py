"""Password gate and the benchmark suite."""

from __future__ import annotations

from osl.auth import hash_password, verify_password
from osl.bench import run_benchmarks
from osl.config import Settings
from osl.data.marketdata import MarketDataProvider
from osl.data.providers import build_provider
from osl.data.tradier import TradierProvider


def test_password_roundtrip():
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong", h)


def test_password_hash_is_hex_sha256():
    h = hash_password("x")
    assert len(h) == 64
    int(h, 16)  # parses as hex


def test_build_provider_extra_providers():
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert isinstance(build_provider("tradier", s), TradierProvider)
    assert isinstance(build_provider("marketdata", s), MarketDataProvider)


def test_benchmarks_run():
    results = run_benchmarks()
    assert set(results) == {"bsm_price_2k_strikes", "svi_fit", "backtest_120d"}
    assert all(t >= 0 for t in results.values())
