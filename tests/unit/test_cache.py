"""Disk cache decorator: memoization, key separation, and clearing."""

from __future__ import annotations

from pathlib import Path

from osl.data.cache import disk_cached, make_cache


def test_memoizes_repeated_calls(tmp_path: Path) -> None:
    cache = make_cache(tmp_path / "c")
    calls = {"n": 0}

    @disk_cached(cache, ttl=60)
    def expensive(x: int) -> int:
        calls["n"] += 1
        return x * 2

    assert expensive(21) == 42
    assert expensive(21) == 42
    assert calls["n"] == 1  # second call served from cache


def test_distinct_args_distinct_entries(tmp_path: Path) -> None:
    cache = make_cache(tmp_path / "c")
    calls = {"n": 0}

    @disk_cached(cache, ttl=60)
    def f(x: int) -> int:
        calls["n"] += 1
        return x

    f(1)
    f(2)
    f(1)
    assert calls["n"] == 2


def test_cache_clear(tmp_path: Path) -> None:
    cache = make_cache(tmp_path / "c")
    calls = {"n": 0}

    @disk_cached(cache, ttl=60)
    def f(x: int) -> int:
        calls["n"] += 1
        return x

    f(5)
    f.cache_clear()  # type: ignore[attr-defined]
    f(5)
    assert calls["n"] == 2
