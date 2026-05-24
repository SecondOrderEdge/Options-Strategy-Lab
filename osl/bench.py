"""Lightweight benchmark suite for the nightly CI job.

Times a few representative operations on synthetic data and returns a
``{name: seconds}`` mapping. Run as ``python -m osl.bench``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from osl.backtest.config import BacktestConfig
from osl.backtest.demo import synthetic_history
from osl.backtest.engine import run_backtest
from osl.backtest.rules import ShortPut16Delta
from osl.pricing.bsm import bsm_price
from osl.surface.svi import fit_svi


def _time(fn: Callable[[], object], *, repeat: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - start) / repeat


def _bench_pricing() -> None:
    strikes = np.linspace(50, 150, 2000)
    bsm_price(100.0, strikes, 0.5, 0.03, 0.0, 0.2, "c")


def _bench_svi_fit() -> None:
    k = np.linspace(-0.4, 0.4, 25)
    w = 0.04 + 0.1 * (0.3 * k + np.sqrt(k * k + 0.01))
    fit_svi(k, w, T=0.25)


def _bench_backtest() -> None:
    days = synthetic_history("BENCH", n_days=120, seed=1)
    run_backtest(days, ShortPut16Delta(), BacktestConfig(fill_model="mid"))


def run_benchmarks() -> dict[str, float]:
    """Return wall-clock seconds for each benchmarked operation."""
    return {
        "bsm_price_2k_strikes": _time(_bench_pricing, repeat=20),
        "svi_fit": _time(_bench_svi_fit, repeat=10),
        "backtest_120d": _time(_bench_backtest),
    }


def main() -> int:
    results = run_benchmarks()
    width = max(len(k) for k in results)
    for name, seconds in results.items():
        print(f"{name:<{width}}  {seconds * 1000:8.2f} ms")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
