"""IV rank/percentile and volatility cone behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osl.volatility.ranks import iv_percentile, iv_rank, vol_cone


def test_iv_rank_extremes():
    s = pd.Series(np.linspace(0.1, 0.5, 100))
    assert iv_rank(s) == pytest.approx(1.0)  # last is the max
    assert iv_rank(s[::-1].reset_index(drop=True)) == pytest.approx(0.0)  # last is the min


def test_iv_rank_midpoint():
    s = pd.Series([0.2, 0.4, 0.3])  # last=0.3, lo=0.2, hi=0.4 -> 0.5
    assert iv_rank(s) == pytest.approx(0.5)


def test_iv_percentile():
    s = pd.Series([0.1, 0.2, 0.3, 0.4])  # last is the max => 100%
    assert iv_percentile(s) == pytest.approx(1.0)
    s2 = pd.Series([0.5, 0.1, 0.2, 0.3])  # last=0.3, two of four <= 0.3 (0.1,0.2,0.3)
    assert iv_percentile(s2) == pytest.approx(0.75)


def test_iv_rank_flat_series_is_nan():
    s = pd.Series([0.25] * 10)
    assert np.isnan(iv_rank(s))


def test_vol_cone_shape_and_monotonic():
    rng = np.random.default_rng(7)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))))
    cone = vol_cone(close, windows=(10, 20, 30), percentiles=(10, 50, 90))
    assert list(cone.index) == [10, 20, 30]
    for col in ("p10", "p50", "p90", "current"):
        assert col in cone.columns
    # Percentiles ordered within each window.
    assert (cone["p10"] <= cone["p50"]).all()
    assert (cone["p50"] <= cone["p90"]).all()
