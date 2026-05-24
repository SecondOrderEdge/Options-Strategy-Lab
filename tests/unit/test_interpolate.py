"""Variance-spline fallback: interpolation at knots, IV, dedup, extrapolation."""

from __future__ import annotations

import numpy as np
import pytest

from osl.surface.interpolate import VarianceSpline


def test_interpolates_through_knots():
    k = np.array([-0.3, -0.1, 0.0, 0.2, 0.4])
    w = np.array([0.05, 0.04, 0.038, 0.045, 0.06])
    spline = VarianceSpline.fit(k, w)
    np.testing.assert_allclose(spline.total_variance(k), w, atol=1e-9)


def test_iv_from_total_variance():
    k = np.array([-0.2, 0.0, 0.2])
    T = 0.5
    w = np.array([0.02, 0.02, 0.02])  # flat total variance
    spline = VarianceSpline.fit(k, w)
    np.testing.assert_allclose(spline.iv(k, T), np.sqrt(0.02 / T), atol=1e-9)


def test_flat_extrapolation_beyond_knots():
    k = np.array([-0.2, 0.0, 0.2])
    w = np.array([0.05, 0.03, 0.06])
    spline = VarianceSpline.fit(k, w)
    # Outside the knot range, value clamps to the nearest endpoint.
    assert spline.total_variance(-5.0) == pytest.approx(w[0])
    assert spline.total_variance(5.0) == pytest.approx(w[-1])


def test_dedup_and_sort():
    k = np.array([0.2, 0.0, 0.0, -0.2])  # unsorted with a duplicate knot
    w = np.array([0.06, 0.03, 0.03, 0.05])
    spline = VarianceSpline.fit(k, w)
    assert list(spline.k) == [-0.2, 0.0, 0.2]


def test_requires_two_knots():
    with pytest.raises(ValueError, match=">= 2 distinct knots"):
        VarianceSpline.fit([0.0, 0.0], [0.03, 0.03])
