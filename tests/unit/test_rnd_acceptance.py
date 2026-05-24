"""M4 acceptance: Breeden-Litzenberger RND integrates to 1 with mean = forward."""

from __future__ import annotations

import numpy as np
import pytest

from osl.surface.rnd import risk_neutral_density
from osl.surface.svi import SVIParams, make_iv_of_strike

SPOT = 100.0
RATE = 0.03
DIV = 0.01
T = 0.25
# Arbitrage-free, mildly-skewed equity smile.
PARAMS = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)


def _density():
    forward = SPOT * np.exp((RATE - DIV) * T)
    iv_fn = make_iv_of_strike(PARAMS, forward=forward, T=T)
    return risk_neutral_density(
        iv_fn, spot=SPOT, rate=RATE, dividend_yield=DIV, T=T, k_min=-2.0, k_max=2.0, n=2500
    )


def test_rnd_integrates_to_one():
    strikes, density = _density()
    area = float(np.trapz(density, strikes))
    assert area == pytest.approx(1.0, abs=0.01)


def test_rnd_mean_equals_forward():
    strikes, density = _density()
    area = float(np.trapz(density, strikes))
    mean = float(np.trapz(strikes * density, strikes)) / area
    forward = SPOT * np.exp((RATE - DIV) * T)
    assert mean == pytest.approx(forward, rel=0.005)
