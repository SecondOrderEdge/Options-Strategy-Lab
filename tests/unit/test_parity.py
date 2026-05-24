"""Put-call parity: synthetic prices satisfy parity; perturbations are flagged."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osl.pricing.bsm import bsm_price
from osl.pricing.parity import parity_gap, put_call_parity_violations


def test_parity_holds_synthetic():
    S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.04, 0.02, 0.3
    call = float(bsm_price(S, K, T, r, q, sigma, "c").item())
    put = float(bsm_price(S, K, T, r, q, sigma, "p").item())
    gap = float(parity_gap(call, put, S, K, T, r, q))
    assert gap == pytest.approx(0.0, abs=1e-9)


def _synthetic_chain(strikes, *, S, T, r, q, sigma, dte):
    rows = []
    for K in strikes:
        for right in ("C", "P"):
            flag = "c" if right == "C" else "p"
            mid = float(bsm_price(S, K, T, r, q, sigma, flag).item())
            rows.append(
                {
                    "right": right,
                    "strike": K,
                    "expiration": pd.Timestamp("2026-06-19"),
                    "dte": dte,
                    "mid": mid,
                }
            )
    return pd.DataFrame(rows)


def test_violations_none_for_consistent_chain():
    S, r, q, sigma, dte = 100.0, 0.04, 0.02, 0.3, 182
    T = dte / 365.0
    chain = _synthetic_chain([90, 95, 100, 105, 110], S=S, T=T, r=r, q=q, sigma=sigma, dte=dte)
    result = put_call_parity_violations(chain, spot=S, rate=r, dividend_yield=q, tolerance=0.01)
    assert len(result) == 5
    assert not result["violation"].any()
    assert np.allclose(result["gap"], 0.0, atol=1e-8)


def test_violation_flagged_when_put_mispriced():
    S, r, q, sigma, dte = 100.0, 0.04, 0.02, 0.3, 182
    T = dte / 365.0
    chain = _synthetic_chain([100], S=S, T=T, r=r, q=q, sigma=sigma, dte=dte)
    chain.loc[chain["right"] == "P", "mid"] += 1.0  # corrupt the put
    result = put_call_parity_violations(chain, spot=S, rate=r, dividend_yield=q, tolerance=0.05)
    assert result["violation"].all()
    assert abs(result["gap"].iloc[0]) == pytest.approx(1.0, abs=1e-6)
