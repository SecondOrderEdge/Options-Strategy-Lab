"""Smoke tests for the Plotly chart builders (figures construct with data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from osl.surface.prepare import Smile
from osl.surface.rnd import risk_neutral_density
from osl.surface.svi import fit_svi
from osl.viz import charts

T = 0.25
FORWARD = 100.0


def _smile() -> Smile:
    strikes = np.arange(80, 121, 5.0)
    k = np.log(strikes / FORWARD)
    iv = 0.2 + 0.001 * (100 - strikes)
    return Smile(
        expiration=pd.Timestamp("2026-08-21").date(),
        T=T,
        forward=FORWARD,
        strikes=strikes,
        k=k,
        iv=iv,
        total_variance=iv * iv * T,
        weights=np.ones_like(strikes),
    )


def _history() -> pd.DataFrame:
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300)))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close})


def test_all_charts_build_with_data():
    smile = _smile()
    fit = fit_svi(smile.k, smile.total_variance, weights=smile.weights, T=T)
    cone = pd.DataFrame(
        {"p10": [0.1, 0.12], "p50": [0.2, 0.22], "p90": [0.3, 0.33], "current": [0.25, 0.24]},
        index=[10, 20],
    )
    strikes, density = risk_neutral_density(
        lambda K: np.full_like(np.asarray(K, float), 0.2), spot=100, rate=0.03, T=T, n=200
    )

    figs = [
        charts.price_history_chart(_history()),
        charts.iv_surface_3d([smile]),
        charts.iv_heatmap([smile]),
        charts.skew_chart([smile], {str(smile.expiration): fit.params}),
        charts.term_structure_chart([0.1, 0.3], [0.2, 0.22]),
        charts.vol_cone_chart(cone),
        charts.iv_vs_rv_chart(pd.Series([0.2, 0.21]), pd.Series([0.18, 0.19])),
        charts.rnd_chart(strikes, density),
        charts.scatter_chart(
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            labels=["a", "b", "c"],
            x_title="risk",
            y_title="ev",
            title="frontier",
        ),
        charts.radar_chart(
            ["POP", "EV", "Theta", "Vega", "Convexity", "Liquidity"],
            [0.6, 0.4, 0.3, 0.5, 0.2, 0.8],
            name="Iron Condor",
        ),
    ]
    for fig in figs:
        assert len(fig.data) >= 1
        # Every figure carries the research-only disclaimer annotation.
        assert any("Research and education only" in (a.text or "") for a in fig.layout.annotations)
