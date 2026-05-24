"""Plotly chart builders for volatility diagnostics and the surface.

Every figure carries a research-only footnote and (where relevant) the model it
was produced from, per the blueprint's disclosure rules.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go

from osl.surface.prepare import Smile
from osl.surface.svi import SVIParams

FloatArray = npt.NDArray[np.float64]

DISCLAIMER = "Research and education only — not investment advice."


def _footer(fig: go.Figure, note: str = DISCLAIMER) -> go.Figure:
    fig.add_annotation(
        text=note,
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.18,
        showarrow=False,
        font={"size": 10, "color": "gray"},
        align="left",
    )
    fig.update_layout(margin={"b": 80})
    return fig


def price_history_chart(history: pd.DataFrame, *, title: str = "Price history") -> go.Figure:
    """Line chart of closing prices from a get_history DataFrame."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history["close"], mode="lines", name="Close"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return _footer(fig)


def iv_surface_3d(smiles: Sequence[Smile]) -> go.Figure:
    """3D implied-vol surface over (log-moneyness, T)."""
    fig = go.Figure()
    for smile in smiles:
        fig.add_trace(
            go.Scatter3d(
                x=smile.k,
                y=np.full(smile.k.shape, smile.T),
                z=smile.iv,
                mode="lines+markers",
                name=str(smile.expiration),
            )
        )
    fig.update_layout(
        title="Implied volatility surface",
        scene={
            "xaxis_title": "log-moneyness ln(K/F)",
            "yaxis_title": "T (years)",
            "zaxis_title": "IV",
        },
    )
    return _footer(fig)


def iv_heatmap(smiles: Sequence[Smile]) -> go.Figure:
    """Heatmap of IV across strikes (x) and expiries (y)."""
    expirations = [str(s.expiration) for s in smiles]
    all_strikes = sorted({float(k) for s in smiles for k in s.strikes})
    z: list[list[float]] = []
    for s in smiles:
        lookup = dict(zip(s.strikes.tolist(), s.iv.tolist(), strict=True))
        z.append([lookup.get(k, float("nan")) for k in all_strikes])
    fig = go.Figure(go.Heatmap(z=z, x=all_strikes, y=expirations, colorbar={"title": "IV"}))
    fig.update_layout(title="IV heatmap", xaxis_title="Strike", yaxis_title="Expiration")
    return _footer(fig)


def skew_chart(
    smiles: Sequence[Smile],
    fits: dict[str, SVIParams] | None = None,
) -> go.Figure:
    """IV vs log-moneyness per expiry, with optional SVI fit overlays."""
    fig = go.Figure()
    for smile in smiles:
        label = str(smile.expiration)
        fig.add_trace(go.Scatter(x=smile.k, y=smile.iv, mode="markers", name=f"{label} (mkt)"))
        if fits and label in fits:
            grid = np.linspace(float(smile.k.min()), float(smile.k.max()), 100)
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=fits[label].iv(grid, smile.T),
                    mode="lines",
                    name=f"{label} (SVI)",
                )
            )
    fig.update_layout(
        title="Volatility skew", xaxis_title="log-moneyness ln(K/F)", yaxis_title="IV"
    )
    return _footer(fig, f"{DISCLAIMER}  |  Model: raw SVI per expiry")


def term_structure_chart(tenors: Sequence[float], atm_iv: Sequence[float]) -> go.Figure:
    """ATM IV vs time to expiry."""
    fig = go.Figure(go.Scatter(x=list(tenors), y=list(atm_iv), mode="lines+markers"))
    fig.update_layout(title="ATM term structure", xaxis_title="T (years)", yaxis_title="ATM IV")
    return _footer(fig)


def vol_cone_chart(cone: pd.DataFrame, *, current: pd.Series | None = None) -> go.Figure:
    """Volatility cone: realized-vol percentile bands across rolling windows."""
    fig = go.Figure()
    for col in [c for c in cone.columns if c.startswith("p")]:
        fig.add_trace(go.Scatter(x=cone.index, y=cone[col], mode="lines", name=col))
    if "current" in cone.columns:
        fig.add_trace(
            go.Scatter(
                x=cone.index,
                y=cone["current"],
                mode="markers",
                name="current RV",
                marker={"size": 9, "color": "black"},
            )
        )
    if current is not None:
        fig.add_trace(go.Scatter(x=current.index, y=current, mode="markers", name="current IV"))
    fig.update_layout(
        title="Volatility cone", xaxis_title="Window (days)", yaxis_title="Annualized vol"
    )
    return _footer(fig)


def iv_vs_rv_chart(iv: pd.Series, rv: pd.Series) -> go.Figure:
    """30D implied vs realized volatility, with the spread shaded."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iv.index, y=iv, mode="lines", name="IV (30D)"))
    fig.add_trace(go.Scatter(x=rv.index, y=rv, mode="lines", name="RV (30D, YZ)"))
    fig.update_layout(title="IV vs RV", xaxis_title="Date", yaxis_title="Annualized vol")
    return _footer(fig)


def scatter_chart(
    x: Sequence[float],
    y: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    x_title: str,
    y_title: str,
    title: str,
) -> go.Figure:
    """Generic labelled scatter (EV-vs-risk frontier, POP-vs-risk, etc.)."""
    fig = go.Figure(
        go.Scatter(
            x=list(x),
            y=list(y),
            mode="markers",
            text=list(labels) if labels is not None else None,
            hovertemplate="%{text}<br>" + x_title + ": %{x:.3f}<br>" + y_title + ": %{y:.3f}",
            marker={"size": 10},
        )
    )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return _footer(fig)


def radar_chart(categories: Sequence[str], values: Sequence[float], *, name: str) -> go.Figure:
    """Normalized strategy radar (POP, EV, theta, vega, convexity, liquidity)."""
    cats = [*categories, categories[0]]
    vals = [*values, values[0]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself", name=name))
    fig.update_layout(title=name, polar={"radialaxis": {"visible": True, "range": [0, 1]}})
    return _footer(fig)


def rnd_chart(
    strikes: FloatArray,
    density: FloatArray,
    *,
    lognormal: FloatArray | None = None,
    title: str = "Risk-neutral density",
) -> go.Figure:
    """Market-implied RND, optionally overlaid with a lognormal benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=density, mode="lines", name="Market-implied RN"))
    if lognormal is not None:
        fig.add_trace(go.Scatter(x=strikes, y=lognormal, mode="lines", name="Lognormal (flat IV)"))
    fig.update_layout(title=title, xaxis_title="Strike", yaxis_title="Density")
    return _footer(fig, f"{DISCLAIMER}  |  Method: Breeden-Litzenberger (risk-neutral)")
