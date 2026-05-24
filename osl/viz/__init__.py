"""Plotly figure builders for the Streamlit pages.

These functions return ``plotly.graph_objects.Figure`` objects and contain no
Streamlit calls, so pages stay thin. Importing this package requires the ``ui``
extra (plotly); the library core never imports it.
"""

from __future__ import annotations

from osl.viz.charts import (
    iv_heatmap,
    iv_surface_3d,
    iv_vs_rv_chart,
    price_history_chart,
    rnd_chart,
    skew_chart,
    term_structure_chart,
    vol_cone_chart,
)

__all__ = [
    "iv_heatmap",
    "iv_surface_3d",
    "iv_vs_rv_chart",
    "price_history_chart",
    "rnd_chart",
    "skew_chart",
    "term_structure_chart",
    "vol_cone_chart",
]
