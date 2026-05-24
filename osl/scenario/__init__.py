"""Payoff and scenario analysis: mark-to-market, P&L surfaces, stress tests."""

from __future__ import annotations

from osl.scenario.payoff import payoff_curve, pnl_at, value_at
from osl.scenario.pnl_surface import pnl_grid
from osl.scenario.stress import ScenarioResult, stress_scenarios

__all__ = [
    "ScenarioResult",
    "payoff_curve",
    "pnl_at",
    "pnl_grid",
    "stress_scenarios",
    "value_at",
]
