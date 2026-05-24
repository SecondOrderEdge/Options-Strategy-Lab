"""Scripted stress scenarios: standard-deviation moves and an earnings shock.

Sigma moves use the strategy's reference IV over its primary horizon
(``move = spot * sigma * sqrt(dte/365)``). The earnings scenario applies the
IV-implied expected move together with an IV crush (a negative vol shock).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from osl.scenario.payoff import pnl_at
from osl.strategy.legs import Strategy
from osl.strategy.metrics import reference_sigma
from osl.utils.time import DAYS_PER_YEAR


@dataclass(frozen=True)
class ScenarioResult:
    label: str
    spot: float
    vol_shock: float
    days_forward: float
    pnl: float


def stress_scenarios(
    strategy: Strategy,
    *,
    sigma: float | None = None,
    days_forward: float = 0.0,
    earnings_iv_crush: float = -0.0,
) -> list[ScenarioResult]:
    """Return scripted ±1σ/±2σ and earnings-move scenarios with their P&L."""
    sig = sigma if sigma is not None else reference_sigma(strategy)
    spot = strategy.spot
    horizon = math.sqrt(max(strategy.primary_dte, 1) / DAYS_PER_YEAR)
    move = spot * sig * horizon if math.isfinite(sig) else 0.0

    specs: list[tuple[str, float, float]] = [
        ("-2 sigma", spot - 2 * move, 0.0),
        ("-1 sigma", spot - move, 0.0),
        ("flat", spot, 0.0),
        ("+1 sigma", spot + move, 0.0),
        ("+2 sigma", spot + 2 * move, 0.0),
        ("earnings down (IV crush)", spot - move, earnings_iv_crush),
        ("earnings up (IV crush)", spot + move, earnings_iv_crush),
    ]

    results: list[ScenarioResult] = []
    for label, scen_spot, vshock in specs:
        pnl = float(
            pnl_at(strategy, np.array([scen_spot]), days_forward=days_forward, vol_shock=vshock)[0]
        )
        results.append(
            ScenarioResult(
                label=label, spot=scen_spot, vol_shock=vshock, days_forward=days_forward, pnl=pnl
            )
        )
    return results
