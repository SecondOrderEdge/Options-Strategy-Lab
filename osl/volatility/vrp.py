"""Variance risk premium estimators.

The variance risk premium is the gap between option-implied and realized
variance (positive on average for equity indices; Carr & Wu 2009). Here it is a
**proxy**: implied minus realized using ATM IV and a realized-vol estimate, not
a full variance-swap replication.
"""

from __future__ import annotations


def variance_risk_premium(iv: float, rv: float) -> float:
    """Implied minus realized **variance** (IV^2 - RV^2)."""
    return iv * iv - rv * rv


def vol_point_premium(iv: float, rv: float) -> float:
    """Implied minus realized **volatility** (vol points)."""
    return iv - rv
