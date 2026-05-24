"""Risk-free rate curve.

M0 ships a flat curve plus a FRED-backed Treasury-zero curve. Rates are stored
as continuously-compounded zeros keyed by tenor in years and linearly
interpolated (flat extrapolation beyond the endpoints). The FRED fetcher is
injectable so the curve can be unit-tested without network access.

Note: FRED Treasury constant-maturity series are bond-equivalent yields; OSL
treats them as continuously-compounded for the short tenors that matter for
options. Documented as a known approximation, not a defect.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

# FRED series id -> tenor in years.
FRED_TENORS: dict[str, float] = {
    "DGS1MO": 1.0 / 12.0,
    "DGS3MO": 3.0 / 12.0,
    "DGS6MO": 6.0 / 12.0,
    "DGS1": 1.0,
    "DGS2": 2.0,
}

FredFetcher = Callable[[str, str], float]
"""Callable ``(series_id, api_key) -> latest value in percent``."""


@dataclass(frozen=True)
class RiskFreeCurve:
    """A continuously-compounded zero curve over tenors (in years)."""

    tenors: tuple[float, ...]
    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.rates):
            raise ValueError("tenors and rates must have equal length")
        if not self.tenors:
            raise ValueError("curve must contain at least one tenor")
        if list(self.tenors) != sorted(self.tenors):
            raise ValueError("tenors must be sorted ascending")

    def rate(self, t_years: float) -> float:
        """Return the cc zero rate at ``t_years`` (flat beyond endpoints)."""
        return float(np.interp(t_years, self.tenors, self.rates))


def flat_curve(rate: float) -> RiskFreeCurve:
    """A single-rate curve usable at any tenor."""
    return RiskFreeCurve(tenors=(0.0, 100.0), rates=(rate, rate))


def _default_fred_fetcher(series_id: str, api_key: str) -> float:
    """Fetch the most recent observation for ``series_id`` from FRED (percent)."""
    import requests

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": "1",
        },
        timeout=15,
    )
    resp.raise_for_status()
    obs = resp.json()["observations"]
    return float(obs[0]["value"])


def from_fred(
    api_key: str,
    *,
    series: Mapping[str, float] = FRED_TENORS,
    fetch: FredFetcher = _default_fred_fetcher,
) -> RiskFreeCurve:
    """Build a curve from FRED Treasury series.

    ``fetch`` is injectable for testing. Series whose value cannot be parsed
    (FRED returns ``"."`` for missing) are skipped.
    """
    pairs: list[tuple[float, float]] = []
    for series_id, tenor in series.items():
        try:
            pct = fetch(series_id, api_key)
        except (ValueError, KeyError):
            continue
        pairs.append((tenor, pct / 100.0))
    if not pairs:
        raise ValueError("no usable FRED observations returned")
    pairs.sort()
    tenors, rates = zip(*pairs, strict=True)
    return RiskFreeCurve(tenors=tenors, rates=rates)
