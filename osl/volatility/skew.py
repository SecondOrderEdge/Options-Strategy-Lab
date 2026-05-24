"""25-delta risk reversal and butterfly from a fitted smile.

Risk reversal = IV(25Δ put) − IV(25Δ call); a positive value means downside
puts are bid relative to calls (the usual equity skew). Butterfly =
½(IV(25Δp)+IV(25Δc)) − IV(ATM), the smile's curvature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from osl.surface.prepare import Smile


@dataclass(frozen=True)
class DeltaSkew:
    risk_reversal_25: float  # IV(25Δ put) - IV(25Δ call)
    butterfly_25: float  # 0.5*(IV25p+IV25c) - IV(ATM)
    atm_iv: float


def delta_skew_25(smile: Smile) -> DeltaSkew:
    """Compute the 25Δ risk reversal and butterfly by interpolating IV vs call-delta."""
    sig_t = smile.iv * np.sqrt(smile.T)
    # Forward call delta (undiscounted): N(d1), d1 = (-k + 0.5 sigma^2 T)/(sigma sqrt(T)).
    d1 = (-smile.k + 0.5 * smile.iv * smile.iv * smile.T) / sig_t
    call_delta = np.asarray(ndtr(d1), dtype=np.float64)

    order = np.argsort(call_delta)
    deltas = call_delta[order]
    ivs = smile.iv[order]

    iv_25c = float(np.interp(0.25, deltas, ivs))  # 25Δ call (high strike)
    iv_25p = float(np.interp(0.75, deltas, ivs))  # 25Δ put (call-delta 0.75)
    atm_iv = float(np.interp(0.5, deltas, ivs))

    return DeltaSkew(
        risk_reversal_25=iv_25p - iv_25c,
        butterfly_25=0.5 * (iv_25p + iv_25c) - atm_iv,
        atm_iv=atm_iv,
    )
