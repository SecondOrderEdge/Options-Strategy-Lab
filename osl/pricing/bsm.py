"""Vectorized Black-Scholes-Merton pricing and greeks (continuous dividend yield).

Conventions (see blueprint §4.1):

* ``right`` is ``"c"``/``"p"`` (case-insensitive) or an array thereof.
* ``T`` is time to expiry in years (ACT/365F); ``r``, ``q``, ``sigma`` are
  decimals (0.20 == 20% vol).
* **theta** is returned **per calendar day** (annual θ ÷ 365).
* **vega** is returned **per 1 vol point** (∂V/∂σ × 0.01).
* **rho** is returned **per 1.00** rate change (the UI scales to per-bp).
* second-order greeks (vanna, charm, vomma) use per-1.00 σ / per-day scalings
  consistent with the first-order set.

All functions broadcast their inputs and return ``float64`` arrays (0-d for
scalar inputs); call ``.item()`` for a Python float.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

FloatArray = npt.NDArray[np.float64]

DAYS_PER_YEAR = 365.0
VOL_POINT = 0.01


def _arr(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _call_mask(right: npt.ArrayLike, shape: tuple[int, ...]) -> npt.NDArray[np.bool_]:
    """Broadcast ``right`` to a boolean call-mask (True == call)."""
    raw = np.asarray(right)
    if raw.dtype.kind in ("U", "S", "O"):
        flat = np.array([str(v).lower().startswith("c") for v in raw.ravel()])
        mask = flat.reshape(raw.shape) if raw.shape else flat.reshape(())
    else:
        mask = raw.astype(np.float64) > 0
    return np.broadcast_to(mask, shape)


def _pdf(x: FloatArray) -> FloatArray:
    return np.asarray(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi), dtype=np.float64)


def _cdf(x: FloatArray) -> FloatArray:
    return np.asarray(ndtr(x), dtype=np.float64)


def _d1_d2(
    S: FloatArray, K: FloatArray, T: FloatArray, r: FloatArray, q: FloatArray, sigma: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    sqrt_t = np.sqrt(np.where(T > 0, T, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
    return _arr(d1), _arr(d2), _arr(sqrt_t)


def bsm_price(
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    q: npt.ArrayLike,
    sigma: npt.ArrayLike,
    right: npt.ArrayLike,
) -> FloatArray:
    """Black-Scholes-Merton option price."""
    S_, K_, T_, r_, q_, sig_ = (_arr(x) for x in (S, K, T, r, q, sigma))
    bshape = np.broadcast_shapes(S_.shape, K_.shape, T_.shape, r_.shape, q_.shape, sig_.shape)
    call = _call_mask(right, bshape)

    d1, d2, _ = _d1_d2(S_, K_, T_, r_, q_, sig_)
    disc_q = np.exp(-q_ * T_)
    disc_r = np.exp(-r_ * T_)

    call_px = S_ * disc_q * _cdf(d1) - K_ * disc_r * _cdf(d2)
    put_px = K_ * disc_r * _cdf(-d2) - S_ * disc_q * _cdf(-d1)
    bsm = np.where(call, call_px, put_px)

    intrinsic = np.where(call, np.maximum(S_ - K_, 0.0), np.maximum(K_ - S_, 0.0))
    valid = (T_ > 0) & (sig_ > 0)
    return _arr(np.where(valid, bsm, intrinsic))


def bsm_greeks(
    S: npt.ArrayLike,
    K: npt.ArrayLike,
    T: npt.ArrayLike,
    r: npt.ArrayLike,
    q: npt.ArrayLike,
    sigma: npt.ArrayLike,
    right: npt.ArrayLike,
) -> dict[str, FloatArray]:
    """Return greeks: delta, gamma, vega, theta, rho, vanna, charm, vomma.

    See module docstring for scalings (theta/day, vega/vol-point, rho/1.00).
    """
    S_, K_, T_, r_, q_, sig_ = (_arr(x) for x in (S, K, T, r, q, sigma))
    bshape = np.broadcast_shapes(S_.shape, K_.shape, T_.shape, r_.shape, q_.shape, sig_.shape)
    call = _call_mask(right, bshape)

    d1, d2, sqrt_t = _d1_d2(S_, K_, T_, r_, q_, sig_)
    disc_q = np.exp(-q_ * T_)
    disc_r = np.exp(-r_ * T_)
    pdf_d1 = _pdf(d1)

    delta = np.where(call, disc_q * _cdf(d1), -disc_q * _cdf(-d1))
    gamma = disc_q * pdf_d1 / (S_ * sig_ * sqrt_t)
    vega_raw = S_ * disc_q * pdf_d1 * sqrt_t  # per 1.00 vol

    theta_common = -(S_ * disc_q * pdf_d1 * sig_) / (2.0 * sqrt_t)
    theta_call = theta_common - r_ * K_ * disc_r * _cdf(d2) + q_ * S_ * disc_q * _cdf(d1)
    theta_put = theta_common + r_ * K_ * disc_r * _cdf(-d2) - q_ * S_ * disc_q * _cdf(-d1)
    theta_annual = np.where(call, theta_call, theta_put)

    rho = np.where(call, K_ * T_ * disc_r * _cdf(d2), -K_ * T_ * disc_r * _cdf(-d2))

    vanna = -disc_q * pdf_d1 * d2 / sig_  # ∂delta/∂σ, same for calls/puts

    charm_common = (
        -disc_q * pdf_d1 * (2.0 * (r_ - q_) * T_ - d2 * sig_ * sqrt_t) / (2.0 * T_ * sig_ * sqrt_t)
    )
    charm_call = q_ * disc_q * _cdf(d1) + charm_common
    charm_put = -q_ * disc_q * _cdf(-d1) + charm_common
    charm_annual = np.where(call, charm_call, charm_put)  # ∂delta/∂t per year

    vomma = vega_raw * d1 * d2 / sig_  # ∂vega_raw/∂σ

    valid = (T_ > 0) & (sig_ > 0)
    zero = np.zeros(bshape)

    def _g(value: npt.ArrayLike) -> FloatArray:
        return _arr(np.where(valid, np.broadcast_to(value, bshape), zero))

    return {
        "delta": _g(delta),
        "gamma": _g(gamma),
        "vega": _g(vega_raw * VOL_POINT),
        "theta": _g(theta_annual / DAYS_PER_YEAR),
        "rho": _g(rho),
        "vanna": _g(vanna),
        "charm": _g(charm_annual / DAYS_PER_YEAR),
        "vomma": _g(vomma),
    }
