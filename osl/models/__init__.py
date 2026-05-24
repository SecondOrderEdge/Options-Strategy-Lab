"""Advanced (experimental) models: Heston, Merton jumps, local vol.

Gated by ``enable_experimental`` at the UI layer. The library functions are
always importable; they require the ``QuantLib`` runtime dependency.
"""

from __future__ import annotations

from osl.models.heston import HestonParams, calibrate_heston, heston_price
from osl.models.jump_diffusion import MertonParams, merton_price
from osl.models.local_vol import dupire_local_vol

__all__ = [
    "HestonParams",
    "MertonParams",
    "calibrate_heston",
    "dupire_local_vol",
    "heston_price",
    "merton_price",
]
