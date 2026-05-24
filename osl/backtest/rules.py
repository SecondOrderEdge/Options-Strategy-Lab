"""Entry/exit systems for the backtester, including three reference strategies.

A *system* pairs an entry rule (select + size a position) with an exit rule
(return a reason string to close, or ``None``). The three reference systems:

1. 45-DTE 16Δ short put, closed at 21 DTE.
2. 45-DTE ~1σ iron condor, closed at 50% credit captured or 21 DTE.
3. 30-DTE ATM long straddle, closed at 100% gain or 14 DTE.
"""

from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from osl.backtest.config import BacktestConfig
from osl.backtest.portfolio import BTLeg, BTPosition, OpenOrder, position_value_mid
from osl.pricing.bsm import bsm_greeks
from osl.utils.time import DAYS_PER_YEAR

if TYPE_CHECKING:
    from osl.backtest.engine import BTDay

_RATE = 0.0  # rate used only for delta-based strike selection


def _nearest_expiry(chain: pd.DataFrame, target_dte: int) -> tuple[object, int] | None:
    dtes = chain.groupby(chain["expiration"].dt.date)["dte"].first()
    if dtes.empty:
        return None
    idx = (dtes - target_dte).abs().idxmin()
    return idx, int(dtes.loc[idx])


def _expiry_slice(chain: pd.DataFrame, expiration: object, right: str) -> pd.DataFrame:
    mask = (chain["expiration"].dt.date == expiration) & (chain["right"] == right)
    return cast(pd.DataFrame, chain[mask].sort_values("strike"))


def _atm_iv(chain: pd.DataFrame, expiration: object, spot: float) -> float:
    calls = _expiry_slice(chain, expiration, "C")
    if calls.empty:
        return float("nan")
    i = int((calls["strike"] - spot).abs().to_numpy().argmin())
    return float(calls["iv"].iloc[i])


def _mid(chain: pd.DataFrame, expiration: object, strike: float, right: str) -> float:
    sub = chain[
        (chain["expiration"].dt.date == expiration)
        & (np.isclose(chain["strike"].to_numpy(dtype=float), strike))
        & (chain["right"] == right)
    ]
    return float(sub["mid"].iloc[0]) if not sub.empty else float("nan")


def _size(capital: float, config: BacktestConfig, risk_per_unit: float) -> int:
    if risk_per_unit <= 0:
        return 1
    return max(1, int(config.max_position_pct * capital / risk_per_unit))


def _current_dte(position: BTPosition, today: date) -> int:
    return min((leg.expiration - today).days for leg in position.legs)


class ShortPut16Delta:
    """45-DTE 16Δ short put; close at 21 DTE."""

    name = "45D 16Δ Short Put"

    def __init__(
        self, target_dte: int = 45, target_delta: float = 0.16, exit_dte: int = 21
    ) -> None:
        self.target_dte = target_dte
        self.target_delta = target_delta
        self.exit_dte = exit_dte

    def enter(self, day: BTDay, capital: float, config: BacktestConfig) -> OpenOrder | None:
        nearest = _nearest_expiry(day.chain, self.target_dte)
        if nearest is None:
            return None
        expiration, dte = nearest
        puts = _expiry_slice(day.chain, expiration, "P")
        if puts.empty:
            return None
        T = dte / DAYS_PER_YEAR
        deltas = np.array(
            [
                abs(float(bsm_greeks(day.spot, k, T, _RATE, 0.0, iv, "p")["delta"].item()))
                for k, iv in zip(puts["strike"], puts["iv"], strict=True)
            ]
        )
        i = int(np.abs(deltas - self.target_delta).argmin())
        strike = float(puts["strike"].iloc[i])
        credit = _mid(day.chain, expiration, strike, "P")
        risk = strike * 100 - credit * 100
        qty = _size(capital, config, risk)
        leg = BTLeg("P", strike, expiration, side=-1, quantity=qty, multiplier=100)  # type: ignore[arg-type]
        return OpenOrder(self.name, (leg,), risk)

    def should_exit(self, day: BTDay, position: BTPosition) -> str | None:
        if _current_dte(position, day.date) <= self.exit_dte:
            return f"{self.exit_dte} DTE"
        return None


class IronCondor1Sigma:
    """45-DTE ~1σ iron condor; close at 50% credit captured or 21 DTE."""

    name = "45D 1σ Iron Condor"

    def __init__(
        self, target_dte: int = 45, wing: int = 1, exit_dte: int = 21, profit_target: float = 0.5
    ) -> None:
        self.target_dte = target_dte
        self.wing = wing
        self.exit_dte = exit_dte
        self.profit_target = profit_target

    def enter(self, day: BTDay, capital: float, config: BacktestConfig) -> OpenOrder | None:
        nearest = _nearest_expiry(day.chain, self.target_dte)
        if nearest is None:
            return None
        expiration, dte = nearest
        calls = _expiry_slice(day.chain, expiration, "C")
        puts = _expiry_slice(day.chain, expiration, "P")
        if len(calls) < self.wing + 2 or len(puts) < self.wing + 2:
            return None
        T = dte / DAYS_PER_YEAR
        iv = _atm_iv(day.chain, expiration, day.spot)
        if not math.isfinite(iv):
            return None
        move = day.spot * iv * math.sqrt(T)

        put_strikes = puts["strike"].to_numpy(dtype=float)
        call_strikes = calls["strike"].to_numpy(dtype=float)
        sp_i = int(np.abs(put_strikes - (day.spot - move)).argmin())
        sc_i = int(np.abs(call_strikes - (day.spot + move)).argmin())
        if sp_i - self.wing < 0 or sc_i + self.wing >= len(call_strikes):
            return None
        short_put, long_put = float(put_strikes[sp_i]), float(put_strikes[sp_i - self.wing])
        short_call, long_call = float(call_strikes[sc_i]), float(call_strikes[sc_i + self.wing])

        credit = (
            _mid(day.chain, expiration, short_put, "P")
            - _mid(day.chain, expiration, long_put, "P")
            + _mid(day.chain, expiration, short_call, "C")
            - _mid(day.chain, expiration, long_call, "C")
        )
        width = max(short_put - long_put, long_call - short_call)
        risk = max(width - credit, 0.01) * 100
        qty = _size(capital, config, risk)
        legs = (
            BTLeg("P", long_put, expiration, side=1, quantity=qty),  # type: ignore[arg-type]
            BTLeg("P", short_put, expiration, side=-1, quantity=qty),  # type: ignore[arg-type]
            BTLeg("C", short_call, expiration, side=-1, quantity=qty),  # type: ignore[arg-type]
            BTLeg("C", long_call, expiration, side=1, quantity=qty),  # type: ignore[arg-type]
        )
        return OpenOrder(self.name, legs, risk)

    def should_exit(self, day: BTDay, position: BTPosition) -> str | None:
        if _current_dte(position, day.date) <= self.exit_dte:
            return f"{self.exit_dte} DTE"
        credit = position.entry_cash  # positive net credit
        if credit > 0:
            profit = credit + position_value_mid(day.chain, position)
            if profit >= self.profit_target * credit:
                return f"{int(self.profit_target * 100)}% credit"
        return None


class LongStraddle:
    """30-DTE ATM long straddle; close at 100% gain or 14 DTE."""

    name = "30D ATM Long Straddle"

    def __init__(
        self, target_dte: int = 30, exit_dte: int = 14, profit_target: float = 1.0
    ) -> None:
        self.target_dte = target_dte
        self.exit_dte = exit_dte
        self.profit_target = profit_target

    def enter(self, day: BTDay, capital: float, config: BacktestConfig) -> OpenOrder | None:
        nearest = _nearest_expiry(day.chain, self.target_dte)
        if nearest is None:
            return None
        expiration, _dte = nearest
        calls = _expiry_slice(day.chain, expiration, "C")
        if calls.empty:
            return None
        i = int((calls["strike"] - day.spot).abs().to_numpy().argmin())
        strike = float(calls["strike"].iloc[i])
        debit = _mid(day.chain, expiration, strike, "C") + _mid(day.chain, expiration, strike, "P")
        risk = debit * 100
        qty = _size(capital, config, risk)
        legs = (
            BTLeg("C", strike, expiration, side=1, quantity=qty),  # type: ignore[arg-type]
            BTLeg("P", strike, expiration, side=1, quantity=qty),  # type: ignore[arg-type]
        )
        return OpenOrder(self.name, legs, risk)

    def should_exit(self, day: BTDay, position: BTPosition) -> str | None:
        if _current_dte(position, day.date) <= self.exit_dte:
            return f"{self.exit_dte} DTE"
        debit = -position.entry_cash  # positive cost paid
        if debit > 0:
            gain = position_value_mid(day.chain, position) + position.entry_cash
            if gain >= self.profit_target * debit:
                return f"{int(self.profit_target * 100)}% gain"
        return None
