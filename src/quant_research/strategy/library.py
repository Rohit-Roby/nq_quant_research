"""Concrete intraday NQ strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_research.features.bars import RTH_OPEN_MIN, opening_range
from quant_research.strategy.base import Strategy


class OpeningRangeBreakout(Strategy):
    """Trade the first close outside the RTH opening range; ATR stop; flat by exit_time.

    At most one trade per session. Stop is evaluated on bar closes (conservative for a
    close-based system: the fill is at the next bar's open, which includes the slippage).
    """

    name = "opening_range_breakout"
    default_params = {"or_minutes": 30, "stop_atr_mult": 1.5, "exit_time": "15:55"}

    def _positions(self, bars: pd.DataFrame) -> pd.Series:
        orng = opening_range(bars, int(self.params["or_minutes"]))
        b = bars.merge(orng, on=["symbol", "trade_date"], how="left")
        start = RTH_OPEN_MIN + int(self.params["or_minutes"])
        k = float(self.params["stop_atr_mult"])
        exit_min = self.exit_minute()
        pos = np.zeros(len(b))

        for _, idx in b.groupby(["symbol", "trade_date"]).indices.items():
            day = b.iloc[idx]
            close = day["close"].to_numpy()
            atr = day["atr"].to_numpy() if "atr" in day else np.full(len(day), np.nan)
            mod = day["minute_of_day"].to_numpy()
            hi, lo = day["or_high"].to_numpy(), day["or_low"].to_numpy()
            state, stop, traded = 0, np.nan, False
            for j in range(len(day)):
                eligible = (mod[j] + self.bar_minutes) >= start and (
                    mod[j] + self.bar_minutes
                ) < exit_min
                if (
                    state == 0
                    and not traded
                    and eligible
                    and np.isfinite(hi[j])
                    and np.isfinite(atr[j])
                ):
                    if close[j] > hi[j]:
                        state, stop, traded = 1, close[j] - k * atr[j], True
                    elif close[j] < lo[j]:
                        state, stop, traded = -1, close[j] + k * atr[j], True
                elif (state == 1 and close[j] <= stop) or (state == -1 and close[j] >= stop):
                    state = 0
                pos[idx[j]] = state
        return pd.Series(pos, index=bars.index)


class VwapReversion(Strategy):
    """Fade stretched moves away from session VWAP; exit on the VWAP cross or at exit_time."""

    name = "vwap_reversion"
    default_params = {"entry_z": 2.0, "lookback": 24, "exit_time": "15:55", "skip_open_minutes": 15}

    def _positions(self, bars: pd.DataFrame) -> pd.Series:
        lookback = int(self.params["lookback"])
        entry_z = float(self.params["entry_z"])
        dev = bars["close"] - bars["vwap"]
        sd = dev.groupby([bars["symbol"], bars["trade_date"]]).transform(
            lambda s: s.rolling(lookback, min_periods=max(4, lookback // 2)).std()
        )
        z = (dev / sd.replace(0, np.nan)).to_numpy()
        d = dev.to_numpy()
        mod = bars["minute_of_day"].to_numpy()
        earliest = RTH_OPEN_MIN + int(self.params["skip_open_minutes"])
        pos = np.zeros(len(bars))

        for _, idx in bars.groupby(["symbol", "trade_date"]).indices.items():
            state = 0
            for i in idx:
                if state == 0:
                    if mod[i] >= earliest and np.isfinite(z[i]):
                        if z[i] > entry_z:
                            state = -1
                        elif z[i] < -entry_z:
                            state = 1
                elif (state == 1 and d[i] >= 0) or (state == -1 and d[i] <= 0):
                    state = 0
                pos[i] = state
        return pd.Series(pos, index=bars.index)


class RthBuyAndHold(Strategy):
    """Benchmark: long from the first RTH bar, flat at exit_time, never overnight."""

    name = "rth_buy_and_hold"
    default_params = {"exit_time": "15:55"}

    def _positions(self, bars: pd.DataFrame) -> pd.Series:
        return bars["is_rth"].astype(float)


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (OpeningRangeBreakout, VwapReversion, RthBuyAndHold)
}


def make_strategy(name: str, bar_minutes: int = 5, **params: Any) -> Strategy:
    try:
        return STRATEGIES[name](bar_minutes=bar_minutes, **params)
    except KeyError as exc:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from exc
