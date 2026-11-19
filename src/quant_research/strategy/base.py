"""Strategy interface.

A strategy maps Gold bars to a **target position per bar, decided at that bar's close**
using only information available at that close. The backtest engine shifts it by one
bar (fill at the next bar's open), so look-ahead is impossible by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "symbol",
    "timestamp",
    "trade_date",
    "minute_of_day",
    "is_rth",
    "open",
    "high",
    "low",
    "close",
}


def parse_hhmm(value: str | int) -> int:
    if isinstance(value, int):
        return value
    hh, mm = str(value).split(":")
    return int(hh) * 60 + int(mm)


class Strategy(ABC):
    name: ClassVar[str]
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, bar_minutes: int = 5, **params: Any) -> None:
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}")
        self.bar_minutes = bar_minutes
        self.params: dict[str, Any] = {**self.default_params, **params}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"

    def exit_minute(self) -> int:
        return parse_hhmm(self.params.get("exit_time", "15:55"))

    def flat_mask(self, bars: pd.DataFrame) -> pd.Series:
        """Bars whose close is at/after the exit time or outside RTH -> must be flat."""
        closes_at = bars["minute_of_day"] + self.bar_minutes
        return (~bars["is_rth"]) | (closes_at >= self.exit_minute())

    def target_positions(self, bars: pd.DataFrame) -> pd.Series:
        missing = REQUIRED_COLUMNS - set(bars.columns)
        if missing:
            raise ValueError(f"bars missing columns {sorted(missing)}")
        pos = self._positions(bars.reset_index(drop=True))
        pos = pos.where(~self.flat_mask(bars.reset_index(drop=True)), 0.0)
        out = pd.Series(np.sign(pos.to_numpy(dtype=float)), index=bars.index, name=self.name)
        return out

    @abstractmethod
    def _positions(self, bars: pd.DataFrame) -> pd.Series:
        """Raw target position in {-1, 0, +1} per bar (index aligned to ``bars``)."""
