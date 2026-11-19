"""Bronze -> Silver data-quality rules (pandas engine).

The same rule set is implemented for Spark in :mod:`quant_research.transform.validation`
so the local pipeline / Airflow tasks and the AWS Glue job produce identical Silver data.
Parity is enforced by ``tests/unit/test_engine_parity.py``.

Rule order (first failing rule wins the quarantine ``reason``):

1. ``schema_error``          symbol/timestamp missing or unparseable
2. ``conflicting_duplicate`` same (symbol, timestamp) with *different* values -> every
                             version is quarantined because we cannot know which is right
3. ``null_field``            any OHLCV value missing
4. ``non_positive_price``    any price <= 0
5. ``ohlc_violation``        high < max(open, close, low) or low > min(open, close)
6. ``negative_volume``       volume < 0
7. ``off_tick``              a price not on the contract tick grid (0.25 for NQ/MNQ)

Exact duplicates are then dropped (kept once) and counted, and price spikes are
*flagged* (``is_spike``) rather than removed, because a real fast market looks like a spike.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from quant_research import BAR_COLUMNS, PRICE_COLUMNS

KEY = ["symbol", "timestamp"]


class Reason(StrEnum):
    SCHEMA = "schema_error"
    CONFLICT = "conflicting_duplicate"
    NULL = "null_field"
    NON_POSITIVE = "non_positive_price"
    OHLC = "ohlc_violation"
    NEGATIVE_VOLUME = "negative_volume"
    OFF_TICK = "off_tick"


@dataclass
class DQReport:
    rows_in: int
    rows_silver: int
    exact_duplicates_dropped: int
    quarantined_by_reason: dict[str, int] = field(default_factory=dict)
    spikes_flagged: int = 0
    intrasession_gaps: int = 0

    @property
    def rows_quarantined(self) -> int:
        return sum(self.quarantined_by_reason.values())

    @property
    def quarantine_rate(self) -> float:
        return self.rows_quarantined / self.rows_in if self.rows_in else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rows_quarantined"] = self.rows_quarantined
        d["quarantine_rate"] = round(self.quarantine_rate, 6)
        return d


@dataclass
class ValidationResult:
    silver: pd.DataFrame
    quarantine: pd.DataFrame
    report: DQReport


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast raw bronze columns to the canonical bar schema (UTC timestamps, float prices)."""
    missing = set(BAR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"bronze data missing columns: {sorted(missing)}")
    out = df[BAR_COLUMNS].copy()
    out["symbol"] = out["symbol"].astype("string")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for c in PRICE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("Float64")
    return out


def off_tick_mask(df: pd.DataFrame, tick_size: float) -> pd.Series:
    ticks = df[PRICE_COLUMNS] / tick_size
    return (ticks - ticks.round()).abs().gt(1e-6).any(axis=1)


def conflicting_keys_mask(df: pd.DataFrame) -> pd.Series:
    distinct = df.drop_duplicates(subset=BAR_COLUMNS)
    counts = distinct.groupby(KEY, dropna=False).size()
    bad = counts[counts > 1].index
    if len(bad) == 0:
        return pd.Series(False, index=df.index)
    return pd.MultiIndex.from_frame(df[KEY]).isin(bad)  # type: ignore[return-value]


def flag_spikes(silver: pd.DataFrame, k: float = 12.0, window: int = 120) -> pd.Series:
    """Flag bars whose |log return| exceeds ``k`` x rolling median absolute return."""
    if silver.empty:
        return pd.Series(dtype=bool)
    r = np.log(silver["close"]).groupby(silver["symbol"]).diff().abs()
    scale = r.groupby(silver["symbol"]).transform(
        lambda s: s.rolling(window, min_periods=20).median().shift(1)
    )
    return (r > k * scale).fillna(False).astype(bool)


def count_intrasession_gaps(
    silver: pd.DataFrame, bar: str = "1min", tz: str = "America/New_York"
) -> int:
    """Count missing bars inside a Globex session (ignores the 17:00-18:00 ET halt/weekends)."""
    if silver.empty:
        return 0
    gaps = 0
    step = pd.Timedelta(bar)
    for _, g in silver.groupby("symbol"):
        ts = g["timestamp"].sort_values()
        session = (ts.dt.tz_convert(tz) + pd.Timedelta(hours=6)).dt.date
        diffs = ts.diff()
        same_session = session.eq(session.shift())
        n_steps = diffs.dt.total_seconds() / step.total_seconds()
        missing = (n_steps - 1).where(same_session & diffs.gt(step), 0)
        gaps += int(missing.fillna(0).sum())
    return gaps


def validate_bars(bronze: pd.DataFrame, tick_size: float = 0.25) -> ValidationResult:
    """Apply all Silver rules: clean rows, quarantined rows (with ``reason``) and a report."""
    rows_in = len(bronze)
    df = coerce_schema(bronze).reset_index(drop=True)
    reason = pd.Series(pd.NA, index=df.index, dtype="string")

    def assign(mask: pd.Series, r: Reason) -> None:
        reason.loc[mask & reason.isna()] = r.value

    assign(df["symbol"].isna() | df["timestamp"].isna(), Reason.SCHEMA)
    assign(conflicting_keys_mask(df) & reason.isna(), Reason.CONFLICT)
    assign(df[PRICE_COLUMNS + ["volume"]].isna().any(axis=1), Reason.NULL)

    filled = df[PRICE_COLUMNS].fillna(np.nan)
    assign(filled.le(0).any(axis=1), Reason.NON_POSITIVE)
    ohlc_bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close"]].min(axis=1)
    )
    assign(ohlc_bad, Reason.OHLC)
    assign(df["volume"].lt(0).fillna(False).astype(bool), Reason.NEGATIVE_VOLUME)
    assign(off_tick_mask(df.fillna({c: 0.0 for c in PRICE_COLUMNS}), tick_size), Reason.OFF_TICK)

    bad = reason.notna()
    quarantine = df[bad].assign(reason=reason[bad]).reset_index(drop=True)

    clean = df[~bad]
    before = len(clean)
    clean = clean.drop_duplicates(subset=BAR_COLUMNS)
    dropped = before - len(clean)
    silver = clean.sort_values(KEY).reset_index(drop=True)
    silver["volume"] = silver["volume"].astype("int64")
    silver["is_spike"] = flag_spikes(silver).to_numpy() if len(silver) else []

    report = DQReport(
        rows_in=rows_in,
        rows_silver=len(silver),
        exact_duplicates_dropped=dropped,
        quarantined_by_reason={
            str(k): int(v) for k, v in quarantine["reason"].value_counts().sort_index().items()
        },
        spikes_flagged=int(silver["is_spike"].sum()) if len(silver) else 0,
        intrasession_gaps=count_intrasession_gaps(silver),
    )
    return ValidationResult(silver=silver, quarantine=quarantine, report=report)
