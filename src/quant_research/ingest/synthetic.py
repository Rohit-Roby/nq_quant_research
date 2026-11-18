"""Deterministic synthetic NQ 1-minute bar generator.

Used when no vendor data is available (CI, demos, this repo's default config).
It is designed to *look* like NQ microstructure so the pipeline is exercised
realistically, not to contain tradable alpha:

* CME Globex hours: Sun-Fri 18:00-17:00 ET with the daily 17:00-18:00 halt
* intraday volatility U-shape with a spike at the 09:30 ET cash open
* GARCH(1,1) volatility clustering
* prices rounded to the 0.25 tick, volume correlated with volatility
* optional injected defects (duplicates, conflicting duplicates, bad OHLC, nulls,
  negative volume, off-tick prices, dropped bars) so Silver validation has work to do

Any backtest result on this data is a pipeline/engine demonstration, not evidence of edge.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quant_research import BAR_COLUMNS

ET = "America/New_York"
MINUTES_PER_YEAR = 252 * 23 * 60


def globex_minutes(start: date, end: date) -> pd.DatetimeIndex:
    """All 1-minute bar open times (UTC) for Globex sessions whose trade date is in range."""
    days = pd.bdate_range(start, end)
    stamps: list[pd.DatetimeIndex] = []
    for d in days:
        # Session for trade date d opens 18:00 ET on the previous calendar day.
        open_et = pd.Timestamp(d.date(), tz=ET) - pd.Timedelta(hours=6)
        close_et = pd.Timestamp(d.date(), tz=ET) + pd.Timedelta(hours=17)
        stamps.append(pd.date_range(open_et, close_et, freq="1min", inclusive="left"))
    if not stamps:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.DatetimeIndex(stamps[0].append(stamps[1:])).tz_convert("UTC")


def intraday_vol_profile(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    """Relative volatility multiplier by time of day (ET)."""
    et = ts_utc.tz_convert(ET)
    mins = et.hour * 60 + et.minute
    rth = (mins >= 570) & (mins < 960)
    prof = np.where(rth, 1.6, 0.55)
    since_open = mins - 570
    prof = prof + np.where((since_open >= 0) & (since_open < 30), 2.2 * np.exp(-since_open / 8), 0)
    to_close = 960 - mins
    prof = prof + np.where((to_close > 0) & (to_close <= 15), 0.8, 0)
    # 08:30 ET macro releases
    prof = prof + np.where((mins >= 510) & (mins < 515), 1.5, 0)
    return np.asarray(prof / prof.mean(), dtype=float)


def generate_bars(
    start: date,
    end: date,
    symbol: str = "NQ",
    seed: int = 42,
    start_price: float = 21_000.0,
    annual_vol: float = 0.22,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """Clean synthetic 1-minute OHLCV bars."""
    ts = globex_minutes(start, end)
    n = len(ts)
    if n == 0:
        return pd.DataFrame(columns=BAR_COLUMNS)
    rng = np.random.default_rng(seed)

    base_var = (annual_vol**2) / MINUTES_PER_YEAR
    omega, alpha, beta = base_var * 0.02, 0.06, 0.92
    shocks = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    var = np.empty(n)
    ret = np.empty(n)
    var[0] = base_var
    prof = intraday_vol_profile(ts)
    for i in range(n):
        if i:
            var[i] = omega + alpha * ret[i - 1] ** 2 / prof[i - 1] + beta * var[i - 1]
        ret[i] = np.sqrt(var[i] * prof[i]) * shocks[i]
    ret += 0.05 / MINUTES_PER_YEAR  # small positive drift

    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[start_price], close[:-1]])
    sigma = np.sqrt(var * prof) * close
    wick_hi = np.abs(rng.normal(0, 0.6, n)) * sigma
    wick_lo = np.abs(rng.normal(0, 0.6, n)) * sigma
    high = np.maximum(open_, close) + wick_hi
    low = np.minimum(open_, close) - wick_lo

    def to_tick(x: np.ndarray) -> np.ndarray:
        return np.round(x / tick_size) * tick_size

    o, c = to_tick(open_), to_tick(close)
    h = np.maximum(to_tick(high), np.maximum(o, c))
    lo = np.minimum(to_tick(low), np.minimum(o, c))
    vol = rng.poisson(lam=np.clip(40 * prof * np.sqrt(var / base_var), 1, None) * 8)

    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": ts,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": vol.astype("int64"),
        }
    )


def inject_defects(df: pd.DataFrame, rate: float = 0.002, seed: int = 7) -> pd.DataFrame:
    """Corrupt a small fraction of rows the way real vendor feeds do."""
    if rate <= 0 or df.empty:
        return df.copy()
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    k = max(1, int(n * rate / 7))

    def pick() -> np.ndarray:
        return rng.choice(n, size=k, replace=False)

    out.loc[pick(), "high"] = out["low"] - 1.0  # ohlc_violation
    out.loc[pick(), "close"] = np.nan  # null_field
    out.loc[pick(), "volume"] = -1  # negative_volume
    out.loc[pick(), "open"] = out["open"] + 0.1  # off_tick (may break ohlc)
    exact = out.iloc[pick()]  # exact duplicates
    conflict = out.iloc[pick()].copy()  # conflicting duplicates
    conflict["close"] = conflict["close"] + 2.0
    conflict["high"] = np.maximum(conflict["high"], conflict["close"])
    drop = pick()  # missing bars
    out = pd.concat([out.drop(out.index[drop]), exact, conflict], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
