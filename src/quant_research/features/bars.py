"""Silver 1-minute bars -> Gold research bars with session-aware features."""

from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"
RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60


def resample_bars(bars_1m: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    """OHLCV resample per symbol. Bars are labelled by their open time (left/left)."""
    if bars_1m.empty:
        return bars_1m.copy()
    rule = f"{minutes}min"
    frames = []
    for sym, g in bars_1m.groupby("symbol"):
        g = g.set_index(pd.to_datetime(g["timestamp"], utc=True)).sort_index()
        agg = (
            g.resample(rule, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["open"])
        )
        agg["symbol"] = sym
        frames.append(agg.reset_index(names="timestamp"))
    out = pd.concat(frames, ignore_index=True)
    return out[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    et = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(ET)
    out["trade_date"] = (et + pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%d")
    out["minute_of_day"] = (et.dt.hour * 60 + et.dt.minute).astype("int32")
    out["is_rth"] = out["minute_of_day"].between(RTH_OPEN_MIN, RTH_CLOSE_MIN - 1)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price anchored at each RTH open; NaN outside RTH."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    rth = df["is_rth"]
    pv = (tp * df["volume"]).where(rth, 0.0)
    v = df["volume"].where(rth, 0.0)
    keys = [df["symbol"], df["trade_date"]]
    cum_pv = pv.groupby(keys).cumsum()
    cum_v = v.groupby(keys).cumsum()
    return (cum_pv / cum_v.replace(0, np.nan)).where(rth)


def build_gold_bars(bars_1m: pd.DataFrame, minutes: int = 5, atr_window: int = 14) -> pd.DataFrame:
    """Gold dataset consumed by strategies and the dashboard."""
    clean = bars_1m
    if "is_spike" in clean.columns:
        clean = clean.drop(columns=["is_spike"])
    bars = add_session_columns(resample_bars(clean, minutes))
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    sym = bars["symbol"]
    bars["ret"] = np.log(bars["close"]).groupby(sym).diff()
    prev_close = bars["close"].groupby(sym).shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    bars["atr"] = tr.groupby(sym).transform(
        lambda s: s.rolling(atr_window, min_periods=atr_window).mean()
    )
    bars["vwap"] = session_vwap(bars)
    bars["vwap_dev"] = bars["close"] - bars["vwap"]
    return bars


def daily_summary(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per trade date: RTH OHLC, volume and realised volatility."""
    rth = bars[bars["is_rth"]]
    agg = rth.groupby(["symbol", "trade_date"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        realised_vol=("ret", lambda r: float(np.sqrt((r**2).sum()) * np.sqrt(252))),
    )
    return agg.reset_index()


def opening_range(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """High/low of the first ``minutes`` of RTH per (symbol, trade_date)."""
    window = bars[bars["minute_of_day"].between(RTH_OPEN_MIN, RTH_OPEN_MIN + minutes - 1)]
    return (
        window.groupby(["symbol", "trade_date"])
        .agg(or_high=("high", "max"), or_low=("low", "min"))
        .reset_index()
    )
