import numpy as np

from quant_research.features.bars import daily_summary, opening_range, resample_bars


def test_resample_preserves_ohlcv(clean_bars):
    five = resample_bars(clean_bars, 5)
    first = clean_bars.iloc[:5]
    row = five.iloc[0]
    assert row["open"] == first["open"].iloc[0]
    assert row["close"] == first["close"].iloc[-1]
    assert row["high"] == first["high"].max()
    assert row["low"] == first["low"].min()
    assert row["volume"] == first["volume"].sum()
    assert five["volume"].sum() == clean_bars["volume"].sum()


def test_gold_columns_and_sessions(gold_bars):
    for c in ("trade_date", "minute_of_day", "is_rth", "ret", "atr", "vwap", "vwap_dev"):
        assert c in gold_bars
    assert gold_bars["trade_date"].nunique() == 10
    rth = gold_bars[gold_bars["is_rth"]]
    assert rth.groupby("trade_date").size().eq(78).all()  # 6.5h / 5m
    assert gold_bars.loc[~gold_bars["is_rth"], "vwap"].isna().all()


def test_vwap_within_session_range(gold_bars):
    rth = gold_bars[gold_bars["is_rth"]]
    g = rth.groupby("trade_date")
    assert (rth["vwap"] <= g["high"].transform("cummax") + 1e-9).all()
    assert (rth["vwap"] >= g["low"].transform("cummin") - 1e-9).all()


def test_opening_range_and_daily(gold_bars):
    orng = opening_range(gold_bars, 30)
    assert len(orng) == 10 and (orng["or_high"] >= orng["or_low"]).all()
    daily = daily_summary(gold_bars)
    assert len(daily) == 10 and np.isfinite(daily["realised_vol"]).all()
