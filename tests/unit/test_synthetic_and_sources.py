from datetime import date

import pandas as pd

from quant_research.ingest.sources import CsvSource, SyntheticSource
from quant_research.ingest.synthetic import generate_bars, inject_defects
from quant_research.transform.rules import validate_bars


def test_globex_hours_and_halt(clean_bars):
    et = clean_bars["timestamp"].dt.tz_convert("America/New_York")
    assert not (et.dt.hour == 17).any(), "17:00-18:00 ET halt must be empty"
    assert not (et.dt.dayofweek == 5).any(), "no Saturday bars"
    assert len(clean_bars) == 10 * 23 * 60  # 10 sessions x 23h


def test_clean_bars_are_valid(clean_bars):
    res = validate_bars(clean_bars)
    assert res.report.rows_quarantined == 0
    assert res.report.intrasession_gaps == 0


def test_deterministic():
    a = generate_bars(date(2025, 3, 3), date(2025, 3, 4), seed=1)
    b = generate_bars(date(2025, 3, 3), date(2025, 3, 4), seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_defects_are_caught(clean_bars):
    dirty = inject_defects(clean_bars, rate=0.01, seed=5)
    res = validate_bars(dirty)
    assert res.report.rows_quarantined > 0
    assert res.report.exact_duplicates_dropped > 0
    assert set(res.report.quarantined_by_reason) >= {
        "ohlc_violation",
        "null_field",
        "negative_volume",
        "conflicting_duplicate",
    }


def test_synthetic_source_subrange_is_consistent():
    src = SyntheticSource("NQ", anchor=date(2025, 3, 3), defect_rate=0.0)
    full = src.fetch(date(2025, 3, 3), date(2025, 3, 7))
    part = src.fetch(date(2025, 3, 5), date(2025, 3, 5))
    merged = full.merge(part, on=["timestamp"], suffixes=("", "_p"))
    assert len(merged) == len(part) > 0
    assert (merged["close"] == merged["close_p"]).all()
    assert {"source", "ingested_at"} <= set(full.columns)


def test_csv_source_normalises_vendor_headers(tmp_path):
    raw = (
        generate_bars(date(2025, 3, 3), date(2025, 3, 3))
        .rename(columns={"timestamp": "ts_event", "volume": "Vol"})
        .drop(columns=["symbol"])
    )
    raw["ts_event"] = raw["ts_event"].dt.tz_convert(None)
    path = tmp_path / "nq.csv"
    raw.to_csv(path, index=False)
    out = CsvSource(path, symbol="NQ").fetch(date(2025, 3, 3), date(2025, 3, 3))
    assert len(out) == len(raw)
    assert out["symbol"].eq("NQ").all()
    assert str(out["timestamp"].dt.tz) == "UTC"
