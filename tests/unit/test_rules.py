import numpy as np
import pandas as pd
import pytest

from quant_research.transform.rules import Reason, validate_bars


def bar(minute, o=25000.0, h=25020.0, lo=24990.0, c=25010.0, v=100, sym="NQ"):
    return {
        "symbol": sym,
        "timestamp": f"2026-10-26 14:{minute:02d}:00",
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "volume": v,
    }


def run(rows):
    return validate_bars(pd.DataFrame(rows))


def test_clean_rows_pass_through():
    res = run([bar(0), bar(1), bar(2)])
    assert len(res.silver) == 3
    assert res.quarantine.empty
    assert res.report.quarantine_rate == 0
    assert str(res.silver["timestamp"].dt.tz) == "UTC"


@pytest.mark.parametrize(
    "row,reason",
    [
        (bar(1, c=np.nan), Reason.NULL),
        (bar(1, o=0.0, lo=0.0), Reason.NON_POSITIVE),
        (bar(1, h=24980.0), Reason.OHLC),
        (bar(1, lo=25005.0, o=25000.0), Reason.OHLC),
        (bar(1, v=-1), Reason.NEGATIVE_VOLUME),
        (bar(1, c=25010.1), Reason.OFF_TICK),
        ({**bar(1), "timestamp": "not-a-time"}, Reason.SCHEMA),
    ],
)
def test_each_rule(row, reason):
    res = run([bar(0), row])
    assert len(res.silver) == 1
    assert res.quarantine["reason"].tolist() == [reason.value]


def test_conflicting_duplicates_quarantine_every_version():
    res = run([bar(0), bar(1), bar(1, c=25015.0)])
    assert len(res.silver) == 1
    assert (res.quarantine["reason"] == Reason.CONFLICT.value).sum() == 2


def test_exact_duplicates_kept_once():
    res = run([bar(0), bar(0), bar(0), bar(1)])
    assert len(res.silver) == 2
    assert res.report.exact_duplicates_dropped == 2


def test_first_failing_rule_wins():
    # null close also breaks nothing else measurable -> null wins over off-tick open
    res = run([bar(0, o=25000.1, c=np.nan)])
    assert res.quarantine["reason"].iloc[0] == Reason.NULL.value


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing columns"):
        validate_bars(pd.DataFrame([{"symbol": "NQ"}]))


def test_gaps_and_spikes_on_synthetic(clean_bars):
    dropped = clean_bars.drop(index=[100, 101, 102])
    res = validate_bars(dropped)
    assert res.report.intrasession_gaps == 3
    spiky = clean_bars.copy()
    i = 2000
    for c in ("open", "high", "low", "close"):
        spiky.loc[i, c] = spiky.loc[i, c] * 1.05 // 0.25 * 0.25
    spiky.loc[i, "high"] = spiky.loc[i, ["open", "high", "low", "close"]].max()
    spiky.loc[i, "low"] = spiky.loc[i, ["open", "high", "low", "close"]].min()
    res = validate_bars(spiky)
    assert res.silver["is_spike"].sum() >= 1
    assert len(res.silver) == len(spiky)  # flagged, not removed
