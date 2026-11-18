"""Original Spark validation tests, kept and extended."""

import pytest

from quant_research.transform.validation import (
    BAR_SCHEMA,
    convert_timestamp,
    find_confliction_duplicates,
    process_market_data,
    validate_duplicates,
    validate_ohlcv,
)

pytestmark = pytest.mark.spark

RECORDS = [
    ("NQ", "2026-10-26 14:00:00", 25000.0, 25020.0, 24990.0, 25010.0, 1500),
    ("NQ", "2026-10-26 14:15:00", 25010.0, 25035.0, 25005.0, 25030.0, 1800),
    ("NQ", "2026-10-26 14:30:00", 25030.0, 25040.0, 25010.0, 25015.0, 1200),
    ("NQ", "2026-10-26 14:45:00", 25000.0, 24980.0, 24990.0, 25010.0, 1000),
    ("NQ", "2026-10-26 15:00:00", 25000.0, None, 24990.0, 25010.0, 1500),
]


def _df(spark, records):
    return spark.createDataFrame([convert_timestamp(r) for r in records], schema=BAR_SCHEMA)


def test_validate_ohlcv(spark):
    df = _df(spark, RECORDS)
    valid_df, invalid_df = validate_ohlcv(df)
    assert valid_df.count() == 3
    assert invalid_df.count() == 2
    assert validate_duplicates(df).count() == 0


def test_conflicting_duplicates_are_quarantined_exact_are_dropped(spark):
    records = [
        *RECORDS[:3],
        ("NQ", "2026-10-26 14:00:00", 25000.0, 25020.0, 24990.0, 25010.0, 1500),  # exact dup
        ("NQ", "2026-10-26 14:15:00", 25010.0, 25035.0, 25005.0, 25025.0, 1700),  # conflict
    ]
    df = _df(spark, records)
    assert find_confliction_duplicates(df).count() == 1
    silver, quarantine, stats = process_market_data(df)
    assert silver.count() == 2
    reasons = {r["reason"] for r in quarantine.collect()}
    assert reasons == {"conflicting_duplicate"}
    assert quarantine.count() == 2
    assert stats["exact_duplicates_dropped"] == 1


def test_reason_codes(spark):
    records = [
        ("NQ", "2026-10-26 14:00:00", 25000.0, 25020.0, 24990.0, 25010.0, 1500),
        ("NQ", "2026-10-26 14:01:00", 25000.1, 25020.0, 24990.0, 25010.0, 1500),  # off tick
        ("NQ", "2026-10-26 14:02:00", 25000.0, 25020.0, 24990.0, 25010.0, -5),  # neg vol
        ("NQ", "2026-10-26 14:03:00", 0.0, 25020.0, 0.0, 25010.0, 5),  # non-positive
    ]
    _, quarantine, _ = process_market_data(_df(spark, records))
    got = {r["timestamp"].minute: r["reason"] for r in quarantine.collect()}
    assert got == {1: "off_tick", 2: "negative_volume", 3: "non_positive_price"}
