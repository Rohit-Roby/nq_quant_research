"""Bronze -> Silver validation (PySpark engine, used by the AWS Glue job).

Mirrors :mod:`quant_research.transform.rules` rule-for-rule; see that module for the
rule order. This module has **no import-time side effects** (no SparkSession is created
at import), so it can be shipped to Glue via ``--extra-py-files`` and unit-tested with
the shared ``spark`` fixture.
"""

from __future__ import annotations

from datetime import datetime
from functools import reduce
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

PRICE_COLS = ["open", "high", "low", "close"]
BAR_COLS = ["symbol", "timestamp", *PRICE_COLS, "volume"]
KEY = ["symbol", "timestamp"]

BAR_SCHEMA = StructType(
    [
        StructField("symbol", StringType(), nullable=True),
        StructField("timestamp", TimestampType(), nullable=True),
        StructField("open", DoubleType(), nullable=True),
        StructField("high", DoubleType(), nullable=True),
        StructField("low", DoubleType(), nullable=True),
        StructField("close", DoubleType(), nullable=True),
        StructField("volume", LongType(), nullable=True),
    ]
)

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def convert_timestamp(record: tuple) -> tuple:
    """Parse the string timestamp in position 1 of a raw record tuple."""
    ts = datetime.strptime(record[1], TIMESTAMP_FORMAT)
    return record[:1] + (ts,) + record[2:]


def _all(cols: list[Column]) -> Column:
    return reduce(lambda a, b: a & b, cols)


def _any(cols: list[Column]) -> Column:
    return reduce(lambda a, b: a | b, cols)


def nan_to_null(df: DataFrame) -> DataFrame:
    """Spark treats NaN as a value (greater than every number), not as missing.

    Data arriving from pandas/CSV often encodes missing prices as NaN, so normalise to
    NULL first or the null rule is silently bypassed and the row is misclassified.
    """
    for c in PRICE_COLS:
        df = df.withColumn(c, F.when(F.isnan(F.col(c)), F.lit(None)).otherwise(F.col(c)))
    return df


def not_null_condition() -> Column:
    return _all([F.col(c).isNotNull() for c in [*PRICE_COLS, "volume"]])


def ohlc_condition() -> Column:
    """True when the bar is internally consistent (high is the max, low is the min)."""
    return (
        (F.col("high") >= F.col("low"))
        & (F.col("high") >= F.col("open"))
        & (F.col("high") >= F.col("close"))
        & (F.col("low") <= F.col("open"))
        & (F.col("low") <= F.col("close"))
    )


def validate_ohlcv(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split into (valid, invalid) on null + OHLC + volume rules. Kept for backwards compat."""
    cond = not_null_condition() & ohlc_condition() & (F.col("volume") >= 0)
    return df.filter(cond), df.filter(~cond)


def validate_nulls(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    cond = not_null_condition()
    return df.filter(cond), df.filter(~cond)


def validate_duplicates(df: DataFrame) -> DataFrame:
    """Keys that appear more than once (exact or conflicting)."""
    return df.groupBy(*KEY).count().filter(F.col("count") > 1)


def find_conflicting_duplicates(df: DataFrame) -> DataFrame:
    """Keys that have more than one *distinct* version of the bar."""
    return df.select(*BAR_COLS).distinct().groupBy(*KEY).count().filter(F.col("count") > 1)


# Backwards-compatible alias for the original (misspelt) function name.
find_confliction_duplicates = find_conflicting_duplicates


def quarantine_conflicts(df: DataFrame, conflicts_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    keys = conflicts_df.select(*KEY)
    clean = df.join(keys, on=KEY, how="left_anti")
    quarantined = df.join(keys, on=KEY, how="inner")
    return clean, quarantined


def reason_column(tick_size: float = 0.25) -> Column:
    """First failing row-level rule, or NULL when the row passes."""
    off_tick = _any(
        [
            F.abs(F.col(c) / F.lit(tick_size) - F.round(F.col(c) / F.lit(tick_size))) > F.lit(1e-6)
            for c in PRICE_COLS
        ]
    )
    non_positive = _any([F.col(c) <= 0 for c in PRICE_COLS])
    return (
        F.when(F.col("symbol").isNull() | F.col("timestamp").isNull(), "schema_error")
        .when(~not_null_condition(), "null_field")
        .when(non_positive, "non_positive_price")
        .when(~ohlc_condition(), "ohlc_violation")
        .when(F.col("volume") < 0, "negative_volume")
        .when(off_tick, "off_tick")
    )


def process_market_data(
    df: DataFrame, tick_size: float = 0.25
) -> tuple[DataFrame, DataFrame, dict[str, int]]:
    """Full Bronze -> Silver pass.

    Returns ``(silver_df, quarantine_df, stats)`` where ``quarantine_df`` carries a
    ``reason`` column with the same codes as the pandas engine.
    """
    df = nan_to_null(df.select(*BAR_COLS))
    rows_in = df.count()

    schema_bad = df.filter(F.col("symbol").isNull() | F.col("timestamp").isNull())
    keyed = df.filter(F.col("symbol").isNotNull() & F.col("timestamp").isNotNull())

    conflicts = find_conflicting_duplicates(keyed)
    remaining, conflict_q = quarantine_conflicts(keyed, conflicts)

    tagged = remaining.withColumn("reason", reason_column(tick_size))
    row_q = tagged.filter(F.col("reason").isNotNull())
    valid = tagged.filter(F.col("reason").isNull()).drop("reason")

    silver = valid.dropDuplicates(BAR_COLS)
    quarantine = (
        schema_bad.withColumn("reason", F.lit("schema_error"))
        .unionByName(conflict_q.withColumn("reason", F.lit("conflicting_duplicate")))
        .unionByName(row_q)
    )
    n_valid = valid.count()
    n_silver = silver.count()
    stats = {
        "rows_in": rows_in,
        "rows_silver": n_silver,
        "exact_duplicates_dropped": n_valid - n_silver,
        "rows_quarantined": quarantine.count(),
    }
    return silver.orderBy(*KEY), quarantine, stats


def _demo() -> None:  # pragma: no cover - manual smoke run
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[*]").appName("NQValidationDemo").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        records = [
            ("NQ", "2026-10-26 14:00:00", 25000.0, 25020.0, 24990.0, 25010.0, 1500),
            ("NQ", "2026-10-26 14:15:00", 25010.0, 25035.0, 25005.0, 25030.0, 1800),
            ("NQ", "2026-10-26 14:45:00", 25000.0, 24980.0, 24990.0, 25010.0, 1000),
            ("NQ", "2026-10-26 14:45:00", 25000.0, 24950.0, 24970.0, 25000.0, 1200),
            ("NQ", "2026-10-26 15:00:00", 25000.0, None, 24990.0, 25010.0, 1500),
        ]
        df = spark.createDataFrame([convert_timestamp(r) for r in records], schema=BAR_SCHEMA)
        silver, quarantine, stats = process_market_data(df)
        silver.show()
        quarantine.show()
        print(stats)
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover
    _demo()
