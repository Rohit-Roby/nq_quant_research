"""The pandas (local/Airflow) and Spark (Glue) engines must agree exactly."""

from datetime import date

import pytest

from quant_research.ingest.synthetic import generate_bars, inject_defects
from quant_research.transform.rules import validate_bars
from quant_research.transform.validation import BAR_COLS, process_market_data

pytestmark = pytest.mark.spark


def test_pandas_and_spark_engines_agree(spark):
    bronze = inject_defects(
        generate_bars(date(2025, 3, 3), date(2025, 3, 5), seed=3), rate=0.02, seed=9
    )
    pd_res = validate_bars(bronze)

    sdf = spark.createDataFrame(bronze[BAR_COLS].astype({"volume": "float64"}))
    sdf = sdf.withColumn("volume", sdf["volume"].cast("long"))
    silver, quarantine, stats = process_market_data(sdf)

    assert stats["rows_silver"] == pd_res.report.rows_silver
    assert stats["exact_duplicates_dropped"] == pd_res.report.exact_duplicates_dropped
    spark_reasons = {
        r["reason"]: r["count"] for r in quarantine.groupBy("reason").count().collect()
    }
    assert spark_reasons == pd_res.report.quarantined_by_reason

    spark_keys = {r["timestamp"].replace(tzinfo=None) for r in silver.select("timestamp").collect()}
    pd_keys = set(pd_res.silver["timestamp"].dt.tz_convert(None).dt.to_pydatetime())
    assert spark_keys == pd_keys
