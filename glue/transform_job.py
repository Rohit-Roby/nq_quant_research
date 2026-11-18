"""AWS Glue (Spark) job: Bronze -> Silver for NQ 1-minute bars.

Deploy with ``src/quant_research/transform/validation.py`` packaged via
``--extra-py-files`` (see docs/DEPLOYMENT in README). Job arguments::

    --JOB_NAME       glue job name
    --BRONZE_PATH    s3://bucket/lake/bronze/bars_1m/
    --SILVER_PATH    s3://bucket/lake/silver/bars_1m/
    --QUARANTINE_PATH s3://bucket/lake/silver/quarantine/
    --TRADE_DATE     optional YYYY-MM-DD; process one partition only
    --TICK_SIZE      optional, default 0.25

Run locally (no Glue libraries) for testing::

    python glue/transform_job.py --local --BRONZE_PATH data/lake/bronze/bars_1m \
        --SILVER_PATH /tmp/silver --QUARANTINE_PATH /tmp/quarantine
"""

from __future__ import annotations

import argparse
import json
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from quant_research.transform.validation import BAR_COLS, process_market_data

REQUIRED = ["BRONZE_PATH", "SILVER_PATH", "QUARANTINE_PATH"]


def parse_args(argv: list[str]) -> tuple[dict[str, str], bool]:
    if "--local" in argv:
        p = argparse.ArgumentParser()
        p.add_argument("--local", action="store_true")
        for name in REQUIRED:
            p.add_argument(f"--{name}", required=True)
        p.add_argument("--TRADE_DATE", default="")
        p.add_argument("--TICK_SIZE", default="0.25")
        ns = p.parse_args(argv)
        return {k: v for k, v in vars(ns).items() if k != "local"}, True

    from awsglue.utils import getResolvedOptions  # type: ignore[import-not-found]

    args = getResolvedOptions(argv, ["JOB_NAME", *REQUIRED])
    for opt in ("TRADE_DATE", "TICK_SIZE"):
        if f"--{opt}" in argv:
            args.update(getResolvedOptions(argv, [opt]))
    return args, False


def read_bronze(spark: SparkSession, path: str, trade_date: str) -> DataFrame:
    df = spark.read.option("basePath", path).parquet(path)
    if trade_date:
        df = df.filter(F.col("trade_date") == trade_date)
    return df.withColumn("timestamp", F.to_timestamp("timestamp")).withColumn(
        "volume", F.col("volume").cast("long")
    )


def write_partitioned(df: DataFrame, path: str) -> None:
    (
        df.write.mode("overwrite")
        .option("partitionOverwriteMode", "dynamic")
        .partitionBy("trade_date")
        .parquet(path)
    )


def run(spark: SparkSession, args: dict[str, str]) -> dict[str, int]:
    tick = float(args.get("TICK_SIZE") or 0.25)
    bronze = read_bronze(spark, args["BRONZE_PATH"], args.get("TRADE_DATE", ""))
    silver, quarantine, stats = process_market_data(bronze.select(*BAR_COLS), tick_size=tick)
    trade_date = F.date_format(
        F.from_utc_timestamp(F.col("timestamp"), "America/New_York") + F.expr("INTERVAL 6 HOURS"),
        "yyyy-MM-dd",
    )
    write_partitioned(silver.withColumn("trade_date", trade_date), args["SILVER_PATH"])
    write_partitioned(
        quarantine.withColumn("trade_date", F.coalesce(trade_date, F.lit("unknown"))),
        args["QUARANTINE_PATH"],
    )
    return stats


def main(argv: list[str] | None = None) -> dict[str, int]:
    argv = sys.argv[1:] if argv is None else argv
    args, local = parse_args(argv)
    if local:
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("nq-silver-local")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
            .getOrCreate()
        )
        try:
            stats = run(spark, args)
        finally:
            spark.stop()
    else:  # pragma: no cover - requires the Glue runtime
        from awsglue.context import GlueContext  # type: ignore[import-not-found]
        from awsglue.job import Job  # type: ignore[import-not-found]
        from pyspark.context import SparkContext

        glue_ctx = GlueContext(SparkContext.getOrCreate())
        spark = glue_ctx.spark_session
        spark.conf.set("spark.sql.session.timeZone", "UTC")
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        job = Job(glue_ctx)
        job.init(args["JOB_NAME"], args)
        stats = run(spark, args)
        job.commit()
    print(json.dumps(stats))
    return stats


if __name__ == "__main__":
    main()
