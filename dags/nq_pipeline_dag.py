"""Airflow DAG: daily NQ medallion pipeline with a data-quality gate.

Runs at 17:30 New York time on weekdays, after the CME Globex daily close (17:00 ET),
and processes the trade date that just closed (the run's ``data_interval_end``).
Every task is idempotent (partition overwrite), so retries and
``airflow dags backfill`` are safe.

    ingest_bronze -> validate_silver (DQ gate) -> build_gold -> run_research
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pendulum

try:  # Airflow 3.x
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

TZ = "America/New_York"
DEFAULT_ARGS = {
    "owner": "quant-research",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}


def _pipeline() -> Any:  # imported lazily so the scheduler parses the DAG fast
    from quant_research.config import load_settings
    from quant_research.pipeline import Pipeline

    return Pipeline(load_settings())


def _trade_date(data_interval_end: Any) -> date:
    if data_interval_end is None:
        return pendulum.now(TZ).date()
    return pendulum.instance(data_interval_end).in_timezone(TZ).date()


@dag(
    dag_id="nq_medallion_pipeline",
    description="NQ futures: bronze -> silver (validated) -> gold features -> backtests",
    schedule="30 17 * * 1-5",
    start_date=pendulum.datetime(2025, 1, 2, tz=TZ),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["nq", "futures", "medallion", "quant"],
    params={"research_lookback_days": 365},
)
def nq_medallion_pipeline() -> None:
    @task
    def ingest_bronze(data_interval_end: Any = None) -> dict:
        d = _trade_date(data_interval_end)
        return _pipeline().ingest(d, d).to_dict()

    @task
    def validate_silver(upstream: dict) -> dict:
        """Raises DataQualityError (task fails, downstream skipped) on a DQ breach."""
        d = date.fromisoformat(upstream["start"])
        out = _pipeline().validate(d, d).to_dict()
        return {
            "start": upstream["start"],
            "rows": out["rows"],
            "quarantine_rate": out["quarantine_rate"],
        }

    @task
    def build_gold(upstream: dict) -> dict:
        d = date.fromisoformat(upstream["start"])
        out = _pipeline().build_gold(d, d).to_dict()
        return {"start": upstream["start"], "rows": out["rows"]}

    @task
    def run_research(upstream: dict, params: dict | None = None) -> dict:
        lookback = int((params or {}).get("research_lookback_days", 365))
        end = date.fromisoformat(upstream["start"])
        out = _pipeline().backtest(end - timedelta(days=lookback), end).to_dict()
        return {"start": upstream["start"], "rows": out["rows"]}

    run_research(build_gold(validate_silver(ingest_bronze())))


nq_medallion_pipeline()
