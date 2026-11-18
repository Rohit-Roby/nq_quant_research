"""Glue job (local Spark), Airflow DAG structure, and the Streamlit dashboard."""

import importlib.util
from datetime import date
from pathlib import Path

import pytest

from quant_research.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.spark
def test_glue_job_matches_pandas_engine(isolated_settings, spark, tmp_path):
    pipe = Pipeline(isolated_settings)
    pipe.ingest(date(2025, 3, 3), date(2025, 3, 5))
    expected = pipe.validate(date(2025, 3, 3), date(2025, 3, 5)).details

    spec = importlib.util.spec_from_file_location("glue_job", ROOT / "glue" / "transform_job.py")
    glue_job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(glue_job)
    stats = glue_job.run(
        spark,
        {
            "BRONZE_PATH": pipe.lake.path("bronze", "bars_1m"),
            "SILVER_PATH": str(tmp_path / "silver"),
            "QUARANTINE_PATH": str(tmp_path / "quarantine"),
            "TRADE_DATE": "",
            "TICK_SIZE": "0.25",
        },
    )
    assert stats["rows_silver"] == expected["rows_silver"]
    assert stats["rows_quarantined"] == expected["rows_quarantined"]
    assert len(list((tmp_path / "silver").glob("trade_date=*"))) == 3


def test_airflow_dag_structure():
    pytest.importorskip("airflow")
    try:
        from airflow.dag_processing.dagbag import DagBag  # Airflow 3.x
    except ImportError:
        from airflow.models import DagBag  # Airflow 2.x

    bag = DagBag(dag_folder=str(ROOT / "dags"))
    assert not bag.import_errors, bag.import_errors
    dag = bag.get_dag("nq_medallion_pipeline")
    assert [t.task_id for t in dag.topological_sort()] == [
        "ingest_bronze",
        "validate_silver",
        "build_gold",
        "run_research",
    ]
    assert dag.default_args["retries"] == 2


def test_dashboard_renders(isolated_settings):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    Pipeline(isolated_settings).run_all()
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5
    assert any(m.label == "Silver rows" for m in at.metric)
