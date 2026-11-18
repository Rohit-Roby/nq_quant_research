from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_research.config import load_settings
from quant_research.features.bars import build_gold_bars
from quant_research.ingest.synthetic import generate_bars


def _spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="session")
def spark():
    if not _spark_available():
        pytest.skip("pyspark not installed")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("NQQuantResearchTests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def clean_bars() -> pd.DataFrame:
    return generate_bars(date(2025, 3, 3), date(2025, 3, 14), seed=11)


@pytest.fixture(scope="session")
def gold_bars(clean_bars) -> pd.DataFrame:
    return build_gold_bars(clean_bars, minutes=5)


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("NQQR_LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.setenv("NQQR_WAREHOUSE_PATH", str(tmp_path / "wh.duckdb"))
    monkeypatch.setenv("NQQR_REPORTS_DIR", str(tmp_path / "reports"))
    return load_settings(start_date=date(2025, 3, 3), end_date=date(2025, 3, 21))
