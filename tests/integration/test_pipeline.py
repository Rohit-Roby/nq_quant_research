"""End-to-end: synthetic source -> lake -> DuckDB -> backtests, plus idempotency and DQ gate."""

import json
from datetime import date

import pytest

from quant_research.cli import main as cli_main
from quant_research.pipeline import DataQualityError, Pipeline
from quant_research.storage.warehouse import Warehouse


@pytest.fixture
def ran(isolated_settings):
    pipe = Pipeline(isolated_settings)
    results = pipe.run_all()
    return pipe, results


def test_run_all_writes_every_layer(ran):
    pipe, results = ran
    assert [r.stage for r in results] == ["ingest", "validate", "gold", "backtest"]
    assert len(pipe.lake.partitions("silver", "bars_1m")) == 15
    assert len(pipe.lake.partitions("gold", "bars_5m")) == 15
    tables = set(pipe.warehouse.tables())
    assert {"dq_history", "bt_metrics", "bt_daily", "bt_grid"} <= tables
    summary = json.loads((pipe.reports / "backtest_summary.json").read_text())
    assert set(summary) == {"opening_range_breakout", "vwap_reversion", "rth_buy_and_hold"}


def test_duckdb_views_query_the_lake(ran):
    pipe, _ = ran
    wh = Warehouse(pipe.s.pipeline.warehouse_path, pipe.s.pipeline.lake_root)
    df = wh.query("SELECT trade_date, count(*) n FROM bars_1m GROUP BY 1 ORDER BY 1")
    assert len(df) == 15
    assert df["n"].between(1300, 1400).all()
    assert wh.query("SELECT count(*) c FROM quarantine")["c"].iloc[0] > 0


def test_rerun_is_idempotent(ran):
    pipe, results = ran
    before = len(pipe.lake.read("silver", "bars_1m"))
    pipe.ingest(date(2025, 3, 10), date(2025, 3, 12))
    pipe.validate(date(2025, 3, 10), date(2025, 3, 12))
    assert len(pipe.lake.read("silver", "bars_1m")) == before


def test_dq_gate_fails_the_run(isolated_settings):
    s = isolated_settings.model_copy(deep=True)
    s.pipeline.max_quarantine_rate = 0.0
    pipe = Pipeline(s)
    pipe.ingest()
    with pytest.raises(DataQualityError):
        pipe.validate()


def test_cli_single_stage(isolated_settings, capsys):
    assert cli_main(["run", "ingest", "--start", "2025-03-03", "--end", "2025-03-04"]) == 0
    out = capsys.readouterr().out
    assert '"stage": "ingest"' in out
