"""Pipeline stages: ingest -> validate (Silver) -> gold features -> research/backtest.

Each stage reads its input from the lake and writes its output back, so stages are
independently retryable and the Airflow DAG passes only small metadata between tasks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research.backtest.walkforward import WalkForwardResult, walk_forward
from quant_research.config import Settings
from quant_research.features.bars import build_gold_bars
from quant_research.ingest.sources import CsvSource, MarketDataSource, SyntheticSource
from quant_research.storage.lake import DataLake
from quant_research.storage.warehouse import Warehouse
from quant_research.transform.rules import validate_bars

log = logging.getLogger(__name__)
GOLD_WARMUP_DAYS = 7


class DataQualityError(RuntimeError):
    """Raised when the quarantine rate breaches ``pipeline.max_quarantine_rate``."""


@dataclass
class StageResult:
    stage: str
    rows: int
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "rows": self.rows, **self.details}


class Pipeline:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.lake = DataLake(settings.pipeline.lake_root)
        self.warehouse = Warehouse(settings.pipeline.warehouse_path, settings.pipeline.lake_root)
        self.reports = Path(settings.pipeline.reports_dir)
        self.reports.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _range(self, start: date | None, end: date | None) -> tuple[date, date]:
        return start or self.s.pipeline.start_date, end or self.s.pipeline.end_date

    @property
    def tick_size(self) -> float:
        sym = self.s.pipeline.symbol
        spec = self.s.instruments.get(sym) or self.s.contract
        return spec.tick_size

    def source(self) -> MarketDataSource:
        p = self.s.pipeline
        if p.source == "csv":
            if not p.csv_path:
                raise ValueError("pipeline.csv_path is required when source == 'csv'")
            return CsvSource(p.csv_path, symbol=p.symbol)
        syn = self.s.synthetic
        return SyntheticSource(
            p.symbol,
            anchor=p.start_date,
            seed=syn.seed,
            start_price=syn.start_price,
            annual_vol=syn.annual_vol,
            defect_rate=syn.defect_rate,
            tick_size=self.tick_size,
        )

    # ------------------------------------------------------------------ stages
    def ingest(self, start: date | None = None, end: date | None = None) -> StageResult:
        start, end = self._range(start, end)
        bronze = self.source().fetch(start, end)
        n = self.lake.write(bronze, "bronze", "bars_1m")
        log.info("ingest %s..%s: %d bronze rows", start, end, n)
        return StageResult(
            "ingest", n, {"start": str(start), "end": str(end), "source": self.s.pipeline.source}
        )

    def validate(
        self, start: date | None = None, end: date | None = None, enforce_gate: bool = True
    ) -> StageResult:
        start, end = self._range(start, end)
        bronze = self.lake.read("bronze", "bars_1m", start, end)
        if bronze.empty:
            raise RuntimeError(f"no bronze data for {start}..{end}; run ingest first")
        res = validate_bars(bronze, tick_size=self.tick_size)
        self.lake.write(res.silver, "silver", "bars_1m")
        if len(res.quarantine):
            # unparseable timestamps land in the trade_date=unknown partition
            self.lake.write(res.quarantine, "silver", "quarantine")
        report = res.report.to_dict()
        report.update(
            {"start": str(start), "end": str(end), "run_at": datetime.now(UTC).isoformat()}
        )
        (self.reports / f"dq_{start}_{end}.json").write_text(json.dumps(report, indent=2))
        hist = pd.DataFrame(
            [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in report.items()}]
        )
        self.warehouse.write_table("dq_history", hist, replace=False)
        log.info("validate: %s", report)
        if enforce_gate and res.report.quarantine_rate > self.s.pipeline.max_quarantine_rate:
            raise DataQualityError(
                f"quarantine rate {res.report.quarantine_rate:.4%} exceeds "
                f"{self.s.pipeline.max_quarantine_rate:.2%}"
            )
        return StageResult("validate", len(res.silver), report)

    def build_gold(self, start: date | None = None, end: date | None = None) -> StageResult:
        start, end = self._range(start, end)
        silver = self.lake.read("silver", "bars_1m", start - timedelta(days=GOLD_WARMUP_DAYS), end)
        if silver.empty:
            raise RuntimeError("no silver data; run validate first")
        gold = build_gold_bars(silver, minutes=self.s.backtest.bar_minutes)
        gold = gold[(gold["trade_date"] >= str(start)) & (gold["trade_date"] <= str(end))]
        n = self.lake.write(gold, "gold", "bars_5m")
        return StageResult("gold", n, {"sessions": int(gold["trade_date"].nunique())})

    def backtest(self, start: date | None = None, end: date | None = None) -> StageResult:
        start, end = self._range(start, end)
        bars = self.lake.read("gold", "bars_5m", start, end)
        if bars.empty:
            raise RuntimeError("no gold data; run build_gold first")
        bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        bt = self.s.backtest
        results: dict[str, WalkForwardResult] = {}
        for name, spec in self.s.strategies.items():
            if not spec.enabled:
                continue
            results[name] = walk_forward(
                bars,
                name,
                spec.params,
                spec.grid,
                self.s.contract,
                self.s.cost,
                contracts=bt.contracts,
                initial_capital=bt.initial_capital,
                in_sample_fraction=bt.in_sample_fraction,
                bar_minutes=bt.bar_minutes,
                slippage_grid=self.s.slippage_sensitivity,
            )
        summary = self._persist_backtests(results)
        return StageResult("backtest", len(bars), {"strategies": summary})

    def run_all(self, start: date | None = None, end: date | None = None) -> list[StageResult]:
        return [
            self.ingest(start, end),
            self.validate(start, end),
            self.build_gold(start, end),
            self.backtest(start, end),
        ]

    # ------------------------------------------------------------------ persistence
    def _persist_backtests(self, results: dict[str, WalkForwardResult]) -> dict[str, Any]:
        metrics_rows, daily, trades, grids, sens = [], [], [], [], []
        summary: dict[str, Any] = {}
        for name, wf in results.items():
            for period, m in (
                ("in_sample", wf.in_sample),
                ("out_of_sample", wf.out_of_sample),
                ("full", wf.full.metrics),
            ):
                metrics_rows.append({"strategy": name, "period": period, **m})
            daily.append(wf.full.daily.assign(strategy=name))
            if len(wf.full.trades):
                trades.append(wf.full.trades.assign(strategy=name))
            grids.append(
                wf.grid.astype(
                    {c: "string" for c in wf.grid.columns if wf.grid[c].dtype == object}
                ).assign(strategy=name)
            )
            if len(wf.cost_sensitivity):
                sens.append(wf.cost_sensitivity.assign(strategy=name))
            summary[name] = {
                "chosen_params": wf.chosen_params,
                "grid_size": wf.grid_size,
                "oos_start": wf.split_date,
                "in_sample": wf.in_sample,
                "out_of_sample": wf.out_of_sample,
            }

        wh = self.warehouse
        wh.write_table("bt_metrics", pd.DataFrame(metrics_rows))
        wh.write_table("bt_daily", pd.concat(daily, ignore_index=True))
        if trades:
            wh.write_table("bt_trades", pd.concat(trades, ignore_index=True))
        wh.write_table(
            "bt_grid",
            pd.concat(grids, ignore_index=True)
            .melt(id_vars=["strategy", "candidate", "is_sharpe", "is_net_pnl", "is_trades"])
            .astype({"value": "string"}),
        )
        if sens:
            wh.write_table("bt_cost_sensitivity", pd.concat(sens, ignore_index=True))
        (self.reports / "backtest_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return summary
