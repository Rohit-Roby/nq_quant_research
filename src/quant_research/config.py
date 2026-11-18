"""Typed configuration loaded from ``configs/*.yaml`` and validated with pydantic."""

from __future__ import annotations

import os
from datetime import date, time
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

CONFIG_DIR = Path(
    os.environ.get("NQQR_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs")
)

# Environment variables that override pipeline.* keys (useful for Airflow/containers/tests).
ENV_OVERRIDES = {
    "lake_root": "NQQR_LAKE_ROOT",
    "warehouse_path": "NQQR_WAREHOUSE_PATH",
    "reports_dir": "NQQR_REPORTS_DIR",
    "source": "NQQR_SOURCE",
    "csv_path": "NQQR_CSV_PATH",
}


class ContractSpec(BaseModel):
    """Futures contract specification."""

    symbol: str
    description: str = ""
    exchange: str = "CME Globex"
    multiplier: float = Field(gt=0, description="USD per index point")
    tick_size: float = Field(gt=0)
    currency: str = "USD"

    @property
    def tick_value(self) -> float:
        return self.multiplier * self.tick_size


class SessionSpec(BaseModel):
    timezone: str = "America/New_York"
    globex_open: time = time(18, 0)
    globex_close: time = time(17, 0)
    rth_open: time = time(9, 30)
    rth_close: time = time(16, 0)


class CostSpec(BaseModel):
    commission_per_side: float = Field(ge=0)
    slippage_ticks_per_side: float = Field(ge=0)


class StrategySpec(BaseModel):
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)


class BacktestSpec(BaseModel):
    instrument: str = "MNQ"
    contracts: int = Field(default=1, ge=1)
    initial_capital: float = Field(default=25_000.0, gt=0)
    bar_minutes: int = Field(default=5, ge=1)
    in_sample_fraction: float = Field(default=0.7, gt=0, lt=1)


class SyntheticSpec(BaseModel):
    seed: int = 42
    start_price: float = 21_000.0
    annual_vol: float = 0.22
    defect_rate: float = Field(default=0.002, ge=0, le=0.2)


class PipelineSpec(BaseModel):
    symbol: str = "NQ"
    source: Literal["synthetic", "csv"] = "synthetic"
    csv_path: str | None = None
    start_date: date = date(2025, 1, 2)
    end_date: date = date(2025, 12, 31)
    lake_root: str = "data/lake"
    warehouse_path: str = "data/warehouse.duckdb"
    reports_dir: str = "data/reports"
    max_quarantine_rate: float = Field(default=0.02, ge=0, le=1)

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: date, info: Any) -> date:
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError("end_date must be on or after start_date")
        return v


class Settings(BaseModel):
    """All project settings in one validated object."""

    instruments: dict[str, ContractSpec]
    session: SessionSpec
    costs: dict[str, CostSpec]
    slippage_sensitivity: list[float]
    backtest: BacktestSpec
    strategies: dict[str, StrategySpec]
    pipeline: PipelineSpec
    synthetic: SyntheticSpec

    model_config = {"arbitrary_types_allowed": True}

    @cached_property
    def contract(self) -> ContractSpec:
        return self.instruments[self.backtest.instrument]

    @cached_property
    def cost(self) -> CostSpec:
        return self.costs[self.backtest.instrument]


def _read(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_settings(config_dir: str | Path | None = None, **pipeline_overrides: Any) -> Settings:
    """Load and validate every YAML in ``config_dir``.

    ``pipeline_overrides`` lets callers (CLI, Airflow) override pipeline keys such as
    ``start_date``/``end_date``/``lake_root`` without editing files.
    """
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    inst = _read(cdir / "instruments.yaml")
    costs = _read(cdir / "costs.yaml")
    strat = _read(cdir / "strategy.yaml")
    pipe = _read(cdir / "pipeline.yaml")

    env = {key: os.environ[var] for key, var in ENV_OVERRIDES.items() if os.environ.get(var)}
    pipeline_cfg = {
        **pipe.get("pipeline", {}),
        **env,
        **{k: v for k, v in pipeline_overrides.items() if v is not None},
    }
    return Settings(
        instruments={k: ContractSpec(symbol=k, **v) for k, v in inst["instruments"].items()},
        session=SessionSpec(**inst.get("session", {})),
        costs={k: CostSpec(**v) for k, v in costs["costs"].items()},
        slippage_sensitivity=costs.get("sensitivity", {}).get("slippage_ticks", [0, 1, 2]),
        backtest=BacktestSpec(**strat.get("backtest", {})),
        strategies={k: StrategySpec(**v) for k, v in strat.get("strategies", {}).items()},
        pipeline=PipelineSpec(**pipeline_cfg),
        synthetic=SyntheticSpec(**pipe.get("synthetic", {})),
    )
