"""Market-data sources. Each returns *bronze* bars: raw values plus lineage columns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from quant_research import BAR_COLUMNS
from quant_research.ingest.synthetic import generate_bars, inject_defects

# Common vendor header spellings -> canonical names.
COLUMN_ALIASES = {
    "ts_event": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "time": "timestamp",
    "ticker": "symbol",
    "instrument": "symbol",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
}


class MarketDataSource(ABC):
    name: str

    @abstractmethod
    def fetch(self, start: date, end: date) -> pd.DataFrame:
        """Return raw bars for trade dates in [start, end]."""

    def to_bronze(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        out["source"] = self.name
        out["ingested_at"] = datetime.now(UTC)
        return out


class SyntheticSource(MarketDataSource):
    """Deterministic: generates from a fixed anchor date so any sub-range is reproducible."""

    name = "synthetic"

    def __init__(
        self,
        symbol: str,
        anchor: date,
        seed: int = 42,
        start_price: float = 21_000.0,
        annual_vol: float = 0.22,
        defect_rate: float = 0.002,
        tick_size: float = 0.25,
    ) -> None:
        self.symbol, self.anchor, self.seed = symbol, anchor, seed
        self.start_price, self.annual_vol = start_price, annual_vol
        self.defect_rate, self.tick_size = defect_rate, tick_size

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        clean = generate_bars(
            min(self.anchor, start),
            end,
            self.symbol,
            self.seed,
            self.start_price,
            self.annual_vol,
            self.tick_size,
        )
        dirty = inject_defects(clean, self.defect_rate, seed=self.seed + 1)
        trade_date = (
            dirty["timestamp"].dt.tz_convert("America/New_York") + pd.Timedelta(hours=6)
        ).dt.date
        mask = (trade_date >= start) & (trade_date <= end)
        return self.to_bronze(dirty[mask].reset_index(drop=True))


class CsvSource(MarketDataSource):
    """Vendor CSV/Parquet export (Databento OHLCV-1m, FirstRate, Kibot, TradingView ...).

    Naive timestamps are interpreted in ``source_tz`` (``UTC`` by default).
    """

    name = "csv"

    def __init__(self, path: str | Path, symbol: str | None = None, source_tz: str = "UTC") -> None:
        self.path, self.symbol, self.source_tz = Path(path), symbol, source_tz

    def fetch(self, start: date, end: date) -> pd.DataFrame:
        if self.path.suffix == ".parquet":
            raw = pd.read_parquet(self.path)
        else:
            raw = pd.read_csv(self.path)
        raw.columns = [
            COLUMN_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in raw.columns
        ]
        if "symbol" not in raw.columns:
            if not self.symbol:
                raise ValueError("CSV has no symbol column; pass symbol=")
            raw["symbol"] = self.symbol
        ts = pd.to_datetime(raw["timestamp"], errors="coerce")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(self.source_tz)
        raw["timestamp"] = ts.dt.tz_convert("UTC")
        trade_date = (
            raw["timestamp"].dt.tz_convert("America/New_York") + pd.Timedelta(hours=6)
        ).dt.date
        raw = raw[(trade_date >= start) & (trade_date <= end)]
        return self.to_bronze(raw[BAR_COLUMNS].reset_index(drop=True))
