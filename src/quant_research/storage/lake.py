"""Medallion data lake on Parquet, Hive-partitioned by ``trade_date``.

Layout::

    {root}/bronze/bars_1m/trade_date=2025-01-02/*.parquet   raw + lineage
    {root}/silver/bars_1m/...                               validated, deduplicated
    {root}/silver/quarantine/...                            rejected rows + reason
    {root}/gold/bars_5m/...                                 session features for research

Writes are idempotent: re-running a date range replaces exactly those partitions
(``existing_data_behavior="delete_matching"``), which is what makes Airflow retries
and backfills safe. ``root`` can be a local path or ``s3://bucket/prefix``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

LAYERS = ("bronze", "silver", "gold")


def add_trade_date(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Globex trade date: bars from 18:00 ET belong to the next calendar day's session."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["trade_date"] = (ts.dt.tz_convert(tz) + pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%d")
    out["trade_date"] = out["trade_date"].fillna("unknown")
    return out


class DataLake:
    def __init__(self, root: str | Path) -> None:
        self.root = str(root).rstrip("/")

    def path(self, layer: str, dataset: str) -> str:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}")
        return f"{self.root}/{layer}/{dataset}"

    def write(self, df: pd.DataFrame, layer: str, dataset: str) -> int:
        """Write ``df`` partitioned by trade_date, replacing the partitions it touches."""
        if df.empty:
            return 0
        if "trade_date" not in df.columns:
            df = add_trade_date(df)
        target = self.path(layer, dataset)
        if not target.startswith("s3://"):
            Path(target).mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=target,
            partition_cols=["trade_date"],
            existing_data_behavior="delete_matching",
        )
        return len(df)

    def read(
        self, layer: str, dataset: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        target = self.path(layer, dataset)
        if not target.startswith("s3://") and not Path(target).exists():
            return pd.DataFrame()
        dset = ds.dataset(target, format="parquet", partitioning="hive")
        filt = None
        if start is not None:
            filt = ds.field("trade_date") >= start.isoformat()
        if end is not None:
            e = ds.field("trade_date") <= end.isoformat()
            filt = e if filt is None else filt & e
        df = dset.to_table(filter=filt).to_pandas()
        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].astype(str)
        if "timestamp" in df.columns:
            df = df.sort_values(["timestamp"]).reset_index(drop=True)
        return df

    def partitions(self, layer: str, dataset: str) -> list[str]:
        p = Path(self.path(layer, dataset))
        if not p.exists():
            return []
        return sorted(
            d.name.split("=", 1)[1] for d in p.iterdir() if d.name.startswith("trade_date=")
        )
