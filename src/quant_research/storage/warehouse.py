"""DuckDB analytical layer over the Parquet lake.

DuckDB reads the lake in place (no copy) through views, and stores small derived
tables (backtest metrics, equity curves, DQ history) so the dashboard and ad-hoc SQL
share one queryable time-series store. Swappable for TimescaleDB/ClickHouse/kdb+
in production; the view names are the contract.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

VIEWS = {
    "bars_1m": ("silver", "bars_1m"),
    "quarantine": ("silver", "quarantine"),
    "bars_5m": ("gold", "bars_5m"),
}


class Warehouse:
    def __init__(self, db_path: str | Path, lake_root: str | Path) -> None:
        self.db_path = str(db_path)
        self.lake_root = str(lake_root).rstrip("/")
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def connect(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(self.db_path, read_only=read_only)
        self._register_views(con)
        return con

    def _register_views(self, con: duckdb.DuckDBPyConnection) -> None:
        for view, (layer, dataset) in VIEWS.items():
            path = Path(self.lake_root) / layer / dataset
            if not path.exists() or not any(path.rglob("*.parquet")):
                continue
            glob = (path / "**" / "*.parquet").as_posix()
            con.execute(
                f"CREATE OR REPLACE TEMP VIEW {view} AS "
                f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true, "
                f"hive_types = {{'trade_date': VARCHAR}})"
            )

    def write_table(self, name: str, df: pd.DataFrame, replace: bool = True) -> None:
        with duckdb.connect(self.db_path) as con:
            con.register("_df", df)
            if replace:
                con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
            else:
                con.execute(f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM _df LIMIT 0")
                con.execute(f"INSERT INTO {name} SELECT * FROM _df")

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        with self.connect(read_only=False) as con:
            return con.execute(sql, params or []).df()

    def tables(self) -> list[str]:
        with duckdb.connect(self.db_path) as con:
            return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
