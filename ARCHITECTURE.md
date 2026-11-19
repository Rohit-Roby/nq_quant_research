# Architecture

## Layers

| Layer | Path | Contents | Written by |
|---|---|---|---|
| Bronze | `lake/bronze/bars_1m/trade_date=…` | Raw vendor values with `source` and `ingested_at` | `Pipeline.ingest` |
| Silver | `lake/silver/bars_1m/…` | Typed, validated, deduplicated 1-minute bars with an `is_spike` flag | `Pipeline.validate` or the Glue job |
| Quarantine | `lake/silver/quarantine/…` | Rejected rows plus a `reason` code. Rows with unparseable timestamps go to `trade_date=unknown` | same |
| Gold | `lake/gold/bars_5m/…` | 5-minute bars with session columns, `ret`, `atr`, `vwap`, `vwap_dev` | `Pipeline.build_gold` |
| Warehouse | `warehouse.duckdb` | Views over the lake, plus `dq_history`, `bt_metrics`, `bt_daily`, `bt_trades`, `bt_grid`, `bt_cost_sensitivity` | pipeline |

All partitions use the **Globex trade date**: a bar belongs to the session that opens at
18:00 ET on the previous day (`ET timestamp + 6h → date`). This keeps an overnight session
in a single partition.

## Idempotency and orchestration

- Each stage reads its input from the lake and overwrites only the partitions it touches
  (`existing_data_behavior="delete_matching"` locally, `partitionOverwriteMode=dynamic` in
  Spark). Retries and backfills therefore cannot duplicate data. An integration test checks this.
- The DAG passes only small metadata through XCom. The data itself stays in the lake.
- The gold stage reads 7 extra days of silver data so that rolling features (ATR) are warmed
  up when it runs on a single day.
- **DQ gate**: `validate` raises `DataQualityError` when the quarantine rate exceeds
  `pipeline.max_quarantine_rate`. In Airflow this fails the task and skips downstream tasks,
  so research never runs on bad data.

## Validation rules (identical in both engines)

1. `schema_error`: symbol or timestamp missing or unparseable
2. `conflicting_duplicate`: the same key with different values. Every version is quarantined.
3. `null_field`
4. `non_positive_price`
5. `ohlc_violation`
6. `negative_volume`
7. `off_tick`: a price not on the 0.25 grid

The first failing rule supplies the reason. Exact duplicates are then kept once. Spikes
(|r| > 12 × the rolling median |r|) are flagged rather than removed, because a real fast
market looks the same as a bad tick.

## Backtest timing

`signal[t]` is decided at the close of bar *t*. The position is held from the open of bar *t+1*.
Bar P&L = carried position × gap + new position × intrabar move. Costs are
`|Δposition| × (commission + slippage_ticks × tick_value)`. All strategies are flat outside
RTH and from the configured exit time.

## Extension points

- New data vendor: subclass `MarketDataSource`.
- New strategy: subclass `Strategy`, implement `_positions`, and register it in `STRATEGIES`.
- Production store: keep the DuckDB view names (`bars_1m`, `bars_5m`, `quarantine`) and swap
  the engine (TimescaleDB, ClickHouse, ArcticDB).
