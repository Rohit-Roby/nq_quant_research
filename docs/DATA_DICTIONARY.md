# Data dictionary

## silver.bars_1m

| Column | Type | Description |
|---|---|---|
| symbol | string | Root symbol (`NQ`) |
| timestamp | timestamp (UTC) | Bar **open** time |
| open, high, low, close | float64 | Index points, on the 0.25 tick grid |
| volume | int64 | Contracts traded in the bar |
| is_spike | bool | \|log return\| > 12 × rolling 120-bar median \|log return\|. Flag only |
| trade_date | string `YYYY-MM-DD` | Globex session date (partition key) |

## silver.quarantine

The silver columns (raw values, possibly null) plus `reason`, one of: `schema_error`,
`conflicting_duplicate`, `null_field`, `non_positive_price`, `ohlc_violation`,
`negative_volume`, `off_tick`.

## gold.bars_5m

| Column | Description |
|---|---|
| symbol, timestamp, open, high, low, close, volume | 5-minute OHLCV, left-labelled |
| trade_date | Globex session date |
| minute_of_day | Minutes since midnight ET of the bar open |
| is_rth | Bar opens between 09:30 and 15:55 ET |
| ret | Log return close-to-close |
| atr | 14-bar average true range (points) |
| vwap | Session VWAP anchored at 09:30 ET. Null outside RTH |
| vwap_dev | close − vwap |

## Warehouse tables

| Table | Grain |
|---|---|
| dq_history | One row per validate run (counts, reasons as JSON, rate, run_at) |
| bt_metrics | strategy × period (in_sample / out_of_sample / full) |
| bt_daily | strategy × trade_date: gross, costs, net, contracts traded, equity |
| bt_trades | One row per round-trip trade |
| bt_grid | Every grid candidate's in-sample score (long format) |
| bt_cost_sensitivity | strategy × slippage ticks: OOS and full-period P&L |
