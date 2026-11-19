# Requirements

## Functional

| ID | Requirement | Where | Verified by |
|---|---|---|---|
| F1 | Ingest NQ/MNQ 1-minute OHLCV from a vendor file or a deterministic synthetic source | `ingest/` | `test_synthetic_and_sources.py` |
| F2 | Validate bars with explicit, ordered rules and quarantine failures with a reason | `transform/rules.py` | `test_rules.py` |
| F3 | Run the same rules in Spark for AWS Glue, with proven parity | `transform/validation.py`, `glue/` | `test_engine_parity.py`, `test_glue_job_matches_pandas_engine` |
| F4 | Store bronze, silver and gold as trade-date-partitioned Parquet with idempotent writes | `storage/lake.py` | `test_rerun_is_idempotent` |
| F5 | Expose the lake through SQL | `storage/warehouse.py` | `test_duckdb_views_query_the_lake` |
| F6 | Build session-aware research features (RTH, VWAP, ATR, opening range) | `features/bars.py` | `test_features.py` |
| F7 | Backtest intraday strategies on NQ/MNQ with realistic costs and no look-ahead | `backtest/engine.py` | `test_backtest.py` |
| F8 | Select parameters in-sample only, report OOS results and cost sensitivity | `backtest/walkforward.py` | `test_walk_forward_selects_on_in_sample_only` |
| F9 | Price options on futures (Black-76), compute Greeks and IV, and fit an arbitrage-checked SVI smile | `pricing/` | `test_pricing.py` |
| F10 | Orchestrate daily with retries and a DQ gate | `dags/` | `test_airflow_dag_structure`, `airflow dags test` |
| F11 | Dashboard for DQ, market data, strategy P&L/risk and the vol smile | `dashboard/app.py` | `test_dashboard_renders` |

## Non-functional

- Python 3.12+, typed (`mypy`), linted (`ruff`), CI on 3.12 and 3.13.
- One year of 1-minute bars runs end to end in under 60 s on a laptop.
- Configuration lives in YAML and is validated by pydantic. Environment variables can
  override storage paths.
- All randomness is seeded, so results are reproducible.
