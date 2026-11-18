# Decision log

| # | Decision | Alternatives | Why |
|---|---|---|---|
| 1 | Medallion layout (bronze/silver/gold) on Hive-partitioned Parquet keyed by Globex trade date | Single table; calendar-date partitions | Raw data is always replayable. Trade-date keys keep an overnight session in one partition, so daily reruns touch exactly one partition |
| 2 | Quarantine instead of drop, with a reason code | Silent filtering | Rejected data stays auditable and the DQ trend becomes measurable |
| 3 | Quarantine **all** versions of a conflicting duplicate | Keep first or last | Neither version can be known to be correct. Choosing one would hide a feed problem |
| 4 | Flag spikes, do not remove them | Remove outliers | Real fast markets (CPI, FOMC) look like spikes. Removing them biases volatility and strategy results |
| 5 | Two validation engines (pandas + Spark) with a parity test | Spark only | Spark is right for Glue at scale but slow to start for daily local runs and CI. The parity test keeps the two engines in sync and caught the NaN-vs-null bug |
| 6 | Removed import-time SparkSession and demo from `validation.py` | Keep as a script | Importing the module started and stopped Spark, which broke reuse in Glue and tests |
| 7 | DuckDB as the query layer | Postgres/Timescale; kdb+ | Zero-ops and reads Parquet in place. The view names are the contract, so the engine can be swapped later |
| 8 | Signal at close, fill at the next open | Fill at signal close | Removes look-ahead by construction. A dedicated test enforces it |
| 9 | Parameters chosen in-sample only; grid size reported | Optimise on full history | Without this the backtest only measures overfitting. The demo data shows it directly (IS and OOS Sharpe flip sign) |
| 10 | MNQ as the default backtest instrument | NQ | Realistic size for a $25k account. NQ is one config line away |
| 11 | Synthetic generator with injected defects as the default source | Require vendor data | CI and reviewers can run everything offline. The defects exercise every validation rule |
| 12 | Black-76 + raw SVI with a Durrleman check | Black-Scholes on spot; SABR | Black-76 is the market standard for options on futures. SVI is the standard listed-index smile parameterisation and has a simple butterfly-arbitrage test |
