"""Vectorised, cost-aware futures backtester.

Timing convention (no look-ahead):
    signal[t]   decided at the close of bar t (strategy output)
    held[t]     = signal[t-1]   -> position carried *during* bar t, filled at open[t]

P&L of bar t in index points per contract::

    held[t-1] * (open[t] - close[t-1])     gap carried from the previous bar
  + held[t]   * (close[t] - open[t])       intrabar move on the new position

Costs are charged on every change in position, per contract per side:
    |held[t] - held[t-1]| * (commission + slippage_ticks * tick_value)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_research.backtest.metrics import compute_metrics, extract_trades
from quant_research.config import ContractSpec, CostSpec


@dataclass
class BacktestResult:
    strategy: str
    params: dict
    bars: pd.DataFrame  # per-bar positions and P&L
    daily: pd.DataFrame  # per-session P&L and equity
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def cost_per_contract_side(
    contract: ContractSpec, cost: CostSpec, slippage_ticks: float | None = None
) -> float:
    ticks = cost.slippage_ticks_per_side if slippage_ticks is None else slippage_ticks
    return cost.commission_per_side + ticks * contract.tick_value


def run_backtest(
    bars: pd.DataFrame,
    signal: pd.Series,
    contract: ContractSpec,
    cost: CostSpec,
    contracts: int = 1,
    initial_capital: float = 25_000.0,
    slippage_ticks: float | None = None,
    strategy: str = "strategy",
    params: dict | None = None,
) -> BacktestResult:
    if len(bars) != len(signal):
        raise ValueError("bars and signal must be aligned")
    df = bars[["symbol", "timestamp", "trade_date", "open", "high", "low", "close"]].copy()
    df = df.reset_index(drop=True)
    sig = pd.Series(np.asarray(signal, dtype=float), index=df.index).fillna(0.0)

    held = sig.shift(1).fillna(0.0) * contracts
    prev_held = held.shift(1).fillna(0.0)
    prev_close = df["close"].shift(1)
    gap = (df["open"] - prev_close).fillna(0.0)
    points = prev_held * gap + held * (df["close"] - df["open"])

    unit_cost = cost_per_contract_side(contract, cost, slippage_ticks)
    traded = (held - prev_held).abs()

    df["signal"] = sig
    df["position"] = held
    df["gross_pnl"] = points * contract.multiplier
    df["costs"] = traded * unit_cost
    df["net_pnl"] = df["gross_pnl"] - df["costs"]
    df["equity"] = initial_capital + df["net_pnl"].cumsum()

    df["contracts_traded"] = traded
    daily = (
        df.groupby("trade_date")
        .agg(
            gross_pnl=("gross_pnl", "sum"),
            costs=("costs", "sum"),
            net_pnl=("net_pnl", "sum"),
            contracts_traded=("contracts_traded", "sum"),
        )
        .reset_index()
    )
    daily["equity"] = initial_capital + daily["net_pnl"].cumsum()

    trades = extract_trades(df, contract.multiplier, unit_cost)
    metrics = compute_metrics(daily, trades, initial_capital, df)
    return BacktestResult(
        strategy=strategy, params=params or {}, bars=df, daily=daily, trades=trades, metrics=metrics
    )
