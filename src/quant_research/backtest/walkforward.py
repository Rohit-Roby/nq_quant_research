"""In-sample parameter search, out-of-sample evaluation and cost sensitivity.

Parameters are chosen **only** on in-sample sessions; the out-of-sample block is
touched once with the chosen parameters. The number of grid points tried is reported
so readers can discount the IS Sharpe for multiple testing.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quant_research.backtest.engine import BacktestResult, run_backtest
from quant_research.backtest.metrics import compute_metrics
from quant_research.config import ContractSpec, CostSpec
from quant_research.strategy.library import make_strategy


@dataclass
class WalkForwardResult:
    strategy: str
    chosen_params: dict[str, Any]
    grid_size: int
    split_date: str
    in_sample: dict
    out_of_sample: dict
    full: BacktestResult
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)


def split_sessions(bars: pd.DataFrame, in_sample_fraction: float) -> tuple[list[str], list[str]]:
    sessions = sorted(bars["trade_date"].unique())
    cut = max(1, int(len(sessions) * in_sample_fraction))
    return sessions[:cut], sessions[cut:]


def _subset_metrics(res: BacktestResult, sessions: list[str], capital: float) -> dict:
    daily = res.daily[res.daily["trade_date"].isin(sessions)]
    trades = res.trades[res.trades["trade_date"].isin(sessions)] if len(res.trades) else res.trades
    bars = res.bars[res.bars["trade_date"].isin(sessions)]
    return compute_metrics(daily, trades, capital, bars)


def _grid(base: dict[str, Any], grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [dict(base)]
    keys = list(grid)
    return [
        {**base, **dict(zip(keys, combo, strict=True))}
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def walk_forward(
    bars: pd.DataFrame,
    strategy: str,
    base_params: dict[str, Any],
    grid: dict[str, list[Any]],
    contract: ContractSpec,
    cost: CostSpec,
    contracts: int = 1,
    initial_capital: float = 25_000.0,
    in_sample_fraction: float = 0.7,
    bar_minutes: int = 5,
    slippage_grid: list[float] | None = None,
    min_trades: int = 10,
) -> WalkForwardResult:
    is_days, oos_days = split_sessions(bars, in_sample_fraction)
    rows, results = [], {}
    for i, params in enumerate(_grid(base_params, grid)):
        strat = make_strategy(strategy, bar_minutes=bar_minutes, **params)
        res = run_backtest(
            bars,
            strat.target_positions(bars),
            contract,
            cost,
            contracts,
            initial_capital,
            strategy=strategy,
            params=params,
        )
        m = _subset_metrics(res, is_days, initial_capital)
        rows.append(
            {
                "candidate": i,
                **params,
                "is_sharpe": m["sharpe"],
                "is_net_pnl": m["total_net_pnl"],
                "is_trades": m["trades"],
            }
        )
        results[i] = res
    grid_df = pd.DataFrame(rows)
    eligible = grid_df[grid_df["is_trades"] >= min_trades]
    pool = eligible if len(eligible) else grid_df
    best = int(pool.sort_values("is_sharpe", ascending=False).iloc[0]["candidate"])
    chosen = results[best]

    sens = []
    for ticks in slippage_grid or []:
        strat = make_strategy(strategy, bar_minutes=bar_minutes, **chosen.params)
        r = run_backtest(
            bars,
            strat.target_positions(bars),
            contract,
            cost,
            contracts,
            initial_capital,
            slippage_ticks=ticks,
            strategy=strategy,
        )
        oos = _subset_metrics(r, oos_days, initial_capital)
        sens.append(
            {
                "slippage_ticks": ticks,
                "oos_sharpe": oos["sharpe"],
                "oos_net_pnl": oos["total_net_pnl"],
                "full_net_pnl": r.metrics["total_net_pnl"],
            }
        )

    return WalkForwardResult(
        strategy=strategy,
        chosen_params=chosen.params,
        grid_size=len(grid_df),
        split_date=oos_days[0] if oos_days else "",
        in_sample=_subset_metrics(chosen, is_days, initial_capital),
        out_of_sample=_subset_metrics(chosen, oos_days, initial_capital) if oos_days else {},
        full=chosen,
        grid=grid_df,
        cost_sensitivity=pd.DataFrame(sens),
    )
