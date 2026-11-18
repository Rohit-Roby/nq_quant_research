"""Performance and risk statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def extract_trades(bars: pd.DataFrame, multiplier: float, unit_cost: float) -> pd.DataFrame:
    """Round-trip trades from the per-bar ``position`` column.

    A trade is a maximal run of constant non-zero position. Its cost is the entry
    plus exit side (a flip counts as exit + new entry).
    """
    pos = bars["position"].to_numpy()
    if not np.any(pos):
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "trade_date",
                "side",
                "contracts",
                "entry_price",
                "exit_price",
                "bars",
                "gross_pnl",
                "costs",
                "net_pnl",
            ]
        )
    run_id = np.concatenate([[0], np.cumsum(pos[1:] != pos[:-1])])
    df = bars.assign(_run=run_id)
    out = []
    n = len(df)
    opens = df["open"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    stamps = df["timestamp"].to_numpy()
    dates = df["trade_date"].to_numpy()
    for _, g in df[df["position"] != 0].groupby("_run"):
        first, last = int(g.index[0]), int(g.index[-1])
        size = float(g["position"].iloc[0])
        nxt = last + 1
        exit_price = float(opens[nxt] if nxt < n else closes[last])
        exit_time = stamps[nxt] if nxt < n else stamps[last]
        entry_price = float(opens[first])
        gross = size * (exit_price - entry_price) * multiplier
        costs = 2 * abs(size) * unit_cost
        out.append(
            {
                "entry_time": stamps[first],
                "exit_time": exit_time,
                "trade_date": dates[first],
                "side": "long" if size > 0 else "short",
                "contracts": abs(size),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "bars": len(g),
                "gross_pnl": gross,
                "costs": costs,
                "net_pnl": gross - costs,
            }
        )
    return pd.DataFrame(out)


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """(max drawdown in currency, as a fraction of the running peak)."""
    if equity.empty:
        return 0.0, 0.0
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()), float((dd / peak).min())


def compute_metrics(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
    bars: pd.DataFrame | None = None,
) -> dict:
    pnl = daily["net_pnl"].astype(float)
    rets = pnl / initial_capital
    n_days = len(pnl)
    std = rets.std(ddof=1) if n_days > 1 else np.nan
    downside = rets[rets < 0].std(ddof=1) if (rets < 0).sum() > 1 else np.nan
    sharpe = float(rets.mean() / std * np.sqrt(TRADING_DAYS)) if std and std > 0 else 0.0
    sortino = (
        float(rets.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0
    )
    equity = initial_capital + pnl.cumsum()
    dd_abs, dd_pct = max_drawdown(pd.concat([pd.Series([initial_capital]), equity]))
    total = float(pnl.sum())
    ann_ret = float(rets.mean() * TRADING_DAYS)
    calmar = ann_ret / abs(dd_pct) if dd_pct < 0 else 0.0
    var95 = float(-np.percentile(pnl, 5)) if n_days else 0.0
    es95 = float(-pnl[pnl <= np.percentile(pnl, 5)].mean()) if n_days else 0.0

    n_tr = len(trades)
    wins = trades[trades["net_pnl"] > 0]["net_pnl"] if n_tr else pd.Series(dtype=float)
    losses = trades[trades["net_pnl"] <= 0]["net_pnl"] if n_tr else pd.Series(dtype=float)
    pf = (
        float(wins.sum() / -losses.sum())
        if n_tr and losses.sum() < 0
        else float("inf")
        if n_tr and wins.sum() > 0
        else 0.0
    )
    exposure = float((bars["position"] != 0).mean()) if bars is not None and len(bars) else 0.0

    return {
        "sessions": n_days,
        "total_net_pnl": round(total, 2),
        "total_gross_pnl": round(float(daily["gross_pnl"].sum()), 2),
        "total_costs": round(float(daily["costs"].sum()), 2),
        "annualised_return_pct": round(100 * ann_ret, 2),
        "annualised_vol_pct": round(100 * float(std * np.sqrt(TRADING_DAYS)), 2)
        if std == std
        else 0.0,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown": round(dd_abs, 2),
        "max_drawdown_pct": round(100 * dd_pct, 2),
        "calmar": round(calmar, 3),
        "daily_var_95": round(var95, 2),
        "daily_es_95": round(es95, 2),
        "trades": n_tr,
        "win_rate_pct": round(100 * len(wins) / n_tr, 2) if n_tr else 0.0,
        "avg_trade": round(float(trades["net_pnl"].mean()), 2) if n_tr else 0.0,
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "exposure_pct": round(100 * exposure, 2),
    }
