"""Streamlit research dashboard: data quality, market view, strategy results, vol surface.

Run with ``nqqr dashboard`` or ``streamlit run dashboard/app.py`` after ``nqqr run all``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_research.config import load_settings
from quant_research.features.bars import opening_range
from quant_research.pricing import black76
from quant_research.pricing.svi import SVIParams, fit_svi, synthetic_chain
from quant_research.storage.warehouse import Warehouse

# Validated categorical order (blue, orange, aqua) — colour follows the strategy, not rank.
SERIES = {
    "opening_range_breakout": "#2a78d6",
    "vwap_reversion": "#eb6834",
    "rth_buy_and_hold": "#1baf7a",
}
NEUTRAL = "#52514e"
LAYOUT = dict(
    margin=dict(l=10, r=10, t=70, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
    height=400,
)

st.set_page_config(page_title="NQ Quant Research", layout="wide")
settings = load_settings()
wh = Warehouse(settings.pipeline.warehouse_path, settings.pipeline.lake_root)


@st.cache_data(ttl=60)
def q(sql: str) -> pd.DataFrame:
    try:
        return wh.query(sql)
    except Exception as exc:  # noqa: BLE001 - surface any missing-table error in the UI
        st.session_state["_last_error"] = str(exc)
        return pd.DataFrame()


st.title("NQ / MNQ Quant Research")
data_note = (
    ("synthetic NQ-like data. Results show what the pipeline and engine do, not evidence of edge")
    if settings.pipeline.source == "synthetic"
    else f"vendor CSV ({settings.pipeline.csv_path})"
)
st.caption(
    f"Source: {data_note}. Instrument for backtests: {settings.backtest.instrument} "
    f"× {settings.backtest.contracts}, costs per side: "
    f"${settings.cost.commission_per_side} + {settings.cost.slippage_ticks_per_side} tick."
)

metrics = q("SELECT * FROM bt_metrics")
if metrics.empty:
    st.warning("No results yet. Run `nqqr run all` first.")
    st.stop()

tab_overview, tab_dq, tab_mkt, tab_strat, tab_vol = st.tabs(
    ["Overview", "Data quality", "Market", "Strategies", "Vol & Greeks"]
)

# ------------------------------------------------------------------ overview
with tab_overview:
    oos = metrics[metrics["period"] == "out_of_sample"].set_index("strategy")
    cols = st.columns(len(oos))
    for col, (name, row) in zip(cols, oos.iterrows(), strict=False):
        col.metric(
            name.replace("_", " ").title(),
            f"${row['total_net_pnl']:,.0f}",
            f"OOS Sharpe {row['sharpe']:.2f}",
            delta_color="off",
        )
    st.subheader("In-sample vs out-of-sample")
    show = [
        "strategy",
        "period",
        "sharpe",
        "total_net_pnl",
        "max_drawdown_pct",
        "trades",
        "win_rate_pct",
        "profit_factor",
        "total_costs",
        "daily_var_95",
    ]
    st.dataframe(metrics[metrics["period"] != "full"][show], hide_index=True, width="stretch")
    summary_path = Path(settings.pipeline.reports_dir) / "backtest_summary.json"
    if summary_path.exists():
        with st.expander("Chosen parameters (selected on in-sample only)"):
            st.json(
                {
                    k: {
                        "params": v["chosen_params"],
                        "grid_size": v["grid_size"],
                        "oos_start": v["oos_start"],
                    }
                    for k, v in json.loads(summary_path.read_text()).items()
                }
            )

# ------------------------------------------------------------------ data quality
with tab_dq:
    dq = q("SELECT * FROM dq_history ORDER BY run_at DESC")
    if dq.empty:
        st.info("No DQ runs recorded.")
    else:
        last = dq.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows in", f"{int(last['rows_in']):,}")
        c2.metric("Silver rows", f"{int(last['rows_silver']):,}")
        c3.metric("Quarantine rate", f"{100 * float(last['quarantine_rate']):.3f}%")
        c4.metric("Duplicates dropped", f"{int(last['exact_duplicates_dropped']):,}")
        reasons = pd.Series(json.loads(last["quarantined_by_reason"])).sort_values()
        fig = go.Figure(
            go.Bar(
                x=reasons.values,
                y=reasons.index,
                orientation="h",
                marker_color=SERIES["opening_range_breakout"],
                hovertemplate="%{y}: %{x:,} rows<extra></extra>",
            )
        )
        fig.update_layout(
            title="Quarantined rows by rule", **{**LAYOUT, "height": 300, "hovermode": "closest"}
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Also flagged (kept): {int(last['spikes_flagged'])} spike bars; "
            f"{int(last['intrasession_gaps'])} missing 1-minute bars inside sessions."
        )
        st.subheader("Quarantine sample")
        st.dataframe(
            q("SELECT * FROM quarantine ORDER BY timestamp LIMIT 200"),
            hide_index=True,
            width="stretch",
        )
        st.subheader("Run history")
        st.dataframe(dq, hide_index=True, width="stretch")

# ------------------------------------------------------------------ market
with tab_mkt:
    dates = q("SELECT DISTINCT trade_date FROM bars_5m ORDER BY trade_date")
    if dates.empty:
        st.info("No gold bars.")
    else:
        d = st.select_slider(
            "Session",
            options=dates["trade_date"].astype(str).tolist(),
            value=str(dates["trade_date"].iloc[-1]),
        )
        bars = q(f"SELECT * FROM bars_5m WHERE trade_date = '{d}' AND is_rth ORDER BY timestamp")
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(
            "America/New_York"
        )
        orng = opening_range(bars, 30)
        fig = go.Figure(
            go.Candlestick(
                x=bars["timestamp"],
                open=bars["open"],
                high=bars["high"],
                low=bars["low"],
                close=bars["close"],
                name="NQ 5m",
                increasing_line_color="#1baf7a",
                decreasing_line_color="#e34948",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bars["timestamp"],
                y=bars["vwap"],
                name="Session VWAP",
                line=dict(color=SERIES["vwap_reversion"], width=2),
            )
        )
        if len(orng):
            for lvl, label in (
                (orng["or_high"].iloc[0], "OR high (30m)"),
                (orng["or_low"].iloc[0], "OR low (30m)"),
            ):
                fig.add_hline(
                    y=lvl,
                    line=dict(color=NEUTRAL, width=1, dash="dot"),
                    annotation_text=label,
                    annotation_position="top left",
                )
        fig.update_layout(
            title=f"RTH session {d} (ET)",
            xaxis_rangeslider_visible=False,
            **{**LAYOUT, "height": 480},
        )
        st.plotly_chart(fig, width="stretch")
        daily = q(
            "SELECT trade_date, sum(volume) AS volume, "
            "stddev_samp(ret) * sqrt(78 * 252) AS rv FROM bars_5m WHERE is_rth "
            "GROUP BY 1 ORDER BY 1"
        )
        fig = go.Figure(
            go.Scatter(
                x=daily["trade_date"],
                y=daily["rv"] * 100,
                line=dict(color=SERIES["opening_range_breakout"], width=2),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(title="RTH realised volatility (annualised, %)", **LAYOUT)
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ strategies
with tab_strat:
    daily = q("SELECT * FROM bt_daily ORDER BY trade_date")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    oos_start = next(iter(summary.values()), {}).get("oos_start")
    fig = go.Figure()
    for name, g in daily.groupby("strategy"):
        fig.add_trace(
            go.Scatter(
                x=g["trade_date"],
                y=g["equity"],
                name=name,
                line=dict(color=SERIES.get(name, NEUTRAL), width=2),
                hovertemplate="$%{y:,.0f}",
            )
        )
    if oos_start:
        fig.add_vline(x=oos_start, line=dict(color=NEUTRAL, dash="dash", width=1))
        fig.add_annotation(
            x=oos_start,
            y=1,
            yref="paper",
            text="out-of-sample →",
            showarrow=False,
            xanchor="left",
            font=dict(color=NEUTRAL),
        )
    fig.update_layout(title="Equity curve, net of costs (USD)", **LAYOUT)
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for name, g in daily.groupby("strategy"):
        dd = g["equity"] - g["equity"].cummax()
        fig.add_trace(
            go.Scatter(
                x=g["trade_date"],
                y=dd,
                name=name,
                line=dict(color=SERIES.get(name, NEUTRAL), width=2),
                hovertemplate="$%{y:,.0f}",
            )
        )
    fig.update_layout(title="Drawdown (USD)", **LAYOUT)
    st.plotly_chart(fig, width="stretch")

    sens = q("SELECT * FROM bt_cost_sensitivity")
    if not sens.empty:
        fig = go.Figure()
        for name, g in sens.groupby("strategy"):
            fig.add_trace(
                go.Scatter(
                    x=g["slippage_ticks"],
                    y=g["oos_net_pnl"],
                    name=name,
                    mode="lines+markers",
                    marker=dict(size=8),
                    line=dict(color=SERIES.get(name, NEUTRAL), width=2),
                )
            )
        fig.update_layout(
            title="Cost sensitivity: OOS net P&L vs slippage ticks per side",
            xaxis_title="slippage (ticks per side)",
            **LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")

    trades = q("SELECT * FROM bt_trades ORDER BY entry_time DESC LIMIT 500")
    st.subheader("Recent trades")
    st.dataframe(trades, hide_index=True, width="stretch")

# ------------------------------------------------------------------ vol
with tab_vol:
    st.markdown(
        "Black-76 pricing for options on NQ futures and a raw-SVI smile fit. "
        "The chain is generated from a known smile plus quote noise, so you can see "
        "the fit recover it."
    )
    c1, c2, c3, c4 = st.columns(4)
    F = c1.number_input("Futures price F", value=21000.0, step=25.0)
    days = c2.number_input("Days to expiry", value=30, min_value=1, max_value=365)
    r = c3.number_input("Rate r", value=0.045, step=0.005, format="%.3f")
    noise = c4.slider("Quote noise (vol pts)", 0.0, 1.0, 0.2) / 100
    T = days / 365
    true = SVIParams(a=0.002, b=0.04, rho=-0.7, m=0.02, sigma=0.08)
    chain = synthetic_chain(F, T, r, true, noise_vol=noise)
    iv_mkt = np.array(
        [
            black76.implied_vol(p, F, k, T, r, bool(c))
            for p, k, c in zip(chain["price"], chain["strike"], chain["is_call"], strict=True)
        ]
    )
    fit = fit_svi(chain["log_moneyness"], iv_mkt, T)
    grid = np.linspace(chain["log_moneyness"].min(), chain["log_moneyness"].max(), 200)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=F * np.exp(chain["log_moneyness"]),
            y=iv_mkt * 100,
            mode="markers",
            name="Market IV (from prices)",
            marker=dict(size=8, color=SERIES["opening_range_breakout"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=F * np.exp(grid),
            y=fit.params.implied_vol(grid, T) * 100,
            name="SVI fit",
            line=dict(color=SERIES["vwap_reversion"], width=2),
        )
    )
    fig.update_layout(
        title=f"Smile, {days}d expiry — RMSE {fit.rmse_vol * 100:.3f} vol pts, "
        f"butterfly-arb-free: {fit.butterfly_arbitrage_free}",
        xaxis_title="strike",
        yaxis_title="implied vol (%)",
        **{**LAYOUT, "hovermode": "closest"},
    )
    st.plotly_chart(fig, width="stretch")
    st.json({k: round(v, 5) for k, v in fit.params.as_dict().items()})

    st.subheader("Greeks at the fitted vol")
    K = st.number_input("Strike", value=float(round(F / 25) * 25), step=25.0)
    vol = float(fit.params.implied_vol(np.log(K / F), T))
    rows = []
    for is_call in (True, False):
        g = black76.greeks(F, K, T, r, vol, is_call)
        px = float(black76.price(F, K, T, r, vol, is_call))
        rows.append(
            {
                "type": "call" if is_call else "put",
                "iv_%": round(vol * 100, 2),
                "price_pts": round(px, 2),
                "price_usd_NQ": round(px * settings.instruments["NQ"].multiplier, 2),
                "delta": round(float(g.delta), 4),
                "gamma": round(float(g.gamma), 6),
                "vega_per_volpt": round(float(g.vega) / 100, 3),
                "theta_per_day": round(float(g.theta) / 365, 3),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
