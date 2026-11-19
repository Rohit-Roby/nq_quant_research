import numpy as np
import pandas as pd
import pytest

from quant_research.backtest.engine import cost_per_contract_side, run_backtest
from quant_research.backtest.metrics import max_drawdown
from quant_research.backtest.walkforward import walk_forward
from quant_research.config import ContractSpec, CostSpec
from quant_research.strategy.library import STRATEGIES, make_strategy

MNQ = ContractSpec(symbol="MNQ", multiplier=2.0, tick_size=0.25)
FREE = CostSpec(commission_per_side=0.0, slippage_ticks_per_side=0.0)


def tiny_bars():
    return pd.DataFrame(
        {
            "symbol": "NQ",
            "timestamp": pd.date_range("2025-03-03 14:30", periods=4, freq="5min", tz="UTC"),
            "trade_date": "2025-03-03",
            "open": [100.0, 101.0, 103.0, 102.0],
            "high": [101.0, 104.0, 104.0, 103.0],
            "low": [99.0, 100.0, 101.0, 101.0],
            "close": [101.0, 103.0, 102.0, 102.5],
        }
    )


def test_hand_computed_pnl_and_fill_timing():
    bars = tiny_bars()
    # long decided at close of bar 0 -> filled at open of bar 1 (101), exit decided bar 2
    # close -> filled at open of bar 3 (102). Gross = (102 - 101) * $2 = $2.
    res = run_backtest(bars, pd.Series([1, 1, 0, 0]), MNQ, FREE)
    assert res.bars["position"].tolist() == [0, 1, 1, 0]
    assert res.metrics["total_gross_pnl"] == pytest.approx(2.0)
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert (t["entry_price"], t["exit_price"]) == (101.0, 102.0)


def test_costs_per_side():
    cost = CostSpec(commission_per_side=0.62, slippage_ticks_per_side=1.0)
    assert cost_per_contract_side(MNQ, cost) == pytest.approx(1.12)
    res = run_backtest(tiny_bars(), pd.Series([1, 1, 0, 0]), MNQ, cost, contracts=2)
    assert res.metrics["total_costs"] == pytest.approx(2 * 2 * 1.12)
    assert res.metrics["total_net_pnl"] == pytest.approx(2 * 2.0 - 4.48)


def test_short_and_flip():
    res = run_backtest(tiny_bars(), pd.Series([-1, 1, 0, 0]), MNQ, FREE)
    # short at 101 (bar1 open), flip long at 103 (bar2 open), exit at 102 (bar3 open)
    assert res.metrics["total_gross_pnl"] == pytest.approx((-(103 - 101) + (102 - 103)) * 2)
    assert len(res.trades) == 2


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategies_flat_outside_rth_and_at_close(name, gold_bars):
    strat = make_strategy(name)
    pos = strat.target_positions(gold_bars)
    assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}
    assert (pos[~gold_bars["is_rth"]] == 0).all()
    last_bar = (
        gold_bars.groupby("trade_date")["timestamp"].transform("max") == gold_bars["timestamp"]
    )
    assert (pos[last_bar] == 0).all()


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_no_lookahead(name, gold_bars):
    """Changing future prices must not change any earlier decision."""
    cut = len(gold_bars) // 2
    strat = make_strategy(name)
    base = strat.target_positions(gold_bars)
    shocked = gold_bars.copy()
    for c in ("open", "high", "low", "close", "vwap"):
        shocked.loc[cut:, c] = shocked.loc[cut:, c] * 1.10
    after = strat.target_positions(shocked)
    pd.testing.assert_series_equal(base.iloc[:cut], after.iloc[:cut])


def test_trade_pnl_reconciles_with_bar_pnl(gold_bars, settings):
    strat = make_strategy("opening_range_breakout")
    res = run_backtest(
        gold_bars, strat.target_positions(gold_bars), settings.contract, settings.cost
    )
    assert res.trades["net_pnl"].sum() == pytest.approx(res.daily["net_pnl"].sum(), abs=1e-6)


def test_unknown_params_and_strategy():
    with pytest.raises(ValueError):
        make_strategy("vwap_reversion", bad=1)
    with pytest.raises(ValueError):
        make_strategy("nope")


def test_max_drawdown():
    dd, pct = max_drawdown(pd.Series([100.0, 120.0, 90.0, 130.0]))
    assert dd == -30 and pct == pytest.approx(-0.25)


def test_walk_forward_selects_on_in_sample_only(gold_bars, settings):
    wf = walk_forward(
        gold_bars,
        "vwap_reversion",
        {"entry_z": 2.0, "lookback": 24},
        {"entry_z": [1.5, 2.5]},
        settings.contract,
        settings.cost,
        in_sample_fraction=0.6,
        slippage_grid=[0, 1],
        min_trades=1,
    )
    assert wf.grid_size == 2
    best = wf.grid.sort_values("is_sharpe", ascending=False).iloc[0]["entry_z"]
    assert wf.chosen_params["entry_z"] == best
    assert wf.in_sample["sessions"] == 6 and wf.out_of_sample["sessions"] == 4
    assert len(wf.cost_sensitivity) == 2
