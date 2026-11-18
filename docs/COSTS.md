# Cost model

Costs are charged **per contract, per side** on every change in position:

```
cost = |Δposition| × (commission_per_side + slippage_ticks_per_side × tick_value)
```

| | NQ | MNQ |
|---|---|---|
| Multiplier | $20 / pt | $2 / pt |
| Tick | 0.25 pt = $5.00 | 0.25 pt = $0.50 |
| Commission per side (assumed, all-in) | $2.25 | $0.62 |
| Slippage per side (assumed) | 1 tick = $5.00 | 1 tick = $0.50 |
| **Round trip per contract** | **$14.50** | **$2.24** |

The commission figures are planning assumptions that cover broker, exchange, clearing and
NFA fees. Replace them with your broker's schedule in `configs/costs.yaml`.

Why slippage matters more than commission: for MNQ, one extra tick per side adds $1.00 per
round trip, almost half the total cost. Every backtest therefore re-runs the chosen
parameters at 0, 0.5, 1, 2 and 3 ticks (`bt_cost_sensitivity`). A strategy whose OOS P&L
turns negative at 2 ticks is not robust enough to trade.
