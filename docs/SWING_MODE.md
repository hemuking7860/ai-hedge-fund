# Aggressive swing mode

Swing-trading layer for the AI hedge fund: technical timing, event risk,
risk-based sizing, and a real position lifecycle. Backward compatible — a
fund without `swing:` in its mandate behaves exactly as upstream.

## Why this exists

Upstream v2.2.0 is a **weight-targeting rebalancer**. On each rebalance date
it computes target weights from analyst convictions and diffs them against
the book. There are no entries, no exits, no stops, and no holding-period
clock anywhere in the codebase, and no technical analysis of any kind — every
alpha model either reads fundamentals through an LLM persona or reads earnings
surprises (PEAD).

For a swing desk, the exit *is* the strategy. This layer adds the missing half.

## What was added

| Module | Role |
|---|---|
| `config/swing_trading_config.py` | `conservative` / `balanced` / `aggressive_swing` profiles |
| `signals/swing_momentum.py` | Technical timing: EMA stack, RSI zone, volume, ATR regime, structure |
| `signals/catalyst.py` | Event risk with four states incl. `UNKNOWN` fallback |
| `risk/swing.py` | Conviction+ATR sizing, portfolio heat, sector caps, event gate |
| `backtesting/exits.py` | ATR stop → breakeven → trail, time stop, pre-earnings flat |
| `pipeline/swing.py` | Binds the above onto the standard cycle |
| `data/yahoo.py` | Keyless OHLCV client so any of this can be backtested |

## Profiles

| Knob | conservative | balanced | **aggressive_swing** |
|---|---|---|---|
| max position | 0.10 | 0.15 | **0.25** |
| risk per trade | 0.005 | 0.01 | **0.03** |
| max portfolio heat | 0.03 | 0.06 | **0.12** |
| max positions/sector | 2 | 3 | **3** |
| min risk:reward | 2.0 | 2.0 | **3.0** |
| min / max hold days | 3 / 20 | 2 / 15 | **2 / 10** |
| avoid binary events | true | true | **true** |
| exit before earnings | 3d | 2d | **1d** |

## Running it

```bash
poetry run aihf mandates/aggressive-swing.yaml --tickers AMZN,MSFT,IBIT,HCA --backtest --start 2026-02-17 --date 2026-08-13 --data-source yahoo --data-cache .cache/bars
```

CLI overrides: `--profile`, `--swing`, `--risk-per-trade`, `--max-hold-days`,
`--max-position-size`, `--max-portfolio-heat`, `--avoid-earnings`.

## Design decisions worth knowing

**Sizing is volatility-normalised.** `weight = risk_budget / (stop_distance /
entry)`, so every trade contributes the same dollar risk regardless of the
name's volatility. That is what makes portfolio heat additive.

> Consequence: with `risk_per_trade=0.03` and a 2-ATR stop, the formula only
> asks for less than the 25% cap once ATR exceeds ~6% of price. Typical large
> caps run at 1–3%, so for them **`max_position_size`, not `risk_per_trade`,
> is the binding constraint** and every name arrives at the cap.

**Gaps fill at the open, not the stop.** If a bar's low breaches the stop, the
fill is `min(open, stop)`. Pretending a gap-down fills at the stop manufactures
free money on exactly the days that hurt most.

**Exits run on daily bars, not the rebalance cadence.** A stop checked weekly
is not a stop, which is why swing mandates use `rebalance: daily`.

**Abstain ≠ neutral.** Both new models abstain when they lack data.
`blend_signals` excludes abstentions from the average, so "no information"
does not get averaged in as "opinion: neutral".

**A rebalance band keeps positions still.** Without it, a daily grid re-derives
every target from a wobbling ATR and the executor trades the difference — that
was 64% of all orders in the 12-month backtest. See below.

## Measured results (2025-08-14 → 2026-08-13, AMZN/MSFT/IBIT/HCA)

| Arm | Return | Bench | Sharpe | MaxDD | Trades | Win% | PF |
|---|---|---|---|---|---|---|---|
| baseline weekly | +4.06% | +22.23% | 1.22 | 1.97% | 9 | 77.8% | 1.74 |
| baseline daily | +14.13% | +21.95% | 1.68 | 4.07% | 9 | 55.6% | 2.37 |
| **aggressive_swing** | **−0.69%** | +21.95% | −0.05 | 9.08% | 94 | 33.0% | 0.65 |

**The swing mode underperformed.** It is slower, riskier and less profitable
than the baseline over this sample. The mechanics are correct and tested; the
edge is not there. Do not deploy this.

Diagnosis: expectancy is negative because a 33% win rate needs a win/loss
ratio above 2.0 and this gets 1.21 (avg win +2.85%, avg loss −2.35%). The
baseline's average win is +9.50% — it has no exits, so its winners run. In a
sample where SPY rose 22%, any long-only system that keeps going flat
structurally loses to holding.

Relaxing `max_hold_days` (tested at 20/40/90) does **not** fix it — profit
factor gets *worse*, which rules out the time stop as the cause.

## Known gaps

Zero transaction costs are modelled. At 5bps/order the swing arm's 356 orders
cost roughly 1.1% of capital, versus 44 orders for the baseline. See the
project report for the full gap list.
