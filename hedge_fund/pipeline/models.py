"""Pipeline records — the serialized truth of every cycle.

A CycleRecord captures one tick of the fund end to end: what the analysts
saw, what they said, how views became weights, what risk clamped, what was
ordered and filled, and what the book looks like after. The ledger persists
these; `fund why AAPL` will answer from them alone.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hedge_fund.brokers.models import Fill, Order
from hedge_fund.fund.spec import FundSpec
from hedge_fund.models import Signal
from hedge_fund.risk.limits import ClampEvent
from hedge_fund.risk.swing import SwingClamp


class TickerSkip(BaseModel):
    """A requested name that could not be traded this cycle, and why."""

    ticker: str
    reason: str


class StrategyRecord(BaseModel):
    """One strategy's slice of a cycle: its analysts' views and its sleeve."""

    name: str
    slice: float                        # normalized capital slice of the fund
    signals: list[Signal]               # this strategy's analysts x tradeable tickers
    convictions: dict[str, float]       # blended views, pre-scaling
    weights: dict[str, float]           # the sleeve, before netting across strategies


class SwingExitRecord(BaseModel):
    """One position the swing exit engine closed, and why."""

    ticker: str
    reason: str                         # STOP_LOSS, TRAILING_STOP, MAX_HOLD, ...
    price: float                        # honest fill, gap-aware
    r_multiple: float                   # P&L in units of initial risk
    bars_held: int
    detail: str = ""


class CycleRecord(BaseModel):
    """One tick of the fund, fully serialized — every stage's inputs and
    outputs. `model_dump_json()` round-trips; nothing about a decision
    lives anywhere else."""

    fund: str
    as_of: str
    spec: FundSpec                      # self-contained audit copy
    universe: list[str]                 # the tickers this cycle was asked to trade
    marks: dict[str, float]             # ticker -> close used for sizing and NAV
    skipped: list[TickerSkip]
    strategies: list[StrategyRecord]    # every sleeve, incl. each thesis
    target_weights: dict[str, float]    # the NETTED book, pre-risk
    clamps: list[ClampEvent]
    final_weights: dict[str, float]     # post-risk
    equity_before: float
    cash_before: float
    orders: list[Order]
    fills: list[Fill]
    positions: dict[str, int]           # signed shares after fills
    cash: float
    nav: float                          # cash + sum(shares * mark)

    # -- swing mode; all empty/None on a stock (non-swing) cycle ----------
    swing_clamps: list[SwingClamp] = Field(default_factory=list)
    swing_exits: list[SwingExitRecord] = Field(default_factory=list)
    portfolio_heat: float | None = None  # summed open risk after swing risk
    stops: dict[str, float] = Field(default_factory=dict)  # ticker -> live stop
