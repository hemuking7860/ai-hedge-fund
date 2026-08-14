"""Swing exit engine — the position lifecycle upstream does not model.

The stock pipeline is a weight-targeting rebalancer: each rebalance date it
computes target weights and diffs them against the book. There are no
entries, no exits, no stops and no holding-period clock anywhere — a
position simply drifts until the target weights happen to change.

That is fine for a monthly fundamental fund and useless for a swing desk,
where the exit IS the strategy. This module adds the missing layer:

    initial stop        `atr_stop_multiple` ATRs below entry
    breakeven move      stop -> entry once the trade is `breakeven_at_r` in
    trailing stop       `trail_atr_multiple` ATRs under the high-water mark
    profit target       optional, off by default
    max holding period  hard time stop
    pre-earnings exit   close N days before a known print

Two modelling choices are worth stating plainly, because they are where
backtests usually lie about stops:

1. FILL PRICE. If a bar's low breaches the stop, the fill is
   `min(open, stop)` — not the stop. A name that gaps down through the
   level fills at the open, and pretending otherwise manufactures free
   money on exactly the days that hurt most in real trading.

2. EVALUATION FREQUENCY. Exits are checked on every DAILY bar, not on the
   rebalance cadence. A stop checked weekly is not a stop. Swing mandates
   therefore run a daily grid; see `rebalance: daily` in the swing mandate.

Still not modelled, and not pretended otherwise: intrabar path (we cannot
know whether the high or the low came first from a daily bar), and
intraday stop-running. See the gaps section of the project report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date

from hedge_fund.config.swing_trading_config import SwingProfile

STOP_LOSS = "STOP_LOSS"
BREAKEVEN_STOP = "BREAKEVEN_STOP"
TRAILING_STOP = "TRAILING_STOP"
PROFIT_TARGET = "PROFIT_TARGET"
MAX_HOLD = "MAX_HOLD"
PRE_EARNINGS = "PRE_EARNINGS"


@dataclass
class PositionState:
    """Everything the exit engine needs to remember about one open trade."""

    ticker: str
    entry_price: float
    entry_date: str
    initial_stop: float
    current_stop: float
    high_water: float
    bars_held: int = 0
    breakeven_armed: bool = False

    @property
    def risk_per_share(self) -> float:
        """One R, in price units — the denominator for every R multiple."""
        return max(self.entry_price - self.initial_stop, 0.0)

    def r_multiple(self, price: float) -> float:
        """How many R the trade is currently up (or down)."""
        unit = self.risk_per_share
        if unit <= 0:
            return 0.0
        return (price - self.entry_price) / unit


@dataclass
class ExitDecision:
    """A position the engine wants closed, and the honest fill price."""

    ticker: str
    reason: str
    price: float
    r_multiple: float
    bars_held: int
    detail: str = ""


@dataclass
class SwingExitEngine:
    """Tracks open swing positions and decides when each one dies.

    State lives here rather than on the broker because the broker models
    fills, not intent: the stop that sizing assumed and the stop the exit
    honours must be the same number, and that number is a strategy fact.
    """

    profile: SwingProfile
    positions: dict[str, PositionState] = field(default_factory=dict)

    # -- lifecycle ---------------------------------------------------------

    def open_position(
        self,
        ticker: str,
        entry_price: float,
        entry_date: str,
        atr_value: float,
    ) -> PositionState | None:
        """Start tracking a new position. Returns None if it cannot be sized.

        No ATR means no stop, and a swing position without a stop is not a
        position this engine is willing to own.
        """
        if entry_price <= 0 or atr_value <= 0:
            return None
        stop = entry_price - atr_value * self.profile.atr_stop_multiple
        if stop <= 0:
            return None

        state = PositionState(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date,
            initial_stop=stop,
            current_stop=stop,
            high_water=entry_price,
        )
        self.positions[ticker] = state
        return state

    def close_position(self, ticker: str) -> None:
        self.positions.pop(ticker, None)

    # -- per-bar evaluation -------------------------------------------------

    def evaluate(
        self,
        ticker: str,
        bar,
        *,
        days_until_earnings: int | None = None,
    ) -> ExitDecision | None:
        """Advance one position by one daily *bar* and decide its fate.

        The bar carries its own date, so no separate as-of is taken — one
        source of truth for "when is this".

        Checks run worst-case first: a stop that was breached matters more
        than a target that was also touched, because on a daily bar we
        cannot know which came first and assuming the favourable one is how
        backtests flatter themselves.
        """
        state = self.positions.get(ticker)
        if state is None:
            return None

        state.bars_held += 1
        state.high_water = max(state.high_water, bar.high)

        # 1. Pre-earnings exit, evaluated before price: a scheduled print is
        #    known in advance, so a desk would already be flat.
        decision = self._check_earnings(state, bar, days_until_earnings)
        if decision is not None:
            return decision

        # 2. Stop breach — the worst case, checked before any upside.
        decision = self._check_stop(state, bar)
        if decision is not None:
            return decision

        # 3. Profit target (opt-in).
        decision = self._check_target(state, bar)
        if decision is not None:
            return decision

        # 4. Time stop.
        decision = self._check_max_hold(state, bar)
        if decision is not None:
            return decision

        # 5. No exit — ratchet the stop for tomorrow.
        self._advance_stop(state, bar)
        return None

    # -- individual checks, each small and independently testable -----------

    def _check_stop(self, state: PositionState, bar) -> ExitDecision | None:
        if bar.low > state.current_stop:
            return None

        # Gap realism: a name that opens below the stop fills at the open.
        fill = min(bar.open, state.current_stop)
        if state.breakeven_armed and state.current_stop >= state.entry_price:
            reason = (
                TRAILING_STOP
                if state.current_stop > state.entry_price
                else BREAKEVEN_STOP
            )
        else:
            reason = STOP_LOSS

        gapped = bar.open < state.current_stop
        return ExitDecision(
            ticker=state.ticker,
            reason=reason,
            price=fill,
            r_multiple=round(state.r_multiple(fill), 4),
            bars_held=state.bars_held,
            detail=(
                f"gapped through {state.current_stop:.2f}, filled at open"
                if gapped else f"stop {state.current_stop:.2f} touched"
            ),
        )

    def _check_target(self, state: PositionState, bar) -> ExitDecision | None:
        if self.profile.profit_target_r <= 0:
            return None
        unit = state.risk_per_share
        if unit <= 0:
            return None
        target = state.entry_price + unit * self.profile.profit_target_r
        if bar.high < target:
            return None
        fill = max(bar.open, target)  # a gap up fills better, not worse
        return ExitDecision(
            ticker=state.ticker, reason=PROFIT_TARGET, price=fill,
            r_multiple=round(state.r_multiple(fill), 4),
            bars_held=state.bars_held,
            detail=f"target {self.profile.profit_target_r:.1f}R reached",
        )

    def _check_max_hold(self, state: PositionState, bar) -> ExitDecision | None:
        if state.bars_held < self.profile.max_hold_days:
            return None
        return ExitDecision(
            ticker=state.ticker, reason=MAX_HOLD, price=bar.close,
            r_multiple=round(state.r_multiple(bar.close), 4),
            bars_held=state.bars_held,
            detail=f"held {state.bars_held} bars, max {self.profile.max_hold_days}",
        )

    def _check_earnings(
        self,
        state: PositionState,
        bar,
        days_until_earnings: int | None,
    ) -> ExitDecision | None:
        window = self.profile.exit_before_earnings_days
        if window <= 0 or days_until_earnings is None:
            return None
        if days_until_earnings > window:
            return None
        # The min-hold floor never traps a position in front of a binary —
        # event risk outranks the holding-period policy.
        return ExitDecision(
            ticker=state.ticker, reason=PRE_EARNINGS, price=bar.close,
            r_multiple=round(state.r_multiple(bar.close), 4),
            bars_held=state.bars_held,
            detail=f"earnings in {days_until_earnings}d, flat before the print",
        )

    def _advance_stop(self, state: PositionState, bar) -> None:
        """Ratchet the stop upward. It may never move down."""
        profile = self.profile
        unit = state.risk_per_share
        if unit <= 0:
            return

        new_stop = state.current_stop

        if profile.breakeven_at_r > 0 and not state.breakeven_armed:
            if state.r_multiple(bar.close) >= profile.breakeven_at_r:
                state.breakeven_armed = True
                new_stop = max(new_stop, state.entry_price)

        if state.breakeven_armed and profile.trail_atr_multiple > 0:
            # Trail off the entry-time R unit, so the trail distance stays
            # the same risk the position was sized on.
            trail_distance = unit / profile.atr_stop_multiple * profile.trail_atr_multiple
            new_stop = max(new_stop, state.high_water - trail_distance)

        # Monotonic by construction — a stop that can loosen is not a stop.
        state.current_stop = max(state.current_stop, new_stop)


def trading_days_until(calendar: list[str], as_of: str) -> int | None:
    """Calendar days from *as_of* to the next date in *calendar*.

    Named for what callers want (time until the event); calendar days are
    the honest unit here because the reference dates are calendar dates and
    converting to trading days would need an exchange calendar we do not
    have. The difference matters only across long weekends.
    """
    today = _date.fromisoformat(as_of)
    future = sorted(d for d in calendar if _date.fromisoformat(d) > today)
    if not future:
        return None
    return (_date.fromisoformat(future[0]) - today).days
