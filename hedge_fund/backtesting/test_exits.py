"""Tests for the swing exit engine — stops, ratchets, gaps, time and events."""

from __future__ import annotations

import pytest

from hedge_fund.backtesting.exits import (
    BREAKEVEN_STOP,
    MAX_HOLD,
    PRE_EARNINGS,
    PROFIT_TARGET,
    STOP_LOSS,
    TRAILING_STOP,
    SwingExitEngine,
    trading_days_until,
)
from hedge_fund.config.swing_trading_config import AGGRESSIVE_SWING
from hedge_fund.data.models import Price


def bar(o, h, l, c, day="2026-03-02", volume=1_000_000) -> Price:
    return Price(open=o, high=h, low=l, close=c, volume=volume, time=day)


def engine(**overrides) -> SwingExitEngine:
    profile = AGGRESSIVE_SWING.model_copy(update=overrides) if overrides else AGGRESSIVE_SWING
    return SwingExitEngine(profile=profile)


def opened(eng, entry=100.0, atr=2.0):
    """Open a position at *entry* with a 2-ATR stop (= entry - 2*atr)."""
    eng.open_position("A", entry, "2026-03-01", atr)
    return eng


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------

def test_open_sets_a_two_atr_stop():
    state = engine().open_position("A", 100.0, "2026-03-01", 3.0)
    assert state is not None
    assert state.initial_stop == pytest.approx(94.0)
    assert state.risk_per_share == pytest.approx(6.0)


def test_refuses_to_open_without_atr():
    """No volatility estimate means no stop, and no stop means no trade."""
    assert engine().open_position("A", 100.0, "2026-03-01", 0.0) is None
    assert engine().open_position("A", 0.0, "2026-03-01", 2.0) is None


# ---------------------------------------------------------------------------
# Stop loss and gap realism
# ---------------------------------------------------------------------------

def test_stop_fills_at_the_stop_when_merely_touched():
    eng = opened(engine())                       # stop = 96.0
    decision = eng.evaluate("A", bar(99, 99.5, 95.5, 97))

    assert decision is not None
    assert decision.reason == STOP_LOSS
    assert decision.price == pytest.approx(96.0)


def test_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """The whole point: a gap down does NOT get the stop price."""
    eng = opened(engine())                       # stop = 96.0
    decision = eng.evaluate("A", bar(90, 91, 88, 89))

    assert decision is not None
    assert decision.price == pytest.approx(90.0), "must fill at the open"
    assert decision.r_multiple == pytest.approx(-2.5)   # lost 2.5R, not 1R
    assert "gapped" in decision.detail


def test_untouched_stop_does_not_exit():
    eng = opened(engine())
    assert eng.evaluate("A", bar(100, 102, 97, 101)) is None


# ---------------------------------------------------------------------------
# Ratchet: breakeven then trail
# ---------------------------------------------------------------------------

def test_stop_moves_to_breakeven_after_one_r():
    eng = opened(engine())                       # entry 100, stop 96, 1R = 4
    eng.evaluate("A", bar(100, 105, 99, 104.5))  # +1.125R on close

    assert eng.positions["A"].breakeven_armed is True
    assert eng.positions["A"].current_stop >= 100.0


def test_stop_trails_the_high_water_mark():
    eng = opened(engine())                       # 1R = 4; trail = 4/2*2.5 = 5
    eng.evaluate("A", bar(100, 110, 99, 109))    # arms breakeven, HWM 110

    # 110 - 5 = 105, which beats breakeven at 100.
    assert eng.positions["A"].current_stop == pytest.approx(105.0)


def test_stop_never_loosens():
    eng = opened(engine())
    eng.evaluate("A", bar(100, 112, 99, 111))    # trails up
    raised = eng.positions["A"].current_stop

    eng.evaluate("A", bar(108, 108, 106, 106.5))  # pulls back, no new high
    assert eng.positions["A"].current_stop == raised


def test_trailing_exit_is_labelled_as_trailing():
    eng = opened(engine())
    eng.evaluate("A", bar(100, 112, 99, 111))    # stop trails to 107
    decision = eng.evaluate("A", bar(108, 109, 100, 101))

    assert decision is not None
    assert decision.reason == TRAILING_STOP
    assert decision.r_multiple > 0, "a trailed stop should bank a profit"


def test_breakeven_exit_is_labelled_separately():
    eng = opened(engine(trail_atr_multiple=0.0))  # breakeven only, no trail
    eng.evaluate("A", bar(100, 105, 99, 104.5))   # arms breakeven at 100
    decision = eng.evaluate("A", bar(101, 102, 98, 99))

    assert decision is not None
    assert decision.reason == BREAKEVEN_STOP
    assert decision.price == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Time stop
# ---------------------------------------------------------------------------

def test_max_hold_closes_a_going_nowhere_trade():
    eng = opened(engine(max_hold_days=3))
    flat = bar(100, 100.5, 99.5, 100)

    assert eng.evaluate("A", flat) is None
    assert eng.evaluate("A", flat) is None
    decision = eng.evaluate("A", flat)

    assert decision is not None
    assert decision.reason == MAX_HOLD
    assert decision.bars_held == 3


def test_stop_outranks_max_hold_on_the_same_bar():
    """Worst case first: if both fire, the loss is what actually happened."""
    eng = opened(engine(max_hold_days=1))
    decision = eng.evaluate("A", bar(99, 99, 90, 91))

    assert decision is not None
    assert decision.reason == STOP_LOSS


# ---------------------------------------------------------------------------
# Profit target
# ---------------------------------------------------------------------------

def test_profit_target_is_off_by_default():
    eng = opened(engine())
    assert eng.profile.profit_target_r == 0.0
    assert eng.evaluate("A", bar(100, 130, 99, 129)) is None


def test_profit_target_fires_when_enabled():
    eng = opened(engine(profit_target_r=3.0))    # 1R = 4 -> target 112
    decision = eng.evaluate("A", bar(100, 115, 99, 114))

    assert decision is not None
    assert decision.reason == PROFIT_TARGET
    assert decision.price == pytest.approx(112.0)


# ---------------------------------------------------------------------------
# Event exit
# ---------------------------------------------------------------------------

def test_exits_before_earnings():
    eng = opened(engine())                       # exit_before_earnings_days = 1
    decision = eng.evaluate(
        "A", bar(100, 101, 99, 100.5), days_until_earnings=1
    )

    assert decision is not None
    assert decision.reason == PRE_EARNINGS


def test_earnings_further_out_does_not_exit():
    eng = opened(engine())
    assert eng.evaluate(
        "A", bar(100, 101, 99, 100.5), days_until_earnings=5
    ) is None


def test_unknown_earnings_date_does_not_exit():
    """Missing calendar must degrade to 'carry on', not to a phantom exit."""
    eng = opened(engine())
    assert eng.evaluate(
        "A", bar(100, 101, 99, 100.5), days_until_earnings=None
    ) is None


def test_earnings_exit_outranks_a_profitable_hold():
    """Event risk beats the holding-period policy, by design."""
    eng = opened(engine())
    decision = eng.evaluate(
        "A", bar(100, 106, 99, 105), days_until_earnings=0
    )
    assert decision is not None
    assert decision.reason == PRE_EARNINGS


# ---------------------------------------------------------------------------
# Calendar helper
# ---------------------------------------------------------------------------

def test_trading_days_until_picks_the_next_future_date():
    calendar = ["2026-01-28", "2026-04-29", "2026-07-29"]
    assert trading_days_until(calendar, "2026-04-27") == 2
    assert trading_days_until(calendar, "2026-07-29") == 0 or \
        trading_days_until(calendar, "2026-07-29") is None


def test_trading_days_until_returns_none_past_the_calendar():
    assert trading_days_until(["2026-01-01"], "2026-06-01") is None


def test_untracked_ticker_evaluates_to_none():
    assert engine().evaluate("GHOST", bar(1, 2, 0.5, 1)) is None
