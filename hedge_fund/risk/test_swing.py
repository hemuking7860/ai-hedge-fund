"""Tests for the swing risk layer — sizing, heat, sector caps, event gate."""

from __future__ import annotations

import pytest

from hedge_fund.config.swing_trading_config import (
    AGGRESSIVE_SWING,
    PROFILES,
    SwingProfile,
    get_profile,
)
from hedge_fund.risk.swing import (
    apply_event_gate,
    apply_portfolio_heat,
    apply_sector_caps,
    apply_swing_risk,
    size_from_risk,
    stop_distance,
    trade_risk,
)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def test_aggressive_swing_matches_the_specified_defaults():
    p = AGGRESSIVE_SWING
    assert p.max_position_size == 0.25
    assert p.risk_per_trade == 0.03
    assert p.max_portfolio_heat == 0.12
    assert p.max_positions_per_sector == 3
    assert p.min_risk_reward == 3.0
    assert p.min_hold_days == 2
    assert p.max_hold_days == 10
    assert p.avoid_binary_events is True
    assert p.exit_before_earnings_days == 1


def test_all_three_profiles_registered():
    assert set(PROFILES) == {"conservative", "balanced", "aggressive_swing"}


def test_unknown_profile_fails_loud():
    with pytest.raises(ValueError, match="unknown swing profile"):
        get_profile("yolo")


def test_incoherent_hold_window_rejected():
    with pytest.raises(ValueError, match="min_hold_days"):
        SwingProfile(
            name="bad", max_position_size=0.1, risk_per_trade=0.01,
            max_portfolio_heat=0.05, max_positions_per_sector=2,
            min_risk_reward=2.0, min_hold_days=10, max_hold_days=5,
        )


def test_risk_exceeding_heat_rejected():
    """A profile where one trade blows the whole heat budget is unusable."""
    with pytest.raises(ValueError, match="max_portfolio_heat"):
        SwingProfile(
            name="bad", max_position_size=0.5, risk_per_trade=0.10,
            max_portfolio_heat=0.05, max_positions_per_sector=2,
            min_risk_reward=2.0, min_hold_days=1, max_hold_days=5,
        )


def test_max_concurrent_positions_derives_from_heat():
    assert AGGRESSIVE_SWING.max_concurrent_positions == 4  # 0.12 / 0.03


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def test_size_risks_exactly_the_budget():
    """The core identity: a full-conviction trade risks risk_per_trade.

    ATR is set high enough (8% of price) that the sizing formula, not the
    max_position_size cap, is what determines the answer.
    """
    entry, atr_value = 100.0, 8.0            # stop 2 ATR = 16.0 away -> 16%
    weight = size_from_risk(entry, atr_value, AGGRESSIVE_SWING, conviction=1.0)

    assert weight < AGGRESSIVE_SWING.max_position_size, "cap must not bind here"
    stop = entry - stop_distance(entry, atr_value, AGGRESSIVE_SWING)
    assert trade_risk(weight, entry, stop) == pytest.approx(0.03, abs=1e-9)


def test_tighter_stop_earns_a_bigger_position():
    """Volatility-aware sizing: same dollar risk, different share count."""
    quiet = size_from_risk(100.0, 7.0, AGGRESSIVE_SWING)
    wild = size_from_risk(100.0, 12.0, AGGRESSIVE_SWING)

    assert quiet > wild
    # Both still risk the same fraction of equity.
    quiet_stop = 100.0 - stop_distance(100.0, 7.0, AGGRESSIVE_SWING)
    wild_stop = 100.0 - stop_distance(100.0, 12.0, AGGRESSIVE_SWING)
    assert trade_risk(quiet, 100.0, quiet_stop) == pytest.approx(
        trade_risk(wild, 100.0, wild_stop), abs=1e-9
    )


def test_position_cap_binds_for_typical_low_vol_names():
    """Documented consequence of the aggressive_swing defaults.

    With risk_per_trade=0.03 and a 2-ATR stop, the risk formula only asks
    for less than the 25% cap once ATR exceeds ~6% of price. Large caps run
    far tighter than that, so for them max_position_size — not
    risk_per_trade — is the binding constraint, and every name arrives at
    the cap regardless of its volatility.
    """
    typical_large_cap_atr = 2.0              # 2% of a 100 price
    weight = size_from_risk(100.0, typical_large_cap_atr, AGGRESSIVE_SWING)

    assert weight == AGGRESSIVE_SWING.max_position_size


def test_size_is_capped_by_max_position_size():
    """A very tight stop must not imply a 90% position."""
    weight = size_from_risk(100.0, 0.05, AGGRESSIVE_SWING)
    assert weight == AGGRESSIVE_SWING.max_position_size


def test_conviction_scales_the_risk_budget():
    full = size_from_risk(100.0, 8.0, AGGRESSIVE_SWING, conviction=1.0)
    half = size_from_risk(100.0, 8.0, AGGRESSIVE_SWING, conviction=0.5)
    assert half == pytest.approx(full / 2)


def test_no_atr_means_no_position():
    """Refusing to size is the safe failure when volatility is unknown."""
    assert size_from_risk(100.0, 0.0, AGGRESSIVE_SWING) == 0.0
    assert size_from_risk(0.0, 2.0, AGGRESSIVE_SWING) == 0.0
    assert size_from_risk(100.0, 2.0, AGGRESSIVE_SWING, conviction=-0.5) == 0.0


# ---------------------------------------------------------------------------
# Portfolio heat
# ---------------------------------------------------------------------------

def test_heat_under_budget_is_untouched():
    weights = {"A": 0.25, "B": 0.25}
    entries = {"A": 100.0, "B": 100.0}
    stops = {"A": 96.0, "B": 96.0}          # 4% stop -> 1% risk each

    out, clamps, heat = apply_portfolio_heat(
        weights, stops, entries, AGGRESSIVE_SWING
    )

    assert out == weights
    assert clamps == []
    assert heat == pytest.approx(0.02)


def test_heat_over_budget_scales_book_down_proportionally():
    """Six full-risk trades want 18% heat against a 12% cap."""
    weights = {t: 0.25 for t in "ABCDEF"}
    entries = {t: 100.0 for t in "ABCDEF"}
    stops = {t: 88.0 for t in "ABCDEF"}     # 12% stop -> 3% risk each

    out, clamps, heat = apply_portfolio_heat(
        weights, stops, entries, AGGRESSIVE_SWING
    )

    assert heat == pytest.approx(0.12)
    assert len(clamps) == 1
    assert clamps[0].limit == "max_portfolio_heat"
    # Proportional: relative shape preserved, total risk inside the cap.
    assert all(w == pytest.approx(0.25 * (0.12 / 0.18)) for w in out.values())


def test_heat_never_increases_exposure():
    """Risk disposes; it must never upsize an under-risked book."""
    weights = {"A": 0.05}
    out, _, _ = apply_portfolio_heat(
        weights, {"A": 99.0}, {"A": 100.0}, AGGRESSIVE_SWING
    )
    assert out["A"] <= weights["A"]


# ---------------------------------------------------------------------------
# Sector concentration
# ---------------------------------------------------------------------------

def test_sector_cap_drops_the_weakest_names():
    weights = {"A": 0.20, "B": 0.10, "C": 0.25, "D": 0.05}
    sectors = dict.fromkeys("ABCD", "Information Technology")

    out, clamps = apply_sector_caps(weights, sectors, AGGRESSIVE_SWING)

    # Cap is 3; the weakest (D) is dropped entirely, not trimmed.
    assert out["D"] == 0.0
    assert out["C"] == 0.25 and out["A"] == 0.20 and out["B"] == 0.10
    assert [c.ticker for c in clamps] == ["D"]


def test_sector_cap_leaves_diversified_book_alone():
    weights = {"A": 0.2, "B": 0.2, "C": 0.2}
    sectors = {"A": "Health Care", "B": "Information Technology", "C": "Energy"}

    out, clamps = apply_sector_caps(weights, sectors, AGGRESSIVE_SWING)

    assert out == weights
    assert clamps == []


def test_unknown_sector_is_capped_not_exempted():
    """An unseen sector is unmeasured concentration, not zero concentration."""
    weights = {t: 0.1 for t in "ABCD"}

    out, clamps = apply_sector_caps(weights, {}, AGGRESSIVE_SWING)

    assert sum(1 for w in out.values() if w == 0.0) == 1
    assert clamps[0].detail is not None and "UNKNOWN" in clamps[0].detail


# ---------------------------------------------------------------------------
# Event gate
# ---------------------------------------------------------------------------

def test_event_gate_blocks_new_entry():
    out, clamps = apply_event_gate(
        {"A": 0.25}, {"A": True}, held={}, profile=AGGRESSIVE_SWING
    )
    assert out["A"] == 0.0
    assert clamps[0].limit == "event_gate"


def test_event_gate_allows_holding_and_reducing():
    """Blocking new exposure must not force a fire-sale of what is held."""
    held = {"A": 0.20}

    hold, _ = apply_event_gate({"A": 0.20}, {"A": True}, held, AGGRESSIVE_SWING)
    assert hold["A"] == 0.20

    reduce_, _ = apply_event_gate({"A": 0.05}, {"A": True}, held, AGGRESSIVE_SWING)
    assert reduce_["A"] == 0.05

    add, _ = apply_event_gate({"A": 0.25}, {"A": True}, held, AGGRESSIVE_SWING)
    assert add["A"] == 0.20          # capped at what was already held


def test_event_gate_is_noop_when_profile_disables_it():
    profile = AGGRESSIVE_SWING.model_copy(update={"avoid_binary_events": False})
    out, clamps = apply_event_gate({"A": 0.25}, {"A": True}, {}, profile)

    assert out["A"] == 0.25
    assert clamps == []


# ---------------------------------------------------------------------------
# Full stack
# ---------------------------------------------------------------------------

def test_full_stack_orders_stages_and_only_shrinks():
    weights = {t: 0.25 for t in ["A", "B", "C", "D", "E"]}
    marks = {t: 100.0 for t in weights}
    atrs = {t: 6.0 for t in weights}          # 2x6 = 12% stop -> 3% risk each
    sectors = {
        "A": "Tech", "B": "Tech", "C": "Tech", "D": "Tech", "E": "Energy",
    }

    result = apply_swing_risk(
        weights, AGGRESSIVE_SWING,
        entries=marks, atrs=atrs, sectors=sectors,
        blocked_by_event={"A": True}, held_weights={},
    )

    # A was gated out by the event; the Tech cap then applies to B/C/D only.
    assert result.weights["A"] == 0.0
    assert result.heat_used <= AGGRESSIVE_SWING.max_portfolio_heat + 1e-9
    assert all(
        abs(result.weights[t]) <= abs(weights[t]) + 1e-9 for t in weights
    ), "swing risk must never increase a position"
    assert {c.limit for c in result.clamps} >= {"event_gate"}


def test_full_stack_returns_the_stops_sizing_assumed():
    result = apply_swing_risk(
        {"A": 0.25}, AGGRESSIVE_SWING,
        entries={"A": 100.0}, atrs={"A": 2.0},
    )
    # 2 ATR below a 100 entry.
    assert result.stops["A"] == pytest.approx(96.0)
