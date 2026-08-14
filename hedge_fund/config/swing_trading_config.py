"""Swing-trading profiles — risk and holding-period policy as data.

A *profile* is the knob-set a swing desk actually argues about: how big a
position may get, how much of the book may be at risk at once, how long a
trade is allowed to live, and how it behaves around binary events. Three
ship by name:

    conservative      small size, tight heat, long minimum reward
    balanced          the middle
    aggressive_swing  bigger size, more heat, short leash

These are POLICY, not edge. A profile never decides what to buy — it bounds
what happens after something is bought. That separation is why the same
profile can front any alpha model.

Backward compatibility: nothing in the stock pipeline reads this module. A
fund only acquires swing behaviour when its mandate asks for it (`swing:` in
the FundSpec) or the CLI passes --profile/--swing. Absent that, the engine
behaves exactly as upstream.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProfileName = Literal["conservative", "balanced", "aggressive_swing"]


class SwingProfile(BaseModel):
    """One named risk/holding policy.

    Fractions are of *equity*, not of notional: risk_per_trade=0.03 means a
    trade that gets stopped out costs ~3% of the book, which is the only
    definition of "risk per trade" that composes into portfolio heat.
    """

    model_config = ConfigDict(extra="forbid")

    name: str

    # -- sizing ----------------------------------------------------------
    max_position_size: float = Field(
        gt=0, le=1.0,
        description="hard cap on |weight| for one ticker, as a fraction of equity",
    )
    risk_per_trade: float = Field(
        gt=0, le=0.25,
        description="equity fraction lost if this trade hits its initial stop; "
        "with a stop distance this is what determines position size",
    )
    max_portfolio_heat: float = Field(
        gt=0, le=1.0,
        description="cap on summed open risk across the book — the number that "
        "stops ten uncorrelated-looking 3% trades from being a 30% bet",
    )
    max_positions_per_sector: int = Field(
        gt=0,
        description="concentration cap; swing setups cluster by sector and a "
        "sector shock hits every one of them on the same morning",
    )

    # -- trade quality ---------------------------------------------------
    min_risk_reward: float = Field(
        gt=0,
        description="minimum reward:risk to take the trade at all",
    )

    # -- holding period --------------------------------------------------
    min_hold_days: int = Field(
        ge=0, description="do not time-stop before this many trading days"
    )
    max_hold_days: int = Field(
        gt=0, description="close the position once it has lived this long, "
        "regardless of P&L — a swing trade that has not worked is dead money",
    )

    # -- event risk ------------------------------------------------------
    avoid_binary_events: bool = Field(
        default=True,
        description="refuse new entries into a known near-term binary event",
    )
    exit_before_earnings_days: int = Field(
        ge=0, default=1,
        description="close an open position this many trading days before its "
        "earnings print; 0 disables",
    )

    # -- stop mechanics --------------------------------------------------
    atr_stop_multiple: float = Field(
        gt=0, default=2.0,
        description="initial stop distance in ATRs below entry",
    )
    atr_period: int = Field(gt=1, default=14)
    breakeven_at_r: float = Field(
        ge=0, default=1.0,
        description="move the stop to entry once the trade is this many R in "
        "favour; 0 disables",
    )
    trail_atr_multiple: float = Field(
        ge=0, default=2.5,
        description="once breakeven is armed, trail this many ATRs below the "
        "high-water mark; 0 disables trailing",
    )
    profit_target_r: float = Field(
        ge=0, default=0.0,
        description="take profit at this many R; 0 disables. Off by default "
        "because a hard target caps the winners that pay for the losers",
    )

    @model_validator(mode="after")
    def _hold_window_is_coherent(self) -> "SwingProfile":
        if self.min_hold_days > self.max_hold_days:
            raise ValueError(
                f"{self.name}: min_hold_days ({self.min_hold_days}) exceeds "
                f"max_hold_days ({self.max_hold_days})"
            )
        return self

    @model_validator(mode="after")
    def _heat_admits_at_least_one_trade(self) -> "SwingProfile":
        if self.risk_per_trade > self.max_portfolio_heat:
            raise ValueError(
                f"{self.name}: risk_per_trade ({self.risk_per_trade}) exceeds "
                f"max_portfolio_heat ({self.max_portfolio_heat}) — no trade "
                "could ever be opened"
            )
        return self

    @property
    def max_concurrent_positions(self) -> int:
        """How many full-risk trades fit inside the heat budget."""
        return int(self.max_portfolio_heat / self.risk_per_trade)


CONSERVATIVE = SwingProfile(
    name="conservative",
    max_position_size=0.10,
    risk_per_trade=0.005,
    max_portfolio_heat=0.03,
    max_positions_per_sector=2,
    min_risk_reward=2.0,
    min_hold_days=3,
    max_hold_days=20,
    avoid_binary_events=True,
    exit_before_earnings_days=3,
    atr_stop_multiple=2.5,
    breakeven_at_r=1.0,
    trail_atr_multiple=3.0,
)

BALANCED = SwingProfile(
    name="balanced",
    max_position_size=0.15,
    risk_per_trade=0.01,
    max_portfolio_heat=0.06,
    max_positions_per_sector=3,
    min_risk_reward=2.0,
    min_hold_days=2,
    max_hold_days=15,
    avoid_binary_events=True,
    exit_before_earnings_days=2,
    atr_stop_multiple=2.0,
    breakeven_at_r=1.0,
    trail_atr_multiple=2.5,
)

AGGRESSIVE_SWING = SwingProfile(
    name="aggressive_swing",
    max_position_size=0.25,
    risk_per_trade=0.03,
    max_portfolio_heat=0.12,
    max_positions_per_sector=3,
    min_risk_reward=3.0,
    min_hold_days=2,
    max_hold_days=10,
    avoid_binary_events=True,
    exit_before_earnings_days=1,
    atr_stop_multiple=2.0,
    breakeven_at_r=1.0,
    trail_atr_multiple=2.5,
)

PROFILES: dict[str, SwingProfile] = {
    p.name: p for p in (CONSERVATIVE, BALANCED, AGGRESSIVE_SWING)
}


def get_profile(name: str) -> SwingProfile:
    """Look up a profile by name. Unknown names fail loud with the options."""
    if name not in PROFILES:
        raise ValueError(
            f"unknown swing profile {name!r}; available: {sorted(PROFILES)}"
        )
    return PROFILES[name]
