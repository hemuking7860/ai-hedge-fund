"""Named configuration profiles for the fund."""

from hedge_fund.config.swing_trading_config import (
    AGGRESSIVE_SWING,
    BALANCED,
    CONSERVATIVE,
    PROFILES,
    SwingProfile,
    get_profile,
)

__all__ = [
    "AGGRESSIVE_SWING",
    "BALANCED",
    "CONSERVATIVE",
    "PROFILES",
    "SwingProfile",
    "get_profile",
]
