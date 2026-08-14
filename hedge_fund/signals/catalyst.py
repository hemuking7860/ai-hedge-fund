"""Catalyst / event-risk alpha model — how exposed is this name to a binary?

A swing trade held 2-5 days that straddles an earnings print is not a swing
trade; it is a coin flip wearing one. This model reads the earnings calendar
and answers one question: how close is the next binary event, and did one
just happen?

It emits one of four states, exactly as specified, and degrades honestly:

    HIGH_CATALYST_RISK   a print lands inside the swing horizon -> negative
    MEDIUM_CATALYST      a print is near but outside the danger window
    NO_NEAR_CATALYST     clear runway -> mildly positive
    UNKNOWN              no calendar for this name -> ABSTAIN, not neutral

The UNKNOWN/abstain distinction is the important one. Under a data client
with no earnings coverage, returning 0.0 would silently vote "neutral" in
the blend and dilute every other model's view. Abstaining says "I have no
information", which blend_signals correctly excludes from the average.

The sign convention is deliberately asymmetric. Event risk is a RISK model:
its job is to veto, not to generate alpha. Clear-runway names get a small
positive nudge; imminent-print names get a large negative one. It should
never be the reason a trade is put on.

Point-in-time: an event is only "known" if its date is in the future
relative to *date*; past prints are used only for the just-reported check,
which is what a real desk would know at that moment.
"""

from __future__ import annotations

from datetime import date as _date

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

HIGH_CATALYST_RISK = "HIGH_CATALYST_RISK"
MEDIUM_CATALYST = "MEDIUM_CATALYST"
NO_NEAR_CATALYST = "NO_NEAR_CATALYST"
UNKNOWN = "UNKNOWN"


class CatalystRiskModel(QuantModel):
    """Score a ticker's near-term binary-event exposure."""

    def __init__(
        self,
        *,
        danger_days: int = 5,
        watch_days: int = 15,
        post_event_drift_days: int = 3,
        earnings_limit: int = 16,
    ) -> None:
        self._danger_days = danger_days
        self._watch_days = watch_days
        self._post_event_drift_days = post_event_drift_days
        self._earnings_limit = earnings_limit
        # predict() is called once per ticker per rebalance; cache the
        # calendar so a long backtest fetches each name once.
        self._cache: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "catalyst"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        as_of = _date.fromisoformat(date)
        calendar = self._calendar(ticker, data_client)

        if not calendar:
            return Signal(
                model_name=self.name, ticker=ticker, date=date, value=0.0,
                reasoning="no earnings calendar available for this name",
                metadata={"abstained": True, "catalyst_state": UNKNOWN},
            )

        days_until = _days_until_next(calendar, as_of)
        days_since = _days_since_last(calendar, as_of)
        state, value = self._classify(days_until, days_since)

        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=value,
            reasoning=_explain(state, days_until, days_since),
            components={
                "days_until_next": float(days_until if days_until is not None else -1),
                "days_since_last": float(days_since if days_since is not None else -1),
            },
            metadata={
                "abstained": False,
                "catalyst_state": state,
                "days_until_next_earnings": days_until,
                "days_since_last_earnings": days_since,
                # What the risk layer actually consumes to veto an entry.
                "blocks_new_entry": state == HIGH_CATALYST_RISK,
            },
        )

    def _classify(
        self,
        days_until: int | None,
        days_since: int | None,
    ) -> tuple[str, float]:
        """Map calendar distance to a state and a conviction."""
        if days_until is not None and days_until <= self._danger_days:
            # Closer = worse: -0.6 at the edge of the window, -1.0 on the eve.
            severity = 1.0 - (days_until / max(self._danger_days, 1)) * 0.4
            return HIGH_CATALYST_RISK, -round(severity, 4)

        if days_until is not None and days_until <= self._watch_days:
            return MEDIUM_CATALYST, -0.2

        # Freshly reported and past the event: the binary is behind us, which
        # is the cleanest runway a swing trade gets.
        if days_since is not None and days_since <= self._post_event_drift_days:
            return NO_NEAR_CATALYST, 0.3

        return NO_NEAR_CATALYST, 0.15

    def _calendar(self, ticker: str, data_client: DataClient) -> list[str]:
        """Sorted report dates for *ticker*, cached per process."""
        if ticker in self._cache:
            return self._cache[ticker]

        records = data_client.get_earnings_history(ticker, limit=self._earnings_limit)
        dates = sorted({r.filing_date[:10] for r in records if r.filing_date})
        self._cache[ticker] = dates
        return dates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_until_next(calendar: list[str], as_of: _date) -> int | None:
    """Calendar days to the next scheduled report, or None if none known."""
    future = [d for d in calendar if _date.fromisoformat(d) > as_of]
    if not future:
        return None
    return (_date.fromisoformat(future[0]) - as_of).days


def _days_since_last(calendar: list[str], as_of: _date) -> int | None:
    """Calendar days since the most recent report on or before *as_of*."""
    past = [d for d in calendar if _date.fromisoformat(d) <= as_of]
    if not past:
        return None
    return (as_of - _date.fromisoformat(past[-1])).days


def _explain(state: str, days_until: int | None, days_since: int | None) -> str:
    if state == HIGH_CATALYST_RISK:
        return f"earnings in {days_until}d — inside the swing horizon"
    if state == MEDIUM_CATALYST:
        return f"earnings in {days_until}d — approaching, not yet blocking"
    if days_since is not None and days_since <= 3:
        return f"reported {days_since}d ago — binary is behind us, runway clear"
    nxt = f"{days_until}d" if days_until is not None else "none scheduled"
    return f"no near catalyst (next: {nxt})"
