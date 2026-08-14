"""SwingContext — the swing-mode adapter bolted onto the standard cycle.

run_cycle stays one code path. When a fund is not in swing mode it receives
`swing=None` and behaves exactly as upstream; when it is, this object owns
the three things swing mode adds:

    resize()   turn blended convictions into risk-based position weights
    gate()     run the swing risk stack (event, sector, heat)
    exits()    ask the exit engine which open positions die today

Keeping the state here rather than inside run_cycle preserves run_cycle's
purity story: the pipeline is still "data -> analysts -> blend -> risk ->
execution -> record", with this object supplying the risk stage's extra
inputs and the execution stage's forced closes.

Per-ticker facts (ATR, sector, earnings calendar) are fetched once and
cached for the life of the context, because a backtest calls into this
object once per ticker per day.
"""

from __future__ import annotations

import pandas as pd

from hedge_fund.backtesting.exits import (
    ExitDecision,
    SwingExitEngine,
    trading_days_until,
)
from hedge_fund.config.swing_trading_config import SwingProfile
from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.risk.swing import SwingRiskResult, apply_swing_risk, size_from_risk
from hedge_fund.signals.swing_momentum import atr, bars_to_frame

_ATR_HISTORY_DAYS = 120


class SwingContext:
    """Swing-mode state for one fund run (a cycle or a whole backtest)."""

    def __init__(self, profile: SwingProfile) -> None:
        self.profile = profile
        self.engine = SwingExitEngine(profile=profile)
        self._sectors: dict[str, str] = {}
        self._calendars: dict[str, list[str]] = {}
        self._frames: dict[str, pd.DataFrame] = {}

    # -- per-ticker facts, cached -----------------------------------------

    def sector(self, ticker: str, data_client: DataClient) -> str:
        if ticker not in self._sectors:
            facts = data_client.get_company_facts(ticker)
            self._sectors[ticker] = (facts.sector if facts and facts.sector else "UNKNOWN")
        return self._sectors[ticker]

    def calendar(self, ticker: str, data_client: DataClient) -> list[str]:
        if ticker not in self._calendars:
            records = data_client.get_earnings_history(ticker, limit=16)
            self._calendars[ticker] = sorted(
                {r.filing_date[:10] for r in records if r.filing_date}
            )
        return self._calendars[ticker]

    def atr_for(self, ticker: str, as_of: str, data_client: DataClient) -> float:
        """ATR as of *as_of*, point-in-time.

        Not cached across dates — ATR is a moving quantity and caching it by
        ticker alone would freeze volatility at whatever the first cycle saw.
        """
        from datetime import date as _date
        from datetime import timedelta

        start = (_date.fromisoformat(as_of) - timedelta(days=_ATR_HISTORY_DAYS)).isoformat()
        bars = data_client.get_prices(ticker, start, as_of)
        if not bars:
            return 0.0
        frame = bars_to_frame(bars)
        if frame.empty or len(frame) < self.profile.atr_period + 1:
            return 0.0
        return atr(frame, self.profile.atr_period)

    def bar_on(self, ticker: str, as_of: str, data_client: DataClient):
        """The bar for *as_of* exactly, or None if the name did not trade."""
        bars = data_client.get_prices(ticker, as_of, as_of)
        return bars[-1] if bars else None

    # -- the three swing stages -------------------------------------------

    def resize(
        self,
        convictions: dict[str, float],
        marks: dict[str, float],
        as_of: str,
        data_client: DataClient,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Blended convictions -> risk-based weights, plus the ATRs used.

        This REPLACES upstream's cross-sectional normalisation for swing
        mode. Upstream splits a fixed gross target across names in
        proportion to conviction, which means a lone weak view still gets
        the whole book. Risk-based sizing instead asks "how much can I own
        such that the stop costs me risk_per_trade?" — the position size
        falls out of volatility and conviction, and the book is whatever
        those independently justify.
        """
        weights: dict[str, float] = {}
        atrs: dict[str, float] = {}
        for ticker, conviction in convictions.items():
            if conviction <= 0:
                weights[ticker] = 0.0        # long-only swing book
                continue
            entry = marks.get(ticker, 0.0)
            atr_value = self.atr_for(ticker, as_of, data_client)
            atrs[ticker] = atr_value
            weights[ticker] = size_from_risk(
                entry, atr_value, self.profile, conviction
            )
        return weights, atrs

    def gate(
        self,
        weights: dict[str, float],
        marks: dict[str, float],
        atrs: dict[str, float],
        signals: list[Signal],
        held_weights: dict[str, float],
        as_of: str,
        data_client: DataClient,
    ) -> SwingRiskResult:
        """Run the swing risk stack over *weights*."""
        sectors = {t: self.sector(t, data_client) for t in weights}
        blocked = _blocked_by_event(signals)
        return apply_swing_risk(
            weights,
            self.profile,
            entries=marks,
            atrs=atrs,
            sectors=sectors,
            blocked_by_event=blocked,
            held_weights=held_weights,
        )

    def exits(
        self,
        held: list[str],
        as_of: str,
        data_client: DataClient,
    ) -> list[ExitDecision]:
        """Advance every tracked position one bar and collect the closes."""
        decisions: list[ExitDecision] = []
        for ticker in sorted(held):
            bar = self.bar_on(ticker, as_of, data_client)
            if bar is None:
                continue  # no session for this name today; nothing to evaluate
            decision = self.engine.evaluate(
                ticker, bar,
                days_until_earnings=trading_days_until(
                    self.calendar(ticker, data_client), as_of
                ),
            )
            if decision is not None:
                decisions.append(decision)
        return decisions

    def sync_positions(
        self,
        positions_after: dict[str, int],
        marks: dict[str, float],
        atrs: dict[str, float],
        as_of: str,
    ) -> None:
        """Reconcile the exit engine's book with the broker's after fills.

        Newly opened names start being tracked (with the stop their sizing
        assumed); names that left the book stop being tracked. Positions
        that merely changed size keep their original entry and stop — a
        top-up does not reset the trade's risk clock, which is what a desk
        means by "the same trade".
        """
        for ticker, shares in positions_after.items():
            if shares <= 0 or ticker in self.engine.positions:
                continue
            self.engine.open_position(
                ticker, marks.get(ticker, 0.0), as_of, atrs.get(ticker, 0.0)
            )

        for ticker in list(self.engine.positions):
            if positions_after.get(ticker, 0) <= 0:
                self.engine.close_position(ticker)


def _blocked_by_event(signals: list[Signal]) -> dict[str, bool]:
    """Which tickers the catalyst model flagged as un-enterable."""
    blocked: dict[str, bool] = {}
    for signal in signals:
        if signal.metadata.get("blocks_new_entry"):
            blocked[signal.ticker] = True
    return blocked
