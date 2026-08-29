"""Tests for the swing momentum and catalyst alpha models."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from hedge_fund.data.models import EarningsData, EarningsRecord, Price
from hedge_fund.signals.catalyst import (
    HIGH_CATALYST_RISK,
    MEDIUM_CATALYST,
    NO_NEAR_CATALYST,
    UNKNOWN,
    CatalystRiskModel,
)
from hedge_fund.signals.swing_momentum import (
    SwingMomentumModel,
    atr,
    bars_to_frame,
    ema,
)


# ---------------------------------------------------------------------------
# Fake data client
# ---------------------------------------------------------------------------

class FakeClient:
    """Minimal DataClient stand-in: bars plus an optional earnings calendar."""

    def __init__(self, bars=None, earnings=None, facts=None):
        self._bars = bars or []
        self._earnings = earnings or []
        self._facts = facts

    def get_prices(self, ticker, start_date, end_date, *a, **kw):
        return [b for b in self._bars if start_date <= b.time <= end_date]

    def get_earnings_history(self, ticker, limit=12):
        return self._earnings[:limit]

    def get_company_facts(self, ticker):
        return self._facts

    def get_financial_metrics(self, *a, **kw): return []
    def get_news(self, *a, **kw): return []
    def get_insider_trades(self, *a, **kw): return []
    def get_earnings(self, *a, **kw): return None
    def get_market_cap(self, *a, **kw): return None


def series(closes, *, start="2025-09-01", volume=1_000_000, spread=0.01):
    """Build daily bars from a close series, with a plausible OHLC envelope."""
    bars = []
    day = date.fromisoformat(start)
    for i, close in enumerate(closes):
        prior = closes[i - 1] if i else close
        bars.append(Price(
            open=round(prior, 4),
            high=round(max(prior, close) * (1 + spread), 4),
            low=round(min(prior, close) * (1 - spread), 4),
            close=round(close, 4),
            volume=volume,
            time=(day + timedelta(days=i)).isoformat(),
        ))
    return bars


def uptrend(n=120, start_price=100.0, step=0.6):
    return [start_price + i * step for i in range(n)]


def downtrend(n=120, start_price=180.0, step=0.6):
    return [start_price - i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def test_ema_tracks_a_rising_series():
    frame = bars_to_frame(series(uptrend(60)))
    fast = ema(frame["close"], 20)
    assert fast.iloc[-1] > fast.iloc[-20]
    assert fast.iloc[-1] < frame["close"].iloc[-1], "EMA lags a rising series"


def test_atr_is_positive_and_scales_with_range():
    calm = bars_to_frame(series(uptrend(60), spread=0.005))
    wild = bars_to_frame(series(uptrend(60), spread=0.05))
    assert atr(calm) > 0
    assert atr(wild) > atr(calm)


def test_bars_to_frame_sorts_oldest_first():
    bars = list(reversed(series([100, 101, 102])))
    frame = bars_to_frame(bars)
    assert list(frame["time"]) == sorted(frame["time"])


# ---------------------------------------------------------------------------
# Momentum scoring
# ---------------------------------------------------------------------------

def test_clean_uptrend_scores_positive():
    client = FakeClient(bars=series(uptrend()))
    signal = SwingMomentumModel().predict("A", "2025-12-30", client)

    assert signal.metadata["abstained"] is False
    assert signal.value > 0.3
    assert signal.components["trend_stack"] > 0


def test_broken_downtrend_scores_negative():
    client = FakeClient(bars=series(downtrend()))
    signal = SwingMomentumModel().predict("A", "2025-12-30", client)

    assert signal.value < 0
    assert signal.components["trend_stack"] < 0
    assert signal.metadata["setup"] == "BROKEN_TREND"


def test_returns_structured_components_not_just_a_number():
    """The brief: a numeric score AND a structured explanation."""
    client = FakeClient(bars=series(uptrend()))
    signal = SwingMomentumModel().predict("A", "2025-12-30", client)

    assert set(signal.components) == {
        "trend_stack", "trend_slope", "rsi_zone",
        "volume_expansion", "volatility_regime", "structure",
    }
    assert signal.reasoning and "score" in signal.reasoning
    assert "setup" in signal.metadata
    assert signal.metadata["atr"] > 0


def test_insufficient_history_abstains_rather_than_voting_neutral():
    """Abstain != neutral: blend_signals excludes one and averages the other."""
    client = FakeClient(bars=series(uptrend(10)))
    signal = SwingMomentumModel().predict("A", "2025-12-30", client)

    assert signal.metadata["abstained"] is True
    assert signal.value == 0.0
    assert "insufficient history" in signal.reasoning


def test_no_bars_at_all_abstains():
    signal = SwingMomentumModel().predict("A", "2025-12-30", FakeClient())
    assert signal.metadata["abstained"] is True


def test_is_point_in_time():
    """Bars after the as-of date must not influence the score."""
    bars = series(uptrend(90))
    model = SwingMomentumModel()
    cut = bars[70].time

    full = model.predict("A", bars[-1].time, FakeClient(bars=bars))
    early = model.predict("A", cut, FakeClient(bars=bars))

    assert full.metadata["close"] != early.metadata["close"]
    assert early.metadata["close"] == pytest.approx(bars[70].close)


def test_overbought_rsi_is_penalised():
    """A vertical blow-off is a bad swing entry, not a great one.

    The healthy series must contain pullbacks: a strictly monotonic rise has
    no down closes at all, so its RSI pins at 100 and it would score as
    'exhausted' too. Real uptrends breathe.
    """
    model = SwingMomentumModel()
    healthy = bars_to_frame(series(
        [100 + i * 0.4 + (3.0 if i % 5 == 0 else 0) - (2.5 if i % 7 == 0 else 0)
         for i in range(120)]
    ))
    parabolic = bars_to_frame(series(
        [100 + i * 0.3 for i in range(100)] + [130 + i * 6 for i in range(20)]
    ))

    healthy_rsi = model.score_components(healthy)["rsi_zone"]
    assert healthy_rsi > -1.0, "a breathing uptrend should not read as exhausted"
    assert model.score_components(parabolic)["rsi_zone"] < healthy_rsi


def test_volume_expansion_rewards_participation():
    model = SwingMomentumModel()
    quiet = bars_to_frame(series(uptrend(60), volume=1_000_000))

    heavy_bars = series(uptrend(60), volume=1_000_000)
    heavy_bars[-1] = heavy_bars[-1].model_copy(update={"volume": 3_000_000})
    heavy = bars_to_frame(heavy_bars)

    assert model.score_components(heavy)["volume_expansion"] > \
        model.score_components(quiet)["volume_expansion"]


def test_breakout_is_labelled():
    closes = [100.0] * 60 + [100.0 + i * 0.1 for i in range(40)] + [120.0]
    frame = bars_to_frame(series(closes))
    components = SwingMomentumModel().score_components(frame)

    assert components["structure"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Catalyst / event risk
# ---------------------------------------------------------------------------

def earnings_on(*dates):
    return [
        EarningsRecord(
            ticker="A", report_period=d, source_type="8-K", filing_date=d,
            quarterly=EarningsData(eps_surprise="BEAT"),
        )
        for d in dates
    ]


def test_imminent_earnings_is_high_risk_and_blocks_entry():
    client = FakeClient(earnings=earnings_on("2026-05-01"))
    signal = CatalystRiskModel().predict("A", "2026-04-28", client)

    assert signal.metadata["catalyst_state"] == HIGH_CATALYST_RISK
    assert signal.metadata["blocks_new_entry"] is True
    assert signal.value < 0


def test_closer_earnings_scores_worse():
    client = FakeClient(earnings=earnings_on("2026-05-01"))
    model = CatalystRiskModel()

    eve = model.predict("A", "2026-04-30", client)
    edge = model.predict("A", "2026-04-26", client)

    assert eve.value < edge.value


def test_approaching_earnings_is_medium_and_does_not_block():
    client = FakeClient(earnings=earnings_on("2026-05-01"))
    signal = CatalystRiskModel().predict("A", "2026-04-20", client)

    assert signal.metadata["catalyst_state"] == MEDIUM_CATALYST
    assert signal.metadata["blocks_new_entry"] is False


def test_clear_runway_is_mildly_positive():
    client = FakeClient(earnings=earnings_on("2026-01-28", "2026-07-29"))
    signal = CatalystRiskModel().predict("A", "2026-03-15", client)

    assert signal.metadata["catalyst_state"] == NO_NEAR_CATALYST
    assert 0 < signal.value <= 0.3, "event risk should never drive a trade"


def test_just_reported_is_the_cleanest_runway():
    client = FakeClient(earnings=earnings_on("2026-01-28", "2026-07-29"))
    model = CatalystRiskModel()

    fresh = model.predict("A", "2026-01-30", client)
    stale = model.predict("A", "2026-03-15", client)

    assert fresh.value > stale.value


def test_missing_calendar_degrades_to_unknown_and_abstains():
    """The required fallback: UNKNOWN, and abstain rather than vote neutral."""
    signal = CatalystRiskModel().predict("A", "2026-03-15", FakeClient())

    assert signal.metadata["catalyst_state"] == UNKNOWN
    assert signal.metadata["abstained"] is True
    assert signal.value == 0.0


def test_all_four_states_are_reachable():
    model = CatalystRiskModel()
    calendar = FakeClient(earnings=earnings_on("2026-01-28", "2026-05-01"))

    # The UNKNOWN case needs its own model: the calendar cache is keyed by
    # ticker, so a warm instance would serve the cached dates instead.
    states = {
        model.predict("A", "2026-04-29", calendar).metadata["catalyst_state"],
        model.predict("A", "2026-04-20", calendar).metadata["catalyst_state"],
        model.predict("A", "2026-03-01", calendar).metadata["catalyst_state"],
        CatalystRiskModel().predict(
            "A", "2026-03-01", FakeClient()
        ).metadata["catalyst_state"],
    }
    assert states == {
        HIGH_CATALYST_RISK, MEDIUM_CATALYST, NO_NEAR_CATALYST, UNKNOWN,
    }


def test_catalyst_is_point_in_time():
    """A future print must not be treated as already reported."""
    client = FakeClient(earnings=earnings_on("2026-05-01"))
    signal = CatalystRiskModel().predict("A", "2026-02-01", client)

    assert signal.components["days_since_last"] == -1.0   # nothing in the past
    assert signal.components["days_until_next"] > 0
