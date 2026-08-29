"""Swing momentum alpha model — technical timing, scored and explained.

Upstream ships no technical analysis at all: every alpha model is either an
LLM persona reading fundamentals or PEAD reading earnings surprises. This
model is the missing piece for a swing desk — it reads bars.

The score is a weighted blend of six checks a discretionary swing trader
would run down a chart, each normalised to [-1, +1]:

    trend_stack      price above 20EMA and 50EMA, 20EMA above 50EMA
    trend_slope      is the 20EMA actually rising
    rsi_zone         bullish but not exhausted (peaks ~55-65, punished >75)
    volume_expansion today's volume against its own 20-day average
    volatility_regime ATR% expanding modestly is good, blowing out is not
    structure        breakout above recent highs, or a held pullback

Two things come back, per the brief: a numeric score in `value`, and a
structured decomposition in `components` so a downstream agent (or a human
reading the record) can see WHICH leg carried the view rather than being
handed one opaque number.

Point-in-time by construction: only bars with time <= date are ever loaded,
and every indicator is computed on that truncated series.

Insufficient history abstains (metadata.abstained=True) rather than
returning 0.0. That distinction is load-bearing upstream: blend_signals
excludes abstentions from the average, so "I cannot see enough bars" does
not get averaged in as "I am neutral".
"""

from __future__ import annotations

from datetime import date as _date
from datetime import timedelta

import pandas as pd

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

# Enough calendar days to cover the longest lookback (50-period EMA needs
# ~50 trading days; 100 calendar days of slack covers holidays comfortably).
_HISTORY_DAYS = 400

_WEIGHTS = {
    "trend_stack": 0.30,
    "trend_slope": 0.15,
    "rsi_zone": 0.15,
    "volume_expansion": 0.15,
    "volatility_regime": 0.10,
    "structure": 0.15,
}


class SwingMomentumModel(QuantModel):
    """Score a name's swing-trade timing from its own bars.

    Long-only in spirit but signed in output: a fully broken-down stack
    scores negative, which a market-neutral sleeve can short and a long-only
    sleeve simply will not buy.
    """

    def __init__(
        self,
        *,
        fast_ema: int = 20,
        slow_ema: int = 50,
        rsi_period: int = 14,
        atr_period: int = 14,
        volume_lookback: int = 20,
        breakout_lookback: int = 20,
        min_bars: int = 60,
    ) -> None:
        self._fast = fast_ema
        self._slow = slow_ema
        self._rsi_period = rsi_period
        self._atr_period = atr_period
        self._volume_lookback = volume_lookback
        self._breakout_lookback = breakout_lookback
        self._min_bars = min_bars

    @property
    def name(self) -> str:
        return "swing_momentum"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        frame = _load_bars(ticker, date, data_client, self._min_bars)
        if frame is None or len(frame) < self._min_bars:
            n = 0 if frame is None else len(frame)
            return Signal(
                model_name=self.name, ticker=ticker, date=date, value=0.0,
                reasoning=(
                    f"insufficient history: {n} bars, need {self._min_bars}"
                ),
                metadata={"abstained": True, "bars": n},
            )

        components = self.score_components(frame)
        value = sum(_WEIGHTS[k] * v for k, v in components.items())
        value = self._normalize_to_signal(value)

        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=round(value, 6),
            reasoning=_explain(components, value),
            components={k: round(v, 4) for k, v in components.items()},
            metadata={
                "abstained": False,
                "atr": round(atr(frame, self._atr_period), 4),
                "atr_pct": round(atr_pct(frame, self._atr_period), 6),
                "close": float(frame["close"].iloc[-1]),
                "setup": _setup_label(components),
            },
        )

    # ------------------------------------------------------------------
    # Scoring — one small helper per leg, all pure and separately testable
    # ------------------------------------------------------------------

    def score_components(self, frame: pd.DataFrame) -> dict[str, float]:
        """Every leg of the score, each in [-1, +1]."""
        return {
            "trend_stack": self._trend_stack(frame),
            "trend_slope": self._trend_slope(frame),
            "rsi_zone": self._rsi_zone(frame),
            "volume_expansion": self._volume_expansion(frame),
            "volatility_regime": self._volatility_regime(frame),
            "structure": self._structure(frame),
        }

    def _trend_stack(self, frame: pd.DataFrame) -> float:
        """Price > 20EMA, price > 50EMA, 20EMA > 50EMA. Three votes."""
        close = float(frame["close"].iloc[-1])
        fast = float(ema(frame["close"], self._fast).iloc[-1])
        slow = float(ema(frame["close"], self._slow).iloc[-1])

        votes = [close > fast, close > slow, fast > slow]
        return (sum(1 if v else -1 for v in votes)) / 3.0

    def _trend_slope(self, frame: pd.DataFrame) -> float:
        """Is the fast EMA rising? Measured in ATRs so it is comparable
        across a $35 ETF and a $500 hospital chain."""
        fast_series = ema(frame["close"], self._fast)
        if len(fast_series) < 6:
            return 0.0
        change = float(fast_series.iloc[-1] - fast_series.iloc[-6])
        unit = atr(frame, self._atr_period)
        if unit <= 0:
            return 0.0
        return self._normalize_to_signal(change / unit)

    def _rsi_zone(self, frame: pd.DataFrame) -> float:
        """Bullish but not exhausted.

        Rewards the 50-70 band, peaks around 60, and turns negative both
        below 40 (no demand) and above 80 (blow-off, poor swing entry).
        """
        value = self._compute_rsi(frame["close"], self._rsi_period)
        if value >= 80:
            return -0.5 - min((value - 80) / 20.0, 0.5)
        if value >= 70:
            return 1.0 - (value - 70) / 10.0 * 1.5   # 70 -> 1.0, 80 -> -0.5
        if value >= 50:
            return 0.5 + (value - 50) / 20.0 * 0.5   # 50 -> 0.5, 70 -> 1.0
        if value >= 40:
            return (value - 40) / 10.0 * 0.5         # 40 -> 0.0, 50 -> 0.5
        return max(-1.0, (value - 40) / 40.0)        # 40 -> 0.0, 0 -> -1.0

    def _volume_expansion(self, frame: pd.DataFrame) -> float:
        """Today's volume vs its own 20-day average.

        Participation confirms a move. 1.0x average scores 0; 2x or better
        saturates at +1; a dried-up half-volume tape scores negative.
        """
        volume = frame["volume"].tail(self._volume_lookback)
        if len(volume) < 5:
            return 0.0
        average = float(volume.mean())
        if average <= 0:
            return 0.0
        ratio = float(frame["volume"].iloc[-1]) / average
        return self._normalize_to_signal((ratio - 1.0))

    def _volatility_regime(self, frame: pd.DataFrame) -> float:
        """Mild ATR expansion is a swing trader's friend; a blow-out is not.

        Compares current ATR to its own median over the lookback. Expansion
        up to ~1.5x scores positive, beyond ~2x scores negative — that is
        news-shock territory, where stops get gapped rather than hit.
        """
        true_range = _true_range(frame)
        if len(true_range) < self._atr_period * 2:
            return 0.0
        current = float(true_range.tail(self._atr_period).mean())
        baseline = float(true_range.tail(self._atr_period * 4).median())
        if baseline <= 0:
            return 0.0
        ratio = current / baseline
        if ratio <= 1.0:
            return (ratio - 1.0) * 2.0            # 0.5x -> -1.0, 1.0x -> 0
        if ratio <= 1.5:
            return (ratio - 1.0) / 0.5            # 1.0x -> 0, 1.5x -> +1.0
        return max(-1.0, 1.0 - (ratio - 1.5) * 2.0)  # 2.0x -> 0, 2.5x -> -1.0

    def _structure(self, frame: pd.DataFrame) -> float:
        """Breakout continuation, or a pullback that held its fast EMA.

        Two setups a swing desk actually trades, scored on one axis:
        pushing through the recent high is best; sitting just under it after
        a shallow pullback that held the 20EMA is next best; losing the
        20EMA inside a downtrend is worst.
        """
        window = frame.tail(self._breakout_lookback + 1)
        if len(window) < 5:
            return 0.0

        close = float(frame["close"].iloc[-1])
        prior_high = float(window["high"].iloc[:-1].max())
        prior_low = float(window["low"].iloc[:-1].min())
        span = prior_high - prior_low
        if span <= 0:
            return 0.0

        fast = float(ema(frame["close"], self._fast).iloc[-1])

        if close > prior_high:
            return 1.0                            # clean breakout
        position = (close - prior_low) / span     # 0 at range low, 1 at high
        if close >= fast:
            return self._normalize_to_signal(position * 1.2 - 0.1)
        return self._normalize_to_signal(position - 0.9)


# ---------------------------------------------------------------------------
# Indicator helpers — module level so risk/exit code can reuse them
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _true_range(frame: pd.DataFrame) -> pd.Series:
    """Wilder's true range: the widest of today's range and the two gaps."""
    high, low = frame["high"], frame["low"]
    prior_close = frame["close"].shift(1)
    return pd.concat([
        high - low,
        (high - prior_close).abs(),
        (low - prior_close).abs(),
    ], axis=1).max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> float:
    """Latest Average True Range in price units. 0.0 if not computable."""
    true_range = _true_range(frame)
    if len(true_range) < 2:
        return 0.0
    value = true_range.tail(period).mean()
    return 0.0 if pd.isna(value) else float(value)


def atr_pct(frame: pd.DataFrame, period: int = 14) -> float:
    """ATR as a fraction of last close — the comparable volatility unit."""
    close = float(frame["close"].iloc[-1])
    if close <= 0:
        return 0.0
    return atr(frame, period) / close


def bars_to_frame(bars) -> pd.DataFrame:
    """Price bars -> a sorted OHLCV DataFrame indexed by date string."""
    frame = pd.DataFrame([{
        "time": b.time[:10],
        "open": b.open, "high": b.high, "low": b.low,
        "close": b.close, "volume": b.volume,
    } for b in bars])
    if frame.empty:
        return frame
    return frame.sort_values("time").reset_index(drop=True)


def _load_bars(
    ticker: str,
    date: str,
    data_client: DataClient,
    min_bars: int,
) -> pd.DataFrame | None:
    """Point-in-time bar history ending at *date*, as a DataFrame."""
    start = (_date.fromisoformat(date) - timedelta(days=_HISTORY_DAYS)).isoformat()
    bars = data_client.get_prices(ticker, start, date)
    if not bars:
        return None
    frame = bars_to_frame(bars)
    if frame.empty:
        return None
    # Defensive: the client is contracted to window, but a mis-windowed
    # client would leak lookahead straight into the backtest.
    return frame[frame["time"] <= date].reset_index(drop=True)


def _setup_label(components: dict[str, float]) -> str:
    """Name the setup, for the record and for downstream agents."""
    if components["structure"] >= 1.0:
        return "BREAKOUT"
    if components["trend_stack"] > 0 and components["structure"] > 0:
        return "PULLBACK_CONTINUATION"
    if components["trend_stack"] < 0:
        return "BROKEN_TREND"
    return "NO_SETUP"


def _explain(components: dict[str, float], value: float) -> str:
    """A one-line rationale naming the legs that actually moved the score."""
    ranked = sorted(
        components.items(), key=lambda kv: abs(kv[1] * _WEIGHTS[kv[0]]), reverse=True
    )
    drivers = ", ".join(f"{k} {v:+.2f}" for k, v in ranked[:3])
    return f"{_setup_label(components)} score {value:+.2f} — driven by {drivers}"
