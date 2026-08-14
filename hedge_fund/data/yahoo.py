"""Yahoo chart-API data client — a keyless DataClient for research runs.

Why this exists: the stock FDClient needs a FINANCIAL_DATASETS_API_KEY, which
gates every backtest behind a paid credential. This client implements the same
`DataClient` protocol against Yahoo's public chart endpoint, so the engine can
be exercised end-to-end with real OHLCV bars and no API key.

Scope, stated honestly — this is a PRICE client, not a fundamentals client:

    get_prices              real daily OHLCV from Yahoo
    get_earnings_history    real report dates from a bundled reference file
    get_company_facts       real sector from a bundled reference file
    get_financial_metrics   [] — not available here
    get_news                [] — not available here
    get_insider_trades      [] — not available here
    get_market_cap          None — not available here

The empty returns are a deliberate, documented degradation, and they are
LEGAL under the DataClient contract ("empty list / None means the data
genuinely doesn't exist"). The consequence is real and must not be glossed:
the LLM investor agents (Buffett, Munger, Graham, Lynch) read fundamentals,
so under this client they see nothing and abstain. Fund mandates run against
this client should staff quant models. Infrastructure failures still RAISE,
per the protocol — a network error must never masquerade as "no signal".

Bars are split- and dividend-adjusted (Yahoo's `adjclose` ratio is applied to
OHLC), so a backtest sees a continuous series across splits.
"""

from __future__ import annotations

import json
import time as _time
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from hedge_fund.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsData,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
)

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-hedge-fund research)"}
_REFERENCE_DIR = Path(__file__).parent / "reference"
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 2


class YahooClientError(RuntimeError):
    """Infrastructure failure talking to Yahoo. Fail loud, never return []."""


class YahooDataClient:
    """Keyless DataClient backed by Yahoo bars + bundled reference data.

    One network call per ticker per process: the full requested history is
    fetched once and sliced in memory, then optionally persisted to
    *cache_dir* so repeat backtests are byte-identical and offline.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        reference_dir: str | Path | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._reference_dir = Path(reference_dir) if reference_dir else _REFERENCE_DIR
        self._bars: dict[str, list[Price]] = {}
        self._reference: dict[str, dict] | None = None

    # -- context manager, to mirror FDClient's usage in run.py -------------

    def __enter__(self) -> "YahooDataClient":
        return self

    def __exit__(self, *exc) -> None:
        return None

    # -- prices ------------------------------------------------------------

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        interval_multiplier: int = 1,
        **kwargs,
    ) -> list[Price]:
        """Daily adjusted OHLCV for *ticker* in [start_date, end_date].

        Empty list means the window genuinely has no bars (pre-IPO, delisted,
        a holiday-only range). Network/HTTP failures raise YahooClientError.

        Only daily bars are served. A caller asking for anything else raises
        rather than silently receiving daily data — a backtest that thinks it
        is running on 5-minute bars but is fed daily ones is a silent lie.
        """
        if interval != "day" or interval_multiplier != 1:
            raise YahooClientError(
                f"YahooDataClient serves daily bars only; got "
                f"interval={interval!r}, interval_multiplier={interval_multiplier}"
            )
        bars = self._all_bars(ticker.upper())
        return [b for b in bars if start_date <= b.time[:10] <= end_date]

    def _all_bars(self, ticker: str) -> list[Price]:
        if ticker in self._bars:
            return self._bars[ticker]

        cached = self._read_cache(ticker)
        if cached is not None:
            self._bars[ticker] = cached
            return cached

        bars = self._fetch(ticker)
        self._bars[ticker] = bars
        self._write_cache(ticker, bars)
        return bars

    def _fetch(self, ticker: str) -> list[Price]:
        """Pull max daily history for *ticker*. Raises on infrastructure failure."""
        params = {"range": "10y", "interval": "1d", "events": "div,split"}
        last_error: Exception | None = None

        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = requests.get(
                    _CHART_URL.format(ticker=ticker),
                    params=params,
                    headers=_HEADERS,
                    timeout=30,
                )
            except requests.RequestException as exc:
                last_error = exc
                _time.sleep(_BACKOFF_SECONDS * (attempt + 1))
                continue

            if response.status_code == 404:
                # Yahoo says this symbol does not exist. That is a real
                # "no data" answer, not an infrastructure failure.
                return []
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = YahooClientError(
                    f"GET chart/{ticker} returned {response.status_code}"
                )
                _time.sleep(_BACKOFF_SECONDS * (attempt + 1))
                continue
            if response.status_code != 200:
                raise YahooClientError(
                    f"GET chart/{ticker} returned {response.status_code}: "
                    f"{response.text[:200]}"
                )
            return _parse_chart(response.json(), ticker)

        raise YahooClientError(
            f"GET chart/{ticker} failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    # -- disk cache --------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{ticker}.json"

    def _read_cache(self, ticker: str) -> list[Price] | None:
        path = self._cache_path(ticker)
        if path is None or not path.exists():
            return None
        raw = json.loads(path.read_text())
        return [Price(**bar) for bar in raw]

    def _write_cache(self, ticker: str, bars: list[Price]) -> None:
        path = self._cache_path(ticker)
        if path is None or not bars:
            return
        path.write_text(json.dumps([b.model_dump() for b in bars], indent=1))

    # -- reference data ----------------------------------------------------

    def _ref(self, ticker: str) -> dict:
        """Bundled per-ticker reference: sector + earnings report dates."""
        if self._reference is None:
            path = self._reference_dir / "tickers.json"
            self._reference = json.loads(path.read_text()) if path.exists() else {}
        return self._reference.get(ticker.upper(), {})

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        ref = self._ref(ticker)
        if not ref:
            return None
        return CompanyFacts(
            ticker=ticker.upper(),
            name=ref.get("name"),
            sector=ref.get("sector"),
            industry=ref.get("industry"),
            exchange=ref.get("exchange"),
        )

    def get_earnings_history(
        self,
        ticker: str,
        limit: int = 12,
    ) -> list[EarningsRecord]:
        """Report dates from the bundled reference file.

        Only the fields a swing desk actually needs are populated: the filing
        date (when the market learned) and the EPS surprise label. Balance
        sheet / cash flow fields stay None — this client has no fundamentals.
        """
        records: list[EarningsRecord] = []
        for row in self._ref(ticker).get("earnings", [])[:limit]:
            records.append(EarningsRecord(
                ticker=ticker.upper(),
                report_period=row["report_period"],
                source_type=row.get("source_type", "8-K"),
                filing_date=row["report_date"],
                quarterly=EarningsData(
                    earnings_per_share=row.get("actual_eps"),
                    estimated_earnings_per_share=row.get("estimate_eps"),
                    eps_surprise=row.get("eps_surprise"),
                ),
            ))
        return records

    def get_earnings(self, ticker: str) -> Earnings | None:
        history = self.get_earnings_history(ticker, limit=1)
        if not history:
            return None
        latest = history[0]
        return Earnings(
            ticker=ticker.upper(),
            report_period=latest.report_period,
            quarterly=latest.quarterly,
        )

    # -- unavailable under this client (documented degradation) ------------

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        return []

    def get_news(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        return []

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        return []

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_chart(payload: dict, ticker: str) -> list[Price]:
    """Turn a Yahoo chart response into adjusted Price bars, oldest first.

    Yahoo reports raw OHLC plus an `adjclose` series. Applying the
    adjclose/close ratio to each of O/H/L/C yields a split- and
    dividend-adjusted bar, which is what a backtest needs to avoid fake
    gaps on ex-div and split dates.
    """
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise YahooClientError(f"chart/{ticker}: {chart['error']}")

    results = chart.get("result") or []
    if not results:
        return []

    result = results[0]
    stamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adjclose_series = (
        ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
        or []
    )

    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    bars: list[Price] = []
    for i, stamp in enumerate(stamps):
        o, h, l, c = (
            _at(opens, i), _at(highs, i), _at(lows, i), _at(closes, i)
        )
        if None in (o, h, l, c) or c == 0:
            continue  # Yahoo pads halted/missing sessions with nulls

        adj = _at(adjclose_series, i)
        ratio = (adj / c) if (adj is not None and c) else 1.0

        bars.append(Price(
            open=round(o * ratio, 6),
            high=round(h * ratio, 6),
            low=round(l * ratio, 6),
            close=round(c * ratio, 6),
            volume=int(_at(volumes, i) or 0),
            time=datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d"),
        ))

    bars.sort(key=lambda b: b.time)
    return bars


def _at(series: list, i: int):
    """Index *series* defensively — Yahoo arrays can be short or hold nulls."""
    if i < len(series):
        return series[i]
    return None
