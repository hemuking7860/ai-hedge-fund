"""Tests for YahooDataClient — parsing, contract compliance, degradation.

The parsing tests use recorded payload shapes rather than the network, so
they run offline and deterministically. One opt-in live test (marked, skipped
by default) checks the real endpoint still answers in the expected shape.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hedge_fund.data.protocol import DataClient
from hedge_fund.data.yahoo import YahooClientError, YahooDataClient, _parse_chart


def _chart(timestamps, opens, highs, lows, closes, volumes, adjcloses=None):
    """Build a Yahoo chart payload of the shape the real endpoint returns."""
    indicators = {"quote": [{
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    }]}
    if adjcloses is not None:
        indicators["adjclose"] = [{"adjclose": adjcloses}]
    return {"chart": {"result": [{
        "timestamp": timestamps, "indicators": indicators,
    }], "error": None}}


def _stamp(day: str) -> int:
    """Epoch seconds for *day* at 13:30 UTC — a US cash-open bar stamp.

    Derived rather than hardcoded so the expected dates in each test read
    directly off the input.
    """
    return int(
        datetime.strptime(day, "%Y-%m-%d")
        .replace(hour=13, minute=30, tzinfo=timezone.utc)
        .timestamp()
    )


_T1, _T2 = _stamp("2026-08-03"), _stamp("2026-08-04")


def test_parses_bars_oldest_first():
    payload = _chart(
        [_T2, _T1],
        opens=[276.17, 278.29], highs=[281.07, 287.2],
        lows=[275.82, 278.0], closes=[277.42, 284.02],
        volumes=[36849608, 52639201],
    )
    bars = _parse_chart(payload, "AMZN")

    assert [b.time for b in bars] == ["2026-08-03", "2026-08-04"]
    assert bars[0].close == 284.02
    assert bars[0].high == 287.2
    assert bars[1].volume == 36849608


def test_applies_adjclose_ratio_to_ohlc():
    """A 2:1 split shows up as adjclose = close/2; every leg must scale."""
    payload = _chart(
        [_T1], opens=[200.0], highs=[210.0], lows=[190.0],
        closes=[200.0], volumes=[1000], adjcloses=[100.0],
    )
    bar = _parse_chart(payload, "X")[0]

    assert bar.close == 100.0
    assert bar.open == 100.0
    assert bar.high == 105.0
    assert bar.low == 95.0


def test_skips_null_padded_sessions():
    """Yahoo pads halted sessions with nulls; those are not tradeable bars."""
    payload = _chart(
        [_T1, _T2], opens=[100.0, None], highs=[101.0, None],
        lows=[99.0, None], closes=[100.5, None], volumes=[10, None],
    )
    bars = _parse_chart(payload, "X")

    assert len(bars) == 1
    assert bars[0].time == "2026-08-03"


def test_empty_result_is_empty_not_error():
    assert _parse_chart({"chart": {"result": [], "error": None}}, "X") == []


def test_chart_error_raises():
    payload = {"chart": {"result": None, "error": {"code": "Not Found"}}}
    with pytest.raises(YahooClientError):
        _parse_chart(payload, "NOPE")


def test_satisfies_dataclient_protocol():
    assert isinstance(YahooDataClient(), DataClient)


def test_get_prices_slices_the_window(tmp_path):
    """get_prices must window by date, and read from cache without network."""
    cache = tmp_path / "bars"
    cache.mkdir()
    (cache / "FAKE.json").write_text(json.dumps([
        {"open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10, "time": "2026-01-01"},
        {"open": 2, "high": 3, "low": 2, "close": 2.5, "volume": 20, "time": "2026-01-02"},
        {"open": 3, "high": 4, "low": 3, "close": 3.5, "volume": 30, "time": "2026-01-03"},
    ]))
    client = YahooDataClient(cache_dir=cache)

    window = client.get_prices("FAKE", "2026-01-02", "2026-01-03")

    assert [b.time for b in window] == ["2026-01-02", "2026-01-03"]


def test_reference_lookups(tmp_path):
    """Sector and earnings come from the bundled reference file."""
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "tickers.json").write_text(json.dumps({
        "ACME": {
            "name": "Acme", "sector": "Industrials",
            "earnings": [
                {"report_period": "2026-06-30", "report_date": "2026-07-30",
                 "eps_surprise": "BEAT"},
            ],
        }
    }))
    client = YahooDataClient(reference_dir=ref)

    facts = client.get_company_facts("acme")
    assert facts is not None and facts.sector == "Industrials"

    history = client.get_earnings_history("ACME")
    assert len(history) == 1
    assert history[0].filing_date == "2026-07-30"
    assert history[0].quarterly.eps_surprise == "BEAT"

    assert client.get_company_facts("UNKNOWN") is None
    assert client.get_earnings_history("UNKNOWN") == []


def test_unavailable_endpoints_degrade_not_raise():
    """Documented degradation: no fundamentals, but never a fake failure."""
    client = YahooDataClient()

    assert client.get_financial_metrics("AMZN", "2026-01-01") == []
    assert client.get_news("AMZN", "2026-01-01") == []
    assert client.get_insider_trades("AMZN", "2026-01-01") == []
    assert client.get_market_cap("AMZN", "2026-01-01") is None


def test_real_sector_data_is_bundled():
    """The shipped reference file must actually cover the research basket."""
    client = YahooDataClient()

    for ticker, sector in [
        ("AMZN", "Consumer Discretionary"),
        ("MSFT", "Information Technology"),
        ("HCA", "Health Care"),
    ]:
        facts = client.get_company_facts(ticker)
        assert facts is not None, f"{ticker} missing from reference data"
        assert facts.sector == sector


@pytest.mark.live
def test_live_fetch_returns_bars():
    """Opt-in: hits the real endpoint. Run with -m live."""
    bars = YahooDataClient().get_prices("AAPL", "2026-01-05", "2026-01-09")
    assert bars, "expected bars for AAPL in a normal trading week"
    assert all(b.high >= b.low for b in bars)
