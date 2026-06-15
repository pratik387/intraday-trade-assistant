"""Tests for MockBroker.fetch_daily_window — the daily-panel data source the
multi-day CNC/MTF setups need in PAPER mode (LiveDailyPanelProvider injects it as
fetch_fn). Assembles per-symbol daily bars (via the wired data SDK's get_daily)
into the long [date, symbol, OHLCV] panel the ranker consumes, windowed to
[start, end].
"""
from datetime import date

import pandas as pd
import pytest

from broker.mock.mock_broker import MockBroker


class _StubSDK:
    """Minimal data SDK: returns date-indexed OHLCV daily bars per symbol."""
    def get_daily(self, symbol, days):
        idx = pd.date_range("2026-01-01", periods=12, freq="B", name="date")
        base = {"AAA": 100.0, "BBB": 50.0}.get(symbol)
        if base is None:
            return pd.DataFrame()
        return pd.DataFrame(
            {"open": base, "high": base + 2, "low": base - 2, "close": base + 1, "volume": 1000},
            index=idx,
        ).tail(int(days))

    # MockBroker may probe these in paper mode; keep them harmless.
    def get_symbol_map(self):
        return {}


def _broker():
    return MockBroker(path_json="nse_all.json", slippage_bps=0.0, data_sdk=_StubSDK())


def test_fetch_daily_window_returns_long_panel_for_all_symbols():
    df = _broker().fetch_daily_window(["AAA", "BBB"], date(2026, 1, 5), date(2026, 1, 9))
    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert set(df["symbol"]) == {"AAA", "BBB"}


def test_fetch_daily_window_filters_to_inclusive_window():
    start, end = date(2026, 1, 5), date(2026, 1, 9)
    df = _broker().fetch_daily_window(["AAA"], start, end)
    assert df["date"].min() >= pd.Timestamp(start)
    assert df["date"].max() <= pd.Timestamp(end)
    assert len(df) > 0


def test_fetch_daily_window_strips_nse_prefix():
    df = _broker().fetch_daily_window(["NSE:AAA"], date(2026, 1, 5), date(2026, 1, 9))
    assert set(df["symbol"]) == {"AAA"}


def test_fetch_daily_window_skips_unknown_symbols_gracefully():
    df = _broker().fetch_daily_window(["AAA", "ZZZ_UNKNOWN"], date(2026, 1, 5), date(2026, 1, 9))
    assert set(df["symbol"]) == {"AAA"}  # ZZZ returns empty -> dropped, no raise


def test_fetch_daily_window_empty_when_no_data():
    df = _broker().fetch_daily_window(["ZZZ_UNKNOWN"], date(2026, 1, 5), date(2026, 1, 9))
    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Current-day bar synthesis from intraday 5m (the live-entry fix): get_daily
# DROPS today's partial bar, but the ranker requires a row dated exactly == the
# session date, so fetch_daily_window must synthesize today's daily bar from the
# session's 5m (open=first, high=max, low=min, close=last, volume=sum).
# ---------------------------------------------------------------------------

class _StubSDKNoToday:
    """get_daily returns history that STOPS the day before end_date (mimics the
    drop-today behavior); get_intraday_5m supplies the current day's 5m bars."""
    END = pd.Timestamp("2026-01-09")

    def get_daily(self, symbol, days):
        # history through 2026-01-08 only (no end_date bar)
        idx = pd.date_range("2026-01-01", "2026-01-08", freq="B", name="date")
        return pd.DataFrame(
            {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1000},
            index=idx,
        ).tail(int(days))

    def get_intraday_5m(self, symbol):
        # five 5m bars on END day: open 200, then rising; close last = 212
        ts = [pd.Timestamp("2026-01-09 09:15"), pd.Timestamp("2026-01-09 09:20"),
              pd.Timestamp("2026-01-09 13:00"), pd.Timestamp("2026-01-09 15:20"),
              pd.Timestamp("2026-01-09 15:25")]
        return pd.DataFrame(
            {"open": [200, 205, 208, 210, 211],
             "high": [206, 209, 215, 212, 213],
             "low":  [199, 204, 207, 209, 210],
             "close": [205, 208, 210, 211, 212],
             "volume": [100, 200, 300, 400, 500]},
            index=pd.DatetimeIndex(ts),
        )

    def get_symbol_map(self):
        return {}


def _broker_no_today():
    from broker.mock.mock_broker import MockBroker
    return MockBroker(path_json="nse_all.json", slippage_bps=0.0, data_sdk=_StubSDKNoToday())


def test_fetch_daily_window_synthesizes_today_bar_from_5m():
    """When get_daily lacks the end_date bar, it is built from that day's 5m."""
    df = _broker_no_today().fetch_daily_window(["AAA"], date(2026, 1, 5), date(2026, 1, 9))
    today = df[df["date"] == pd.Timestamp("2026-01-09")]
    assert len(today) == 1, "current-day bar must be present so the ranker can rank"
    row = today.iloc[0]
    assert row["open"] == 200.0          # first 5m open
    assert row["high"] == 215.0          # max 5m high
    assert row["low"] == 199.0           # min 5m low
    assert row["close"] == 212.0         # last 5m close
    assert row["volume"] == 1500.0       # summed 5m volume


def test_fetch_daily_window_no_synth_when_already_present():
    """If get_daily already returns the end_date bar, no 5m synthesis / no dup."""
    df = _broker().fetch_daily_window(["AAA"], date(2026, 1, 5), date(2026, 1, 9))
    assert (df["date"] == pd.Timestamp("2026-01-09")).sum() <= 1


# ---------------------------------------------------------------------------
# Intraday-5m prefetch: run_eod batch-fetches today's 5m for the universe ONCE
# (mirrors overnight_handlers' asyncio.run(async_fetch_intraday_5m_batch)) and
# stashes it on the broker, so fetch_daily_window's per-symbol get_intraday_5m
# are cache hits instead of ~1300 synchronous API calls.
# ---------------------------------------------------------------------------

class _SDKExplodes5m:
    """get_intraday_5m raises — proves a prefetch HIT avoids the live API."""
    def get_daily(self, symbol, days):
        return pd.DataFrame()  # force the synth path (no daily history)

    def get_intraday_5m(self, symbol):
        raise AssertionError("get_intraday_5m must not be called when prefetch has the symbol")

    def get_symbol_map(self):
        return {}


def _5m_frame():
    ts = [pd.Timestamp("2026-01-09 09:15"), pd.Timestamp("2026-01-09 15:25")]
    return pd.DataFrame(
        {"open": [200, 211], "high": [206, 213], "low": [199, 210],
         "close": [205, 212], "volume": [100, 500]},
        index=pd.DatetimeIndex(ts),
    )


def test_intraday_5m_prefetch_is_used_before_live_api():
    from broker.mock.mock_broker import MockBroker
    b = MockBroker(path_json="nse_all.json", slippage_bps=0.0, data_sdk=_SDKExplodes5m())
    b.set_intraday_5m_prefetch({"NSE:AAA": _5m_frame()})
    # get_intraday_5m returns the prefetched frame without hitting the SDK
    got = b.get_intraday_5m("NSE:AAA")
    assert got is not None and len(got) == 2
    # and fetch_daily_window synthesizes today's bar from it (no SDK call -> no raise)
    df = b.fetch_daily_window(["AAA"], date(2026, 1, 5), date(2026, 1, 9))
    today = df[df["date"] == pd.Timestamp("2026-01-09")]
    assert len(today) == 1 and today.iloc[0]["close"] == 212.0
