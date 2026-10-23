"""Daily-candle prewarm / cache fixes (2026-10-23).

Live symptoms 10-16..10-23: both intraday daemons refetched ~1540 symbols of
daily candles at the 09:21 first scan from one IP, blew Upstox's 2000/30min
cap, stalled scans 10-50 min and left 130-250 symbols with EMPTY daily data.
Causes covered here:
  - the fetch window was 210 CALENDAR days (~150 bars), so a disk-loaded cache
    (stamped len(df)) never satisfied days=210 and was refetched in full;
  - a failed refetch overwrote good cached data with an empty frame;
  - prewarm skipped entirely at >=90% coverage instead of fetching the gaps,
    and never retried failures.
"""
import threading
from datetime import date, timedelta

import pandas as pd
import pytest

import broker.upstox.upstox_data_client as mod
from broker.upstox.upstox_data_client import UpstoxDataClient


def _bars(n, last=date(2026, 10, 22)):
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=n)
    return pd.DataFrame({c: [1.0] * n for c in ["open", "high", "low", "close", "volume"]},
                        index=idx)


def _client(fetch, symbols=("A", "B", "C")):
    c = UpstoxDataClient.__new__(UpstoxDataClient)
    c._daily_cache = {}
    c._daily_cache_days = {}
    c._daily_cache_day = "2026-10-23"
    c._daily_cache_lock = threading.RLock()
    c._reset_daily_cache_if_new_day = lambda: None
    c._instrument_key_for = lambda s: s
    c._equity_instruments = list(symbols)
    c._rps = 8.0
    c._prewarm_retry_rounds = 2
    c._prewarm_retry_wait_sec = 60.0
    c._sleep = lambda s: c._slept.append(s)
    c._slept = []
    c._fetch_daily_candles = fetch
    return c


# --- fetch window: trading days, not calendar days -------------------------

def test_window_start_counts_trading_days(monkeypatch):
    monkeypatch.setattr(mod, "is_trading_day", lambda d: pd.Timestamp(d).weekday() < 5)
    c = UpstoxDataClient.__new__(UpstoxDataClient)
    c._window_extra_calendar_days = 15
    c._window_start_memo = {}
    c._window_start_lock = threading.Lock()
    end = date(2026, 10, 23)                       # Friday
    start = c._daily_window_start(end, 210)
    n_trading = len(pd.bdate_range(start + timedelta(days=15), end - timedelta(days=1)))
    assert n_trading == 210                        # 210 trading days before today, then +15 cal buffer
    assert (end - start).days > 290                # old code asked for only 220 calendar days


# --- a failed refetch must not wipe good cached data -----------------------

def test_empty_refetch_keeps_cached_data():
    calls = []

    def fetch(ikey, days):
        calls.append(days)
        return _bars(5) if len(calls) == 1 else pd.DataFrame()

    c = _client(fetch)
    c.get_daily("A", days=5)
    df = c.get_daily("A", days=210)                # deeper request, refetch fails
    assert calls == [5, 210]
    assert len(df) == 5                            # served the old bars, not empty
    assert len(c._daily_cache["A"]) == 5


# --- injected cache carries the depth it was fetched at --------------------

def test_injected_cache_stamped_with_requested_depth():
    c = _client(lambda ikey, days: pytest.fail("must not refetch"))
    c.set_daily_cache({"A": _bars(150)}, days=210)  # short-history symbol, fetched at 210
    assert len(c.get_daily("A", days=210)) == 150


# --- prewarm fetches only the gaps, retries failures -----------------------

def test_prewarm_fetches_only_missing_and_retries():
    attempts = {}

    def fetch(ikey, days):
        attempts[ikey] = attempts.get(ikey, 0) + 1
        if ikey == "C" and attempts[ikey] < 3:     # rate-limited twice, then ok
            return pd.DataFrame()
        return _bars(210)

    c = _client(fetch)
    c.set_daily_cache({"A": _bars(210)}, days=210)
    res = c.prewarm_daily_cache(symbols=["A", "B", "C"], days=210)

    assert "A" not in attempts                     # already satisfied
    assert attempts == {"B": 1, "C": 3}
    assert c._slept == [60.0, 60.0]
    assert res["failed"] == 0 and res["source"] == "api"


def test_prewarm_reports_symbols_still_missing():
    c = _client(lambda ikey, days: pd.DataFrame())
    res = c.prewarm_daily_cache(symbols=["A", "B"], days=210)
    assert res["failed"] == 2
    assert res["missing"] == ["A", "B"]
    assert len(c._slept) == 2                      # both rounds used


def test_prewarm_all_cached_is_cache_source():
    c = _client(lambda ikey, days: pytest.fail("must not fetch"))
    c.set_daily_cache({"A": _bars(210), "B": _bars(210)}, days=210)
    res = c.prewarm_daily_cache(symbols=["A", "B"], days=210)
    assert res["source"] == "cache" and res["failed"] == 0


# --- review follow-ups -------------------------------------------------------

def test_snapshot_export_excludes_shallow_entries():
    # A days=2 prevday entry must not be saved under the 210 stamp.
    c = _client(lambda ikey, days: _bars(days))
    c.get_daily("A", days=2)
    c.get_daily("B", days=210)
    assert set(c.get_daily_cache(min_days=210)) == {"B"}


def test_empty_refetch_over_prior_stamps_to_stop_repeat_calls():
    calls = []

    def fetch(ikey, days):
        calls.append(days)
        return _bars(2) if len(calls) == 1 else pd.DataFrame()

    c = _client(fetch)
    c.get_daily("A", days=2)
    c.get_daily("A", days=210)       # empty -> keeps prior, stamps 210
    c.get_daily("A", days=210)       # must NOT hit the API again
    assert calls == [2, 210]


def test_api_prewarm_survives_first_get_daily(monkeypatch):
    # Live 2026-10-23 restart: an API prewarm (no snapshot) left the cache's
    # day unset, so the first scan's get_daily "reset for a new day", cleared
    # the cache and refetched all 1541 symbols. Uses the REAL day-reset.
    calls = []

    def fetch(ikey, days):
        calls.append(ikey)
        return _bars(210)

    c = _client(fetch)
    del c._reset_daily_cache_if_new_day          # use the class method
    c._daily_cache_day = None                     # fresh process
    monkeypatch.setattr(mod, "_now_naive_ist", lambda: pd.Timestamp("2026-10-23 14:00"))
    c.prewarm_daily_cache(symbols=["A", "B"], days=210)
    c.get_daily("A", days=210)
    c.get_daily("B", days=210)
    assert calls == ["A", "B"]                    # no refetch after prewarm
