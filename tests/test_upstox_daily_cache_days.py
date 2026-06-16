"""get_daily must refetch when a LARGER `days` is requested than was cached.

Bug: the cache check `len(cached) >= min(days, len(cached))` is always true, so a
prior small-`days` fetch (e.g. get_prevday_levels uses days=2) poisons later
large requests — the regime layer asks for 210 and silently gets the 2-bar cache.
Fix: track the requested days; refetch only when more is asked, without thrashing
short-history symbols.
"""
import threading
import pandas as pd

from broker.upstox.upstox_data_client import UpstoxDataClient


def _client():
    c = UpstoxDataClient.__new__(UpstoxDataClient)
    c._daily_cache = {}
    c._daily_cache_days = {}
    c._daily_cache_day = "2026-06-16"
    c._daily_cache_lock = threading.RLock()
    c._reset_daily_cache_if_new_day = lambda: None
    c._instrument_key_for = lambda s: "KEY"
    c._fetched = []

    def fake_fetch(ikey, days):
        c._fetched.append(days)
        # Symbol has deep history: returns exactly `days` bars.
        return pd.DataFrame({"open": range(days), "high": range(days),
                             "low": range(days), "close": range(days),
                             "volume": range(days)})
    c._fetch_daily_candles = fake_fetch
    return c


def test_larger_days_request_refetches_after_small_cache():
    c = _client()
    c.get_daily("NSE:NIFTY 50", days=2)        # prevday-style small fetch
    df = c.get_daily("NSE:NIFTY 50", days=210)  # regime needs 210
    assert c._fetched == [2, 210]               # refetched, did NOT serve 2-bar cache
    assert len(df) == 210


def test_smaller_or_equal_request_hits_cache():
    c = _client()
    c.get_daily("NSE:NIFTY 50", days=210)
    c.get_daily("NSE:NIFTY 50", days=2)         # served from the 210 cache
    assert c._fetched == [210]


def test_short_history_symbol_does_not_thrash():
    # Symbol with only 50 bars of history: fetched at 210 returns 50; a later 210
    # request must NOT refetch (we asked for 210 already).
    c = UpstoxDataClient.__new__(UpstoxDataClient)
    c._daily_cache = {}; c._daily_cache_days = {}; c._daily_cache_day = "2026-06-16"
    c._daily_cache_lock = threading.RLock()
    c._reset_daily_cache_if_new_day = lambda: None
    c._instrument_key_for = lambda s: "KEY"
    c._fetched = []
    def fake_fetch(ikey, days):
        c._fetched.append(days)
        n = min(days, 50)
        return pd.DataFrame({"open": range(n), "high": range(n), "low": range(n),
                             "close": range(n), "volume": range(n)})
    c._fetch_daily_candles = fake_fetch
    c.get_daily("X", days=210)
    c.get_daily("X", days=210)
    assert c._fetched == [210]   # second call hit cache despite only 50 bars
