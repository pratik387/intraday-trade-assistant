"""Tests for broker.upstox.upstox_data_client.concurrent_get_daily.

Thread-pools an injected get_daily callable so the multi-day CNC/MTF panel
provider can warm the daily cache for the whole MTF universe concurrently
(instead of ~1300 serial calls = minutes). Reuses get_daily (caching, today-drop,
parsing, thread-safe rate limit), so this helper just fans it out.
"""
import threading

import pandas as pd

from broker.upstox.upstox_data_client import concurrent_get_daily


def test_concurrent_get_daily_calls_every_symbol_with_days():
    calls = []
    lock = threading.Lock()

    def fake_get_daily(sym, days):
        with lock:
            calls.append((sym, days))
        if sym == "NSE:BAD":
            return pd.DataFrame()  # empty -> counted as fail
        return pd.DataFrame({"open": [1.0], "close": [1.0]})

    n_ok = concurrent_get_daily(fake_get_daily, ["NSE:A", "NSE:B", "NSE:BAD"],
                                days=300, max_workers=4)
    assert n_ok == 2                                   # A, B succeeded; BAD empty
    assert sorted(calls) == [("NSE:A", 300), ("NSE:B", 300), ("NSE:BAD", 300)]


def test_concurrent_get_daily_tolerates_exceptions():
    def boom(sym, days):
        if sym == "NSE:ERR":
            raise RuntimeError("transient")
        return pd.DataFrame({"close": [1.0]})

    n_ok = concurrent_get_daily(boom, ["NSE:ERR", "NSE:OK"], days=100, max_workers=2)
    assert n_ok == 1  # ERR swallowed, OK counted


def test_concurrent_get_daily_empty_input():
    assert concurrent_get_daily(lambda s, days: pd.DataFrame({"c": [1]}), [], days=50) == 0
