"""DailyCachePersistence: a snapshot is current by CONTENT, not filename.

Bug (2026-10): snapshots are only written right after a morning API fetch, so
`daily_cache_<D>.pkl` holds bars up to D-1. The next day it was loaded as
"current" (filename == previous trading day) and that session ran without the
previous day's bar. Also rejects snapshots fetched shallower than requested.
"""
import pickle
from datetime import date

import pandas as pd

from services.state.daily_cache_persistence import DailyCachePersistence


def _bars(last, n=5):
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=n)
    return pd.DataFrame({c: [1.0] * n for c in ["open", "high", "low", "close", "volume"]},
                        index=idx)


def _write(dirp, name, cache, days=210, saved="2026-10-22"):
    with open(dirp / name, "wb") as f:
        pickle.dump({"date": saved, "cache": cache, "days": days}, f)


TODAY = date(2026, 10, 23)
PREV = date(2026, 10, 22)


def test_rejects_snapshot_missing_previous_day_bar(tmp_path):
    # Saved the morning of 10-22: last bar 10-21 -> stale on 10-23.
    _write(tmp_path, "daily_cache_2026-10-22.pkl", {"A": _bars("2026-10-21"), "B": _bars("2026-10-21")})
    p = DailyCachePersistence(cache_dir=tmp_path)
    assert p.load_current(today=TODAY, prev_trading_day=PREV, days=210) is None


def test_accepts_today_snapshot_ending_previous_day(tmp_path):
    _write(tmp_path, "daily_cache_2026-10-23.pkl",
           {"A": _bars("2026-10-22"), "B": _bars("2026-10-22")}, saved="2026-10-23")
    p = DailyCachePersistence(cache_dir=tmp_path)
    cache = p.load_current(today=TODAY, prev_trading_day=PREV, days=210)
    assert set(cache) == {"A", "B"}


def test_ignores_empty_frames_when_judging_freshness(tmp_path):
    # Empty frames (failed fetches) must not count either way; and are dropped.
    _write(tmp_path, "daily_cache_2026-10-23.pkl",
           {"A": _bars("2026-10-22"), "B": pd.DataFrame()}, saved="2026-10-23")
    p = DailyCachePersistence(cache_dir=tmp_path)
    cache = p.load_current(today=TODAY, prev_trading_day=PREV, days=210)
    assert set(cache) == {"A"}


def test_rejects_shallower_or_legacy_snapshot(tmp_path):
    _write(tmp_path, "daily_cache_2026-10-23.pkl", {"A": _bars("2026-10-22")}, days=30, saved="2026-10-23")
    p = DailyCachePersistence(cache_dir=tmp_path)
    assert p.load_current(today=TODAY, prev_trading_day=PREV, days=210) is None

    with open(tmp_path / "daily_cache_2026-10-23.pkl", "wb") as f:   # pre-fix format: no "days"
        pickle.dump({"date": "2026-10-23", "cache": {"A": _bars("2026-10-22")}}, f)
    assert p.load_current(today=TODAY, prev_trading_day=PREV, days=210) is None


def test_save_records_depth_and_round_trips(tmp_path):
    p = DailyCachePersistence(cache_dir=tmp_path)
    assert p.save({"A": _bars("2026-10-22")}, days=210, today=TODAY)
    cache = p.load_current(today=TODAY, prev_trading_day=PREV, days=210)
    assert set(cache) == {"A"}


def test_prewarm_lock_is_exclusive(tmp_path):
    p = DailyCachePersistence(cache_dir=tmp_path)
    with p.prewarm_lock(timeout_sec=5):
        other = DailyCachePersistence(cache_dir=tmp_path)
        assert other.try_prewarm_lock() is False
    other = DailyCachePersistence(cache_dir=tmp_path)
    assert other.try_prewarm_lock() is True
    other.release_prewarm_lock()
