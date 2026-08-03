"""Tests for services/event_feeds.py.

Regression cover for the 2026-08-03 silent failure: an ENABLED event-driven setup
whose feed was never deployed built an EMPTY universe every bar for a whole
session and raised nothing. These assert that both failure modes (missing feed,
stale feed) are detected and surfaced at CRITICAL.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from services.event_feeds import (
    FeedStatus,
    refresh_and_validate_all,
    validate_event_feed,
)

NOW = pd.Timestamp("2026-08-03 08:00:00")  # IST-naive, as the caller passes


def _write_feed(tmp_path: Path, newest: str, rows: int = 3) -> Path:
    p = tmp_path / "events.parquet"
    dates = pd.to_datetime([newest] * rows)
    pd.DataFrame({"announce_date": dates, "symbol": [f"S{i}" for i in range(rows)]}).to_parquet(p)
    return p


# --------------------------- validate_event_feed ---------------------------

def test_missing_feed_is_not_ok(tmp_path):
    st = validate_event_feed(
        tmp_path / "nope.parquet",
        date_column="announce_date", max_staleness_days=4, now=NOW,
    )
    assert not st.ok
    assert st.reason == "missing"
    # message must name the deployment cause — this is the whole point
    assert "gitignored" in st.message


def test_fresh_feed_is_ok(tmp_path):
    st = validate_event_feed(
        _write_feed(tmp_path, "2026-08-03"),
        date_column="announce_date", max_staleness_days=4, now=NOW,
    )
    assert st.ok and st.reason == "ok"
    assert st.age_days == 0 and st.rows == 3


def test_feed_at_the_staleness_limit_is_still_ok(tmp_path):
    """Boundary is inclusive: age == limit must pass, not fail."""
    st = validate_event_feed(
        _write_feed(tmp_path, "2026-07-30"),  # exactly 4 days
        date_column="announce_date", max_staleness_days=4, now=NOW,
    )
    assert st.ok, st.message
    assert st.age_days == 4


def test_feed_one_day_past_the_limit_is_stale(tmp_path):
    st = validate_event_feed(
        _write_feed(tmp_path, "2026-07-29"),  # 5 days
        date_column="announce_date", max_staleness_days=4, now=NOW,
    )
    assert not st.ok and st.reason == "stale"
    assert st.age_days == 5


def test_the_actual_2026_08_03_condition_is_caught(tmp_path):
    """The real incident: calendar frozen at 2026-07-28 while running on 08-03."""
    st = validate_event_feed(
        _write_feed(tmp_path, "2026-07-28"),
        date_column="announce_date", max_staleness_days=4, now=NOW,
    )
    assert not st.ok and st.reason == "stale"
    assert st.age_days == 6


def test_unreadable_feed_is_reported_not_raised(tmp_path):
    bad = tmp_path / "corrupt.parquet"
    bad.write_bytes(b"not a parquet file")
    st = validate_event_feed(
        bad, date_column="announce_date", max_staleness_days=4, now=NOW,
    )
    assert not st.ok and st.reason == "unreadable"


def test_missing_date_column_is_unreadable(tmp_path):
    p = tmp_path / "e.parquet"
    pd.DataFrame({"other": [1, 2]}).to_parquet(p)
    st = validate_event_feed(
        p, date_column="announce_date", max_staleness_days=4, now=NOW,
    )
    assert not st.ok and st.reason == "unreadable"


# ------------------------- refresh_and_validate_all -------------------------

class _Logger:
    def __init__(self):
        self.critical_msgs, self.info_msgs, self.error_msgs = [], [], []

    def critical(self, m): self.critical_msgs.append(m)
    def info(self, m): self.info_msgs.append(m)
    def error(self, m): self.error_msgs.append(m)


def _cfg(enabled: bool, feed: dict | None) -> dict:
    setup = {"enabled": enabled}
    if feed is not None:
        setup["event_feed"] = feed
    return {"setups": {"some_event_setup": setup}}


BASE_FEED = {
    "path": "events.parquet",
    "date_column": "announce_date",
    "refresh_on_start": False,
    "max_staleness_days": 4,
}


def test_enabled_setup_with_missing_feed_logs_critical(tmp_path):
    log = _Logger()
    refresh_and_validate_all(_cfg(True, BASE_FEED), repo_root=tmp_path, now=NOW, logger=log)
    assert len(log.critical_msgs) == 1
    assert "FEED MISSING" in log.critical_msgs[0]


def test_disabled_setup_is_not_checked(tmp_path):
    """A disabled setup cannot fire, so a missing feed is not a problem."""
    log = _Logger()
    refresh_and_validate_all(_cfg(False, BASE_FEED), repo_root=tmp_path, now=NOW, logger=log)
    assert log.critical_msgs == []


def test_setup_without_event_feed_is_skipped(tmp_path):
    log = _Logger()
    refresh_and_validate_all(_cfg(True, None), repo_root=tmp_path, now=NOW, logger=log)
    assert log.critical_msgs == []


def test_fresh_feed_logs_info_not_critical(tmp_path):
    _write_feed(tmp_path, "2026-08-02")
    feed = dict(BASE_FEED, path="events.parquet")
    log = _Logger()
    refresh_and_validate_all(_cfg(True, feed), repo_root=tmp_path, now=NOW, logger=log)
    assert log.critical_msgs == []
    assert any("newest=2026-08-02" in m for m in log.info_msgs)


def test_stale_feed_logs_critical(tmp_path):
    _write_feed(tmp_path, "2026-07-20")
    log = _Logger()
    refresh_and_validate_all(_cfg(True, dict(BASE_FEED)), repo_root=tmp_path, now=NOW, logger=log)
    assert len(log.critical_msgs) == 1
    assert "STALE" in log.critical_msgs[0]


def test_failed_refresh_does_not_prevent_validation(tmp_path, monkeypatch):
    """A refresh failure must degrade to 'possibly stale', never kill the daemon."""
    import services.event_feeds as ef
    monkeypatch.setattr(ef, "refresh_event_feed", lambda *a, **k: (False, "refresh FAILED rc=1"))
    _write_feed(tmp_path, "2026-08-03")
    feed = dict(BASE_FEED, refresh_on_start=True, refresh_lookback_days=7,
                refresh_module="x", refresh_sleep_secs=5, refresh_timeout_sec=60)
    log = _Logger()
    refresh_and_validate_all(_cfg(True, feed), repo_root=tmp_path, now=NOW, logger=log)
    assert any("refresh FAILED" in m for m in log.error_msgs)
    assert log.critical_msgs == []          # feed itself is fine, so no CRITICAL
