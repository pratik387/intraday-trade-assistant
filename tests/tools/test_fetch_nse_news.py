"""Tests for tools.news_feed.fetch_nse_news (NSE announcements + event calendar).

No network: the module's ``requests.Session`` is swapped for a scripted fake and
``time.sleep`` is a no-op. Covers the refresh contract used by
services/event_feeds.py (``--start --end --sleep-secs``), per-day chunking,
cookie re-bootstrap on 403 / empty body, IST-naive an_dt parsing, merge that
preserves rows outside the fetched window, and event-calendar range replacement.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from tools.news_feed import fetch_nse_news as mod


# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        if body is None:
            self.content = b""
        elif isinstance(body, (bytes, bytearray)):
            self.content = bytes(body)
        elif isinstance(body, str):
            self.content = body.encode("utf-8")
        else:
            self.content = json.dumps(body).encode("utf-8")
        self.headers = headers or {}

    def json(self):
        return json.loads(self.content.decode("utf-8"))


class FakeHTTP:
    """Scripted requests.Session stand-in.

    `script` maps an URL to a list of FakeResponse returned in order (the last
    one repeats). Every call is recorded in `calls` as (url, params).
    """

    def __init__(self, script: dict):
        self.script = {k: list(v) for k, v in script.items()}
        self.headers = {}
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params) if params else None))
        queue = self.script.get(url)
        if not queue:
            return FakeResponse(404, "")
        return queue.pop(0) if len(queue) > 1 else queue[0]


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)


def _install(monkeypatch, script: dict) -> FakeHTTP:
    fake = FakeHTTP(script)
    monkeypatch.setattr(mod.requests, "Session", lambda: fake)
    return fake


def _ann(symbol, an_dt, desc="Outcome of Board Meeting", text="x", name="Co Ltd"):
    return {
        "symbol": symbol, "an_dt": an_dt, "desc": desc, "attchmntText": text,
        "sm_name": name, "sort_date": "2026-10-06 20:04:49",
    }


def _cal(symbol, day, purpose="Financial Results", bm_desc="results", company="Co"):
    return {"symbol": symbol, "date": day, "purpose": purpose, "bm_desc": bm_desc,
            "company": company}


# ---------------------------------------------------------------------------
# Parsing.
# ---------------------------------------------------------------------------

def test_parse_an_dt_is_ist_naive_timestamp():
    ts = mod.parse_an_dt("06-Oct-2026 16:16:44")
    assert isinstance(ts, pd.Timestamp)
    assert ts.tzinfo is None
    assert ts == pd.Timestamp("2026-10-06 16:16:44")
    assert mod.parse_an_dt("garbage") is None
    assert mod.parse_an_dt(None) is None


def test_parse_announcements_schema_and_text_truncation():
    rows = mod.parse_announcements([
        _ann("KARAMTARA", "06-Oct-2026 20:04:49", text="A" * 900),
        {"symbol": "", "an_dt": "06-Oct-2026 20:04:49"},          # no symbol -> drop
        {"symbol": "X", "an_dt": None, "sort_date": None},         # no time -> drop
        {"symbol": "y", "an_dt": "bad", "sort_date": "2026-10-06 09:01:02"},  # fallback
    ])
    assert [r["symbol"] for r in rows] == ["KARAMTARA", "Y"]
    r = rows[0]
    assert set(r) == set(mod.ANN_COLUMNS)
    assert r["an_dt"] == pd.Timestamp("2026-10-06 20:04:49") and r["an_dt"].tzinfo is None
    assert r["filing_date"] == pd.Timestamp("2026-10-06")
    assert len(r["text"]) == 500
    assert r["source"] == "nse_announcements"
    assert rows[1]["an_dt"] == pd.Timestamp("2026-10-06 09:01:02")


def test_parse_event_calendar_schema():
    fa = pd.Timestamp("2026-10-06 18:00:00")
    rows = mod.parse_event_calendar(
        [_cal("ARCIL", "06-Oct-2026"), _cal("NOPE", "-"), _cal("", "06-Oct-2026")],
        fetched_at=fa,
    )
    assert len(rows) == 1
    r = rows[0]
    assert set(r) == set(mod.CAL_COLUMNS)
    assert r["meeting_date"] == pd.Timestamp("2026-10-06") and r["meeting_date"].tzinfo is None
    assert r["purpose"] == "Financial Results"
    assert r["fetched_at"] == fa


# ---------------------------------------------------------------------------
# Chunking + HTTP behaviour.
# ---------------------------------------------------------------------------

def test_day_chunks_inclusive():
    assert mod.day_chunks(date(2026, 10, 5), date(2026, 10, 6)) == [
        date(2026, 10, 5), date(2026, 10, 6),
    ]
    assert mod.day_chunks(date(2026, 10, 5), date(2026, 10, 5)) == [date(2026, 10, 5)]
    with pytest.raises(ValueError):
        mod.day_chunks(date(2026, 10, 6), date(2026, 10, 5))


def test_announcements_one_call_per_day_with_same_from_to(monkeypatch, no_sleep):
    fake = _install(monkeypatch, {
        mod._NSE_HOME: [FakeResponse(200, "<html>")],
        mod._NSE_ANN_API: [FakeResponse(200, [_ann("A", "05-Oct-2026 10:00:00")])],
    })
    sess = mod.NSENewsSession(sleep_secs=0)
    rows, stats = mod.fetch_announcements_range(
        sess, date(2026, 10, 4), date(2026, 10, 6), sleep_secs=0,
    )
    ann_calls = [p for u, p in fake.calls if u == mod._NSE_ANN_API]
    assert len(ann_calls) == 3
    assert [(p["from_date"], p["to_date"]) for p in ann_calls] == [
        ("04-10-2026", "04-10-2026"), ("05-10-2026", "05-10-2026"), ("06-10-2026", "06-10-2026"),
    ]
    assert all(p["index"] == "equities" and "symbol" not in p for p in ann_calls)
    assert stats["chunks_ok"] == 3 and stats["chunks_failed"] == 0
    assert len(rows) == 3


def test_rebootstrap_on_403_then_success(monkeypatch, no_sleep):
    fake = _install(monkeypatch, {
        mod._NSE_HOME: [FakeResponse(200, "<html>")],
        mod._NSE_ANN_API: [
            FakeResponse(403, ""),
            FakeResponse(200, [_ann("A", "05-Oct-2026 10:00:00")]),
        ],
    })
    sess = mod.NSENewsSession(sleep_secs=0)
    assert sess.bootstrap_count == 1
    payload = sess.get_json(mod._NSE_ANN_API, {"index": "equities"})
    assert isinstance(payload, list) and payload[0]["symbol"] == "A"
    assert sess.bootstrap_count == 2
    home_calls = [u for u, _ in fake.calls if u == mod._NSE_HOME]
    assert len(home_calls) == 2


def test_rebootstrap_on_empty_body_and_non_json(monkeypatch, no_sleep):
    _install(monkeypatch, {
        mod._NSE_HOME: [FakeResponse(200, "<html>")],
        mod._NSE_ANN_API: [
            FakeResponse(200, ""),            # empty body
            FakeResponse(200, "<html>chal"),  # Akamai-style HTML 200
            FakeResponse(200, [_ann("A", "05-Oct-2026 10:00:00")]),
        ],
    })
    sess = mod.NSENewsSession(sleep_secs=0)
    payload = sess.get_json(mod._NSE_ANN_API, {})
    assert isinstance(payload, list)
    assert sess.bootstrap_count == 3


def test_exhausted_retries_returns_none_and_chunk_is_skipped_not_raised(monkeypatch, no_sleep):
    _install(monkeypatch, {
        mod._NSE_HOME: [FakeResponse(200, "<html>")],
        mod._NSE_ANN_API: [FakeResponse(403, "")],
    })
    sess = mod.NSENewsSession(sleep_secs=0, max_retries=2)
    rows, stats = mod.fetch_announcements_range(
        sess, date(2026, 10, 5), date(2026, 10, 5), sleep_secs=0,
    )
    assert rows == []
    assert stats["chunks_failed"] == 1 and stats["chunks_ok"] == 0


# ---------------------------------------------------------------------------
# Persistence semantics.
# ---------------------------------------------------------------------------

def test_merge_announcements_dedupes_and_keeps_rows_outside_window(tmp_path):
    out = tmp_path / "nse_announcements.parquet"
    old_rows = mod.parse_announcements([
        _ann("OLD", "01-Sep-2026 10:00:00", desc="Press Release"),
        _ann("DUP", "05-Oct-2026 10:00:00"),
    ])
    mod.merge_announcements(old_rows, out)

    new_rows = mod.parse_announcements([
        _ann("DUP", "05-Oct-2026 10:00:00", text="same key, newer text"),  # dup key
        _ann("NEW", "06-Oct-2026 11:00:00"),
    ])
    df = mod.merge_announcements(new_rows, out)
    assert list(df.columns) == mod.ANN_COLUMNS
    assert sorted(df["symbol"]) == ["DUP", "NEW", "OLD"]       # OLD preserved, DUP once
    assert df.loc[df["symbol"] == "DUP", "text"].iloc[0] == "same key, newer text"
    assert str(df["an_dt"].dtype) == "datetime64[ns]"
    assert df["an_dt"].dt.tz is None

    back = pd.read_parquet(out)
    assert len(back) == 3
    assert back["filing_date"].min() == pd.Timestamp("2026-09-01")


def test_merge_announcements_empty_fetch_keeps_existing(tmp_path):
    out = tmp_path / "a.parquet"
    mod.merge_announcements(mod.parse_announcements([_ann("A", "01-Sep-2026 10:00:00")]), out)
    df = mod.merge_announcements([], out)
    assert len(df) == 1


def test_event_calendar_range_replacement(tmp_path):
    out = tmp_path / "cal.parquet"
    fa0 = pd.Timestamp("2026-10-01 18:00:00")
    stored = mod.parse_event_calendar([
        _cal("KEEP_PAST", "01-Oct-2026"),                     # outside range -> kept
        _cal("RESCHED", "08-Oct-2026"),                       # in range, moves to the 12th
        _cal("CANCELLED", "09-Oct-2026"),                     # in range, gone from fresh fetch
        _cal("KEEP_FUTURE", "25-Oct-2026"),                   # outside range -> kept
    ], fetched_at=fa0)
    mod.merge_event_calendar(stored, out, replace_windows=[(date(2026, 10, 1), date(2026, 10, 31))])

    fa1 = pd.Timestamp("2026-10-06 18:00:00")
    fresh = mod.parse_event_calendar([
        _cal("RESCHED", "12-Oct-2026"),
        _cal("NEW", "10-Oct-2026"),
        _cal("NEW", "10-Oct-2026"),                           # duplicate within the fetch
    ], fetched_at=fa1)
    df = mod.merge_event_calendar(
        fresh, out, replace_windows=[(date(2026, 10, 6), date(2026, 10, 16))],
    )
    got = {(r.symbol, r.meeting_date.date()) for r in df.itertuples()}
    assert got == {
        ("KEEP_PAST", date(2026, 10, 1)),
        ("KEEP_FUTURE", date(2026, 10, 25)),
        ("RESCHED", date(2026, 10, 12)),
        ("NEW", date(2026, 10, 10)),
    }
    assert df["meeting_date"].dt.tz is None
    assert df.loc[df["symbol"] == "NEW", "fetched_at"].iloc[0] == fa1
    assert df.loc[df["symbol"] == "KEEP_PAST", "fetched_at"].iloc[0] == fa0


def test_event_calendar_failed_window_does_not_wipe_stored_rows(tmp_path):
    out = tmp_path / "cal.parquet"
    stored = mod.parse_event_calendar([_cal("A", "08-Oct-2026"), _cal("B", "15-Oct-2026")],
                                      fetched_at=pd.Timestamp("2026-10-01"))
    mod.merge_event_calendar(stored, out, replace_windows=[(date(2026, 10, 1), date(2026, 10, 31))])
    # Second window (13..19) failed: only the first window is replaced.
    df = mod.merge_event_calendar(
        [], out, replace_windows=[(date(2026, 10, 6), date(2026, 10, 12))],
    )
    assert df["symbol"].tolist() == ["B"]


# ---------------------------------------------------------------------------
# CLI contract (services/event_feeds.py passes --start --end --sleep-secs).
# ---------------------------------------------------------------------------

def test_main_end_to_end_with_mock_http(monkeypatch, no_sleep, tmp_path):
    cal_rows = [_cal("ARCIL", "07-Oct-2026"), _cal("BBB", "10-Oct-2026")]
    fake = _install(monkeypatch, {
        mod._NSE_HOME: [FakeResponse(200, "<html>")],
        mod._NSE_ANN_API: [FakeResponse(200, [_ann("A", "05-Oct-2026 10:00:00"),
                                             _ann("B", "06-Oct-2026 16:16:44")])],
        mod._NSE_CAL_API: [FakeResponse(200, cal_rows)],
    })
    monkeypatch.setattr(mod, "_now_naive_ist", lambda: pd.Timestamp("2026-10-06 18:00:00"))
    rc = mod.main([
        "--start", "2026-10-05", "--end", "2026-10-06", "--sleep-secs", "0",
        "--forward-days", "10", "--out-dir", str(tmp_path),
    ])
    assert rc == 0
    ann = pd.read_parquet(tmp_path / "nse_announcements.parquet")
    cal = pd.read_parquet(tmp_path / "nse_event_calendar.parquet")
    # 2 days x 2 rows each from the repeating script, deduped on (symbol, an_dt, desc)
    assert len(ann) == 2
    assert sorted(ann["symbol"]) == ["A", "B"]
    assert len(cal) == 2
    cal_calls = [p for u, p in fake.calls if u == mod._NSE_CAL_API]
    # [end, end+10] = Oct 6..16 -> two 7-day windows.
    assert [(p["from_date"], p["to_date"]) for p in cal_calls] == [
        ("06-10-2026", "12-10-2026"), ("13-10-2026", "16-10-2026"),
    ]


def test_main_requires_forward_days_and_sleep_secs(tmp_path):
    with pytest.raises(SystemExit):
        mod.main(["--start", "2026-10-05", "--end", "2026-10-06", "--sleep-secs", "0",
                  "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        mod.main(["--start", "2026-10-05", "--end", "2026-10-06", "--forward-days", "3",
                  "--out-dir", str(tmp_path)])


def test_main_total_failure_exits_nonzero_but_single_bad_chunk_does_not(monkeypatch, no_sleep, tmp_path):
    # Everything 403 -> both streams total failure -> rc 4.
    _install(monkeypatch, {
        mod._NSE_HOME: [FakeResponse(200, "<html>")],
        mod._NSE_ANN_API: [FakeResponse(403, "")],
        mod._NSE_CAL_API: [FakeResponse(403, "")],
    })
    monkeypatch.setattr(mod.NSENewsSession, "max_retries", 1, raising=False)
    rc = mod.main(["--start", "2026-10-05", "--end", "2026-10-06", "--sleep-secs", "0",
                   "--forward-days", "2", "--out-dir", str(tmp_path)])
    assert rc == 4

    # One of two announcement days fails, calendar fine -> rc 0, partial data written.
    _install(monkeypatch, {
        mod._NSE_HOME: [FakeResponse(200, "<html>")],
        mod._NSE_ANN_API: [FakeResponse(403, ""), FakeResponse(403, ""), FakeResponse(403, ""),
                           FakeResponse(403, ""), FakeResponse(403, ""),
                           FakeResponse(200, [_ann("OK", "06-Oct-2026 10:00:00")])],
        mod._NSE_CAL_API: [FakeResponse(200, [_cal("C", "07-Oct-2026")])],
    })
    rc = mod.main(["--start", "2026-10-05", "--end", "2026-10-06", "--sleep-secs", "0",
                   "--forward-days", "2", "--out-dir", str(tmp_path)])
    assert rc == 0
    ann = pd.read_parquet(tmp_path / "nse_announcements.parquet")
    assert ann["symbol"].tolist() == ["OK"]
