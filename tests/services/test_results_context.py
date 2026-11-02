"""Results-context tag for up_spike signals (2026-11-02, logging only)."""
import pandas as pd
import pytest

from services import results_context as rc


@pytest.fixture
def cal(tmp_path):
    rows = [
        # AMC yesterday -> reaction today
        ("NSE:AMC", "2026-11-02 18:30", "AMC", "2026-11-03", None, "2026-11-02"),
        # BMO today
        ("NSE:BMO", "2026-11-03 08:10", "BMO", "2026-11-03", None, "2026-11-03"),
        # scheduled meeting today (live: nothing released yet)
        ("NSE:SCHED", "2026-11-03 09:00", "scheduled", "2026-11-03", "2026-11-03", "2026-11-03"),
        # released intraday today (history: the meeting row was replaced by the release)
        ("NSE:INTRA", "2026-11-03 13:05", "intraday", "2026-11-03", None, "2026-11-03"),
        # released after the close today: the meeting is today, the reaction tomorrow
        ("NSE:LATE", "2026-11-03 19:00", "AMC", "2026-11-04", None, "2026-11-03"),
        # last week
        ("NSE:OLD", "2026-10-27 13:00", "intraday", "2026-10-27", None, "2026-10-27"),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "announce_time", "announce_time_class", "trade_date",
                                     "bm_date", "announce_date"])
    df["announce_time"] = pd.to_datetime(df["announce_time"])
    df["announce_date"] = pd.to_datetime(df["announce_date"])
    p = tmp_path / "earnings_events.parquet"
    df.to_parquet(p)
    rc._tags_for.cache_clear()
    yield p
    rc._tags_for.cache_clear()


@pytest.mark.parametrize("sym, tag", [
    ("NSE:AMC", "results_preopen"), ("NSE:BMO", "results_preopen"),
    ("NSE:SCHED", "board_meeting_today"), ("NSE:INTRA", "board_meeting_today"),
    ("NSE:LATE", "board_meeting_today"), ("NSE:OLD", "none"), ("NSE:NOPE", "none"),
])
def test_tags_on_the_session(cal, sym, tag):
    assert rc.results_context(sym, "2026-11-03", path=cal) == tag


def test_next_day_the_late_release_is_preopen(cal):
    assert rc.results_context("NSE:LATE", "2026-11-04", path=cal) == "results_preopen"


def test_unreadable_calendar_tags_unknown(tmp_path):
    rc._tags_for.cache_clear()
    assert rc.results_context("NSE:X", "2026-11-03", path=tmp_path / "missing.parquet") == "unknown"


def test_up_spike_logs_the_tag_and_refreshes_the_calendar():
    import io, json
    from pathlib import Path
    cfg = json.load(io.open(Path(__file__).resolve().parents[2] / "config" / "configuration.json",
                            encoding="utf-8"))["setups"]["up_spike_fade_short"]
    assert cfg["log_results_context"] is True
    assert cfg["event_feed"]["refresh_on_start"] is True
    assert cfg["event_feed"]["path"] == "data/earnings_calendar/earnings_events.parquet"


def test_schema_drift_or_bad_session_tags_unknown_without_raising(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"symbol": ["NSE:X"]}).to_parquet(p)
    rc._tags_for.cache_clear()
    assert rc.results_context("NSE:X", "2026-11-03", path=p) == "unknown"
    assert rc.results_context("NSE:X", None, path=p) == "unknown"


def test_a_failed_read_is_not_cached(cal, tmp_path):
    missing = tmp_path / "later.parquet"
    assert rc.results_context("NSE:AMC", "2026-11-03", path=missing) == "unknown"
    missing.write_bytes(cal.read_bytes())
    assert rc.results_context("NSE:AMC", "2026-11-03", path=missing) == "results_preopen"
