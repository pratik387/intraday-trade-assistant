"""NSE holiday calendar = cash-market (equity) trading holidays only (2026-11-02).

The refresh job merged every segment of NSE's holiday API, so currency/IRD bank
holidays sat in assets/nse_holidays.json as equity holidays: 10 real trading days in
2025-26 (on 2026-08-26 the paper daemon never started and the overnight entry
skipped the day).
"""
import datetime as dt
import json
import logging
from pathlib import Path

import pytest

from jobs import refresh_holidays as rh

ASSET = Path(__file__).resolve().parents[1] / "assets" / "nse_holidays.json"

# Weekdays the old file called holidays; equities traded on every one of them
# (1,400-2,500 symbols in the 5m archive). 2023-06-28: Bakri Id moved to 06-29.
TRADED = ["2025-01-01", "2025-02-19", "2025-04-01", "2025-05-12", "2025-09-05",
          "2026-01-01", "2026-02-19", "2026-03-19", "2026-04-01", "2026-08-26", "2023-06-28"]
# Closures declared after the annual circular (no session in the archive).
AD_HOC = ["2024-01-22", "2024-05-20", "2024-11-20", "2026-01-15", "2023-06-29"]


def _rows():
    return json.loads(ASSET.read_text(encoding="utf-8"))


def _dates():
    return [dt.datetime.strptime(r["tradingDate"], "%d-%b-%Y").date() for r in _rows()]


def test_asset_has_no_trading_day_marked_as_holiday():
    ds = set(_dates())
    assert not [d for d in TRADED if dt.date.fromisoformat(d) in ds]


def test_asset_has_the_ad_hoc_closures():
    ds = set(_dates())
    assert all(dt.date.fromisoformat(d) in ds for d in AD_HOC)


def test_asset_is_weekday_cm_holidays_for_every_year():
    ds = _dates()
    assert all(d.weekday() < 5 for d in ds)
    assert len(ds) == len(set(ds))
    per_year = {y: sum(1 for d in ds if d.year == y) for y in range(2022, 2027)}
    assert all(12 <= n <= 18 for n in per_year.values()), per_year


def test_weekday_names_match_the_dates():
    for r in _rows():
        d = dt.datetime.strptime(r["tradingDate"], "%d-%b-%Y").date()
        assert r["weekDay"] == d.strftime("%A"), r


# ---- consumers read the corrected calendar ----

def test_utils_is_trading_day():
    from utils.util import is_trading_day
    assert all(is_trading_day(d) for d in TRADED)
    assert not is_trading_day("2026-10-20")          # Dussehra
    assert not is_trading_day("2023-06-29")


def test_overnight_trading_day_math(monkeypatch):
    from services.execution import overnight_handlers as oh
    monkeypatch.setattr(oh, "_NSE_HOLIDAYS", None)
    assert oh._is_trading_day(dt.date(2026, 8, 26))
    assert oh._next_trading_day(dt.date(2026, 8, 25)) == dt.date(2026, 8, 26)
    assert oh._next_trading_day(dt.date(2026, 11, 9)) == dt.date(2026, 11, 11)   # 11-10 Balipratipada


def test_price_band_calendar_excludes_the_traded_days():
    from services.price_bands import load_nse_holidays
    hols = load_nse_holidays(str(ASSET))
    assert not [d for d in TRADED if dt.date.fromisoformat(d) in hols]


def test_dates_past_the_calendar_log_an_error_once(monkeypatch, caplog):
    from services.execution import overnight_handlers as oh
    monkeypatch.setattr(oh, "_NSE_HOLIDAYS", None)
    monkeypatch.setattr(oh, "_UNCOVERED_YEARS_LOGGED", set())
    with caplog.at_level(logging.ERROR):
        assert oh._is_trading_day(dt.date(2027, 1, 26))      # not yet listed -> counted as trading
        oh._is_trading_day(dt.date(2027, 1, 27))
    errs = [r for r in caplog.records if "no 2027 entries" in r.getMessage()]
    assert len(errs) == 1


# ---- refresh job: only the CM segment, validated, never destructive ----

SEGMENTS = {
    "CM": [{"tradingDate": "26-Jan-2027", "weekDay": "Tuesday", "description": "Republic Day"}],
    "CD": [{"tradingDate": "01-Jan-2027", "weekDay": "Friday", "description": "New year"},
           {"tradingDate": "26-Jan-2027", "weekDay": "Tuesday", "description": "Republic Day"}],
    "FO": [{"tradingDate": "26-Jan-2027", "weekDay": "Tuesday", "description": "Republic Day"}],
}


def test_only_the_cash_market_segment_is_taken():
    assert [h["tradingDate"] for h in rh.equity_holidays(SEGMENTS)] == ["26-Jan-2027"]


def test_weekend_and_duplicate_rows_are_dropped():
    cm = {"CM": [{"tradingDate": "26-Jan-2027"}, {"tradingDate": "26-Jan-2027"},
                 {"tradingDate": "14-Mar-2027"}]}                     # Sunday
    assert [h["tradingDate"] for h in rh.equity_holidays(cm)] == ["26-Jan-2027"]


@pytest.mark.parametrize("payload", [{"content": SEGMENTS["CM"]}, {"CD": SEGMENTS["CD"]}, []])
def test_a_response_without_the_cm_segment_is_rejected(payload):
    with pytest.raises(ValueError, match="CM"):
        rh.equity_holidays(payload)


def test_a_malformed_date_is_rejected():
    with pytest.raises(ValueError):
        rh.equity_holidays({"CM": [{"tradingDate": "2027-01-26"}]})


def _fake_api(monkeypatch, payload):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    class Session:
        def __init__(self):
            self.headers = {}

        def get(self, url, timeout):
            return Resp()

    monkeypatch.setattr(rh.requests, "Session", Session)


def _target(tmp_path, monkeypatch, text):
    target = tmp_path / "nse_holidays.json"
    target.write_text(text, encoding="utf-8")
    monkeypatch.setattr(rh, "HOLIDAY_FILE", target)
    return target


def test_download_merges_only_cm_entries(tmp_path, monkeypatch):
    target = _target(tmp_path, monkeypatch, json.dumps([{"tradingDate": "25-Dec-2026", "description": "Christmas"}]))
    _fake_api(monkeypatch, SEGMENTS)
    assert rh.download_nse_holidays()["status"] == "success"
    assert [h["tradingDate"] for h in json.loads(target.read_text(encoding="utf-8"))] == ["25-Dec-2026", "26-Jan-2027"]


@pytest.mark.parametrize("payload", [{"CD": SEGMENTS["CD"]}, {"CM": [{"tradingDate": "bad"}]}])
def test_a_failed_refresh_leaves_the_file_unchanged(tmp_path, monkeypatch, payload):
    before = json.dumps([{"tradingDate": "25-Dec-2026", "description": "Christmas"}])
    target = _target(tmp_path, monkeypatch, before)
    _fake_api(monkeypatch, payload)
    assert rh.download_nse_holidays()["status"] == "failed"
    assert target.read_text(encoding="utf-8") == before


def test_an_unreadable_file_is_not_overwritten(tmp_path, monkeypatch):
    target = _target(tmp_path, monkeypatch, "{ not json")
    _fake_api(monkeypatch, SEGMENTS)
    assert rh.download_nse_holidays()["status"] == "failed"
    assert target.read_text(encoding="utf-8") == "{ not json"
