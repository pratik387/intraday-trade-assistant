"""The scanned bar's label reaches detectors as MarketContext.scan_bar_ts (2026-11-03)."""
import pandas as pd

from services.dispatch.worker import _build_market_context


def _df():
    idx = pd.date_range("2026-11-03 09:15", periods=12, freq="5min")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=idx)


def test_scan_bar_ts_is_carried():
    ctx = _build_market_context("NSE:X", _df(), {}, bar_ts=pd.Timestamp("2026-11-03 10:05"),
                                session_date=pd.Timestamp("2026-11-03").date())
    assert ctx.scan_bar_ts == pd.Timestamp("2026-11-03 10:05")
    assert ctx.timestamp == pd.Timestamp("2026-11-03 10:10")       # last row, unchanged behaviour


def test_scan_bar_ts_is_none_without_bar_ts():
    assert _build_market_context("NSE:X", _df(), {}).scan_bar_ts is None


def test_scan_bar_ts_is_ist_naive():
    ts = pd.Timestamp("2026-11-03 10:05", tz="Asia/Kolkata")
    ctx = _build_market_context("NSE:X", _df(), {}, bar_ts=ts)
    assert ctx.scan_bar_ts.tzinfo is None and ctx.scan_bar_ts == pd.Timestamp("2026-11-03 10:05")
