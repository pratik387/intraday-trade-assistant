"""Circuit-risk sizing: NSE price bands -> distance to the circuit -> notional cap.

HUHTAMAKI 2026-10-28: a short locked at its +20% upper circuit could not be covered
by our stop nor by Zerodha's auto square-off and went to the short-delivery auction.
The loss on a locked position is the circuit distance plus the auction premium,
whatever the stop says, so up_spike_fade_short caps notional by it.
"""
import io
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from services.price_bands import (
    InconsistentReferenceClose, PriceBandDataError, PriceBandStore, SymbolNotInBandList)
from services.risk.intraday_sizing import SizingConfigError, size_intraday_position

ROOT = Path(__file__).resolve().parents[2]
CFG = json.load(io.open(ROOT / "config" / "configuration.json", encoding="utf-8"))


HOLIDAYS = frozenset({date(2026, 10, 20)})


def _store(tmp_path, rows, holidays=HOLIDAYS):
    df = pd.DataFrame(rows, columns=["file_date", "symbol", "series", "band_pct"])
    df["file_date"] = pd.to_datetime(df["file_date"])
    p = tmp_path / "bands.parquet"
    df.to_parquet(p, index=False)
    return PriceBandStore(str(p), holidays, ["EQ", "BE", "SM"], 1.0)


# --- which file a session reads ---------------------------------------------

def test_session_reads_the_file_dated_before_it(tmp_path):
    """sec_list_D carries the bands for the trading day AFTER D (VNL 10->5:
    sec_list_27102026 says 5, VNL still moved -10% on 10-27)."""
    s = _store(tmp_path, [("2026-10-27", "VNL", "EQ", 5.0), ("2026-10-26", "VNL", "EQ", 10.0),
                          ("2026-10-28", "VNL", "EQ", 2.0)])
    assert s.band_pct("NSE:VNL", date(2026, 10, 28)) == 5.0
    assert s.band_pct("NSE:VNL", date(2026, 10, 27)) == 10.0


def test_monday_reads_fridays_file(tmp_path):
    s = _store(tmp_path, [("2026-10-23", "X", "EQ", 10.0)])
    assert s.session(date(2026, 10, 26)).file_date == date(2026, 10, 23)


def test_stale_band_file_is_an_error_not_no_circuit(tmp_path):
    s = _store(tmp_path, [("2026-10-01", "X", "EQ", 10.0)])
    with pytest.raises(PriceBandDataError, match="previous trading"):
        s.session(date(2026, 10, 29))


def test_one_day_old_file_is_the_wrong_days_bands(tmp_path):
    """The evening file was not fetched: yesterday's-yesterday bands miss a change."""
    s = _store(tmp_path, [("2026-10-27", "X", "EQ", 10.0)])
    with pytest.raises(PriceBandDataError, match="2026-10-28"):
        s.session(date(2026, 10, 29))


def test_session_after_a_holiday_reads_the_last_trading_day(tmp_path):
    s = _store(tmp_path, [("2026-10-19", "X", "EQ", 10.0)])     # 10-20 holiday
    assert s.session(date(2026, 10, 21)).file_date == date(2026, 10, 19)


def test_holiday_calendar_loads(tmp_path):
    from services.price_bands import load_nse_holidays
    h = load_nse_holidays(CFG["price_bands"]["holidays_path"])
    assert date(2026, 10, 20) in h and date(2023, 1, 26) in h


def test_missing_parquet_raises(tmp_path):
    s = PriceBandStore(str(tmp_path / "nope.parquet"), HOLIDAYS, ["EQ"], 1.0)
    with pytest.raises(PriceBandDataError, match="missing"):
        s.session(date(2026, 10, 29))


def test_no_file_before_session_raises(tmp_path):
    s = _store(tmp_path, [("2026-10-29", "X", "EQ", 10.0)])
    with pytest.raises(PriceBandDataError, match="no price-band file"):
        s.session(date(2026, 10, 29))


def test_symbol_absent_from_the_list_raises(tmp_path):
    s = _store(tmp_path, [("2026-10-28", "X", "EQ", 10.0)])
    with pytest.raises(SymbolNotInBandList):
        s.band_pct("NSE:NEWLISTING", date(2026, 10, 29))


def test_no_band_means_no_fixed_circuit(tmp_path):
    s = _store(tmp_path, [("2026-10-28", "RELIANCE", "EQ", float("nan"))])
    assert s.band_pct("NSE:RELIANCE", date(2026, 10, 29)) is None
    assert s.circuit_distance_frac("NSE:RELIANCE", date(2026, 10, 29), 100.0, 105.0, "short") is None


def test_eq_series_preferred_over_others(tmp_path):
    s = _store(tmp_path, [("2026-10-28", "X", "BE", 5.0), ("2026-10-28", "X", "EQ", 20.0)])
    assert s.band_pct("X", date(2026, 10, 29)) == 20.0


# --- distance to the circuit that hurts -------------------------------------

def test_short_distance_is_to_the_upper_circuit(tmp_path):
    """HUHTAMAKI 10-28: prev close 191.31, band 20 -> UC 229.57; short at 193.32."""
    s = _store(tmp_path, [("2026-10-27", "HUHTAMAKI", "EQ", 20.0)])
    d = s.circuit_distance_frac("NSE:HUHTAMAKI", date(2026, 10, 28), 191.31, 193.32, "short")
    assert d == pytest.approx(229.572 / 193.32 - 1, rel=1e-4)


def test_long_distance_is_to_the_lower_circuit(tmp_path):
    s = _store(tmp_path, [("2026-10-28", "X", "EQ", 10.0)])
    d = s.circuit_distance_frac("X", date(2026, 10, 29), 100.0, 95.0, "long")
    assert d == pytest.approx(1 - 90.0 / 95.0)


def test_entry_already_at_the_circuit_is_zero_distance(tmp_path):
    s = _store(tmp_path, [("2026-10-28", "X", "EQ", 5.0)])
    assert s.circuit_distance_frac("X", date(2026, 10, 29), 100.0, 106.0, "short") == 0.0


@pytest.mark.parametrize("ref", [None, float("nan"), 0.0, -5.0])
def test_bad_reference_close_is_inconsistent(tmp_path, ref):
    """NaN used to slip through `not ref` and size silently at the tightest cap."""
    s = _store(tmp_path, [("2026-10-28", "X", "EQ", 5.0)])
    with pytest.raises(InconsistentReferenceClose):
        s.circuit_distance_frac("X", date(2026, 10, 29), ref, 100.0, "short")


def test_entry_outside_the_band_means_the_close_is_not_the_base(tmp_path):
    """Unadjusted ex-date (1:1 bonus): PDC 200 vs real base 100, entry 104 is
    'outside' a 20% band around 200 -> distance unknown, never a loose cap."""
    s = _store(tmp_path, [("2026-10-28", "X", "EQ", 20.0)])
    with pytest.raises(InconsistentReferenceClose):
        s.circuit_distance_frac("X", date(2026, 10, 29), 200.0, 104.0, "short")


def test_store_rereads_the_parquet_after_the_morning_refresh(tmp_path):
    import os, time
    s = _store(tmp_path, [("2026-10-27", "X", "EQ", 10.0)])
    with pytest.raises(PriceBandDataError):
        s.session(date(2026, 11, 5))                     # stale on day N+k
    df = pd.DataFrame([("2026-11-04", "X", "EQ", 5.0)],
                      columns=["file_date", "symbol", "series", "band_pct"])
    df["file_date"] = pd.to_datetime(df["file_date"])
    df.to_parquet(s.path, index=False)
    os.utime(s.path, (time.time() + 5, time.time() + 5))
    assert s.band_pct("X", date(2026, 11, 5)) == 5.0


# --- the sizing cap ---------------------------------------------------------

def _size(**kw):
    base = dict(setup_name="s", sizing_mode="notional", entry_price=100.0,
                target_notional_pct=0.06, total_capital_inr=100_000.0,
                book_size_multiplier=10.0, min_notional_inr=2_000.0,
                max_notional_inr=100_000.0)
    base.update(kw)
    return size_intraday_position(**base)


def test_cap_binds_when_the_circuit_is_close():
    # 60k flat; circuit 6% away + 10% premium -> 8000 / 0.16 = 50k
    r = _size(circuit_risk_budget_inr=8_000.0, circuit_distance_frac=0.06, circuit_premium_frac=0.10)
    assert r.clamped == "circuit"
    assert r.notional_inr == pytest.approx(50_000, abs=100)


def test_cap_does_not_bind_when_the_loss_fits_the_budget():
    # 60k x (0.02 + 0.10) = 7.2k <= 8k: flat size kept
    r = _size(circuit_risk_budget_inr=8_000.0, circuit_distance_frac=0.02, circuit_premium_frac=0.10)
    assert r.clamped is None and r.notional_inr == pytest.approx(60_000, abs=100)


def test_locked_tail_loss_never_exceeds_the_budget():
    for dist in (0.0, 0.03, 0.08, 0.16, 0.30):
        r = _size(circuit_risk_budget_inr=8_000.0, circuit_distance_frac=dist,
                  circuit_premium_frac=0.10)
        assert r.notional_inr * (dist + 0.10) <= 8_000.0 + 1e-6


def test_no_band_means_no_cap():
    r = _size(circuit_risk_budget_inr=8_000.0, circuit_distance_frac=None, circuit_premium_frac=0.10)
    assert r.clamped is None and r.notional_inr == pytest.approx(60_000, abs=100)


def test_cap_without_a_premium_is_a_config_error():
    with pytest.raises(SizingConfigError):
        _size(circuit_risk_budget_inr=8_000.0, circuit_distance_frac=0.05, circuit_premium_frac=None)


def test_cap_still_respects_the_floor():
    r = _size(circuit_risk_budget_inr=100.0, circuit_distance_frac=0.10, circuit_premium_frac=0.10)
    assert r.qty == 0 and r.reason == "below_min_notional"


def test_setups_without_a_budget_are_untouched():
    assert _size().notional_inr == pytest.approx(60_000, abs=100)


# --- real config ------------------------------------------------------------

def test_up_spike_live_budget_is_rs8k():
    s = CFG["setups"]["up_spike_fade_short"]
    live_capital = 100_000.0
    budget = (float(s["circuit_risk_budget_pct_of_capital"]) * live_capital
              * float(CFG["intraday_sizing"]["book_size_multiplier"]))
    assert budget == pytest.approx(8_000.0)
    assert float(CFG["intraday_sizing"]["circuit_risk_auction_premium_pct"]) == 10.0


def test_price_band_config_block_complete():
    pb = CFG["price_bands"]
    for k in ("parquet_path", "source_url_template", "http_user_agent", "http_timeout_sec",
              "http_pause_sec", "refresh_lookback_days", "save_every_files",
              "holidays_path", "series_preference", "consistency_tolerance_pct"):
        assert k in pb, f"price_bands.{k} missing"
    assert "{ddmmyyyy}" in pb["source_url_template"]


# --- startup check ----------------------------------------------------------

def test_startup_fails_without_band_data(monkeypatch, tmp_path):
    import services.plan_orchestrator as po
    import services.price_bands as pbm
    cfg = json.loads(json.dumps(CFG))
    cfg["price_bands"]["parquet_path"] = str(tmp_path / "missing.parquet")
    monkeypatch.setattr(po, "_load_root_config", lambda: cfg)
    pbm.reset_price_band_store()
    try:
        with pytest.raises(po.OrchestratorConfigError, match="price bands"):
            po.validate_price_bands(date(2026, 10, 29))
    finally:
        pbm.reset_price_band_store()


def test_startup_passes_with_band_data(monkeypatch, tmp_path):
    import services.plan_orchestrator as po
    import services.price_bands as pbm
    df = pd.DataFrame([("2026-10-28", "X", "EQ", 10.0)],
                      columns=["file_date", "symbol", "series", "band_pct"])
    df["file_date"] = pd.to_datetime(df["file_date"])
    p = tmp_path / "b.parquet"; df.to_parquet(p, index=False)
    cfg = json.loads(json.dumps(CFG))
    cfg["price_bands"]["parquet_path"] = str(p)
    monkeypatch.setattr(po, "_load_root_config", lambda: cfg)
    pbm.reset_price_band_store()
    try:
        assert "up_spike_fade_short" in po.validate_price_bands(date(2026, 10, 29))
    finally:
        pbm.reset_price_band_store()


def test_orchestrator_raises_on_missing_band_data_and_skips_estimated_pdc():
    """Missing session data must not size every symbol to 0 silently; a PDC made
    from today's early bars must not anchor the circuit."""
    import inspect
    import services.plan_orchestrator as po
    src = inspect.getsource(po)
    assert "circuit_pdc_estimated" in src
    i = src.index("except PriceBandDataError as e:", src.index("circuit_ref_inconsistent"))
    assert "raise OrchestratorConfigError" in src[i:i + 200]


def test_fallback_levels_mark_the_estimated_pdc():
    from utils.level_utils import _estimate_levels_from_fallback
    df = pd.DataFrame({"open": [100.0] * 12, "high": [101.0] * 12, "low": [99.0] * 12,
                       "close": [100.5] * 12, "volume": [1] * 12})
    assert _estimate_levels_from_fallback(df).get("PDC_estimated") == 1.0


def test_main_runs_the_startup_check():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "validate_price_bands(" in src


# --- fetcher parsing --------------------------------------------------------

class _Resp:
    def __init__(self, text, code=200):
        self.text, self.status_code = text, code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_fetcher_parses_bands_and_no_band(monkeypatch):
    import tools.fetch_nse_price_bands as f
    csv = 'Symbol,Series,Security Name,Band,Remarks\nA,EQ,A LTD,20,"-"\nB,EQ,B LTD,No Band,"-"\n'
    monkeypatch.setattr(f.requests, "get", lambda *a, **k: _Resp(csv))
    df = f._fetch(date(2026, 10, 28), CFG["price_bands"])
    assert df.set_index("symbol").band_pct["A"] == 20.0
    assert pd.isna(df.set_index("symbol").band_pct["B"])


def test_fetcher_treats_html_as_not_published(monkeypatch):
    import tools.fetch_nse_price_bands as f
    monkeypatch.setattr(f.requests, "get", lambda *a, **k: _Resp("<!DOCTYPE html><html>"))
    assert f._fetch(date(2026, 10, 29), CFG["price_bands"]) is None


def test_fetcher_rejects_unknown_band_values(monkeypatch):
    import tools.fetch_nse_price_bands as f
    csv = 'Symbol,Series,Security Name,Band,Remarks\nA,EQ,A LTD,abc,"-"\n'
    monkeypatch.setattr(f.requests, "get", lambda *a, **k: _Resp(csv))
    with pytest.raises(RuntimeError, match="unparseable"):
        f._fetch(date(2026, 10, 28), CFG["price_bands"])


def test_entry_at_the_circuit_is_not_a_trade_for_cover_setups():
    """KIRIINDUS 2023-03-02 backtest: shorted at 304.80 with the circuit at 304.81; the
    cover closed it on the same bar (fees-only loss)."""
    import inspect
    import services.plan_orchestrator as po
    src = inspect.getsource(po)
    i = src.index('_circ_skip = "circuit_at_entry"')
    assert 'trigger_offset_pct' in src[i - 400:i]


# --- point-in-time series filter (intraday universe) --------------------------

def test_trade_for_trade_series_is_not_intraday_tradeable(tmp_path):
    """PLATIND 2024-03-07 was BE (T2T): the old backtest shorted it intraday."""
    s = _store(tmp_path, [("2026-10-28", "PLATIND", "BE", 5.0), ("2026-10-28", "RELIANCE", "EQ", float("nan")),
                          ("2026-10-28", "SMEX", "SM", 5.0), ("2026-10-28", "BOTH", "BE", 5.0),
                          ("2026-10-28", "BOTH", "EQ", 5.0)])
    d = date(2026, 10, 29)
    assert s.tradeable("NSE:PLATIND", d, ["EQ"]) is False
    assert s.tradeable("NSE:SMEX", d, ["EQ"]) is False
    assert s.tradeable("NSE:RELIANCE", d, ["EQ"]) is True
    assert s.tradeable("NSE:BOTH", d, ["EQ"]) is True


def test_unlisted_name_is_not_judged_by_the_series_filter(tmp_path):
    """Historical lists carry old names of renamed stocks (ADOR was ADORWELD)."""
    s = _store(tmp_path, [("2026-10-28", "ADORWELD", "EQ", 20.0)])
    assert s.tradeable("NSE:ADOR", date(2026, 10, 29), ["EQ"]) is True


def test_screener_applies_the_series_filter_in_every_mode():
    src = (ROOT / "services" / "screener_live.py").read_text(encoding="utf-8")
    i = src.index("SERIES_FILTER")
    block = src[src.rindex("from services.price_bands import get_price_band_store", 0, i):i]
    assert "intraday_series" in block and "DRY_RUN" in block      # backtest uses its session date
    assert CFG["price_bands"]["intraday_series"] == ["EQ"]
