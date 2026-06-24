"""Tests for tools/overnight_paper_slippage.py.

All data is MOCKED — no network, no Upstox creds. The SDK fetch is injected
via the `fetch_fn` parameter so these unit tests never touch aiohttp.

The tool reconstructs the idealized-Rs1L paper-equivalent for every fired
signal on a given entry date and computes real-vs-idealized slippage for the
signals that also traded live.
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.overnight_paper_slippage import (
    PAPER_MARGIN_INR,
    parse_fired_signals,
    reconstruct_paper_trade,
    extract_idealized_prices,
    compute_slippage,
    reconstruct_for_date,
)
from tools.sub7_validation.build_per_setup_pnl import calc_fee_cnc, calc_fee_mtf


SAMPLE_LOG = """\
2026-06-16 15:27:03,112 INFO - run_entry: universe size for close_dn_overnight_long = 41
2026-06-16 15:27:05,001 INFO - close_dn_overnight_long fired | symbol=NSE:PACEDIGITK svr=-1.000 vol_z=24.95 prior_ret=11.23% product=MTF lev=3.34
2026-06-16 15:27:05,442 INFO - close_dn_overnight_long fired | symbol=NSE:SAMPANN svr=-1.000 vol_z=8.10 prior_ret=4.55% product=CNC lev=1.00
2026-06-16 15:27:06,900 INFO - run_entry: complete | fired=2 skipped=0 rejected=0
"""


def test_parse_fired_signals():
    sigs = parse_fired_signals(SAMPLE_LOG)
    assert len(sigs) == 2
    assert sigs[0] == ("NSE:PACEDIGITK", "MTF", 3.34)
    assert sigs[1] == ("NSE:SAMPANN", "CNC", 1.00)


def test_reconstruct_cnc_1L():
    # CNC, lev forced to 1.0 (CNC has no leverage). entry 100, exit 105.
    entry, exit_ = 100.0, 105.0
    lev = 1.0
    rec = reconstruct_paper_trade(
        symbol="NSE:SAMPANN", product="CNC", lev=lev,
        idealized_entry=entry, idealized_exit=exit_, hold_days=1,
    )
    notional = PAPER_MARGIN_INR * lev
    qty = int(notional / entry)            # 100000/100 = 1000
    assert rec["qty"] == qty == 1000
    gross = (exit_ - entry) * qty          # 5 * 1000 = 5000
    assert rec["gross_pnl_inr"] == gross
    fees = calc_fee_cnc(entry * qty, exit_ * qty)
    assert rec["fees_inr"] == fees
    assert rec["net_pnl_inr"] == gross - fees
    assert rec["product"] == "CNC"
    assert rec["source"] == "reconstructed"
    assert rec["exit_reason"] == "reconstructed_paper"


def test_reconstruct_mtf_1L():
    entry, exit_ = 50.0, 48.0   # a loser, MTF leveraged
    lev = 3.34
    rec = reconstruct_paper_trade(
        symbol="NSE:PACEDIGITK", product="MTF", lev=lev,
        idealized_entry=entry, idealized_exit=exit_, hold_days=1,
    )
    notional = PAPER_MARGIN_INR * lev      # 334000
    qty = int(notional / entry)            # 334000/50 = 6680
    assert rec["qty"] == qty == 6680
    gross = (exit_ - entry) * qty          # -2 * 6680 = -13360
    assert rec["gross_pnl_inr"] == gross
    fees = calc_fee_mtf(entry * qty, exit_ * qty, PAPER_MARGIN_INR, 1)
    assert rec["fees_inr"] == fees
    assert rec["net_pnl_inr"] == gross - fees
    assert rec["leverage"] == lev
    assert rec["product"] == "MTF"


def _mock_df_two_days():
    # Entry day 2026-06-16 has a 15:25 bar; exit day 2026-06-17 has 09:15.
    idx = pd.to_datetime([
        "2026-06-16 15:15", "2026-06-16 15:20", "2026-06-16 15:25",
        "2026-06-17 09:15", "2026-06-17 09:20",
    ])
    return pd.DataFrame(
        {
            "open":   [101.0, 102.0, 103.0, 110.0, 111.0],
            "high":   [101.5, 102.5, 103.5, 110.5, 111.5],
            "low":    [100.5, 101.5, 102.5, 109.5, 110.5],
            "close":  [101.2, 102.2, 103.2, 110.2, 111.2],
            "volume": [10, 20, 30, 40, 50],
        },
        index=idx,
    )


def test_idealized_price_extraction():
    df = _mock_df_two_days()
    entry_dt = date(2026, 6, 16)
    exit_dt = date(2026, 6, 17)
    ie, ix = extract_idealized_prices(df, entry_dt, exit_dt)
    # entry = 15:25 bar CLOSE on entry day
    assert ie == 103.2
    # exit = 09:15 bar OPEN on exit day
    assert ix == 110.0


def test_idealized_price_extraction_fallback():
    # No exact 15:25 / 09:15 bar -> use last<=15:25 entry / first>=09:15 exit.
    idx = pd.to_datetime([
        "2026-06-16 15:10", "2026-06-16 15:20",   # last <= 15:25 is 15:20
        "2026-06-17 09:20", "2026-06-17 09:25",   # first >= 09:15 is 09:20
    ])
    df = pd.DataFrame(
        {
            "open":   [1.0, 2.0, 9.0, 9.5],
            "high":   [1.0, 2.0, 9.0, 9.5],
            "low":    [1.0, 2.0, 9.0, 9.5],
            "close":  [1.1, 2.2, 9.1, 9.6],
            "volume": [1, 2, 3, 4],
        },
        index=idx,
    )
    ie, ix = extract_idealized_prices(df, date(2026, 6, 16), date(2026, 6, 17))
    assert ie == 2.2    # close of 15:20 (last <= 15:25)
    assert ix == 9.0    # open of 09:20 (first >= 09:15)


def test_slippage_calc():
    # idealized entry 100 (paper buys at close), real entry 100.5 (paid more =
    # adverse, positive). idealized exit 110 (paper sells at open), real exit
    # 109.4 (sold for less = adverse, positive).
    slip = compute_slippage(
        idealized_entry=100.0, real_entry=100.5,
        idealized_exit=110.0, real_exit=109.4,
    )
    assert round(slip["entry_slip"], 6) == 0.5
    assert round(slip["exit_slip"], 6) == 0.6
    assert round(slip["entry_slip_bps"], 4) == round(0.5 / 100.0 * 1e4, 4)   # 50 bps
    assert round(slip["exit_slip_bps"], 4) == round(0.6 / 110.0 * 1e4, 4)
    # A favorable fill (real entry below idealized) yields a negative slip.
    fav = compute_slippage(
        idealized_entry=100.0, real_entry=99.0,
        idealized_exit=110.0, real_exit=110.0,
    )
    assert fav["entry_slip"] == -1.0
    assert fav["exit_slip"] == 0.0


def _write_log(tmp_path: Path, content: str) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / "overnight_entry_2026-06-16.log"
    p.write_text(content, encoding="utf-8")
    return p


def _fake_fetch_factory():
    def fake_fetch(symbols, from_date_iso, to_date_iso):
        out = {}
        for s in symbols:
            out[s] = _mock_df_two_days()
        return out
    return fake_fetch


def _mock_df_entry_day():
    """Entry-day (2026-06-16) bars only — what the historical endpoint returns
    for the always-past entry date."""
    idx = pd.to_datetime([
        "2026-06-16 15:15", "2026-06-16 15:20", "2026-06-16 15:25",
    ])
    return pd.DataFrame(
        {
            "open":   [101.0, 102.0, 103.0],
            "high":   [101.5, 102.5, 103.5],
            "low":    [100.5, 101.5, 102.5],
            "close":  [101.2, 102.2, 103.2],
            "volume": [10, 20, 30],
        },
        index=idx,
    )


def _mock_df_exit_day():
    """Exit-day (2026-06-17) bars only — what the intraday endpoint returns when
    the exit date is today."""
    idx = pd.to_datetime([
        "2026-06-17 09:15", "2026-06-17 09:20",
    ])
    return pd.DataFrame(
        {
            "open":   [110.0, 111.0],
            "high":   [110.5, 111.5],
            "low":    [109.5, 110.5],
            "close":  [110.2, 111.2],
            "volume": [40, 50],
        },
        index=idx,
    )


def test_idempotent_append(tmp_path, monkeypatch):
    # Seed a paper ledger with one NON-reconstructed (real paper-run) trade
    # whose ts_iso date == the exit date, to prove we never remove it.
    paper_path = tmp_path / "state" / "decay_tripwire_close_dn_overnight_long.json"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    seed = {
        "setup_name": "close_dn_overnight_long",
        "trades": [
            {"net_pnl_inr": 42.0, "ts_iso": "2026-06-17T09:30:00"},   # net-only, keep
        ],
    }
    paper_path.write_text(json.dumps(seed), encoding="utf-8")

    log_path = _write_log(tmp_path, SAMPLE_LOG)
    reports_dir = tmp_path / "reports"

    common = dict(
        entry_date=date(2026, 6, 16),
        log_path=log_path,
        paper_ledger_path=paper_path,
        live_ledger_path=tmp_path / "state" / "does_not_exist_live.json",
        reports_dir=reports_dir,
        fetch_fn=_fake_fetch_factory(),
    )

    res1 = reconstruct_for_date(**common)
    after1 = json.loads(paper_path.read_text())
    recon1 = [t for t in after1["trades"] if t.get("source") == "reconstructed"]
    assert len(recon1) == 2                        # 2 fired signals reconstructed
    assert res1["n_fired"] == 2
    # original non-reconstructed trade preserved
    keep = [t for t in after1["trades"] if t.get("source") != "reconstructed"]
    assert any(t.get("net_pnl_inr") == 42.0 for t in keep)
    # backup written
    assert (paper_path.parent / (paper_path.name + ".bak-2026-06-16")).exists()

    # Re-run for the SAME date -> still exactly 2 reconstructed (no dupes).
    reconstruct_for_date(**common)
    after2 = json.loads(paper_path.read_text())
    recon2 = [t for t in after2["trades"] if t.get("source") == "reconstructed"]
    assert len(recon2) == 2
    keep2 = [t for t in after2["trades"] if t.get("source") != "reconstructed"]
    assert any(t.get("net_pnl_inr") == 42.0 for t in keep2)


def test_slippage_report_with_live_match(tmp_path):
    # Live ledger has a real fill for SAMPANN (recorded at settle = exit date).
    paper_path = tmp_path / "state" / "decay_tripwire_close_dn_overnight_long.json"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(json.dumps({"trades": []}), encoding="utf-8")

    live_path = tmp_path / "state" / "decay_tripwire_close_dn_overnight_long_live.json"
    live_path.write_text(json.dumps({
        "trades": [
            {
                "symbol": "NSE:SAMPANN",
                "entry_price": 103.5,    # real entry > idealized 103.2 -> adverse
                "exit_price": 109.9,     # real exit < idealized 110.0 -> adverse
                "qty": 96, "gross_pnl_inr": 614.4, "fees_inr": 30.0,
                "net_pnl_inr": 584.4, "ts_iso": "2026-06-17T09:30:01",
                "exit_reason": "t1_settle",
            }
        ]
    }), encoding="utf-8")

    log_path = _write_log(tmp_path, SAMPLE_LOG)
    reports_dir = tmp_path / "reports"

    res = reconstruct_for_date(
        entry_date=date(2026, 6, 16),
        log_path=log_path,
        paper_ledger_path=paper_path,
        live_ledger_path=live_path,
        reports_dir=reports_dir,
        fetch_fn=_fake_fetch_factory(),
    )
    assert res["n_fired"] == 2
    assert res["n_traded_live"] == 1

    report_path = reports_dir / "overnight_slippage_2026-06-16.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["date"] == "2026-06-16"
    assert report["n_traded_live"] == 1
    # the matched per-trade row carries real fills + slip
    matched = [r for r in report["per_trade"] if r["real_entry"] is not None]
    assert len(matched) == 1
    row = matched[0]
    assert row["symbol"] == "NSE:SAMPANN"
    assert round(row["entry_slip"], 4) == round(103.5 - 103.2, 4)
    assert round(row["exit_slip"], 4) == round(110.0 - 109.9, 4)
    assert row["live_net_10k"] == 584.4


def test_exit_today_uses_intraday_fetch(tmp_path, monkeypatch):
    # When exit_date == today, the orchestrator must use the INTRADAY fetch for
    # the exit price (the historical endpoint lacks today's bars) and the
    # HISTORICAL fetch for the entry price.
    import tools.overnight_paper_slippage as mod

    # exit date for entry 2026-06-16 is 2026-06-17; pretend today is that date.
    monkeypatch.setattr(mod, "_today_ist", lambda: date(2026, 6, 17))

    hist_calls = []
    intraday_calls = []

    def fake_hist(symbols, from_date_iso, to_date_iso):
        hist_calls.append((tuple(symbols), from_date_iso, to_date_iso))
        # Historical only ever asked for the (past) entry day.
        return {s: _mock_df_entry_day() for s in symbols}

    def fake_intraday(symbols):
        intraday_calls.append(tuple(symbols))
        # Intraday returns TODAY's (exit-day) bars only.
        return {s: _mock_df_exit_day() for s in symbols}

    paper_path = tmp_path / "state" / "decay_tripwire_close_dn_overnight_long.json"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(json.dumps({"trades": []}), encoding="utf-8")
    log_path = _write_log(tmp_path, SAMPLE_LOG)
    reports_dir = tmp_path / "reports"

    res = reconstruct_for_date(
        entry_date=date(2026, 6, 16),
        log_path=log_path,
        paper_ledger_path=paper_path,
        live_ledger_path=tmp_path / "state" / "no_live.json",
        reports_dir=reports_dir,
        fetch_fn=fake_hist,
        intraday_fetch_fn=fake_intraday,
    )

    # Intraday fetch was used for the exit (today); historical for entry.
    assert len(intraday_calls) == 1
    # Historical was called only for the entry day (from==to==entry date).
    assert all(f == t == "2026-06-16" for _, f, t in hist_calls)
    assert res["n_fired"] == 2
    assert res["n_reconstructed"] == 2
    assert res["n_missing_price"] == 0

    # Reconstruction used entry=103.2 (15:25 close) and exit=110.0 (09:15 open).
    after = json.loads(paper_path.read_text())
    recon = [t for t in after["trades"] if t.get("source") == "reconstructed"]
    assert len(recon) == 2
    for t in recon:
        assert t["entry_price"] == 103.2
        assert t["exit_price"] == 110.0


def test_exit_past_uses_historical_fetch(tmp_path, monkeypatch):
    # When exit_date is in the PAST (today is later), the orchestrator must use
    # the HISTORICAL fetch for BOTH entry and exit — the intraday fetch is never
    # called.
    import tools.overnight_paper_slippage as mod

    # Today is well after the exit date 2026-06-17.
    monkeypatch.setattr(mod, "_today_ist", lambda: date(2026, 6, 22))

    hist_calls = []
    intraday_calls = []

    def fake_hist(symbols, from_date_iso, to_date_iso):
        hist_calls.append((tuple(symbols), from_date_iso, to_date_iso))
        if from_date_iso == "2026-06-16":
            return {s: _mock_df_entry_day() for s in symbols}
        return {s: _mock_df_exit_day() for s in symbols}

    def fake_intraday(symbols):
        intraday_calls.append(tuple(symbols))
        return {s: _mock_df_exit_day() for s in symbols}

    paper_path = tmp_path / "state" / "decay_tripwire_close_dn_overnight_long.json"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(json.dumps({"trades": []}), encoding="utf-8")
    log_path = _write_log(tmp_path, SAMPLE_LOG)
    reports_dir = tmp_path / "reports"

    res = reconstruct_for_date(
        entry_date=date(2026, 6, 16),
        log_path=log_path,
        paper_ledger_path=paper_path,
        live_ledger_path=tmp_path / "state" / "no_live.json",
        reports_dir=reports_dir,
        fetch_fn=fake_hist,
        intraday_fetch_fn=fake_intraday,
    )

    # Intraday NEVER called; historical used for both entry and exit days.
    assert intraday_calls == []
    from_to = {(f, t) for _, f, t in hist_calls}
    assert ("2026-06-16", "2026-06-16") in from_to
    assert ("2026-06-17", "2026-06-17") in from_to
    assert res["n_fired"] == 2
    assert res["n_reconstructed"] == 2
    assert res["n_missing_price"] == 0

    after = json.loads(paper_path.read_text())
    recon = [t for t in after["trades"] if t.get("source") == "reconstructed"]
    assert len(recon) == 2
    for t in recon:
        assert t["entry_price"] == 103.2
        assert t["exit_price"] == 110.0
