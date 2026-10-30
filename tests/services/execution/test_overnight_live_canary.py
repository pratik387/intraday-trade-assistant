"""Overnight live canary (Risk Charter restart rules) and the paper mirror under a live pause.

- With live paused (enabled=False, paper_enabled=True) the LIVE cron still detects
  and logs every fire, because the paper mirror is rebuilt from those log lines,
  but places no order.
- Canary on: live orders only for names under max_adv_inr ADV, as CNC, at
  notional_inr; stopped by the monthly stop or by the SPRT accepting 'no edge'.
"""
import json
import logging
import math
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import services.execution.overnight_handlers as oh  # noqa: E402
import services.dispatch.setup_registry as sr  # noqa: E402
from tests.services.execution.test_overnight_handlers import _minimal_config  # noqa: E402

TODAY = "2026-11-04"
NOW = pd.Timestamp(f"{TODAY} 15:26:00")
SETUP = "close_dn_overnight_long"
SPRT = {"mu1_pct": 0.44, "sd_pct": 2.4, "alpha": 0.10, "beta": 0.20}
CANARY = {"enabled": True, "start_date": "2026-11-01", "max_adv_inr": 2500000,
          "notional_inr": 15000, "min_notional_inr": 10000, "monthly_stop_inr": -3000,
          "sprt": SPRT}


class _Detector:
    """Fires on every symbol as MTF at 3x, like the real detector on an MTF name."""
    def __init__(self, cfg):
        pass

    def detect(self, ctx):
        oh.logger.info("close_dn_overnight_long fired | symbol=%s product=MTF lev=3.00", ctx)
        evt = SimpleNamespace(context={"product": "MTF", "leverage": 3.0})
        return SimpleNamespace(structure_detected=True, events=[evt])

    def plan_long_strategy(self, ctx, evt):
        return SimpleNamespace(qty=750, entry_price=100.0)


def _trade(ts, pnl, entry=100.0, qty=150):
    return {"net_pnl_inr": pnl, "ts_iso": ts, "entry_price": entry, "qty": qty}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A live run_entry with fake detection and a recording broker; ADV per symbol below."""
    adv = {"NSE:THIN": 1_500_000, "NSE:THICK": 40_000_000, "NSE:NOADV": None}
    root = tmp_path / "repo"
    (root / "data" / "close_dn_baseline").mkdir(parents=True)
    (root / "data" / "close_dn_baseline" / "candidates_latest.json").write_text(json.dumps({
        "session_date": TODAY,
        "candidates": [{"symbol": s, "adv_inr": a, "prior_close": 100.0, "prev_prior_close": 95.0}
                       for s, a in adv.items()]}))
    monkeypatch.setattr(oh, "__file__", str(root / "services" / "execution" / "overnight_handlers.py"))
    monkeypatch.setattr(sr, "_import_path", lambda p: _Detector)
    monkeypatch.setattr(oh, "_build_market_context", lambda broker, sym, *a, **k: sym)
    monkeypatch.setattr(oh, "_safe_quote", lambda broker, sym: (100.0, 120.0, 80.0))
    monkeypatch.setattr(oh, "_instrument_tick", lambda broker, sym, cfg: 0.05)
    monkeypatch.setattr(oh, "_safe_top_of_book", lambda broker, sym: (None, None, None, None))
    monkeypatch.setattr(oh, "_entry_cutoff_reached", lambda cfg, paper: False)
    monkeypatch.setattr(oh, "_live_poll_fill_ex", lambda broker, oid, timeout_sec: (100.0, {"status": "COMPLETE"}))
    monkeypatch.setattr(oh, "_reconcile_unattached_buys", lambda *a, **k: 0)
    placed = []

    def _buy(broker, **kw):
        placed.append(kw)
        return f"B{len(placed)}"
    monkeypatch.setattr(oh, "_place_buy", _buy)

    cfg = _minimal_config(tmp_path / "overnight_slots.json")
    sc = cfg["setups"][SETUP]
    sc["capital_allocation"]["max_new_positions_per_day"] = 4
    sc["entry_reconcile_delay_sec"] = 0
    sc["live_canary"] = dict(CANARY)
    ledger = tmp_path / "tripwire_live.json"
    sc["decay_tripwire"] = {"window_trades": 30, "pf_floor": 1.2, "sustained_weeks": 3,
                            "state_file": str(ledger), "archive_entries_before": None,
                            "archive_label": None, "archive_regime": None}
    broker = MagicMock()
    broker._data_sdk = None

    def write_ledger(trades):
        ledger.write_text(json.dumps({"setup_name": SETUP, "window_trades": 30, "pf_floor": 1.2,
                                      "sustained_weeks": 3, "first_below_floor_ts": None,
                                      "paused_since": None, "trades": trades}))

    def run():
        return oh.run_entry(cfg, broker, now_ist=NOW, paper_mode=False)
    return SimpleNamespace(cfg=cfg, sc=sc, run=run, placed=placed, write_ledger=write_ledger,
                           cand_path=root / "data" / "close_dn_baseline" / "candidates_latest.json")


def test_live_paused_still_logs_fires_for_the_paper_mirror(env, caplog):
    env.sc["enabled"] = False
    with caplog.at_level(logging.INFO):
        s = env.run()
    assert env.placed == []
    assert s["mirror_only_count"] == 3
    fired = [r for r in caplog.records if "close_dn_overnight_long fired" in r.getMessage()]
    assert len(fired) == 3                      # the lines tools/overnight_paper_slippage.py parses


def test_live_paused_mirror_survives_a_tripped_tripwire(env, caplog):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.write_ledger([])
    led = Path(env.sc["decay_tripwire"]["state_file"])
    led.write_text(json.dumps({**json.loads(led.read_text()), "paused_since": "2026-11-02T09:16:00"}))
    s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3


def test_canary_takes_only_thin_names_as_cnc_at_canary_size(env):
    env.sc["enabled"] = True
    env.write_ledger([])
    s = env.run()
    assert [p["symbol"] for p in env.placed] == ["NSE:THIN"]
    (p,) = env.placed
    assert p["product"] == "CNC"
    assert p["qty"] * p["limit_price"] <= CANARY["notional_inr"]
    assert p["qty"] * p["limit_price"] >= CANARY["min_notional_inr"]
    assert s["canary_adv_skipped"] == 2         # THICK (Rs4cr) and NOADV (unknown)
    slots = json.loads(Path(env.sc["capital_allocation"]["state_file"]).read_text())["slots"]
    (taken,) = [x for x in slots if x["status"] != "free"]
    assert taken["product"] == "CNC" and taken["leverage"] == 1.0
    assert taken["margin_inr"] == pytest.approx(p["qty"] * p["limit_price"])


def test_canary_floor_skips_when_one_share_exceeds_the_notional(env, monkeypatch):
    env.sc["enabled"] = True
    env.write_ledger([])
    monkeypatch.setattr(oh, "_safe_quote", lambda broker, sym: (16000.0, 18000.0, 14000.0))
    env.run()
    assert env.placed == []


def test_monthly_stop_blocks_live_but_not_the_mirror(env, caplog):
    env.sc["enabled"] = True
    env.write_ledger([_trade("2026-11-02T09:16:00", -1800), _trade("2026-11-03T09:16:00", -1300)])
    with caplog.at_level(logging.INFO):
        s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3
    assert any(r.levelno == logging.CRITICAL and "monthly stop" in r.getMessage() for r in caplog.records)


def test_losses_before_the_canary_start_do_not_count(env):
    env.sc["enabled"] = True
    env.write_ledger([_trade("2026-10-28T09:16:00", -9000), _trade("2026-11-01T09:16:00", -9000)])
    env.run()
    assert [p["symbol"] for p in env.placed] == ["NSE:THIN"]


@pytest.mark.parametrize("bad", [{"start_date": None}, {"start_date": "soon"},
                                 {"min_notional_inr": 20000}, {"monthly_stop_inr": 3000},
                                 {"sprt": {**SPRT, "sd_pct": 0}}, {"sprt": {**SPRT, "alpha": 1.0}}])
def test_unusable_canary_config_blocks_live_but_keeps_the_mirror(env, caplog, bad):
    env.sc["enabled"] = True
    env.write_ledger([])
    env.sc["live_canary"].update(bad)
    with caplog.at_level(logging.INFO):
        s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3
    assert any(r.levelno == logging.CRITICAL and "cannot be evaluated" in r.getMessage()
               for r in caplog.records)


def test_missing_ledger_blocks_live_entries(env):
    env.sc["enabled"] = True                    # no write_ledger: the file does not exist
    s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3


def test_corrupt_ledger_blocks_live_but_the_cron_and_mirror_survive(env):
    env.sc["enabled"] = True
    Path(env.sc["decay_tripwire"]["state_file"]).write_text("{not json")
    s = env.run()
    assert env.placed == [] and s["mirror_only_count"] == 3


def test_net_only_ledger_rows_do_not_break_the_gate(env):
    env.sc["enabled"] = True
    env.write_ledger([{"net_pnl_inr": 0.0, "ts_iso": "2026-11-03T09:30:00"}])
    env.run()
    assert [p["symbol"] for p in env.placed] == ["NSE:THIN"]
    # ...and they still count toward the monthly stop
    assert not oh.canary_gate(CANARY, [{"net_pnl_inr": -3100.0, "ts_iso": "2026-11-03T09:30:00"}],
                              date(2026, 11, 20))[0]


@pytest.mark.parametrize("adv, taken", [(120_000, True), (80_000, False)])
def test_participation_cap_uses_the_canary_floor(env, adv, taken):
    """10% of ADV: Rs1.2L ADV caps the order at ~Rs12k (above the Rs10k canary floor:
    shrunk, taken); Rs0.8L caps it at Rs8k (skipped)."""
    env.sc["enabled"] = True
    env.write_ledger([])
    env.sc["participation_cap"] = {"enabled": True, "max_participation_pct": 0.1,
                                   "skip_when_adv_missing": True,
                                   "min_notional_after_cap_inr": 25000,
                                   "min_notional_after_cap_inr_cnc": 25000,
                                   "fallback_to_cnc_below_mtf_floor": False,
                                   "max_tick_pct_of_price": 0.2, "min_qty_after_cap": 1}
    cands = env.cand_path
    data = json.loads(cands.read_text())
    for c in data["candidates"]:
        if c["symbol"] == "NSE:THIN":
            c["adv_inr"] = adv
    cands.write_text(json.dumps(data))
    env.run()
    if taken:
        (p,) = env.placed
        assert 10_000 <= p["qty"] * p["limit_price"] <= 0.1 * adv
    else:
        assert env.placed == []


def test_disabled_canary_is_full_size_live(env):
    env.sc["enabled"] = True
    env.sc["live_canary"]["enabled"] = False
    env.write_ledger([])
    env.run()
    assert sorted(p["symbol"] for p in env.placed) == ["NSE:NOADV", "NSE:THICK", "NSE:THIN"]
    assert {p["product"] for p in env.placed} == {"MTF"}


def test_paper_mode_ignores_the_canary(env, monkeypatch):
    env.sc["enabled"] = True
    monkeypatch.setattr(oh, "_paper_fill_price_entry", lambda broker, sym, d: 100.0)
    oh.run_entry(env.cfg, MagicMock(_data_sdk=None), now_ist=NOW, paper_mode=True)
    assert len(env.placed) == 3 and {p["product"] for p in env.placed} == {"MTF"}


# ---- the gate itself ------------------------------------------------------------

def _gate(trades, today=date(2026, 11, 20)):
    return oh.canary_gate(CANARY, trades, today)


def test_gate_monthly_stop_is_calendar_month_to_date():
    big = lambda pnl: _trade("2026-11-05T09:16:00", pnl, qty=10_000)   # tiny % so the SPRT stays quiet
    assert not _gate([big(-3000)])[0]
    assert _gate([big(-2999)])[0]
    assert _gate([big(-3000)], today=date(2026, 12, 1))[0]             # a new month starts clean


def test_gate_sprt_thresholds():
    lo, hi = math.log(0.2 / 0.9), math.log(0.8 / 0.1)
    # each zero-return trade moves the LLR by -mu1^2/(2 sd^2) = -0.0168: 'no edge' after 90
    zero = lambda n: [_trade(f"2026-11-{2 + i % 18:02d}T09:16:00", 0.0) for i in range(n)]
    step = -(0.44 ** 2) / (2 * 2.4 ** 2)
    assert oh.canary_sprt_llr([0.0] * 10, SPRT) == pytest.approx(10 * step)
    n_h0 = math.ceil(lo / step)
    ok, why = _gate(zero(n_h0))
    assert not ok and "no edge" in why
    ok, why = _gate(zero(n_h0 - 1))
    assert ok and "canary running" in why
    # a run of +1.0% trades accepts 'edge' (allowed, logged for a manual decision)
    win = [_trade(f"2026-11-{2 + i % 18:02d}T09:16:00", 150.0) for i in range(60)]
    assert oh.canary_sprt_llr([1.0] * 60, SPRT) >= hi
    ok, why = oh.canary_gate({**CANARY, "monthly_stop_inr": -1e9}, win, date(2026, 11, 20))
    assert ok and "accepts 'edge'" in why


def test_real_config_has_the_canary_off_until_a_start_date_is_set():
    cfg = json.loads((_REPO_ROOT / "config" / "configuration.json").read_text(encoding="utf-8"))
    sc = cfg["setups"][SETUP]
    c = sc["live_canary"]
    assert c["enabled"] is True and c["start_date"] is None
    assert sc["enabled"] is False                 # live stays paused: enabling needs a start_date
    assert c["max_adv_inr"] == 2_500_000 and 10_000 <= c["min_notional_inr"] <= c["notional_inr"] <= 15_000
    assert c["sprt"] == SPRT and c["monthly_stop_inr"] == -3000
