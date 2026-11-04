"""Overnight slot fill times come from the exchange, not the pass's clock (2026-11-04).

Every 2026-11-03 live buy recorded 15:26:03 (the entry pass's start) though the
orders went out ~15:28, and every exit recorded the 09:16 verify run. The
ledger's ts_iso keeps the settlement time: the SPRT and the tripwire count from it.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import services.execution.overnight_handlers as oh  # noqa: E402
from tests.services.execution.test_overnight_handlers import (  # noqa: E402
    _failsafe_broker, _live_slot_state, _minimal_config, _slot1,
)
from tests.services.execution.test_overnight_live_canary import env  # noqa: E402,F401  (fixture)

FALLBACK = pd.Timestamp("2026-11-03 15:26:03")


def test_exchange_time_is_used_as_ist_naive():
    assert oh._fill_ts_iso({"exchange_timestamp": "2026-11-03T15:28:11"}, FALLBACK) == "2026-11-03T15:28:11"
    utc = datetime(2026, 11, 3, 9, 58, 11, tzinfo=timezone.utc)                # 15:28:11 IST
    assert oh._fill_ts_iso({"exchange_timestamp": utc}, FALLBACK) == "2026-11-03T15:28:11"


def test_no_exchange_time_falls_back_to_the_pass():
    for status in (None, {}, {"exchange_timestamp": None}, {"exchange_timestamp": "not a time"}):
        assert oh._fill_ts_iso(status, FALLBACK) == "2026-11-03T15:26:03"


def test_live_entry_records_the_exchange_fill_time(env, monkeypatch):
    monkeypatch.setattr(oh, "_live_poll_fill_ex", lambda broker, oid, timeout_sec: (
        100.0, {"status": "COMPLETE", "exchange_timestamp": "2026-11-04T15:28:07"}))
    env.sc["enabled"] = True
    env.write_ledger([])
    env.run()
    slots = json.loads(Path(env.sc["capital_allocation"]["state_file"]).read_text())["slots"]
    filled = [s for s in slots if s["status"] == "t0_open"]
    assert filled and all(s["buy_fill_ts"] == "2026-11-04T15:28:07" for s in filled)


def test_live_exit_records_the_amo_fill_time_and_ledger_keeps_settle_time(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    sc = cfg["setups"]["close_dn_overnight_long"]
    sc["enabled"] = True
    ledger = tmp_path / "tripwire.json"
    sc["decay_tripwire"] = {"window_trades": 30, "pf_floor": 1.2, "sustained_weeks": 3,
                            "state_file": str(ledger), "archive_entries_before": None,
                            "archive_label": None, "archive_regime": None}
    _live_slot_state(state_path)
    broker = _failsafe_broker({"AMO-001": {"status": "COMPLETE", "average_price": 37.0,
                                           "filled_quantity": 1388,
                                           "exchange_timestamp": "2026-07-20T09:15:02"}})
    now = pd.Timestamp("2026-07-20 09:16:30")
    summary = oh.run_verify_exit(cfg, broker, now_ist=now, paper_mode=False)
    assert summary["settled_count"] == 1
    assert _slot1(state_path)["sell_fill_ts"] == "2026-07-20T09:15:02"
    (row,) = json.loads(ledger.read_text())["trades"]
    assert row["ts_iso"] == now.isoformat()
