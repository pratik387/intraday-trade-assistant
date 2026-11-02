import os, sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from services.capital_manager import OvernightSlotPool


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "slots.json"


def _config(state_path):
    return {"setups": {"close_dn_overnight_long": {
        "mode": "overnight", "enabled": True, "paper_enabled": True,
        "capital_allocation": {"state_file": str(state_path), "max_concurrent_slots": 2,
                               "margin_per_slot_inr": 10000, "max_new_positions_per_day": 2},
        "catastrophe_stop_pct": 5.0, "gtt_limit_buffer_pct": 0.5, "entry_limit_buffer_pct": 1.0,
    }}}


def _seed_ready_to_settle(state_path):
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool.reserve(symbol="NSE:RELIANCE", product="MTF", leverage=2.5, today=date(2026, 6, 22))
    pool.attach_buy_fill(slot.slot_id, fill_price=140.0, fill_ts_iso="2026-06-22T15:26:00", order_id="BUY1")
    pool.attach_amo_sell(slot.slot_id, "AMO1", date(2026, 6, 23))
    slot.gtt_id = "GTT1"
    pool.persist()
    return slot.slot_id


def test_verify_exit_cancels_gtt_on_settle(monkeypatch, state_path):
    import services.execution.overnight_handlers as oh
    sid = _seed_ready_to_settle(state_path)
    spec = MagicMock(); spec.name = "close_dn_overnight_long"; spec.mode = "overnight"; spec.enabled = True
    spec.raw_config = _config(state_path)["setups"]["close_dn_overnight_long"]
    monkeypatch.setattr(oh, "_select_overnight_setups", lambda config, *, paper_mode: [spec])
    # exits read the managed set (flag-independent) — inject there too
    monkeypatch.setattr(oh, "_managed_overnight_setups", lambda config: [spec])
    monkeypatch.setattr(oh, "_paper_fill_price_exit", lambda b, s, d: 150.0)
    broker = MagicMock(); broker.cancel_gtt.return_value = True
    oh.run_verify_exit(_config(state_path), broker,
                       now_ist=pd.Timestamp("2026-06-23 09:30:00"), paper_mode=True)
    broker.cancel_gtt.assert_called_once_with("GTT1")
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    assert pool._get_slot(sid).gtt_id is None


@pytest.mark.parametrize("boundary, archived", [("2026-06-23", True), ("2026-06-22", False)])
def test_verify_exit_routes_settles_across_the_tripwire_boundary(monkeypatch, state_path, tmp_path,
                                                                   boundary, archived):
    """run_verify_exit passes the slot's entry date, so a position ENTERED before
    decay_tripwire.archive_entries_before settles to the archive, not the live ledger."""
    import json
    import services.execution.overnight_handlers as oh
    _seed_ready_to_settle(state_path)                       # entered 2026-06-22
    ledger = tmp_path / "tw_live.json"
    raw = _config(state_path)["setups"]["close_dn_overnight_long"]
    raw["decay_tripwire"] = {"window_trades": 30, "pf_floor": 1.2, "sustained_weeks": 6,
                             "state_file": str(ledger), "archive_entries_before": boundary,
                             "archive_label": "pre_test", "archive_regime": "before the test"}
    spec = MagicMock(); spec.name = "close_dn_overnight_long"; spec.mode = "overnight"; spec.enabled = True
    spec.raw_config = raw
    monkeypatch.setattr(oh, "_select_overnight_setups", lambda config, *, paper_mode: [spec])
    monkeypatch.setattr(oh, "_managed_overnight_setups", lambda config: [spec])
    monkeypatch.setattr(oh, "_paper_fill_price_exit", lambda b, s, d: 150.0)
    broker = MagicMock(); broker.cancel_gtt.return_value = True
    oh.run_verify_exit({"setups": {"close_dn_overnight_long": raw}}, broker,
                       now_ist=pd.Timestamp("2026-06-23 09:30:00"), paper_mode=True)
    archive = tmp_path / "archive" / "tw_live.pre_test.json"
    if archived:
        assert json.loads(archive.read_text())["trades"][0]["entry_date"] == "2026-06-22"
        assert not ledger.exists() or json.loads(ledger.read_text())["trades"] == []
    else:
        assert not archive.exists()
        (t,) = json.loads(ledger.read_text())["trades"]
        assert t["entry_date"] == "2026-06-22" and t["symbol"] == "NSE:RELIANCE"


def test_a_failed_ledger_write_still_settles_every_slot(monkeypatch, state_path, tmp_path):
    """A ledger error on one settle must not leave the other slots unsettled with live GTTs."""
    import json
    import services.execution.overnight_handlers as oh
    import services.risk.decay_tripwire as dtw
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    for sym, gtt in (("NSE:RELIANCE", "GTT1"), ("NSE:TCS", "GTT2")):
        slot = pool.reserve(symbol=sym, product="CNC", leverage=1.0, today=date(2026, 6, 22))
        pool.attach_buy_fill(slot.slot_id, fill_price=140.0, fill_ts_iso="2026-06-22T15:26:00", order_id="B" + gtt)
        pool.attach_amo_sell(slot.slot_id, "A" + gtt, date(2026, 6, 23))
        slot.gtt_id = gtt
    pool.persist()
    ledger = tmp_path / "tw_live.json"
    raw = _config(state_path)["setups"]["close_dn_overnight_long"]
    raw["decay_tripwire"] = {"window_trades": 30, "pf_floor": 1.2, "sustained_weeks": 6,
                             "state_file": str(ledger), "archive_entries_before": None,
                             "archive_label": None, "archive_regime": None}
    spec = MagicMock(); spec.name = "close_dn_overnight_long"; spec.mode = "overnight"; spec.enabled = True
    spec.raw_config = raw
    monkeypatch.setattr(oh, "_select_overnight_setups", lambda config, *, paper_mode: [spec])
    monkeypatch.setattr(oh, "_managed_overnight_setups", lambda config: [spec])
    monkeypatch.setattr(oh, "_paper_fill_price_exit", lambda b, s, d: 150.0)
    calls = {"n": 0}
    real = dtw.DecayTripwire.record_trade

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("archive is for another setup")
        return real(self, *a, **k)
    monkeypatch.setattr(dtw.DecayTripwire, "record_trade", flaky)
    broker = MagicMock(); broker.cancel_gtt.return_value = True
    oh.run_verify_exit({"setups": {"close_dn_overnight_long": raw}}, broker,
                       now_ist=pd.Timestamp("2026-06-23 09:30:00"), paper_mode=True)
    assert sorted(c.args[0] for c in broker.cancel_gtt.call_args_list) == ["GTT1", "GTT2"]
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    assert all(s.status == "t1_settling" for s in pool.active())
    marker = Path(str(ledger) + ".WRITE_FAILED")
    (line,) = marker.read_text().splitlines()
    assert "another setup" in json.loads(line)["error"]
    assert len(json.loads(ledger.read_text())["trades"]) == 1        # the other settle is booked
