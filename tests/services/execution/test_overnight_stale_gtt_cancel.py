import os, sys
from datetime import date
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


def _seed_stale_slot(state_path):
    """Seed a t0_open slot whose expected_exit_date is a PRIOR session (a missed
    cron). attach_amo_sell keeps status at t0_open and records the exit date, so
    when run_verify_exit fires with today AFTER that date it hits the stale branch.
    """
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool.reserve(symbol="NSE:RELIANCE", product="MTF", leverage=2.5, today=date(2026, 6, 18))
    pool.attach_buy_fill(slot.slot_id, fill_price=140.0, fill_ts_iso="2026-06-18T15:26:00", order_id="BUY1")
    # expected_exit_date in the PAST relative to the verify-exit now_ist below
    pool.attach_amo_sell(slot.slot_id, "AMO_STALE", date(2026, 6, 19))
    slot.gtt_id = "GTT_STALE"
    pool.persist()
    return slot.slot_id


def test_verify_exit_cancels_gtt_on_stale_orphan(monkeypatch, state_path):
    import services.execution.overnight_handlers as oh
    sid = _seed_stale_slot(state_path)
    spec = MagicMock(); spec.name = "close_dn_overnight_long"; spec.mode = "overnight"; spec.enabled = True
    spec.raw_config = _config(state_path)["setups"]["close_dn_overnight_long"]
    monkeypatch.setattr(oh, "_select_overnight_setups", lambda config, *, paper_mode: [spec])
    broker = MagicMock(); broker.cancel_gtt.return_value = True
    # now_ist (2026-06-22) is AFTER expected_exit_date (2026-06-19) -> stale branch
    summary = oh.run_verify_exit(_config(state_path), broker,
                                 now_ist=pd.Timestamp("2026-06-22 09:30:00"), paper_mode=True)
    broker.cancel_gtt.assert_called_once_with("GTT_STALE")
    assert summary["orphan_t0_count"] == 1
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    assert pool._get_slot(sid).gtt_id is None
