"""Paper-mode end-to-end of the overnight AMO+GTT flow via a real MockBroker.

Proves MockBroker's place_gtt_stop / cancel_gtt stubs let the place-exit cron
record a GTT id and the verify-exit cron cancel it — the same flow the live
hybrid broker runs, but fully simulated for the VM paper daemon.
"""
import os
import sys
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from services.capital_manager import OvernightSlotPool
from broker.mock.mock_broker import MockBroker
import services.execution.overnight_handlers as oh


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "slots.json"


def _config(state_path):
    return {"setups": {"close_dn_overnight_long": {
        "mode": "overnight", "enabled": True, "paper_enabled": True,
        "capital_allocation": {"state_file": str(state_path), "max_concurrent_slots": 2,
                               "margin_per_slot_inr": 10000, "max_new_positions_per_day": 2},
        "catastrophe_stop_pct": 5.0, "gtt_limit_buffer_pct": 0.5,
        "entry_limit_buffer_pct": 1.0,
    }}}


def _spec(state_path):
    spec = MagicMock()
    spec.name = "close_dn_overnight_long"; spec.mode = "overnight"; spec.enabled = True
    spec.raw_config = _config(state_path)["setups"]["close_dn_overnight_long"]
    return spec


def test_paper_place_exit_then_verify_cancels_gtt(monkeypatch, state_path):
    # data_sdk set (MagicMock) so MockBroker skips the nse_all.json instrument
    # load; the exit path only needs state + simulated orders/GTT.
    broker = MockBroker(path_json="nse_all.json", slippage_bps=5.0, data_sdk=MagicMock())
    monkeypatch.setattr(oh, "_select_overnight_setups", lambda config, *, paper_mode: [_spec(state_path)])

    # Seed a t0_open slot (post-BUY).
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool.reserve(symbol="NSE:RELIANCE", product="MTF", leverage=2.5, today=date(2026, 6, 22))
    pool.attach_buy_fill(slot.slot_id, fill_price=140.0, fill_ts_iso="2026-06-22T15:26:00", order_id="BUY1")
    pool.persist()

    # place-exit: AMO + GTT both placed in paper (gtt no longer "fails").
    s1 = oh.run_place_exit(_config(state_path), broker,
                           now_ist=pd.Timestamp("2026-06-22 16:05:00"), paper_mode=True)
    assert s1["placed_count"] == 1
    assert s1["gtt_failed_count"] == 0
    pool2 = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    seeded = pool2._get_slot(slot.slot_id)
    assert seeded.gtt_id and seeded.gtt_id.startswith("PAPER_GTT_")
    assert seeded.amo_sell_order_id is not None

    # verify-exit next morning: settle + cancel the GTT.
    monkeypatch.setattr(oh, "_paper_fill_price_exit", lambda b, s, d: 150.0)
    s2 = oh.run_verify_exit(_config(state_path), broker,
                            now_ist=pd.Timestamp("2026-06-23 09:30:00"), paper_mode=True)
    assert s2["settled_count"] == 1
    pool3 = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    assert pool3._get_slot(slot.slot_id).gtt_id is None  # GTT cancelled on settle
