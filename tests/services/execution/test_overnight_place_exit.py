import os, sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from services.capital_manager import OvernightSlotPool


def _seed_t0_open_slot(state_path):
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool.reserve(symbol="NSE:RELIANCE", product="MTF", leverage=2.5, today=date(2026, 6, 22))
    pool.attach_buy_fill(slot.slot_id, fill_price=140.0, fill_ts_iso="2026-06-22T15:26:00", order_id="BUY1")
    pool.persist()
    return slot.slot_id


def _config(state_path):
    return {"setups": {"close_dn_overnight_long": {
        "mode": "overnight", "enabled": True, "paper_enabled": True,
        "capital_allocation": {"state_file": str(state_path), "max_concurrent_slots": 2,
                               "margin_per_slot_inr": 10000, "max_new_positions_per_day": 2},
        "catastrophe_stop_pct": 5.0, "gtt_limit_buffer_pct": 0.5,
    }}}


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "slots.json"


@pytest.fixture
def patched_registry(monkeypatch, state_path):
    spec = MagicMock()
    spec.name = "close_dn_overnight_long"; spec.mode = "overnight"; spec.enabled = True
    spec.raw_config = _config(state_path)["setups"]["close_dn_overnight_long"]
    import services.execution.overnight_handlers as oh
    monkeypatch.setattr(oh, "_select_overnight_setups", lambda config, *, paper_mode: [spec])
    return spec


def test_run_place_exit_places_amo_and_gtt(state_path, patched_registry):
    import services.execution.overnight_handlers as oh
    sid = _seed_t0_open_slot(state_path)
    broker = MagicMock()
    broker.place_order.return_value = "AMO1"
    broker.place_gtt_stop.return_value = "GTT1"
    summary = oh.run_place_exit(_config(state_path), broker,
                                now_ist=pd.Timestamp("2026-06-22 16:05:00"), paper_mode=False)
    assert summary["placed_count"] == 1
    amo_kwargs = broker.place_order.call_args.kwargs
    assert amo_kwargs["side"] == "SELL" and amo_kwargs["variety"] == "amo" and amo_kwargs["product"] == "MTF"
    gtt_kwargs = broker.place_gtt_stop.call_args.kwargs
    assert round(gtt_kwargs["trigger_price"], 2) == 133.0
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool._get_slot(sid)
    assert slot.amo_sell_order_id == "AMO1"
    assert slot.gtt_id == "GTT1"


def test_run_place_exit_idempotent(state_path, patched_registry):
    import services.execution.overnight_handlers as oh
    _seed_t0_open_slot(state_path)
    broker = MagicMock(); broker.place_order.return_value = "AMO1"; broker.place_gtt_stop.return_value = "GTT1"
    cfg = _config(state_path)
    oh.run_place_exit(cfg, broker, now_ist=pd.Timestamp("2026-06-22 16:05:00"), paper_mode=False)
    broker.place_order.reset_mock()
    summary2 = oh.run_place_exit(cfg, broker, now_ist=pd.Timestamp("2026-06-22 16:06:00"), paper_mode=False)
    assert summary2["placed_count"] == 0
    broker.place_order.assert_not_called()


def test_run_place_exit_captures_idealized_entry(state_path, patched_registry, monkeypatch):
    import services.execution.overnight_handlers as oh
    sid = _seed_t0_open_slot(state_path)
    broker = MagicMock()
    broker.place_order.return_value = "AMO1"
    broker.place_gtt_stop.return_value = "GTT1"
    # Idealized entry (15:25 close) returns a known value — stored on the slot.
    monkeypatch.setattr(oh, "_paper_fill_price_entry", lambda b, sym, d: 138.42)
    summary = oh.run_place_exit(_config(state_path), broker,
                                now_ist=pd.Timestamp("2026-06-22 16:05:00"), paper_mode=False)
    assert summary["placed_count"] == 1
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool._get_slot(sid)
    assert slot.idealized_entry_price == 138.42
    # AMO/GTT logic still ran (additive capture didn't disrupt placement).
    assert slot.amo_sell_order_id == "AMO1"
    assert slot.gtt_id == "GTT1"


def test_run_place_exit_idealized_capture_failure_does_not_block(state_path, patched_registry, monkeypatch):
    import services.execution.overnight_handlers as oh
    sid = _seed_t0_open_slot(state_path)
    broker = MagicMock()
    broker.place_order.return_value = "AMO1"
    broker.place_gtt_stop.return_value = "GTT1"

    def _boom(b, sym, d):
        raise RuntimeError("data unavailable")
    monkeypatch.setattr(oh, "_paper_fill_price_entry", _boom)
    summary = oh.run_place_exit(_config(state_path), broker,
                                now_ist=pd.Timestamp("2026-06-22 16:05:00"), paper_mode=False)
    # Placement still succeeds; idealized stays None.
    assert summary["placed_count"] == 1
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool._get_slot(sid)
    assert slot.idealized_entry_price is None
    assert slot.amo_sell_order_id == "AMO1"


def test_run_place_exit_refuses_before_amo_window(state_path, patched_registry):
    import services.execution.overnight_handlers as oh
    _seed_t0_open_slot(state_path)
    broker = MagicMock()
    summary = oh.run_place_exit(_config(state_path), broker,
                                now_ist=pd.Timestamp("2026-06-22 15:30:00"), paper_mode=False)
    assert summary.get("refused_amo_window") is True
    broker.place_order.assert_not_called()
