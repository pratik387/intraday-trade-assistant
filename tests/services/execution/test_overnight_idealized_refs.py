"""Idealized-reference capture for slippage (run_verify_exit settle path).

Asserts run_verify_exit records the persisted idealized_entry/idealized_exit on
the live trade ledger via DecayTripwire.record_trade — distinct from the REAL
AMO fill in live mode. Additive: settle/pause logic unchanged.
"""
import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from services.capital_manager import OvernightSlotPool  # noqa: E402


def _config(state_path: Path, tripwire_path: Path) -> dict:
    return {"setups": {"close_dn_overnight_long": {
        "mode": "overnight", "enabled": True, "paper_enabled": True,
        "capital_allocation": {"state_file": str(state_path), "max_concurrent_slots": 2,
                               "margin_per_slot_inr": 100000, "max_new_positions_per_day": 2},
        "decay_tripwire": {"state_file": str(tripwire_path), "window_trades": 30,
                           "pf_floor": 1.20, "sustained_weeks": 6},
    }}}


@pytest.fixture
def patched_registry(monkeypatch, state_path, tripwire_path):
    spec = MagicMock()
    spec.name = "close_dn_overnight_long"
    spec.mode = "overnight"
    spec.enabled = True
    spec.raw_config = _config(state_path, tripwire_path)["setups"]["close_dn_overnight_long"]
    import services.execution.overnight_handlers as oh
    monkeypatch.setattr(oh, "_select_overnight_setups", lambda config, *, paper_mode: [spec])
    return spec


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "slots.json"


@pytest.fixture
def tripwire_path(tmp_path):
    return tmp_path / "decay_tripwire_live.json"


def _seed_settleable_slot(state_path: Path, *, idealized_entry):
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=100000, max_new_per_day=2)
    slot = pool.reserve(symbol="NSE:RELIANCE", product="CNC", leverage=1.0, today=date(2026, 5, 21))
    pool.attach_buy_fill(slot.slot_id, fill_price=100.0, fill_ts_iso="2026-05-21T15:30:00", order_id="BUY1")
    slot.idealized_entry_price = idealized_entry
    pool.attach_amo_sell(slot.slot_id, "AMO1", date(2026, 5, 22))
    pool.persist()
    return slot.slot_id


def test_verify_exit_records_idealized_refs_live(state_path, tripwire_path, patched_registry, monkeypatch):
    """In LIVE mode the real AMO fill (101.0) differs from the idealized 09:15
    open (100.5); both are recorded — slippage = real - idealized."""
    import services.execution.overnight_handlers as oh
    _seed_settleable_slot(state_path, idealized_entry=99.5)

    broker = MagicMock()
    # Real live AMO fill = 101.0 (distinct from idealized 100.5).
    monkeypatch.setattr(oh, "_live_check_amo_fill", lambda b, oid: 101.0)
    # Idealized exit (09:15 open) = 100.5.
    monkeypatch.setattr(oh, "_paper_fill_price_exit", lambda b, sym, d: 100.5)

    summary = oh.run_verify_exit(_config(state_path, tripwire_path), broker,
                                 now_ist=pd.Timestamp("2026-05-22 09:30:00"), paper_mode=False)
    assert summary["settled_count"] == 1

    # The live trade record carries BOTH the real exit and the persisted idealized refs.
    ledger = json.loads(tripwire_path.read_text(encoding="utf-8"))
    rec = ledger["trades"][0]
    assert rec["exit_price"] == 101.0          # real AMO fill
    assert rec["idealized_exit"] == 100.5      # idealized 09:15 open (distinct)
    assert rec["idealized_entry"] == 99.5      # persisted at place-exit time


def test_verify_exit_idealized_exit_failure_records_none(state_path, tripwire_path, patched_registry, monkeypatch):
    import services.execution.overnight_handlers as oh
    _seed_settleable_slot(state_path, idealized_entry=99.5)

    broker = MagicMock()
    monkeypatch.setattr(oh, "_live_check_amo_fill", lambda b, oid: 101.0)

    def _boom(b, sym, d):
        raise RuntimeError("data unavailable")
    monkeypatch.setattr(oh, "_paper_fill_price_exit", _boom)

    summary = oh.run_verify_exit(_config(state_path, tripwire_path), broker,
                                 now_ist=pd.Timestamp("2026-05-22 09:30:00"), paper_mode=False)
    # Settle still happens; idealized_exit omitted (None -> not persisted).
    assert summary["settled_count"] == 1
    ledger = json.loads(tripwire_path.read_text(encoding="utf-8"))
    rec = ledger["trades"][0]
    assert "idealized_exit" not in rec
    assert rec["idealized_entry"] == 99.5
