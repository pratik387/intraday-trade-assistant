"""Tests for OvernightSlotPool: persistence, lifecycle, capacity, errors."""
import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.capital_manager import OvernightSlot, OvernightSlotPool  # noqa: E402,F401


def _pool(tmp_path: Path, max_slots: int = 4, margin: float = 100000, max_new: int = 2):
    state_path = tmp_path / "overnight_slots.json"
    return OvernightSlotPool(state_path, max_slots, margin, max_new), state_path


def test_pool_loads_empty_state_on_first_run(tmp_path):
    pool, _ = _pool(tmp_path)
    assert pool.free_count() == 4
    assert pool.open_count() == 0
    assert pool.settling_count() == 0


def test_reserve_returns_slot_when_free_capacity(tmp_path):
    pool, _ = _pool(tmp_path)
    slot = pool.reserve("NSE:RELIANCE", "MTF", 3.85, date(2026, 5, 21))
    assert slot is not None
    assert slot.symbol == "NSE:RELIANCE"
    assert slot.product == "MTF"
    assert slot.leverage == 3.85
    assert slot.status == "t0_open"
    assert slot.notional_inr == pytest.approx(385000.0)
    assert pool.free_count() == 3


def test_pool_persists_after_reserve(tmp_path):
    pool, path = _pool(tmp_path)
    pool.reserve("NSE:RELIANCE", "MTF", 3.85, date(2026, 5, 21))
    pool.persist()
    # New instance loads same state
    pool2 = OvernightSlotPool(path, max_slots=4, margin_per_slot=100000, max_new_per_day=2)
    active = pool2.active()
    assert len(active) == 1
    assert active[0].symbol == "NSE:RELIANCE"


def test_reserve_returns_none_when_all_slots_occupied(tmp_path):
    pool, _ = _pool(tmp_path, max_new=10)  # raise per-day cap to test slot cap
    for i in range(4):
        s = pool.reserve(f"NSE:SYM{i}", "MTF", 3.0, date(2026, 5, 21))
        assert s is not None
    # 5th should fail
    extra = pool.reserve("NSE:SYM5", "MTF", 3.0, date(2026, 5, 21))
    assert extra is None


def test_max_new_per_day_cap_blocks_third_reserve(tmp_path):
    pool, _ = _pool(tmp_path, max_new=2)
    today = date(2026, 5, 21)
    s1 = pool.reserve("NSE:A", "MTF", 3.0, today)
    s2 = pool.reserve("NSE:B", "MTF", 3.0, today)
    s3 = pool.reserve("NSE:C", "MTF", 3.0, today)  # blocked by per-day cap
    assert s1 is not None and s2 is not None and s3 is None


def test_full_lifecycle_reserve_buy_amo_settle_release(tmp_path):
    pool, _ = _pool(tmp_path)
    today = date(2026, 5, 21)
    tomorrow = date(2026, 5, 22)
    day_after = date(2026, 5, 23)

    slot = pool.reserve("NSE:RELIANCE", "MTF", 3.85, today)
    pool.attach_buy_fill(slot.slot_id, 2500.0, "2026-05-21T15:30:00", "BUY-001")
    slot.gtt_id = "GTT_X"
    pool.attach_amo_sell(slot.slot_id, "AMO-001", tomorrow)
    pool.settle(slot.slot_id, 2515.0, "2026-05-22T09:15:00", fees_inr=500.0, interest_inr=160.0)

    s = pool.active()[0]
    assert s.status == "t1_settling"
    assert s.realized_pnl_inr is not None
    # qty = round(385000 / 2500) = 154; gross = (2515 - 2500) * 154 = 2310
    # net = 2310 - 500 - 160 = 1650
    assert s.realized_pnl_inr == pytest.approx(1650.0, abs=1.0)

    pool.release(slot.slot_id, day_after)
    assert pool.free_count() == 4
    assert pool.active() == []
    assert pool._get_slot(slot.slot_id).gtt_id is None


def test_settle_requires_t0_open_status(tmp_path):
    pool, _ = _pool(tmp_path)
    with pytest.raises(ValueError, match="status is 'free'"):
        pool.settle(1, 2515.0, "ts", 500.0, 160.0)


def test_release_requires_t1_settling_status(tmp_path):
    pool, _ = _pool(tmp_path)
    pool.reserve("NSE:A", "MTF", 3.0, date(2026, 5, 21))
    # Slot is t0_open, not t1_settling
    with pytest.raises(ValueError, match="expected 't1_settling'"):
        pool.release(1, date(2026, 5, 23))


def test_corrupt_state_file_raises_explicit_error(tmp_path):
    path = tmp_path / "overnight_slots.json"
    path.write_text("not valid json {{{", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        OvernightSlotPool(path, max_slots=4, margin_per_slot=100000, max_new_per_day=2)


def test_state_file_wrong_shape_raises(tmp_path):
    path = tmp_path / "overnight_slots.json"
    path.write_text('["not", "a dict"]', encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected shape"):
        OvernightSlotPool(path, max_slots=4, margin_per_slot=100000, max_new_per_day=2)


def test_state_slot_count_shrink_raises(tmp_path):
    # SHRINKING (state has MORE slots than config) is unsafe — could orphan a
    # slot holding an open position — so it must raise for manual migration.
    path = tmp_path / "overnight_slots.json"
    data = {"slots": [{"slot_id": i, "status": "free", "leverage": 1.0,
                       "margin_inr": 0.0, "notional_inr": 0.0} for i in range(1, 6)]}  # 5 slots
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="has 5 slots but config"):
        OvernightSlotPool(path, max_slots=4, margin_per_slot=100000, max_new_per_day=2)


def test_state_fewer_slots_than_config_auto_expands(tmp_path):
    # EXPANDING (state has FEWER slots than config) is safe and intentional:
    # preserve existing slots, append free slots up to max_slots. This is what
    # happens when max_slots is raised in config (e.g. paper->live sizing).
    path = tmp_path / "overnight_slots.json"
    data = {"slots": [{"slot_id": 1, "status": "free", "leverage": 1.0,
                       "margin_inr": 0.0, "notional_inr": 0.0}]}
    path.write_text(json.dumps(data), encoding="utf-8")
    pool = OvernightSlotPool(path, max_slots=4, margin_per_slot=100000, max_new_per_day=2)
    assert len(pool._slots) == 4


def test_slot_roundtrips_gtt_id(tmp_path):
    from services.capital_manager import OvernightSlotPool
    state = tmp_path / "slots.json"
    pool = OvernightSlotPool(state, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool.reserve(symbol="NSE:RELIANCE", product="MTF", leverage=2.5, today=date(2026, 6, 22))
    slot.gtt_id = "GTT_123"
    pool.persist()
    pool2 = OvernightSlotPool(state, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    assert pool2._get_slot(slot.slot_id).gtt_id == "GTT_123"


def test_from_dict_ignores_unknown_legacy_keys():
    from services.capital_manager import OvernightSlot
    d = {"slot_id": 1, "status": "free", "paper_variant_b": None}  # legacy key not on the dataclass
    slot = OvernightSlot.from_dict(d)
    assert slot.slot_id == 1
    assert slot.gtt_id is None
