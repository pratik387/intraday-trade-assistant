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


def test_release_archives_trade_into_released_trades(tmp_path):
    """Released trades must be archived BEFORE slot fields are wiped (Bug fix 2026-05-23).

    Without this archive, every released slot is invisible to downstream PnL
    aggregators — historical paper-trade lookups + backtest harness see 0
    trades. This test locks the archive behavior.
    """
    pool, _ = _pool(tmp_path)
    pool.reserve("NSE:TEST", "MTF", 3.0, date(2026, 5, 21), paper_variant_b=True)
    pool.attach_buy_fill(1, fill_price=100.0, fill_ts_iso="2026-05-21T15:25", order_id="buy-1")
    pool.attach_amo_sell(1, amo_order_id="sell-1", expected_exit_date=date(2026, 5, 22))
    pool.settle(1, sell_fill_price=102.0, sell_fill_ts_iso="2026-05-22T09:15",
                fees_inr=50.0, interest_inr=40.0)
    # Snapshot the settled slot before release
    slot_pre = pool._get_slot(1)
    assert slot_pre.status == "t1_settling"
    assert slot_pre.sell_fill_price == 102.0
    assert slot_pre.realized_pnl_inr is not None

    pool.release(1, cash_back_date=date(2026, 5, 23))

    # Slot is now "free" with all fields wiped
    slot_post = pool._get_slot(1)
    assert slot_post.status == "free"
    assert slot_post.sell_fill_price is None
    assert slot_post.realized_pnl_inr is None
    assert slot_post.paper_variant_b is None

    # But the archive has the trade
    assert len(pool._released_trades) == 1
    archived = pool._released_trades[0]
    assert archived["symbol"] == "NSE:TEST"
    assert archived["product"] == "MTF"
    assert archived["buy_fill_price"] == 100.0
    assert archived["sell_fill_price"] == 102.0
    assert archived["fees_inr"] == 50.0
    assert archived["interest_inr"] == 40.0
    assert archived["realized_pnl_inr"] is not None
    assert archived["paper_variant_b"] is True
    assert archived["cash_back_date"] == "2026-05-23"


def test_released_trades_persist_across_reloads(tmp_path):
    """Archive must round-trip through state file save/load."""
    pool, path = _pool(tmp_path)
    pool.reserve("NSE:X", "CNC", 1.0, date(2026, 5, 21))
    pool.attach_buy_fill(1, fill_price=50.0, fill_ts_iso="2026-05-21T15:25", order_id="b")
    pool.attach_amo_sell(1, amo_order_id="s", expected_exit_date=date(2026, 5, 22))
    pool.settle(1, sell_fill_price=48.0, sell_fill_ts_iso="2026-05-22T09:15",
                fees_inr=30.0, interest_inr=0.0)
    pool.release(1, cash_back_date=date(2026, 5, 23))
    pool.persist()

    # Reload from disk — archive should survive
    pool2 = OvernightSlotPool(path, max_slots=4, margin_per_slot=100000, max_new_per_day=2)
    assert len(pool2._released_trades) == 1
    assert pool2._released_trades[0]["symbol"] == "NSE:X"
    assert pool2._released_trades[0]["sell_fill_price"] == 48.0


def test_multiple_releases_accumulate_in_archive(tmp_path):
    """Multiple release calls must all be archived (not overwrite each other)."""
    pool, _ = _pool(tmp_path, max_new=3)  # raise per-day cap so 3 reserves fit
    for i, sym in enumerate(["NSE:A", "NSE:B", "NSE:C"], start=1):
        slot = pool.reserve(sym, "MTF", 2.0, date(2026, 5, 21))
        assert slot is not None
        pool.attach_buy_fill(slot.slot_id, fill_price=100.0, fill_ts_iso="ts", order_id=f"b{i}")
        pool.attach_amo_sell(slot.slot_id, amo_order_id=f"s{i}", expected_exit_date=date(2026, 5, 22))
        pool.settle(slot.slot_id, sell_fill_price=101.0, sell_fill_ts_iso="ts2",
                    fees_inr=10.0, interest_inr=5.0)
        pool.release(slot.slot_id, cash_back_date=date(2026, 5, 23))
    assert len(pool._released_trades) == 3
    symbols = [t["symbol"] for t in pool._released_trades]
    assert sorted(symbols) == ["NSE:A", "NSE:B", "NSE:C"]


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


def test_state_slot_count_mismatch_raises(tmp_path):
    path = tmp_path / "overnight_slots.json"
    data = {"slots": [{"slot_id": 1, "status": "free", "leverage": 1.0,
                       "margin_inr": 0.0, "notional_inr": 0.0}]}
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="has 1 slots but config"):
        OvernightSlotPool(path, max_slots=4, margin_per_slot=100000, max_new_per_day=2)
