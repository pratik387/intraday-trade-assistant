"""An exit order that does not fill must never be booked as filled (2026-10-29).

HUHTAMAKI 2026-10-28: the live SL BUY was REJECTED with the stock locked at its
upper circuit. _place_and_log_exit swallowed the error, booked the exit at the
SL price, released the margin and stopped tracking a short still open at Zerodha.
"""
import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

import services.execution.exit_executor as ee
ee.logger = logging.getLogger("test_exit_not_filled")
ee.trade_logger = MagicMock()

from services.execution.exit_executor import ExitExecutor, Position, PositionStore

SYM = "NSE:HUHTAMAKI"


class FakeKite:
    """Live-shaped broker: every order is answered from `statuses`."""
    dry_run = False

    def __init__(self):
        self.placed = []
        self.cancelled = []
        self.statuses = {}          # order_id -> status dict
        self.next_status = "REJECTED"
        self.next_fill = 0.0
        self.open_qty = 514         # broker day position (abs)

    def place_order(self, **kw):
        oid = f"O{len(self.placed) + 1}"
        self.placed.append(kw)
        self.statuses[oid] = {"order_id": oid, "status": self.next_status,
                              "average_price": self.next_fill, "status_message": "circuit"}
        return oid

    def reconcile_exit(self, symbol, order_id, expected_qty, position_qty_before, timeout):
        o = self.statuses[order_id]
        if o["status"] == "COMPLETE":
            return {"avg_price": o["average_price"], "remaining_qty": 0}
        return None

    def get_order_status(self, order_id):
        return dict(self.statuses[order_id])

    def cancel_order(self, order_id, variety="regular"):
        self.cancelled.append(order_id)
        return True

    def get_positions(self):
        return {"day": [{"tradingsymbol": "HUHTAMAKI", "product": "MIS", "quantity": -self.open_qty}]}


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(ee.time, "monotonic", lambda: t["now"])
    return t


@pytest.fixture
def setup(clock):
    broker = FakeKite()
    store = PositionStore()
    cap = MagicMock()
    ex = ExitExecutor(broker=broker, positions=store, get_ltp_ts=lambda s: (229.57, None),
                      capital_manager=cap)
    pos = Position(symbol=SYM, side="SELL", qty=514, avg_price=193.30,
                   plan={"trade_id": "T1", "hard_sl": 223.47, "stop": {"hard": 223.47},
                         "targets": [{"level": 180.0}, {"level": 170.0}], "_state": {}})
    store.upsert(pos)
    return ex, broker, store, cap, pos


TS = pd.Timestamp("2026-10-28 13:40:11")


def test_rejected_exit_keeps_the_position_open_and_margin_blocked(setup):
    ex, broker, store, cap, pos = setup
    pos.plan["_state"]["exit_pending"] = True       # what the tick SL path sets
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert len(broker.placed) == 1
    assert store.get_by_trade_id("T1") is not None
    assert store.get_by_trade_id("T1").qty == 514
    cap.exit_position.assert_not_called()
    assert "exit_pending" not in pos.plan["_state"]  # so the next tick retries


def test_no_second_order_inside_the_cooldown(setup, clock):
    ex, broker, store, cap, pos = setup
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    clock["now"] += ex.exit_retry_cooldown_sec - 1
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert len(broker.placed) == 1


def test_retry_after_the_cooldown_closes_on_fill(setup, clock):
    ex, broker, store, cap, pos = setup
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    clock["now"] += ex.exit_retry_cooldown_sec
    broker.next_status, broker.next_fill = "COMPLETE", 229.57
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert len(broker.placed) == 2
    assert store.get_by_trade_id("T1") is None
    cap.exit_position.assert_called_once()


def test_late_fill_is_booked_from_the_earlier_order_without_a_new_one(setup, clock):
    ex, broker, store, cap, pos = setup
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    broker.statuses["O1"].update(status="COMPLETE", average_price=229.10)
    clock["now"] += ex.exit_retry_cooldown_sec
    assert ex._place_and_log_exit(SYM, pos, 229.57, 514, TS, "hard_sl") == 229.10
    assert len(broker.placed) == 1


def test_positions_api_failure_cannot_book_an_open_short(setup, clock):
    """Kite returns an empty book when positions() fails; that must not read as flat."""
    ex, broker, store, cap, pos = setup
    broker.get_positions = lambda: {"day": [], "net": []}
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    clock["now"] += ex.exit_retry_cooldown_sec
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")           # O1 REJECTED -> dead -> retry
    assert len(broker.placed) == 2
    assert store.get_by_trade_id("T1") is not None      # O2 also rejected: still open
    cap.exit_position.assert_not_called()


@pytest.mark.parametrize("status", ["OPEN", "UNKNOWN", "TRIGGER PENDING"])
def test_no_new_order_while_the_previous_one_may_still_fill(setup, clock, status):
    ex, broker, store, cap, pos = setup
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    broker.statuses["O1"]["status"] = status            # the cancel did not take
    clock["now"] += ex.exit_retry_cooldown_sec
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert len(broker.placed) == 1
    assert broker.cancelled == ["O1"]
    assert store.get_by_trade_id("T1") is not None


def test_late_fill_of_a_different_qty_sends_nothing(setup, clock):
    ex, broker, store, cap, pos = setup
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    broker.statuses["O1"].update(status="COMPLETE", average_price=229.10)
    clock["now"] += ex.exit_retry_cooldown_sec
    assert ex._place_and_log_exit(SYM, pos, 229.57, 300, TS, "t1_partial") is None
    assert len(broker.placed) == 1


def test_one_exit_per_position_at_a_time(setup):
    ex, broker, store, cap, pos = setup
    lock = ex._exit_lock("T1")
    lock.acquire()
    try:
        assert ex._place_and_log_exit(SYM, pos, 229.57, 514, TS, "hard_sl") is None
    finally:
        lock.release()
    assert broker.placed == []


def test_already_closed_position_is_not_exited_again(setup):
    ex, broker, store, cap, pos = setup
    store.close("T1")
    assert ex._place_and_log_exit(SYM, pos, 229.57, 514, TS, "hard_sl") is None
    assert broker.placed == []


def test_missing_order_id_is_not_a_fill(setup):
    ex, broker, store, cap, pos = setup
    broker.place_order = lambda **kw: ""
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert store.get_by_trade_id("T1") is not None


def test_exit_orders_skip_the_mis_margin_check(setup):
    ex, broker, store, cap, pos = setup
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert broker.placed[0]["check_margins"] is False


def test_open_order_is_cancelled_and_a_raced_fill_counts(setup):
    ex, broker, store, cap, pos = setup
    broker.next_status = "OPEN"
    orig = broker.cancel_order

    def cancel_but_it_filled(order_id, variety="regular"):
        broker.statuses[order_id].update(status="COMPLETE", average_price=229.40)
        return orig(order_id, variety)
    broker.cancel_order = cancel_but_it_filled
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert broker.cancelled == ["O1"]
    assert store.get_by_trade_id("T1") is None


def test_open_order_cancelled_stays_open(setup):
    ex, broker, store, cap, pos = setup
    broker.next_status = "OPEN"
    orig = broker.cancel_order

    def cancel(order_id, variety="regular"):
        broker.statuses[order_id].update(status="CANCELLED")
        return orig(order_id, variety)
    broker.cancel_order = cancel
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert store.get_by_trade_id("T1") is not None


def test_failed_t1_partial_is_not_marked_booked(setup):
    ex, broker, store, cap, pos = setup
    pos.plan["risk_per_share"] = 10.0
    stop_before = dict(pos.plan["stop"])
    ex._partial_exit_t1(SYM, pos, 180.0, TS)
    assert len(broker.placed) == 1
    st = pos.plan["_state"]
    assert not st.get("t1_done") and not st.get("_t1_processing")
    assert store.get_by_trade_id("T1").qty == 514
    assert pos.plan["stop"] == stop_before and pos.plan["hard_sl"] == 223.47
    assert st["_exit_last_order_id"] == "O1"           # retry markers survive the rollback


def test_t1_rollback_keeps_concurrent_state_updates(setup):
    ex, broker, store, cap, pos = setup
    pos.plan["risk_per_share"] = 10.0
    st = pos.plan["_state"]
    st["mae"] = -3.0
    ex._partial_exit_t1(SYM, pos, 180.0, TS)
    assert pos.plan["_state"] is st and st["mae"] == -3.0


def test_cooldown_comes_from_config(setup):
    ex = setup[0]
    from config.filters_setup import load_filters
    assert ex.exit_retry_cooldown_sec == float(load_filters()["exit_retry_cooldown_sec"])


def test_cancelled_after_a_partial_fill_never_resends_the_full_qty(setup, clock):
    ex, broker, store, cap, pos = setup
    broker.next_status = "OPEN"
    orig = broker.cancel_order

    def cancel_after_partial(order_id, variety="regular"):
        broker.statuses[order_id].update(status="CANCELLED", filled_quantity=300)
        return orig(order_id, variety)
    broker.cancel_order = cancel_after_partial
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    clock["now"] += ex.exit_retry_cooldown_sec
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert len(broker.placed) == 1
    assert store.get_by_trade_id("T1") is not None


def test_reconcile_error_is_not_a_fill(setup):
    ex, broker, store, cap, pos = setup

    def boom(**kw):
        raise ConnectionError("orders() timed out")
    broker.reconcile_exit = boom
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")           # status REJECTED on the re-read
    assert store.get_by_trade_id("T1") is not None
    cap.exit_position.assert_not_called()


def test_placement_error_that_reached_the_exchange_is_found_not_resent(setup, clock):
    ex, broker, store, cap, pos = setup
    real_place = broker.place_order

    def place_then_time_out(**kw):
        real_place(**kw)                                 # the order did go through
        raise TimeoutError("read timeout")
    broker.place_order = place_then_time_out
    broker.next_status = "OPEN"
    broker.get_orders = lambda: [{"order_id": "O1", "tradingsymbol": "HUHTAMAKI", "transaction_type": "BUY",
                                  "tag": "ITDA_T1"}]
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    clock["now"] += ex.exit_retry_cooldown_sec
    broker.place_order = real_place
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")           # O1 still OPEN -> hold, re-cancel
    assert len(broker.placed) == 1
    assert broker.cancelled == ["O1"]


def test_placement_error_with_no_order_at_the_exchange_retries(setup, clock):
    ex, broker, store, cap, pos = setup

    def refused(**kw):
        raise RuntimeError("Kite: insufficient permissions")
    real_place = broker.place_order
    broker.place_order = refused
    broker.get_orders = lambda: []
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    clock["now"] += ex.exit_retry_cooldown_sec
    broker.place_order = real_place
    broker.next_status, broker.next_fill = "COMPLETE", 229.57
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert store.get_by_trade_id("T1") is None


def test_dry_run_has_no_cooldown(setup):
    ex, broker, store, cap, pos = setup
    broker.dry_run = True
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    broker.next_status, broker.next_fill = "COMPLETE", 229.57
    ex._exit(SYM, pos, 229.57, TS, "hard_sl")
    assert len(broker.placed) == 2
    assert store.get_by_trade_id("T1") is None
