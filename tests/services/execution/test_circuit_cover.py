"""Circuit cover: a resting stop-limit order at the price-band circuit (2026-10-29).

HUHTAMAKI 2026-10-28: short 514 @ 193.30, the stock locked at its +20% upper circuit
229.57. Our MARKET stop BUY and Zerodha's 15:20 auto square-off were both rejected;
the short went to the short-delivery auction. Sellers kept trading AT 229.57 all
afternoon (~46k shares), filled by time priority. The cover is a STOP-LIMIT: inert
until price reaches the trigger just inside the circuit, then a limit AT the circuit.
(A plain LIMIT BUY at the circuit is marketable and would cover at once.)
"""
import json
import logging
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import services.execution.exit_executor as ee
ee.logger = logging.getLogger("test_circuit_cover")
ee.trade_logger = MagicMock()

from services.execution.exit_executor import ExitExecutor, Position, PositionStore

SYM, UC, TICK = "NSE:HUHTAMAKI", 229.57, 0.05
TID = "NSE:HUHTAMAKI_abc123def456"
TS = pd.Timestamp("2026-10-28 15:10:00")


class FakeKite:
    dry_run = False

    def __init__(self):
        self.placed, self.cancelled = [], []
        self.statuses = {}
        self.book = []                       # raw kc.orders() (other orders at the broker)
        self.net = []                        # raw kc.positions()["net"]
        self.quote = {"last_price": 200.0, "upper_circuit_limit": UC, "lower_circuit_limit": 153.05}
        self.exit_status, self.exit_fill = "COMPLETE", 210.0
        self.cover_status = "TRIGGER PENDING"
        self.kc = SimpleNamespace(orders=lambda: list(self.book),
                                  positions=lambda: {"net": list(self.net)})

    def get_quote(self, symbol):
        return dict(self.quote)

    def get_tick_size(self, symbol):
        return TICK

    def place_order(self, **kw):
        oid = f"O{len(self.placed) + 1}"
        self.placed.append(kw)
        if kw.get("order_type") in ("SL", "LIMIT"):
            st = {"status": self.cover_status, "average_price": 0.0, "filled_quantity": 0}
        else:
            st = {"status": self.exit_status, "average_price": self.exit_fill,
                  "filled_quantity": kw["qty"] if self.exit_status == "COMPLETE" else 0}
        self.statuses[oid] = {"order_id": oid, "status_message": "", **st}
        return oid

    def reconcile_exit(self, symbol, order_id, expected_qty, position_qty_before, timeout):
        o = self.statuses[order_id]
        return {"avg_price": o["average_price"], "remaining_qty": 0} if o["status"] == "COMPLETE" else None

    def get_order_status(self, order_id):
        return dict(self.statuses.get(order_id) or {"order_id": order_id, "status": "UNKNOWN",
                                                     "average_price": 0.0, "filled_quantity": 0})

    def cancel_order(self, order_id, variety="regular"):
        self.cancelled.append(order_id)
        o = self.statuses.get(order_id)
        if o and o["status"] != "COMPLETE":
            o["status"] = "CANCELLED"
        return True

    def get_ltp_with_level(self, symbol, check_level=None):
        return 200.0


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(ee.time, "monotonic", lambda: t["now"])
    return t


def _pos(cover=UC, hard_sl=240.0):
    return Position(symbol=SYM, side="SELL", qty=514, avg_price=193.30,
                    plan={"trade_id": TID, "hard_sl": hard_sl, "stop": {"hard": hard_sl},
                          "targets": [{"level": 175.9, "qty_pct": 1.0}],
                          "exits": {"time_stop_hhmm": "15:10", "circuit_cover_px": cover},
                          "_state": {}})


@pytest.fixture
def live(clock, tmp_path):
    broker, store, cap = FakeKite(), PositionStore(), MagicMock()
    ex = ExitExecutor(broker=broker, positions=store, get_ltp_ts=lambda s: (200.0, TS),
                      capital_manager=cap)
    ex.cc_ledger_path = tmp_path / "ledger.json"
    pos = _pos(); store.upsert(pos)
    return ex, broker, store, cap, pos


def _trigger():
    ticks = math.ceil(UC * 0.001 / TICK - 1e-9)
    return round(UC - ticks * TICK, 2)


# --- placing the cover (live) -----------------------------------------------

def test_cover_is_a_stop_limit_with_its_limit_at_the_exchange_circuit(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    o = broker.placed[0]
    assert (o["side"], o["qty"], o["order_type"], o["price"]) == ("BUY", 514, "SL", UC)
    assert o["trigger_price"] == _trigger() < UC
    assert o["check_margins"] is False
    st = pos.plan["_state"]
    assert st["_circ_order_id"] == "O1" and "O1" in st["_exit_known_order_ids"]


def test_never_a_plain_limit_while_price_is_below_the_trigger(live):
    """A LIMIT BUY at the circuit with price below it fills at once."""
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed[0]["order_type"] == "SL"


def test_price_already_at_the_trigger_gets_a_limit_at_the_circuit(live):
    ex, broker, store, cap, pos = live
    broker.quote["last_price"] = UC
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed[0]["order_type"] == "LIMIT" and broker.placed[0]["price"] == UC


def test_placed_once(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    ex._ensure_circuit_order(SYM, pos)
    assert len(broker.placed) == 1


def test_exchange_circuit_wins_over_the_band_file(live):
    ex, broker, store, cap, pos = live
    pos.plan["exits"]["circuit_cover_px"] = 229.572
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed[0]["price"] == UC


def test_restart_adopts_the_cover_already_at_the_broker(live):
    ex, broker, store, cap, pos = live
    broker.book = [{"order_id": "X9", "tradingsymbol": "HUHTAMAKI", "transaction_type": "BUY",
                    "tag": "ITDA_abc123def456", "order_type": "SL", "status": "TRIGGER PENDING",
                    "price": UC, "product": "MIS", "quantity": 514}]
    broker.statuses["X9"] = {"order_id": "X9", "status": "TRIGGER PENDING", "average_price": 0.0,
                             "filled_quantity": 0}
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed == [] and pos.plan["_state"]["_circ_order_id"] == "X9"


def test_rejected_cover_is_re_placed_after_the_retry_wait(live, clock):
    ex, broker, store, cap, pos = live
    broker.cover_status = "REJECTED"
    ex._ensure_circuit_order(SYM, pos)
    assert "_circ_order_id" not in pos.plan["_state"]
    ex._ensure_circuit_order(SYM, pos)
    assert len(broker.placed) == 1
    broker.cover_status = "TRIGGER PENDING"
    clock["now"] += ex.cc_place_retry_sec
    ex._ensure_circuit_order(SYM, pos)
    assert len(broker.placed) == 2 and pos.plan["_state"]["_circ_order_id"] == "O2"


@pytest.mark.parametrize("prev", ["OPEN", "COMPLETE", "UNKNOWN"])
def test_no_cover_while_an_earlier_exit_is_unsettled(live, prev):
    """Even COMPLETE: the position may already be flat; a cover would buy again."""
    ex, broker, store, cap, pos = live
    broker.statuses["E1"] = {"order_id": "E1", "status": prev, "average_price": 0.0, "filled_quantity": 0}
    pos.plan["_state"].update(_exit_last_order_id="E1", _exit_retry_at=0.0)
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed == []


def test_adoption_refuses_a_wrong_size_order(live):
    ex, broker, store, cap, pos = live
    broker.book = [{"order_id": "X9", "tradingsymbol": "HUHTAMAKI", "transaction_type": "BUY",
                    "tag": "ITDA_abc123def456", "order_type": "SL", "status": "TRIGGER PENDING",
                    "price": UC, "product": "MIS", "quantity": 900}]
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed == [] and "_circ_order_id" not in pos.plan["_state"]


def test_late_fill_of_an_earlier_exit_cancels_the_cover(live, clock):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    st = pos.plan["_state"]
    broker.statuses["E1"] = {"order_id": "E1", "status": "COMPLETE", "average_price": 205.0,
                             "filled_quantity": 514}
    st.update(_exit_last_order_id="E1", _exit_last_qty=514, _exit_retry_at=clock["now"])
    ex._exit(SYM, pos, 205.0, TS, "time_stop_15:10")
    assert broker.cancelled == ["O1"]
    assert store.get_by_trade_id(TID) is None
    assert json.loads(ex.cc_ledger_path.read_text()) == {}


def test_no_cover_for_a_closed_position_or_while_an_exit_is_pending(live):
    ex, broker, store, cap, pos = live
    pos.plan["_state"]["exit_pending"] = True
    ex._ensure_circuit_order(SYM, pos)
    pos.plan["_state"].pop("exit_pending")
    store.close(TID)
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed == []


def test_no_cover_without_a_circuit_price_or_for_shadow(live):
    ex, broker, store, cap, pos = live
    pos.plan["exits"]["circuit_cover_px"] = None
    ex._ensure_circuit_order(SYM, pos)
    pos.plan["exits"]["circuit_cover_px"] = UC
    pos.plan["shadow"] = True
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed == []


# --- exits while the cover rests -------------------------------------------

def test_triggered_cover_at_the_circuit_keeps_its_place_in_the_queue(live):
    """Cancel + resend at a locked circuit would go to the back of the queue."""
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    broker.statuses["O1"]["status"] = "OPEN"               # triggered, queued at the circuit
    ex._exit(SYM, pos, UC, TS, "time_stop_15:10")
    assert broker.cancelled == [] and len(broker.placed) == 1
    assert store.get_by_trade_id(TID) is not None
    cap.exit_position.assert_not_called()


def test_untriggered_cover_is_cancelled_then_the_position_exits(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    ex._exit(SYM, pos, 200.0, TS, "time_stop_15:10")
    assert broker.cancelled == ["O1"]
    assert broker.placed[1]["qty"] == 514 and broker.placed[1]["order_type"] not in ("SL", "LIMIT")
    assert store.get_by_trade_id(TID) is None


def test_cover_partly_filled_then_cancelled_exits_only_the_rest(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    broker.statuses["O1"].update(status="CANCELLED", filled_quantity=200, average_price=UC)
    ex._exit(SYM, pos, 205.0, TS, "time_stop_15:10")
    assert broker.placed[1]["qty"] == 314
    assert store.get_by_trade_id(TID) is None


def test_partial_cover_then_a_rejected_exit_never_re_buys_the_covered_part(live, clock):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    broker.statuses["O1"].update(status="CANCELLED", filled_quantity=200, average_price=UC)
    broker.exit_status = "REJECTED"
    ex._exit(SYM, pos, 205.0, TS, "time_stop_15:10")
    assert store.get_by_trade_id(TID).qty == 314            # covered part booked
    assert pos.plan["_state"]["manual_partial_qty"] == 200
    clock["now"] += ex.exit_retry_cooldown_sec
    broker.exit_status = "COMPLETE"
    ex._exit(SYM, pos, 205.0, TS, "time_stop_15:10")
    assert [o["qty"] for o in broker.placed[1:]] == [314, 314]


def test_completed_cover_is_booked_without_a_new_order(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    broker.statuses["O1"].update(status="COMPLETE", filled_quantity=514, average_price=UC)
    ex._exit(SYM, pos, 200.0, TS, "time_stop_15:10")
    assert len(broker.placed) == 1 and broker.cancelled == []
    assert store.get_by_trade_id(TID) is None
    cap.exit_position.assert_called_once()


def test_unknown_cover_status_holds_without_cancelling(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    broker.statuses["O1"]["status"] = "UNKNOWN"
    ex._exit(SYM, pos, 200.0, TS, "time_stop_15:10")
    assert broker.cancelled == [] and len(broker.placed) == 1
    assert store.get_by_trade_id(TID) is not None


def test_dead_cover_does_not_block_the_exit(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    broker.statuses["O1"]["status"] = "REJECTED"
    ex._exit(SYM, pos, UC * 0.995, TS, "time_stop_15:10")
    assert broker.placed[1]["qty"] == 514
    assert store.get_by_trade_id(TID) is None


def test_partial_exit_refused_while_the_cover_rests(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    assert ex._place_and_log_exit(SYM, pos, 200.0, 100, TS, "manual_partial") is None
    assert broker.cancelled == [] and len(broker.placed) == 1


def test_near_the_circuit_a_filled_cover_closes_the_position(live, clock):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    broker.statuses["O1"]["status"] = "OPEN"
    assert ex._circuit_cover_check(SYM, pos, UC, TS) is False      # queued
    broker.statuses["O1"].update(status="COMPLETE", filled_quantity=514, average_price=UC)
    assert ex._circuit_cover_check(SYM, pos, UC, TS) is False      # poll throttled
    clock["now"] += ex.cc_poll_sec
    assert ex._circuit_cover_check(SYM, pos, UC, TS) is True
    assert store.get_by_trade_id(TID) is None
    assert len(broker.placed) == 1


def test_cover_rejected_later_is_dropped_for_re_placement(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    broker.statuses["O1"]["status"] = "REJECTED"
    assert ex._circuit_cover_check(SYM, pos, UC, TS) is False
    assert "_circ_order_id" not in pos.plan["_state"]


def test_far_from_the_circuit_nothing_is_polled(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    broker.get_order_status = MagicMock(side_effect=AssertionError("polled"))
    assert ex._circuit_cover_check(SYM, pos, 200.0, TS) is False


def test_cover_is_recorded_in_the_ledger_and_removed_when_settled(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    assert json.loads(ex.cc_ledger_path.read_text())["O1"]["trade_id"] == TID
    ex._exit(SYM, pos, 200.0, TS, "time_stop_15:10")
    assert json.loads(ex.cc_ledger_path.read_text()) == {}


def test_sweep_cancels_a_ledger_cover_of_a_closed_position_when_flat(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    store.close(TID)                                   # closed outside the exit path
    assert ex.sweep_orphan_circuit_orders() == 1
    assert broker.cancelled == ["O1"] and json.loads(ex.cc_ledger_path.read_text()) == {}


def test_sweep_keeps_the_cover_of_an_untracked_open_position(live):
    """Cancelling it would leave a real short naked."""
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    store.close(TID)
    broker.net = [{"tradingsymbol": "HUHTAMAKI", "product": "MIS", "quantity": -514}]
    assert ex.sweep_orphan_circuit_orders() == 0 and broker.cancelled == []


def test_sweep_never_touches_orders_it_did_not_place(live):
    ex, broker, store, cap, pos = live
    broker.book = [{"order_id": "Z1", "tradingsymbol": "OTHER", "transaction_type": "BUY",
                    "tag": "ITDA_deadbeef0000", "order_type": "LIMIT", "status": "OPEN", "product": "MIS"}]
    assert ex.sweep_orphan_circuit_orders() == 0 and broker.cancelled == []


def test_sweep_leaves_tracked_positions_alone(live):
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    assert ex.sweep_orphan_circuit_orders() == 0 and broker.cancelled == []


# --- paper / backtest: the same order, simulated ----------------------------

class DryBroker:
    dry_run = True

    def __init__(self, touched):
        self.placed, self.touched = [], touched

    def place_order(self, **kw):
        self.placed.append(kw); return "dry"

    def get_ltp_with_level(self, symbol, check_level=None):
        return check_level if check_level in self.touched else 200.0


@pytest.fixture(autouse=True)
def _runtime_capital():
    from services.plan_orchestrator import set_runtime_capital
    set_runtime_capital(100_000.0)       # main.py does this at startup; the daily stop reads it


def _dry(clock, touched=(), hard_sl=240.0):
    store, cap = PositionStore(), MagicMock()
    broker = DryBroker(set(touched))
    ex = ExitExecutor(broker=broker, positions=store, get_ltp_ts=lambda s: (UC, TS), capital_manager=cap)
    pos = _pos(cover=229.572, hard_sl=hard_sl); store.upsert(pos)
    return ex, broker, store, pos


def test_paper_tick_at_the_trigger_covers_at_the_circuit(clock):
    ex, broker, store, pos = _dry(clock)
    assert ex._circuit_cover_check(SYM, pos, 229.40, TS) is True    # >= 229.572 x 0.999
    assert store.get_by_trade_id(TID) is None


def test_paper_below_the_trigger_does_nothing(clock):
    ex, broker, store, pos = _dry(clock)
    assert ex._circuit_cover_check(SYM, pos, 228.0, TS) is False


def test_backtest_bar_touching_the_trigger_covers(clock):
    ex, broker, store, pos = _dry(clock, touched={229.572 * 0.999})
    ex.run_once()
    assert store.get_by_trade_id(TID) is None


def test_backtest_stop_inside_the_circuit_is_hit_first(clock):
    """9% stop 210.70 below the circuit: price passes the stop first."""
    ex, broker, store, pos = _dry(clock, touched={210.70, 229.572 * 0.999}, hard_sl=210.70)
    seen = []
    ex._exit = lambda s, p, px, ts, reason: seen.append(reason)
    ex.run_once()
    assert seen and seen[0] != "circuit_cover"


def test_dry_run_never_rests_a_real_order(clock):
    ex, broker, store, pos = _dry(clock)
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed == []


# --- config / broker --------------------------------------------------------

def test_up_spike_rests_the_cover_and_config_is_complete():
    import io, json
    from pathlib import Path
    cfg = json.load(io.open(Path(__file__).resolve().parents[3] / "config" / "configuration.json",
                            encoding="utf-8"))
    assert cfg["setups"]["up_spike_fade_short"]["circuit_cover_order"] is True
    assert "circuit_risk_budget_pct_of_capital" in cfg["setups"]["up_spike_fade_short"]
    for k in ("near_circuit_pct", "status_poll_sec", "place_retry_sec",
              "quote_mismatch_warn_pct", "trigger_offset_pct"):
        assert k in cfg["circuit_cover"]


def test_kite_broker_sends_stop_limit_with_trigger():
    from broker.kite.kite_broker import KiteBroker
    kb = KiteBroker.__new__(KiteBroker)
    kb.dry_run, kb._market_protection = False, -1
    sent = {}
    kb.kc = SimpleNamespace(TRANSACTION_TYPE_BUY="BUY", TRANSACTION_TYPE_SELL="SELL",
                            ORDER_TYPE_LIMIT="LIMIT", ORDER_TYPE_MARKET="MARKET", ORDER_TYPE_SL="SL",
                            PRODUCT_MIS="MIS", PRODUCT_CNC="CNC", VARIETY_REGULAR="regular",
                            VARIETY_AMO="amo")
    kb._kc_place = lambda variety, params: sent.update(params) or "OID"
    assert kb.place_order(symbol=SYM, side="BUY", qty=5, order_type="SL", price=UC,
                          trigger_price=229.32, check_margins=False) == "OID"
    assert (sent["order_type"], sent["price"], sent["trigger_price"]) == ("SL", UC, 229.32)
    assert "market_protection" not in sent
    with pytest.raises(ValueError):
        kb.place_order(symbol=SYM, side="BUY", qty=5, order_type="SL", price=UC, check_margins=False)


def test_refused_partial_does_not_block_re_placing_a_dead_cover(live, clock):
    """N1: the refusal arms the retry marker without sending any order; once the
    cover dies, it must be re-placed, not blocked by that marker forever."""
    ex, broker, store, cap, pos = live
    ex._ensure_circuit_order(SYM, pos)
    assert ex._place_and_log_exit(SYM, pos, 200.0, 100, TS, "t1_partial") is None
    assert pos.plan["_state"].get("_exit_retry_at") is not None
    broker.statuses["O1"]["status"] = "REJECTED"
    ex._circuit_order_dead(SYM, pos.plan["_state"], "O1", broker.statuses["O1"])
    clock["now"] += max(ex.exit_retry_cooldown_sec, ex.cc_place_retry_sec)
    ex._ensure_circuit_order(SYM, pos)
    assert pos.plan["_state"]["_circ_order_id"] == "O2"


def test_cover_not_re_placed_while_a_failed_exit_order_may_still_fill(live, clock):
    ex, broker, store, cap, pos = live
    broker.statuses["E1"] = {"order_id": "E1", "status": "OPEN", "average_price": 0.0, "filled_quantity": 0}
    pos.plan["_state"].update(_exit_last_order_id="E1", _exit_last_qty=514, _exit_retry_at=clock["now"])
    clock["now"] += 1
    ex._ensure_circuit_order(SYM, pos)
    assert broker.placed == []


def test_corrupt_ledger_is_quarantined_not_fatal(live):
    ex, broker, store, cap, pos = live
    ex.cc_ledger_path.write_text("{not json", encoding="utf-8")
    ex._ensure_circuit_order(SYM, pos)
    assert json.loads(ex.cc_ledger_path.read_text())["O1"]["trade_id"] == TID
    assert list(ex.cc_ledger_path.parent.glob("*.corrupt-*"))
