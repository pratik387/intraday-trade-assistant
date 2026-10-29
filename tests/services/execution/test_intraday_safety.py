"""Intraday real-money safety (2026-10-29 audit, build item 1).

B3 restart: a failed broker read must never 'close' every persisted position.
B4 exit:    a flipped or unreadable broker position is never read as a clean exit.
B5 entry:   a live entry is settled from the order status; OPEN is cancelled, a
            partial fill is the position, a timed-out placement is looked up.
B1 EOD:     a failed exit keeps being retried past 15:15 instead of 12 instant tries;
            a partly filled exit books its fill and retries only the rest.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import services.execution.exit_executor as ee
ee.logger = logging.getLogger("test_intraday_safety")
ee.trade_logger = MagicMock()

from services.execution.exit_executor import ExitExecutor, Position, PositionStore
from services.state.broker_reconciliation import BrokerReconciliation

TS = pd.Timestamp("2026-10-29 15:15:00")


# --- B3: recovery ------------------------------------------------------------

def _pp(sym, side, qty):
    return SimpleNamespace(symbol=sym, side=side, qty=qty, avg_price=100.0, state={})


class ReconBroker:
    def __init__(self, day=None, fail=False, app_syms=()):
        self.day, self.fail, self.app_syms = day or [], fail, app_syms

    def get_positions(self, strict=False):
        if self.fail:
            if strict:
                raise RuntimeError("token dead")
            return {"day": [], "net": []}
        return {"day": self.day, "net": []}

    def get_app_orders(self):
        return [{"exchange": "NSE", "tradingsymbol": s.split(":")[1]} for s in self.app_syms]


def test_failed_broker_read_keeps_every_persisted_position():
    r = BrokerReconciliation(ReconBroker(fail=True)).reconcile({"NSE:A": _pp("NSE:A", "SELL", 10)})
    assert "NSE:A" in r.matched and not r.orphaned_app


def test_cnc_line_in_the_same_stock_is_not_the_intraday_position():
    day = [{"tradingsymbol": "A", "exchange": "NSE", "quantity": 50, "product": "CNC", "average_price": 1}]
    r = BrokerReconciliation(ReconBroker(day)).reconcile({"NSE:A": _pp("NSE:A", "SELL", 10)})
    assert "NSE:A" in r.orphaned_app and "NSE:A" not in r.matched


def test_opposite_side_at_the_broker_is_not_restored():
    day = [{"tradingsymbol": "A", "exchange": "NSE", "quantity": 53, "product": "MIS", "average_price": 1}]
    r = BrokerReconciliation(ReconBroker(day)).reconcile({"NSE:A": _pp("NSE:A", "SELL", 53)})
    assert "NSE:A" in r.side_mismatch and "NSE:A" not in r.matched


def test_same_side_short_matches():
    day = [{"tradingsymbol": "A", "exchange": "NSE", "quantity": -10, "product": "MIS", "average_price": 1}]
    r = BrokerReconciliation(ReconBroker(day)).reconcile({"NSE:A": _pp("NSE:A", "SELL", 10)})
    assert "NSE:A" in r.matched


def test_app_position_with_no_record_is_flagged_untracked():
    day = [{"tradingsymbol": "B", "exchange": "NSE", "quantity": -514, "product": "MIS", "average_price": 1}]
    r = BrokerReconciliation(ReconBroker(day, app_syms=("NSE:B",))).reconcile({})
    assert "NSE:B" in r.untracked_app


# --- B5: live entry settle ---------------------------------------------------

class EntryBroker:
    dry_run = False

    def __init__(self, statuses, after_cancel=None, book=None):
        self.statuses, self.after_cancel = list(statuses), after_cancel
        self.cancelled = []
        self.kc = SimpleNamespace(orders=lambda: list(book or []))

    def get_order_status(self, oid):
        return dict(self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0])

    def cancel_order(self, oid, variety="regular"):
        self.cancelled.append(oid)
        if self.after_cancel:
            self.statuses = [self.after_cancel]
        return True


def _tae(broker, monkeypatch):
    from services.execution.trigger_aware_executor import TriggerAwareExecutor
    import services.execution.trigger_aware_executor as tae
    monkeypatch.setattr(tae.time, "sleep", lambda s: None)
    t = TriggerAwareExecutor.__new__(TriggerAwareExecutor)
    t.broker, t.entry_settle_retries, t.entry_settle_delay_sec = broker, 3, 0.0
    return t


def _o(status, filled=0, avg=0.0):
    return {"status": status, "filled_quantity": filled, "average_price": avg, "status_message": ""}


def test_complete_entry_returns_its_fill(monkeypatch):
    t = _tae(EntryBroker([_o("COMPLETE", 514, 193.32)]), monkeypatch)
    assert t._settle_live_entry("NSE:H", "E1", 514) == (514, 193.32)


def test_open_entry_is_cancelled_and_a_partial_fill_is_the_position(monkeypatch):
    b = EntryBroker([_o("OPEN")], after_cancel=_o("CANCELLED", 200, 100.5))
    t = _tae(b, monkeypatch)
    assert t._settle_live_entry("NSE:X", "E1", 500) == (200, 100.5)
    assert b.cancelled == ["E1"]


def test_entry_that_filled_nothing_is_not_a_trade(monkeypatch):
    t = _tae(EntryBroker([_o("REJECTED")]), monkeypatch)
    assert t._settle_live_entry("NSE:X", "E1", 500) is None


def test_entry_still_unknown_after_cancel_raises(monkeypatch):
    b = EntryBroker([_o("UNKNOWN")], after_cancel=_o("UNKNOWN"))
    t = _tae(b, monkeypatch)
    with pytest.raises(RuntimeError, match="unknown"):
        t._settle_live_entry("NSE:X", "E1", 500)


def test_timed_out_placement_finds_the_order_that_reached_nse(monkeypatch):
    book = [{"order_id": "G1", "tradingsymbol": "X", "transaction_type": "SELL", "tag": "ITDA_abc123def456"}]
    t = _tae(EntryBroker([_o("COMPLETE")], book=book), monkeypatch)
    assert t._find_unacknowledged_entry("NSE:X", "SELL", "NSE:X_abc123def456") == "G1"
    assert t._find_unacknowledged_entry("NSE:X", "BUY", "NSE:X_abc123def456") is None


# --- B1: partial fills and the EOD sweep -------------------------------------

class ExitBroker:
    """Live-shaped: exit orders answered from a script of statuses."""
    dry_run = False

    def __init__(self):
        self.placed, self.cancelled, self.statuses = [], [], {}
        self.script = []          # statuses for successive exit orders
        self.open_at_broker = {"A": -500, "B": -100}

    def place_order(self, **kw):
        oid = f"X{len(self.placed) + 1}"
        self.placed.append(kw)
        st = self.script.pop(0) if self.script else _o("COMPLETE", kw["qty"], 205.0)
        self.statuses[oid] = {"order_id": oid, **st}
        return oid

    def reconcile_exit(self, symbol, order_id, expected_qty, position_qty_before, timeout, position_side=None):
        o = self.statuses[order_id]
        return {"avg_price": o["average_price"], "remaining_qty": 0} if o["status"] == "COMPLETE" else None

    def get_order_status(self, oid):
        return dict(self.statuses[oid])

    def cancel_order(self, oid, variety="regular"):
        self.cancelled.append(oid); return True

    def get_ltp(self, sym):
        return 205.0

    def get_positions(self, strict=False):
        return {"day": [{"tradingsymbol": t, "product": "MIS", "quantity": q} for t, q in self.open_at_broker.items()],
                "net": []}


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(ee.time, "monotonic", lambda: t["now"])
    monkeypatch.setattr(ee.time, "sleep", lambda s: t.__setitem__("now", t["now"] + s))
    return t


def _ex(broker):
    store, cap = PositionStore(), MagicMock()
    ex = ExitExecutor(broker=broker, positions=store, get_ltp_ts=lambda s: (205.0, TS), capital_manager=cap)
    return ex, store, cap


def _short(tid, sym, qty=500):
    return Position(symbol=sym, side="SELL", qty=qty, avg_price=200.0,
                    plan={"trade_id": tid, "hard_sl": 250.0, "stop": {"hard": 250.0}, "_state": {}})


def test_partly_filled_exit_books_its_fill_and_retries_only_the_rest(clock):
    b = ExitBroker()
    b.script = [_o("CANCELLED", 300, 204.0)]                     # first exit: 300 of 500, then cancelled
    ex, store, cap = _ex(b)
    pos = _short("T1", "NSE:A"); store.upsert(pos)
    ex._exit("NSE:A", pos, 205.0, TS, "time_stop")
    assert store.get_by_trade_id("T1") is not None
    clock["now"] += ex.exit_retry_cooldown_sec
    ex._exit("NSE:A", pos, 205.0, TS, "time_stop")
    assert [o["qty"] for o in b.placed] == [500, 200]
    assert store.get_by_trade_id("T1") is None
    assert pos.plan["_state"]["manual_partial_qty"] == 300


def test_eod_sweep_keeps_retrying_through_the_cooldown(clock):
    b = ExitBroker()
    b.script = [_o("REJECTED"), _o("REJECTED")]                  # locked twice, then it fills
    ex, store, cap = _ex(b)
    pos = _short("T1", "NSE:A"); store.upsert(pos)
    ex.square_off_all_open_positions(eod=True)
    assert store.get_by_trade_id("T1") is None
    assert len(b.placed) == 3


def test_eod_sweep_does_not_let_one_stuck_position_block_another(clock):
    b = ExitBroker()
    b.script = [_o("REJECTED"), _o("COMPLETE", 100, 205.0)] + [_o("REJECTED")] * 200
    ex, store, cap = _ex(b)
    stuck, ok = _short("T1", "NSE:A"), _short("T2", "NSE:B", 100)
    store.upsert(stuck); store.upsert(ok)
    ex.square_off_all_open_positions(eod=True)
    assert store.get_by_trade_id("T2") is None                    # closed in the first round
    assert store.get_by_trade_id("T1") is not None                # still tracked, not dropped
    assert clock["now"] - 1000.0 >= ex.eod_flatten_max_wait_sec    # kept trying to the deadline


def test_config_keys_present():
    import io, json
    from pathlib import Path
    cfg = json.load(io.open(Path(__file__).resolve().parents[3] / "config" / "configuration.json", encoding="utf-8"))
    for k in ("eod_flatten_max_wait_sec", "eod_flatten_poll_sec", "entry_settle_retries", "entry_settle_delay_sec"):
        assert k in cfg


def test_restart_after_the_broker_squared_off_sends_nothing(clock):
    """4a: a sweep that starts after 15:25 must not BUY into an already-flat short."""
    b = ExitBroker()
    b.open_at_broker = {}
    ex, store, cap = _ex(b)
    store.upsert(_short("T1", "NSE:A"))
    ex.square_off_all_open_positions(eod=True)
    assert b.placed == [] and store.get_by_trade_id("T1") is None


def test_broker_auto_square_off_is_not_double_traded(clock):
    """15:25: Zerodha flattened it; another BUY would open a long."""
    b = ExitBroker()
    b.script = [_o("REJECTED")] * 50
    ex, store, cap = _ex(b)
    pos = _short("T1", "NSE:A"); store.upsert(pos)
    ex.square_off_all_open_positions(eod=True)                 # round 1 rejected
    assert store.get_by_trade_id("T1") is not None             # still short at the broker: kept
    b.open_at_broker = {}                                       # the broker squared it off
    n = len(b.placed)
    ex.square_off_all_open_positions(eod=True)
    assert store.get_by_trade_id("T1") is None
    assert len(b.placed) <= n + 1                               # at most the first round's order, then none
    cap.exit_position.assert_called()


def test_unreadable_positions_send_nothing_more(clock):
    b = ExitBroker()
    b.script = [_o("REJECTED")] * 50
    b.get_positions = lambda strict=False: (_ for _ in ()).throw(RuntimeError("down"))
    ex, store, cap = _ex(b)
    pos = _short("T1", "NSE:A"); store.upsert(pos)
    ex.square_off_all_open_positions(eod=True)
    # Blind sends could buy into a short the broker already squared off at 15:25.
    assert len(b.placed) == 0 and store.get_by_trade_id("T1") is not None


def test_mid_session_stop_does_not_wait_out_the_eod_window(clock):
    b = ExitBroker()
    b.script = [_o("REJECTED")] * 50
    ex, store, cap = _ex(b)
    store.upsert(_short("T1", "NSE:A"))
    ex.square_off_all_open_positions(eod=False)
    assert clock["now"] - 1000.0 < ex.eod_flatten_poll_sec + 1
    assert len(b.placed) == 1


def test_flipped_exit_books_the_fill_and_blocks_the_symbol(clock):
    from services.execution import symbol_blocks
    symbol_blocks.clear_all()
    b = ExitBroker()
    b.reconcile_exit = lambda **kw: {"avg_price": 205.0, "remaining_qty": 0, "verified": False,
                                     "flipped": True, "broker_qty": 500}
    ex, store, cap = _ex(b)
    pos = _short("T1", "NSE:A"); store.upsert(pos)
    ex._exit("NSE:A", pos, 205.0, TS, "time_stop")
    assert store.get_by_trade_id("T1") is None and len(b.placed) == 1   # no second buy
    assert symbol_blocks.blocked_reason("NSE:A")
    symbol_blocks.clear_all()


def test_blocked_symbol_gets_no_new_entry():
    from services.execution import symbol_blocks
    from services.execution.trigger_aware_executor import TriggerAwareExecutor
    symbol_blocks.clear_all()
    symbol_blocks.block("NSE:A", "test")
    t = TriggerAwareExecutor.__new__(TriggerAwareExecutor)
    t.staleness_seconds = 1e9
    t._get_current_time = lambda: pd.Timestamp("2026-10-29 10:00")
    trade = SimpleNamespace(symbol="NSE:A", trigger_timestamp=None, plan={}, state=None)
    assert t._final_execution_check(trade) is False
    symbol_blocks.clear_all()


def test_network_failure_is_uncertain_not_rejected():
    from kiteconnect.exceptions import NetworkException, InputException
    from broker.kite.kite_broker import KiteBroker, OrderPlacementUncertain
    kb = KiteBroker.__new__(KiteBroker)
    kb.dry_run, kb._market_protection = False, -1
    kb.kc = SimpleNamespace(TRANSACTION_TYPE_BUY="BUY", TRANSACTION_TYPE_SELL="SELL", ORDER_TYPE_LIMIT="LIMIT",
                            ORDER_TYPE_MARKET="MARKET", ORDER_TYPE_SL="SL", PRODUCT_MIS="MIS", PRODUCT_CNC="CNC",
                            VARIETY_REGULAR="regular", VARIETY_AMO="amo")
    def boom(exc):
        def f(v, p):
            raise exc
        return f
    kb._kc_place = boom(NetworkException("timed out"))
    with pytest.raises(OrderPlacementUncertain):
        kb.place_order(symbol="NSE:A", side="SELL", qty=1, check_margins=False)
    kb._kc_place = boom(InputException("bad price"))
    with pytest.raises(RuntimeError) as ei:
        kb.place_order(symbol="NSE:A", side="SELL", qty=1, check_margins=False)
    assert not isinstance(ei.value, OrderPlacementUncertain)



# --- daily stop and broker check ---------------------------------------------

@pytest.fixture
def cap100k():
    from services.plan_orchestrator import set_runtime_capital, _load_root_config
    from services.execution import symbol_blocks
    set_runtime_capital(100_000.0)
    symbol_blocks.clear_all()
    mult = float(_load_root_config()["intraday_sizing"]["book_size_multiplier"])
    yield 100_000.0 * mult
    symbol_blocks.clear_all()


def _px_ex(broker, prices):
    store, cap = PositionStore(), MagicMock()
    midday = pd.Timestamp("2026-10-29 11:00:00")                # not EOD: only the stop may exit
    ex = ExitExecutor(broker=broker, positions=store, get_ltp_ts=lambda s: (prices[s], midday), capital_manager=cap)
    return ex, store


def test_daily_stop_levels_scale_with_the_book(cap100k):
    ex, store = _px_ex(ExitBroker(), {})
    ex._daily_stop_check()
    assert ex._ds_hard_inr == pytest.approx(cap100k * ex.ds_hard_frac)
    assert ex._ds_hard_inr == pytest.approx(7_500.0) and ex._ds_soft_inr == pytest.approx(5_000.0)


def test_soft_stop_blocks_entries_but_does_not_flatten(cap100k, clock):
    from services.execution import symbol_blocks
    b = ExitBroker()
    prices = {"NSE:A": 211.0}                                   # short 500 @200 -> -Rs5,500 MTM
    ex, store = _px_ex(b, prices)
    store.upsert(_short("T1", "NSE:A"))
    ex._daily_stop_check()
    assert ex._ds_state == "soft" and symbol_blocks.blocked_reason("NSE:ZZZ")
    ex.run_once()
    assert b.placed == [] and store.get_by_trade_id("T1") is not None


def test_hard_stop_flattens_everything(cap100k, clock):
    b = ExitBroker()
    prices = {"NSE:A": 216.0, "NSE:B": 201.0}                    # -8,000 and -100
    ex, store = _px_ex(b, prices)
    store.upsert(_short("T1", "NSE:A")); store.upsert(_short("T2", "NSE:B", 100))
    ex.run_once()
    assert ex._ds_state == "hard"
    assert store.get_by_trade_id("T1") is None and store.get_by_trade_id("T2") is None


def test_realized_losses_count_toward_the_stop(cap100k, clock):
    b = ExitBroker()
    b.script = [_o("COMPLETE", 500, 215.0)]                      # closed at -7,500
    ex, store = _px_ex(b, {"NSE:A": 215.0})
    pos = _short("T1", "NSE:A"); store.upsert(pos)
    ex._exit("NSE:A", pos, 215.0, TS, "time_stop")
    assert ex._day_realized == pytest.approx(-7_500.0)
    ex._daily_stop_check()
    assert ex._ds_state == "hard"


def test_no_stop_on_a_normal_day(cap100k):
    ex, store = _px_ex(ExitBroker(), {"NSE:A": 199.0})
    store.upsert(_short("T1", "NSE:A"))
    ex._daily_stop_check()
    assert ex._ds_state is None


def test_broker_mismatch_on_two_checks_blocks_the_symbol(cap100k, clock):
    from services.execution import symbol_blocks
    b = ExitBroker()
    b.open_at_broker = {"A": -500, "HUHTAMAKI": -514}              # we track only A
    ex, store = _px_ex(b, {"NSE:A": 200.0})
    store.upsert(_short("T1", "NSE:A"))
    ex._broker_book_check()
    assert symbol_blocks.blocked_reason("NSE:HUHTAMAKI") is None    # one strike: maybe in flight
    clock["now"] += ex.broker_check_sec
    ex._broker_book_check()
    assert "514" in symbol_blocks.blocked_reason("NSE:HUHTAMAKI")
    assert symbol_blocks.blocked_reason("NSE:A") is None


def test_broker_check_is_throttled_and_live_only(cap100k, clock):
    b = ExitBroker()
    calls = []
    b.get_positions = lambda strict=False: calls.append(1) or {"day": []}
    ex, store = _px_ex(b, {})
    ex._broker_book_check(); ex._broker_book_check()
    assert len(calls) == 1
    b.dry_run = True
    clock["now"] += ex.broker_check_sec
    ex._broker_book_check()
    assert len(calls) == 1



def test_restart_restores_todays_losses_and_the_fired_stop(cap100k, clock, tmp_path):
    from services.execution import symbol_blocks
    b = ExitBroker()
    ex, store = _px_ex(b, {"NSE:A": 216.0})
    ex.persistence, ex.ds_state_path = MagicMock(), tmp_path / "ds.json"
    ex._day_realized = -8_000.0
    ex._daily_stop_check()
    assert ex._ds_state == "hard"
    symbol_blocks.clear_all()
    ex2, _ = _px_ex(b, {})                                       # the daemon restarts
    ex2.persistence, ex2.ds_state_path = MagicMock(), tmp_path / "ds.json"
    ex2._daily_stop_check()
    assert ex2._ds_state == "hard" and ex2._day_realized == -8_000.0
    assert symbol_blocks.blocked_reason("NSE:ANY")


def test_booked_partials_count_without_a_price(cap100k):
    ex, store = _px_ex(ExitBroker(), {})
    ex._get_px_ts = lambda s: (None, None)
    pos = _short("T1", "NSE:A")
    pos.plan["_state"]["t1_profit"] = -3_000.0
    store.upsert(pos)
    assert ex._day_pnl() == pytest.approx(-3_000.0)


def test_a_failing_stop_check_does_not_stop_exits(cap100k, clock):
    b = ExitBroker()
    ex, store = _px_ex(b, {"NSE:A": 200.0})
    ex._daily_stop_check = lambda: (_ for _ in ()).throw(KeyError("total_capital_inr"))
    store.upsert(_short("T1", "NSE:A"))
    ex.run_once()                                                # no exception escapes
