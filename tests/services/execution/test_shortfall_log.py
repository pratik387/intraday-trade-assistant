"""Live execution shortfall ledger (Risk Charter item 6, 2026-10-30).

Oct 2026 live fills vs the price the decision assumed: entries 7.2 bps worse on
average (p90 21), exits 2.9 bps (p90 13). The ledger keeps that measurement per
trade across sessions and alarms when the last 20 trades lose more than 30% of the
backtest edge to execution.
"""
import json
import logging

import pytest

import services.plan_orchestrator as po
from services.execution import shortfall_log as sf
from tests.services.execution.test_circuit_cover import (  # noqa: F401  (fixtures)
    SYM, TID, TS, FakeKite, clock, live)

SETUP = "up_spike_fade_short"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "shortfall.jsonl"
    monkeypatch.setattr(sf, "_ledger_path", lambda: p)
    return p


def _rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def _trade(i, entry_bps, exit_bps, px=100.0):
    """Short trade i: SELL entry at px, BUY cover at px."""
    tid = f"T{i}"
    sf.record_entry(trade_id=tid, setup=SETUP, symbol="NSE:X", side="SELL", qty=10,
                    decision_px=px, fill_px=px * (1 - entry_bps / 1e4))
    sf.record_exit(trade_id=tid, setup=SETUP, symbol="NSE:X", side="BUY", qty=10,
                   decision_px=px, fill_px=px * (1 + exit_bps / 1e4), reason="time_stop_15:10")


def test_adverse_is_positive_on_both_sides():
    assert sf.adverse_bps("SELL", 100.0, 99.9) == pytest.approx(10.0)    # sold lower
    assert sf.adverse_bps("BUY", 100.0, 100.1) == pytest.approx(10.0)    # bought higher
    assert sf.adverse_bps("BUY", 100.0, 99.9) == pytest.approx(-10.0)    # price improvement
    with pytest.raises(ValueError):
        sf.adverse_bps("BUY", 0.0, 1.0)


def test_exit_legs_are_qty_weighted_and_open_or_unmatched_trades_skipped():
    rows = [
        {"setup": SETUP, "trade_id": "A", "leg": "entry", "bps": 10, "qty": 30, "inr": 3, "ts": "1"},
        {"setup": SETUP, "trade_id": "A", "leg": "exit", "bps": 0, "qty": 20, "inr": 0, "ts": "2"},
        {"setup": SETUP, "trade_id": "A", "leg": "exit", "bps": 30, "qty": 10, "inr": 3, "ts": "3"},
        {"setup": SETUP, "trade_id": "OPEN", "leg": "entry", "bps": 5, "qty": 1, "inr": 0, "ts": "4"},
        {"setup": SETUP, "trade_id": "OLD", "leg": "exit", "bps": 5, "qty": 1, "inr": 0, "ts": "5"},
        {"setup": "other", "trade_id": "B", "leg": "entry", "bps": 99, "qty": 1, "inr": 0, "ts": "6"},
    ]
    (t,) = sf.trade_shortfalls(rows, SETUP)
    assert t["trade_id"] == "A"
    assert t["exit_bps"] == pytest.approx(10.0) and t["total_bps"] == pytest.approx(20.0)


def test_no_alarm_value_before_the_window_fills(ledger):
    for i in range(19):
        _trade(i, 40, 40)
    assert sf.check_alarm(SETUP) is None


def test_alarm_above_30pct_of_the_backtest_edge(ledger, caplog):
    edge = po._load_root_config()["setups"][SETUP]["backtest_edge_bps"]
    caplog.set_level(logging.CRITICAL)
    for i in range(20):
        _trade(i, 0.2 * edge, 0.2 * edge)            # 40% of the edge
    assert sf.check_alarm(SETUP) == pytest.approx(0.4 * edge, rel=1e-3)
    assert any("SHORTFALL_ALARM" in r.getMessage() for r in caplog.records)


def test_no_alarm_at_the_october_baseline(ledger, caplog):
    caplog.set_level(logging.CRITICAL)
    for i in range(25):
        _trade(i, 7.2, 2.9)
    assert sf.check_alarm(SETUP) == pytest.approx(10.1, abs=0.05)
    assert not any("SHORTFALL_ALARM" in r.getMessage() for r in caplog.records)


def test_live_exit_is_recorded_against_the_first_decision_price(live, ledger, clock):
    ex, broker, store, cap, pos = live
    pos.plan["setup_type"] = SETUP
    pos.plan["exits"].pop("circuit_cover_px")          # no resting cover in this test
    broker.exit_status = "REJECTED"
    ex._exit(SYM, pos, 200.0, TS, "hard_sl")
    assert not ledger.exists()                        # nothing filled, nothing recorded
    clock["now"] += ex.exit_retry_cooldown_sec
    broker.exit_status, broker.exit_fill = "COMPLETE", 202.0
    ex._exit(SYM, pos, 201.0, TS, "hard_sl")
    (row,) = _rows(ledger)
    assert row["leg"] == "exit" and row["side"] == "BUY" and row["trade_id"] == TID
    assert row["decision_px"] == 200.0 and row["fill_px"] == 202.0 and row["bps"] == pytest.approx(100.0)
    assert "_sf_decision" not in pos.plan["_state"]


def test_a_ledger_failure_never_blocks_the_exit(live, monkeypatch):
    ex, broker, store, cap, pos = live
    pos.plan["setup_type"] = SETUP
    pos.plan["exits"].pop("circuit_cover_px")

    def boom(**kw):
        raise OSError("disk full")
    monkeypatch.setattr(sf, "record_exit", boom)
    ex._exit(SYM, pos, 200.0, TS, "time_stop_15:10")
    assert store.get_by_trade_id(TID) is None


def test_paper_exits_are_not_recorded(live, ledger):
    ex, broker, store, cap, pos = live
    pos.plan["setup_type"] = SETUP
    pos.plan["exits"].pop("circuit_cover_px")
    broker.dry_run = True
    ex._exit(SYM, pos, 200.0, TS, "time_stop_15:10")
    assert not ledger.exists()


def _sf_cfg(tmp_path, **kw):
    c = {"ledger_path": str(tmp_path / "sf.jsonl"), "rolling_trades": 20,
         "alarm_frac_of_backtest_edge": 0.3, "per_trade_cap_bps": 100}
    c.update(kw)
    return c


def test_live_startup_needs_the_edge_on_every_live_setup(monkeypatch, tmp_path):
    root = {"setups": {"a": {"enabled": True, "mode": "intraday", "live_enabled": True}},
            "shortfall_log": _sf_cfg(tmp_path)}
    monkeypatch.setattr(po, "_ROOT_CFG_CACHE", root)
    with pytest.raises(po.OrchestratorConfigError, match="backtest_edge_bps"):
        po.set_runtime_trading_mode("live")
    root["setups"]["a"]["backtest_edge_bps"] = 53
    del root["shortfall_log"]
    with pytest.raises(po.OrchestratorConfigError, match="shortfall_log"):
        po.set_runtime_trading_mode("live")
    root["shortfall_log"] = _sf_cfg(tmp_path)
    assert po.set_runtime_trading_mode("live") == ["a"]
    assert (tmp_path / "sf.jsonl").exists()             # writability checked at startup


@pytest.mark.parametrize("bad", [{"rolling_trades": 0}, {"rolling_trades": 2.5},
                                 {"alarm_frac_of_backtest_edge": 0}, {"per_trade_cap_bps": -1}])
def test_live_startup_rejects_unusable_values(monkeypatch, tmp_path, bad):
    root = {"setups": {"a": {"enabled": True, "mode": "intraday", "live_enabled": True,
                             "backtest_edge_bps": 53}}, "shortfall_log": _sf_cfg(tmp_path, **bad)}
    monkeypatch.setattr(po, "_ROOT_CFG_CACHE", root)
    with pytest.raises(po.OrchestratorConfigError, match=next(iter(bad))):
        po.set_runtime_trading_mode("live")


def test_live_startup_fails_when_the_ledger_is_not_writable(monkeypatch, tmp_path):
    (tmp_path / "f").write_text("a file, not a directory")
    root = {"setups": {"a": {"enabled": True, "mode": "intraday", "live_enabled": True,
                             "backtest_edge_bps": 53}},
            "shortfall_log": _sf_cfg(tmp_path, ledger_path=str(tmp_path / "f" / "sf.jsonl"))}
    monkeypatch.setattr(po, "_ROOT_CFG_CACHE", root)
    with pytest.raises(po.OrchestratorConfigError, match="not writable"):
        po.set_runtime_trading_mode("live")


def test_one_circuit_locked_cover_does_not_trip_the_alarm(ledger, caplog):
    caplog.set_level(logging.CRITICAL)
    for i in range(19):
        _trade(i, 7.2, 2.9)
    _trade(19, 0, 800)                                   # covered 8% above the stop at a locked circuit
    mean = sf.check_alarm(SETUP)
    assert mean < 0.3 * po._load_root_config()["setups"][SETUP]["backtest_edge_bps"]
    assert not any("SHORTFALL_ALARM" in r.getMessage() for r in caplog.records)


def test_a_record_without_trade_id_is_refused():
    with pytest.raises(ValueError):
        sf.record_exit(trade_id=None, setup=SETUP, symbol="NSE:X", side="BUY", qty=1,
                       decision_px=1.0, fill_px=1.0, reason="x")


def test_a_different_exit_starts_from_its_own_price(live, ledger, clock):
    ex, broker, store, cap, pos = live
    pos.plan["setup_type"] = SETUP
    pos.plan["exits"].pop("circuit_cover_px")
    broker.exit_status = "REJECTED"
    ex._exit(SYM, pos, 180.0, TS, "target_t1_full")       # target exit rejected
    clock["now"] += ex.exit_retry_cooldown_sec
    broker.exit_status, broker.exit_fill = "COMPLETE", 210.0
    ex._exit(SYM, pos, 209.0, TS, "hard_sl")              # then the stop fills
    (row,) = _rows(ledger)
    assert row["decision_px"] == 209.0 and row["reason"] == "hard_sl"


def test_eod_retries_keep_the_first_decision_price(live, ledger, clock):
    ex, broker, store, cap, pos = live
    pos.plan["setup_type"] = SETUP
    pos.plan["exits"].pop("circuit_cover_px")
    broker.exit_status = "REJECTED"
    ex._exit(SYM, pos, 200.0, TS, "eod_squareoff")
    clock["now"] += ex.exit_retry_cooldown_sec
    broker.exit_status, broker.exit_fill = "COMPLETE", 204.0
    ex._exit(SYM, pos, 203.0, TS, "eod_squareoff_retry1")
    (row,) = _rows(ledger)
    assert row["decision_px"] == 200.0


def test_a_partly_filled_cover_leg_is_recorded(live, ledger, clock):
    ex, broker, store, cap, pos = live
    pos.plan["setup_type"] = SETUP
    ex._ensure_circuit_order(SYM, pos)
    broker.statuses["O1"].update(status="CANCELLED", filled_quantity=200, average_price=229.57)
    broker.exit_status = "REJECTED"
    ex._exit(SYM, pos, 205.0, TS, "time_stop_15:10")
    (row,) = _rows(ledger)
    assert row["qty"] == 200 and row["fill_px"] == 229.57 and row["decision_px"] == 205.0


def test_every_live_setup_in_the_real_config_has_an_edge():
    cfg = po._load_root_config()
    live = [n for n, r in cfg["setups"].items() if po._is_enabled_intraday(r) and r.get("live_enabled") is True]
    assert live and all(isinstance(cfg["setups"][n]["backtest_edge_bps"], (int, float)) for n in live)


class _EntryKite:
    dry_run = False

    def place_order(self, **kw):
        self.kw = kw
        return "E1"

    def get_order_status(self, oid):
        return {"status": "COMPLETE", "filled_quantity": 514, "average_price": 193.32, "status_message": ""}


def _entry_executor(monkeypatch, broker):
    from types import SimpleNamespace
    import services.execution.trigger_aware_executor as tae
    monkeypatch.setattr(tae.time, "sleep", lambda s: None)
    t = tae.TriggerAwareExecutor.__new__(tae.TriggerAwareExecutor)
    t.broker, t.entry_settle_retries, t.entry_settle_delay_sec = broker, 3, 0.0
    t.live_guards = t.trading_logger = t.capital_manager = t.persistence = t.positions = None
    t.risk = SimpleNamespace(open_positions={})
    plan = {"side": "SELL", "qty": 514, "price": 193.65, "setup_type": SETUP, "hard_sl": 200.0}
    trade = tae.PendingTrade(trade_id="NSE:HUHTAMAKI_abc123def456", symbol="NSE:HUHTAMAKI", plan=plan,
                             state=None, created_at=TS, trigger_price=193.65, trigger_timestamp=TS)
    return t, trade


def test_live_entry_fill_is_recorded_against_the_trigger_price(monkeypatch, ledger):
    t, trade = _entry_executor(monkeypatch, _EntryKite())
    assert t._place_trade_order(trade) is True
    (row,) = _rows(ledger)
    assert row["leg"] == "entry" and row["side"] == "SELL" and row["qty"] == 514
    assert row["decision_px"] == 193.65 and row["fill_px"] == 193.32
    assert row["bps"] == pytest.approx(1e4 * 0.33 / 193.65, abs=0.01)     # sold lower = adverse


def test_a_ledger_failure_never_blocks_the_entry(monkeypatch):
    t, trade = _entry_executor(monkeypatch, _EntryKite())

    def boom(**kw):
        raise OSError("disk full")
    monkeypatch.setattr(sf, "record_entry", boom)
    assert t._place_trade_order(trade) is True
