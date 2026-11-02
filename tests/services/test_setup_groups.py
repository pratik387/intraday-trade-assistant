"""Setup groups (2026-11-03): competing setups on one symbol-day resolve in real
time — first selected tradeable member plan claims the symbol-day, same-scan ties go
to the higher-priority member (validated rule for up_spike + the results-day cells)."""
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from services import plan_orchestrator as po

UP, DAY, LATE = "up_spike_fade_short", "results_breadth_day_fade_short", "results_breadth_late_fade_short"
T1, T2 = pd.Timestamp("2026-11-03 10:05"), pd.Timestamp("2026-11-03 10:20")
CFG = json.load(io.open(Path(__file__).resolve().parents[2] / "config" / "configuration.json", encoding="utf-8"))


def _plan(setup, qty=10, score=1.0):
    return {"strategy": setup, "symbol": "NSE:TESTCO", "eligible": True,
            "sizing": {"qty": qty}, "ranking": {"score": score}}


@pytest.fixture
def orch():
    return po.PlanOrchestrator()


def test_config_group_is_the_up_spike_family():
    g = CFG["setup_groups"]["up_spike_family"]
    assert g["members_in_priority_order"] == [UP, DAY, LATE]
    assert g["one_trade_per_symbol_day"] is True


def test_same_scan_tie_goes_to_up_spike(orch):
    out = orch._filter_setup_groups("NSE:TESTCO", T1, [(_plan(DAY), 9.0), (_plan(UP), 1.0)])
    assert [p["strategy"] for p, _ in out] == [UP]


def test_untradeable_plans_neither_win_nor_claim(orch):
    # up_spike's plan sits at the circuit (qty 0): the tradeable cell plan stays
    out = orch._filter_setup_groups("NSE:TESTCO", T1, [(_plan(UP, qty=0), 1.0), (_plan(DAY), 9.0)])
    assert sorted(p["strategy"] for p, _ in out) == sorted([UP, DAY])
    orch._claim_setup_groups("NSE:TESTCO", T1, [_plan(UP, qty=0)])
    assert not orch._group_claims                                           # qty 0 never claims


def test_first_selected_member_claims_the_symbol_day(orch):
    out = orch._filter_setup_groups("NSE:TESTCO", T1, [(_plan(DAY), 1.0)])
    orch._claim_setup_groups("NSE:TESTCO", T1, [p for p, _ in out])
    assert orch._filter_setup_groups("NSE:TESTCO", T2, [(_plan(UP), 5.0)]) == []
    assert len(orch._filter_setup_groups("NSE:OTHER", T2, [(_plan(UP), 5.0)])) == 1


def test_a_new_session_resets_the_claims(orch):
    orch._claim_setup_groups("NSE:TESTCO", T1, [_plan(UP)])
    nxt = pd.Timestamp("2026-11-04 10:05")
    assert len(orch._filter_setup_groups("NSE:TESTCO", nxt, [(_plan(DAY), 1.0)])) == 1
    orch._claim_setup_groups("NSE:OTHER", nxt, [_plan(UP)])
    assert all(k[1] == nxt.date() for k in orch._group_claims)


def test_setups_outside_any_group_pass_and_do_not_claim(orch):
    other = _plan("close_dn_overnight_long")
    out = orch._filter_setup_groups("NSE:TESTCO", T1, [(other, 1.0), (_plan(DAY), 2.0)])
    assert [p["strategy"] for p, _ in out] == ["close_dn_overnight_long", DAY]
    orch._claim_setup_groups("NSE:TESTCO", T1, [other])
    assert not orch._group_claims


def test_drops_are_persisted_to_the_planning_log(orch, monkeypatch):
    rows = []

    class P:
        def log_reject(self, symbol, **kw):
            rows.append((symbol, kw))
    monkeypatch.setattr(po, "_planning_logger", lambda: P())
    orch._filter_setup_groups("NSE:TESTCO", T1, [(_plan(DAY), 9.0), (_plan(UP), 1.0)])
    assert rows[0][0] == "NSE:TESTCO"
    assert rows[0][1]["reason"] == "setup_group_lower_priority"
    assert rows[0][1]["strategy_type"] == DAY and rows[0][1]["kept_strategy"] == UP and rows[0][1]["qty"] == 10


class _C:
    def __init__(self, s):
        self.setup_type = s


def _stub_planning(orch, monkeypatch, plans):
    monkeypatch.setattr(orch, "process_single_candidate", lambda **kw: dict(plans[kw["setup_type"]]))
    monkeypatch.setattr(orch, "_log_accept", lambda *a, **k: None)


def test_process_candidates_single_best_and_return_all(orch, monkeypatch):
    _stub_planning(orch, monkeypatch, {UP: _plan(UP, score=1.0), DAY: _plan(DAY, score=9.0)})
    kw = dict(df5m=pd.DataFrame(), levels={}, regime="chop", candidates=[_C(DAY), _C(UP)])
    best = orch.process_candidates(symbol="NSE:A", now=T1, **kw)
    assert best["strategy"] == UP                                            # tie: up_spike despite lower score
    allp = orch.process_candidates(symbol="NSE:B", now=T1, return_all_eligible=True, **kw)   # the live path
    assert [p["strategy"] for p in allp] == [UP]
    later = orch.process_candidates(symbol="NSE:B", now=T2, return_all_eligible=True,
                                    df5m=pd.DataFrame(), levels={}, regime="chop", candidates=[_C(DAY)])
    assert later == []                                                       # claimed at the earlier scan


def test_process_candidates_multi_applies_the_group(orch, monkeypatch):
    _stub_planning(orch, monkeypatch, {UP: _plan(UP, score=1.0), DAY: _plan(DAY, score=9.0)})
    monkeypatch.setattr(po, "_get_selection_rules",
                        lambda: {"max_positions_total": 10, "skip_duplicate_symbols": False})
    data = [{"symbol": "NSE:A", "df5m": pd.DataFrame(), "levels": {}, "candidates": [_C(DAY), _C(UP)]}]
    sel = orch.process_candidates_multi(symbols_data=data, regime="chop", now=T1)
    assert [p["strategy"] for p in sel] == [UP]
    assert orch.process_candidates_multi(symbols_data=data, regime="chop", now=T2) == []


def test_unknown_member_fails_at_startup(monkeypatch):
    bad = dict(po._load_root_config(), setup_groups={"g": {"members_in_priority_order": [UP, "no_such_setup"],
                                                           "one_trade_per_symbol_day": True}})
    monkeypatch.setattr(po, "_ROOT_CFG_CACHE", bad)
    with pytest.raises(po.OrchestratorConfigError):
        po.set_runtime_trading_mode("paper")


def test_unreadable_results_calendar_fails_at_startup(monkeypatch, tmp_path):
    real = po._load_root_config()
    setups = dict(real["setups"])
    setups[DAY] = dict(setups[DAY], event_feed=dict(setups[DAY]["event_feed"], path=str(tmp_path / "missing.parquet")))
    monkeypatch.setattr(po, "_ROOT_CFG_CACHE", dict(real, setups=setups))
    with pytest.raises(po.OrchestratorConfigError, match=DAY):
        po.set_runtime_trading_mode("paper")


def test_live_daemon_drops_paper_only_intraday_setups(monkeypatch):
    real = po._load_root_config()
    monkeypatch.setattr(po, "_ROOT_CFG_CACHE", dict(real, runtime_trading_mode="live"))
    assert sorted(po.paper_only_intraday_setups_in_live(real["setups"])) == sorted([DAY, LATE])
    monkeypatch.setattr(po, "_ROOT_CFG_CACHE", dict(real, runtime_trading_mode="paper"))
    assert po.paper_only_intraday_setups_in_live(real["setups"]) == []
