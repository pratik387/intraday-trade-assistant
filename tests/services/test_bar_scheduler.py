"""Tests for services.bar_scheduler.schedule_admits."""
import pandas as pd
import pytest


def _plan(symbol, setup_type, priority, qty=10, entry=100.0, stop_distance=2.0):
    """Build a minimal plan dict that bar_scheduler can consume."""
    return {
        "symbol": symbol,
        "strategy": setup_type,
        "priority": priority,
        "sizing": {"qty": qty, "notional": qty * entry},
        "entry_ref_price": entry,
        "stop": {"hard": entry + stop_distance, "risk_per_share": stop_distance},
    }


class _FakeCapitalManager:
    def __init__(self, total=1_000_000, allow_all=True):
        self.total_capital = total
        self.available_capital = total
        self._allow = allow_all
        self.admits = []

    def can_enter_position(self, symbol, side, entry_price, stop_distance,
                            cap_segment, setup_type=None, **_):
        if not self._allow:
            return False, "no_capital", 0
        return True, "", 100

    def enter_position(self, symbol, side, qty, price, cap_segment, setup_type=None, **_):
        self.admits.append((symbol, setup_type))


class _FakeRisk:
    def can_admit(self, sym, setup_type, ts):
        return True, ""

    def record_admit(self, sym, setup_type, ts):
        pass


def test_schedule_returns_plans_in_priority_order():
    from services.bar_scheduler import schedule_admits

    plans = [
        _plan("NSE:A", "gap_fade_short", priority=40),
        _plan("NSE:B", "earnings_day_intraday_fade", priority=90),
        _plan("NSE:C", "circuit_t1_fade_short", priority=70),
    ]
    cm = _FakeCapitalManager()
    rt = _FakeRisk()
    admitted = schedule_admits(plans, cm, rt, ts=pd.Timestamp("2024-08-29 09:20"))
    # Earnings (90) > circuit (70) > gap (40)
    assert [a["symbol"] for a in admitted] == ["NSE:B", "NSE:C", "NSE:A"]


def test_schedule_stops_when_capital_exhausted():
    from services.bar_scheduler import schedule_admits

    plans = [
        _plan("NSE:A", "gap_fade_short", priority=40),
        _plan("NSE:B", "earnings_day_intraday_fade", priority=90),
    ]
    cm = _FakeCapitalManager(allow_all=False)   # all rejections
    rt = _FakeRisk()
    admitted = schedule_admits(plans, cm, rt, ts=pd.Timestamp("2024-08-29 09:20"))
    assert admitted == []


def test_schedule_respects_setup_risk_tracker():
    from services.bar_scheduler import schedule_admits

    class _DenyAll:
        def can_admit(self, *a, **kw): return False, "blocked"
        def record_admit(self, *a, **kw): pass

    plans = [_plan("NSE:A", "gap_fade_short", priority=70)]
    cm = _FakeCapitalManager()
    admitted = schedule_admits(plans, cm, _DenyAll(), ts=pd.Timestamp("2024-08-29 09:20"))
    assert admitted == []


def test_schedule_records_admit_on_setup_risk():
    from services.bar_scheduler import schedule_admits

    recorded = []
    class _RecordingRisk:
        def can_admit(self, *a, **kw): return True, ""
        def record_admit(self, sym, setup_type, ts):
            recorded.append((sym, setup_type, ts))

    plans = [_plan("NSE:A", "gap_fade_short", priority=70)]
    cm = _FakeCapitalManager()
    rt = _RecordingRisk()
    schedule_admits(plans, cm, rt, ts=pd.Timestamp("2024-08-29 09:20"))
    assert len(recorded) == 1
    assert recorded[0][0] == "NSE:A"
