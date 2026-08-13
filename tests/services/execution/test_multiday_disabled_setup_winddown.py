"""Disabling a multi-day setup must not strand the positions it already holds.

Regression, 2026-08-13. crash2d was disabled on 2026-08-12 while holding 10
filled positions and 3 pending AMO entries. Both the exit leg and the
entry-fill leg iterated the *eligible* (enabled) setup list, so:

  - NSE:CLSEL was due to square off on 08-13 and had NO exit path at all;
    the other 9 filled positions were equally stranded.
  - the 3 pending entries could neither fill nor be cleaned up, so they sat
    in the book forever displaying "pending entry".

The flag must gate NEW exposure only. Wind-down runs for every multi-day
setup. A pending entry belonging to a disabled setup is DROPPED rather than
filled, because filling it would open new exposure in a setup we switched off.
"""
import pytest

from services.execution import mtf_capitulation_handlers as H

MANAGED = H._managed_multiday_setups
ELIGIBLE = H._eligible_multiday_setups


def _cfg(**flags):
    """Two multi-day setups + one intraday setup that must never be included."""
    return {"setups": {
        "zscore_oversold_revert_long": {
            "horizon": "multi_day", "enabled": True, "paper_enabled": True},
        "crash2d_revert_long": {
            "horizon": "multi_day",
            "enabled": flags.get("crash2d_enabled", False),
            "paper_enabled": flags.get("crash2d_paper", False)},
        "some_intraday_setup": {
            "horizon": "intraday", "enabled": True, "paper_enabled": True},
    }}


def test_managed_includes_the_disabled_setup():
    names = [n for n, _ in MANAGED(_cfg())]
    assert "crash2d_revert_long" in names, "disabled setup lost its exit path"
    assert "zscore_oversold_revert_long" in names


def test_managed_excludes_non_multiday_setups():
    assert "some_intraday_setup" not in [n for n, _ in MANAGED(_cfg())]


def test_eligible_still_excludes_the_disabled_setup():
    """The entry gate must keep working — that's the point of the flag."""
    names = [n for n, _ in ELIGIBLE(_cfg(), paper_mode=True)]
    assert "crash2d_revert_long" not in names
    assert "zscore_oversold_revert_long" in names


def test_managed_is_a_superset_of_eligible():
    cfg = _cfg()
    managed = {n for n, _ in MANAGED(cfg)}
    for mode in (True, False):
        assert {n for n, _ in ELIGIBLE(cfg, paper_mode=mode)} <= managed


def test_enabling_makes_the_two_sets_agree():
    cfg = _cfg(crash2d_enabled=True, crash2d_paper=True)
    assert {n for n, _ in MANAGED(cfg)} == {n for n, _ in ELIGIBLE(cfg, paper_mode=True)}


def test_managed_ignores_paper_vs_live_flag_split():
    """paper_enabled=False + enabled=True must still be managed in paper mode."""
    cfg = _cfg(crash2d_enabled=True, crash2d_paper=False)
    assert "crash2d_revert_long" in [n for n, _ in MANAGED(cfg)]
    assert "crash2d_revert_long" not in [n for n, _ in ELIGIBLE(cfg, paper_mode=True)]


def test_empty_config_is_harmless():
    assert MANAGED({}) == []
    assert MANAGED({"setups": {}}) == []


def test_exit_leg_iterates_managed_not_eligible(monkeypatch):
    """run_eod Phase A must call _run_exits for the disabled setup too."""
    cfg = _cfg()
    seen = []
    monkeypatch.setattr(H, "PositionPersistence", lambda *a, **k: object())
    monkeypatch.setattr(H, "_position_state_dir", lambda raw: "unused")
    monkeypatch.setattr(H, "_warn_mtf_delisted", lambda *a, **k: None)
    monkeypatch.setattr(H, "_run_exits",
                        lambda name, *a, **k: seen.append(name))
    H.run_eod(cfg, broker=object(), now_ist=None, paper_mode=True, phase="exits")
    assert "crash2d_revert_long" in seen, "disabled setup was skipped by the exit leg"
    assert "zscore_oversold_revert_long" in seen
    assert "some_intraday_setup" not in seen


class _Pos:
    def __init__(self, entry_date):
        self.state = {"pending_entry_fill": True, "qty": 100}
        self.entry_date = entry_date
        self.order_id = "AMO1"
        self.product = "MTF"
        self.trade_id = "t1"


class _Store:
    def __init__(self, snap):
        self.snap = snap
        self.removed = []
        self.updated = []

    def load_snapshot(self):
        return dict(self.snap)

    def remove_position(self, sym):
        self.removed.append(sym)

    def update_position(self, sym, **kw):
        self.updated.append(sym)


def _run_verify(monkeypatch, cfg, stores):
    monkeypatch.setattr(H, "_position_state_dir", lambda raw: "unused")
    monkeypatch.setattr(H, "PositionPersistence",
                        lambda d, _s=stores: _s.pop(0))
    monkeypatch.setattr(H, "_paper_open_price", lambda *a, **k: 101.0)
    import pandas as pd
    return H.run_verify_entries(cfg, broker=object(),
                                now_ist=pd.Timestamp("2026-08-13 09:33"),
                                paper_mode=True)


def test_pending_entry_of_a_disabled_setup_is_dropped_not_filled(monkeypatch):
    """Filling it would open NEW exposure in a setup we just switched off."""
    zs = _Store({"NSE:GOOD": _Pos("2026-08-13")})
    c2 = _Store({"NSE:AMRUTANJAN": _Pos("2026-08-13")})
    summary = _run_verify(monkeypatch, _cfg(), [zs, c2])
    assert c2.removed == ["NSE:AMRUTANJAN"], "stale pending entry was not cleaned up"
    assert c2.updated == [], "disabled setup opened new exposure"
    assert zs.updated == ["NSE:GOOD"], "enabled setup stopped filling"
    assert summary["filled_count"] == 1 and summary["unfilled_count"] == 1
    assert {"setup": "crash2d_revert_long", "symbol": "NSE:AMRUTANJAN",
            "dropped": "setup_disabled"} in summary["events"]


def test_future_dated_pending_entry_is_left_alone(monkeypatch):
    """Only entries whose day has arrived are resolved — disabled or not."""
    c2 = _Store({"NSE:LATER": _Pos("2026-08-20")})
    _run_verify(monkeypatch, _cfg(), [_Store({}), c2])
    assert c2.removed == [] and c2.updated == []


def test_entry_leg_still_skips_the_disabled_setup(monkeypatch):
    cfg = _cfg()
    passed = {}
    monkeypatch.setattr(H, "PositionPersistence", lambda *a, **k: object())
    monkeypatch.setattr(H, "_position_state_dir", lambda raw: "unused")
    monkeypatch.setattr(H, "_warn_mtf_delisted", lambda *a, **k: None)
    monkeypatch.setattr(H, "_prewarm_daily_universe", lambda *a, **k: None)
    monkeypatch.setattr(H, "_cb_paused_setups", lambda setups: [])
    monkeypatch.setattr(
        H, "_run_entries_composite",
        lambda setups, *a, **k: passed.update(names=[n for n, _ in setups]))
    H.run_eod(cfg, broker=object(), now_ist=None, paper_mode=True, phase="entries")
    assert passed["names"] == ["zscore_oversold_revert_long"]
