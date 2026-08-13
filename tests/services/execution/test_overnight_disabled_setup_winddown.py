"""Disabling an overnight setup must not strand an OPEN real-money position.

Same defect class as the multi-day crash2d incident (2026-08-13), but the
overnight failure is total rather than partial: the exit legs read the shared
slot-pool path from setups[0], so if the enabled flag empties the selector the
leg logs "no overnight setups" and returns, and the open position is never
sold. close_dn_overnight_long is LIVE with real money and a pause is under
consideration for ~Sep-2026, so this path must be flag-independent.
"""
import pytest

from services.execution import overnight_handlers as OH


class _Spec:
    def __init__(self, name, mode, enabled, paper_enabled, wired=True):
        self.name = name
        self.mode = mode
        self.enabled = enabled
        self.raw_config = {"paper_enabled": paper_enabled}
        if wired:
            self.raw_config["capital_allocation"] = {
                "state_file": f"state/{name}_slots.json"}


def _patch_registry(monkeypatch, specs):
    class _Reg:
        _specs = {s.name: s for s in specs}

        @classmethod
        def load_from_config(cls, config):
            return cls()

    import services.dispatch.setup_registry as SR
    monkeypatch.setattr(SR, "SetupRegistry", _Reg)


DISABLED = _Spec("close_dn_overnight_long", "overnight", False, False)
ENABLED = _Spec("close_dn_overnight_long", "overnight", True, True)
INTRADAY = _Spec("some_intraday", "intraday", True, True)
UNWIRED = _Spec("research_overnight", "overnight", False, False, wired=False)


def test_disabled_overnight_setup_is_still_managed(monkeypatch):
    _patch_registry(monkeypatch, [DISABLED])
    assert [s.name for s in OH._managed_overnight_setups({})] == ["close_dn_overnight_long"]


def test_selector_still_excludes_it_for_new_entries(monkeypatch):
    _patch_registry(monkeypatch, [DISABLED])
    for mode in (True, False):
        assert OH._select_overnight_setups({}, paper_mode=mode) == []


def test_managed_excludes_non_overnight_modes(monkeypatch):
    _patch_registry(monkeypatch, [DISABLED, INTRADAY])
    assert [s.name for s in OH._managed_overnight_setups({})] == ["close_dn_overnight_long"]


def test_managed_excludes_setups_with_no_slot_pool(monkeypatch):
    """No capital_allocation => no slot pool => nothing to wind down, and
    reading raw_config['capital_allocation'] would KeyError."""
    _patch_registry(monkeypatch, [UNWIRED])
    assert OH._managed_overnight_setups({}) == []


def test_managed_is_a_superset_of_the_entry_selector(monkeypatch):
    _patch_registry(monkeypatch, [ENABLED])
    managed = {s.name for s in OH._managed_overnight_setups({})}
    for mode in (True, False):
        assert {s.name for s in OH._select_overnight_setups({}, paper_mode=mode)} <= managed


def test_enabled_setup_behaviour_is_unchanged(monkeypatch):
    """The fix must be a no-op while the setup is live — that's the safety case."""
    _patch_registry(monkeypatch, [ENABLED])
    assert ([s.name for s in OH._managed_overnight_setups({})] ==
            [s.name for s in OH._select_overnight_setups({}, paper_mode=True)])


def test_exit_legs_do_not_call_the_entry_selector():
    """Regression guard: run_place_exit / run_verify_exit must read the managed
    set. If someone points them back at the flag-gated selector, fail here."""
    import inspect
    for fn in (OH.run_place_exit, OH.run_verify_exit):
        src = inspect.getsource(fn)
        assert "_managed_overnight_setups" in src, f"{fn.__name__} lost the managed set"
        assert "_select_overnight_setups" not in src, \
            f"{fn.__name__} gates exits on the enabled flag"


def test_entry_leg_still_uses_the_flag_gated_selector():
    import inspect
    assert "_select_overnight_setups" in inspect.getsource(OH.run_entry)
