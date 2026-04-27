"""Fixtures for structure detector tests.

wide_open_mode is patched to False for all structure tests so that
design-inferred filter tests remain meaningful regardless of what
configuration.json says on the current branch.
"""
import pytest


@pytest.fixture(autouse=True)
def force_wide_open_false(monkeypatch):
    """Patch all sub8 _is_wide_open helpers to return False.

    This ensures structure unit tests always exercise the full filter set,
    regardless of the wide_open_mode value in configuration.json.
    """
    import structures.orb_15_structure as _orb
    import structures.narrow_cpr_breakout_structure as _ncpr
    import structures.vwap_first_pullback_structure as _vwap
    import structures.pdh_pdl_reject_structure as _pdh
    import structures.closing_hour_reversal_structure as _chr

    for mod in (_orb, _ncpr, _vwap, _pdh, _chr):
        monkeypatch.setattr(mod, "_is_wide_open", lambda: False)
