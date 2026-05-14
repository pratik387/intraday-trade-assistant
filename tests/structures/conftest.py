"""Fixtures for structure detector tests.

wide_open_mode is patched to False for all structure tests so that
design-inferred filter tests remain meaningful regardless of what
configuration.json says on the current branch.
"""
import pytest


@pytest.fixture(autouse=True)
def force_wide_open_false(monkeypatch):
    """Patch active detectors' _is_wide_open helpers to return False.

    This ensures structure unit tests always exercise the full filter set,
    regardless of the wide_open_mode value in configuration.json.

    Retired-setup helpers removed 2026-05-14 (see docs/retired_setups.md).
    """
    import structures.circuit_t1_fade_short_structure as _circ
    import structures.delivery_pct_anomaly_short_structure as _del_pct
    import structures.gap_fade_short_structure as _gap
    for mod in (_circ, _del_pct, _gap):
        if hasattr(mod, "_is_wide_open"):
            monkeypatch.setattr(mod, "_is_wide_open", lambda: False)
