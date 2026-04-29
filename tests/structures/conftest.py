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
    # narrow_cpr_breakout / vwap_first_pullback removed in earlier cleanup —
    # mis_unwind_short / closing_hour_reversal removed in Phase 1 cleanup
    # (no salvageable cell). Surviving sub8 detectors with _is_wide_open:
    # orb_15, pdh_pdl_reject, pdh_pdl_sweep_reclaim, gap_and_go_continuation,
    # ema5_alert_pullback, camarilla_l3_reversal. gap_fade_short uses a
    # different mechanism.
    import structures.orb_15_structure as _orb
    import structures.pdh_pdl_reject_structure as _pdh
    import structures.pdh_pdl_sweep_reclaim_structure as _pdh_sr
    import structures.gap_and_go_continuation_structure as _gap_and_go
    import structures.ema5_alert_pullback_structure as _ema5
    import structures.camarilla_l3_reversal_structure as _cam_l3

    for mod in (_orb, _pdh, _pdh_sr, _gap_and_go, _ema5, _cam_l3):
        monkeypatch.setattr(mod, "_is_wide_open", lambda: False)
