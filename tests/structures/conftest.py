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

    Sub-9 cleanup (2026-05-01): the 6 sub-7/sub-8 candidate detectors
    (orb_15, pdh_pdl_reject, pdh_pdl_sweep_reclaim, gap_and_go_continuation,
    ema5_alert_pullback, camarilla_l3_reversal) were deleted after
    Phase-1 validation failure.

    2026-05-08: gap_fade_short and capitulation_long_morning gained
    _is_wide_open helpers as part of the wide_open OCI-capture fix —
    added to patch list here so existing unit tests remain isolated.
    """
    import structures.expiry_pin_strike_reversal_structure as _exp_pin
    import structures.circuit_t1_fade_short_structure as _circ
    import structures.delivery_pct_anomaly_short_structure as _del_pct
    import structures.capitulation_long_morning_structure as _cap
    import structures.gap_fade_short_structure as _gap
    import structures.earnings_day_intraday_fade_structure as _earn
    for mod in (_exp_pin, _circ, _del_pct, _cap, _gap, _earn):
        monkeypatch.setattr(mod, "_is_wide_open", lambda: False)
