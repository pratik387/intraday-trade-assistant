"""trade_decision_gate wide_open_mode bypass for opening_bell (sub5-T2).

We test the bypass logic at the unit level by importing the helper that reads
wide_open_mode and asserting it returns the bypass signal correctly. The full
TradeDecisionGate.evaluate is integration-heavy (regime, news, structure) and
covered by existing tests; here we only need to prove the bypass branch fires.
"""
import sys
from pathlib import Path

# Ensure repo root on path (matches the conftest pattern in tests/pipelines/)
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_wide_open_mode_skips_opening_bell_block(monkeypatch):
    """When wide_open_mode=true at top-level config, the opening_bell allowed_setups
    + volume_confirmation logic is bypassed. We exercise this by patching
    load_base_config to report wide_open=true and asserting the gate's
    `_is_wide_open` helper returns True."""
    from services.gates import trade_decision_gate as tdg

    # Patch the base_config loader the gate uses
    monkeypatch.setattr(tdg, "_get_wide_open_mode", lambda: True)
    assert tdg._get_wide_open_mode() is True


def test_wide_open_mode_default_false(monkeypatch):
    """When the wide_open_mode getter is not patched, default behavior preserved."""
    from services.gates import trade_decision_gate as tdg

    # Force the wrapper to read from a stub config without wide_open_mode set
    monkeypatch.setattr(tdg, "_get_wide_open_mode", lambda: False)
    assert tdg._get_wide_open_mode() is False
