"""Mode profile merge: validates the loader applies mode_profiles overrides on top of base config."""
import pytest
from config.filters_setup import load_filters


def test_default_mode_is_production_no_override():
    """When mode=production, no overrides applied (empty profile)."""
    cfg = load_filters(force_reload=True)
    assert cfg["mode"] == "production"
    # entry_cutoff_hhmm in base config (production value) is unaffected
    assert cfg["entry_cutoff_hhmm"] == "15:10"


def test_oci_research_mode_overrides_via_env(monkeypatch):
    """RUN_MODE=oci_research env var picks the oci_research profile."""
    monkeypatch.setenv("RUN_MODE", "oci_research")
    cfg = load_filters(force_reload=True)
    assert cfg["entry_cutoff_hhmm"] == "15:25"
    assert cfg["last_scan_hhmm"] == "15:25"
    assert cfg["eod_squareoff_hhmm"] == "15:25"
    assert cfg["max_trades_per_cycle"] == 10000
    assert cfg["gate_input_logging"]["enabled"] is True


def test_unknown_mode_raises(monkeypatch):
    """Typo in RUN_MODE fails fast at startup."""
    monkeypatch.setenv("RUN_MODE", "oci_resaerch")  # typo
    with pytest.raises(ValueError, match="unknown mode"):
        load_filters(force_reload=True)
