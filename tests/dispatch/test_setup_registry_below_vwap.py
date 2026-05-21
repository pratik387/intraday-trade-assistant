"""Setup registry can load + validate below_vwap_volume_revert_long.

Even though the setup is disabled in live (`enabled: false`), the registry
must be able to PARSE its config without error so the paper-trade pipeline
can opt-in via paper_enabled.
"""
import json
from pathlib import Path

from services.dispatch.setup_registry import SetupRegistry


def test_registry_parses_below_vwap_block():
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "configuration.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    reg = SetupRegistry.load_from_config(cfg)
    spec = reg.get("below_vwap_volume_revert_long")
    assert spec.detector_class_path == (
        "structures.below_vwap_volume_revert_long_structure."
        "BelowVwapVolumeRevertLongStructure"
    )
    assert spec.universe_builder_path == (
        "services.setup_universe.below_vwap_volume_revert_long_universe"
    )
    # Wide active window for paper observability; cell hhmm gate is inside the detector.
    assert spec.active_window[0].strftime("%H:%M") == "10:00"
    assert spec.active_window[1].strftime("%H:%M") == "14:55"


def test_registry_validate_imports_succeeds_if_paper_enabled():
    """If we add `enabled: true`, registry.validate() must succeed (= class
    imports cleanly). We test the import directly rather than mutating config."""
    import importlib
    mod = importlib.import_module("structures.below_vwap_volume_revert_long_structure")
    assert hasattr(mod, "BelowVwapVolumeRevertLongStructure")
    builder = importlib.import_module("services.setup_universe")
    assert hasattr(builder, "below_vwap_volume_revert_long_universe")


def test_setup_categories_has_below_vwap():
    """config.setup_categories must register the new setup as REVERSION (per
    Task 1 code-reviewer flag — without this, services.gates.trade_decision_gate
    would silently treat the setup as uncategorized)."""
    from config.setup_categories import SETUP_CATEGORIES, SetupCategory, get_category
    assert SETUP_CATEGORIES["below_vwap_volume_revert_long"] == SetupCategory.REVERSION
    assert get_category("below_vwap_volume_revert_long") == SetupCategory.REVERSION
