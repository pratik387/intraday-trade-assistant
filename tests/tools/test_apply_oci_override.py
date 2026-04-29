"""tools/apply_oci_override.py tests.

Six tests covering the deep-merge utility used by oci/docker/entrypoint.py
to fold sub8_oci_overrides.json into configuration.json before main.py runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.apply_oci_override import apply_override


def _write(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def _read(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_top_level_scalar_overridden(tmp_path: Path):
    """Top-level scalar (e.g. wide_open_mode) gets replaced."""
    base_path = tmp_path / "base.json"
    over_path = tmp_path / "over.json"
    _write(base_path, {"wide_open_mode": False, "max_trades_per_cycle": 100})
    _write(over_path, {"wide_open_mode": True, "max_trades_per_cycle": 10000})

    summary = apply_override(base_path, over_path)
    out = _read(base_path)
    assert out["wide_open_mode"] is True
    assert out["max_trades_per_cycle"] == 10000
    assert summary["top_keys_changed"] == 2


def test_nested_dict_shallow_merged_not_replaced(tmp_path: Path):
    """A nested dict like gate_input_logging gets shallow-merged: override
    keys win, base keys preserved when not overridden."""
    base_path = tmp_path / "base.json"
    over_path = tmp_path / "over.json"
    _write(base_path, {
        "gate_input_logging": {"enabled": False, "max_size_mb": 500},
    })
    _write(over_path, {
        "gate_input_logging": {"enabled": True},
    })

    apply_override(base_path, over_path)
    out = _read(base_path)
    # 'enabled' overridden, 'max_size_mb' preserved
    assert out["gate_input_logging"] == {"enabled": True, "max_size_mb": 500}


def test_setup_block_preserves_params_only_flips_enabled(tmp_path: Path):
    """A setup block in the override (typically just `enabled: true`) MUST
    preserve all the detector parameters from the base config — flipping
    enabled is the only thing the override is supposed to do for that path."""
    base_path = tmp_path / "base.json"
    over_path = tmp_path / "over.json"
    _write(base_path, {
        "setups": {
            "expiry_pin_strike_reversal": {
                "enabled": False,
                "active_window_start": "13:30",
                "active_window_end": "15:15",
                "min_spot_distance_to_pin_pct": 0.3,
                "rsi_period": 14,
            },
            "gap_fade_short": {"enabled": True, "min_gap_pct_above_pdc": 1.5},
        },
    })
    _write(over_path, {
        "setups": {
            "expiry_pin_strike_reversal": {"enabled": True},
            "gap_fade_short": {"enabled": True},
        },
    })

    summary = apply_override(base_path, over_path)
    out = _read(base_path)
    expiry = out["setups"]["expiry_pin_strike_reversal"]
    assert expiry["enabled"] is True
    # Detector parameters MUST survive — these are critical for the detector
    # to function (no hardcoded defaults per CLAUDE.md rule 1).
    assert expiry["active_window_start"] == "13:30"
    assert expiry["active_window_end"] == "15:15"
    assert expiry["min_spot_distance_to_pin_pct"] == 0.3
    assert expiry["rsi_period"] == 14
    # Only one setup actually flipped (expiry_pin); gap_fade was already true.
    assert summary["setups_enabled_flipped"] == 1
    assert summary["total_setups_in_override"] == 2


def test_idempotent(tmp_path: Path):
    """Running the merge twice produces no further change."""
    base_path = tmp_path / "base.json"
    over_path = tmp_path / "over.json"
    _write(base_path, {
        "wide_open_mode": False,
        "setups": {"expiry_pin": {"enabled": False, "rsi_period": 14}},
    })
    _write(over_path, {
        "wide_open_mode": True,
        "setups": {"expiry_pin": {"enabled": True}},
    })

    apply_override(base_path, over_path)
    after_first = _read(base_path)
    summary2 = apply_override(base_path, over_path)
    after_second = _read(base_path)

    assert after_first == after_second
    assert summary2["setups_enabled_flipped"] == 0
    assert summary2["top_keys_changed"] == 0


def test_ascii_preserved_for_em_dashes(tmp_path: Path):
    """Em-dash characters in the base config (live-status comments etc.) must
    survive the round-trip as \\u2014 escapes — otherwise Windows cp1252
    readers (filters_setup.py) crash with UnicodeDecodeError."""
    base_path = tmp_path / "base.json"
    over_path = tmp_path / "over.json"
    _write(base_path, {
        "_live_status": "PRIMARY — Phase 7 OOS PASS",   # em-dash
        "setups": {"foo": {"enabled": False, "_notes": "Sources — OptionX"}},
    })
    _write(over_path, {"setups": {"foo": {"enabled": True}}})

    apply_override(base_path, over_path)
    raw_bytes = base_path.read_bytes()
    # Confirm — is in the file as the ASCII escape (NOT the raw 3-byte UTF-8 sequence)
    assert b"\\u2014" in raw_bytes, "em-dash must remain ASCII-escaped"
    assert b"\xe2\x80\x94" not in raw_bytes, "raw UTF-8 em-dash bytes must NOT appear"
    # And content survived
    out = _read(base_path)
    assert out["_live_status"] == "PRIMARY — Phase 7 OOS PASS"
    assert out["setups"]["foo"]["enabled"] is True
    assert out["setups"]["foo"]["_notes"] == "Sources — OptionX"


def test_missing_file_raises(tmp_path: Path):
    """Missing base or override raises FileNotFoundError so the caller
    (entrypoint.py) can fail loudly instead of silently running with an
    unmerged config."""
    base_path = tmp_path / "base.json"
    over_path = tmp_path / "over.json"
    _write(base_path, {"a": 1})
    # over_path does not exist
    with pytest.raises(FileNotFoundError):
        apply_override(base_path, over_path)
