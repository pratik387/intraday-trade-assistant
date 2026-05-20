"""Tests for tools.methodology.setup_metadata."""
import json
import os
from pathlib import Path

import pytest

from tools.methodology.setup_metadata import (
    read_setup_block,
    write_setup_block_atomic,
    SetupNotFound,
    SETUP_METADATA_FIELDS,
)


def _write_config(tmp_path: Path, content: dict) -> Path:
    """Write a minimal configuration.json to a temp path."""
    p = tmp_path / "configuration.json"
    p.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return p


def test_read_setup_block_existing(tmp_path):
    """Read a known setup block by name."""
    cfg = {
        "setups": {
            "my_setup": {
                "enabled": True,
                "detector_class": "structures.my.MyStruct",
                "extra_param": 42,
            }
        }
    }
    cfg_path = _write_config(tmp_path, cfg)

    block = read_setup_block(cfg_path, "my_setup")

    assert block["enabled"] is True
    assert block["extra_param"] == 42


def test_read_setup_block_missing_raises(tmp_path):
    """Unknown setup name raises SetupNotFound."""
    cfg_path = _write_config(tmp_path, {"setups": {}})

    with pytest.raises(SetupNotFound):
        read_setup_block(cfg_path, "does_not_exist")


def test_write_setup_block_atomic_merges_updates(tmp_path):
    """Updates merge into existing block, preserving unrelated fields."""
    cfg = {
        "setups": {
            "my_setup": {
                "enabled": True,
                "existing_field": "preserve_me",
            }
        }
    }
    cfg_path = _write_config(tmp_path, cfg)

    write_setup_block_atomic(cfg_path, "my_setup", {
        "mechanism_tags": ["FII_net_flow_positive_30d"],
        "cb_state": "enabled",
    })

    reloaded = read_setup_block(cfg_path, "my_setup")
    assert reloaded["existing_field"] == "preserve_me"
    assert reloaded["enabled"] is True
    assert reloaded["mechanism_tags"] == ["FII_net_flow_positive_30d"]
    assert reloaded["cb_state"] == "enabled"


def test_write_setup_block_atomic_temp_file_cleaned_on_error(tmp_path):
    """If write fails mid-flight, no orphan tempfile is left behind."""
    cfg_path = _write_config(tmp_path, {"setups": {"my_setup": {"enabled": True}}})

    class NotSerializable:
        pass

    with pytest.raises(TypeError):
        write_setup_block_atomic(cfg_path, "my_setup", {"bad": NotSerializable()})

    leftover = list(tmp_path.glob(".configuration.json.*.tmp"))
    assert leftover == [], f"orphan tempfile(s): {leftover}"


def test_write_setup_block_atomic_setup_not_found(tmp_path):
    """Writing to missing setup raises SetupNotFound."""
    cfg_path = _write_config(tmp_path, {"setups": {}})
    with pytest.raises(SetupNotFound):
        write_setup_block_atomic(cfg_path, "missing", {"x": 1})
