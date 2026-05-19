"""Atomic read/write of setup blocks in config/configuration.json.

The config file is shared with the live screener process. Writes use a
temp-file + os.replace pattern so the screener never sees a partially-
written file.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

# Fields managed by walk-forward + circuit breaker tooling.
SETUP_METADATA_FIELDS = (
    "mechanism_tags",
    "mechanism_notes",
    "walk_forward_results",
    "cb_lookback_days",
    "cb_drawdown_threshold",
    "cb_min_trades_for_signal",
    "cb_state",
    "cb_disabled_at",
    "cb_disabled_reason",
    "position_size_multiplier",
)


class SetupNotFound(KeyError):
    """Raised when a setup name is not present under setups.* in config."""


def read_setup_block(config_path: Path, setup_name: str) -> Dict[str, Any]:
    """Return the dict at setups[setup_name] from the config file.

    Raises:
        SetupNotFound: if setup_name not under setups.
        FileNotFoundError: if config_path doesn't exist.
        json.JSONDecodeError: if config is malformed.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    setups = cfg.get("setups", {})
    if setup_name not in setups:
        raise SetupNotFound(setup_name)
    return setups[setup_name]


def write_setup_block_atomic(
    config_path: Path,
    setup_name: str,
    updates: Dict[str, Any],
) -> None:
    """Merge `updates` into setups[setup_name] and atomically rewrite config.

    Atomic: writes to a tempfile in the same directory, then os.replace()
    swaps it into place. Any concurrent reader sees either the old or new
    file, never a partial.

    Raises:
        SetupNotFound: if setup_name not under setups.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    setups = cfg.get("setups", {})
    if setup_name not in setups:
        raise SetupNotFound(setup_name)
    setups[setup_name].update(updates)
    cfg["setups"] = setups

    parent = config_path.parent
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{config_path.name}.", suffix=".tmp", dir=str(parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, str(config_path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
