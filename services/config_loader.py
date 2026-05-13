"""Base config loader + runtime override helpers.

Relocated from `pipelines/base_pipeline.py` as part of the 2026-04-30
architecture refactor (Phase C — flatten the pipelines layer). The
category-pipeline scaffolding (BasePipeline ABC, BREAKOUT/LEVEL/REVERSION/
MOMENTUM subclasses, ScreeningResult / QualityResult / etc. result
dataclasses) is dead code for the current detector roster — every active
sub7+sub8 setup goes through the orchestrator's SUB7 fast path. This
module preserves only the bits that LIVE callers (main.py, every detector
via `_is_wide_open`, screener_live, orchestrator) actually use.

Public API:
  - load_base_config() -> dict — module-level cached read of
    `config/pipelines/base_config.json`. Used by every detector's
    `_is_wide_open()` and by trade_decision_gate.
  - set_base_config_override(key, value) — used by main.py to inject
    runtime risk_per_trade_rupees from capital manager.
  - require_config(config, *keys) — nested-key path lookup that raises
    ConfigurationError on missing keys.
  - ConfigurationError — raised on any missing required config.

The `load_pipeline_config(category)` helper is intentionally NOT
preserved — the category pipelines it served are being deleted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from config.logging_config import get_agent_logger


logger = get_agent_logger()


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


_BASE_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def set_base_config_override(key: str, value: Any) -> None:
    """Override a value in the cached base config.

    Allows main.py to set runtime values (e.g. `risk_per_trade_rupees` from
    the capital manager) that will be picked up by all consumers reading
    `load_base_config()` afterward.

    MUST be called AFTER `load_base_config()` has been called at least once
    (we force-load if not).

    Args:
        key: Config key to override (e.g., "risk_per_trade_rupees")
        value: New value to set
    """
    global _BASE_CONFIG_CACHE
    if _BASE_CONFIG_CACHE is None:
        load_base_config()
    _BASE_CONFIG_CACHE[key] = value
    logger.info(f"CONFIG_OVERRIDE: Set {key}={value}")


def load_base_config() -> Dict[str, Any]:
    """Load base configuration (universal settings shared across all detectors).

    Reads `config/pipelines/base_config.json` once, caches the result
    module-globally. Subsequent calls return the cached dict.

    Returns:
        Base configuration dict (or empty dict if file missing — logged
        as a warning, not raised).

    Raises:
        ConfigurationError: only if the file is present but malformed JSON.
    """
    global _BASE_CONFIG_CACHE
    if _BASE_CONFIG_CACHE is not None:
        return _BASE_CONFIG_CACHE

    # services/ is one level under repo root; base_config.json lives at
    # config/pipelines/base_config.json.
    repo_root = Path(__file__).resolve().parent.parent
    config_file = repo_root / "config" / "pipelines" / "base_config.json"

    if not config_file.exists():
        logger.warning(f"Base config not found: {config_file}, using empty base config")
        _BASE_CONFIG_CACHE = {}
        return _BASE_CONFIG_CACHE

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            _BASE_CONFIG_CACHE = json.load(f)
        logger.debug(f"Loaded base config: {len(_BASE_CONFIG_CACHE)} keys")
        return _BASE_CONFIG_CACHE
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_file}: {e}")
        raise ConfigurationError(f"Invalid JSON in {config_file}: {e}")


def is_wide_open_for_setup(setup_type: Optional[str]) -> bool:
    """Return wide_open flag for a specific setup_type.

    Resolution order:
      1. `setups.<setup_type>.wide_open` in configuration.json (per-setup override)
      2. top-level `wide_open_mode` in configuration.json or base_config.json
         (legacy global fallback)

    Empty / unknown setup_type returns the top-level flag.

    Per-setup wide_open enables a mixed-mode OCI capture: production-ready
    setups (wide_open=false) keep tight entry-zone / duplicate-guard /
    min_bars semantics; research setups (wide_open=true) bypass them so
    every detector signal becomes a captured trade row.
    """
    try:
        from config.filters_setup import load_filters
        cfg = load_filters() or {}
    except Exception:
        cfg = {}
    global_flag = bool(cfg.get("wide_open_mode", load_base_config().get("wide_open_mode", False)))
    if not setup_type:
        return global_flag
    setup_cfg = ((cfg.get("setups") or {}).get(setup_type)) or {}
    if "wide_open" in setup_cfg:
        return bool(setup_cfg["wide_open"])
    return global_flag


def require_config(config: Dict[str, Any], *keys: str) -> Any:
    """Navigate nested config and raise ConfigurationError on missing keys.

    Args:
        config: Configuration dict
        keys: Path to required key (e.g., "screening", "time_windows", "morning_start")

    Returns:
        The value at the specified path.

    Raises:
        ConfigurationError if any key in the path is missing.
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            path_str = " -> ".join(keys)
            raise ConfigurationError(f"Missing required config key: {path_str}")
        current = current[key]
    return current
