"""Apply an OCI override JSON onto a base configuration.json (in-place merge).

Used by oci/docker/entrypoint.py at OCI startup to fold the contents of
`config/sub8_oci_overrides.json` into `config/configuration.json` BEFORE
`main.py` runs. Without this step, the OCI run would read the production
`configuration.json` as-is — wide_open_mode=false, only gap_fade_short
enabled — and the gauntlet would see no signal from the new sub8 detectors.

Also usable for local wide-open smoke runs:
    python tools/apply_oci_override.py \\
        --base config/configuration.json \\
        --override config/sub8_oci_overrides.json

Behavior:
  - Top-level keys present in the override REPLACE the base value.
  - For nested dicts at the top level (gate_input_logging, expiry_day_exclusion,
    setups[name], etc.), values are SHALLOW-MERGED (override keys win, base
    keys preserved if not overridden).
  - File is written back with `ensure_ascii=True` to preserve the existing
    em-dash escaping convention (\\u2014) — Windows cp1252-readable.
  - Idempotent: re-running the same merge is a no-op.

Per CLAUDE.md rule 1: no hardcoded defaults; all values flow from JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _shallow_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Return base with each key from over either replacing it (scalar / list)
    or shallow-merging it (dict). For setups[name] we want to preserve the
    base config's parameters (active_window, thresholds, etc.) and only
    override 'enabled' when the override says so — that's a shallow merge."""
    out: Dict[str, Any] = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def apply_override(base_path: Path, override_path: Path) -> Dict[str, int]:
    """Merge override_path into base_path in place.

    Returns a summary dict for logging:
      {top_keys_changed, setups_enabled_flipped, total_setups_in_override}.
    """
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not override_path.exists():
        raise FileNotFoundError(f"Override config not found: {override_path}")

    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(override_path, encoding="utf-8") as f:
        over = json.load(f)

    summary = {
        "top_keys_changed": 0,
        "setups_enabled_flipped": 0,
        "total_setups_in_override": 0,
    }

    # Top-level keys: shallow-merge nested dicts; replace scalars/lists.
    # `setups` is handled specially below (per-setup shallow merge).
    for k, v in over.items():
        if k == "setups":
            continue
        if k in base and base[k] == v:
            continue
        summary["top_keys_changed"] += 1
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v

    # `setups` requires per-setup shallow merge: we want to keep all the
    # detector parameters (active_window_start, thresholds, etc.) from the
    # base and ONLY override the `enabled` flag (and any other fields the
    # override explicitly sets, which today is just `enabled`).
    base_setups = base.setdefault("setups", {})
    for name, ov_cfg in over.get("setups", {}).items():
        summary["total_setups_in_override"] += 1
        if name in base_setups and isinstance(base_setups[name], dict) \
                and isinstance(ov_cfg, dict):
            old_enabled = bool(base_setups[name].get("enabled", False))
            base_setups[name] = {**base_setups[name], **ov_cfg}
            if bool(base_setups[name].get("enabled", False)) != old_enabled:
                summary["setups_enabled_flipped"] += 1
        else:
            base_setups[name] = ov_cfg
            summary["setups_enabled_flipped"] += 1

    # Write back. ensure_ascii=True preserves the file's existing escape
    # convention (em-dashes etc. as \\u2014) so anything reading the file
    # later with cp1252 (Windows default) can still parse it.
    with open(base_path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2, ensure_ascii=True)
        f.write("\n")

    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply an OCI override JSON onto a base configuration.json (in-place merge)."
    )
    parser.add_argument(
        "--base", type=Path, required=True,
        help="Path to the base configuration.json (modified in place).",
    )
    parser.add_argument(
        "--override", type=Path, required=True,
        help="Path to the override JSON to fold on top.",
    )
    args = parser.parse_args(argv)

    summary = apply_override(args.base, args.override)
    print(
        f"OK: merged {args.override} -> {args.base}: "
        f"top_keys_changed={summary['top_keys_changed']} "
        f"setups_enabled_flipped={summary['setups_enabled_flipped']} "
        f"total_setups_in_override={summary['total_setups_in_override']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
