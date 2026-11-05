"""Results-day size tilt (P&L plan step 4, 2026-11-02; built OFF 2026-11-05).

Audit 2026-11-01/02: up_spike_fade_short trades taken on results days made
about half of the honest backtest's P&L at PF ~3, while the rest were about
break-even after live costs. The tilt sizes results-day trades up and the
rest down, keyed on the tag services/results_context.py computes from the
results calendar.

That tag is only as point-in-time as the calendar.
- Live sees what the start-of-day refresh wrote. A meeting the scrape missed
  tags 'none' and sizes down.
- A backtest reads the repaired history, so on the same day it would tag
  board_meeting_today and size up. The calendar has had multi-week holes:
  event_feed.refresh_lookback_days explains the Aug 2026 one.
- A tilted backtest therefore overstates live results-day size wherever
  live coverage was missing.

The caller multiplies BOTH the setup's notional and its circuit-loss budget
by the returned number. Scaling only the notional would change nothing on
most trades: the circuit cap set the size on all five live up_spike trades
of 2026-11-03..05. The max notional clamp, the min-notional floor and the
capital manager's per-trade margin cap still apply.

Per-setup config. Absent, null or disabled -> 1.0, sizing unchanged:
  "results_size_tilt": {"enabled": false, "max_multiplier": 2.0,
                        "multipliers": {"results_preopen": 2.0, "board_meeting_today": 2.0,
                                        "none": 0.5, "unknown": 0.5}}

Rules:
- Every tag results_context() can return must have a multiplier.
- Each multiplier must be a positive number <= max_multiplier, a second,
  independent value that catches a typo such as 20.0 for 2.0.
- Anything else is a config error.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from services.results_context import ALL_TAGS


class ResultsSizeTiltConfigError(ValueError):
    """Raised when a setup's results_size_tilt block is malformed."""


def _positive_number(v: Any) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def validate_results_size_tilt(cfg: Any, setup_name: str) -> None:
    """Raise unless `cfg` is absent (None) or a complete, well-formed tilt block."""
    if cfg is None:
        return
    where = f"setups.{setup_name}.results_size_tilt"
    if not isinstance(cfg, dict):
        raise ResultsSizeTiltConfigError(f"{where} must be an object")
    if not isinstance(cfg.get("enabled"), bool):
        raise ResultsSizeTiltConfigError(f"{where}.enabled must be true/false")
    ceiling = cfg.get("max_multiplier")
    if not _positive_number(ceiling):
        raise ResultsSizeTiltConfigError(f"{where}.max_multiplier must be a positive number, got {ceiling!r}")
    mult = cfg.get("multipliers")
    if not isinstance(mult, dict):
        raise ResultsSizeTiltConfigError(f"{where}.multipliers must be an object")
    keys = {k for k in mult if not str(k).startswith("_")}
    if keys != ALL_TAGS:
        raise ResultsSizeTiltConfigError(
            f"{where}.multipliers must name exactly {sorted(ALL_TAGS)}; "
            f"missing {sorted(ALL_TAGS - keys)}, unknown {sorted(keys - ALL_TAGS)}")
    for tag in sorted(ALL_TAGS):
        v = mult[tag]
        if not _positive_number(v) or v > ceiling:
            raise ResultsSizeTiltConfigError(
                f"{where}.multipliers.{tag} must be a positive number <= max_multiplier {ceiling}, got {v!r}")


def results_size_multiplier(cfg: Any, tag: Optional[str], setup_name: str) -> float:
    """Size multiplier for one plan: 1.0 when the tilt is absent or disabled,
    else the configured multiplier of `tag` (a results_context() value)."""
    validate_results_size_tilt(cfg, setup_name)
    if cfg is None or not cfg["enabled"]:
        return 1.0
    if tag not in ALL_TAGS:
        raise ResultsSizeTiltConfigError(
            f"setups.{setup_name}.results_size_tilt: results tag {tag!r} is not one of {sorted(ALL_TAGS)}")
    return float(cfg["multipliers"][tag])
