"""Single source of truth for setup metadata.

Loaded from configuration.json setups.* at startup. Every other module that
needs to know "what setups exist, what they need, when" reads from here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time
from typing import Optional


@dataclass(frozen=True)
class Trigger:
    """When a setup's universe is built.

    kind="session_start" — built at scanner warmup (no `at`).
    kind="bar" — built when the 5m bar at `at` closes.
    """
    kind: str
    at: Optional[time] = None

    @classmethod
    def session_start(cls) -> "Trigger":
        return cls(kind="session_start", at=None)

    @classmethod
    def bar(cls, at: time) -> "Trigger":
        return cls(kind="bar", at=at)


_BAR_RE = re.compile(r"^bar:(\d{2}):(\d{2})$")


def parse_trigger(spec: str) -> Trigger:
    """Parse a trigger spec string from config.

    Accepted: "session_start" | "bar:HH:MM" (zero-padded).
    """
    if spec == "session_start":
        return Trigger.session_start()
    m = _BAR_RE.match(spec)
    if not m:
        raise ValueError(f"malformed trigger spec {spec!r}; expected 'session_start' or 'bar:HH:MM'")
    return Trigger.bar(time(int(m.group(1)), int(m.group(2))))


@dataclass(frozen=True)
class SetupSpec:
    """All metadata for one setup. Single source of truth."""
    name: str
    enabled: bool
    detector_class_path: str        # "structures.gap_fade_short_structure.GapFadeShortStructure"
    universe_builder_path: str      # "services.setup_universe.gap_fade_universe"
    universe_trigger: Trigger
    active_window: tuple            # (start: time, end: time), inclusive
    raw_config: dict
    mode: str = "intraday"          # "intraday" (default, daemon) or "overnight" (cron-triggered)

    def __post_init__(self):
        start, end = self.active_window
        if start > end:
            raise ValueError(
                f"setup {self.name!r}: active_window_start <= active_window_end required; "
                f"got start={start} end={end}"
            )


import importlib


def _import_path(path: str):
    """Import 'module.submodule.Symbol' and return the Symbol."""
    module_path, _, symbol = path.rpartition(".")
    if not module_path:
        raise ImportError(f"invalid import path {path!r}; expected 'module.Symbol'")
    mod = importlib.import_module(module_path)
    if not hasattr(mod, symbol):
        raise ImportError(f"{path!r}: module {module_path!r} has no attribute {symbol!r}")
    return getattr(mod, symbol)


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


REQUIRED_KEYS = (
    "detector_class",
    "universe_builder",
    "universe_trigger",
    "active_window_start",
    "active_window_end",
)


class SetupRegistry:
    """Single source of truth for setup metadata."""

    def __init__(self, specs: dict):
        self._specs: dict = specs

    @classmethod
    def load_from_config(cls, root_config: dict) -> "SetupRegistry":
        setups = root_config.get("setups", {})
        specs: dict = {}
        for name, raw in setups.items():
            if not isinstance(raw, dict):
                continue  # skip non-dict entries (e.g., comments at this level)
            # Multi-day cross-sectional setups (horizon='multi_day') are
            # ranker-based (ranker_class/selection_mode) and run via the separate
            # `--mode multi_day` cron path, NOT this intraday/overnight DETECTOR
            # registry. They legitimately have no detector_class/universe_builder/
            # active_window, so skip them here — otherwise their presence in the
            # shared config crashes the intraday daemon's registry load.
            if str(raw.get("horizon")) == "multi_day":
                continue
            for k in REQUIRED_KEYS:
                if k not in raw:
                    raise ValueError(f"setup {name!r}: missing required key {k!r}")
            mode = raw.get("mode", "intraday")
            if mode not in ("intraday", "overnight"):
                raise ValueError(
                    f"setup {name!r}: mode must be 'intraday' or 'overnight', got {mode!r}"
                )
            specs[name] = SetupSpec(
                name=name,
                enabled=bool(raw.get("enabled", False)),
                detector_class_path=raw["detector_class"],
                universe_builder_path=raw["universe_builder"],
                universe_trigger=parse_trigger(raw["universe_trigger"]),
                active_window=(_parse_hhmm(raw["active_window_start"]), _parse_hhmm(raw["active_window_end"])),
                raw_config=raw,
                mode=mode,
            )
        return cls(specs)

    def enabled(self) -> list:
        return [s for s in self._specs.values() if s.enabled]

    def get_active_setups(self, mode: str) -> list:
        """Return enabled setups matching the given mode ('intraday' or 'overnight')."""
        if mode not in ("intraday", "overnight"):
            raise ValueError(f"mode must be 'intraday' or 'overnight', got {mode!r}")
        return [s for s in self._specs.values() if s.enabled and s.mode == mode]

    def get(self, name: str) -> SetupSpec:
        return self._specs[name]

    def validate(self) -> None:
        """Import every enabled setup's detector_class + universe_builder. Fail fast."""
        for spec in self.enabled():
            _import_path(spec.detector_class_path)
            _import_path(spec.universe_builder_path)
