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

    def __post_init__(self):
        start, end = self.active_window
        if start > end:
            raise ValueError(
                f"setup {self.name!r}: active_window_start <= active_window_end required; "
                f"got start={start} end={end}"
            )
