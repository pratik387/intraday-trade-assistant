"""Time-keyed event list derived from SetupRegistry.

Walked at each bar close to evolve TagMap state. Events sorted by
(time, ordering_key); ordering ensures build < open < close at same minute.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

from services.dispatch.setup_registry import SetupRegistry


# Conventional session-open time. session_start-trigger universes build here
# (before the 09:15 scan).
SESSION_OPEN = time(9, 15)


@dataclass(frozen=True)
class TransitionEvent:
    at: time
    kind: str
    setup: str


_KIND_ORDER = {"build_universe": 0, "open_window": 1, "close_window": 2}


class TransitionCalendar:
    def __init__(self, events: list):
        self._events = sorted(events, key=lambda e: (e.at, _KIND_ORDER[e.kind]))

    @classmethod
    def from_registry(cls, registry: SetupRegistry) -> "TransitionCalendar":
        events: list = []
        for spec in registry.enabled():
            build_at = (
                SESSION_OPEN
                if spec.universe_trigger.kind == "session_start"
                else spec.universe_trigger.at
            )
            events.append(TransitionEvent(at=build_at, kind="build_universe", setup=spec.name))
            events.append(TransitionEvent(at=spec.active_window[0], kind="open_window", setup=spec.name))
            events.append(TransitionEvent(at=spec.active_window[1], kind="close_window", setup=spec.name))
        return cls(events)

    def all_events(self) -> list:
        return list(self._events)

    def events_in(self, after: time, until: time) -> list:
        """Return events where after < ev.at <= until, in calendar order."""
        return [ev for ev in self._events if after < ev.at <= until]
