"""Per-day cross-symbol density tracker for regime guards.

Detectors that need a UNIVERSE-wide regime guard cannot count cross-symbol
qualifications themselves — the production screener/orchestrator processes
symbols sequentially. This module is the shared counter all detectors mutate
and query.

Pattern:
    1. Each detector calls `note(setup_key, session_date, symbol)` whenever
       a symbol passes the detector's BROADER filter (regardless of whether
       the narrow trigger fires). Idempotent per symbol/date.
    2. Before firing the narrow trigger, the detector calls
       `get_density(setup_key, session_date)`. If > configured threshold,
       the detector suppresses the fire.

Sequential-ordering note (v1.0 limitation):
    Symbols are processed sequentially within a bar. The Nth symbol sees
    density=N at its decision point, not the final end-of-bar count. On a
    severe panic day with, say, 200 broader matches, the first ~80 (= threshold)
    fires would slip through before the guard activates. This is a known
    bounded leak; a two-pass screener (count first, decide second) would
    eliminate it. Tracked as v1.1.

State is per-process. Tests should call `reset()` in their fixtures.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from threading import Lock
from typing import Dict, Set, Tuple


_Key = Tuple[str, date]


class _Tracker:
    def __init__(self) -> None:
        self._counts: Dict[_Key, int] = defaultdict(int)
        self._noted_symbols: Dict[_Key, Set[str]] = defaultdict(set)
        self._lock = Lock()

    def note(self, setup_key: str, d: date, symbol: str) -> None:
        """Record that `symbol` passed the broader filter for `setup_key`
        on date `d`. Idempotent — re-noting the same triple doesn't bump count.
        """
        with self._lock:
            key = (setup_key, d)
            if symbol not in self._noted_symbols[key]:
                self._noted_symbols[key].add(symbol)
                self._counts[key] += 1

    def get_density(self, setup_key: str, d: date) -> int:
        with self._lock:
            return self._counts.get((setup_key, d), 0)

    def reset(self) -> None:
        """Clear all state. Intended for test isolation."""
        with self._lock:
            self._counts.clear()
            self._noted_symbols.clear()

    def reset_date(self, d: date) -> None:
        with self._lock:
            keys_to_drop = [k for k in list(self._counts.keys()) if k[1] == d]
            for k in keys_to_drop:
                self._counts.pop(k, None)
                self._noted_symbols.pop(k, None)


_GLOBAL = _Tracker()


def note(setup_key: str, d: date, symbol: str) -> None:
    _GLOBAL.note(setup_key, d, symbol)


def get_density(setup_key: str, d: date) -> int:
    return _GLOBAL.get_density(setup_key, d)


def reset() -> None:
    _GLOBAL.reset()


def reset_date(d: date) -> None:
    _GLOBAL.reset_date(d)
