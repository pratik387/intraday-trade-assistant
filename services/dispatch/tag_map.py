"""Per-symbol active-detector state.

Mutated only by DispatchPlanner walking TransitionCalendar. Read by
FetchScopeManager and DispatchPlanner to build per-bar fetch + dispatch
plans.
"""
from __future__ import annotations


class TagMap:
    def __init__(self):
        # setup_name -> set[sym]
        self._universe: dict = {}
        # setup_name -> bool (is its active window currently open?)
        self._window_open: dict = {}

    def add_universe(self, setup: str, syms: set) -> None:
        """Register the universe of qualifying symbols for a setup.

        Called once when the universe builder fires (session_start or bar:HH:MM
        trigger). Does NOT open the active window.
        """
        self._universe[setup] = set(syms)

    def open_window(self, setup: str) -> None:
        """Mark this setup's active window as open. Symbols in its universe
        become eligible for dispatch."""
        self._window_open[setup] = True

    def close_window(self, setup: str) -> None:
        """Mark this setup's active window as closed. No-op if never opened."""
        self._window_open[setup] = False

    def active_tags(self, sym: str) -> set:
        """Detector names whose universe contains `sym` AND whose window is open."""
        return {
            setup
            for setup, univ in self._universe.items()
            if sym in univ and self._window_open.get(setup, False)
        }

    def active_symbols(self) -> set:
        """Union of universes whose windows are currently open."""
        out: set = set()
        for setup, univ in self._universe.items():
            if self._window_open.get(setup, False):
                out |= univ
        return out
