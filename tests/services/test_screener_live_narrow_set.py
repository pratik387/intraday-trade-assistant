"""Unit tests for _compute_build_df5_narrow_set helper.

Per design: spec 2026-05-21-backtest-bar-fetch-narrowing-design.md
Helper narrows the backtest build_df5_map loop to active_symbols when
tag_map is populated, else falls back to full core_symbols (pre-09:15).
"""
from services.screener_live import _compute_build_df5_narrow_set


def test_empty_active_falls_back_to_full_core_symbols():
    """Pre-09:15 / no setup universes built yet → must return full core_symbols
    so 5-arg universe builders (gap_fade, long_panic_gap_down) iterate the
    broader universe at the bar:09:15 trigger."""
    result = _compute_build_df5_narrow_set(set(), ["NSE:A", "NSE:B", "NSE:C"])
    assert result == {"NSE:A", "NSE:B", "NSE:C"}


def test_populated_active_intersects_with_core_symbols():
    """When tag_map has open universes, narrow to their intersection with
    core_symbols. Symbols active but NOT in core are dropped (the bar data
    loop only runs over core anyway)."""
    active = {"NSE:A", "NSE:B", "NSE:X"}  # X not in core
    core = ["NSE:A", "NSE:B", "NSE:C"]
    result = _compute_build_df5_narrow_set(active, core)
    assert result == {"NSE:A", "NSE:B"}


def test_active_disjoint_from_core_returns_empty():
    """If no active symbols are in core_symbols, the narrow set is empty.
    Downstream code (line 1443-1447 in screener_live.py) handles empty
    df5_by_symbol by returning early."""
    active = {"NSE:X", "NSE:Y"}
    core = ["NSE:A", "NSE:B", "NSE:C"]
    result = _compute_build_df5_narrow_set(active, core)
    assert result == set()
