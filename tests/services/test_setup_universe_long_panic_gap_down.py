"""Unit tests for setup_universe.long_panic_gap_down_universe.

Verifies the v1.1 universe contributor produces exactly the set of symbols
that would pass the broader long_panic_gap_down filter at the 09:15 bar.
"""
from datetime import date

import pandas as pd
import pytest

from services.setup_universe import long_panic_gap_down_universe


def _cfg():
    return {
        "allowed_cap_segments": ["small_cap", "mid_cap"],
        "gap_pct_max": -1.0,
        "dist_from_pdh_pct_max": -5.5,
        "broader_dist_from_pdl_pct_max": -1.25,
    }


def _bar(open_, high, low, close):
    return pd.DataFrame({
        "open": [open_], "high": [high], "low": [low], "close": [close], "volume": [50000],
    }, index=[pd.Timestamp("2026-05-20 09:15:00")])


def _daily(close, high, low):
    return pd.DataFrame({"close": [close], "high": [high], "low": [low]})


def test_qualifying_symbol_in_universe():
    """Symbol passing all three filters (gap, dist_pdh, broader dist_pdl) qualifies."""
    df5 = {"AAA": _bar(94.0, 94.5, 89.5, 91.0)}  # gap=-6%, dist_pdh=-13%, dist_pdl=-4.2%
    daily = {"AAA": _daily(close=100.0, high=105.0, low=95.0)}
    cap_map = {"AAA": "small_cap"}
    result = long_panic_gap_down_universe(df5, daily, date(2026, 5, 20), _cfg(), cap_map)
    assert result == {"AAA"}


def test_disallowed_cap_excluded():
    df5 = {"BBB": _bar(94.0, 94.5, 89.5, 91.0)}
    daily = {"BBB": _daily(close=100.0, high=105.0, low=95.0)}
    cap_map = {"BBB": "large_cap"}  # not in allowed
    result = long_panic_gap_down_universe(df5, daily, date(2026, 5, 20), _cfg(), cap_map)
    assert result == set()


def test_gap_too_shallow_excluded():
    # gap = -0.3% > -1% max → excluded
    df5 = {"CCC": _bar(99.7, 99.8, 90.0, 91.0)}
    daily = {"CCC": _daily(close=100.0, high=105.0, low=95.0)}
    cap_map = {"CCC": "mid_cap"}
    result = long_panic_gap_down_universe(df5, daily, date(2026, 5, 20), _cfg(), cap_map)
    assert result == set()


def test_dist_pdh_too_shallow_excluded():
    # close=101 vs pdh=105 → dist_pdh=-3.8% > -5.5% max → excluded
    df5 = {"DDD": _bar(99.0, 101.5, 98.5, 101.0)}
    daily = {"DDD": _daily(close=100.0, high=105.0, low=95.0)}
    cap_map = {"DDD": "small_cap"}
    result = long_panic_gap_down_universe(df5, daily, date(2026, 5, 20), _cfg(), cap_map)
    assert result == set()


def test_dist_pdl_too_shallow_excluded_from_broader():
    # close=94.5 vs pdl=95 → dist_pdl=-0.53% > -1.25% max → excluded
    df5 = {"EEE": _bar(94.0, 95.0, 93.0, 94.5)}
    daily = {"EEE": _daily(close=100.0, high=105.0, low=95.0)}
    cap_map = {"EEE": "small_cap"}
    result = long_panic_gap_down_universe(df5, daily, date(2026, 5, 20), _cfg(), cap_map)
    assert result == set()


def test_loose_dist_pdl_still_qualifies_for_broader():
    """A symbol with dist_pdl=-1.5% (looser than narrow Cell B's -3% floor)
    should still qualify for the BROADER universe — it counts toward density."""
    # close=93.575 vs pdl=95 → dist_pdl=-1.5% < -1.25% max ✓
    # gap = (94/100-1)*100 = -6% ✓
    # dist_pdh = (93.575/105-1)*100 = -10.88% ✓
    df5 = {"FFF": _bar(94.0, 94.5, 93.0, 93.575)}
    daily = {"FFF": _daily(close=100.0, high=105.0, low=95.0)}
    cap_map = {"FFF": "small_cap"}
    result = long_panic_gap_down_universe(df5, daily, date(2026, 5, 20), _cfg(), cap_map)
    assert result == {"FFF"}


def test_mixed_universe_returns_only_qualifying():
    df5 = {
        "QUAL": _bar(94.0, 94.5, 89.5, 91.0),    # passes
        "GAP_SHALLOW": _bar(99.7, 99.8, 90.0, 91.0),  # fails gap
        "WRONG_CAP": _bar(94.0, 94.5, 89.5, 91.0),    # fails cap
    }
    daily = {
        "QUAL": _daily(close=100.0, high=105.0, low=95.0),
        "GAP_SHALLOW": _daily(close=100.0, high=105.0, low=95.0),
        "WRONG_CAP": _daily(close=100.0, high=105.0, low=95.0),
    }
    cap_map = {"QUAL": "small_cap", "GAP_SHALLOW": "mid_cap", "WRONG_CAP": "large_cap"}
    result = long_panic_gap_down_universe(df5, daily, date(2026, 5, 20), _cfg(), cap_map)
    assert result == {"QUAL"}


def test_missing_daily_excluded():
    """No prior-day data for symbol → can't compute pdh/pdl/pdc → excluded."""
    df5 = {"GGG": _bar(94.0, 94.5, 89.5, 91.0)}
    daily = {}
    cap_map = {"GGG": "small_cap"}
    result = long_panic_gap_down_universe(df5, daily, date(2026, 5, 20), _cfg(), cap_map)
    assert result == set()


def test_empty_bar_excluded():
    df5 = {"HHH": pd.DataFrame()}
    daily = {"HHH": _daily(close=100.0, high=105.0, low=95.0)}
    cap_map = {"HHH": "small_cap"}
    result = long_panic_gap_down_universe(df5, daily, date(2026, 5, 20), _cfg(), cap_map)
    assert result == set()
