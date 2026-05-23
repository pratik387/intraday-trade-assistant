"""Calendar utilities for paper-variant gating and calendar-conditional analytics.

Used by:
- close_dn_overnight_long Variant B paper-validation gate
  (specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md §Task 8)

Provides:
- is_expiry_week(d, expiry_dates): True if d is in Mon-Thu of week containing any monthly F&O expiry
- trading_day_of_month(d, holidays): 1-indexed trading-day rank within month (skips weekends + NSE holidays)
- passes_close_dn_variant_b(d): the composite Variant B gate

Lazy-loads:
- Monthly F&O expiry dates from `data/futures_basis/2023_2026_basis.parquet[expiry_date]`
- NSE holidays from `assets/nse_holidays.json` (list of {tradingDate: "DD-Mon-YYYY"} dicts)

All timestamps IST-naive (no tzinfo) per CLAUDE.md.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPIRY_PARQUET = _REPO_ROOT / "data" / "futures_basis" / "2023_2026_basis.parquet"
_HOLIDAYS_JSON = _REPO_ROOT / "assets" / "nse_holidays.json"

# weekday() = 0..6 with Mon=0; this constant clarifies the explicit-exclude rule
_THURSDAY = 3


def _parse_trading_date(s: str) -> date:
    """Parse 'DD-Mon-YYYY' (NSE convention) → date."""
    return datetime.strptime(s, "%d-%b-%Y").date()


@lru_cache(maxsize=1)
def load_expiry_dates() -> FrozenSet[date]:
    """Load monthly F&O expiry dates from `data/futures_basis/2023_2026_basis.parquet`.

    The parquet's `expiry_date` column already includes NSE holiday adjustments
    (e.g., post-2025-09-01 Tuesday-shift per circular FAOP68747). Caching avoids
    re-reading the ~MB parquet on every detector fire.
    """
    if not _EXPIRY_PARQUET.exists():
        raise FileNotFoundError(f"F&O expiry parquet missing: {_EXPIRY_PARQUET}")
    df = pd.read_parquet(_EXPIRY_PARQUET, columns=["expiry_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    if df["expiry_date"].dt.tz is not None:
        df["expiry_date"] = df["expiry_date"].dt.tz_localize(None)
    return frozenset(df["expiry_date"].dt.date.unique())


@lru_cache(maxsize=1)
def load_nse_holidays() -> FrozenSet[date]:
    """Load NSE trading holidays from `assets/nse_holidays.json` (list of dicts).

    Schema: [{"tradingDate": "26-Jan-2023", "weekDay": "Thursday", ...}, ...]
    """
    if not _HOLIDAYS_JSON.exists():
        raise FileNotFoundError(f"NSE holidays JSON missing: {_HOLIDAYS_JSON}")
    with open(_HOLIDAYS_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected nse_holidays.json schema: expected list, got {type(raw)}")
    out = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        td = entry.get("tradingDate")
        if td:
            try:
                out.add(_parse_trading_date(td))
            except ValueError:
                continue
    return frozenset(out)


def is_expiry_week(
    d: date,
    expiry_dates: Optional[FrozenSet[date]] = None,
) -> bool:
    """True iff d is in Mon-Thu of the ISO calendar week containing any monthly F&O expiry.

    The ISO calendar week is used (weeks start on Monday). Friday/Saturday/Sunday return
    False even if the same ISO week contains an expiry — these days don't carry the
    pre-expiry institutional flow signature for our Variant B mechanism.

    Args:
        d: signal date (any weekday accepted; Fri-Sun always return False)
        expiry_dates: frozenset of monthly expiry dates. If None, loads from parquet (cached).

    Returns:
        True iff d.weekday() in {Mon, Tue, Wed, Thu} AND ISO-week(d) == ISO-week(any expiry).
    """
    if d.weekday() > _THURSDAY:
        return False
    if expiry_dates is None:
        expiry_dates = load_expiry_dates()
    d_yr, d_wk, _ = d.isocalendar()
    for e in expiry_dates:
        e_yr, e_wk, _ = e.isocalendar()
        if (d_yr, d_wk) == (e_yr, e_wk):
            return True
    return False


def trading_day_of_month(
    d: date,
    holidays: Optional[FrozenSet[date]] = None,
) -> int:
    """Return 1-indexed trading-day-of-month rank for d.

    Counts weekdays (Mon-Fri) within d's calendar month from the 1st up to and
    including d, excluding NSE holidays. Returns 0 if d itself is not a trading
    day (weekend or holiday) — caller should treat 0 as "non-trading day."

    Args:
        d: any date
        holidays: NSE holidays frozenset. If None, loads from JSON (cached).

    Returns:
        Integer >= 0. Typical range 1-23 across a calendar month.
    """
    if holidays is None:
        holidays = load_nse_holidays()
    if d.weekday() >= 5 or d in holidays:
        return 0
    rank = 0
    day = date(d.year, d.month, 1)
    while day <= d:
        if day.weekday() < 5 and day not in holidays:
            rank += 1
        day = day + timedelta(days=1)
    return rank


def passes_close_dn_variant_b(
    d: date,
    expiry_dates: Optional[FrozenSet[date]] = None,
    holidays: Optional[FrozenSet[date]] = None,
) -> bool:
    """Variant B paper-validation gate for close_dn_overnight_long.

    Rule (per specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md §Task 8):

        (dow == Monday OR is_expiry_week OR trading_day_of_month >= 21)
         AND dow != Thursday

    Calendar-conditioning analysis (2026-05-22) found this gate lifts close_dn_overnight_long
    Holdout PF from 1.59 → 3.04 (+91%), with n=1,663 (well above 30-trade floor).

    The Thursday-exclude is explicit because Thursday is the pre-Friday-expiry day where
    institutional supply tends to OFFSET the weekend-risk-reversal pattern that drives the
    setup's overnight LONG edge.

    Args:
        d: signal date (IST-naive)
        expiry_dates: optional override (testing)
        holidays: optional override (testing)

    Returns:
        True iff d satisfies Variant B composite gate.
    """
    dow = d.weekday()
    if dow == _THURSDAY:
        return False
    if dow == 0:  # Monday
        return True
    if is_expiry_week(d, expiry_dates):
        return True
    if trading_day_of_month(d, holidays) >= 21:
        return True
    return False
