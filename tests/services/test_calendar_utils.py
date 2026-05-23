"""Tests for services.calendar_utils.

Covers Variant B gate semantics for close_dn_overnight_long paper-validation
per specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md §Task 8.
"""
from datetime import date

import pytest

from services.calendar_utils import (
    is_expiry_week,
    passes_close_dn_variant_b,
    trading_day_of_month,
)


# Test fixtures — explicit expiry/holiday sets so tests don't depend on on-disk parquet
# 2024-01-25 was the last Thursday of January 2024 (monthly F&O expiry, pre-Tuesday-shift era)
JAN_2024_EXPIRY = frozenset([date(2024, 1, 25)])
# 2024-05-30 was the last Thursday of May 2024
MAY_2024_EXPIRY = frozenset([date(2024, 5, 30)])
NO_HOLIDAYS = frozenset()
# Common 2024 NSE holidays (subset for tests)
SAMPLE_HOLIDAYS_2024 = frozenset([
    date(2024, 1, 26),   # Republic Day (Friday)
    date(2024, 3, 8),    # Holi (Friday)
    date(2024, 3, 25),   # Holi (Monday)
    date(2024, 3, 29),   # Good Friday
])


# ---------------------------------------------------------------------------
# is_expiry_week
# ---------------------------------------------------------------------------

def test_is_expiry_week_monday_of_expiry_week_true():
    # 2024-01-22 is Monday of the week containing Thursday 2024-01-25
    assert is_expiry_week(date(2024, 1, 22), JAN_2024_EXPIRY)


def test_is_expiry_week_thursday_expiry_day_itself_true():
    # The Thursday expiry day itself counts as expiry-week
    assert is_expiry_week(date(2024, 1, 25), JAN_2024_EXPIRY)


def test_is_expiry_week_tuesday_wednesday_true():
    assert is_expiry_week(date(2024, 1, 23), JAN_2024_EXPIRY)
    assert is_expiry_week(date(2024, 1, 24), JAN_2024_EXPIRY)


def test_is_expiry_week_friday_after_expiry_false():
    # Friday is excluded from Variant B's expiry-week definition (Mon-Thu only)
    assert not is_expiry_week(date(2024, 1, 26), JAN_2024_EXPIRY)


def test_is_expiry_week_weekend_false():
    assert not is_expiry_week(date(2024, 1, 27), JAN_2024_EXPIRY)  # Sat
    assert not is_expiry_week(date(2024, 1, 28), JAN_2024_EXPIRY)  # Sun


def test_is_expiry_week_prior_week_false():
    # 2024-01-15 is in week PRIOR to expiry week
    assert not is_expiry_week(date(2024, 1, 15), JAN_2024_EXPIRY)


def test_is_expiry_week_next_week_false():
    # 2024-01-29 is Monday of week AFTER expiry
    assert not is_expiry_week(date(2024, 1, 29), JAN_2024_EXPIRY)


def test_is_expiry_week_multiple_expiries_in_set():
    # When expiry set has multiple months, any matching week qualifies
    expiries = frozenset([date(2024, 1, 25), date(2024, 2, 29), date(2024, 3, 28)])
    assert is_expiry_week(date(2024, 1, 22), expiries)
    assert is_expiry_week(date(2024, 2, 26), expiries)
    assert is_expiry_week(date(2024, 3, 25), expiries)
    # Mid-month between expiries
    assert not is_expiry_week(date(2024, 2, 5), expiries)


# ---------------------------------------------------------------------------
# trading_day_of_month
# ---------------------------------------------------------------------------

def test_trading_day_of_month_first_trading_day():
    # 2024-01-01 was Monday (no holiday) → tdom = 1
    assert trading_day_of_month(date(2024, 1, 1), NO_HOLIDAYS) == 1


def test_trading_day_of_month_consecutive_weekdays():
    # 2024-01-01 Mon=1, Tue=2, Wed=3, Thu=4, Fri=5
    assert trading_day_of_month(date(2024, 1, 2), NO_HOLIDAYS) == 2
    assert trading_day_of_month(date(2024, 1, 5), NO_HOLIDAYS) == 5


def test_trading_day_of_month_skips_weekend():
    # 2024-01-08 Mon = tdom 6 (Sat/Sun Jan 6-7 not counted)
    assert trading_day_of_month(date(2024, 1, 8), NO_HOLIDAYS) == 6


def test_trading_day_of_month_skips_holiday():
    # 2024-01-26 is Republic Day (holiday). The Monday after = 2024-01-29.
    # Without 26 in holidays: tdom(29) = 21 (1-5, 8-12, 15-19, 22-26, 29 = 5+5+5+5+1)
    # WITH 26 in holidays: tdom(29) = 20 (skips 26)
    assert trading_day_of_month(date(2024, 1, 29), NO_HOLIDAYS) == 21
    assert trading_day_of_month(date(2024, 1, 29), SAMPLE_HOLIDAYS_2024) == 20


def test_trading_day_of_month_weekend_returns_zero():
    # Saturday and Sunday are not trading days
    assert trading_day_of_month(date(2024, 1, 6), NO_HOLIDAYS) == 0  # Sat
    assert trading_day_of_month(date(2024, 1, 7), NO_HOLIDAYS) == 0  # Sun


def test_trading_day_of_month_holiday_itself_returns_zero():
    # Holiday itself is not a trading day
    assert trading_day_of_month(date(2024, 1, 26), SAMPLE_HOLIDAYS_2024) == 0


def test_trading_day_of_month_resets_per_month():
    # 2024-02-01 was Thursday → tdom 1 within Feb
    assert trading_day_of_month(date(2024, 2, 1), NO_HOLIDAYS) == 1
    # 2024-02-05 Mon → tdom 3 (Feb 1=Thu, 2=Fri, 5=Mon)
    assert trading_day_of_month(date(2024, 2, 5), NO_HOLIDAYS) == 3


# ---------------------------------------------------------------------------
# passes_close_dn_variant_b — the composite gate
# ---------------------------------------------------------------------------

def test_variant_b_thursday_always_excluded():
    # Thursday explicit exclude even on a Monday-rule technicality
    # (Thursday is also is_expiry_week=True for expiry Thursday, but the rule excludes it)
    assert not passes_close_dn_variant_b(date(2024, 1, 25), JAN_2024_EXPIRY, NO_HOLIDAYS)


def test_variant_b_monday_passes():
    # Monday qualifies via dow==0 branch, regardless of expiry-week / tdom
    # 2024-02-05 Mon, not in expiry week, tdom=3 (low) — still passes via Monday rule
    expiry_far = frozenset([date(2024, 1, 25)])
    assert passes_close_dn_variant_b(date(2024, 2, 5), expiry_far, NO_HOLIDAYS)


def test_variant_b_expiry_week_tuesday_wednesday_pass():
    # Tuesday and Wednesday of expiry week qualify via is_expiry_week
    assert passes_close_dn_variant_b(date(2024, 1, 23), JAN_2024_EXPIRY, NO_HOLIDAYS)  # Tue
    assert passes_close_dn_variant_b(date(2024, 1, 24), JAN_2024_EXPIRY, NO_HOLIDAYS)  # Wed


def test_variant_b_late_month_friday_passes_via_tdom():
    # 2024-02-23 Fri, tdom=17 — NOT >= 21, NOT Monday, NOT expiry week (expiry Feb 29)
    # Actually Feb 29 Thu = expiry. 2024-02-23 Fri is week BEFORE expiry.
    # tdom: Feb 1=Thu(1), 2=Fri(2), 5-9=3-7, 12-16=8-12, 19-23=13-17.
    # So 2024-02-23 has tdom=17 < 21, not expiry week, not Mon → False
    expiry = frozenset([date(2024, 2, 29)])
    assert not passes_close_dn_variant_b(date(2024, 2, 23), expiry, NO_HOLIDAYS)
    # 2024-02-28 Wed has tdom=20 (last weekday before Feb 29 expiry)
    # but it IS in expiry week (Feb 29 expiry, Feb 26 Mon-Feb 28 Wed are Mon-Wed of that week)
    assert passes_close_dn_variant_b(date(2024, 2, 28), expiry, NO_HOLIDAYS)


def test_variant_b_tdom_gte_21_passes():
    # 2024-04-30 Tue: should have tdom >= 21 in April 2024 (22 weekdays in April)
    # April 2024: 1=Mon(1), ..., 30=Tue. Trading days = ~22.
    expiry = frozenset([date(2024, 4, 25)])  # last Thursday April
    holidays = frozenset()
    # 2024-04-30 Tue: not expiry week (expiry Apr 25, week was Apr 22-26)
    # not Monday, tdom = 22 (>= 21) → passes via tdom
    assert trading_day_of_month(date(2024, 4, 30), holidays) >= 21
    assert passes_close_dn_variant_b(date(2024, 4, 30), expiry, holidays)


def test_variant_b_mid_month_non_expiry_friday_fails():
    # Generic mid-month Friday: not Mon, not expiry week, tdom mid-range → False
    expiry = frozenset([date(2024, 5, 30)])  # late Thu May
    holidays = frozenset()
    # 2024-05-10 Fri: tdom 8 (May 1=Wed,2=Thu,3=Fri,6=Mon,...10=Fri → 8)
    # Not Mon, not expiry week (expiry week May 27-30), tdom 8 < 21 → False
    assert not passes_close_dn_variant_b(date(2024, 5, 10), expiry, holidays)


def test_variant_b_thursday_in_expiry_week_still_excluded():
    # Thursday expiry day satisfies is_expiry_week=True
    # but dow==Thursday explicit exclude wins → False
    assert not passes_close_dn_variant_b(date(2024, 1, 25), JAN_2024_EXPIRY, NO_HOLIDAYS)


def test_variant_b_weekend_dates_fail():
    # Saturday/Sunday never trigger fires anyway, but the gate should explicitly return False
    expiry = JAN_2024_EXPIRY
    # 2024-01-27 Sat, 2024-01-28 Sun — both in same week as Jan 25 expiry but Fri/Sat/Sun excluded
    assert not passes_close_dn_variant_b(date(2024, 1, 27), expiry, NO_HOLIDAYS)
    assert not passes_close_dn_variant_b(date(2024, 1, 28), expiry, NO_HOLIDAYS)


# ---------------------------------------------------------------------------
# Smoke test that on-disk loaders work (skipped if files missing)
# ---------------------------------------------------------------------------

def test_load_expiry_dates_smoke():
    """Smoke test: on-disk expiry parquet loads without error and contains expected dates."""
    from services.calendar_utils import load_expiry_dates
    try:
        expiries = load_expiry_dates()
    except FileNotFoundError:
        pytest.skip("data/futures_basis/2023_2026_basis.parquet not present in test env")
    assert len(expiries) >= 30, f"expected >=30 monthly expiries, got {len(expiries)}"
    # Verify 2024-01-25 (known Thursday expiry) is in the set
    assert date(2024, 1, 25) in expiries


def test_load_nse_holidays_smoke():
    """Smoke test: on-disk holidays JSON loads and contains expected dates."""
    from services.calendar_utils import load_nse_holidays
    try:
        holidays = load_nse_holidays()
    except FileNotFoundError:
        pytest.skip("assets/nse_holidays.json not present in test env")
    assert len(holidays) >= 20, f"expected >=20 NSE holidays across 2023-2026, got {len(holidays)}"
    # Verify Republic Day 2023 (well-known Indian holiday) is in the set
    assert date(2023, 1, 26) in holidays


# ---------------------------------------------------------------------------
# passes_long_panic_variant_bp — Variant Bp gate for long_panic_gap_down
# ---------------------------------------------------------------------------

def test_variant_bp_tuesday_passes():
    from services.calendar_utils import passes_long_panic_variant_bp
    # 2024-01-02 was Tuesday
    assert passes_long_panic_variant_bp(date(2024, 1, 2))


def test_variant_bp_wednesday_passes():
    from services.calendar_utils import passes_long_panic_variant_bp
    # 2024-01-03 was Wednesday
    assert passes_long_panic_variant_bp(date(2024, 1, 3))


def test_variant_bp_friday_passes():
    from services.calendar_utils import passes_long_panic_variant_bp
    # 2024-01-05 was Friday
    assert passes_long_panic_variant_bp(date(2024, 1, 5))


def test_variant_bp_monday_excluded():
    from services.calendar_utils import passes_long_panic_variant_bp
    # 2024-01-01 was Monday
    assert not passes_long_panic_variant_bp(date(2024, 1, 1))


def test_variant_bp_thursday_excluded():
    from services.calendar_utils import passes_long_panic_variant_bp
    # 2024-01-04 was Thursday
    assert not passes_long_panic_variant_bp(date(2024, 1, 4))


def test_variant_bp_weekend_returns_false():
    from services.calendar_utils import passes_long_panic_variant_bp
    # 2024-01-06 Saturday, 2024-01-07 Sunday
    assert not passes_long_panic_variant_bp(date(2024, 1, 6))
    assert not passes_long_panic_variant_bp(date(2024, 1, 7))


def test_variant_bp_independent_of_expiry_or_holidays():
    """Variant Bp depends ONLY on dow — not on expiry or holiday calendars.

    Rationale: war-regime-decomposed analysis (2026-05-23) showed the Mon/Thu
    drag is a pure dow effect across all months/expiry cycles, not an expiry
    week interaction.
    """
    from services.calendar_utils import passes_long_panic_variant_bp
    # 2024-01-25 was Thursday AND monthly F&O expiry — excluded by dow rule alone
    assert not passes_long_panic_variant_bp(date(2024, 1, 25))
    # 2024-01-23 was Tuesday during expiry week — passes by dow rule
    assert passes_long_panic_variant_bp(date(2024, 1, 23))
