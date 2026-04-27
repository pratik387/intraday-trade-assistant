"""universe_filter tests (sub8-T1, rev2 — adds expiry/circuit helpers)."""
from datetime import date

import pandas as pd
import pytest

from services.universe_filter import (
    in_nifty50,
    in_banknifty,
    in_fno_liquid_200,
    in_universe,
    is_expiry_day,
    near_circuit_band,
)


def test_nifty50_member_returns_true():
    # RELIANCE is always in Nifty 50
    assert in_nifty50("NSE:RELIANCE") is True


def test_nifty50_non_member_returns_false():
    # SWSOLAR is small-cap — never in Nifty 50
    assert in_nifty50("NSE:SWSOLAR") is False


def test_banknifty_member_returns_true():
    # HDFCBANK is always in Bank Nifty
    assert in_banknifty("NSE:HDFCBANK") is True


def test_banknifty_non_member_returns_false():
    assert in_banknifty("NSE:RELIANCE") is False  # in Nifty50, not BankNifty


def test_fno_liquid_200_includes_index_majors():
    assert in_fno_liquid_200("NSE:RELIANCE") is True
    assert in_fno_liquid_200("NSE:HDFCBANK") is True


def test_in_universe_dispatches_by_key():
    assert in_universe("NSE:RELIANCE", "nifty50") is True
    assert in_universe("NSE:RELIANCE", "banknifty") is False
    assert in_universe("NSE:RELIANCE", "nifty50_banknifty") is True
    assert in_universe("NSE:HDFCBANK", "nifty50_banknifty") is True
    assert in_universe("NSE:RELIANCE", "fno_liquid_200") is True


def test_in_universe_unknown_key_raises():
    with pytest.raises(KeyError):
        in_universe("NSE:RELIANCE", "nonexistent_universe")


# ---------------------------------------------------------------------------
# Cross-cutting exclusion helpers (rev2)
# ---------------------------------------------------------------------------


def test_is_expiry_day_thursday_pre_2025_09():
    """Pre-2025-09-01 fallback: Thursday is Nifty weekly expiry."""
    # 2024-06-06 is a Thursday
    assert is_expiry_day(date(2024, 6, 6)) is True


def test_is_expiry_day_tuesday_post_2025_09():
    """Post-2025-09-01 fallback: Tuesday is Nifty weekly expiry."""
    # 2025-10-07 is a Tuesday
    assert is_expiry_day(date(2025, 10, 7)) is True


def test_is_expiry_day_non_expiry_returns_false():
    # 2024-06-05 is a Wednesday (Nifty was Thursday pre-Sep-2025)
    assert is_expiry_day(date(2024, 6, 5)) is False


def test_is_expiry_day_accepts_pandas_timestamp():
    ts = pd.Timestamp("2024-06-06")  # Thursday
    assert is_expiry_day(ts) is True


def test_is_expiry_day_accepts_iso_string():
    assert is_expiry_day("2024-06-06") is True  # Thursday


def test_is_expiry_day_handles_none():
    assert is_expiry_day(None) is False


def test_near_circuit_band_at_upper_circuit_returns_true():
    # PDC=100, 10% upper circuit = 110, proximity 2% = 2 → flagged at >=108
    assert near_circuit_band(current_price=109.0, prev_close=100.0,
                             circuit_pct=10.0, proximity_pct=2.0) is True


def test_near_circuit_band_at_lower_circuit_returns_true():
    # PDC=100, 10% lower circuit = 90, proximity 2% = 2 → flagged at <=92
    assert near_circuit_band(current_price=91.0, prev_close=100.0,
                             circuit_pct=10.0, proximity_pct=2.0) is True


def test_near_circuit_band_safe_zone_returns_false():
    # PDC=100, current=100 (mid) — far from both circuits
    assert near_circuit_band(current_price=100.0, prev_close=100.0,
                             circuit_pct=10.0, proximity_pct=2.0) is False


def test_near_circuit_band_invalid_prices_excludes_safely():
    # Zero/negative prices treated as "exclude" (be safe)
    assert near_circuit_band(current_price=0, prev_close=100.0) is True
    assert near_circuit_band(current_price=100.0, prev_close=0) is True


def test_near_circuit_band_5pct_band_for_smaller_stocks():
    # SME / smaller stocks may have 5% circuit. Custom circuit_pct supported.
    # PDC=100, 5% upper = 105, proximity 2 → flag at >=103
    assert near_circuit_band(current_price=104.0, prev_close=100.0,
                             circuit_pct=5.0, proximity_pct=2.0) is True
    assert near_circuit_band(current_price=102.0, prev_close=100.0,
                             circuit_pct=5.0, proximity_pct=2.0) is False
