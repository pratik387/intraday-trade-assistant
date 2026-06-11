"""Tests for calc_fee_cnc, calc_fee_mtf, calc_fee_by_mode."""
import pytest
from tools.sub7_validation.build_per_setup_pnl import (
    calc_fee, calc_fee_cnc, calc_fee_mtf, calc_fee_by_mode,
)


def test_calc_fee_cnc_baseline_matches_sanity():
    """Rs 1L buy / Rs 1L*1.005 sell -> ~Rs 222.96.

    Post-Jun-2026 fee audit:
      - Brokerage: Rs 0 (was Rs 20 flat each side; Zerodha delivery free)
      - STT: 0.1% on BOTH sides (was sell-only)
      - Txn: 0.00307%/side (was 0.00345%)
    Net per-side breakdown: stt(200.50) + stamp(15.00) + txn(6.155) + sebi(0.20) + gst(1.108)
    """
    fee = calc_fee_cnc(100000.0, 100500.0)
    # Allow Rs 1 tolerance for rounding
    assert 222 <= fee <= 224, f"expected ~222.96, got {fee:.4f}"


def test_calc_fee_cnc_zero_for_zero_values():
    assert calc_fee_cnc(0.0, 0.0) == 0.0 or calc_fee_cnc(0.0, 0.0) < 0.01


def test_calc_fee_mtf_one_night_includes_interest():
    """Rs 2.79L buy / Rs 2.80L sell / Rs 1L margin / 1 day -> fee > CNC equivalent due to interest + pledge."""
    cnc_equivalent = calc_fee_cnc(279000.0, 280000.0)
    mtf = calc_fee_mtf(279000.0, 280000.0, 100000.0, 1)
    # MTF fee = CNC base + interest + pledge + unpledge
    # interest = (279000 - 100000) * 0.0004 * 1 = 71.6
    # pledge + unpledge = 15*1.18 + 15*1.18 = 35.4
    expected_extra = 71.6 + 35.4
    assert abs(mtf - cnc_equivalent - expected_extra) < 0.5, (
        f"mtf={mtf:.2f}, cnc={cnc_equivalent:.2f}, "
        f"extra={mtf - cnc_equivalent:.2f}, expected_extra~={expected_extra}"
    )


def test_calc_fee_mtf_friday_to_monday_hold_days_3():
    """Hold days scales interest linearly."""
    mtf_1day = calc_fee_mtf(279000.0, 280000.0, 100000.0, 1)
    mtf_3day = calc_fee_mtf(279000.0, 280000.0, 100000.0, 3)
    # Difference is 2 additional days of interest: 179000 * 0.0004 * 2 = 143.2
    expected_diff = 179000.0 * 0.0004 * 2
    actual_diff = mtf_3day - mtf_1day
    assert abs(actual_diff - expected_diff) < 0.5, (
        f"3-day MTF minus 1-day MTF = {actual_diff:.2f}, expected ~={expected_diff:.2f}"
    )


def test_calc_fee_mtf_rejects_invalid_margin():
    """margin_inr <= 0 returns 0 (mirrors calc_fee defensive check)."""
    assert calc_fee_mtf(279000.0, 280000.0, 0.0, 1) == 0.0
    assert calc_fee_mtf(279000.0, 280000.0, -100.0, 1) == 0.0


def test_calc_fee_by_mode_dispatches_cnc():
    by_mode = calc_fee_by_mode(100000.0, 100500.0, mode="delivery_cnc")
    direct = calc_fee_cnc(100000.0, 100500.0)
    assert by_mode == direct


def test_calc_fee_by_mode_dispatches_mtf():
    by_mode = calc_fee_by_mode(279000.0, 280000.0, mode="mtf",
                                margin_inr=100000.0, hold_days=1)
    direct = calc_fee_mtf(279000.0, 280000.0, 100000.0, 1)
    assert by_mode == direct


def test_calc_fee_by_mode_mtf_without_margin_raises():
    with pytest.raises(ValueError, match="margin_inr"):
        calc_fee_by_mode(279000.0, 280000.0, mode="mtf")


def test_calc_fee_by_mode_unsupported_mode_raises():
    with pytest.raises(ValueError, match="unsupported mode"):
        calc_fee_by_mode(100000.0, 100500.0, mode="bracket_order")


def test_existing_intraday_calc_fee_unaffected():
    """Regression: don't break existing intraday MIS fee math."""
    # qty=10, entry=500, exit=501.5 (+0.3% intraday move)
    fee_old = calc_fee(500.0, 501.5, 10, side="BUY", mis_leverage=1.0)
    assert fee_old > 0
    # Same trade computed via mode dispatcher (delivery_cnc, would be DIFFERENT --
    # confirms the dispatcher is correctly routing and the modes have DIFFERENT fee schedules)
    buy_value = 500 * 10  # = 5000
    sell_value = 501.5 * 10  # = 5015
    fee_cnc = calc_fee_cnc(buy_value, sell_value)
    # CNC should be HIGHER than intraday because delivery STT (0.1% on BOTH sides
    # = 0.2% round-trip notional) dominates vs intraday STT (0.025% sell-only).
    # Delivery brokerage is Rs 0 since Zerodha made it free.
    assert fee_cnc > fee_old
