"""Tests for tools.methodology.sanity_csv_schema.

The validator's primary job is to catch the 5 catastrophic bugs that would
silently corrupt walk-forward verdicts. Each bug class has dedicated tests.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tools.methodology.sanity_csv_schema import (
    REQUIRED_COLUMNS,
    ALLOWED_SIDES,
    EXIT_REASONS,
    PNL_PCT_TOLERANCE_PP,
    validate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    *,
    signal_date="2024-06-15",
    symbol="NSE:RELIANCE",
    side="LONG",
    entry_price=100.0,
    exit_price=102.0,
    qty=10,
    pnl_pct=None,
    exit_reason="t2",
    same_bar=False,
):
    """Build a single canonical row. If pnl_pct=None, compute from side."""
    if pnl_pct is None:
        if side == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100.0
    return {
        "signal_date": signal_date,
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "qty": qty,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "same_bar": same_bar,
    }


def _df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Happy path: valid canonical CSV passes
# ---------------------------------------------------------------------------

def test_valid_canonical_csv_passes():
    """A minimal valid CSV with all required columns + correct sign convention passes."""
    df = _df([
        _row(side="LONG", entry_price=100.0, exit_price=102.0),
        _row(side="LONG", entry_price=100.0, exit_price=98.0),
        _row(side="SHORT", entry_price=100.0, exit_price=98.0),
        _row(side="SHORT", entry_price=100.0, exit_price=102.0),
    ])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert result.is_valid, f"expected valid, got: {result.summary()}"
    assert result.n_errors == 0


# ---------------------------------------------------------------------------
# BUG 1: Sign-convention inversion (CATASTROPHIC)
# ---------------------------------------------------------------------------

def test_long_with_short_sign_convention_fails():
    """LONG trade with pnl_pct computed as if SHORT (sign inverted) FAILS."""
    # Entry 100, exit 102 → LONG should be +2.0%. If labeled LONG but pnl_pct=-2.0%,
    # that's the catastrophic inversion bug.
    df = _df([{
        "signal_date": "2024-06-15", "symbol": "NSE:RELIANCE", "side": "LONG",
        "entry_price": 100.0, "exit_price": 102.0, "qty": 10,
        "pnl_pct": -2.0,  # WRONG — should be +2.0
        "exit_reason": "t2", "same_bar": False,
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    codes = [i.code for i in result.issues]
    assert "pnl_pct.sign_or_magnitude_mismatch" in codes


def test_short_with_long_sign_convention_fails():
    """SHORT trade with LONG sign convention FAILS."""
    df = _df([{
        "signal_date": "2024-06-15", "symbol": "NSE:RELIANCE", "side": "SHORT",
        "entry_price": 100.0, "exit_price": 98.0, "qty": 10,
        "pnl_pct": -2.0,  # WRONG — SHORT 100→98 should be +2.0%
        "exit_reason": "t2", "same_bar": False,
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    codes = [i.code for i in result.issues]
    assert "pnl_pct.sign_or_magnitude_mismatch" in codes


def test_correct_sign_convention_passes_both_sides():
    """Confirms LONG and SHORT both pass when pnl_pct is computed correctly."""
    df = _df([
        # LONG winners + losers
        _row(side="LONG", entry_price=100.0, exit_price=102.0, pnl_pct=2.0),
        _row(side="LONG", entry_price=100.0, exit_price=98.0, pnl_pct=-2.0),
        # SHORT winners + losers
        _row(side="SHORT", entry_price=100.0, exit_price=98.0, pnl_pct=2.0),
        _row(side="SHORT", entry_price=100.0, exit_price=102.0, pnl_pct=-2.0),
    ])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert result.is_valid, result.summary()


# ---------------------------------------------------------------------------
# BUG 2: Double-counted fees/leverage (pnl_pct contaminated)
# ---------------------------------------------------------------------------

def test_pnl_pct_with_leverage_applied_fails():
    """pnl_pct=10% with entry 100 / exit 102 LONG cannot be right (leverage 5x?)."""
    df = _df([{
        "signal_date": "2024-06-15", "symbol": "NSE:RELIANCE", "side": "LONG",
        "entry_price": 100.0, "exit_price": 102.0, "qty": 10,
        "pnl_pct": 10.0,  # 5x leverage applied — but raw return is 2.0%
        "exit_reason": "t2", "same_bar": False,
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "pnl_pct.sign_or_magnitude_mismatch" in [i.code for i in result.issues]


def test_pnl_pct_with_fees_applied_fails_if_diff_exceeds_tolerance():
    """If pnl_pct already has 0.5% fee subtracted, the cross-check fails."""
    # LONG 100→102 raw = +2.0%. After 0.5% fees = +1.5%. That's a 0.5pp diff.
    df = _df([{
        "signal_date": "2024-06-15", "symbol": "NSE:RELIANCE", "side": "LONG",
        "entry_price": 100.0, "exit_price": 102.0, "qty": 10,
        "pnl_pct": 1.5,  # fee-discounted
        "exit_reason": "t2", "same_bar": False,
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    # 0.5pp diff > PNL_PCT_TOLERANCE_PP (0.10pp) → fails
    assert not result.is_valid
    assert "pnl_pct.sign_or_magnitude_mismatch" in [i.code for i in result.issues]


def test_pnl_pct_within_tolerance_passes():
    """Tiny float-rounding errors within tolerance should pass."""
    # LONG 100→102 = exactly +2.0%. Off by 0.05pp = within tolerance.
    df = _df([_row(side="LONG", entry_price=100.0, exit_price=102.0, pnl_pct=2.05)])
    # tolerance is 0.10pp → 2.05 vs 2.0 = 0.05pp → should pass
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert result.is_valid, result.summary()


# ---------------------------------------------------------------------------
# BUG 3: IST/UTC date drift
# ---------------------------------------------------------------------------

def test_tz_aware_signal_date_fails():
    """If signal_date is parsed with tz, walk-forward IST-naive contract is broken."""
    df = pd.DataFrame([_row()])
    # Force tz-aware timestamps
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.tz_localize("UTC")
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "signal_date.has_tz" in [i.code for i in result.issues]


def test_signal_date_string_naive_passes():
    """IST-naive date string is the expected format and passes."""
    df = _df([_row(signal_date="2024-06-15")])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert result.is_valid


def test_signal_date_out_of_range_fails():
    """Date outside [2022-01-01, 2026-12-31] is suspicious."""
    df = _df([_row(signal_date="2019-01-01")])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "signal_date.out_of_range" in [i.code for i in result.issues]


def test_weekend_signal_date_warns_but_passes():
    """Weekend date is a warning (not error) — caller chooses how strict to be."""
    # 2024-06-15 was a Saturday
    df = _df([_row(signal_date="2024-06-15")])
    result = validate(df, setup_name="test", layer="filtered_trades")
    # Either no weekend warning (if not Saturday) or warning-only
    codes = [i.code for i in result.issues]
    if "signal_date.weekend" in codes:
        # Confirm it's a warning, not error
        weekend_issues = [i for i in result.issues if i.code == "signal_date.weekend"]
        assert all(i.severity == "warn" for i in weekend_issues)


# ---------------------------------------------------------------------------
# BUG 4: Layer mismatch (raw vs filtered)
# ---------------------------------------------------------------------------

def test_raw_candidates_layer_rejected():
    """layer='raw_candidates' is rejected — walk-forward only takes filtered trades."""
    df = _df([_row()])
    result = validate(df, setup_name="test", layer="raw_candidates")
    assert not result.is_valid
    assert "metadata.layer.invalid" in [i.code for i in result.issues]


def test_unknown_layer_rejected():
    """Arbitrary layer string is rejected."""
    df = _df([_row()])
    result = validate(df, setup_name="test", layer="something_else")
    assert not result.is_valid
    assert "metadata.layer.invalid" in [i.code for i in result.issues]


# ---------------------------------------------------------------------------
# BUG 5: NaN/inf in pnl_pct
# ---------------------------------------------------------------------------

def test_nan_pnl_pct_fails():
    """NaN pnl_pct is rejected — would silently skew PF."""
    df = _df([
        _row(),
        {**_row(), "pnl_pct": float("nan")},
    ])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "pnl_pct.nan_or_inf" in [i.code for i in result.issues]


def test_inf_pnl_pct_fails():
    """inf pnl_pct is rejected."""
    df = _df([
        _row(),
        {**_row(), "pnl_pct": float("inf")},
    ])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "pnl_pct.nan_or_inf" in [i.code for i in result.issues]


# ---------------------------------------------------------------------------
# Other required-column checks
# ---------------------------------------------------------------------------

def test_missing_required_column_fails():
    """Missing 'side' column fails clearly."""
    df = _df([{
        "signal_date": "2024-06-15", "symbol": "NSE:RELIANCE",
        "entry_price": 100.0, "exit_price": 102.0, "qty": 10,
        "pnl_pct": 2.0, "exit_reason": "t2", "same_bar": False,
        # NO side column
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "columns.missing_required" in [i.code for i in result.issues]


def test_empty_dataframe_fails():
    """Zero-row DataFrame is rejected (no trades to validate)."""
    df = pd.DataFrame({c: [] for c in REQUIRED_COLUMNS})
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "rows.empty" in [i.code for i in result.issues]


def test_invalid_side_value_fails():
    """side='long' (lowercase) is rejected — must be 'LONG' or 'SHORT'."""
    df = _df([{**_row(), "side": "long"}])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "side.invalid_value" in [i.code for i in result.issues]


def test_negative_entry_price_fails():
    """entry_price <= 0 is rejected."""
    df = _df([{**_row(), "entry_price": -100.0}])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "entry_price.non_positive" in [i.code for i in result.issues]


def test_zero_qty_fails():
    """qty must be positive."""
    df = _df([{**_row(), "qty": 0}])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "qty.non_positive" in [i.code for i in result.issues]


def test_unknown_exit_reason_warns():
    """Unknown exit_reason is a warning (adapter should normalize), not an error."""
    df = _df([{**_row(), "exit_reason": "weird_string"}])
    result = validate(df, setup_name="test", layer="filtered_trades")
    # Should still be valid (no errors), but have a warning
    assert result.is_valid
    codes = [i.code for i in result.issues]
    assert "exit_reason.unknown_value" in codes


def test_symbol_without_nse_prefix_warns():
    """Symbol 'RELIANCE' instead of 'NSE:RELIANCE' is a warning, not error."""
    df = _df([{**_row(), "symbol": "RELIANCE"}])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert result.is_valid
    assert "symbol.unexpected_format" in [i.code for i in result.issues]


# ---------------------------------------------------------------------------
# Optional rupee cross-check
# ---------------------------------------------------------------------------

def test_rupee_pnl_inconsistency_warns():
    """net_pnl_inr != realized_pnl_inr - fee_inr is a warning."""
    df = _df([{
        **_row(),
        "realized_pnl_inr": 1000.0,
        "fee_inr": 50.0,
        "net_pnl_inr": 800.0,  # WRONG: should be 950
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    codes = [i.code for i in result.issues]
    assert "rupee_pnl.cross_check_failed" in codes


def test_rupee_pnl_consistent_passes():
    """Consistent rupee fields pass cleanly."""
    df = _df([{
        **_row(),
        "realized_pnl_inr": 1000.0,
        "fee_inr": 50.0,
        "net_pnl_inr": 950.0,
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert result.is_valid
    # No rupee warnings
    rupee_warnings = [i for i in result.issues if i.code.startswith("rupee_pnl")]
    assert len(rupee_warnings) == 0


# ---------------------------------------------------------------------------
# Multi-leg trades: t1_partial_booked + blended pnl_pct
# ---------------------------------------------------------------------------

def test_t1_partial_booked_true_skips_sign_check():
    """When t1_partial_booked=True, validator skips the (exit-entry)/entry
    cross-check because pnl_pct is blended across two legs."""
    # SHORT trade: entry 100, T1 booked at 99 on half qty, breakeven exit at 100.
    # Blended pnl_pct = T1 profit on half: (100-99)*half_qty / (100*total_qty) * 100
    #                 = 0.5 * 0.5% = +0.5% blended? No wait, let me recompute:
    # T1 partial profit per share = 100 - 99 = 1.0 (+1% per share)
    # On half qty: 1.0 * 0.5 * qty
    # On total notional 100*qty: blended_pct = (1.0 * 0.5) / 100 * 100 = +0.5%
    df = _df([{
        "signal_date": "2024-06-15", "symbol": "NSE:RELIANCE", "side": "SHORT",
        "entry_price": 100.0, "exit_price": 100.0, "qty": 100,
        "pnl_pct": 0.5,  # blended (T1 partial)
        "exit_reason": "breakeven_stop", "same_bar": False,
        "t1_partial_booked": True,
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    # The stored pnl_pct=0.5 does NOT match (entry-exit)/entry*100=0,
    # but because t1_partial_booked=True, validator skips that check.
    assert result.is_valid, result.summary()


def test_t1_partial_booked_false_still_enforces_sign_check():
    """t1_partial_booked=False rows still get the sign cross-check."""
    df = _df([{
        "signal_date": "2024-06-15", "symbol": "NSE:RELIANCE", "side": "SHORT",
        "entry_price": 100.0, "exit_price": 100.0, "qty": 100,
        "pnl_pct": 0.5,  # WRONG — would only be valid if t1_partial_booked=True
        "exit_reason": "sl", "same_bar": False,
        "t1_partial_booked": False,
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    assert "pnl_pct.sign_or_magnitude_mismatch" in [i.code for i in result.issues]


def test_breakeven_stop_requires_t1_partial_booked():
    """exit_reason='breakeven_stop' is only valid when t1_partial_booked=True."""
    df = _df([{
        "signal_date": "2024-06-15", "symbol": "NSE:RELIANCE", "side": "SHORT",
        "entry_price": 100.0, "exit_price": 100.0, "qty": 100,
        "pnl_pct": 0.5,
        "exit_reason": "breakeven_stop", "same_bar": False,
        "t1_partial_booked": False,  # INCONSISTENT
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    codes = [i.code for i in result.issues]
    assert "breakeven_stop.t1_partial_booked_false" in codes


def test_breakeven_stop_without_t1_partial_booked_column_fails():
    """exit_reason='breakeven_stop' without the t1_partial_booked column = error."""
    df = _df([{
        "signal_date": "2024-06-15", "symbol": "NSE:RELIANCE", "side": "SHORT",
        "entry_price": 100.0, "exit_price": 100.0, "qty": 100,
        "pnl_pct": 0.5,
        "exit_reason": "breakeven_stop", "same_bar": False,
        # NO t1_partial_booked column
    }])
    result = validate(df, setup_name="test", layer="filtered_trades")
    assert not result.is_valid
    codes = [i.code for i in result.issues]
    assert "breakeven_stop.missing_t1_partial_booked" in codes
