"""Tests for tools.methodology.walk_forward."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tools.methodology.walk_forward import (
    build_windows,
    run_walk_forward,
    classify_tier,
    Window,
    WalkForwardResult,
    Tier,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_trades.csv"


def test_build_windows_13_non_overlapping_3month_windows():
    """13 windows × 3 months covers Jan 2023 → Mar 2026."""
    windows = build_windows(
        start=date(2023, 1, 1),
        end=date(2026, 3, 31),
        window_months=3,
        n_windows=13,
    )

    assert len(windows) == 13
    assert windows[0].start == date(2023, 1, 1)
    assert windows[0].end == date(2023, 3, 31)
    assert windows[-1].start == date(2026, 1, 1)
    assert windows[-1].end == date(2026, 3, 31)
    for i in range(1, len(windows)):
        assert windows[i].start > windows[i-1].end


def test_classify_tier_green_at_9_of_13():
    assert classify_tier(pass_rate=9/13, n_windows_total=13) == Tier.GREEN

def test_classify_tier_amber_at_6_of_13():
    assert classify_tier(pass_rate=6/13, n_windows_total=13) == Tier.AMBER

def test_classify_tier_amber_at_8_of_13():
    assert classify_tier(pass_rate=8/13, n_windows_total=13) == Tier.AMBER

def test_classify_tier_red_at_5_of_13():
    assert classify_tier(pass_rate=5/13, n_windows_total=13) == Tier.RED

def test_classify_tier_red_at_0():
    assert classify_tier(pass_rate=0.0, n_windows_total=13) == Tier.RED

def test_classify_tier_green_at_13_of_13():
    assert classify_tier(pass_rate=1.0, n_windows_total=13) == Tier.GREEN


def test_run_walk_forward_handles_string_signal_date_column():
    """Object-dtype signal_date column (e.g., strings from CSV with no date parsing)
    must be converted, not silently broken."""
    trades = pd.DataFrame({
        "signal_date": ["2023-02-15", "2023-02-20", "2023-03-10",
                        "2024-06-15", "2024-06-20", "2024-07-10",
                        "2025-06-15", "2025-06-20", "2025-07-10",
                        "2025-12-15", "2025-12-20", "2026-01-10"] * 5,
        "symbol": ["SYM01"] * 60,
        "pnl_pct": [0.5, -0.3, 0.4, 0.6, -0.2, 0.5, -0.4, -0.5, -0.6, -0.3, -0.4, -0.5] * 5,
    })
    # Note: signal_date is object dtype (strings) — NOT datetime

    result = run_walk_forward(
        setup_name="dtype_test",
        trades_df=trades,
        start=date(2023, 1, 1),
        end=date(2026, 3, 31),
        window_months=3, n_windows=13,
        bootstrap_n=200,
    )

    # Should not raise. Verify it ran end-to-end.
    assert result.windows_total == 13
    # At least one window should have trades (not all n=0)
    assert any(w.n > 0 for w in result.windows)


def test_run_walk_forward_raises_on_missing_columns():
    """Missing required column → ValueError with clear message."""
    trades = pd.DataFrame({"signal_date": ["2023-01-15"], "wrong_pnl": [0.5]})

    with pytest.raises(ValueError, match="missing required columns"):
        run_walk_forward(
            setup_name="missing_cols",
            trades_df=trades,
            start=date(2023, 1, 1),
            end=date(2026, 3, 31),
        )


def test_compute_per_trade_net_pnl_default_fee_matches_real_indian_intraday():
    """Calibration regression test (added 2026-05-20).

    Default fee_pct_round_trip=0.25 was derived by tracing real Indian
    retail intraday trades (pre_results_t1_v2 Discovery, 100-trade sample
    on 2026-05-20). Implied fee_pct on capital = 0.2484 +/- 0.0002 across
    a range of trade sizes (Rs 50K-500K notional).

    The PRIOR default 0.5% was wrong by 2x. It systematically punished every
    trade by 0.25pp net, which flipped multiple production-GREEN setups to
    RED on walk-forward (e.g. delivery_pct_anomaly_short, long_panic_gap_down,
    or_window_failure_fade_short showed RED when production claimed GREEN).
    Don't revert this without a stronger calibration argument.
    """
    from tools.methodology.walk_forward import _compute_per_trade_net_pnl

    # Trade reality: SHORT entry 462.75, exit 464.13825 → pnl_pct = -0.3%
    # Notional Rs 333K, capital Rs 66.6K (5x MIS leverage)
    # Actual fee = Rs 165.48 = 0.0497% of notional = 0.2485% of capital
    # Actual net PnL = -1.748% of capital
    net = _compute_per_trade_net_pnl(pnl_pct=-0.3, fee_pct=0.25, mis_leverage=5.0)
    # Formula gives: -0.3 × 5 - 0.25 = -1.75%
    assert abs(net - (-1.75)) < 1e-9
    # Within 0.05pp of measured reality (-1.748%)
    assert abs(net - (-1.748)) < 0.05


def test_run_walk_forward_on_fixture_detects_regime_break():
    """Fixture has positive edge pre-2025, negative post-2025.
    Walk-forward should classify as AMBER (mixed) or RED."""
    trades = pd.read_csv(FIXTURE)
    trades["signal_date"] = pd.to_datetime(trades["signal_date"]).dt.date

    result = run_walk_forward(
        setup_name="fixture_test",
        trades_df=trades,
        start=date(2023, 1, 1),
        end=date(2026, 3, 31),
        window_months=3,
        n_windows=13,
        fee_pct_round_trip=0.5,
        mis_leverage=5.0,
        bootstrap_n=500,
    )

    assert isinstance(result, WalkForwardResult)
    assert result.windows_total == 13
    assert result.tier in (Tier.AMBER, Tier.RED)
    assert result.windows[0].passes_gate is True
    assert result.windows[-1].passes_gate is False
