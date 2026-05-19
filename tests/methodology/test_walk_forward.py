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
