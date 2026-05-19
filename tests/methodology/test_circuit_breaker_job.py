"""Tests for jobs.check_circuit_breakers."""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from jobs.check_circuit_breakers import check_setup_circuit_breaker, CBState


def _trade_report(tmp_path: Path, rows: list) -> Path:
    """Write a minimal trade_report.csv fixture."""
    p = tmp_path / "trade_report.csv"
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return p


def test_circuit_breaker_does_not_trip_when_pnl_above_threshold(tmp_path):
    """Setup with trailing 60d PnL > threshold stays enabled."""
    trades = _trade_report(tmp_path, [
        {"signal_date": (date.today() - timedelta(days=10)).isoformat(),
         "setup_type": "my_setup", "actual_pnl_after_charges": 5000.0},
        {"signal_date": (date.today() - timedelta(days=20)).isoformat(),
         "setup_type": "my_setup", "actual_pnl_after_charges": 8000.0},
    ] * 20)

    state = check_setup_circuit_breaker(
        setup_name="my_setup",
        trades_csv=trades,
        lookback_days=60,
        threshold=-50000.0,
        min_trades=30,
        today=date.today(),
    )

    assert state.action == "no_change"
    assert state.trailing_pnl > 0


def test_circuit_breaker_trips_when_pnl_below_threshold(tmp_path):
    """Setup with trailing 60d PnL < threshold trips -> disabled."""
    trades = _trade_report(tmp_path, [
        {"signal_date": (date.today() - timedelta(days=10)).isoformat(),
         "setup_type": "my_setup", "actual_pnl_after_charges": -3000.0},
    ] * 40)

    state = check_setup_circuit_breaker(
        setup_name="my_setup",
        trades_csv=trades,
        lookback_days=60,
        threshold=-50000.0,
        min_trades=30,
        today=date.today(),
    )

    assert state.action == "disable"
    assert state.trailing_pnl < -50000.0
    assert state.reason == "60d_pnl_below_threshold"


def test_circuit_breaker_does_not_trip_when_insufficient_trades(tmp_path):
    """Insufficient n in window -> no action even if loss large."""
    trades = _trade_report(tmp_path, [
        {"signal_date": (date.today() - timedelta(days=10)).isoformat(),
         "setup_type": "my_setup", "actual_pnl_after_charges": -10000.0},
    ] * 5)

    state = check_setup_circuit_breaker(
        setup_name="my_setup",
        trades_csv=trades,
        lookback_days=60,
        threshold=-50000.0,
        min_trades=30,
        today=date.today(),
    )

    assert state.action == "no_change"
    assert state.reason == "insufficient_trades"
