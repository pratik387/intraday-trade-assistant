"""Tests for data_loader — loading, joining, filtering."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tools.edge_discovery.data_loader import load_run, GauntletData
from tests.edge_discovery.fixtures.make_fixtures import make_mini_run


@pytest.fixture
def mini_run(tmp_path):
    root = tmp_path / "mini_run"
    dates = ["2023-01-02", "2023-01-15", "2023-02-01", "2023-02-15"]
    make_mini_run(root, dates, trades_per_session=20)
    return root


def test_loads_sessions(mini_run):
    data = load_run(mini_run)
    assert isinstance(data, GauntletData)
    assert len(data.trades) > 0
    assert set(data.trades["session_date"].unique()) == {
        "2023-01-02", "2023-01-15", "2023-02-01", "2023-02-15",
    }


def test_is_final_exit_filter(mini_run):
    """Loader MUST drop is_final_exit=False rows. This is the core correctness
    property — without it, partial T1 exits inflate WR."""
    data = load_run(mini_run)
    # Fixture writes 20 trades/session + 7 partial exits/session (every 3rd).
    # Only 20 should remain per session.
    assert len(data.trades) == 4 * 20  # 4 sessions × 20 final exits


def test_trades_have_required_columns(mini_run):
    data = load_run(mini_run)
    required = {
        "trade_id", "session_date", "setup_type", "total_trade_pnl",
        "regime", "cap_segment", "hour_bucket", "fy",
    }
    assert required.issubset(set(data.trades.columns))


def test_hour_bucket_values(mini_run):
    """Hour bucket comes from trade_report.csv minute_of_day joined into analytics."""
    data = load_run(mini_run)
    buckets = set(data.trades["hour_bucket"].dropna().unique())
    # At least one valid bucket must appear
    valid = {"opening", "morning", "lunch", "afternoon", "late"}
    assert buckets.issubset(valid)


def test_fy_assignment(mini_run):
    data = load_run(mini_run)
    # All fixture dates are in Jan-Feb 2023 → FY2022-23
    assert set(data.trades["fy"].unique()) == {"FY2022-23"}


def test_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No session"):
        load_run(empty)
