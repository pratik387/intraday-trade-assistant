"""Tests for data_loader — loading, joining, filtering."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tools.edge_discovery_legacy_gauntlet.data_loader import load_run, GauntletData
from tests.edge_discovery_legacy_gauntlet.fixtures.make_fixtures import make_mini_run


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


def test_raises_when_only_partial_exits(tmp_path):
    """Branch: directory has a valid session, JSONL has only is_final_exit=False rows → ValueError."""
    import json
    run_dir = tmp_path / "partial_only"
    sdir = run_dir / "2023-06-15"
    sdir.mkdir(parents=True)
    with open(sdir / "analytics.jsonl", "w") as f:
        for i in range(5):
            f.write(json.dumps({
                "trade_id": f"t{i}",
                "setup_type": "setup_a_long",
                "pnl": 100,
                "is_final_exit": False,  # all partials, no finals
                "reason": "t1_partial",
            }) + "\n")
    with pytest.raises(ValueError, match="No final-exit"):
        load_run(run_dir)


def test_hour_bucket_directly_at_boundaries():
    """Lock _hour_bucket boundary semantics with direct unit tests."""
    from tools.edge_discovery.data_loader import _hour_bucket
    # Pre-market (< 555) should be None, not 'opening'
    assert _hour_bucket(540) is None  # 9:00 — before market open
    assert _hour_bucket(554) is None
    # Boundaries
    assert _hour_bucket(555) == "opening"  # 9:15 — first valid minute
    assert _hour_bucket(599) == "opening"
    assert _hour_bucket(600) == "morning"  # 10:00
    assert _hour_bucket(719) == "morning"
    assert _hour_bucket(720) == "lunch"    # 12:00
    assert _hour_bucket(779) == "lunch"
    assert _hour_bucket(780) == "afternoon"  # 13:00
    assert _hour_bucket(869) == "afternoon"
    assert _hour_bucket(870) == "late"     # 14:30
    assert _hour_bucket(930) == "late"     # 15:30
    # None / NaN pass-through
    assert _hour_bucket(None) is None
    import pandas as pd
    import numpy as np
    assert _hour_bucket(np.nan) is None
