"""Stage 3 tests: 1-way and 2-way structural conditioner analysis.

Pass criteria per cell:
  - N >= 100
  - PF >= 1.3 on cell
  - PF >= 1.1 in both sub-periods
"""
from datetime import date
from pathlib import Path
import pandas as pd
import pytest

from tools.edge_discovery.periods import DiscoveryConfig
from tools.edge_discovery.stages.stage3_conditional import run_stage3


def _cfg():
    return DiscoveryConfig(
        discovery_start=date(2023, 1, 1),
        discovery_end=date(2024, 12, 31),
        validation_start=date(2025, 1, 1),
        validation_end=date(2025, 9, 30),
        holdout_start=date(2025, 10, 1),
        holdout_end=date(2026, 3, 31),
    )


def _make_df(rows):
    """rows: list of dicts with regime/cap_segment/hour_bucket/pnl/date."""
    df = pd.DataFrame(rows)
    df["setup_type"] = "test_setup"
    df["total_trade_pnl"] = df["pnl"]
    df["session_date_dt"] = pd.to_datetime(df["date"]).dt.date
    return df


def test_stage3_finds_regime_specific_edge(tmp_path):
    """Setup positive in trend_down, negative in chop — trend_down cell should pass."""
    rows = []
    for _ in range(150):
        rows.append({"regime": "trend_down", "cap_segment": "mid_cap",
                     "hour_bucket": "morning", "pnl": 100, "date": "2023-06-01"})
        rows.append({"regime": "trend_down", "cap_segment": "mid_cap",
                     "hour_bucket": "morning", "pnl": -50, "date": "2024-06-01"})
    for _ in range(100):
        rows.append({"regime": "chop", "cap_segment": "mid_cap",
                     "hour_bucket": "morning", "pnl": -50, "date": "2023-06-01"})
        rows.append({"regime": "chop", "cap_segment": "mid_cap",
                     "hour_bucket": "morning", "pnl": -100, "date": "2024-06-01"})
    df = _make_df(rows)

    result = run_stage3(
        df, cfg=_cfg(),
        survivors_input=["test_setup"],
        report_path=tmp_path / "03.md",
        survivors_json=tmp_path / "s3.json",
    )
    test_cells = [c for c in result if c["setup"] == "test_setup"]
    passing = [c for c in test_cells if c["passed"]]
    assert any(c.get("conditioner") == "regime" and c.get("cell_value") == "trend_down"
               for c in passing)


def test_stage3_skips_cells_below_min_n(tmp_path):
    """Cells with N < 100 are not evaluated."""
    rows = []
    for _ in range(50):
        rows.append({"regime": "trend_down", "cap_segment": "mid_cap",
                     "hour_bucket": "morning", "pnl": 100, "date": "2023-06-01"})
    df = _make_df(rows)
    result = run_stage3(
        df, cfg=_cfg(),
        survivors_input=["test_setup"],
        report_path=tmp_path / "03.md",
        survivors_json=tmp_path / "s3.json",
    )
    assert not any(c["passed"] for c in result)


def test_stage3_runs_2way_only_for_1way_passing(tmp_path):
    """2-way combos only evaluated where 1-way passed."""
    rows = []
    for _ in range(150):
        rows.append({"regime": "trend_down", "cap_segment": "mid_cap",
                     "hour_bucket": "morning", "pnl": 200, "date": "2023-06-01"})
        rows.append({"regime": "trend_down", "cap_segment": "mid_cap",
                     "hour_bucket": "morning", "pnl": -50, "date": "2024-06-01"})
    df = _make_df(rows)
    result = run_stage3(
        df, cfg=_cfg(),
        survivors_input=["test_setup"],
        report_path=tmp_path / "03.md",
        survivors_json=tmp_path / "s3.json",
    )
    two_way = [c for c in result if c.get("dim_count", 0) == 2]
    assert len(two_way) > 0
