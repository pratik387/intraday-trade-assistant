"""Stage 5b tests: ruleset simulation as union filter + aggregate stats."""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tools.edge_discovery.stages.stage5b_ruleset_simulation import (
    apply_filter,
    aggregate_stats,
    run_stage5b,
)


def _trades(rows):
    return pd.DataFrame(rows)


def test_apply_filter_matches_setup_and_all_conditions():
    trades = _trades([
        {"setup_type": "a", "regime": "chop", "cap_segment": "small_cap",
         "hour_bucket": "morning", "total_trade_pnl": 100, "session_date_dt": date(2023, 1, 1)},
        {"setup_type": "a", "regime": "chop", "cap_segment": "large_cap",
         "hour_bucket": "morning", "total_trade_pnl": 200, "session_date_dt": date(2023, 1, 1)},
        {"setup_type": "b", "regime": "chop", "cap_segment": "small_cap",
         "hour_bucket": "morning", "total_trade_pnl": 300, "session_date_dt": date(2023, 1, 1)},
    ])
    rules = [{"setup": "a", "conditions": [("regime", "chop"), ("cap_segment", "small_cap")]}]
    result = apply_filter(trades, rules)
    # Only the first row matches setup=a AND all conditions
    assert len(result) == 1
    assert result.iloc[0]["total_trade_pnl"] == 100


def test_apply_filter_union_semantics():
    """A trade matches if ANY rule matches — OR across rules."""
    trades = _trades([
        {"setup_type": "a", "regime": "chop", "cap_segment": "small_cap",
         "hour_bucket": "morning", "total_trade_pnl": 100, "session_date_dt": date(2023, 1, 1)},
        {"setup_type": "a", "regime": "trend_up", "cap_segment": "mid_cap",
         "hour_bucket": "morning", "total_trade_pnl": 200, "session_date_dt": date(2023, 1, 1)},
    ])
    rules = [
        {"setup": "a", "conditions": [("regime", "chop")]},
        {"setup": "a", "conditions": [("cap_segment", "mid_cap")]},
    ]
    # Both rows match exactly one rule
    result = apply_filter(trades, rules)
    assert len(result) == 2


def test_apply_filter_counts_each_trade_once_even_if_multiple_rules_match():
    trades = _trades([
        {"setup_type": "a", "regime": "chop", "cap_segment": "small_cap",
         "hour_bucket": "morning", "total_trade_pnl": 100, "session_date_dt": date(2023, 1, 1)},
    ])
    rules = [
        {"setup": "a", "conditions": [("regime", "chop")]},
        {"setup": "a", "conditions": [("cap_segment", "small_cap")]},
        {"setup": "a", "conditions": [("regime", "chop"), ("cap_segment", "small_cap")]},
    ]
    result = apply_filter(trades, rules)
    assert len(result) == 1  # not 3


def test_aggregate_stats_basic():
    df = _trades([
        {"session_date_dt": date(2023, 1, 1), "total_trade_pnl": 100},
        {"session_date_dt": date(2023, 1, 1), "total_trade_pnl": -50},
        {"session_date_dt": date(2023, 1, 2), "total_trade_pnl": 200},
        {"session_date_dt": date(2023, 1, 2), "total_trade_pnl": -100},
    ])
    s = aggregate_stats(df, "test")
    assert s["n_trades"] == 4
    assert s["n_sessions"] == 2
    assert s["trades_per_day"] == 2.0
    assert s["pf"] == pytest.approx(300 / 150)
    assert s["wr_pct"] == pytest.approx(50.0)
    assert s["total_pnl"] == 150


def test_run_stage5b_produces_report_and_json(tmp_path):
    trades = pd.DataFrame([
        {"setup_type": "a", "regime": "chop", "cap_segment": "small_cap",
         "hour_bucket": "morning", "total_trade_pnl": 100,
         "session_date_dt": date(2023, 1, 1) + timedelta(days=i)}
        for i in range(20)
    ])
    rules = [{"setup": "a", "conditions": [("regime", "chop")]}]
    result = run_stage5b(
        trades=trades,
        approved_rules=rules,
        report_path=tmp_path / "06.md",
        summary_json=tmp_path / "s5b.json",
    )
    assert (tmp_path / "06.md").exists()
    assert (tmp_path / "s5b.json").exists()
    assert "scenarios" in result
    assert "per_hour" in result
    assert "per_setup" in result
    # Baseline scenario + filtered + 3 hour-subsets = 5 scenarios
    assert len(result["scenarios"]) == 5
