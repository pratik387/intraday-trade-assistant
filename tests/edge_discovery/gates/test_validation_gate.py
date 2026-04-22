"""Validation gate tests — per-rule OOS check with 3 gates (PF, N, WR delta)."""
from pathlib import Path

import pandas as pd
import pytest

from tools.edge_discovery.gates.validation_gate import (
    validate_rule,
    run_validation_gate,
)


def _trades(setup, pnls, cap="small_cap", regime="chop", hour_bucket="morning"):
    """Minimal trade rows."""
    import datetime
    return pd.DataFrame([
        {
            "setup_type": setup,
            "regime": regime,
            "cap_segment": cap,
            "hour_bucket": hour_bucket,
            "minute_of_day": 650,
            "session_date_dt": datetime.date(2025, 1, (i % 28) + 1),
            "total_trade_pnl": p,
        }
        for i, p in enumerate(pnls)
    ])


def test_rule_passes_when_all_criteria_met():
    # 100 trades, PF 2.0 (100 winners @ 100, 100 losers @ -50) → WR 50%
    pnls = [100] * 100 + [-50] * 100
    trades = _trades("setup_a", pnls)
    rule = {"setup": "setup_a", "conditions": [("regime", "chop")]}
    discovery_stats = {"wr_pct": 50.0, "pf": 2.0, "n": 1000}
    result = validate_rule(trades, rule, discovery_stats)
    assert result["passed"] is True
    assert result["n_val"] == 200
    assert result["pf_val"] == pytest.approx(2.0)
    assert result["wr_val"] == pytest.approx(50.0)


def test_rule_fails_when_pf_below_1():
    # PF 0.5: 60 winners @ 100, 60 losers @ 200
    pnls = [100] * 60 + [-200] * 60
    trades = _trades("setup_a", pnls)
    rule = {"setup": "setup_a", "conditions": [("regime", "chop")]}
    discovery_stats = {"wr_pct": 50.0, "pf": 1.5, "n": 1000}
    result = validate_rule(trades, rule, discovery_stats)
    assert result["passed"] is False
    assert result["pf_val"] < 1.0
    assert "pf" in " ".join(result["fail_reasons"]).lower()


def test_rule_fails_when_n_below_50():
    # Only 30 trades
    pnls = [100] * 20 + [-50] * 10
    trades = _trades("setup_a", pnls)
    rule = {"setup": "setup_a", "conditions": [("regime", "chop")]}
    discovery_stats = {"wr_pct": 50.0, "pf": 2.0, "n": 1000}
    result = validate_rule(trades, rule, discovery_stats)
    assert result["passed"] is False
    assert result["n_val"] == 30
    assert "n" in " ".join(result["fail_reasons"]).lower()


def test_rule_fails_when_wr_delta_exceeds_10pp():
    # WR 70% in validation but Discovery was 50% → 20pp delta
    pnls = [100] * 140 + [-50] * 60
    trades = _trades("setup_a", pnls)
    rule = {"setup": "setup_a", "conditions": [("regime", "chop")]}
    discovery_stats = {"wr_pct": 50.0, "pf": 2.0, "n": 1000}
    result = validate_rule(trades, rule, discovery_stats)
    assert result["passed"] is False
    assert abs(result["wr_val"] - 50.0) > 10.0
    assert "wr" in " ".join(result["fail_reasons"]).lower()


def test_rule_conditions_are_applied():
    """Only trades matching ALL rule conditions are counted."""
    pnls_chop = [100] * 50 + [-50] * 50  # 100 trades in chop, PF 2.0
    trades_chop = _trades("setup_a", pnls_chop, regime="chop")
    trades_trend = _trades("setup_a", [100] * 5, regime="trend_up")
    all_trades = pd.concat([trades_chop, trades_trend])
    rule = {"setup": "setup_a", "conditions": [("regime", "chop")]}
    discovery_stats = {"wr_pct": 50.0, "pf": 2.0, "n": 1000}
    result = validate_rule(all_trades, rule, discovery_stats)
    # Only the 100 chop trades are counted, not the 5 trend_up ones
    assert result["n_val"] == 100


def test_empty_trade_subset_handled():
    """Rule with zero matching trades fails (N < 50)."""
    trades = _trades("setup_b", [100] * 5)  # different setup
    rule = {"setup": "setup_a", "conditions": [("regime", "chop")]}
    discovery_stats = {"wr_pct": 50.0, "pf": 2.0, "n": 1000}
    result = validate_rule(trades, rule, discovery_stats)
    assert result["passed"] is False
    assert result["n_val"] == 0


def test_run_validation_gate_writes_report_and_json(tmp_path):
    """End-to-end: runs a ruleset, writes markdown + JSON survivors."""
    pnls_good = [100] * 60 + [-50] * 40  # passing rule
    pnls_bad = [100] * 30 + [-150] * 30   # failing rule (PF 0.67)
    trades = pd.concat([
        _trades("setup_good", pnls_good),
        _trades("setup_bad", pnls_bad),
    ])
    rules = [
        {"setup": "setup_good", "conditions": [("regime", "chop")]},
        {"setup": "setup_bad", "conditions": [("regime", "chop")]},
    ]
    discovery_stats_by_rule = {
        "setup_good__regime=chop": {"wr_pct": 60.0, "pf": 2.5, "n": 500},
        "setup_bad__regime=chop": {"wr_pct": 50.0, "pf": 2.0, "n": 500},
    }
    result = run_validation_gate(
        trades=trades,
        approved_rules=rules,
        discovery_stats_by_rule=discovery_stats_by_rule,
        report_path=tmp_path / "06.md",
        survivors_json=tmp_path / "validation_survivors.json",
    )
    assert (tmp_path / "06.md").exists()
    assert (tmp_path / "validation_survivors.json").exists()
    # result contains per-rule stats
    assert len(result["per_rule"]) == 2
    # one passed, one failed
    passed = [r for r in result["per_rule"] if r["passed"]]
    failed = [r for r in result["per_rule"] if not r["passed"]]
    assert len(passed) == 1
    assert len(failed) == 1
