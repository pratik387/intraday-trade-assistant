"""trial.py tests (sub5-T6) — single-config evaluator."""
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


def _build_minimal_full_cfg(survivors_path: Path) -> dict:
    """Config shape accepted by LiveGateChain.__init__."""
    return {
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(survivors_path)},
        "cross_sectional_gate": {
            "enabled": True,
            "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
            "f1_applicable_caps": [], "f1_skip_hour_buckets": [],
            "f1_min_history_sessions": 5, "f1_rolling_window_sessions": 20,
            "f2_crowdedness_enabled": False, "f2_crowdedness_threshold": 100,
            "f2_crowdedness_window_min": 5,
        },
        "conviction_gate": {
            "enabled": True,
            "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
            "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
            "daily_cap": 50, "min_predicted_r": -100.0,
        },
        "dedup_gate": {"enabled": False, "cooloff_bars": 6, "require_setup_change": True},
        "rank_pctl_min": 0.80,
    }


def test_trial_returns_metrics_for_matched_admits(tmp_path):
    """Given a 3-bar gate_input + pnl_index with 2 matching admits, trial returns
    metrics reflecting those 2 matched PnLs."""
    from tools.gauntlet_v2.trial import run_trial

    pnl_rows = [
        {"session_date": "2025-01-02", "ts": "2025-01-02T09:20:00",
         "symbol": "NSE:SYM1", "setup_type": "premium_zone_short",
         "total_trade_pnl": 100.0, "r_multiple": 1.0, "gross_exit_qty": 10},
        {"session_date": "2025-01-02", "ts": "2025-01-02T09:25:00",
         "symbol": "NSE:SYM2", "setup_type": "premium_zone_short",
         "total_trade_pnl": -50.0, "r_multiple": -0.5, "gross_exit_qty": 10},
    ]
    pnl_index = pd.DataFrame(pnl_rows)
    base_cfg = _build_minimal_full_cfg(FIXTURES / "minimal_survivors.json")

    metrics = run_trial(
        cfg_overrides={},
        gate_input_dir=FIXTURES / "mini_gate_input",
        pnl_index=pnl_index,
        base_cfg=base_cfg,
    )

    assert metrics["n_trades"] == 2
    assert metrics["n_sessions"] == 1
    assert metrics["total_pnl"] == 50.0
    assert metrics["wr"] == pytest.approx(0.5)
    assert metrics["trades_per_day"] == pytest.approx(2.0)


def test_trial_cfg_override_deep_merges():
    """cfg_overrides must merge into base_cfg without wiping siblings."""
    from tools.gauntlet_v2.trial import _merge_cfg

    base = {"conviction_gate": {"daily_cap": 50, "min_predicted_r": 0.3}}
    overrides = {"conviction_gate": {"daily_cap": 10}}
    merged = _merge_cfg(base, overrides)

    assert merged["conviction_gate"]["daily_cap"] == 10
    assert merged["conviction_gate"]["min_predicted_r"] == 0.3
