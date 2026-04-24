"""search.py tests (sub5-T7) — 10-trial Optuna smoke."""
import json
from pathlib import Path

import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"


def test_search_10_trial_smoke_writes_best_config(tmp_path):
    """10-trial Optuna run on the minimal gate_input + synthetic PnL.
    Verifies best_config.json + trials.csv + study.db are written."""
    from tools.gauntlet_v2.search import run_search

    pnl_rows = [
        {"session_date": "2025-01-02", "ts": "2025-01-02T09:20:00",
         "symbol": "NSE:SYM1", "setup_type": "premium_zone_short",
         "total_trade_pnl": 100.0, "r_multiple": 1.0, "gross_exit_qty": 10},
        {"session_date": "2025-01-02", "ts": "2025-01-02T09:25:00",
         "symbol": "NSE:SYM2", "setup_type": "premium_zone_short",
         "total_trade_pnl": -50.0, "r_multiple": -0.5, "gross_exit_qty": 10},
    ]
    pnl_index = pd.DataFrame(pnl_rows)

    base_cfg = {
        "live_gate_chain": {"enabled": True},
        "rule_filter_gate": {"survivors_path": str(FIXTURES / "minimal_survivors.json")},
        "cross_sectional_gate": {
            "enabled": True, "f1_rvol_enabled": False, "f1_rvol_threshold_pct": 90.0,
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

    out_dir = tmp_path / "study"
    run_search(
        base_cfg=base_cfg,
        gate_input_dir=FIXTURES / "mini_gate_input",
        pnl_index=pnl_index,
        output_dir=out_dir,
        n_trials=10,
        n_jobs=1,           # in-process for determinism in test
        min_n_trades=0,     # allow tiny-N trials in smoke test
    )

    assert (out_dir / "best_config.json").exists()
    assert (out_dir / "trials.csv").exists()
    assert (out_dir / "study.db").exists()
    best = json.loads((out_dir / "best_config.json").read_text())
    # Best config must include at least one searched override
    assert "conviction_gate" in best or "dedup_gate" in best or "cross_sectional_gate" in best
