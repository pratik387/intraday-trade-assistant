"""validate.py tests (sub5-T8) — OOS one-shot discipline."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


def _write_minimal_config_files(tmp_path):
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
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_cfg))

    frozen_overrides = {"conviction_gate": {"daily_cap": 10}}
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen_overrides))

    pnl_rows = [
        {"session_date": "2025-01-02", "ts": "2025-01-02T09:20:00",
         "symbol": "NSE:SYM1", "setup_type": "premium_zone_short",
         "total_trade_pnl": 100.0, "r_multiple": 1.0, "gross_exit_qty": 10},
        {"session_date": "2025-01-02", "ts": "2025-01-02T09:25:00",
         "symbol": "NSE:SYM2", "setup_type": "premium_zone_short",
         "total_trade_pnl": -50.0, "r_multiple": -0.5, "gross_exit_qty": 10},
    ]
    pnl_path = tmp_path / "pnl.parquet"
    pd.DataFrame(pnl_rows).to_parquet(pnl_path, index=False)
    return base_path, frozen_path, pnl_path


def test_validate_one_shot_refuses_second_run_without_force(tmp_path):
    base, frozen, pnl = _write_minimal_config_files(tmp_path)
    out = tmp_path / "out"

    r1 = subprocess.run(
        [sys.executable, "tools/gauntlet_v2/validate.py",
         "--base-cfg", str(base), "--config", str(frozen),
         "--gate-input-dir", str(FIXTURES / "mini_gate_input"),
         "--pnl-index", str(pnl),
         "--period", "validation",
         "--output-dir", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    # First run completes (returncode 0 or 2 depending on pass/fail). Either way, the result file exists.
    assert r1.returncode in (0, 2), f"unexpected returncode {r1.returncode}; stderr: {r1.stderr}"
    assert (out / "07-validation-result.json").exists()

    r2 = subprocess.run(
        [sys.executable, "tools/gauntlet_v2/validate.py",
         "--base-cfg", str(base), "--config", str(frozen),
         "--gate-input-dir", str(FIXTURES / "mini_gate_input"),
         "--pnl-index", str(pnl),
         "--period", "validation",
         "--output-dir", str(out)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert r2.returncode == 1, f"second run must refuse with exit 1; got {r2.returncode}"
    assert "already" in r2.stderr.lower() or "--force" in r2.stderr.lower()


def test_validate_force_overwrites_for_holdout(tmp_path):
    base, frozen, pnl = _write_minimal_config_files(tmp_path)
    out = tmp_path / "out"

    for _ in range(2):
        r = subprocess.run(
            [sys.executable, "tools/gauntlet_v2/validate.py",
             "--base-cfg", str(base), "--config", str(frozen),
             "--gate-input-dir", str(FIXTURES / "mini_gate_input"),
             "--pnl-index", str(pnl),
             "--period", "holdout",
             "--output-dir", str(out), "--force"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        # --force on both runs; second must succeed (returncode 0 or 2 — pass or fail, but NOT 1=refused)
        assert r.returncode in (0, 2), f"--force run unexpectedly refused: {r.stderr}"
    assert (out / "09-holdout-result.json").exists()
