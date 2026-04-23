"""Parity simulator tests (sub-project #4)."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


def test_cli_help_runs():
    """CLI must respond to --help without ImportError."""
    result = subprocess.run(
        [sys.executable, "tools/shadow/parity_simulator.py", "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "--gate-input" in result.stdout
    assert "--config" in result.stdout
    assert "--output" in result.stdout


def _build_minimal_config(survivors_path: Path) -> dict:
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
    }


def test_single_session_replay_admits_known_setups():
    """Given 3 candidates (2 premium_zone_short + 1 vwap_lose_short) with only PZ in survivors,
    sim must admit both PZ and reject vwap_lose_short at rule_filter."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = _build_minimal_config(FIXTURES / "minimal_survivors.json")
        cfg_path = td / "cfg.json"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        out_csv = td / "sim_admits.csv"
        result = subprocess.run(
            [sys.executable, "tools/shadow/parity_simulator.py",
             "--gate-input", str(FIXTURES / "minimal_gate_input.jsonl"),
             "--config", str(cfg_path),
             "--output", str(out_csv)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        rows = list(csv.DictReader(out_csv.open()))
        admits = [r for r in rows if r["stage"] == "admitted"]
        assert len(admits) == 2, f"expected 2 admits, got {len(admits)}: {admits}"
        assert {r["symbol"] for r in admits} == {"NSE:SYM1", "NSE:SYM2"}
        rejects = [r for r in rows if r["stage"] != "admitted"]
        assert any(r["symbol"] == "NSE:SYM3" and r["stage"] == "rule_filter" for r in rejects)


def test_on_bar_close_receives_datetime_not_string():
    """Regression test for bar_ts type bug: chain.on_bar_close requires datetime,
    not string. Without _parse_dt conversion, RVOL warmup silently fails because
    the except-Exception masks the AttributeError on str.hour."""
    from unittest.mock import MagicMock
    from datetime import datetime
    from tools.shadow.parity_simulator import _replay_one_session

    chain = MagicMock()
    chain.evaluate.return_value = []
    rows = [{
        "ts": "2025-01-02T09:20:00",
        "session_date": "2025-01-02",
        "candidates": [],
        "bar_volumes": {"NSE:SYM1": 1000},
        "symbol_caps": {"NSE:SYM1": "small_cap"},
    }]
    _replay_one_session(rows, chain)
    chain.on_bar_close.assert_called_once()
    call_kwargs = chain.on_bar_close.call_args.kwargs
    assert isinstance(call_kwargs["bar_ts"], datetime), \
        f"on_bar_close received {type(call_kwargs['bar_ts']).__name__} for bar_ts; expected datetime"
