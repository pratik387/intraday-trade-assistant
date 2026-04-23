"""Parity check tests (sub-project #4)."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent


def _write_sim_csv(path: Path, rows):
    """rows = list of dicts with at least ts, symbol, setup_type, stage."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session_date", "ts", "symbol", "setup_type",
                                            "predicted_r", "gate_reject_reason", "stage"])
        w.writeheader()
        for r in rows:
            w.writerow({**{"session_date": "", "predicted_r": "", "gate_reject_reason": ""}, **r})


def _write_live_jsonl(path: Path, admits):
    """admits = list of dicts with timestamp, symbol, strategy_type."""
    with open(path, "w", encoding="utf-8") as f:
        for a in admits:
            f.write(json.dumps({**{"action": "admit"}, **a}) + "\n")


def test_parity_mode_match_exits_zero():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sim = td / "sim.csv"
        live = td / "live.jsonl"
        _write_sim_csv(sim, [
            {"ts": "2025-01-02T09:20:00", "symbol": "NSE:A", "setup_type": "premium_zone_short", "stage": "admitted"},
            {"ts": "2025-01-02T09:25:00", "symbol": "NSE:B", "setup_type": "premium_zone_short", "stage": "admitted"},
        ])
        _write_live_jsonl(live, [
            {"timestamp": "2025-01-02T09:20:00", "symbol": "NSE:A", "strategy_type": "premium_zone_short"},
            {"timestamp": "2025-01-02T09:25:00", "symbol": "NSE:B", "strategy_type": "premium_zone_short"},
        ])
        result = subprocess.run(
            [sys.executable, "tools/shadow/parity_check.py",
             "--live", str(live), "--sim", str(sim)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "MATCH 2/2" in result.stdout


def test_parity_mode_divergence_exits_one():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sim = td / "sim.csv"
        live = td / "live.jsonl"
        _write_sim_csv(sim, [
            {"ts": "2025-01-02T09:20:00", "symbol": "NSE:A", "setup_type": "premium_zone_short", "stage": "admitted"},
            {"ts": "2025-01-02T09:25:00", "symbol": "NSE:C", "setup_type": "premium_zone_short", "stage": "admitted"},
        ])
        _write_live_jsonl(live, [
            {"timestamp": "2025-01-02T09:20:00", "symbol": "NSE:A", "strategy_type": "premium_zone_short"},
            {"timestamp": "2025-01-02T09:25:00", "symbol": "NSE:B", "strategy_type": "premium_zone_short"},
        ])
        result = subprocess.run(
            [sys.executable, "tools/shadow/parity_check.py",
             "--live", str(live), "--sim", str(sim)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 1
        assert "DIVERGENCE" in result.stdout
        assert "NSE:B" in result.stdout  # in live, missing from sim
        assert "NSE:C" in result.stdout  # in sim, missing from live


def test_ab_mode_reports_deltas():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        baseline = td / "baseline.csv"
        variant = td / "variant.csv"
        _write_sim_csv(baseline, [
            {"ts": "2025-01-02T09:20:00", "symbol": "NSE:A", "setup_type": "premium_zone_short", "stage": "admitted"},
            {"ts": "2025-01-02T09:25:00", "symbol": "NSE:B", "setup_type": "premium_zone_short", "stage": "admitted"},
        ])
        _write_sim_csv(variant, [
            {"ts": "2025-01-02T09:20:00", "symbol": "NSE:A", "setup_type": "premium_zone_short", "stage": "admitted"},
            {"ts": "2025-01-02T10:30:00", "symbol": "NSE:C", "setup_type": "range_bounce_short", "stage": "admitted"},
            {"ts": "2025-01-02T13:05:00", "symbol": "NSE:D", "setup_type": "range_bounce_short", "stage": "admitted"},
        ])
        result = subprocess.run(
            [sys.executable, "tools/shadow/parity_check.py",
             "--baseline", str(baseline), "--variant", str(variant)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "admit count: 2 -> 3" in result.stdout
        assert "premium_zone_short" in result.stdout
        assert "range_bounce_short" in result.stdout
