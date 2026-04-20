"""Stage 1 tests: N ≥ 500 AND PF ≥ 0.8."""
from pathlib import Path
import pandas as pd
import pytest

from tools.edge_discovery.stages.stage1_universe_prune import run_stage1


def _synth_trades(setup_pnls: dict) -> pd.DataFrame:
    """Build a trades DataFrame from {setup_name: [pnl_list]}."""
    rows = []
    for setup, pnls in setup_pnls.items():
        for pnl in pnls:
            rows.append({
                "setup_type": setup,
                "total_trade_pnl": pnl,
                "session_date": "2023-06-01",
            })
    return pd.DataFrame(rows)


def test_stage1_passes_when_N_and_PF_met(tmp_path):
    # setup_a: 600 trades with PF > 0.8; setup_b: 300 trades (fails N)
    df = _synth_trades({
        "setup_a_long": [100] * 350 + [-50] * 250,  # N=600, PF = 35000/12500 = 2.8
        "setup_b_short": [100] * 200 + [-50] * 100,  # N=300, fails
    })
    out = tmp_path / "01-universe-pruning.md"
    result = run_stage1(df, report_path=out, survivors_json=tmp_path / "s1.json")
    survivors = {r["setup"] for r in result if r["passed"]}
    assert "setup_a_long" in survivors
    assert "setup_b_short" not in survivors


def test_stage1_fails_when_PF_below_threshold(tmp_path):
    # setup_a: N=600 but PF < 0.8 (losers dominate)
    df = _synth_trades({
        "setup_a_long": [50] * 200 + [-200] * 400,  # N=600, PF = 10000/80000 = 0.125
    })
    out = tmp_path / "01.md"
    result = run_stage1(df, report_path=out, survivors_json=tmp_path / "s.json")
    assert result[0]["passed"] is False
    assert result[0]["pf"] < 0.8


def test_stage1_writes_report_file(tmp_path):
    df = _synth_trades({
        "a": [100] * 500 + [-50] * 100,
    })
    out = tmp_path / "01.md"
    run_stage1(df, report_path=out, survivors_json=tmp_path / "s.json")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Stage 1" in text
    assert "a" in text


def test_stage1_edge_case_exactly_500(tmp_path):
    """N=500 is the boundary — must pass (≥ 500 is inclusive)."""
    df = _synth_trades({
        "setup_edge": [100] * 300 + [-50] * 200,  # N=500, PF = 30000/10000 = 3.0
    })
    result = run_stage1(df, report_path=tmp_path / "r.md", survivors_json=tmp_path / "s.json")
    assert result[0]["passed"] is True
