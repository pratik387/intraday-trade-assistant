"""Stage 1 tests: N ≥ 500 AND PF ≥ 0.8."""
from pathlib import Path
import pandas as pd
import pytest

from tools.edge_discovery_legacy_gauntlet.stages.stage1_universe_prune import run_stage1


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


def test_stage1_all_winners_inf_pf(tmp_path):
    """All-winners setup produces PF=inf internally; must serialize as 999.0 in JSON and mark passed=True."""
    import json
    df = _synth_trades({
        "all_winners": [100] * 600,  # N=600, zero losers → PF=inf
    })
    out_md = tmp_path / "01.md"
    out_json = tmp_path / "s.json"
    result = run_stage1(df, report_path=out_md, survivors_json=out_json)
    assert result[0]["passed"] is True
    assert result[0]["pf"] == 999.0  # inf clamped to 999 for JSON
    # JSON file must be valid and contain the expected schema
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["stage"] == "1"
    assert "all_winners" in loaded["survivors"]
    assert loaded["details"][0]["pf"] == 999.0


def test_stage1_pf_exactly_at_threshold(tmp_path):
    """PF exactly 0.8 must pass (>= is inclusive, symmetric with N=500 boundary)."""
    # 250 winners at +80 Rs, 500 losers at -50 Rs → winners=20000, losers=25000, PF=0.8
    df = _synth_trades({
        "pf_edge": [80] * 250 + [-50] * 500,  # N=750, PF=20000/25000=0.8
    })
    result = run_stage1(df, report_path=tmp_path / "r.md", survivors_json=tmp_path / "s.json")
    assert result[0]["passed"] is True
    assert result[0]["pf"] == 0.8


def test_stage1_sort_order_pass_before_fail_then_n_desc(tmp_path):
    """Passed setups come before failed; within each group, sorted by N descending."""
    df = _synth_trades({
        "small_pass": [100] * 400 + [-50] * 200,  # N=600, PF=4.0 PASS
        "big_pass":   [100] * 600 + [-50] * 200,  # N=800, PF=6.0 PASS
        "big_fail":   [10]  * 900 + [-100] * 200, # N=1100, PF=0.45 FAIL
        "small_fail": [10]  * 400 + [-100] * 100, # N=500, PF=0.4 FAIL
    })
    result = run_stage1(df, report_path=tmp_path / "r.md", survivors_json=tmp_path / "s.json")
    setups_in_order = [r["setup"] for r in result]
    # Pass group first, sorted by N desc within group
    assert setups_in_order == ["big_pass", "small_pass", "big_fail", "small_fail"]
