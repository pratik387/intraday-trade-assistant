"""Stage 2 tests: PF ≥ 1.2 full, PF ≥ 1.0 both halves, Sharpe ≥ 0.7, DD < 30%."""
from datetime import date
from pathlib import Path
import pandas as pd
import pytest

from tools.edge_discovery.periods import DiscoveryConfig
from tools.edge_discovery.stages.stage2_univariate import run_stage2


def _cfg():
    return DiscoveryConfig(
        discovery_start=date(2023, 1, 1),
        discovery_end=date(2024, 12, 31),
        validation_start=date(2025, 1, 1),
        validation_end=date(2025, 9, 30),
        holdout_start=date(2025, 10, 1),
        holdout_end=date(2026, 3, 31),
    )


def _trades(setup: str, h1_pnls: list, h2_pnls: list) -> pd.DataFrame:
    """Spreads trades across multiple distinct session dates so session_sharpe
    is computable (needs >= 2 distinct dates)."""
    from datetime import timedelta
    rows = []
    # H1 trades span Jan-Jun 2023, one trade per date cycling through 10 dates
    h1_dates = [date(2023, 1, 1) + timedelta(days=30 * i) for i in range(10)]
    for i, p in enumerate(h1_pnls):
        rows.append({"setup_type": setup, "total_trade_pnl": p,
                     "session_date_dt": h1_dates[i % len(h1_dates)]})
    # H2 trades span Jul-Dec 2024, one trade per date cycling through 10 dates
    h2_dates = [date(2024, 3, 1) + timedelta(days=30 * i) for i in range(10)]
    for i, p in enumerate(h2_pnls):
        rows.append({"setup_type": setup, "total_trade_pnl": p,
                     "session_date_dt": h2_dates[i % len(h2_dates)]})
    return pd.DataFrame(rows)


def test_stage2_passes_when_all_criteria_met(tmp_path):
    """Both sub-periods positive, PF full > 1.2, session_sharpe > 0.7, DD < 30%.

    With trades spread across 20 distinct dates (10 per half) and the
    [100, 100, -50] cycle giving consistent positive daily PnL, session Sharpe
    is very high (low daily variance).
    """
    df = _trades("winner", [100, 100, -50] * 100, [100, 100, -50] * 100)
    result = run_stage2(
        df, cfg=_cfg(),
        survivors_input=["winner"],
        report_path=tmp_path / "02.md",
        survivors_json=tmp_path / "s2.json",
    )
    assert result[0]["passed"] is True
    assert result[0]["session_sharpe"] >= 0.7


def test_stage2_fails_when_h1_positive_h2_negative(tmp_path):
    """Regime change — one half good, other half bad."""
    df = _trades("fluky", [200, -50] * 100, [-100, 50] * 100)
    result = run_stage2(
        df, cfg=_cfg(),
        survivors_input=["fluky"],
        report_path=tmp_path / "02.md",
        survivors_json=tmp_path / "s2.json",
    )
    assert result[0]["passed"] is False


def test_stage2_fails_when_full_pf_below_1_2(tmp_path):
    """PF full Discovery < 1.2 → fail even if both halves positive."""
    # PF = 0.5: 60 winners * 100 vs 60 losers * 200 = 6000/12000 = 0.5
    df = _trades("weak", [100] * 30 + [-200] * 30, [100] * 30 + [-200] * 30)
    result = run_stage2(
        df, cfg=_cfg(),
        survivors_input=["weak"],
        report_path=tmp_path / "02.md",
        survivors_json=tmp_path / "s2.json",
    )
    assert result[0]["passed"] is False
    assert result[0]["pf_full"] < 1.2


def test_stage2_only_evaluates_input_survivors(tmp_path):
    """Stage 2 must NOT evaluate setups that weren't passed from Stage 1."""
    df = pd.concat([
        _trades("keeper", [200, -100] * 100, [200, -100] * 100),
        _trades("killed_in_stage1", [100] * 10, [100] * 10),
    ])
    result = run_stage2(
        df, cfg=_cfg(),
        survivors_input=["keeper"],
        report_path=tmp_path / "02.md",
        survivors_json=tmp_path / "s2.json",
    )
    assert {r["setup"] for r in result} == {"keeper"}


def test_stage2_output_contains_h1_n_and_h2_n(tmp_path):
    """h1_n and h2_n exposed in result so empty-subperiod is visible in audit trail."""
    df = _trades("w", [100, 100, -50] * 50, [100, 100, -50] * 60)
    # h1 has 150 trades, h2 has 180
    result = run_stage2(
        df, cfg=_cfg(),
        survivors_input=["w"],
        report_path=tmp_path / "02.md",
        survivors_json=tmp_path / "s2.json",
    )
    row = result[0]
    assert "h1_n" in row
    assert "h2_n" in row
    assert row["h1_n"] == 150
    assert row["h2_n"] == 180
    assert row["n"] == 330


def test_stage2_empty_subperiod_is_treated_as_fail(tmp_path):
    """Empty sub-period → PF=0.0 → fails sub-period gate. h2_n=0 makes it transparent."""
    df = _trades("h1_only", [100, 100, -50] * 100, [])
    result = run_stage2(
        df, cfg=_cfg(),
        survivors_input=["h1_only"],
        report_path=tmp_path / "02.md",
        survivors_json=tmp_path / "s2.json",
    )
    row = result[0]
    assert row["h2_n"] == 0
    assert row["pf_h2"] == 0.0
    assert row["passed"] is False


def test_stage2_gates_on_session_sharpe_not_per_trade_sharpe(tmp_path):
    """Regression guard: Stage 2 must use session_sharpe (daily-aggregated),
    not per-trade sharpe. A setup with ~300 trades/day at +₹25 avg and high
    per-trade variance has per-trade Sharpe ~0.24 (would FAIL old 0.7 gate)
    but session Sharpe is infinite because daily PnL is constant (stable edge
    emerges at the session level)."""
    from datetime import timedelta
    # Per-trade: alternating [+100, -50] → mean=25, std≈106, sharpe≈0.24
    # PF = 100/50 = 2.0
    # Each session: 150 × 100 + 150 × -50 = +₹7,500 (constant daily PnL)
    # Session sharpe → inf (zero daily std)
    rows = []
    for d_offset in range(20):
        sd = date(2023, 3, 1) + timedelta(days=d_offset * 7)
        for p in [100, -50] * 150:  # 300 trades/session
            rows.append({"setup_type": "high_volume", "total_trade_pnl": p,
                         "session_date_dt": sd})
    for d_offset in range(20):
        sd = date(2024, 3, 1) + timedelta(days=d_offset * 7)
        for p in [100, -50] * 150:
            rows.append({"setup_type": "high_volume", "total_trade_pnl": p,
                         "session_date_dt": sd})
    df = pd.DataFrame(rows)
    result = run_stage2(
        df, cfg=_cfg(),
        survivors_input=["high_volume"],
        report_path=tmp_path / "02.md",
        survivors_json=tmp_path / "s2.json",
    )
    row = result[0]
    # Per-trade sharpe well below 0.7 — proves we would have FAILED with the old metric
    assert row["sharpe_per_trade"] < 0.5
    # Session sharpe passes — proves the new metric credits real edge
    assert row["session_sharpe"] >= 0.7
    assert row["passed"] is True
