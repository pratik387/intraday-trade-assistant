"""Tests for tools.methodology.cell_sweep.

Each test pins one of the bug-prone behaviors that recurred in per-setup
sweep scripts before this helper existed (Lesson #5 failure modes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.methodology.cell_sweep import (
    ALLOWED_SIDES,
    CellResult,
    CellSweepConfig,
    FORBIDDEN_DIM_PREFIXES,
    SchemaIssue,
    lock_cell,
    run_cell_sweep,
    select_best_cell,
    simulate_exit,
    validate_candidates_schema,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _good_df(n=10):
    return pd.DataFrame({
        "entry_ts": pd.date_range("2024-01-02 10:00", periods=n, freq="5min"),
        "entry_price": [100.0] * n,
        "qty": [50] * n,
        "mfe_r": [0.8] * n,
        "mae_r": [0.5] * n,
        "R_per_share": [1.0] * n,
        "close_at_1500": [99.5] * n,
        "cap_segment": (["small_cap", "mid_cap"] * n)[:n],
    })


def test_schema_validator_passes_well_formed_df():
    df = _good_df(20)
    r_grid = [(1.0, 2.0, 1500)]
    result = validate_candidates_schema(df, dim_pool=["cap_segment"], r_grid=r_grid)
    assert result.is_valid


def test_schema_validator_rejects_lookahead_dim():
    """REGRESSION: Lesson #5 failure mode #1 — day_high/day_low/day_close at signal."""
    df = _good_df()
    df["day_high"] = 105.0
    r_grid = [(1.0, 2.0, 1500)]
    result = validate_candidates_schema(df, dim_pool=["day_high"], r_grid=r_grid)
    assert not result.is_valid
    assert any(i.code == "lookahead_dim" for i in result.issues)


def test_schema_validator_blocks_close_off_high_bucket():
    """Concrete case from _circuit_release_fade — column was removed 2026-05-16."""
    df = _good_df()
    df["close_off_high_bucket"] = "0-0.3"
    r_grid = [(1.0, 2.0, 1500)]
    result = validate_candidates_schema(df, dim_pool=["close_off_high_bucket"], r_grid=r_grid)
    assert not result.is_valid
    assert any(i.code == "lookahead_dim" for i in result.issues)


def test_schema_validator_catches_missing_close_column():
    df = _good_df()
    # r_grid mentions ts=1300 but df only has close_at_1500
    r_grid = [(1.0, 2.0, 1500), (0.5, 1.0, 1300)]
    result = validate_candidates_schema(df, dim_pool=["cap_segment"], r_grid=r_grid)
    assert not result.is_valid
    assert any(i.code == "close_col_missing" for i in result.issues)


def test_schema_validator_catches_missing_required_columns():
    df = _good_df().drop(columns=["mfe_r"])
    result = validate_candidates_schema(
        df, dim_pool=["cap_segment"], r_grid=[(1.0, 2.0, 1500)],
    )
    assert not result.is_valid
    assert any(i.code == "missing_required" for i in result.issues)


def test_schema_validator_catches_sign_convention_violation():
    df = _good_df()
    df.loc[df.index[0], "mfe_r"] = -0.5  # NEGATIVE mfe is wrong
    result = validate_candidates_schema(
        df, dim_pool=["cap_segment"], r_grid=[(1.0, 2.0, 1500)],
    )
    assert not result.is_valid
    assert any(i.code == "negative_mfe" for i in result.issues)


# ---------------------------------------------------------------------------
# simulate_exit semantics
# ---------------------------------------------------------------------------

def test_simulate_exit_short_full_t2_winner():
    """SHORT with mfe=2.5R, mae=0.5R: T1+T2 both hit, no stop."""
    out = simulate_exit(
        side="SHORT", entry_price=100.0, qty=50, R_per_share=1.0,
        mfe_r=2.5, mae_r=0.5, close_at_ts=98.0,
        T1_R=1.0, T2_R=2.0,
    )
    assert out is not None
    assert out.exit_reason == "t2_full"
    # SHORT favorable means exit_price < entry_price
    assert out.exit_price < 100.0
    assert out.net_pnl_inr > 0


def test_simulate_exit_short_same_bar_stop_pessimism():
    """REGRESSION (Lesson #5 failure mode #4): when both stop and T2 hit on
    the same bar, sanity must pick STOP (pessimistic). This was the
    mis_unwind 88% same-bar-lookahead bug."""
    out = simulate_exit(
        side="SHORT", entry_price=100.0, qty=50, R_per_share=1.0,
        mfe_r=3.0,  # T2 hit
        mae_r=1.0,  # ALSO stop hit on same bar
        close_at_ts=98.0,
        T1_R=1.0, T2_R=2.0,
    )
    assert out is not None
    assert out.exit_reason == "sl", "stop must win when both stop and T2 hit"
    # SHORT stop = entry + R (price went UP against)
    assert out.exit_price == pytest.approx(101.0)
    assert out.net_pnl_inr < 0


def test_simulate_exit_long_t1_partial():
    """LONG with mfe=1.2R, mae=0.5R: T1 partial booked, remainder time-stop."""
    out = simulate_exit(
        side="LONG", entry_price=100.0, qty=50, R_per_share=1.0,
        mfe_r=1.2, mae_r=0.5, close_at_ts=100.8,
        T1_R=1.0, T2_R=2.0,
    )
    assert out is not None
    assert out.exit_reason == "t1_partial"
    assert out.net_pnl_inr > 0


def test_simulate_exit_time_stop_when_nothing_hit():
    """mfe=0.3R, mae=0.4R: nothing hit, exit at time-stop close."""
    out = simulate_exit(
        side="LONG", entry_price=100.0, qty=50, R_per_share=1.0,
        mfe_r=0.3, mae_r=0.4, close_at_ts=100.2,
        T1_R=1.0, T2_R=2.0,
    )
    assert out is not None
    assert out.exit_reason == "time_stop"
    # Small profit per share, minus fees -- could be positive or negative
    assert out.exit_price == pytest.approx(100.2)


def test_simulate_exit_pnl_sign_is_side_aware():
    """SHORT and LONG with same MFE/MAE/close should have OPPOSITE PnL signs."""
    common = dict(
        entry_price=100.0, qty=50, R_per_share=1.0,
        mfe_r=0.3, mae_r=0.4, close_at_ts=101.0,  # close UP
        T1_R=1.0, T2_R=2.0,
    )
    long_out = simulate_exit(side="LONG", **common)
    short_out = simulate_exit(side="SHORT", **common)
    assert long_out.net_pnl_inr > 0  # LONG benefits from close up
    assert short_out.net_pnl_inr < 0  # SHORT suffers


def test_simulate_exit_returns_none_for_bad_inputs():
    """Zero qty / zero R / NaN price -> None, do not crash sweep."""
    out = simulate_exit(
        side="LONG", entry_price=100.0, qty=0, R_per_share=1.0,
        mfe_r=0.5, mae_r=0.5, close_at_ts=100.0,
        T1_R=1.0, T2_R=2.0,
    )
    assert out is None

    out = simulate_exit(
        side="LONG", entry_price=float("nan"), qty=50, R_per_share=1.0,
        mfe_r=0.5, mae_r=0.5, close_at_ts=100.0,
        T1_R=1.0, T2_R=2.0,
    )
    assert out is None


def test_simulate_exit_rejects_unknown_side():
    with pytest.raises(ValueError):
        simulate_exit(
            side="FLAT", entry_price=100.0, qty=50, R_per_share=1.0,
            mfe_r=0.5, mae_r=0.5, close_at_ts=100.0,
            T1_R=1.0, T2_R=2.0,
        )


# ---------------------------------------------------------------------------
# run_cell_sweep end-to-end
# ---------------------------------------------------------------------------

def _synthetic_winning_candidates(n=300, win_rate=0.65, seed=42):
    """A SHORT candidates df where the small_cap subset has edge."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # Half small_cap, half mid_cap
        cap = "small_cap" if i % 2 == 0 else "mid_cap"
        # small_cap wins 70%, mid_cap wins 50%
        wr = 0.70 if cap == "small_cap" else 0.50
        is_win = rng.random() < wr
        if is_win:
            mfe = rng.uniform(2.0, 3.0)
            mae = rng.uniform(0.0, 0.6)
        else:
            mfe = rng.uniform(0.0, 0.5)
            mae = rng.uniform(1.0, 1.5)
        rows.append({
            "entry_ts": pd.Timestamp("2024-01-02 10:00") + pd.Timedelta(minutes=5 * i),
            "entry_price": 100.0,
            "qty": 50,
            "mfe_r": mfe,
            "mae_r": mae,
            "R_per_share": 1.0,
            "close_at_1500": 100.0 - 0.5 * mfe + 0.5 * mae,
            "cap_segment": cap,
        })
    return pd.DataFrame(rows)


def test_run_cell_sweep_finds_winning_cell():
    df = _synthetic_winning_candidates(n=400)
    cfg = CellSweepConfig(
        side="SHORT",
        r_grid=[(1.0, 2.0, 1500, "baseline")],
        dim_pool=["cap_segment"],
        n_min_floor=50, pf_min_floor=1.10,
        n_min_ship=80, pf_min_ship=1.30,
    )
    results = run_cell_sweep(df, cfg)
    # small_cap should pass; mid_cap should not (PF<1.0 by construction)
    assert not results.empty
    assert (results["cell_label"].str.contains("small_cap")).any()


def test_run_cell_sweep_raises_on_lookahead_dim():
    df = _synthetic_winning_candidates(n=100)
    df["day_high"] = 105.0
    cfg = CellSweepConfig(
        side="SHORT",
        r_grid=[(1.0, 2.0, 1500, "baseline")],
        dim_pool=["day_high"],
    )
    with pytest.raises(ValueError, match="lookahead_dim"):
        run_cell_sweep(df, cfg)


def test_select_best_cell_returns_none_when_no_ship_eligible():
    """REGRESSION (Lesson #2 anti-salvage): if no cell is ship-eligible, return
    None — kill signal, NOT silently pick the marginal best."""
    df = pd.DataFrame([{
        "dims": ["cap_segment"], "cell_label": "cap_segment=small_cap",
        "r_label": "baseline", "T1_R": 1.0, "T2_R": 2.0, "TS": 1500,
        "n": 150, "pf": 1.15, "wr_pct": 55.0,
        "net_pnl_inr": 1000.0, "expectancy_inr": 6.67,
    }])
    cfg = CellSweepConfig(
        side="SHORT", r_grid=[(1.0, 2.0, 1500, "b")], dim_pool=["cap_segment"],
        n_min_ship=200, pf_min_ship=1.30,
    )
    # n=150 < n_min_ship=200 -> not eligible
    assert select_best_cell(df, cfg) is None


def test_select_best_cell_picks_top_pf_when_eligible():
    df = pd.DataFrame([
        {"dims": ["cap_segment"], "cell_label": "A", "r_label": "b",
         "T1_R": 1.0, "T2_R": 2.0, "TS": 1500,
         "n": 250, "pf": 1.40, "wr_pct": 60.0,
         "net_pnl_inr": 5000.0, "expectancy_inr": 20.0},
        {"dims": ["cap_segment"], "cell_label": "B", "r_label": "b",
         "T1_R": 1.0, "T2_R": 2.0, "TS": 1500,
         "n": 500, "pf": 1.32, "wr_pct": 58.0,
         "net_pnl_inr": 7500.0, "expectancy_inr": 15.0},
    ])
    cfg = CellSweepConfig(
        side="SHORT", r_grid=[(1.0, 2.0, 1500, "b")], dim_pool=["cap_segment"],
        n_min_ship=200, pf_min_ship=1.30,
    )
    best = select_best_cell(df, cfg)
    assert best is not None
    assert best["cell_label"] == "A"  # higher PF wins


# ---------------------------------------------------------------------------
# lock_cell discipline
# ---------------------------------------------------------------------------

def test_lock_cell_writes_json(tmp_path):
    sel = {"cell_label": "cap_segment=small_cap", "pf": 1.42, "n": 234}
    out = tmp_path / "locked.json"
    written = lock_cell(sel, setup_name="my_setup", window_label="Discovery", output_path=out)
    assert written.exists()
    import json
    payload = json.loads(written.read_text())
    assert payload["setup_name"] == "my_setup"
    assert payload["selected_cell"]["cell_label"] == "cap_segment=small_cap"


def test_lock_cell_refuses_overwrite_without_force(tmp_path):
    """REGRESSION (Lesson #2 anti-p-hack): a lock cannot be re-selected on the
    same data without explicit acknowledgment via force=True."""
    sel = {"cell_label": "A", "pf": 1.3}
    out = tmp_path / "lock.json"
    lock_cell(sel, setup_name="s", window_label="Discovery", output_path=out)
    with pytest.raises(FileExistsError):
        lock_cell(sel, setup_name="s", window_label="Discovery", output_path=out)


def test_lock_cell_allows_overwrite_with_force(tmp_path):
    sel = {"cell_label": "A", "pf": 1.3}
    out = tmp_path / "lock.json"
    lock_cell(sel, setup_name="s", window_label="Discovery", output_path=out)
    lock_cell(sel, setup_name="s", window_label="Discovery", output_path=out, force=True)
