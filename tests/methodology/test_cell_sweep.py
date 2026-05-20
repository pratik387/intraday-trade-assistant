"""Tests for tools.methodology.cell_sweep v2 (R / pct / structural target modes).

Each test pins one of the bug-prone behaviors that recurred in per-setup
sweep scripts before this helper existed (Lesson #5 failure modes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.methodology.cell_sweep import (
    ALLOWED_PARTIAL_MODES,
    ALLOWED_SIDES,
    CellSweepConfig,
    FORBIDDEN_DIM_EXACT,
    FORBIDDEN_DIM_PREFIXES,
    GridEntry,
    lock_cell,
    run_cell_sweep,
    select_best_cell,
    simulate_exit,
    validate_candidates_schema,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _good_df_R(n=10):
    """Candidates df valid for target_unit='R'."""
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


def _good_df_pct(n=10):
    return pd.DataFrame({
        "entry_ts": pd.date_range("2024-01-02 10:00", periods=n, freq="5min"),
        "entry_price": [100.0] * n,
        "qty": [50] * n,
        "mfe_pct": [0.8] * n,
        "mae_pct": [0.5] * n,
        "close_at_1500": [99.5] * n,
        "cap_segment": (["small_cap"] * n)[:n],
    })


def _good_df_structural(n=10, side="SHORT"):
    sign = -1 if side == "SHORT" else 1
    return pd.DataFrame({
        "entry_ts": pd.date_range("2024-01-02 10:00", periods=n, freq="5min"),
        "entry_price": [100.0] * n,
        "qty": [50] * n,
        "mfe_r": [0.8] * n,
        "mae_r": [0.5] * n,
        "R_per_share": [1.0] * n,
        "t1_price": [100.0 + sign * 0.5] * n,    # 0.5R favorable
        "t2_price": [100.0 + sign * 1.5] * n,    # 1.5R favorable
        "close_at_1500": [99.5] * n,
        "cap_segment": (["small_cap"] * n)[:n],
    })


def test_schema_validator_passes_R_mode():
    df = _good_df_R(20)
    result = validate_candidates_schema(
        df, target_unit="R", dim_pool=["cap_segment"], ts_hhmms=[1500],
    )
    assert result.is_valid


def test_schema_validator_passes_pct_mode():
    df = _good_df_pct(20)
    result = validate_candidates_schema(
        df, target_unit="pct", dim_pool=["cap_segment"], ts_hhmms=[1500],
    )
    assert result.is_valid


def test_schema_validator_passes_structural_mode():
    df = _good_df_structural(20)
    result = validate_candidates_schema(
        df, target_unit="structural", dim_pool=["cap_segment"], ts_hhmms=[1500],
    )
    assert result.is_valid


def test_schema_validator_blocks_lookahead_dim_exact_match():
    """Lesson #5 failure mode #1 — day_high/day_low/day_vwap at signal."""
    df = _good_df_R()
    df["day_high"] = 105.0
    result = validate_candidates_schema(
        df, target_unit="R", dim_pool=["day_high"], ts_hhmms=[1500],
    )
    assert not result.is_valid
    assert any(i.code == "lookahead_dim" for i in result.issues)


def test_schema_validator_blocks_close_off_high_prefix():
    """Concrete case from _circuit_release_fade — removed 2026-05-16."""
    df = _good_df_R()
    df["close_off_high_bucket"] = "0-0.3"
    result = validate_candidates_schema(
        df, target_unit="R", dim_pool=["close_off_high_bucket"],
        ts_hhmms=[1500],
    )
    assert not result.is_valid
    assert any(i.code == "lookahead_dim" for i in result.issues)


def test_schema_validator_ALLOWS_day_gain_bucket_v2_relaxed_rule():
    """REGRESSION (v1 over-broad rule): day_gain_bucket is legitimate when the
    underlying value is computed from session_high_so_far at signal time.

    v1 had `FORBIDDEN_DIM_PREFIXES = ("day_", ...)` which incorrectly rejected
    this dim. v2 uses an exact blocklist (day_high/low/vwap/close/volume/range/atr)
    plus prefix blocklist for close_off_high/EOD_/eod_/session_close_.
    """
    df = _good_df_R()
    df["day_gain_bucket"] = "5-10"
    result = validate_candidates_schema(
        df, target_unit="R", dim_pool=["day_gain_bucket"], ts_hhmms=[1500],
    )
    assert result.is_valid, f"day_gain_bucket should be allowed; issues: {result.issues}"


def test_schema_validator_catches_missing_required_columns_pct_mode():
    df = _good_df_pct().drop(columns=["mfe_pct"])
    result = validate_candidates_schema(
        df, target_unit="pct", dim_pool=["cap_segment"], ts_hhmms=[1500],
    )
    assert not result.is_valid
    assert any(i.code == "missing_required" for i in result.issues)


def test_schema_validator_catches_missing_structural_columns():
    df = _good_df_structural().drop(columns=["t1_price"])
    result = validate_candidates_schema(
        df, target_unit="structural", dim_pool=["cap_segment"], ts_hhmms=[1500],
    )
    assert not result.is_valid
    assert any(i.code == "missing_required" for i in result.issues)


def test_schema_validator_catches_missing_close_col():
    df = _good_df_R()
    # ts_hhmms includes 1300 but only close_at_1500 exists
    result = validate_candidates_schema(
        df, target_unit="R", dim_pool=["cap_segment"], ts_hhmms=[1500, 1300],
    )
    assert not result.is_valid
    assert any(i.code == "close_col_missing" for i in result.issues)


def test_schema_validator_catches_sign_violation_pct():
    df = _good_df_pct()
    df.loc[df.index[0], "mfe_pct"] = -0.5
    result = validate_candidates_schema(
        df, target_unit="pct", dim_pool=["cap_segment"], ts_hhmms=[1500],
    )
    assert not result.is_valid
    assert any(i.code == "negative_mfe" for i in result.issues)


# ---------------------------------------------------------------------------
# simulate_exit — R mode
# ---------------------------------------------------------------------------

def _grid_R(t1=1.0, t2=2.0, ts=1500, partial="partial_50_no_trail", label="b"):
    return GridEntry(label=label, ts_hhmm=ts, partial_mode=partial,
                     t1=t1, t2=t2, sl=1.0)


def test_simulate_exit_R_short_full_t2():
    out = simulate_exit(
        target_unit="R", side="SHORT", grid=_grid_R(),
        entry_price=100.0, qty=50, close_at_ts=98.0,
        mfe=2.5, mae=0.5, R_per_share=1.0,
    )
    assert out is not None
    assert out.exit_reason == "t2_full"
    assert out.exit_price < 100.0
    assert out.net_pnl_inr > 0


def test_simulate_exit_R_same_bar_stop_pessimism():
    """REGRESSION (Lesson #5 FM #4): stop wins when both stop and T2 hit."""
    out = simulate_exit(
        target_unit="R", side="SHORT", grid=_grid_R(),
        entry_price=100.0, qty=50, close_at_ts=98.0,
        mfe=3.0, mae=1.0, R_per_share=1.0,
    )
    assert out.exit_reason == "sl"
    assert out.exit_price == pytest.approx(101.0)
    assert out.net_pnl_inr < 0


def test_simulate_exit_R_all_in_mode_no_t1_partial():
    """all_in: T1 alone doesn't fire; only T2 or TS resolves the trade."""
    out = simulate_exit(
        target_unit="R", side="LONG",
        grid=_grid_R(partial="all_in"),
        entry_price=100.0, qty=50, close_at_ts=100.5,
        mfe=1.2, mae=0.5, R_per_share=1.0,
    )
    # T1=1.0R hit (mfe=1.2), T2=2.0R not hit (mfe<2.0). all_in -> TS exit.
    assert out.exit_reason == "time_stop"


def test_simulate_exit_R_partial_50_no_trail():
    """T1 partial + remainder at TS close."""
    out = simulate_exit(
        target_unit="R", side="LONG",
        grid=_grid_R(partial="partial_50_no_trail"),
        entry_price=100.0, qty=50, close_at_ts=100.5,
        mfe=1.2, mae=0.5, R_per_share=1.0,
    )
    assert out.exit_reason == "t1_partial"


def test_simulate_exit_R_partial_50_be_trail_triggers_when_mae_high():
    """BE trail (conservative): T1 hit AND mae >= 0.75 -> remaining 50% at BE."""
    out = simulate_exit(
        target_unit="R", side="LONG",
        grid=_grid_R(partial="partial_50_be_trail"),
        entry_price=100.0, qty=50, close_at_ts=99.5,
        mfe=1.2, mae=0.85, R_per_share=1.0,  # mae>=0.75 -> BE trail trips
    )
    assert out.exit_reason == "t1_be_trail"
    assert out.exit_price == pytest.approx(100.0)  # remaining exits at entry (BE)


def test_simulate_exit_R_partial_50_be_trail_does_not_trip_low_mae():
    """BE trail does NOT trip when mae stays small after T1."""
    out = simulate_exit(
        target_unit="R", side="LONG",
        grid=_grid_R(partial="partial_50_be_trail"),
        entry_price=100.0, qty=50, close_at_ts=100.8,
        mfe=1.2, mae=0.3, R_per_share=1.0,  # mae<0.75 -> stays in partial mode
    )
    assert out.exit_reason == "t1_partial"


# ---------------------------------------------------------------------------
# simulate_exit — pct mode
# ---------------------------------------------------------------------------

def test_simulate_exit_pct_short_t2_hit():
    grid = GridEntry(label="b", ts_hhmm=1500, t1=0.5, t2=1.5, sl=1.0)
    out = simulate_exit(
        target_unit="pct", side="SHORT", grid=grid,
        entry_price=100.0, qty=50, close_at_ts=98.5,
        mfe=2.0, mae=0.4,  # mfe=2% > t2=1.5%
    )
    assert out is not None
    assert out.exit_reason == "t2_full"
    assert out.exit_price == pytest.approx(100.0 * (1 - 1.5 / 100.0))


def test_simulate_exit_pct_stop_hit():
    grid = GridEntry(label="b", ts_hhmm=1500, t1=0.5, t2=1.5, sl=1.0)
    out = simulate_exit(
        target_unit="pct", side="SHORT", grid=grid,
        entry_price=100.0, qty=50, close_at_ts=101.0,
        mfe=0.2, mae=1.2,  # mae=1.2% > sl=1.0% -> stop
    )
    assert out.exit_reason == "sl"
    assert out.exit_price == pytest.approx(101.0)


# ---------------------------------------------------------------------------
# simulate_exit — structural mode
# ---------------------------------------------------------------------------

def test_simulate_exit_structural_short_t2_hit():
    """SHORT setup with t1=99.5, t2=98.5 (T1=0.5R, T2=1.5R)."""
    grid = GridEntry(label="b", ts_hhmm=1500)
    out = simulate_exit(
        target_unit="structural", side="SHORT", grid=grid,
        entry_price=100.0, qty=50, close_at_ts=99.0,
        mfe=2.0, mae=0.4, R_per_share=1.0,
        t1_price=99.5, t2_price=98.5,
    )
    assert out is not None
    assert out.exit_reason == "t2_full"
    assert out.exit_price == pytest.approx(98.5)


def test_simulate_exit_structural_rejects_targets_on_wrong_side():
    """Structural t1/t2 on adverse side of entry should fail validation."""
    grid = GridEntry(label="b", ts_hhmm=1500)
    out = simulate_exit(
        target_unit="structural", side="SHORT", grid=grid,
        entry_price=100.0, qty=50, close_at_ts=99.0,
        mfe=0.5, mae=0.3, R_per_share=1.0,
        t1_price=101.0,  # WRONG: SHORT t1 should be < entry
        t2_price=102.0,
    )
    assert out is None  # _to_r_units raises ValueError, simulate_exit catches it


# ---------------------------------------------------------------------------
# simulate_exit — edges
# ---------------------------------------------------------------------------

def test_simulate_exit_pnl_sign_is_side_aware():
    grid = _grid_R()
    common = dict(
        target_unit="R", grid=grid, entry_price=100.0, qty=50,
        close_at_ts=101.0,  # close UP
        mfe=0.3, mae=0.4, R_per_share=1.0,
    )
    long_out = simulate_exit(side="LONG", **common)
    short_out = simulate_exit(side="SHORT", **common)
    assert long_out.net_pnl_inr > 0
    assert short_out.net_pnl_inr < 0


def test_simulate_exit_rejects_unknown_side():
    with pytest.raises(ValueError):
        simulate_exit(
            target_unit="R", side="FLAT", grid=_grid_R(),
            entry_price=100.0, qty=50, close_at_ts=100.0,
            mfe=0.5, mae=0.5, R_per_share=1.0,
        )


def test_simulate_exit_returns_none_for_zero_qty():
    out = simulate_exit(
        target_unit="R", side="LONG", grid=_grid_R(),
        entry_price=100.0, qty=0, close_at_ts=100.0,
        mfe=0.5, mae=0.5, R_per_share=1.0,
    )
    assert out is None


# ---------------------------------------------------------------------------
# CellSweepConfig validation
# ---------------------------------------------------------------------------

def test_config_rejects_R_grid_entry_missing_t1():
    with pytest.raises(ValueError, match="R mode requires"):
        CellSweepConfig(
            side="SHORT", target_unit="R", dim_pool=["cap_segment"],
            grid=[GridEntry(label="b", ts_hhmm=1500, t2=2.0)],  # t1 missing
        )


def test_config_rejects_pct_grid_entry_missing_sl():
    with pytest.raises(ValueError, match="pct mode requires"):
        CellSweepConfig(
            side="SHORT", target_unit="pct", dim_pool=["cap_segment"],
            grid=[GridEntry(label="b", ts_hhmm=1500, t1=0.5, t2=1.5)],  # sl missing
        )


def test_config_accepts_structural_with_no_t1_t2_in_grid():
    cfg = CellSweepConfig(
        side="SHORT", target_unit="structural", dim_pool=["cap_segment"],
        grid=[GridEntry(label="b", ts_hhmm=1500)],  # no t1/t2/sl
    )
    assert cfg.target_unit == "structural"


def test_grid_entry_rejects_unknown_partial_mode():
    with pytest.raises(ValueError, match="partial_mode"):
        GridEntry(label="b", ts_hhmm=1500, partial_mode="all_or_nothing")


# ---------------------------------------------------------------------------
# run_cell_sweep — end-to-end across modes
# ---------------------------------------------------------------------------

def _synthetic_R_candidates(n=400, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        cap = "small_cap" if i % 2 == 0 else "mid_cap"
        wr = 0.70 if cap == "small_cap" else 0.50
        is_win = rng.random() < wr
        if is_win:
            mfe = rng.uniform(2.0, 3.0); mae = rng.uniform(0.0, 0.6)
        else:
            mfe = rng.uniform(0.0, 0.5); mae = rng.uniform(1.0, 1.5)
        rows.append({
            "entry_ts": pd.Timestamp("2024-01-02 10:00") + pd.Timedelta(minutes=5 * i),
            "entry_price": 100.0, "qty": 50,
            "mfe_r": mfe, "mae_r": mae, "R_per_share": 1.0,
            "close_at_1500": 100.0 - 0.5 * mfe + 0.5 * mae,
            "cap_segment": cap,
        })
    return pd.DataFrame(rows)


def test_run_cell_sweep_R_mode_finds_winning_cell():
    df = _synthetic_R_candidates(n=400)
    cfg = CellSweepConfig(
        side="SHORT", target_unit="R", dim_pool=["cap_segment"],
        grid=[_grid_R(t1=1.0, t2=2.0, ts=1500, label="baseline")],
        n_min_floor=50, pf_min_floor=1.10,
        n_min_ship=80, pf_min_ship=1.30,
    )
    results = run_cell_sweep(df, cfg)
    assert not results.empty
    assert (results["cell_label"].str.contains("small_cap")).any()


def test_run_cell_sweep_raises_on_lookahead_dim():
    df = _synthetic_R_candidates(n=100)
    df["day_high"] = 105.0
    cfg = CellSweepConfig(
        side="SHORT", target_unit="R", dim_pool=["day_high"],
        grid=[_grid_R()],
    )
    with pytest.raises(ValueError, match="lookahead_dim"):
        run_cell_sweep(df, cfg)


def test_run_cell_sweep_sweeps_multiple_partial_modes():
    """A grid with all 3 partial modes produces results for each."""
    df = _synthetic_R_candidates(n=400)
    cfg = CellSweepConfig(
        side="SHORT", target_unit="R", dim_pool=["cap_segment"],
        grid=[
            _grid_R(partial="all_in", label="all_in"),
            _grid_R(partial="partial_50_no_trail", label="no_trail"),
            _grid_R(partial="partial_50_be_trail", label="be_trail"),
        ],
        n_min_floor=20, pf_min_floor=1.0,
    )
    results = run_cell_sweep(df, cfg)
    if not results.empty:
        assert results["partial_mode"].nunique() >= 1


# ---------------------------------------------------------------------------
# Selection + lock
# ---------------------------------------------------------------------------

def test_select_best_cell_returns_none_when_no_ship_eligible():
    """Anti-salvage (Lesson #2): no eligible cell -> kill signal, not soft-pick."""
    df = pd.DataFrame([{
        "dims": ["cap_segment"], "cell_label": "x", "grid_label": "b",
        "ts_hhmm": 1500, "partial_mode": "partial_50_no_trail",
        "t1": 1.0, "t2": 2.0, "sl": 1.0,
        "n": 150, "pf": 1.15, "wr_pct": 55.0,
        "net_pnl_inr": 1000.0, "expectancy_inr": 6.67,
    }])
    cfg = CellSweepConfig(
        side="SHORT", target_unit="R", dim_pool=["cap_segment"],
        grid=[_grid_R()],
        n_min_ship=200, pf_min_ship=1.30,
    )
    assert select_best_cell(df, cfg) is None


def test_lock_cell_refuses_overwrite_without_force(tmp_path):
    sel = {"cell_label": "A", "pf": 1.3}
    out = tmp_path / "lock.json"
    lock_cell(sel, setup_name="s", window_label="Discovery", output_path=out)
    with pytest.raises(FileExistsError):
        lock_cell(sel, setup_name="s", window_label="Discovery", output_path=out)


def test_lock_cell_allows_force_overwrite(tmp_path):
    sel = {"cell_label": "A", "pf": 1.3}
    out = tmp_path / "lock.json"
    lock_cell(sel, setup_name="s", window_label="Discovery", output_path=out)
    lock_cell(sel, setup_name="s", window_label="Discovery",
              output_path=out, force=True)
