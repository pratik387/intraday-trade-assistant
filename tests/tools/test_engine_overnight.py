"""Unit tests for tools/engine_overnight.py — OCI-style overnight backtest harness.

These tests cover the pure helper functions (no subprocess invocation) since
the full driver requires market data + state-file infrastructure that isn't
available in CI. End-to-end validation is via manual range runs.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.engine_overnight import (
    _aggregate_trades,
    _daterange,
    _load_setups_config,
    _next_trading_day,
    _profit_factor,
    _state_files_from_config,
)


# ---------------------------------------------------------------------------
# _daterange
# ---------------------------------------------------------------------------

def test_daterange_inclusive_both_ends():
    days = list(_daterange(date(2026, 4, 20), date(2026, 4, 22)))
    assert days == [date(2026, 4, 20), date(2026, 4, 21), date(2026, 4, 22)]


def test_daterange_single_day():
    assert list(_daterange(date(2026, 4, 20), date(2026, 4, 20))) == [date(2026, 4, 20)]


# ---------------------------------------------------------------------------
# _next_trading_day
# ---------------------------------------------------------------------------

def test_next_trading_day_skips_weekend():
    # 2026-04-24 is a Friday → next trading day is Monday 2026-04-27
    nd = _next_trading_day(date(2026, 4, 24))
    assert nd == date(2026, 4, 27)


def test_next_trading_day_normal_weekday():
    # 2026-04-20 is a Monday → next trading day is Tuesday 2026-04-21
    nd = _next_trading_day(date(2026, 4, 20))
    assert nd == date(2026, 4, 21)


# ---------------------------------------------------------------------------
# _profit_factor
# ---------------------------------------------------------------------------

def test_profit_factor_basic():
    # 3 wins of 100 each, 2 losses of 50 each → PF = 300/100 = 3.0
    pf = _profit_factor([100.0, 100.0, 100.0, -50.0, -50.0])
    assert pf == 3.0


def test_profit_factor_no_losses_returns_inf_with_wins():
    pf = _profit_factor([100.0, 50.0, 25.0])
    assert pf == float("inf")


def test_profit_factor_no_trades_returns_none():
    assert _profit_factor([]) is None


def test_profit_factor_all_zero_returns_none():
    # All zero is not strictly >0 nor <0 → no wins, no losses → None
    assert _profit_factor([0.0, 0.0]) is None


# ---------------------------------------------------------------------------
# _load_setups_config
# ---------------------------------------------------------------------------

def test_load_setups_config_returns_setups_block():
    """The harness loads configuration.json directly (NOT base_config.json)
    so it sees the `setups.*` blocks."""
    cfg = _load_setups_config()
    assert "setups" in cfg, "configuration.json must have a setups block"
    assert isinstance(cfg["setups"], dict)
    # Sanity: close_dn_overnight_long exists
    assert "close_dn_overnight_long" in cfg["setups"]


# ---------------------------------------------------------------------------
# _state_files_from_config
# ---------------------------------------------------------------------------

def test_state_files_from_config_finds_overnight_setup_state_files():
    cfg = _load_setups_config()
    paths = _state_files_from_config(cfg)
    # Expect at least slot pool + decay tripwire for close_dn_overnight_long
    assert len(paths) >= 2, f"expected >=2 state files, got {paths}"
    names = {p.name for p in paths}
    assert "overnight_slots.json" in names
    assert "decay_tripwire_close_dn_overnight_long.json" in names


def test_state_files_from_config_ignores_intraday_setups():
    """Only mode=overnight setups contribute state files."""
    fake_cfg = {
        "setups": {
            "intraday_setup": {
                "mode": "intraday",
                "capital_allocation": {"state_file": "state/should_not_appear.json"},
            },
            "overnight_setup": {
                "mode": "overnight",
                "capital_allocation": {"state_file": "state/should_appear.json"},
            },
        }
    }
    paths = _state_files_from_config(fake_cfg)
    names = {p.name for p in paths}
    assert "should_appear.json" in names
    assert "should_not_appear.json" not in names


# ---------------------------------------------------------------------------
# _aggregate_trades
# ---------------------------------------------------------------------------

def test_aggregate_trades_emits_settled_only(tmp_path):
    """Aggregator reads from `released_trades` archive (full lifecycle) PLUS
    in-flight `t1_settling` slots (settled but not yet released).

    Released-trade archive was added in capital_manager.py 2026-05-23 because
    pool.release() wiped slot fields, making historical trades invisible. The
    aggregator must read BOTH archive + still-held t1_settling positions to
    give a complete view for backtest harness output.
    """
    state_path = tmp_path / "overnight_slots.json"
    blob = {
        "released_trades": [
            # Fully-released winner (Variant B cohort)
            {
                "slot_id": 1, "symbol": "NSE:TEST1", "product": "MTF",
                "buy_fill_price": 100.0, "sell_fill_price": 102.0,
                "buy_fill_ts": "2026-04-20T15:25", "sell_fill_ts": "2026-04-21T09:15",
                "notional_inr": 100000.0, "margin_inr": 25000.0, "leverage": 4.0,
                "fees_inr": 50.0, "interest_inr": 40.0, "realized_pnl_inr": 1910.0,
                "reserved_today": "2026-04-20", "expected_exit_date": "2026-04-21",
                "paper_variant_b": True, "cash_back_date": "2026-04-22",
            },
            # Fully-released loser (baseline cohort)
            {
                "slot_id": 3, "symbol": "NSE:TEST2", "product": "CNC",
                "buy_fill_price": 200.0, "sell_fill_price": 195.0,
                "notional_inr": 200000.0, "margin_inr": 200000.0, "leverage": 1.0,
                "fees_inr": 80.0, "interest_inr": 0.0, "realized_pnl_inr": -1080.0,
                "reserved_today": "2026-04-21", "expected_exit_date": "2026-04-22",
                "paper_variant_b": False, "cash_back_date": "2026-04-23",
            },
        ],
        "slots": [
            # Still open (no sell fill yet) — must be excluded
            {
                "slot_id": 2, "symbol": "NSE:OPEN1",
                "buy_fill_price": 50.0, "sell_fill_price": None,
                "notional_inr": 50000.0, "margin_inr": 12500.0, "leverage": 4.0,
                "status": "t0_open",
            },
            # In-flight t1_settling — INCLUDED in aggregation (already has sell fill)
            {
                "slot_id": 4, "symbol": "NSE:INFLIGHT", "product": "MTF",
                "buy_fill_price": 300.0, "sell_fill_price": 305.0,
                "notional_inr": 300000.0, "margin_inr": 75000.0, "leverage": 4.0,
                "fees_inr": 100.0, "interest_inr": 50.0, "realized_pnl_inr": 1350.0,
                "reserved_today": "2026-04-29", "expected_exit_date": "2026-04-30",
                "paper_variant_b": False, "status": "t1_settling",
            },
        ]
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(blob, f)

    out_dir = tmp_path / "out"
    summary = _aggregate_trades([state_path], out_dir)

    # 2 released + 1 t1_settling = 3 trades. NSE:OPEN1 (t0_open) excluded.
    assert summary["n_trades"] == 3
    assert summary["wins"] == 2   # TEST1 +1910 + INFLIGHT +1350
    assert summary["losses"] == 1  # TEST2 -1080
    assert summary["sum_realized_pnl_inr"] == pytest.approx(1910.0 + 1350.0 - 1080.0)
    # Cohort split: TEST1 is variant_b=True, the rest are baseline
    cs = summary["cohort_split"]
    assert cs["variant_b_true_n"] == 1
    assert cs["variant_b_true_sum"] == pytest.approx(1910.0)
    assert cs["baseline_only_n"] == 2
    assert cs["baseline_only_sum"] == pytest.approx(1350.0 - 1080.0)

    csv_path = out_dir / "trades.csv"
    assert csv_path.exists()
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "NSE:TEST1" in csv_content
    assert "NSE:TEST2" in csv_content
    assert "NSE:INFLIGHT" in csv_content
    assert "NSE:OPEN1" not in csv_content


def test_aggregate_trades_handles_missing_state_file(tmp_path):
    """Aggregator must not crash when state file is absent (e.g., 0 fires)."""
    missing = tmp_path / "does_not_exist.json"
    out_dir = tmp_path / "out"
    summary = _aggregate_trades([missing], out_dir)
    assert summary["n_trades"] == 0
    assert summary["sum_realized_pnl_inr"] == 0.0
    assert summary["gross_PF"] is None
    # Placeholder CSV exists
    assert (out_dir / "trades.csv").exists()


def test_aggregate_trades_empty_slots_list(tmp_path):
    """Aggregator must handle state file with empty slots array."""
    state_path = tmp_path / "overnight_slots.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"slots": []}, f)
    out_dir = tmp_path / "out"
    summary = _aggregate_trades([state_path], out_dir)
    assert summary["n_trades"] == 0
    assert summary["gross_PF"] is None
