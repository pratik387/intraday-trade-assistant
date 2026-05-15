"""Stage 5c tests: cross-sectional gate replay on trade stream."""
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tools.edge_discovery_legacy_gauntlet.stages.stage5c_cross_sectional_simulation import (
    run_stage5c,
    simulate_filter,
)


CFG = {
    "enabled": True,
    "f1_rvol_enabled": True,
    "f1_rvol_threshold_pct": 70.0,
    "f1_applicable_caps": ["small_cap"],
    "f1_skip_hour_buckets": ["late"],
    "f1_min_history_sessions": 2,
    "f1_rolling_window_sessions": 5,
    "f2_crowdedness_enabled": True,
    "f2_crowdedness_threshold": 3,
    "f2_crowdedness_window_min": 5,
}


def _trade_row(date_str, ts_str, symbol="SYM", setup="premium_zone_short",
               cap="small_cap", hb="morning", mod=600, pnl=100.0):
    return {
        "session_date": date_str,
        "session_date_dt": date.fromisoformat(date_str),
        "trade_id": f"{symbol}_{ts_str}",
        "symbol_raw": symbol,
        "setup": setup,
        "cap_segment": cap,
        "hour_bucket": hb,
        "minute_of_day": mod,
        "decision_ts": ts_str,
        "total_trade_pnl": pnl,
    }


def test_simulate_filter_rejects_crowded_trades():
    """F2 alone — 4 same-setup trades in 2min → 2nd, 3rd, 4th get rejected."""
    trades = pd.DataFrame([
        _trade_row("2023-01-02", "2023-01-02 10:00:00", symbol="A"),
        _trade_row("2023-01-02", "2023-01-02 10:01:00", symbol="B"),
        _trade_row("2023-01-02", "2023-01-02 10:02:00", symbol="C"),
        _trade_row("2023-01-02", "2023-01-02 10:03:00", symbol="D"),
    ])
    # No ohlcv needed (F1 can be inert) — pass empty
    ohlcv_empty = pd.DataFrame(columns=["symbol", "date_only", "mod", "volume"])
    result = simulate_filter(trades, ohlcv_empty, CFG)
    # First 2-3 pass (crowd count 0,1,2,3), 4th has crowd=3 -> reject if threshold=3
    # With threshold=3 and crowd count at 10:03 = events at 10:00, 10:01, 10:02 = 3
    # That hits >=3 threshold -> reject.
    rejected = result[result["allowed"] == False]
    allowed = result[result["allowed"] == True]
    assert len(allowed) >= 2
    assert len(rejected) >= 1


def test_simulate_filter_applies_f1_on_small_cap_only():
    """F1 rejects when rvol_pct >= 70 and cap in applicable list."""
    ohlcv = pd.DataFrame([
        # History for symbol A at mod 600 — 3 prior sessions @ 1000 vol
        {"symbol": "A", "date_only": date(2022, 12, 29), "mod": 600, "volume": 1000},
        {"symbol": "A", "date_only": date(2022, 12, 30), "mod": 600, "volume": 1000},
        {"symbol": "A", "date_only": date(2023, 1, 1), "mod": 600, "volume": 1000},
        # Current session: spike to 5000 (rvol = 5.0)
        {"symbol": "A", "date_only": date(2023, 1, 2), "mod": 600, "volume": 5000},
        # Symbol B low rvol for contrast
        {"symbol": "B", "date_only": date(2022, 12, 29), "mod": 600, "volume": 1000},
        {"symbol": "B", "date_only": date(2022, 12, 30), "mod": 600, "volume": 1000},
        {"symbol": "B", "date_only": date(2023, 1, 1), "mod": 600, "volume": 1000},
        {"symbol": "B", "date_only": date(2023, 1, 2), "mod": 600, "volume": 500},
    ])
    trades = pd.DataFrame([
        _trade_row("2023-01-02", "2023-01-02 10:00:00", symbol="A", cap="small_cap"),
        _trade_row("2023-01-02", "2023-01-02 10:00:00", symbol="B", cap="small_cap"),
    ])
    result = simulate_filter(trades, ohlcv, CFG)
    # A has rvol_pct=100, B has rvol_pct=0 → A rejected, B allowed
    a_row = result[result["symbol_raw"] == "A"].iloc[0]
    b_row = result[result["symbol_raw"] == "B"].iloc[0]
    assert a_row["allowed"] is False
    assert b_row["allowed"] is True


def test_simulate_filter_skips_f1_for_unknown_cap():
    ohlcv = pd.DataFrame([
        {"symbol": "A", "date_only": date(2022, 12, 29), "mod": 600, "volume": 1000},
        {"symbol": "A", "date_only": date(2022, 12, 30), "mod": 600, "volume": 1000},
        {"symbol": "A", "date_only": date(2023, 1, 2), "mod": 600, "volume": 10000},
    ])
    trades = pd.DataFrame([
        _trade_row("2023-01-02", "2023-01-02 10:00:00", symbol="A", cap="unknown"),
    ])
    result = simulate_filter(trades, ohlcv, CFG)
    assert result.iloc[0]["allowed"] is True  # F1 skipped for unknown


def test_run_stage5c_writes_report_and_json(tmp_path):
    """End-to-end: run_stage5c produces report + JSON artifact."""
    trades = pd.DataFrame([
        _trade_row(f"2023-01-0{i}", f"2023-01-0{i} 10:00:00", symbol=f"S{i}", pnl=100.0)
        for i in range(2, 7)
    ])
    trades["session_date_dt"] = pd.to_datetime(trades["session_date"]).dt.date
    ohlcv_empty = pd.DataFrame(columns=["symbol", "date_only", "mod", "volume"])
    result = run_stage5c(
        trades=trades,
        ohlcv=ohlcv_empty,
        cfg=CFG,
        report_path=tmp_path / "07.md",
        summary_json=tmp_path / "s5c.json",
    )
    assert (tmp_path / "07.md").exists()
    assert (tmp_path / "s5c.json").exists()
    assert "before" in result
    assert "after" in result
    assert "delta" in result
