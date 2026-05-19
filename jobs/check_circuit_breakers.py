"""Daily-EOD circuit breaker check.

For each enabled setup with cb_drawdown_threshold defined, computes
trailing-60-day NET PnL from trade_report.csv and auto-disables the
setup if PnL falls below threshold (and trade count meets minimum).

Usage:
    .venv/Scripts/python jobs/check_circuit_breakers.py \\
        --trades-csv reports/run_latest/trade_report.csv \\
        --config config/configuration.json

Run via cron at 16:00 IST (after market close, after EOD reports).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.methodology.setup_metadata import write_setup_block_atomic


@dataclass(frozen=True)
class CBState:
    setup_name: str
    action: str               # "disable" | "no_change"
    trailing_pnl: float
    trades_in_window: int
    reason: str


def check_setup_circuit_breaker(
    setup_name: str,
    trades_csv: Path,
    lookback_days: int,
    threshold: float,
    min_trades: int,
    today: date,
) -> CBState:
    """Compute trailing-N-day NET PnL for `setup_name`; return CBState with action."""
    df = pd.read_csv(trades_csv)
    if "signal_date" not in df.columns:
        return CBState(setup_name, "no_change", 0.0, 0, "missing_signal_date_column")
    if "setup_type" not in df.columns:
        return CBState(setup_name, "no_change", 0.0, 0, "missing_setup_type_column")
    if "actual_pnl_after_charges" not in df.columns:
        return CBState(setup_name, "no_change", 0.0, 0, "missing_actual_pnl_column")

    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
    cutoff = today - timedelta(days=lookback_days)
    win = df[(df["setup_type"] == setup_name) & (df["signal_date"] >= cutoff)]

    n = len(win)
    pnl = float(win["actual_pnl_after_charges"].sum()) if n > 0 else 0.0

    if n < min_trades:
        return CBState(setup_name, "no_change", pnl, n, "insufficient_trades")

    if pnl < threshold:
        return CBState(setup_name, "disable", pnl, n, "60d_pnl_below_threshold")
    return CBState(setup_name, "no_change", pnl, n, "above_threshold")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--trades-csv", type=Path, required=True)
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "config" / "configuration.json")
    p.add_argument("--today", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                   default=date.today())
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    setups = cfg.get("setups", {})

    n_checked = 0
    n_disabled = 0
    for name, block in setups.items():
        if not block.get("enabled", False):
            continue
        threshold = block.get("cb_drawdown_threshold")
        if threshold is None:
            continue
        if block.get("cb_state") == "disabled":
            continue  # already disabled; manual re-enable required
        lookback = block.get("cb_lookback_days", 60)
        min_trades = block.get("cb_min_trades_for_signal", 30)

        state = check_setup_circuit_breaker(
            setup_name=name,
            trades_csv=args.trades_csv,
            lookback_days=lookback,
            threshold=float(threshold),
            min_trades=int(min_trades),
            today=args.today,
        )
        n_checked += 1
        print(f"[cb] {name}: action={state.action} "
              f"trailing_pnl={state.trailing_pnl:.2f} n={state.trades_in_window} "
              f"reason={state.reason}")

        if state.action == "disable":
            n_disabled += 1
            write_setup_block_atomic(args.config, name, {
                "cb_state": "disabled",
                "cb_disabled_at": args.today.isoformat(),
                "cb_disabled_reason": state.reason,
            })

    print(f"\n[cb] checked={n_checked} disabled={n_disabled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
