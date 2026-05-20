"""CLI driver: run walk-forward on a setup, write results to config.

Usage:
    .venv/Scripts/python tools/methodology/run_walk_forward.py \\
        --setup pre_results_t1_fade \\
        --trades-csv reports/sub9_sanity/_pre_results_t1_v2_trades_combined.csv \\
        --config config/configuration.json \\
        --start 2023-01-01 --end 2026-03-31

The trades CSV must have columns: signal_date, symbol, pnl_pct.
Combine Disc + OOS + HO trade ledgers into one CSV before running.

Writes walk_forward_results + cb_drawdown_threshold + cb_state + position_size_multiplier
into the setup block. Refuses to run if mechanism_tags is not pre-registered.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.methodology.data_health import check_all as data_health_check
from tools.methodology.pre_registration import (
    check_mechanism_pre_registered, PreRegistrationError,
)
from tools.methodology.setup_metadata import write_setup_block_atomic
from tools.methodology.walk_forward import run_walk_forward


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv=None):
    p = argparse.ArgumentParser(description="Run walk-forward validation on a setup.")
    p.add_argument("--setup", required=True,
                   help="Setup name (key under setups.* in config)")
    p.add_argument("--trades-csv", type=Path, required=True,
                   help="Combined Disc+OOS+HO trades CSV")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "config" / "configuration.json")
    p.add_argument("--start", type=_parse_date, default=date(2023, 1, 1))
    p.add_argument("--end", type=_parse_date, default=date(2026, 3, 31))
    p.add_argument("--n-windows", type=int, default=13)
    p.add_argument("--window-months", type=int, default=3)
    p.add_argument("--fee-pct", type=float, default=0.25,
                   help="Round-trip fee on CAPITAL basis (default 0.25%). "
                        "Verified against real trades 2026-05-20: 0.05% on notional "
                        "x 5x MIS leverage = 0.25%% on capital.")
    p.add_argument("--mis-leverage", type=float, default=5.0)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--skip-pre-registration-check", action="store_true",
                   help="DANGER: skip git-timestamp check. Only for dry-runs.")
    p.add_argument("--skip-data-health", action="store_true",
                   help="DANGER: skip data_health.py drift detection. Only for dry-runs.")
    p.add_argument("--data-health-log-dir", type=Path,
                   default=REPO_ROOT / "reports" / "data_health",
                   help="Directory to write JSONL audit log of health issues.")
    args = p.parse_args(argv)

    # Pre-registration check
    if not args.skip_pre_registration_check:
        try:
            check_mechanism_pre_registered(REPO_ROOT, args.config, args.setup)
        except PreRegistrationError as e:
            print(f"PRE-REGISTRATION FAILED: {e}", file=sys.stderr)
            return 2

    # Load trades
    if not args.trades_csv.exists():
        print(f"trades CSV not found: {args.trades_csv}", file=sys.stderr)
        return 3
    trades = pd.read_csv(args.trades_csv)
    required_cols = {"signal_date", "pnl_pct"}
    missing = required_cols - set(trades.columns)
    if missing:
        print(
            f"trades CSV missing required columns: {sorted(missing)}; "
            f"got: {list(trades.columns)}",
            file=sys.stderr,
        )
        return 3

    # Data health check (Layers 1 + 3 on trades CSV; Layer 2 requires
    # explicit source data — caller adds via separate script).
    # Pass start/end so data_health uses the same window range as walk-forward
    # (prevents partial leading/trailing quarters from triggering false anomalies).
    if not args.skip_data_health:
        report = data_health_check(
            trades, args.setup,
            audit_log_dir=args.data_health_log_dir,
            start_date=args.start, end_date=args.end,
        )
        if report.issues:
            print(f"\n{report.summary()}")
        if report.has_blocking_issues:
            print(
                f"\nDATA HEALTH BLOCKED — refusing to run walk-forward. "
                f"Investigate {report.n_block} blocking issue(s) above. "
                f"Override with --skip-data-health (NOT RECOMMENDED for ship decisions).",
                file=sys.stderr,
            )
            return 5

    # Run walk-forward
    print(f"[walk-forward] setup={args.setup} n_trades={len(trades)} "
          f"windows={args.n_windows}")
    result = run_walk_forward(
        setup_name=args.setup,
        trades_df=trades,
        start=args.start, end=args.end,
        window_months=args.window_months, n_windows=args.n_windows,
        fee_pct_round_trip=args.fee_pct, mis_leverage=args.mis_leverage,
        bootstrap_n=args.bootstrap_n,
    )

    # Print per-window summary
    print(f"\n=== {args.setup} walk-forward result ===")
    print(f"{'#':>3} {'start':<12} {'end':<12} {'n':>5} {'PF_net':>7} {'CI_lo':>7} {'pass':>5}")
    for s in result.windows:
        ci = f"{s.bootstrap.ci_lower:.3f}" if s.bootstrap else "n/a"
        pf_net_str = f"{s.pf_net:>7.3f}" if s.pf_net != float("inf") else "    inf"
        print(f"{s.window.index:>3} {s.window.start} {s.window.end} {s.n:>5} "
              f"{pf_net_str} {ci:>7} {'YES' if s.passes_gate else 'no':>5}")

    print(f"\npass: {result.windows_pass}/{result.windows_total} "
          f"({result.pass_rate:.1%})")
    print(f"tier: {result.tier.value}")
    print(f"cb_drawdown_threshold: {result.cb_drawdown_threshold:.2f}")

    # Write back to config
    tier_value = result.tier.value
    if tier_value == "GREEN":
        cb_state = "enabled"
        position_size_multiplier = 1.0
    elif tier_value == "AMBER":
        cb_state = "forward_validation"
        position_size_multiplier = 0.25
    else:  # RED
        cb_state = "disabled"
        position_size_multiplier = 0.0

    updates = {
        "walk_forward_results": {
            "windows_pass": result.windows_pass,
            "windows_total": result.windows_total,
            "pass_rate": result.pass_rate,
            "tier": tier_value,
            "evaluated_at": datetime.now().date().isoformat(),
            "engine_version": "1.0",
        },
        "cb_drawdown_threshold": result.cb_drawdown_threshold,
        "cb_lookback_days": 60,
        "cb_min_trades_for_signal": 30,
        "cb_state": cb_state,
        "position_size_multiplier": position_size_multiplier,
    }
    write_setup_block_atomic(args.config, args.setup, updates)
    print(f"\nwrote results to {args.config} (setups.{args.setup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
