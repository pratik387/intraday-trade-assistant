"""One-time helper to combine per-period trade CSVs into a single file
suitable for the walk-forward CLI.

The walk-forward engine expects columns `signal_date` and `pnl_pct`. Some
existing sanity scripts emit different column names (e.g., `trade_date`,
`T0_signal_date`, raw rupee `net_pnl` without `pnl_pct`). This script
normalizes per-setup.

Usage:
    .venv/Scripts/python tools/methodology/_combine_trades.py \\
        --setup pre_results_t1_fade \\
        --inputs reports/sub9_sanity/_pre_results_t1_v2_trades_discovery.csv \\
                 reports/sub9_sanity/_pre_results_t1_v2_trades_oos.csv \\
                 reports/sub9_sanity/_pre_results_t1_v2_trades_holdout.csv \\
        --output reports/sub9_sanity/_walkfwd_combined_pre_results_t1_fade.csv \\
        --side SHORT

For setups that emit pnl_pct directly, --side is ignored. For setups
that emit only entry_price + exit_price, --side is required to compute
pnl_pct correctly (LONG: exit-entry; SHORT: entry-exit).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


_SIGNAL_DATE_CANDIDATES = ("signal_date", "trade_date", "T0_signal_date", "entry_date")


def _normalize(df: pd.DataFrame, side: str | None) -> pd.DataFrame:
    """Normalize a single setup's trades DataFrame to (signal_date, symbol, pnl_pct)."""
    # signal_date
    sd_col = next((c for c in _SIGNAL_DATE_CANDIDATES if c in df.columns), None)
    if sd_col is None:
        raise ValueError(
            f"No signal_date column found; tried {_SIGNAL_DATE_CANDIDATES}. "
            f"Columns: {list(df.columns)}"
        )
    df = df.copy()
    df["signal_date"] = pd.to_datetime(df[sd_col]).dt.date

    # symbol
    if "symbol" not in df.columns:
        if "bare_symbol" in df.columns:
            df["symbol"] = df["bare_symbol"]
        else:
            df["symbol"] = "UNKNOWN"

    # pnl_pct
    if "pnl_pct" not in df.columns:
        if "entry_price" in df.columns and "exit_price" in df.columns:
            if side is None:
                raise ValueError(
                    "pnl_pct not in columns and --side not provided. "
                    "Pass --side LONG or --side SHORT to compute from entry/exit."
                )
            entry = df["entry_price"].astype(float)
            exit_ = df["exit_price"].astype(float)
            if side.upper() == "LONG":
                df["pnl_pct"] = (exit_ - entry) / entry * 100.0
            else:
                df["pnl_pct"] = (entry - exit_) / entry * 100.0
        else:
            raise ValueError(
                "Cannot compute pnl_pct: missing pnl_pct + (entry_price, exit_price). "
                f"Columns: {list(df.columns)}"
            )

    return df[["signal_date", "symbol", "pnl_pct"]]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--setup", required=True)
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--side", choices=["LONG", "SHORT", "long", "short"], default=None,
        help="Required if pnl_pct not in CSVs (computed from entry/exit_price).",
    )
    args = p.parse_args(argv)

    frames = []
    for f in args.inputs:
        if not f.exists():
            print(f"WARNING: input not found, skipping: {f}", file=sys.stderr)
            continue
        df = pd.read_csv(f)
        norm = _normalize(df, args.side)
        frames.append(norm)
        print(f"  loaded {len(norm)} from {f.name}")

    if not frames:
        print("ERROR: no input files loaded", file=sys.stderr)
        return 2

    out = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nwrote {len(out)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
