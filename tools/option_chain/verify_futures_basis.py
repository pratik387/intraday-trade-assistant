"""Quick verification report for the futures-basis time-series.

Reads the latest basis parquet and prints:
  1. Total (symbol, date) row count + symbol/date coverage.
  2. Median basis_bps by days_to_expiry bucket (1-5, 6-10, 11-30) — basis
     should converge to near-zero as expiry approaches.
  3. Top-20 absolute |basis_bps| rows with days_to_expiry == 1 — for
     spot-checking data integrity (these are the most likely to flag
     ingestion bugs since T-1 basis must be ≤ a few bps under
     cash-settlement convergence).

Per specs/2026-05-07-sub-project-9-brief-stock_futures_basis_convergence.md §11.

Usage:
    python tools/option_chain/verify_futures_basis.py
    python tools/option_chain/verify_futures_basis.py --path data/futures_basis/2023_2026_basis.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DIR = _REPO_ROOT / "data" / "futures_basis"


def _resolve_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    candidates = sorted(_DEFAULT_DIR.glob("*_basis.parquet"))
    if not candidates:
        raise SystemExit(f"no basis parquet under {_DEFAULT_DIR}")
    return candidates[-1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default=None,
                   help="basis parquet path (defaults to most recent under data/futures_basis/)")
    args = p.parse_args()

    path = _resolve_path(args.path)
    print(f"reading: {path}")
    df = pd.read_parquet(path)

    n_rows = len(df)
    n_syms = df["symbol"].nunique()
    n_dates = df["session_date"].nunique()
    print()
    print("=" * 60)
    print("Futures basis time-series -- verification")
    print("=" * 60)
    print(f"  rows                  : {n_rows:,}")
    print(f"  symbols               : {n_syms}")
    print(f"  trading sessions      : {n_dates}")
    print(f"  date range            : {df['session_date'].min()} -> {df['session_date'].max()}")
    print()

    # Bucketize days_to_expiry
    def _bucket(d: int) -> str:
        if 1 <= d <= 5:
            return "1-5"
        if 6 <= d <= 10:
            return "6-10"
        if 11 <= d <= 30:
            return "11-30"
        return ">30"

    df["dte_bucket"] = df["days_to_expiry"].map(_bucket)
    by_bucket = (
        df.groupby("dte_bucket")
          .agg(n=("basis_bps", "size"),
               median_bps=("basis_bps", "median"),
               p10_bps=("basis_bps", lambda s: s.quantile(0.10)),
               p90_bps=("basis_bps", lambda s: s.quantile(0.90)),
               abs_median_bps=("basis_bps", lambda s: s.abs().median()))
          .reindex(["1-5", "6-10", "11-30", ">30"])
          .dropna(how="all")
    )
    print("Median basis_bps by days_to_expiry bucket:")
    print("(basis SHOULD converge to ~0 bps as expiry approaches)")
    print(by_bucket.to_string(float_format=lambda x: f"{x:8.2f}"))
    print()

    # Top-20 anomalous |basis_bps| at days_to_expiry == 1
    t1 = df[df["days_to_expiry"] == 1].copy()
    if t1.empty:
        print("(no rows with days_to_expiry == 1 found)")
    else:
        t1["abs_bps"] = t1["basis_bps"].abs()
        top20 = t1.sort_values("abs_bps", ascending=False).head(20)
        print("Top-20 |basis_bps| at days_to_expiry == 1 (data-integrity spot-check):")
        cols = ["session_date", "symbol", "expiry_date",
                "futures_close", "spot_close", "basis_pct", "basis_bps"]
        print(top20[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:10.2f}",
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
