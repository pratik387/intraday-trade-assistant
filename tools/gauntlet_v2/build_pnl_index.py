"""Build pnl_by_admit.parquet from a directory of per-session trade_report.csv files.

Sub-project #5 Step 2. One-time ETL after the OCI wide-open capture.

Expected input tree:
    <oci-dir>/
      2023-01-02/trade_report.csv
      2023-01-03/trade_report.csv
      ...

Output: parquet keyed by (session_date, ts, symbol, setup_type).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


KEY_COLS = ("session_date", "ts", "symbol", "setup_type")
OUT_COLS = ("session_date", "ts", "symbol", "setup_type",
            "total_trade_pnl", "r_multiple", "gross_exit_qty")


def build(oci_dir: Path) -> pd.DataFrame:
    rows = []
    for sess_dir in sorted(oci_dir.iterdir()):
        if not sess_dir.is_dir():
            continue
        tr = sess_dir / "trade_report.csv"
        if not tr.exists():
            continue
        sess_date = sess_dir.name  # "YYYY-MM-DD" by convention
        df = pd.read_csv(tr)
        # trade_report.csv schemas vary across runs; reduce to canonical cols
        if "realized_pnl" in df.columns:
            pnl_col = "realized_pnl"
        elif "total_trade_pnl" in df.columns:
            pnl_col = "total_trade_pnl"
        else:
            pnl_col = None
        # Normalize decision_ts to ISO 8601 with 'T' separator so the join key
        # matches gate_input.jsonl (which uses pd.Timestamp.isoformat()). Live's
        # trade_report.csv writes "YYYY-MM-DD HH:MM:SS" (space) by default.
        ts_normalized = df["decision_ts"].astype(str).str.replace(" ", "T", regex=False)
        out = pd.DataFrame({
            "session_date": sess_date,
            "ts": ts_normalized,
            "symbol": df["symbol"].astype(str),
            "setup_type": df["setup_type"].astype(str),
            "total_trade_pnl": df[pnl_col].astype(float) if pnl_col else 0.0,
            "r_multiple": df.get("r_multiple", pd.Series([0.0]*len(df))).astype(float),
            "gross_exit_qty": df.get("gross_exit_qty", pd.Series([0]*len(df))).astype(int),
        })
        rows.append(out)
    if not rows:
        raise SystemExit(f"[build_pnl_index] no trade_report.csv found under {oci_dir}")
    big = pd.concat(rows, ignore_index=True)

    # Duplicate-key handling: wide_open_mode OCI captures bypass dedup, so a
    # single bar can contain multiple trades on the same (sym, ts, setup) —
    # each a separate trade_id. Any dedup-enabled config being A/B tested
    # would admit only the FIRST of those. Keep the first-seen row per key
    # so PnL joins are unambiguous. Extra same-key admits from looser
    # configs silently won't match (a minor bias for very loose configs).
    dup_mask = big.duplicated(subset=list(KEY_COLS), keep=False)
    ndup = int(dup_mask.sum())
    if ndup > 0:
        n_before = len(big)
        big = big.drop_duplicates(subset=list(KEY_COLS), keep="first").reset_index(drop=True)
        n_after = len(big)
        print(f"[build_pnl_index] deduplicated {n_before - n_after} duplicate-key rows "
              f"({ndup} total matched duplicates); kept first of each key")
    return big[list(OUT_COLS)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oci-dir", required=True, help="Path to OCI run dir (session subfolders with trade_report.csv each)")
    p.add_argument("--output", required=True, help="Output parquet path")
    args = p.parse_args()

    df = build(Path(args.oci_dir))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[build_pnl_index] wrote {len(df)} admit rows to {out}")


if __name__ == "__main__":
    main()
