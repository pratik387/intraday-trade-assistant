"""Build IV-rank time series from atm_iv (per symbol, rolling 252-day min-max).

Usage:
    python tools/option_chain/build_iv_rank.py --in data/options_iv/2023_2024_iv_timeseries.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--lookback", type=int, default=252)
    args = p.parse_args()

    df = pd.read_parquet(args.in_path)
    print(f"input: {len(df):,} rows, {df['symbol'].nunique()} symbols")

    df = df.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    LB = args.lookback

    # Rolling per-symbol min/max with min_periods to allow earlier ranks once enough history exists
    df["iv_rank_252d_min"] = df.groupby("symbol")["atm_iv"].transform(
        lambda s: s.rolling(LB, min_periods=max(20, LB // 4)).min()
    )
    df["iv_rank_252d_max"] = df.groupby("symbol")["atm_iv"].transform(
        lambda s: s.rolling(LB, min_periods=max(20, LB // 4)).max()
    )
    rng = df["iv_rank_252d_max"] - df["iv_rank_252d_min"]
    df["iv_rank"] = (df["atm_iv"] - df["iv_rank_252d_min"]) / rng.replace(0, pd.NA)
    df["iv_rank"] = df["iv_rank"].clip(0.0, 1.0)

    out_path = args.in_path.parent / args.in_path.name.replace("iv_timeseries", "iv_rank")
    out_df = df[["session_date", "symbol", "atm_iv", "iv_rank",
                 "iv_rank_252d_min", "iv_rank_252d_max"]]
    out_df.to_parquet(out_path)

    print(f"wrote: {out_path}")
    print(f"  rows with iv_rank computed: {out_df['iv_rank'].notna().sum():,}")
    cov = out_df["iv_rank"].notna().sum() / len(out_df)
    print(f"  coverage: {100*cov:.1f}%")
    high = (out_df["iv_rank"] >= 0.80).sum()
    low = (out_df["iv_rank"] <= 0.20).sum()
    mid = ((out_df["iv_rank"] > 0.20) & (out_df["iv_rank"] < 0.80)).sum()
    print(f"  IV-rank distribution: high (>=0.80) {high:,} | mid {mid:,} | low (<=0.20) {low:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
