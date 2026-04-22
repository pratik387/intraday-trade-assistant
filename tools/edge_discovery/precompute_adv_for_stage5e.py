"""Precompute 20-day rolling ADV_rupees per (symbol, date) for Stage 5e budgeted selector.

Reads monthly 5m-enriched feathers, aggregates to per-day turnover (sum of
volume * close across all 5m bars in a day), computes 20-day rolling average
per symbol, saves as parquet for Stage 5e to join on (symbol, session_date).

Output: models/gauntlet/stage5e_adv_rupees.parquet
Columns: symbol, date, adv_rupees_20d
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
MONTHLY_DIR = ROOT / "backtest-cache-download" / "monthly"
OUT_DIR = ROOT / "models" / "gauntlet"
OUT_PARQUET = OUT_DIR / "stage5e_adv_rupees.parquet"


def main():
    files = sorted(MONTHLY_DIR.glob("*_5m_enriched.feather"))
    if not files:
        raise FileNotFoundError(f"No monthly feathers in {MONTHLY_DIR}")
    print(f"Found {len(files)} monthly feathers")

    parts = []
    for f in files:
        df = pd.read_feather(f, columns=["date", "symbol", "volume", "close"])
        df["date_only"] = df["date"].dt.date if df["date"].dt.tz is None else df["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.date
        df["rupee_vol"] = df["volume"].astype("float64") * df["close"].astype("float64")
        parts.append(df[["symbol", "date_only", "rupee_vol"]])
        print(f"  {f.name}: {len(df):,} rows")

    print("Concat + daily aggregate...")
    all_bars = pd.concat(parts, ignore_index=True)
    del parts
    daily = all_bars.groupby(["symbol", "date_only"], as_index=False)["rupee_vol"].sum()
    daily = daily.rename(columns={"rupee_vol": "daily_turnover_rupees"})
    daily = daily.sort_values(["symbol", "date_only"]).reset_index(drop=True)
    print(f"Daily rows: {len(daily):,}")

    print("Rolling 20-day ADV per symbol...")
    daily["adv_rupees_20d"] = (
        daily.groupby("symbol")["daily_turnover_rupees"]
        .rolling(window=20, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # Rows with <5 prior sessions get NaN — that's acceptable; stage5e will treat NaN as "unknown ADV, skip"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily[["symbol", "date_only", "adv_rupees_20d"]].to_parquet(OUT_PARQUET, index=False)
    print(f"Saved: {OUT_PARQUET}")
    print(f"  Rows: {len(daily):,}")
    print(f"  Unique symbols: {daily['symbol'].nunique():,}")
    print(f"  Date range: {daily['date_only'].min()} to {daily['date_only'].max()}")
    print(f"  ADV_rupees percentiles (non-NaN):")
    adv = daily["adv_rupees_20d"].dropna()
    for p in [10, 25, 50, 75, 90, 99]:
        print(f"    {p}%: Rs {adv.quantile(p/100):,.0f}")


if __name__ == "__main__":
    main()
