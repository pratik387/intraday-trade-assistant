"""Build cross-day RVOL baseline lookup parquet.

Computes per (symbol, hhmm) the 20-prior-session rolling mean volume
keyed by (symbol, date, hhmm). Loaded by services/cross_day_rvol_enrichment.py
at screener daily-seed time so detectors can divide today's bar volume
by the historical baseline without needing 20+ days of 5m bars in df_5m.

Why this exists: structures/delivery_pct_anomaly_short_structure.py's
_cross_day_rvol() expects df_5m to span 20 prior same-tod bars. The
screener feeds df_5m as tail(20) (today's intraday only), so the
function always returned None → 0 fires in OCI run 20260508-230433_full
even though sanity confirmed 261 SHORT trades cell-locked.

Output schema:
    symbol      str
    date        date  (the T+0 trading date this baseline applies to)
    hhmm        int16 (e.g., 930 for 09:30)
    vol_mean20  float64 (mean of prior 20 same-tod bars in same symbol)

Storage estimate: 2,000 symbols × 500 days × 16 bars (09:30-11:00 window)
= 16M rows × ~16 bytes ≈ 256MB. Active-window narrow build (09:30-11:00).
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MONTHLY_DIR = REPO / "backtest-cache-download" / "monthly"
OUT_DIR = REPO / "data" / "cross_day_rvol"
OUT_FP = OUT_DIR / "rvol_baseline.parquet"

# Active window: 09:30-11:00 (covers detector window 09:30-10:00 + buffer)
HHMM_MIN = 930
HHMM_MAX = 1100
ROLLING_DAYS = 20


def main(start: str, end: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine months to load (need ROLLING_DAYS history before `start`)
    s = pd.Timestamp(start) - pd.Timedelta(days=40)
    e = pd.Timestamp(end)
    months = pd.date_range(s.replace(day=1), e, freq="MS")
    files = []
    for m in months:
        fp = MONTHLY_DIR / f"{m.year}_{m.month:02d}_5m_enriched.feather"
        if fp.exists():
            files.append(fp)
    print(f"Loading {len(files)} monthly feathers ({s.date()} to {e.date()})...")

    parts = []
    for fp in files:
        df = pd.read_feather(fp, columns=["date", "symbol", "volume"])
        df["ts"] = pd.to_datetime(df["date"])
        df["session_date"] = df["ts"].dt.date
        df["hhmm"] = df["ts"].dt.hour * 100 + df["ts"].dt.minute
        df = df[(df["hhmm"] >= HHMM_MIN) & (df["hhmm"] <= HHMM_MAX)]
        parts.append(df[["symbol", "session_date", "hhmm", "volume"]])
    big = pd.concat(parts, ignore_index=True)
    print(f"Loaded {len(big):,} rows in active window")

    # Sort by (symbol, hhmm, session_date) so groupby+rolling works correctly
    big = big.sort_values(["symbol", "hhmm", "session_date"]).reset_index(drop=True)

    # Per (symbol, hhmm): rolling 20-prior-session mean of volume
    print(f"Computing rolling-{ROLLING_DAYS} mean volume per (symbol, hhmm)...")
    t0 = time.time()
    big["vol_mean20"] = big.groupby(["symbol", "hhmm"])["volume"].transform(
        lambda s: s.shift(1).rolling(ROLLING_DAYS, min_periods=5).mean()
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # Drop rows where baseline is NaN (insufficient history)
    big = big.dropna(subset=["vol_mean20"])
    print(f"Rows with valid baseline: {len(big):,}")

    # Trim to requested date range
    big = big[
        (big["session_date"] >= pd.Timestamp(start).date())
        & (big["session_date"] <= pd.Timestamp(end).date())
    ]
    print(f"Rows after date trim: {len(big):,}")

    # Optimize dtypes
    big["hhmm"] = big["hhmm"].astype("int16")
    big["vol_mean20"] = big["vol_mean20"].astype("float32")
    big = big[["symbol", "session_date", "hhmm", "vol_mean20"]].rename(
        columns={"session_date": "date"}
    )

    big.to_parquet(OUT_FP, compression="zstd", index=False)
    size_mb = OUT_FP.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_FP} ({size_mb:.1f} MB, {len(big):,} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: build_baseline.py START END  (YYYY-MM-DD)")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
