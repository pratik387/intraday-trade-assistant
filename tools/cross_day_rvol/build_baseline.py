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
    vol_mean20  float32 (mean of prior 20 same-tod bars in same symbol)

Storage estimate: 2,000 symbols × 500 days × 75 bars (09:15-15:25 window)
= ~75M rows × ~16 bytes ≈ 1.2GB. Full-session build (extended 2026-05-21
for below_vwap_volume_revert_long afternoon cell).

# Memory strategy

85M rows of (symbol, session_date, hhmm, volume) does not fit a single
sort + rolling-transform operation. Process in SYMBOL CHUNKS — load all
years for ~200 symbols at a time, compute rolling baseline, accumulate.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MONTHLY_DIR = REPO / "backtest-cache-download" / "monthly"
OUT_DIR = REPO / "data" / "cross_day_rvol"
OUT_FP = OUT_DIR / "rvol_baseline.parquet"

# Active window: 09:15-15:25 (full session). Original morning-only build was
# 09:30-11:00 — extended 2026-05-21 to cover below_vwap_volume_revert_long
# (afternoon-cell setup at 13:00-14:55). See specs/2026-05-21-below_vwap_volume_revert_long-paper-trade-spec.md.
HHMM_MIN = 915
HHMM_MAX = 1525
ROLLING_DAYS = 20

# Symbols per chunk. With 75 hhmms/day × 500 days × 200 symbols ≈ 7.5M rows
# per chunk — well within memory limits, even with the sort/transform copy.
SYMBOL_CHUNK = 200


def _load_active_window(start_load: pd.Timestamp, end_load: pd.Timestamp) -> pd.DataFrame:
    """Load all monthly feathers in [start_load, end_load], filtered to active hhmm window."""
    months = pd.date_range(start_load.replace(day=1), end_load, freq="MS")
    files = [MONTHLY_DIR / f"{m.year}_{m.month:02d}_5m_enriched.feather" for m in months]
    files = [fp for fp in files if fp.exists()]
    print(f"Loading {len(files)} monthly feathers ({start_load.date()} to {end_load.date()})...",
          flush=True)
    parts = []
    for fp in files:
        df = pd.read_feather(fp, columns=["date", "symbol", "volume"])
        df["ts"] = pd.to_datetime(df["date"])
        df["session_date"] = df["ts"].dt.date
        df["hhmm"] = (df["ts"].dt.hour * 100 + df["ts"].dt.minute).astype("int16")
        df = df[(df["hhmm"] >= HHMM_MIN) & (df["hhmm"] <= HHMM_MAX)]
        parts.append(df[["symbol", "session_date", "hhmm", "volume"]])
    big = pd.concat(parts, ignore_index=True)
    return big


def _process_symbol_chunk(sub: pd.DataFrame, start_keep, end_keep) -> pd.DataFrame:
    """Sort + rolling-baseline + filter for one symbol chunk."""
    sub = sub.sort_values(["symbol", "hhmm", "session_date"]).reset_index(drop=True)
    sub["vol_mean20"] = sub.groupby(["symbol", "hhmm"], observed=True)["volume"].transform(
        lambda s: s.shift(1).rolling(ROLLING_DAYS, min_periods=5).mean()
    )
    sub = sub.dropna(subset=["vol_mean20"])
    sub = sub[
        (sub["session_date"] >= start_keep) & (sub["session_date"] <= end_keep)
    ]
    sub["vol_mean20"] = sub["vol_mean20"].astype("float32")
    return sub[["symbol", "session_date", "hhmm", "vol_mean20"]].rename(
        columns={"session_date": "date"}
    )


def main(start: str, end: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_keep = pd.Timestamp(start).date()
    end_keep = pd.Timestamp(end).date()
    start_load = pd.Timestamp(start) - pd.Timedelta(days=40)
    end_load = pd.Timestamp(end)

    big = _load_active_window(start_load, end_load)
    print(f"Loaded {len(big):,} total rows in active window", flush=True)

    symbols: List[str] = sorted(big["symbol"].unique().tolist())
    print(f"Universe: {len(symbols):,} symbols. Processing in chunks of {SYMBOL_CHUNK}.",
          flush=True)

    out_parts: List[pd.DataFrame] = []
    t0 = time.time()
    for ci in range(0, len(symbols), SYMBOL_CHUNK):
        chunk_syms = set(symbols[ci:ci + SYMBOL_CHUNK])
        sub = big[big["symbol"].isin(chunk_syms)].copy()
        if sub.empty:
            continue
        processed = _process_symbol_chunk(sub, start_keep, end_keep)
        out_parts.append(processed)
        del sub, processed
        gc.collect()
        done = min(ci + SYMBOL_CHUNK, len(symbols))
        print(f"  [{done}/{len(symbols)}] elapsed {time.time()-t0:.0f}s, "
              f"accumulated rows {sum(len(p) for p in out_parts):,}", flush=True)

    final = pd.concat(out_parts, ignore_index=True)
    print(f"Rows with valid baseline (after date trim): {len(final):,}", flush=True)

    final.to_parquet(OUT_FP, compression="zstd", index=False)
    size_mb = OUT_FP.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_FP} ({size_mb:.1f} MB, {len(final):,} rows)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: build_baseline.py START END  (YYYY-MM-DD)")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
