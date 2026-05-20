"""Phase 2 attention-crowdedness signature test.

Hypothesis: when a stock has a sudden intraday volume burst (vol_ratio >>
its 20-day baseline at the same intraday minute), it appears on NSE's
published "most active" list, attracting retail attention. The crowd-in
overshoots within 30-60 min, then exhausts. Trade the exhaustion (fade).

Design notes:
- We do NOT have NSE's historical intraday top-10 archives (NSE doesn't
  publish them historically).
- We do have a partial 20-day rolling baseline at
  data/cross_day_rvol/rvol_baseline.parquet but it covers ONLY 09:30-11:00.
- Instead, we compute the full-day baseline INLINE from the 5m feathers.
  Per (symbol, hhmm): rolling-20d mean of volume.

Phase 2 measures:
  1. ATTENTION EVENT = first 5m bar in a session where vol_ratio >=
     THRESHOLD (e.g., 3.0) for that (symbol, hhmm).
  2. Forward returns 5m/15m/30m/60m after attention event.
  3. Direction conditioning: UP-direction event (bar_return > 0) vs DOWN.
  4. Baseline: random bars (non-event) on the SAME symbol/day for fair
     comparison. If forward returns post-event differ materially from
     non-event baseline, there's a signal.

Universe: stocks with sufficient data (>= 200 trading days in window),
which naturally selects liquid F&O-eligible + mid/large-cap names.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"

WINDOW_START = date(2023, 1, 2)
WINDOW_END = date(2024, 12, 31)
ROLLING_DAYS = 20
MIN_TRADING_DAYS_PER_SYMBOL = 200
VOL_RATIO_THRESHOLD = 3.0      # event threshold
MIN_TOTAL_VOLUME = 1000        # exclude near-no-trade bars
MIN_DAILY_AVG_VOLUME = 50_000  # rough liquidity filter
FWD_HORIZONS_MIN = [5, 15, 30, 60]


def _months_in_window() -> List[Tuple[int, int]]:
    out = []
    y, m = WINDOW_START.year, WINDOW_START.month
    while (y, m) <= (WINDOW_END.year, WINDOW_END.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _load_all_data() -> pd.DataFrame:
    """Load 5m feathers for the Discovery window, return concatenated df."""
    chunks = []
    months = _months_in_window()
    print(f"Loading {len(months)} months of 5m data...", flush=True)
    for (yyyy, mm) in months:
        p = _MONTHLY_DIR / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
        if not p.exists():
            print(f"  [warn] missing {p.name}", flush=True)
            continue
        df = pd.read_feather(p, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        chunks.append(df)
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    print(f"  Loaded {len(df):,} bars across {df['symbol'].nunique()} symbols", flush=True)
    return df


def _filter_to_liquid_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to symbols with enough trading days + reasonable daily liquidity."""
    print("Filtering to liquid universe...", flush=True)
    # Trading days per symbol
    days_per_sym = df.groupby("symbol")["d"].nunique()
    keep_days = days_per_sym[days_per_sym >= MIN_TRADING_DAYS_PER_SYMBOL].index
    # Daily avg volume per symbol
    daily_vol = df.groupby(["symbol", "d"])["volume"].sum().groupby(level=0).mean()
    keep_vol = daily_vol[daily_vol >= MIN_DAILY_AVG_VOLUME].index
    keep = set(keep_days) & set(keep_vol)
    out = df[df["symbol"].isin(keep)].reset_index(drop=True)
    print(f"  {len(keep):,} symbols pass filter ({len(out):,} bars)", flush=True)
    return out


def _compute_rvol_baseline_and_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, hhmm), compute rolling-20d mean of volume and per-bar vol_ratio.

    Adds columns: vol_mean20, vol_ratio.
    """
    print("Computing rolling-20d vol baseline + per-bar vol_ratio...", flush=True)
    df = df.sort_values(["symbol", "hhmm", "d"]).reset_index(drop=True)
    # Rolling mean per (symbol, hhmm), shifted to exclude today
    df["vol_mean20"] = (
        df.groupby(["symbol", "hhmm"])["volume"]
        .transform(lambda s: s.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS).mean().shift(1))
    )
    df["vol_ratio"] = df["volume"] / df["vol_mean20"]
    n_valid = df["vol_ratio"].notna().sum()
    print(f"  {n_valid:,} bars have valid vol_ratio", flush=True)
    return df


def _identify_attention_events(df: pd.DataFrame) -> pd.DataFrame:
    """Find FIRST bar per (symbol, date) where vol_ratio >= threshold.

    Memory-efficient: filter via boolean mask, no .copy() of 40M-row df.
    """
    print(f"Identifying attention events (vol_ratio >= {VOL_RATIO_THRESHOLD})...", flush=True)
    mask = (
        df["vol_ratio"].notna() &
        (df["vol_ratio"] >= VOL_RATIO_THRESHOLD) &
        (df["volume"] >= MIN_TOTAL_VOLUME)
    )
    # .loc with a boolean mask + column list returns a much smaller df
    cols_needed = ["symbol", "d", "date", "hhmm", "open", "close", "volume", "vol_ratio"]
    events = df.loc[mask, cols_needed].sort_values(["symbol", "d", "date"])
    events = events.drop_duplicates(["symbol", "d"], keep="first").reset_index(drop=True)
    print(f"  {len(events):,} attention events identified", flush=True)
    return events


def _hhmm_add(hhmm: str, minutes: int) -> str:
    h, m = hhmm.split(":")
    total = int(h) * 60 + int(m) + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def _measure_forward_returns(events: pd.DataFrame, all_df: pd.DataFrame) -> pd.DataFrame:
    """For each event, compute fwd_return at each horizon.

    Memory-efficient: instead of dict lookup over 40M rows, merge events
    against a slim (symbol, d, hhmm, close) view per horizon.
    """
    print("Measuring forward returns...", flush=True)
    out = events.copy()
    out["signal_close"] = out["close"]
    out["bar_return_pct"] = (out["close"] - out["open"]) / out["open"] * 100.0
    out["bar_direction"] = np.where(out["close"] > out["open"], "UP", "DOWN")

    slim = all_df[["symbol", "d", "hhmm", "close"]]

    for h_min in FWD_HORIZONS_MIN:
        target_hhmm = out["hhmm"].apply(lambda x: _hhmm_add(x, h_min))
        merge_key = pd.DataFrame({
            "symbol": out["symbol"].values,
            "d": out["d"].values,
            "hhmm": target_hhmm.values,
            "_row_idx": np.arange(len(out)),
        })
        merged = merge_key.merge(slim, on=["symbol", "d", "hhmm"], how="left")
        merged = merged.sort_values("_row_idx").reset_index(drop=True)
        fwd_close = merged["close"].values
        col = f"fwd_{h_min}m_pct"
        out[col] = np.where(
            (fwd_close > 0) & np.isfinite(fwd_close),
            (fwd_close - out["close"].values) / out["close"].values * 100.0,
            np.nan,
        )

    # Drop the original close column (we have signal_close)
    out = out.drop(columns=["close"])
    n_valid = out["fwd_30m_pct"].notna().sum()
    print(f"  {len(out):,} events with forward returns measured ({n_valid:,} have valid fwd_30m)",
          flush=True)
    return out


def _summarize(events: pd.DataFrame) -> None:
    """Print summary stats: distribution of forward returns by direction + by hour."""
    print()
    print("=" * 90)
    print(f"ATTENTION EVENT FORWARD-RETURN SIGNATURE (vol_ratio >= {VOL_RATIO_THRESHOLD})")
    print("=" * 90)
    if events.empty:
        print("  no events"); return

    # Overall
    print(f"\nOverall (n={len(events):,}):")
    for h in FWD_HORIZONS_MIN:
        col = f"fwd_{h}m_pct"
        v = events[col].dropna()
        if not len(v):
            continue
        print(f"  fwd_{h}m: mean={v.mean():+.4f}%  median={v.median():+.4f}%  "
              f"n={len(v):,}  win_rate(>0)={100*(v>0).mean():.1f}%")

    # By bar direction
    print(f"\nBy signal-bar direction (the 'crowd-in' direction):")
    for direction in ("UP", "DOWN"):
        sub = events[events["bar_direction"] == direction]
        if not len(sub):
            continue
        print(f"  [{direction}] n={len(sub):,}")
        for h in FWD_HORIZONS_MIN:
            col = f"fwd_{h}m_pct"
            v = sub[col].dropna()
            if not len(v):
                continue
            print(f"    fwd_{h}m: mean={v.mean():+.4f}%  median={v.median():+.4f}%  "
                  f"win_rate={100*(v>0).mean():.1f}%")

    # By bar direction + vol_ratio bucket
    print(f"\nBy vol_ratio strength (top quartile = most extreme):")
    events["vr_q"] = pd.qcut(events["vol_ratio"], q=4,
                              labels=["Q1_low", "Q2", "Q3", "Q4_extreme"])
    for q in ("Q4_extreme", "Q3", "Q2", "Q1_low"):
        sub = events[events["vr_q"] == q]
        if not len(sub):
            continue
        for direction in ("UP", "DOWN"):
            sub2 = sub[sub["bar_direction"] == direction]
            if len(sub2) < 30:
                continue
            mean30 = float(sub2["fwd_30m_pct"].mean())
            mean60 = float(sub2["fwd_60m_pct"].mean())
            wr30 = 100 * float((sub2["fwd_30m_pct"] > 0).mean())
            print(f"  [{q}] [{direction}] n={len(sub2):,}  "
                  f"fwd_30m_mean={mean30:+.4f}%  fwd_60m_mean={mean60:+.4f}%  "
                  f"fwd_30m_win_rate={wr30:.1f}%")

    # By time of day
    print(f"\nBy time-of-day (when attention event fires):")
    events["hour"] = events["hhmm"].str[:2].astype(int)
    for hr in sorted(events["hour"].unique()):
        sub = events[events["hour"] == hr]
        if len(sub) < 50:
            continue
        mean30 = float(sub["fwd_30m_pct"].mean())
        mean30_up = float(sub[sub["bar_direction"] == "UP"]["fwd_30m_pct"].mean())
        mean30_dn = float(sub[sub["bar_direction"] == "DOWN"]["fwd_30m_pct"].mean())
        print(f"  {hr:02d}:xx  n={len(sub):,}  fwd_30m_mean={mean30:+.4f}%  "
              f"(UP={mean30_up:+.4f}%, DOWN={mean30_dn:+.4f}%)")


def main() -> int:
    print("=" * 90)
    print("Phase 2 signature: attention-crowdedness (vol_ratio extreme events)")
    print("=" * 90)
    print(f"  Window: {WINDOW_START} to {WINDOW_END}")
    print(f"  vol_ratio threshold: {VOL_RATIO_THRESHOLD}")
    print(f"  Rolling baseline: {ROLLING_DAYS} prior days")
    print(f"  Forward horizons: {FWD_HORIZONS_MIN} minutes")

    df = _load_all_data()
    df = _filter_to_liquid_universe(df)
    df = _compute_rvol_baseline_and_ratio(df)
    events = _identify_attention_events(df)
    results = _measure_forward_returns(events, df)

    _summarize(results)

    out_path = _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_attention_crowdedness.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nRaw events written to: {out_path}  ({len(results):,} events)")

    # Verdict — look for clear directional signature
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    if results.empty:
        print("  KILL: no events found (threshold may be too high)")
        return 1
    # Signal exists if mean fwd_30m by direction differs from zero by >= 0.1%
    up = results[results["bar_direction"] == "UP"]["fwd_30m_pct"].dropna()
    dn = results[results["bar_direction"] == "DOWN"]["fwd_30m_pct"].dropna()
    up_mean = float(up.mean()) if len(up) else 0
    dn_mean = float(dn.mean()) if len(dn) else 0
    print(f"  UP-event mean fwd_30m_pct: {up_mean:+.4f}%   (n={len(up):,})")
    print(f"  DOWN-event mean fwd_30m_pct: {dn_mean:+.4f}%   (n={len(dn):,})")
    print(f"  Spread (UP - DOWN): {up_mean - dn_mean:+.4f}%")
    if abs(up_mean) >= 0.10 or abs(dn_mean) >= 0.10:
        print("\n  -> PROCEED: at least one direction shows >= 0.10% mean fwd return — investigate.")
    elif abs(up_mean - dn_mean) >= 0.15:
        print("\n  -> PROCEED: UP-vs-DOWN spread suggests directional structure.")
    else:
        print("\n  -> KILL: no meaningful directional signature at 30m horizon.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
