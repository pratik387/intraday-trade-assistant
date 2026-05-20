"""Phase 2 v2: attention-crowdedness with baseline-controlled forward returns.

v1 showed mild directional pattern (UP events fade -0.015%, DOWN events bounce
+0.04%). But absolute magnitudes are below fee floor. v2 question: is the
attention-event signal DISTINCT from the generic intraday mean-reversion that
exists in any large bar (regardless of volume)?

Test design:
  - Event cohort: bars with vol_ratio >= threshold (3, 5, 7, 10)
  - Baseline cohort: bars with vol_ratio in [0.8, 1.5] (NORMAL volume)
  - Bin both cohorts by bar_return_pct decile (within direction)
  - At each (direction x decile), compute event_mean_fwd30 - baseline_mean_fwd30
  - That delta is the attention-specific signal, controlling for movement size

Tradeability gate: delta_fwd30 >= 0.15% on at least one cell with n >= 1000.
At 5x MIS leverage = 0.75%, minus 0.30% fees = +0.45% net. Worth pursuing.
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
MIN_DAILY_AVG_VOLUME = 50_000

# Event thresholds to test
VOL_RATIO_THRESHOLDS = [3.0, 5.0, 7.0, 10.0]
# Baseline (non-event) cohort
BASELINE_VOL_RATIO_MIN = 0.8
BASELINE_VOL_RATIO_MAX = 1.5

MIN_TOTAL_VOLUME = 1000
FWD_HORIZONS_MIN = [15, 30, 60]
BASELINE_SAMPLE_SIZE = 1_000_000  # cap on non-event sample for efficiency
SAMPLE_SEED = 20260520


def _months_in_window() -> List[Tuple[int, int]]:
    out = []
    y, m = WINDOW_START.year, WINDOW_START.month
    while (y, m) <= (WINDOW_END.year, WINDOW_END.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _load_data() -> pd.DataFrame:
    chunks = []
    for (yyyy, mm) in _months_in_window():
        p = _MONTHLY_DIR / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
        if not p.exists():
            continue
        df = pd.read_feather(p, columns=["date", "symbol", "open", "close", "volume"])
        # Downcast numeric cols early to halve memory
        df["open"] = df["open"].astype("float32")
        df["close"] = df["close"].astype("float32")
        df["volume"] = df["volume"].astype("float32")
        chunks.append(df)
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    # Categorize symbol to save memory (small dictionary lookup vs string per row)
    df["symbol"] = df["symbol"].astype("category")
    # Drop the full 'date' col now that we have d + hhmm
    df = df.drop(columns=["date"])
    return df


def _filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    days = df.groupby("symbol")["d"].nunique()
    keep_days = days[days >= MIN_TRADING_DAYS_PER_SYMBOL].index
    daily_vol = df.groupby(["symbol", "d"])["volume"].sum().groupby(level=0).mean()
    keep_vol = daily_vol[daily_vol >= MIN_DAILY_AVG_VOLUME].index
    keep = set(keep_days) & set(keep_vol)
    return df[df["symbol"].isin(keep)].reset_index(drop=True)


def _add_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "hhmm", "d"])
    df["vol_mean20"] = (
        df.groupby(["symbol", "hhmm"], observed=True)["volume"]
        .transform(lambda s: s.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS).mean().shift(1))
        .astype("float32")
    )
    df["vol_ratio"] = (df["volume"] / df["vol_mean20"]).astype("float32")
    df["bar_return_pct"] = ((df["close"] - df["open"]) / df["open"] * 100.0).astype("float32")
    # Drop vol_mean20 (no longer needed once vol_ratio computed)
    df = df.drop(columns=["vol_mean20"])
    return df


def _hhmm_add(hhmm: str, minutes: int) -> str:
    h, m = hhmm.split(":")
    total = int(h) * 60 + int(m) + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def _merge_fwd_returns(events: pd.DataFrame, all_df: pd.DataFrame,
                        horizons: List[int]) -> pd.DataFrame:
    """Add fwd_<h>m_pct cols via merge for each horizon."""
    out = events.copy()
    slim = all_df[["symbol", "d", "hhmm", "close"]].rename(columns={"close": "fwd_close"})
    for h in horizons:
        target_hhmm = out["hhmm"].apply(lambda x: _hhmm_add(x, h))
        merge_key = pd.DataFrame({
            "symbol": out["symbol"].values,
            "d": out["d"].values,
            "hhmm": target_hhmm.values,
            "_idx": np.arange(len(out)),
        })
        merged = merge_key.merge(slim, on=["symbol", "d", "hhmm"], how="left")
        merged = merged.sort_values("_idx").reset_index(drop=True)
        fwd_close = merged["fwd_close"].values
        signal_close = out["close"].values
        col = f"fwd_{h}m_pct"
        out[col] = np.where(
            (fwd_close > 0) & np.isfinite(fwd_close) & (signal_close > 0),
            (fwd_close - signal_close) / signal_close * 100.0,
            np.nan,
        )
    return out


def main() -> int:
    print("=" * 95)
    print("Phase 2 v2: attention-event vs baseline (controlled for bar_return magnitude)")
    print("=" * 95)
    print(f"  Window: {WINDOW_START} to {WINDOW_END}")
    print(f"  Event thresholds: {VOL_RATIO_THRESHOLDS}")
    print(f"  Baseline vol_ratio range: [{BASELINE_VOL_RATIO_MIN}, {BASELINE_VOL_RATIO_MAX}]")
    print()

    print("Loading data...", flush=True)
    df = _load_data()
    print(f"  {len(df):,} bars / {df['symbol'].nunique()} symbols", flush=True)
    df = _filter_universe(df)
    print(f"  After universe filter: {len(df):,} bars / {df['symbol'].nunique()} symbols", flush=True)
    df = _add_baseline(df)
    # Filter to valid rows in-place; do NOT reset_index (that triggers a deep copy)
    df = df[df["vol_ratio"].notna() & df["bar_return_pct"].notna()]
    print(f"  After dropping NaN baselines: {len(df):,} bars", flush=True)

    # Identify event + baseline cohorts
    event_masks = {
        f"vr>={t:.1f}": (df["vol_ratio"] >= t) & (df["volume"] >= MIN_TOTAL_VOLUME)
        for t in VOL_RATIO_THRESHOLDS
    }
    baseline_mask = (
        (df["vol_ratio"] >= BASELINE_VOL_RATIO_MIN) &
        (df["vol_ratio"] <= BASELINE_VOL_RATIO_MAX) &
        (df["volume"] >= MIN_TOTAL_VOLUME)
    )

    # Sample baseline if too large
    cols_needed = ["symbol", "d", "hhmm", "open", "close", "vol_ratio", "bar_return_pct"]
    baseline_pool = df.loc[baseline_mask, cols_needed]
    print(f"  Baseline pool size: {len(baseline_pool):,} bars", flush=True)
    if len(baseline_pool) > BASELINE_SAMPLE_SIZE:
        baseline_sample = baseline_pool.sample(BASELINE_SAMPLE_SIZE, random_state=SAMPLE_SEED)
    else:
        baseline_sample = baseline_pool
    print(f"  Baseline sample size: {len(baseline_sample):,} bars", flush=True)

    print("\nComputing forward returns for baseline...", flush=True)
    baseline_with_fwd = _merge_fwd_returns(baseline_sample, df, FWD_HORIZONS_MIN)

    # Bin bar_return into deciles (within direction)
    print("\nBinning by bar_return deciles...", flush=True)
    # Use only baseline to compute decile edges (so they're not biased by events)
    up_edges = pd.qcut(
        baseline_with_fwd.loc[baseline_with_fwd["bar_return_pct"] > 0, "bar_return_pct"],
        q=5, retbins=True, duplicates="drop",
    )[1]
    dn_edges = pd.qcut(
        baseline_with_fwd.loc[baseline_with_fwd["bar_return_pct"] < 0, "bar_return_pct"],
        q=5, retbins=True, duplicates="drop",
    )[1]
    print(f"  UP quintile edges: {[f'{x:.3f}' for x in up_edges]}")
    print(f"  DN quintile edges: {[f'{x:.3f}' for x in dn_edges]}")

    def _bin(row):
        ret = row["bar_return_pct"]
        if ret > 0:
            for i in range(len(up_edges) - 1):
                if up_edges[i] <= ret <= up_edges[i + 1]:
                    return f"UP_Q{i+1}"
            return f"UP_Q{len(up_edges)-1}"
        elif ret < 0:
            for i in range(len(dn_edges) - 1):
                if dn_edges[i] <= ret <= dn_edges[i + 1]:
                    return f"DN_Q{i+1}"
            return f"DN_Q{len(dn_edges)-1}"
        return "ZERO"

    print("\nBinning baseline...", flush=True)
    baseline_with_fwd["bucket"] = baseline_with_fwd.apply(_bin, axis=1)
    baseline_stats = baseline_with_fwd.groupby("bucket")[
        [f"fwd_{h}m_pct" for h in FWD_HORIZONS_MIN]
    ].mean()
    baseline_n = baseline_with_fwd.groupby("bucket").size()

    print("\nBaseline per-bucket means:")
    print(f"  {'bucket':<8} {'n':>8} {'fwd_15m':>9} {'fwd_30m':>9} {'fwd_60m':>9}")
    for bucket in baseline_stats.index:
        n = baseline_n[bucket]
        row = baseline_stats.loc[bucket]
        print(f"  {bucket:<8} {n:>8,} {row['fwd_15m_pct']:>+9.4f} {row['fwd_30m_pct']:>+9.4f} "
              f"{row['fwd_60m_pct']:>+9.4f}")

    # Now for each event threshold, compute event mean per bucket + delta vs baseline
    all_results = []
    for thresh_label, mask in event_masks.items():
        events = df.loc[mask, cols_needed]
        if events.empty:
            continue
        print(f"\n[{thresh_label}] {len(events):,} events", flush=True)
        events_with_fwd = _merge_fwd_returns(events, df, FWD_HORIZONS_MIN)
        events_with_fwd["bucket"] = events_with_fwd.apply(_bin, axis=1)
        event_stats = events_with_fwd.groupby("bucket")[
            [f"fwd_{h}m_pct" for h in FWD_HORIZONS_MIN]
        ].mean()
        event_n = events_with_fwd.groupby("bucket").size()

        print(f"  {'bucket':<8} {'n_evt':>7} {'evt_30m':>9} {'base_30m':>9} {'delta_30m':>10} "
              f"{'delta_15m':>10} {'delta_60m':>10}")
        for bucket in sorted(set(event_stats.index) & set(baseline_stats.index)):
            if event_n[bucket] < 100:
                continue
            evt = event_stats.loc[bucket]
            base = baseline_stats.loc[bucket]
            d15 = evt["fwd_15m_pct"] - base["fwd_15m_pct"]
            d30 = evt["fwd_30m_pct"] - base["fwd_30m_pct"]
            d60 = evt["fwd_60m_pct"] - base["fwd_60m_pct"]
            print(f"  {bucket:<8} {event_n[bucket]:>7,} {evt['fwd_30m_pct']:>+9.4f} "
                  f"{base['fwd_30m_pct']:>+9.4f} {d30:>+10.4f} {d15:>+10.4f} {d60:>+10.4f}")
            all_results.append({
                "thresh": thresh_label, "bucket": bucket, "n_evt": int(event_n[bucket]),
                "evt_15m": evt["fwd_15m_pct"], "evt_30m": evt["fwd_30m_pct"], "evt_60m": evt["fwd_60m_pct"],
                "base_15m": base["fwd_15m_pct"], "base_30m": base["fwd_30m_pct"], "base_60m": base["fwd_60m_pct"],
                "delta_15m": d15, "delta_30m": d30, "delta_60m": d60,
            })

    # Save + verdict
    if all_results:
        out_df = pd.DataFrame(all_results)
        out_path = _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_attention_v2_baseline.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path}")

        # Find best cells (largest |delta_30m|)
        print("\n" + "=" * 95)
        print("TOP 10 CELLS BY |delta_30m| (event - baseline; positive = trade direction works)")
        print("=" * 95)
        out_df["abs_delta_30m"] = out_df["delta_30m"].abs()
        top = out_df.sort_values("abs_delta_30m", ascending=False).head(10)
        print(f"  {'thresh':<10} {'bucket':<8} {'n_evt':>7} {'delta_30m':>10} {'delta_60m':>10}")
        for _, r in top.iterrows():
            print(f"  {r['thresh']:<10} {r['bucket']:<8} {r['n_evt']:>7,} "
                  f"{r['delta_30m']:>+10.4f} {r['delta_60m']:>+10.4f}")

        # Verdict
        max_abs = top["abs_delta_30m"].max()
        print(f"\n  Strongest cell: |delta_30m| = {max_abs:.4f}%")
        if max_abs >= 0.15:
            print("  -> PROCEED: at least one cell shows tradeable delta vs baseline")
        elif max_abs >= 0.10:
            print("  -> BORDERLINE: weak signal; would need leverage > 5x or fee reduction")
        else:
            print("  -> KILL: no cell shows delta >= 0.10% vs baseline at any threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
