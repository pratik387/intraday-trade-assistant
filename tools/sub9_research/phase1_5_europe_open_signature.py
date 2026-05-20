"""Phase 1.5 signature feasibility for the 'Indian-market reaction to European open'
hypothesis.

Per the MPC kill lesson (docs/setup_lifecycle.md Stage 2): before writing a
400-line brief or any sanity script, do a 15-30 min signature test on
existing 5m feathers to confirm the mechanism leaves a measurable footprint.
If no footprint -> kill candidate cheap.

# Hypothesis

The Indian market opens 09:15 IST, runs ~4 hours, THEN European markets open.
This makes India unique: it's the only major market trading BEFORE European
liquidity arrives. The cross-border information / hedging / arb flow at
European-open time should leave a signature in Europe-exposed Indian names.

European open times in IST (DST matters):
  - Winter (Oct-Mar): London 08:00 GMT = 13:30 IST, Frankfurt 09:00 = 13:30 IST
  - Summer (Apr-Sep, BST): London 08:00 BST = 12:30 IST, Frankfurt 09:00 = 12:30 IST
  - European pre-market (XETRA pre-trading): starts 07:30 local
    -> 13:00 IST (winter) / 12:00 IST (summer)

We test the 12:30 / 13:00 / 13:30 IST 5m bars (open-of-bar timestamps).

# Cohort

Europe-exposed Indian large/mid-cap:
  - IT exporters (Europe = #2 revenue source after US)
  - Banks with ADR/GDR (HDFCBANK, ICICIBANK, AXISBANK)
  - Pharma exporters (Europe regulatory exposure)
  - Auto with EU exposure (TATAMOTORS via JLR)
  - Metals pegged to LME (TATASTEEL, JSWSTEEL, etc.)

Total: ~22 names.

Baseline cohort: similar-cap Indian names with NEGLIGIBLE Europe exposure
(domestic-consumption tilt). Comparison removes the "any large-cap moves at
13:00" confound.

# Signature measured

For each (date, symbol) in cohort and baseline:
  - vol_ratio_<HHMM>      = bar.volume / prior_30d_same_HHMM_bar_mean
  - bar_return_<HHMM>     = (bar.close - bar.open) / bar.open * 100
  - fwd_return_30m        = (close_at_HHMM+30 - close_at_HHMM) / close_at_HHMM * 100
  - fwd_return_60m
  - fwd_return_90m

We compare cohort vs baseline along three axes:
  1. Volume burst: is mean(vol_ratio) materially > 1.0 at 12:30/13:00/13:30
     for Europe cohort vs baseline?
  2. Directional impulse: |bar_return| at signal bar — does cohort show
     larger moves than baseline?
  3. Forward-return signature: corr(bar_return_signal, fwd_return_30m) — is
     it positive (continuation) or negative (revert) on cohort but ~0 on
     baseline?

Per Lesson #3 Phase 2: net drift < 0.1% on cohort delta = kill the
hypothesis. Otherwise proceed.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# Europe-exposed Indian large/mid-cap (22 names)
EUROPE_COHORT = [
    # IT exporters (Europe revenue 25-40%)
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "COFORGE",
    # Banks with ADR/GDR listings
    "HDFCBANK", "ICICIBANK", "AXISBANK",
    # Pharma exporters (Europe regulatory matters: EMA, MHRA)
    "SUNPHARMA", "DRREDDY", "LUPIN", "CIPLA", "AUROPHARMA", "TORNTPHARM",
    # Auto with JLR / Europe exposure
    "TATAMOTORS",
    # Metals pegged to LME (London Metal Exchange opens 13:00 IST winter)
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL",
]

# Baseline: similar-cap Indian large/mid-cap with NEGLIGIBLE Europe revenue
# (domestic-consumption, PSU, India-focused financials)
BASELINE_COHORT = [
    # FMCG (domestic)
    "HINDUNILVR", "ITC", "DABUR", "GODREJCP", "MARICO", "NESTLEIND",
    # Domestic banks/finance (low FII exposure)
    "SBIN", "PNB", "BANKBARODA", "CANBK", "LICHSGFIN",
    # PSU / domestic-utility
    "ONGC", "COALINDIA", "POWERGRID", "NTPC", "GAIL",
    # Domestic-consumption
    "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
    "TITAN", "ASIANPAINT",
]

# Time anchors (5m bar open timestamps in IST)
TIME_ANCHORS = ["12:30", "13:00", "13:30"]

# Window
WINDOW_START = date(2023, 1, 2)
WINDOW_END = date(2024, 12, 31)

VOL_BASELINE_DAYS = 30           # rolling for prior-30d same-HHMM vol mean
MIN_OBSERVATIONS_PER_SYMBOL = 50  # need enough days to compute rolling


# ---------------------------------------------------------------------------
# Bar loading
# ---------------------------------------------------------------------------

def _months_in_window() -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    y, m = WINDOW_START.year, WINDOW_START.month
    while (y, m) <= (WINDOW_END.year, WINDOW_END.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1; y += 1
    return out


def _load_5m_for_month(yyyy: int, mm: int, cohort: List[str]) -> pd.DataFrame:
    """Load month feather, filter to cohort, index by (symbol, date_d, hhmm)."""
    p = _MONTHLY_DIR / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_feather(p, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    df = df[df["symbol"].isin(cohort)]
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    return df.reset_index(drop=True)


def _build_close_lookup(df_filtered: pd.DataFrame) -> dict:
    """Dict[(symbol, d, hhmm)] -> close for O(1) lookups across months."""
    if df_filtered.empty:
        return {}
    return dict(zip(
        zip(df_filtered["symbol"], df_filtered["d"], df_filtered["hhmm"]),
        df_filtered["close"].astype(float),
    ))


def _build_volume_lookup(df_filtered: pd.DataFrame) -> dict:
    if df_filtered.empty:
        return {}
    return dict(zip(
        zip(df_filtered["symbol"], df_filtered["d"], df_filtered["hhmm"]),
        df_filtered["volume"].astype(float),
    ))


def _build_open_lookup(df_filtered: pd.DataFrame) -> dict:
    if df_filtered.empty:
        return {}
    return dict(zip(
        zip(df_filtered["symbol"], df_filtered["d"], df_filtered["hhmm"]),
        df_filtered["open"].astype(float),
    ))


def _add_hhmm(hhmm: str, minutes: int) -> str:
    h, m = hhmm.split(":")
    total = int(h) * 60 + int(m) + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Signature measurement
# ---------------------------------------------------------------------------

def measure_cohort(cohort: List[str], label: str) -> pd.DataFrame:
    """Build (date, symbol, anchor_hhmm) measurements:
    vol_ratio, bar_return, fwd_return_30m/60m/90m.
    """
    print(f"\n[{label}] loading {len(cohort)} symbols across {len(_months_in_window())} months",
          flush=True)
    # Load all months consolidated
    chunks = []
    for (yyyy, mm) in _months_in_window():
        chunk = _load_5m_for_month(yyyy, mm, cohort)
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        print(f"[{label}] no data loaded", flush=True)
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    print(f"[{label}] loaded {len(df):,} bars total", flush=True)

    # Pre-build lookups
    close_lu = _build_close_lookup(df)
    vol_lu = _build_volume_lookup(df)
    open_lu = _build_open_lookup(df)

    # Per (symbol, hhmm), build rolling-30d vol baseline using pandas groupby
    # Easier: per symbol+hhmm, sort by date, compute rolling mean shifted by 1
    rows = []
    for sym in cohort:
        sym_df = df[df["symbol"] == sym]
        if len(sym_df) < MIN_OBSERVATIONS_PER_SYMBOL:
            continue
        for anchor in TIME_ANCHORS:
            anchor_df = sym_df[sym_df["hhmm"] == anchor].copy()
            if len(anchor_df) < VOL_BASELINE_DAYS + 5:
                continue
            anchor_df = anchor_df.sort_values("d").reset_index(drop=True)
            # Rolling-30d volume mean of PRIOR days (shift(1) excludes current bar)
            anchor_df["vol_baseline"] = (
                anchor_df["volume"].rolling(VOL_BASELINE_DAYS, min_periods=VOL_BASELINE_DAYS)
                .mean().shift(1)
            )
            anchor_df["vol_ratio"] = anchor_df["volume"] / anchor_df["vol_baseline"]
            anchor_df["bar_return_pct"] = (anchor_df["close"] - anchor_df["open"]) / anchor_df["open"] * 100.0

            # Forward returns
            for fwd_min in (30, 60, 90):
                future_hhmm = _add_hhmm(anchor, fwd_min)
                fwd_close = anchor_df["d"].apply(
                    lambda d: close_lu.get((sym, d, future_hhmm), np.nan)
                )
                anchor_df[f"fwd_close_{fwd_min}m"] = fwd_close
                anchor_df[f"fwd_return_{fwd_min}m_pct"] = (
                    (fwd_close - anchor_df["close"]) / anchor_df["close"] * 100.0
                )

            for _, r in anchor_df.iterrows():
                if pd.isna(r["vol_ratio"]):
                    continue
                if pd.isna(r["bar_return_pct"]):
                    continue
                rows.append({
                    "label": label,
                    "symbol": sym,
                    "d": r["d"],
                    "anchor_hhmm": anchor,
                    "vol_ratio": float(r["vol_ratio"]),
                    "bar_return_pct": float(r["bar_return_pct"]),
                    "fwd_return_30m_pct": float(r.get("fwd_return_30m_pct", np.nan)),
                    "fwd_return_60m_pct": float(r.get("fwd_return_60m_pct", np.nan)),
                    "fwd_return_90m_pct": float(r.get("fwd_return_90m_pct", np.nan)),
                })
    print(f"[{label}] yielded {len(rows):,} observations", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_cohort(df: pd.DataFrame, label: str) -> dict:
    print(f"\n--- {label}: per-anchor signature ---")
    if df.empty:
        print("  (no data)")
        return {}
    stats = {}
    for anchor in TIME_ANCHORS:
        sub = df[df["anchor_hhmm"] == anchor].dropna(subset=["vol_ratio", "bar_return_pct"])
        if sub.empty:
            continue
        # Signature stats
        n = len(sub)
        mean_vol_ratio = float(sub["vol_ratio"].mean())
        p50_vol_ratio = float(sub["vol_ratio"].median())
        mean_abs_return = float(sub["bar_return_pct"].abs().mean())
        # Direction-conditioned forward returns: when bar moved UP, what's fwd_30m?
        # When bar moved DOWN, what's fwd_30m?
        up = sub[sub["bar_return_pct"] > 0]
        dn = sub[sub["bar_return_pct"] < 0]
        # Mean signed fwd return per direction
        mean_fwd_30_up = float(up["fwd_return_30m_pct"].dropna().mean()) if len(up) else np.nan
        mean_fwd_30_dn = float(dn["fwd_return_30m_pct"].dropna().mean()) if len(dn) else np.nan
        # Continuation/reversal correlation
        valid = sub.dropna(subset=["bar_return_pct", "fwd_return_30m_pct"])
        corr_30 = float(valid["bar_return_pct"].corr(valid["fwd_return_30m_pct"])) if len(valid) > 1 else np.nan
        valid60 = sub.dropna(subset=["bar_return_pct", "fwd_return_60m_pct"])
        corr_60 = float(valid60["bar_return_pct"].corr(valid60["fwd_return_60m_pct"])) if len(valid60) > 1 else np.nan
        # Vol-conditioned: only high-vol bars
        high_vol = sub[sub["vol_ratio"] >= 1.5]
        if len(high_vol) >= 5:
            hv_corr_30 = float(
                high_vol.dropna(subset=["bar_return_pct", "fwd_return_30m_pct"])
                ["bar_return_pct"].corr(
                    high_vol.dropna(subset=["bar_return_pct", "fwd_return_30m_pct"])
                    ["fwd_return_30m_pct"]
                )
            )
        else:
            hv_corr_30 = np.nan
        print(f"  [{anchor}] n={n:,}  mean(vol_ratio)={mean_vol_ratio:.2f}  p50(vol_ratio)={p50_vol_ratio:.2f}  "
              f"mean|bar_ret|={mean_abs_return:.3f}%  "
              f"corr(bar,fwd30)={corr_30:+.3f}  corr(bar,fwd60)={corr_60:+.3f}  "
              f"high-vol-corr30={hv_corr_30:+.3f}")
        print(f"           mean fwd_30 | UP bar = {mean_fwd_30_up:+.4f}%   DOWN bar = {mean_fwd_30_dn:+.4f}%")
        stats[anchor] = {
            "n": n,
            "mean_vol_ratio": mean_vol_ratio,
            "p50_vol_ratio": p50_vol_ratio,
            "mean_abs_return_pct": mean_abs_return,
            "mean_fwd_30_up": mean_fwd_30_up,
            "mean_fwd_30_dn": mean_fwd_30_dn,
            "corr_bar_fwd30": corr_30,
            "corr_bar_fwd60": corr_60,
            "hv_corr_30": hv_corr_30,
        }
    return stats


def compare_signatures(eur_stats: dict, base_stats: dict) -> None:
    print("\n" + "=" * 78)
    print("SIGNATURE DELTA (Europe cohort vs Baseline)")
    print("=" * 78)
    for anchor in TIME_ANCHORS:
        es = eur_stats.get(anchor, {})
        bs = base_stats.get(anchor, {})
        if not es or not bs:
            print(f"  [{anchor}] insufficient data")
            continue
        d_vol = es["mean_vol_ratio"] - bs["mean_vol_ratio"]
        d_abs_ret = es["mean_abs_return_pct"] - bs["mean_abs_return_pct"]
        d_corr30 = es["corr_bar_fwd30"] - bs["corr_bar_fwd30"]
        print(f"  [{anchor}] d_vol_ratio={d_vol:+.3f}  "
              f"d_mean|bar_ret|={d_abs_ret:+.3f}pp  "
              f"d_corr(bar,fwd30)={d_corr30:+.3f}")
    print()
    print("Interpretation key:")
    print("  Δvol_ratio > +0.10        : European cohort has materially MORE vol burst at this anchor")
    print("  Δmean|bar_ret| > +0.05pp  : European cohort moves materially MORE at this anchor")
    print("  Δcorr(bar,fwd30) << 0     : European cohort REVERTS more (fade-the-impulse setup)")
    print("  Δcorr(bar,fwd30) >> 0     : European cohort CONTINUES more (follow-the-impulse setup)")


def main() -> int:
    print("=" * 78)
    print("Phase 1.5 signature: Indian-market reaction to European open (~12:30-13:30 IST)")
    print("=" * 78)
    print(f"  Discovery window: {WINDOW_START} to {WINDOW_END}")
    print(f"  Europe cohort: {len(EUROPE_COHORT)} symbols")
    print(f"  Baseline cohort: {len(BASELINE_COHORT)} symbols")
    print(f"  Time anchors: {TIME_ANCHORS}")
    print(f"  Vol baseline: prior {VOL_BASELINE_DAYS}-day rolling mean of same-anchor volume")

    eur_df = measure_cohort(EUROPE_COHORT, "Europe-cohort")
    base_df = measure_cohort(BASELINE_COHORT, "Baseline")

    eur_stats = report_cohort(eur_df, "Europe-cohort")
    base_stats = report_cohort(base_df, "Baseline")

    compare_signatures(eur_stats, base_stats)

    # Save raw measurements
    out_path = _REPO_ROOT / "reports" / "sub9_sanity" / "_phase1_5_europe_open_signature.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full = pd.concat([eur_df, base_df], ignore_index=True)
    full.to_csv(out_path, index=False)
    print(f"\nRaw measurements written to: {out_path}  ({len(full):,} rows)")

    # Verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not eur_stats or not base_stats:
        print("  Insufficient data — abort.")
        return 1
    # Pick best anchor by |Δcorr(bar,fwd30)|
    best_anchor = max(
        TIME_ANCHORS,
        key=lambda a: abs(eur_stats.get(a, {}).get("corr_bar_fwd30", 0)
                          - base_stats.get(a, {}).get("corr_bar_fwd30", 0))
    )
    es = eur_stats[best_anchor]
    bs = base_stats[best_anchor]
    d_corr = es["corr_bar_fwd30"] - bs["corr_bar_fwd30"]
    d_vol = es["mean_vol_ratio"] - bs["mean_vol_ratio"]
    print(f"  Best anchor (largest fwd-corr divergence vs baseline): {best_anchor}")
    print(f"    Δcorr(bar,fwd30) = {d_corr:+.3f}")
    print(f"    Δvol_ratio = {d_vol:+.3f}")
    if abs(d_corr) < 0.10 and abs(d_vol) < 0.10:
        print("\n  -> KILL: no meaningful signature divergence between Europe cohort and baseline")
        print("     at any of the three anchors. European open does not produce a measurable")
        print("     intraday impulse / reversion pattern in Europe-exposed Indian names.")
    else:
        direction = "FADE" if d_corr < 0 else "FOLLOW"
        print(f"\n  -> PROCEED TO STAGE 3: signature present at {best_anchor}, direction = {direction}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
