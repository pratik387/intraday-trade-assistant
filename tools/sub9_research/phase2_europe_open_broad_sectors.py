"""Phase 2 (redone): European-open signature test on BROAD MARKET + SECTOR COHORTS.

Per user's correction (2026-05-20): the previous narrow Europe-revenue cohort
was a guess. Phase 1 redone identified 5 candidate channels (FPI flow, GIFT
Nifty lead, EUR/INR FX, Brent crude, EUREX index futures). FPI flow channel
in particular suggests the signature should hit the WHOLE FPI book (broad
large-cap), not a revenue-exposed sub-cohort.

# Test scope (Step 1 of 5)

Broad market: NIFTYBEES (Nifty 50 ETF on NSE, high vol, real-time tracker)
Sectors built from on-disk constituent stocks:
  - BANK (large-cap private + state banks)
  - IT (export-heavy, USD/EUR revenue)
  - PHARMA (EMA/MHRA regulated)
  - AUTO
  - METAL (LME-pegged)
  - FMCG (domestic consumption — baseline-like)
  - ENERGY/OIL (Brent crude sensitive)
  - REALTY
  - DOMESTIC_BANK (PSU + low-FII — baseline-like)

# Signature measured at 12:30 / 13:00 / 13:30 IST anchors

For each cohort + anchor:
  - mean(vol_ratio_30d_anchor)
  - mean abs(bar_return) on signal bar
  - corr(bar_return_signal, fwd_return_30m)
  - sign(mean fwd_return) on UP bars vs DOWN bars

Additional comparison: 12:30 anchor vs 11:30 anchor (lunch quiet) vs 14:30
anchor on the BROAD market — is 12:30 uniquely elevated, or just the generic
post-noon pickup the NSE J/U-shape predicts?

If broad market shows NO signature distinct from 11:30/14:30 anchors,
KILL (no point fetching sectoral indices for Step 2).
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
# Cohort definitions (all symbols already in monthly feathers)
# ---------------------------------------------------------------------------

COHORTS: Dict[str, List[str]] = {
    "BROAD_MARKET_NIFTYBEES": ["NIFTYBEES"],
    "BANK": ["HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN", "KOTAKBANK", "INDUSINDBK"],
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA"],
    "AUTO": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT"],
    "METAL": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA"],
    "FMCG_DOMESTIC": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
    "OIL_ENERGY": ["RELIANCE", "ONGC", "IOC", "BPCL", "HPCL"],
    "REALTY": ["DLF", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD"],
    "PSU_BANK_DOMESTIC": ["SBIN", "PNB", "BANKBARODA", "CANBK", "BANKINDIA"],
}

# Time anchors (5m bar open timestamps in IST)
EUROPE_ANCHORS = ["12:30", "13:00", "13:30"]
# Control anchors (NOT European-open times — for "is 12:30 unique?" test)
CONTROL_ANCHORS = ["11:30", "14:30"]
ALL_ANCHORS = EUROPE_ANCHORS + CONTROL_ANCHORS

WINDOW_START = date(2023, 1, 2)
WINDOW_END = date(2024, 12, 31)

VOL_BASELINE_DAYS = 30
MIN_OBSERVATIONS_PER_SYMBOL = 50


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


def _load_5m_for_month(yyyy: int, mm: int, symbols: List[str]) -> pd.DataFrame:
    p = _MONTHLY_DIR / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_feather(p, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    return df.reset_index(drop=True)


def _load_all_data(all_symbols: List[str]) -> pd.DataFrame:
    """Single-pass load: concatenate all months for all unique symbols."""
    chunks = []
    months = _months_in_window()
    print(f"  Loading {len(all_symbols)} symbols across {len(months)} months...", flush=True)
    for (yyyy, mm) in months:
        c = _load_5m_for_month(yyyy, mm, all_symbols)
        if not c.empty:
            chunks.append(c)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df):,} total bars", flush=True)
    return df


def _add_hhmm(hhmm: str, minutes: int) -> str:
    h, m = hhmm.split(":")
    total = int(h) * 60 + int(m) + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Per-symbol per-anchor measurements
# ---------------------------------------------------------------------------

def measure_symbol_anchor(df_sym: pd.DataFrame, anchor: str,
                          close_lookup: dict) -> pd.DataFrame:
    """For a single symbol's bars, compute per-day measurements at this anchor."""
    anchor_df = df_sym[df_sym["hhmm"] == anchor].copy()
    if len(anchor_df) < VOL_BASELINE_DAYS + 5:
        return pd.DataFrame()
    anchor_df = anchor_df.sort_values("d").reset_index(drop=True)
    anchor_df["vol_baseline"] = (
        anchor_df["volume"].rolling(VOL_BASELINE_DAYS, min_periods=VOL_BASELINE_DAYS)
        .mean().shift(1)
    )
    anchor_df["vol_ratio"] = anchor_df["volume"] / anchor_df["vol_baseline"]
    anchor_df["bar_return_pct"] = (
        (anchor_df["close"] - anchor_df["open"]) / anchor_df["open"] * 100.0
    )
    sym = anchor_df["symbol"].iloc[0]
    for fwd_min in (30, 60, 90):
        future_hhmm = _add_hhmm(anchor, fwd_min)
        anchor_df[f"fwd_close_{fwd_min}m"] = anchor_df["d"].apply(
            lambda d: close_lookup.get((sym, d, future_hhmm), np.nan)
        )
        anchor_df[f"fwd_return_{fwd_min}m_pct"] = (
            (anchor_df[f"fwd_close_{fwd_min}m"] - anchor_df["close"])
            / anchor_df["close"] * 100.0
        )
    out = anchor_df[[
        "symbol", "d", "vol_ratio", "bar_return_pct",
        "fwd_return_30m_pct", "fwd_return_60m_pct", "fwd_return_90m_pct",
    ]].copy()
    out["anchor"] = anchor
    return out.dropna(subset=["vol_ratio", "bar_return_pct"])


def measure_cohort(df: pd.DataFrame, cohort_name: str, symbols: List[str],
                    close_lookup: dict) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        sym_df = df[df["symbol"] == sym]
        if len(sym_df) < MIN_OBSERVATIONS_PER_SYMBOL:
            continue
        for anchor in ALL_ANCHORS:
            sub = measure_symbol_anchor(sym_df, anchor, close_lookup)
            if not sub.empty:
                rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["cohort"] = cohort_name
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def cohort_signature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per (cohort, anchor) summary stats."""
    rows = []
    for (cohort, anchor), sub in df.groupby(["cohort", "anchor"]):
        n = len(sub)
        if n < 30:
            continue
        sub = sub.dropna(subset=["bar_return_pct"])
        mean_vol_ratio = float(sub["vol_ratio"].mean())
        p50_vol_ratio = float(sub["vol_ratio"].median())
        mean_abs_ret = float(sub["bar_return_pct"].abs().mean())
        mean_signed_ret = float(sub["bar_return_pct"].mean())
        # Forward-return correlation
        valid30 = sub.dropna(subset=["fwd_return_30m_pct"])
        corr_30 = float(valid30["bar_return_pct"].corr(valid30["fwd_return_30m_pct"])) if len(valid30) > 1 else np.nan
        valid60 = sub.dropna(subset=["fwd_return_60m_pct"])
        corr_60 = float(valid60["bar_return_pct"].corr(valid60["fwd_return_60m_pct"])) if len(valid60) > 1 else np.nan
        # Vol-conditioned: only high-vol bars
        high_vol = sub[sub["vol_ratio"] >= 1.5]
        if len(high_vol) >= 30:
            hv30 = high_vol.dropna(subset=["fwd_return_30m_pct"])
            hv_corr = float(hv30["bar_return_pct"].corr(hv30["fwd_return_30m_pct"])) if len(hv30) > 1 else np.nan
            hv_mean_abs = float(high_vol["bar_return_pct"].abs().mean())
        else:
            hv_corr = np.nan
            hv_mean_abs = np.nan
        rows.append({
            "cohort": cohort, "anchor": anchor, "n": n,
            "mean_vol_ratio": round(mean_vol_ratio, 3),
            "p50_vol_ratio": round(p50_vol_ratio, 3),
            "mean_abs_ret_pct": round(mean_abs_ret, 4),
            "mean_signed_ret_pct": round(mean_signed_ret, 4),
            "corr_30m": round(corr_30, 4) if not np.isnan(corr_30) else np.nan,
            "corr_60m": round(corr_60, 4) if not np.isnan(corr_60) else np.nan,
            "hv_n": int((sub["vol_ratio"] >= 1.5).sum()),
            "hv_mean_abs_ret_pct": round(hv_mean_abs, 4) if not np.isnan(hv_mean_abs) else np.nan,
            "hv_corr_30m": round(hv_corr, 4) if not np.isnan(hv_corr) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 90)
    print("Phase 2 (redone): broad market + sector signature at European-open anchors")
    print("=" * 90)
    print(f"  Discovery window: {WINDOW_START} to {WINDOW_END}")
    print(f"  Europe anchors: {EUROPE_ANCHORS}")
    print(f"  Control anchors (no European event): {CONTROL_ANCHORS}")
    print(f"  Cohorts: {list(COHORTS.keys())}")

    all_symbols = sorted({s for syms in COHORTS.values() for s in syms})
    df = _load_all_data(all_symbols)
    if df.empty:
        print("ABORT: no data loaded")
        return 1

    # Build close lookup ONCE for all symbols
    print("  Building close lookup...", flush=True)
    close_lookup = dict(zip(
        zip(df["symbol"], df["d"], df["hhmm"]),
        df["close"].astype(float),
    ))
    print(f"  Lookup size: {len(close_lookup):,}", flush=True)

    cohort_dfs = []
    for cohort_name, symbols in COHORTS.items():
        print(f"\n[{cohort_name}] measuring {len(symbols)} symbols...", flush=True)
        cdf = measure_cohort(df, cohort_name, symbols, close_lookup)
        if not cdf.empty:
            cohort_dfs.append(cdf)
            print(f"  {len(cdf):,} observations", flush=True)

    if not cohort_dfs:
        print("ABORT: no cohort data")
        return 1

    full = pd.concat(cohort_dfs, ignore_index=True)
    summary = cohort_signature_table(full)

    print("\n" + "=" * 90)
    print("PER-COHORT SIGNATURE TABLE")
    print("=" * 90)
    print(f"{'cohort':<25} {'anchor':<7} {'n':>6} {'vol_r':>6} {'|ret|%':>7} {'signed%':>7} "
          f"{'corr30':>7} {'corr60':>7} {'hv_n':>5} {'hv_|ret|':>8} {'hv_corr30':>9}")
    print("-" * 110)
    for _, r in summary.sort_values(["cohort", "anchor"]).iterrows():
        print(f"{r['cohort']:<25} {r['anchor']:<7} {r['n']:>6} "
              f"{r['mean_vol_ratio']:>6.2f} {r['mean_abs_ret_pct']:>7.3f} "
              f"{r['mean_signed_ret_pct']:>+7.4f} "
              f"{r['corr_30m']:>+7.3f} {r['corr_60m']:>+7.3f} "
              f"{r['hv_n']:>5} "
              f"{r['hv_mean_abs_ret_pct']:>8.3f} "
              f"{r['hv_corr_30m']:>+9.3f}")

    # Verdict logic
    print("\n" + "=" * 90)
    print("VERDICT — does a 12:30/13:00/13:30 signature exist DISTINCT from 11:30/14:30 controls?")
    print("=" * 90)

    # For broad market: check if any European anchor has materially different metrics
    # from the control anchors (11:30, 14:30)
    bm = summary[summary["cohort"] == "BROAD_MARKET_NIFTYBEES"]
    if bm.empty:
        print("  BROAD MARKET data missing — cannot check kill criterion")
        return 1
    print("\n  Broad market (NIFTYBEES) per-anchor:")
    for _, r in bm.iterrows():
        print(f"    {r['anchor']}  n={r['n']:>4}  vol_r={r['mean_vol_ratio']:.2f}  "
              f"|ret|={r['mean_abs_ret_pct']:.3f}%  corr30={r['corr_30m']:+.3f}")
    # Pick controls
    control_metrics = bm[bm["anchor"].isin(CONTROL_ANCHORS)]
    europe_metrics = bm[bm["anchor"].isin(EUROPE_ANCHORS)]
    if control_metrics.empty or europe_metrics.empty:
        print("\n  Cannot verdict — insufficient data on broad market controls or europe anchors")
        return 1
    ctrl_mean_vol = float(control_metrics["mean_vol_ratio"].mean())
    eur_mean_vol = float(europe_metrics["mean_vol_ratio"].mean())
    ctrl_mean_abs = float(control_metrics["mean_abs_ret_pct"].mean())
    eur_mean_abs = float(europe_metrics["mean_abs_ret_pct"].mean())
    ctrl_corr = float(control_metrics["corr_30m"].mean())
    eur_corr = float(europe_metrics["corr_30m"].mean())
    d_vol = eur_mean_vol - ctrl_mean_vol
    d_abs = eur_mean_abs - ctrl_mean_abs
    d_corr = eur_corr - ctrl_corr
    print(f"\n  EUROPE vs CONTROL (broad market):")
    print(f"    delta_vol_ratio = {d_vol:+.3f}  (Europe avg {eur_mean_vol:.3f}, control avg {ctrl_mean_vol:.3f})")
    print(f"    delta_|ret|%    = {d_abs:+.4f}pp (Europe avg {eur_mean_abs:.4f}, control avg {ctrl_mean_abs:.4f})")
    print(f"    delta_corr30    = {d_corr:+.3f}  (Europe avg {eur_corr:+.3f}, control avg {ctrl_corr:+.3f})")
    # Kill thresholds
    KILL_VOL = 0.10
    KILL_ABS = 0.02  # pp
    KILL_CORR = 0.10
    has_signal = (
        abs(d_vol) >= KILL_VOL or abs(d_abs) >= KILL_ABS or abs(d_corr) >= KILL_CORR
    )
    if not has_signal:
        print("\n  -> KILL: broad market shows NO distinguishable signature at European anchors")
        print("     vs same-day control anchors (11:30, 14:30). The 12:30-13:30 IST window is")
        print("     not materially different at the broad-market scale. Stop here. No need to")
        print("     fetch sectoral indices (Step 2).")
    else:
        print("\n  -> PROCEED TO STEP 2: broad market shows possible signature.")
        # Show strongest sector at European anchors
        sector_eur = summary[(summary["cohort"] != "BROAD_MARKET_NIFTYBEES") &
                              (summary["anchor"].isin(EUROPE_ANCHORS))]
        if not sector_eur.empty:
            print("     Top sectors by |corr30| at European anchors:")
            sector_eur_sorted = sector_eur.assign(
                abs_corr=lambda x: x["corr_30m"].abs()
            ).sort_values("abs_corr", ascending=False)
            for _, r in sector_eur_sorted.head(5).iterrows():
                print(f"       {r['cohort']:<25} {r['anchor']:<5} corr30={r['corr_30m']:+.3f} "
                      f"|ret|={r['mean_abs_ret_pct']:.3f}%")

    # Save raw + summary
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_dir / "_phase2_europe_open_broad_sectors_raw.csv", index=False)
    summary.to_csv(out_dir / "_phase2_europe_open_broad_sectors_summary.csv", index=False)
    print(f"\nRaw + summary CSVs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
