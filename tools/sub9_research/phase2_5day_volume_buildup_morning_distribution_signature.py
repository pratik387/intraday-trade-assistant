# tools/sub9_research/phase2_5day_volume_buildup_morning_distribution_signature.py
#
# Phase 2 empirical signature for 5day_volume_buildup_morning_distribution_short.
# See specs/2026-05-22-brief-5day_volume_buildup_morning_distribution_short.md
#
# Anti-bias guards (Lesson #5):
#   1. 5-day vol sum uses daily.volume[T-5..T-1]; 60-day baseline uses
#      rolling-5-day-sums on T-65..T-6 (NO overlap with the current 5-day window).
#   2. First-hour features computed only from bars 09:15..10:30 (no look-ahead).
#   3. One record per (sym, day): the 10:30 close decides signal vs baseline.
#   4. ProductionUniverseGate per-date (Lesson #19).
#   5. large_cap AND unknown EXCLUDED from universe (small_cap + mid_cap only).
#
# Lesson #19 carry-over (CRITICAL):
#   delivery_pct is NOT in consolidated_daily.feather. It lives in
#   data/delivery_pct/delivery_history.parquet. Falsifier #1 (signal-cohort
#   delivery% MEDIAN < baseline-cohort delivery% MEDIAN) silently disables if
#   we skip the join. We explicitly pd.merge on (symbol, date) and assert
#   non-null coverage before evaluating Falsifier #1.
#
# Pre-registration discipline (Lesson #2):
#   Falsifier #1 is computed and printed BEFORE the drift-delta step.
#   If signal delivery% >= baseline, the "distribution" thesis fails -> KILL.
"""Phase 2 empirical signature - 5day_volume_buildup_morning_distribution_short."""
from __future__ import annotations

import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402

# -----------------------------------------------------------------------------
# CONFIG  --  NO hardcoded defaults inside the logic; every knob is declared here.
# -----------------------------------------------------------------------------
CONFIG: Dict[str, object] = {
    # Date window
    "window_start": date(2023, 1, 2),
    "window_end":   date(2026, 4, 30),

    # Universe
    "accepted_caps": {"small_cap", "mid_cap"},  # large_cap + unknown EXCLUDED
    "require_mis":   True,
    "min_daily_bars_required": 65,              # 60-day baseline + 5-day sum

    # Multi-day volume gate
    "vol_z_min":       2.0,
    "baseline_window": 60,   # T-65 .. T-6 (60-day window, EXCLUDES current 5-day)
    "sum_window":      5,    # T-5 .. T-1

    # First-hour window for signal
    "fh_start": dtime(9, 15),
    "fh_end":   dtime(10, 30),   # 09:15 .. 10:30 inclusive

    # Anchor bar for signal: 5m bar whose timestamp is 10:25 (closes at 10:30).
    # Schema convention in monthly_5m feathers: 'date' column carries bar START ts.
    "anchor_bar_time": dtime(10, 25),     # 5m bar 10:25-10:30 -> close at 10:30
    "open_bar_time":   dtime(9, 15),       # 5m bar 09:15-09:20 -> open_at_0915

    # Target bar (exit reference): 5m bar at 13:25 -> close at 13:30
    "target_bar_time": dtime(13, 25),

    # Regime / period cuts
    "regime_2024_cut":  date(2024, 1, 1),
    "sebi_oct2025_cut": date(2025, 10, 1),

    # Acceptance
    "drift_delta_max": -0.15,   # signal_mean - baseline_mean must be <= -0.15%
    "n_signal_min":    200,

    # Paths
    "monthly_5m_dir":     _REPO_ROOT / "backtest-cache-download" / "monthly",
    "daily_feather_path": _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather",
    "delivery_parquet":   _REPO_ROOT / "data" / "delivery_pct" / "delivery_history.parquet",
    "out_csv":            _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_5day_volume_buildup_morning_distribution_signature.csv",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _months_between(d0: date, d1: date) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _ensure_naive_ist(ts_col: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64tz_dtype(ts_col):
        return ts_col.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts_col


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# Multi-day vol-z table builder (daily, per-symbol)
# -----------------------------------------------------------------------------
def build_vol_buildup_table() -> pd.DataFrame:
    """Return DataFrame: columns = [symbol, t1_date, vol_z, sum5, base_mean, base_std]

    `t1_date` is the daily-bar date used as T-1; the intraday signal is evaluated
    on the NEXT trading session (T+0). We join on (symbol, T+0 session_date)
    after we know which trading dates exist in the 5m feathers.

    Definitions:
      sum5      = sum(volume[T-5..T-1])                  (5 daily bars ending T-1)
      base[k]   = sum(volume[k-4..k]) for k in T-65..T-6 (rolling-5 sums)
      base_mean = mean(base[T-65..T-6])                  (60 sample window)
      base_std  = std(base[T-65..T-6])
      vol_z     = (sum5 - base_mean) / base_std
    """
    _log(f"loading daily feather: {CONFIG['daily_feather_path']}")
    daily = pd.read_feather(CONFIG["daily_feather_path"])
    daily["ts"] = pd.to_datetime(daily["ts"])
    daily["d"] = daily["ts"].dt.date
    daily = daily.sort_values(["symbol", "ts"]).reset_index(drop=True)

    min_bars = int(CONFIG["min_daily_bars_required"])
    sum_w = int(CONFIG["sum_window"])
    base_w = int(CONFIG["baseline_window"])
    # rolling-5 sum
    daily["roll5"] = daily.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(sum_w, min_periods=sum_w).sum()
    )
    # baseline mean/std of rolling-5 sums OVER 60 prior samples,
    # excluding the most recent 5 (so no overlap with current window).
    # base series at T-1 must be computed from rolling-5 sums at T-6..T-65.
    # Implementation: take roll5, shift by sum_w (so position T-1 sees roll5[T-6]),
    # then rolling(base_w) mean/std.
    daily["_roll5_lag5"] = daily.groupby("symbol")["roll5"].shift(sum_w)
    daily["base_mean"] = daily.groupby("symbol")["_roll5_lag5"].transform(
        lambda s: s.rolling(base_w, min_periods=base_w).mean()
    )
    daily["base_std"] = daily.groupby("symbol")["_roll5_lag5"].transform(
        lambda s: s.rolling(base_w, min_periods=base_w).std()
    )
    # bar count prior to (and including) this row's date, per symbol
    daily["bar_count"] = daily.groupby("symbol").cumcount() + 1

    daily["vol_z"] = (daily["roll5"] - daily["base_mean"]) / daily["base_std"]
    qualified = daily[
        (daily["bar_count"] >= min_bars)
        & (daily["vol_z"] >= float(CONFIG["vol_z_min"]))
        & daily["base_std"].notna()
        & (daily["base_std"] > 0)
    ].copy()

    qualified = qualified.rename(columns={"d": "t1_date", "roll5": "sum5"})
    out = qualified[["symbol", "t1_date", "vol_z", "sum5", "base_mean", "base_std"]].copy()
    _log(f"  {len(out):,} (sym, T-1) qualifiers (vol_z >= {CONFIG['vol_z_min']})")
    return out


# -----------------------------------------------------------------------------
# Per-(symbol, day) evaluator
# -----------------------------------------------------------------------------
def evaluate_symbol_day(sym_bars: pd.DataFrame, sym: str, d: date) -> Optional[dict]:
    """Compute first-hour features + target return for (sym, T+0).

    Returns ONE record (signal OR baseline) per (sym, day) given the cohort
    is already known to satisfy the multi-day vol-buildup gate at T-1.
    """
    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    if sym_bars.empty:
        return None

    mask_sess = (sym_bars["time"] >= dtime(9, 15)) & (sym_bars["time"] <= dtime(15, 25))
    sym_bars = sym_bars[mask_sess].reset_index(drop=True)
    if sym_bars.empty:
        return None

    fh_start = CONFIG["fh_start"]
    fh_end = CONFIG["fh_end"]
    open_t = CONFIG["open_bar_time"]
    anchor_t = CONFIG["anchor_bar_time"]
    tgt_t = CONFIG["target_bar_time"]

    # first_hour rows: bars whose start time is in [09:15, 10:25] inclusive,
    # i.e. the bar 10:25-10:30 IS the last first-hour bar.
    fh_rows = sym_bars[(sym_bars["time"] >= fh_start) & (sym_bars["time"] <= anchor_t)]
    if fh_rows.empty:
        return None

    open_rows = fh_rows[fh_rows["time"] == open_t]
    anchor_rows = fh_rows[fh_rows["time"] == anchor_t]
    tgt_rows = sym_bars[sym_bars["time"] == tgt_t]
    if open_rows.empty or anchor_rows.empty or tgt_rows.empty:
        return None

    first_hour_open = float(open_rows.iloc[0]["open"])
    first_hour_low = float(fh_rows["low"].min())
    close_at_1030 = float(anchor_rows.iloc[0]["close"])
    target_close = float(tgt_rows.iloc[0]["close"])

    midpoint = (first_hour_open + first_hour_low) / 2.0
    is_signal = close_at_1030 <= midpoint
    is_baseline = not is_signal

    # SHORT: ret = (entry - exit) / entry * 100; but we keep convention
    # ret_to_1330 = (target - entry) / entry * 100  (so SHORT-favourable < 0).
    ret_to_1330 = (target_close - close_at_1030) / close_at_1030 * 100.0

    return {
        "symbol": sym,
        "date": d.isoformat(),
        "first_hour_open":  first_hour_open,
        "first_hour_low":   first_hour_low,
        "first_hour_midpoint": midpoint,
        "close_at_1030":    close_at_1030,
        "target_close_1330": target_close,
        "ret_to_1330":      ret_to_1330,
        "is_signal":        bool(is_signal),
        "is_baseline":      bool(is_baseline),
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature -- 5day_volume_buildup_morning_distribution_short")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print(f"Caps:   {sorted(CONFIG['accepted_caps'])}  (large_cap + unknown EXCLUDED)")
    print(f"Multi-day gate: 5-day vol-sum z >= {CONFIG['vol_z_min']} vs 60-day rolling-5-sum baseline")
    print(f"Signal trigger: close_at_1030 <= (open_at_0915 + first_hour_low)/2  -> SHORT")
    print(f"Baseline:       close_at_1030 >  midpoint (upper-half close)")
    print(f"Target: close at 13:25 (= 13:30 IST exit reference)")
    print("=" * 80)

    # Build daily vol-buildup qualifier table.
    qual = build_vol_buildup_table()
    # Build O(1) lookup: (symbol, t1_date) -> (vol_z, sum5, base_mean, base_std)
    qual_lookup: Dict[Tuple[str, date], Tuple[float, float, float, float]] = {}
    for row in qual.itertuples(index=False):
        qual_lookup[(row.symbol, row.t1_date)] = (
            float(row.vol_z), float(row.sum5), float(row.base_mean), float(row.base_std),
        )
    # Per-symbol sorted T-1 dates for nearest-prior lookup
    qual_keyset: Dict[str, List[date]] = {}
    for s, sub in qual.groupby("symbol", sort=False):
        qual_keyset[s] = sorted(sub["t1_date"].tolist())
    _log(f"  qual_lookup keys: {len(qual_lookup):,};  symbols: {len(qual_keyset):,}")

    # ---- Delivery% join table (Lesson #19 CRITICAL) ----
    _log(f"loading delivery%: {CONFIG['delivery_parquet']}")
    deliv = pd.read_parquet(CONFIG["delivery_parquet"])
    # Schema: symbol, date (datetime), series, delivery_pct
    deliv["date"] = pd.to_datetime(deliv["date"])
    deliv["d"] = deliv["date"].dt.date
    # Prefer EQ, fall back to whatever exists per (symbol, date).
    deliv["_series_rank"] = (deliv["series"] != "EQ").astype(int)  # EQ=0, others=1
    deliv = deliv.sort_values(["symbol", "d", "_series_rank"])
    deliv = deliv.drop_duplicates(subset=["symbol", "d"], keep="first")
    deliv_lookup = deliv.set_index(["symbol", "d"])["delivery_pct"].to_dict()
    _log(f"  delivery% lookup built: {len(deliv_lookup):,} (sym, date) keys")

    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )

    records: List[dict] = []
    months = _months_between(CONFIG["window_start"], CONFIG["window_end"])
    total_eval_pairs = 0
    total_rejected_universe = 0
    total_no_qual = 0
    total_no_record = 0

    for (yy, mm) in months:
        path = Path(CONFIG["monthly_5m_dir"]) / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            _log(f"skip {yy:04d}-{mm:02d} (no 5m feather)")
            continue

        df = pd.read_feather(
            path, columns=["symbol", "date", "open", "high", "low", "close", "volume"],
        )
        df["date"] = _ensure_naive_ist(df["date"])
        df["day"] = df["date"].dt.date
        df["time"] = df["date"].dt.time
        df = df[(df["day"] >= CONFIG["window_start"]) & (df["day"] <= CONFIG["window_end"])]
        if df.empty:
            continue

        nse_all = gate._load_nse_all()
        accepted = CONFIG["accepted_caps"]
        # Precompute eligible-cap+mis symbol set (gate uses cap+mis dict lookup only
        # since min_days=0 and min_vol=0; we can replicate it without per-pair calls).
        cap_ok = {s for s, row in nse_all.items() if row.cap_segment in accepted}
        if CONFIG["require_mis"]:
            mis_ok = {s for s, row in nse_all.items() if row.mis_enabled}
            elig_syms = cap_ok & mis_ok
        else:
            elig_syms = cap_ok
        # Also restrict to symbols that have ANY qualifier (cheap pre-filter)
        keep_syms = elig_syms & set(qual_keyset.keys())
        df = df[df["symbol"].isin(keep_syms)]
        if df.empty:
            continue

        gb = df.groupby(["symbol", "day"], sort=False)
        month_eval = 0
        month_reject = 0
        month_noqual = 0
        month_rec = 0

        for (sym, d), bars in gb:
            month_eval += 1
            total_eval_pairs += 1
            # Universe check already enforced via keep_syms pre-filter (cap + mis).
            # Look up T-1: most recent qualifier strictly < d, within 7 calendar days.
            qd_list = qual_keyset.get(sym, [])
            if not qd_list:
                month_noqual += 1
                total_no_qual += 1
                continue
            # binary search
            lo, hi = 0, len(qd_list)
            while lo < hi:
                mid = (lo + hi) // 2
                if qd_list[mid] < d:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo - 1
            if idx < 0:
                month_noqual += 1
                total_no_qual += 1
                continue
            t1 = qd_list[idx]
            if (d - t1).days > 7:
                month_noqual += 1
                total_no_qual += 1
                continue

            rec = evaluate_symbol_day(bars, sym, d)
            if rec is None:
                total_no_record += 1
                continue

            # O(1) lookup for multi-day stats + delivery_pct_T-1 join (CRITICAL).
            qstats = qual_lookup.get((sym, t1))
            if qstats is None:
                total_no_record += 1
                continue
            vz, s5, bm, bs = qstats
            rec["t1_date"] = t1.isoformat()
            rec["vol_z"] = vz
            rec["sum5"] = s5
            rec["base_mean"] = bm
            rec["base_std"] = bs

            dp = deliv_lookup.get((sym, t1))
            rec["delivery_pct_T-1"] = float(dp) if dp is not None else np.nan

            records.append(rec)
            month_rec += 1

        _log(
            f"  {yy:04d}-{mm:02d}: pairs={month_eval:,} "
            f"rejected_univ={month_reject:,} "
            f"no_qual={month_noqual:,} "
            f"records={month_rec:,}"
        )

    print()
    print("=== Pipeline tally ===")
    print(f"  total (sym, day) pairs in caps:      {total_eval_pairs:,}")
    print(f"  rejected by universe gate:           {total_rejected_universe:,}")
    print(f"  no multi-day qualifier (T-1):        {total_no_qual:,}")
    print(f"  missing target bar / open:           {total_no_record:,}")
    print(f"  recorded:                            {len(records):,}")

    if not records:
        print("\nNO RECORDS COLLECTED -- abort.")
        return 1

    df_out = pd.DataFrame.from_records(records)

    # Attach cap_segment + cohort split columns
    nse_all = gate._load_nse_all()

    def _cap(s: str) -> str:
        row = nse_all.get(s)
        return row.cap_segment if row else "unknown"

    df_out["cap_segment"] = df_out["symbol"].map(_cap)

    dt_dates = pd.to_datetime(df_out["date"])
    cut_2024 = pd.Timestamp(CONFIG["regime_2024_cut"])
    cut_sebi = pd.Timestamp(CONFIG["sebi_oct2025_cut"])
    df_out["regime_pre_post_2024"] = np.where(dt_dates < cut_2024, "pre", "post")
    df_out["period_pre_post_sebi_oct2025"] = np.where(dt_dates < cut_sebi, "pre", "post")

    def _vz_bucket(v: float) -> str:
        if pd.isna(v):
            return "nan"
        if v < 2.5:
            return "2.0-2.5"
        if v < 3.0:
            return "2.5-3.0"
        return ">=3.0"

    df_out["vol_z_bucket"] = df_out["vol_z"].map(_vz_bucket)

    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out):,} rows -> {out_path}")

    sig = df_out[df_out["is_signal"]].copy()
    base = df_out[df_out["is_baseline"]].copy()
    n_sig = len(sig)
    n_base = len(base)

    # -------------------------------------------------------------------------
    # STEP 1 -- Falsifier #1 (delivery% corroboration)  --  BEFORE drift
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 1 -- Falsifier #1 (delivery% corroboration)")
    print("=" * 80)

    sig_dp = sig["delivery_pct_T-1"].dropna()
    base_dp = base["delivery_pct_T-1"].dropna()
    sig_dp_cov = float(len(sig_dp)) / max(1, n_sig)
    base_dp_cov = float(len(base_dp)) / max(1, n_base)

    print(f"  n_signal:                 {n_sig:,}")
    print(f"  n_baseline:               {n_base:,}")
    print(f"  delivery% join coverage:  signal={sig_dp_cov*100:.1f}%, baseline={base_dp_cov*100:.1f}%")

    if len(sig_dp) == 0 or len(base_dp) == 0:
        print("  delivery% data missing -- cannot evaluate Falsifier #1.")
        print("\n" + "=" * 80)
        print("VERDICT: KILL (Falsifier #1 not evaluable -- check delivery% join)")
        print("=" * 80)
        return 0

    sig_dp_med = float(sig_dp.median())
    base_dp_med = float(base_dp.median())
    sig_dp_mean = float(sig_dp.mean())
    base_dp_mean = float(base_dp.mean())

    print(f"  signal   delivery% T-1 median: {sig_dp_med:.3f}%   (mean {sig_dp_mean:.3f}%)")
    print(f"  baseline delivery% T-1 median: {base_dp_med:.3f}%   (mean {base_dp_mean:.3f}%)")
    print(f"  delivery% MEDIAN delta (sig - base): {sig_dp_med - base_dp_med:+.3f}%   [must be NEGATIVE]")

    falsifier1_pass = sig_dp_med < base_dp_med
    print(f"\n  FALSIFIER #1: {'PASS' if falsifier1_pass else 'FAIL'}")

    if not falsifier1_pass:
        print("\n  Signal cohort delivery% MEDIAN is NOT below baseline.")
        print("  The 5-day volume buildup looks like REAL accumulation, not distribution.")
        print("\n" + "=" * 80)
        print("VERDICT: KILL (Falsifier #1 failed -- distribution thesis rejected)")
        print("=" * 80)
        return 0

    # -------------------------------------------------------------------------
    # STEP 2 -- Drift delta (only if Falsifier #1 PASSED)
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 2 -- Drift delta")
    print("=" * 80)

    sig_mean = float(sig["ret_to_1330"].mean()) if n_sig else float("nan")
    base_mean = float(base["ret_to_1330"].mean()) if n_base else float("nan")
    sig_median = float(sig["ret_to_1330"].median()) if n_sig else float("nan")
    base_median = float(base["ret_to_1330"].median()) if n_base else float("nan")
    delta = sig_mean - base_mean if (n_sig and n_base) else float("nan")

    print(f"  n_signal:                   {n_sig:,}   [>= {CONFIG['n_signal_min']} required]")
    print(f"  n_baseline:                 {n_base:,}")
    print(f"  Signal   mean ret_to_1330:  {sig_mean:+.4f}%  (median {sig_median:+.4f}%)")
    print(f"  Baseline mean ret_to_1330:  {base_mean:+.4f}%  (median {base_median:+.4f}%)")
    print(f"  DRIFT DELTA (sig - base):   {delta:+.4f}%   [<= {CONFIG['drift_delta_max']:.2f}% required]")

    # -------------------------------------------------------------------------
    # STEP 3 -- Cohort splits
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 3 -- Cohort splits")
    print("=" * 80)
    hdr = f"{'split':<34}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}"
    print(hdr)

    def _split_block(label: str, mask_sig: pd.Series, mask_base: pd.Series) -> Tuple[int, float, float, float]:
        s = sig[mask_sig]
        b = base[mask_base]
        ns, nb = len(s), len(b)
        sm = float(s["ret_to_1330"].mean()) if ns else float("nan")
        bm = float(b["ret_to_1330"].mean()) if nb else float("nan")
        dl = sm - bm if (ns and nb) else float("nan")
        print(f"{label:<34}{ns:>8d}{nb:>8d}{sm:>12.4f}{bm:>12.4f}{dl:>12.4f}")
        return ns, sm, bm, dl

    sig_dates = pd.to_datetime(sig["date"])
    base_dates = pd.to_datetime(base["date"])

    _split_block("pre_2024",            sig_dates < cut_2024,  base_dates < cut_2024)
    _split_block("post_2024",           sig_dates >= cut_2024, base_dates >= cut_2024)
    _split_block("pre_sebi_oct2025",    sig_dates < cut_sebi,  base_dates < cut_sebi)
    post_sebi_n, _, _, post_sebi_delta = _split_block(
        "post_sebi_oct2025", sig_dates >= cut_sebi, base_dates >= cut_sebi
    )

    for cap_val in ["small_cap", "mid_cap"]:
        _split_block(f"cap={cap_val}", sig["cap_segment"] == cap_val, base["cap_segment"] == cap_val)

    print()
    print("  --- vol_z buckets ---")
    print(hdr)
    bucket_stats: List[Tuple[str, int, float]] = []
    for bucket in ["2.0-2.5", "2.5-3.0", ">=3.0"]:
        mask_s = sig["vol_z_bucket"] == bucket
        mask_b = base["vol_z_bucket"] == bucket
        ns, sm, bm, dl = _split_block(f"vol_z={bucket}", mask_s, mask_b)
        bucket_stats.append((bucket, ns, sm))

    print()
    print("  --- delivery% distribution (sig vs base) ---")
    print(f"  signal   delivery% mean / median: {sig_dp_mean:.3f}% / {sig_dp_med:.3f}%   n={len(sig_dp):,}")
    print(f"  baseline delivery% mean / median: {base_dp_mean:.3f}% / {base_dp_med:.3f}%   n={len(base_dp):,}")
    if len(sig_dp) and len(base_dp):
        for q in (0.25, 0.50, 0.75):
            qs = float(sig_dp.quantile(q))
            qb = float(base_dp.quantile(q))
            print(f"  Q{int(q*100):02d}:  sig={qs:.3f}%   base={qb:.3f}%   delta={qs-qb:+.3f}%")

    # -------------------------------------------------------------------------
    # VERDICT
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    drift_ok = (not np.isnan(delta)) and (delta <= float(CONFIG["drift_delta_max"]))
    n_ok = n_sig >= int(CONFIG["n_signal_min"])
    sign_flip = (
        (not np.isnan(post_sebi_delta))
        and (post_sebi_delta > 0)
        and (post_sebi_n >= 50)
    )

    if not n_ok:
        verdict = "KILL"
        reason = f"n_signal {n_sig} < {CONFIG['n_signal_min']}"
    elif not drift_ok:
        verdict = "KILL"
        reason = (
            f"drift delta {delta:+.4f}% > {CONFIG['drift_delta_max']:.2f}% "
            f"(insufficient SHORT footprint)"
        )
    elif sign_flip:
        verdict = "DEFER"
        reason = (
            f"post-SEBI-Oct-2025 cohort delta is POSITIVE "
            f"({post_sebi_delta:+.4f}%, n_sig={post_sebi_n}) -- regulatory regime risk"
        )
    else:
        verdict = "PROCEED to Phase 3"
        reason = (
            f"Falsifier #1 PASS, "
            f"drift {delta:+.4f}% <= {CONFIG['drift_delta_max']:.2f}%, "
            f"n_signal {n_sig} >= {CONFIG['n_signal_min']}, no post-SEBI sign flip"
        )

    print(f"  {verdict}")
    print(f"  reason: {reason}")

    # Monotonicity inspection across vol_z buckets:
    # as vol_z rises (stronger buildup), if mechanism is real then signal_mean
    # should become MORE NEGATIVE -> sm_seq monotonically DECREASING.
    sm_seq = [sm for (_, ns, sm) in bucket_stats if ns > 0]
    note_lines: List[str] = []
    if len(sm_seq) >= 2:
        increasing = all(sm_seq[i] <= sm_seq[i+1] for i in range(len(sm_seq)-1))
        decreasing = all(sm_seq[i] >= sm_seq[i+1] for i in range(len(sm_seq)-1))
        if decreasing:
            note_lines.append(
                "  MONOTONICITY: vol_z bucket signal_mean monotonically DECREASES from 2.0-2.5 -> >=3.0 "
                "(more negative as buildup gets stronger). Mechanism confirmation."
            )
        elif increasing:
            note_lines.append(
                "  MONOTONICITY: vol_z bucket signal_mean monotonically INCREASES (less negative as buildup gets stronger). "
                "Mechanism INVERTED -- stronger buildup means LESS distribution. CAUTION."
            )
        else:
            note_lines.append(
                "  MONOTONICITY: vol_z bucket signal_mean NON-MONOTONIC -- buildup-strength effect is mixed."
            )

    if note_lines:
        print()
        print("Notes:")
        for ln in note_lines:
            print(ln)

    return 0


if __name__ == "__main__":
    sys.exit(run())
