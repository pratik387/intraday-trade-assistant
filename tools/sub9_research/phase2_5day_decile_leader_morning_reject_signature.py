# tools/sub9_research/phase2_5day_decile_leader_morning_reject_signature.py
#
# Phase 2 empirical signature for 5day_decile_leader_morning_reject_short.
# See specs/2026-05-22-brief-5day_decile_leader_morning_reject_short.md
#
# Mechanism (one sentence):
#   Small/mid-cap MIS-eligible stocks that as of T-1 close rank in the TOP DECILE
#   of 5-day cumulative return WITHIN their cap_segment, AND on T+0 the 10:25-10:30
#   5m bar close < the 09:15-09:20 bar open (first-hour failure to extend) -> SHORT
#   at 10:30 close, exit 13:30.
#
# Anti-bias guards (Lesson #5 + Lesson #19):
#   1. 5-day rank uses T-6..T-1 daily close ONLY (no T+0 data) -- no look-ahead
#   2. Per-(cap_segment, date) np.quantile of 5day_cumret -- cross-sectional
#   3. First-fire-per-day latch (sym, date) -- single record per (sym, date)
#   4. ProductionUniverseGate per-date (mirrors production setup_universe builders)
#   5. large_cap AND unknown EXCLUDED from universe (per brief)
#
# Pre-registration discipline (Lesson #2):
#   - Falsifier #1 (rank stability T-1 vs T-2) computed FIRST
#   - KILL if mean stability < 50% regardless of drift
#   - CAUTION if 50-60%; PROCEED if >= 60%
"""Phase 2 empirical signature -- 5day_decile_leader_morning_reject_short."""
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
    # Date window  (5m feather coverage 2023-01 .. 2026-04; daily covers same)
    "window_start": date(2023, 1, 10),   # >= 2023-01-10 so T-6 daily lookback resolves
    "window_end":   date(2026, 4, 30),

    # Universe
    "accepted_caps": {"small_cap", "mid_cap"},  # large_cap + unknown EXCLUDED (per brief)
    "require_mis":   True,

    # 5-day cross-sectional rank
    "lookback_days":         5,            # ret = close[T-1] / close[T-6] - 1
    "decile_pct":            0.90,         # top-decile = >= 90th percentile within cap_segment
    "min_cohort_size":       20,           # need >= 20 stocks per (cap_segment, date) for stable quantile

    # Bar times for intraday signal (5m bars labeled by START of bar)
    "open_bar_time":         dtime(9, 15),    # 09:15 5m bar open == 09:15 open
    "signal_bar_time":       dtime(10, 25),   # 10:25 5m bar close == 10:30 IST close
    "target_bar_time":       dtime(13, 25),   # 13:25 5m bar close == 13:30 IST exit

    # Regime / period cuts
    "regime_2024_cut":       date(2024, 1, 1),
    "sebi_oct2025_cut":      date(2025, 10, 1),

    # Acceptance
    "drift_delta_max":       -0.15,         # signal_mean - baseline_mean must be <= -0.15%
    "n_signal_min":          200,

    # Falsifier #1 (rank stability)
    "stability_kill_below":  0.50,
    "stability_pass_at":     0.60,

    # OR-fail proxy thresholds
    "or_fail_high_buf_frac": 0.005,         # high-in-first-hour > open * (1 + 0.5%)

    # Paths
    "monthly_5m_dir":  _REPO_ROOT / "backtest-cache-download" / "monthly",
    "daily_path":      _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather",
    "out_csv":         _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_5day_decile_leader_morning_reject_signature.csv",
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
# Cross-sectional rank precomputation
# -----------------------------------------------------------------------------
def build_decile_leader_index(
    daily_df: pd.DataFrame,
    sym_to_cap: Dict[str, str],
    cfg: Dict[str, object],
) -> Tuple[Dict[Tuple[str, date], int], Dict[Tuple[str, date], float], Dict[date, Dict[str, int]]]:
    """For each (sym, T-1) build a flag = is_top_decile within (cap_segment, T-1).

    Returns:
      leader_flag[(sym, T_minus_1_date)] -> 1/0
      cumret_value[(sym, T_minus_1_date)] -> 5-day return value (for splits)
      per_date_top_set[T-1] -> {cap_segment: set(top-decile syms)}  (for stability)
    """
    lookback = int(cfg["lookback_days"])
    accepted = set(cfg["accepted_caps"])
    decile_pct = float(cfg["decile_pct"])
    min_cohort = int(cfg["min_cohort_size"])

    # daily_df columns: ts (datetime64), open/high/low/close/volume, symbol
    # Use ts.date() as the calendar date
    daily_df = daily_df[["symbol", "ts", "close"]].copy()
    daily_df["d"] = daily_df["ts"].dt.date

    # Restrict universe to accepted caps (huge speedup)
    accepted_syms = {s for s, cap in sym_to_cap.items() if cap in accepted}
    daily_df = daily_df[daily_df["symbol"].isin(accepted_syms)]
    if daily_df.empty:
        return {}, {}, {}

    daily_df = daily_df.sort_values(["symbol", "d"]).reset_index(drop=True)

    # Build per-symbol close array indexed by trading-day position so we can
    # compute close[T-1] / close[T-6] - 1 in vectorized form per symbol.
    out_records: List[Tuple[str, date, float]] = []
    for sym, sym_df in daily_df.groupby("symbol", sort=False):
        if len(sym_df) <= lookback:
            continue
        closes = sym_df["close"].to_numpy(dtype=np.float64)
        dates = sym_df["d"].to_numpy()
        # 5-day cumulative return: close[i] / close[i - lookback] - 1
        # i ranges from lookback to len-1; the "T-1 date" associated is dates[i].
        if len(closes) <= lookback:
            continue
        ratio = closes[lookback:] / closes[:-lookback] - 1.0
        d_t_minus_1 = dates[lookback:]
        for r, dd in zip(ratio, d_t_minus_1):
            if not np.isfinite(r):
                continue
            out_records.append((sym, dd.item() if hasattr(dd, "item") else dd, float(r)))

    if not out_records:
        return {}, {}, {}

    rk = pd.DataFrame.from_records(out_records, columns=["symbol", "d", "cumret"])
    rk["cap_segment"] = rk["symbol"].map(sym_to_cap)
    rk = rk[rk["cap_segment"].isin(accepted)]

    # Per (cap_segment, d) quantile threshold
    grp = rk.groupby(["cap_segment", "d"], sort=False)
    # Compute per-group threshold and cohort size, then merge back
    cohort_size = grp["cumret"].size().rename("cohort_n")
    thresholds = grp["cumret"].quantile(decile_pct).rename("q_thresh")
    summary = pd.concat([cohort_size, thresholds], axis=1).reset_index()
    rk = rk.merge(summary, on=["cap_segment", "d"], how="left")

    # Only keep dates where cohort_n >= min_cohort_size (otherwise quantile is unreliable)
    rk = rk[rk["cohort_n"] >= min_cohort]

    rk["is_top"] = (rk["cumret"] >= rk["q_thresh"]).astype(int)

    leader_flag: Dict[Tuple[str, date], int] = {}
    cumret_value: Dict[Tuple[str, date], float] = {}
    for sym, dd, ist, ret in zip(rk["symbol"], rk["d"], rk["is_top"], rk["cumret"]):
        # dd already a python date object
        key = (sym, dd)
        leader_flag[key] = int(ist)
        cumret_value[key] = float(ret)

    # Per-date top-set by cap_segment for stability test
    per_date_top_set: Dict[date, Dict[str, set]] = {}
    top_only = rk[rk["is_top"] == 1]
    for (cap, dd), g in top_only.groupby(["cap_segment", "d"], sort=False):
        per_date_top_set.setdefault(dd, {})[cap] = set(g["symbol"].tolist())

    return leader_flag, cumret_value, per_date_top_set


# -----------------------------------------------------------------------------
# Falsifier #1: T-1 vs T-2 decile-leader stability (per cap_segment)
# -----------------------------------------------------------------------------
def compute_stability(
    per_date_top_set: Dict[date, Dict[str, set]],
    cfg: Dict[str, object],
) -> Dict[str, Tuple[float, int]]:
    """For each cap_segment, mean fraction of (T-1 top-decile) ∩ (T-2 top-decile) / |T-1 top|.

    Returns: {cap_segment: (mean_stability, n_date_pairs_evaluated)}
    """
    win_start: date = cfg["window_start"]  # type: ignore
    all_dates = sorted([d for d in per_date_top_set.keys() if d >= win_start])
    per_cap_stabilities: Dict[str, List[float]] = {"small_cap": [], "mid_cap": []}

    # For each date d, find the previous date in per_date_top_set (i.e. previous T-1 entry)
    # Note: per_date_top_set is keyed by T-1 dates; consecutive T-1 dates are typically
    # consecutive trading days (since we computed cumret for every trading day from lookback+).
    for i in range(1, len(all_dates)):
        d_t1 = all_dates[i]
        d_t2 = all_dates[i - 1]
        for cap in ("small_cap", "mid_cap"):
            top_t1 = per_date_top_set.get(d_t1, {}).get(cap, set())
            top_t2 = per_date_top_set.get(d_t2, {}).get(cap, set())
            if not top_t1:
                continue
            overlap = top_t1 & top_t2
            per_cap_stabilities[cap].append(len(overlap) / len(top_t1))

    out: Dict[str, Tuple[float, int]] = {}
    for cap, vals in per_cap_stabilities.items():
        if vals:
            out[cap] = (float(np.mean(vals)), len(vals))
        else:
            out[cap] = (float("nan"), 0)
    return out


# -----------------------------------------------------------------------------
# Per-(symbol, day) intraday evaluator
# -----------------------------------------------------------------------------
def evaluate_symbol_day(
    sym_bars: pd.DataFrame, sym: str, d: date, cfg: Dict[str, object]
) -> Optional[dict]:
    """For (sym, T+0) compute open_at_0915 + close_at_1030 + close_at_1330 + first-hour high.

    Returns a SINGLE record per (sym, date):
      - is_signal: True if close_at_1030 < open_at_0915 (first-hour failure)
      - is_baseline: True if close_at_1030 >= open_at_0915 (continued)
      - ret_to_1330: (close_at_1330 - close_at_1030) / close_at_1030 * 100
      - or_fail_proxy: True if (close_at_1030 < open_at_0915) AND (high_first_hr > open_at_0915 * (1+0.005))
      - None if any required bar is missing
    """
    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    if sym_bars.empty:
        return None

    open_t = cfg["open_bar_time"]
    sig_t = cfg["signal_bar_time"]
    tgt_t = cfg["target_bar_time"]
    or_buf_frac = float(cfg["or_fail_high_buf_frac"])

    # Locate the three reference bars
    open_rows = sym_bars[sym_bars["time"] == open_t]
    sig_rows = sym_bars[sym_bars["time"] == sig_t]
    tgt_rows = sym_bars[sym_bars["time"] == tgt_t]
    if open_rows.empty or sig_rows.empty or tgt_rows.empty:
        return None

    open_at_0915 = float(open_rows.iloc[0]["open"])
    close_at_1030 = float(sig_rows.iloc[0]["close"])
    close_at_1330 = float(tgt_rows.iloc[0]["close"])
    sig_ts = sig_rows.iloc[0]["date"]

    if open_at_0915 <= 0:
        return None

    # First-hour high: bars between 09:15 (inclusive) and 10:15 (inclusive of 10:10 5m bar
    # which closes at 10:15). We use 09:15 <= time <= 10:10 (since 5m bar @10:10 closes at 10:15).
    first_hr_mask = (sym_bars["time"] >= dtime(9, 15)) & (sym_bars["time"] <= dtime(10, 10))
    first_hr_bars = sym_bars[first_hr_mask]
    if first_hr_bars.empty:
        return None
    high_first_hr = float(first_hr_bars["high"].max())

    is_signal = close_at_1030 < open_at_0915
    is_baseline = not is_signal
    or_fail_proxy = bool(is_signal and (high_first_hr > open_at_0915 * (1.0 + or_buf_frac)))

    ret = (close_at_1330 - close_at_1030) / close_at_1030 * 100.0

    return {
        "symbol": sym,
        "date": d.isoformat(),
        "signal_bar_ts": pd.Timestamp(sig_ts).isoformat(),
        "open_at_0915": open_at_0915,
        "close_at_1030": close_at_1030,
        "close_at_1330": close_at_1330,
        "high_first_hr": high_first_hr,
        "is_signal": bool(is_signal),
        "is_baseline": bool(is_baseline),
        "or_fail_proxy": bool(or_fail_proxy),
        "ret_to_1330": ret,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature -- 5day_decile_leader_morning_reject_short")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print(f"Caps:   {sorted(CONFIG['accepted_caps'])}  (large_cap + unknown EXCLUDED)")
    print(f"Signal:    close_at_1030 < open_at_0915  (decile-leader cohort only)")
    print(f"Baseline:  close_at_1030 >= open_at_0915 (decile-leader cohort only)")
    print(f"Target:    ret_to_1330 = (close_1330 - close_1030)/close_1030 * 100   [SHORT]")
    print("=" * 80)

    # --- Production universe gate ---
    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )
    nse_all = gate._load_nse_all()
    sym_to_cap: Dict[str, str] = {s: row.cap_segment for s, row in nse_all.items()}

    # --- Build decile-leader index from consolidated_daily.feather ---
    _log("Loading consolidated_daily.feather ...")
    daily_df = pd.read_feather(CONFIG["daily_path"])
    daily_df["ts"] = pd.to_datetime(daily_df["ts"])
    # IST-naive (strip any tz that may exist)
    if pd.api.types.is_datetime64tz_dtype(daily_df["ts"]):
        daily_df["ts"] = daily_df["ts"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    _log("Computing 5-day cumulative returns and per-(cap_segment,date) decile thresholds ...")
    leader_flag, cumret_value, per_date_top_set = build_decile_leader_index(
        daily_df, sym_to_cap, CONFIG
    )
    _log(
        f"  decile-leader index entries: {len(leader_flag):,}   "
        f"(top-decile sym-days: {sum(leader_flag.values()):,})"
    )

    # --- Falsifier #1: T-1 vs T-2 stability ---
    _log("Computing Falsifier #1 (T-1 vs T-2 decile-rank stability) ...")
    stab = compute_stability(per_date_top_set, CONFIG)
    print()
    print("=" * 80)
    print("STEP 1 -- Falsifier #1: decile-rank STABILITY (T-1 vs T-2)")
    print("=" * 80)
    print(f"{'cap_segment':<14}{'mean_stability':>18}{'n_date_pairs':>16}")
    overall_pairs: List[float] = []
    for cap in ("small_cap", "mid_cap"):
        ms, npairs = stab.get(cap, (float("nan"), 0))
        print(f"{cap:<14}{ms:>18.4f}{npairs:>16d}")
        if not np.isnan(ms):
            overall_pairs.extend([ms] * npairs)
    overall_mean = float(np.mean(overall_pairs)) if overall_pairs else float("nan")
    print(f"{'OVERALL':<14}{overall_mean:>18.4f}{len(overall_pairs):>16d}")

    kill_thr = float(CONFIG["stability_kill_below"])
    pass_thr = float(CONFIG["stability_pass_at"])
    falsifier1_state: str
    if any((not np.isnan(stab[c][0])) and stab[c][0] < kill_thr for c in ("small_cap", "mid_cap")):
        falsifier1_state = "KILL"
    elif all((not np.isnan(stab[c][0])) and stab[c][0] >= pass_thr for c in ("small_cap", "mid_cap")):
        falsifier1_state = "PASS"
    else:
        falsifier1_state = "CAUTION"
    print(f"\n  FALSIFIER #1 STATE: {falsifier1_state}   "
          f"(kill if any cap < {kill_thr:.2f}; pass if both >= {pass_thr:.2f})")

    if falsifier1_state == "KILL":
        print("\n  STOP -- Falsifier #1 FAILED. Verdict: KILL (signal is rank-noise, not positioning saturation).")
        print("\n" + "=" * 80)
        print("VERDICT: KILL (Falsifier #1 failed -- decile-rank stability < 50%)")
        print("=" * 80)
        return 0

    # --- Iterate monthly 5m feathers, score decile-leader sym-days only ---
    _log("Scanning monthly 5m feathers for decile-leader sym-days only ...")
    records: List[dict] = []
    months = _months_between(CONFIG["window_start"], CONFIG["window_end"])

    total_eval_pairs = 0
    total_rejected_universe = 0
    total_non_leader = 0
    total_no_record = 0

    for (yy, mm) in months:
        path = Path(CONFIG["monthly_5m_dir"]) / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            _log(f"  skip {yy:04d}-{mm:02d} (no 5m feather)")
            continue

        df = pd.read_feather(
            path, columns=["symbol", "date", "open", "high", "low", "close", "volume"]
        )
        df["date"] = _ensure_naive_ist(df["date"])
        df["day"] = df["date"].dt.date
        df["time"] = df["date"].dt.time
        df = df[(df["day"] >= CONFIG["window_start"]) & (df["day"] <= CONFIG["window_end"])]
        if df.empty:
            continue

        # Pre-filter to accepted caps
        accepted_syms = {s for s, cap in sym_to_cap.items() if cap in CONFIG["accepted_caps"]}
        df = df[df["symbol"].isin(accepted_syms)]
        if df.empty:
            continue

        gb = df.groupby(["symbol", "day"], sort=False)
        month_eval = 0
        month_reject = 0
        month_non_leader = 0
        month_rec = 0

        for (sym, d), bars in gb:
            month_eval += 1
            total_eval_pairs += 1

            # Decile-leader check: leader flag is keyed by T-1 = day BEFORE the signal day.
            # Find the most recent T-1 trading date BEFORE d for which we have a leader entry.
            # Cheapest: lookup (sym, prev_trading_day). Here we approximate prev_trading_day
            # by stepping back 1..5 calendar days; the decile index only has trading dates.
            t_minus_1: Optional[date] = None
            for back in range(1, 6):
                candidate = date.fromordinal(d.toordinal() - back)
                if (sym, candidate) in leader_flag:
                    t_minus_1 = candidate
                    break
            if t_minus_1 is None:
                month_non_leader += 1
                total_non_leader += 1
                continue
            if leader_flag[(sym, t_minus_1)] != 1:
                month_non_leader += 1
                total_non_leader += 1
                continue

            # Universe filter (production-aligned, per-date)
            if not gate.is_eligible(sym, d):
                month_reject += 1
                total_rejected_universe += 1
                continue

            rec = evaluate_symbol_day(bars, sym, d, CONFIG)
            if rec is None:
                total_no_record += 1
                continue

            # Attach the 5-day cumret value for split analysis
            rec["cumret_5d_at_t_minus_1"] = float(cumret_value.get((sym, t_minus_1), float("nan")))
            rec["t_minus_1_date"] = t_minus_1.isoformat()
            records.append(rec)
            month_rec += 1

        _log(
            f"  {yy:04d}-{mm:02d}: pairs={month_eval:,} "
            f"non_leader={month_non_leader:,} rejected_univ={month_reject:,} records={month_rec:,}"
        )

    print()
    print("=== Pipeline tally ===")
    print(f"  total (sym, day) pairs in caps:           {total_eval_pairs:,}")
    print(f"  non-decile-leader (no T-1 top flag):      {total_non_leader:,}")
    print(f"  rejected by universe gate:                {total_rejected_universe:,}")
    print(f"  no record (missing bars):                 {total_no_record:,}")
    print(f"  recorded:                                 {len(records):,}")

    if not records:
        print("\nNO RECORDS COLLECTED -- abort.")
        return 1

    df_out = pd.DataFrame.from_records(records)
    df_out["cap_segment"] = df_out["symbol"].map(sym_to_cap).fillna("unknown")

    dt_dates = pd.to_datetime(df_out["date"])
    cut_2024 = pd.Timestamp(CONFIG["regime_2024_cut"])
    cut_sebi = pd.Timestamp(CONFIG["sebi_oct2025_cut"])
    df_out["regime_pre_post_2024"] = np.where(dt_dates < cut_2024, "pre", "post")
    df_out["period_pre_post_sebi_oct2025"] = np.where(dt_dates < cut_sebi, "pre", "post")

    def _cumret_bucket(r: float) -> str:
        if pd.isna(r):
            return "nan"
        pct = r * 100.0
        if pct < 5.0:
            return "<5%"
        if pct < 10.0:
            return "5-10%"
        if pct < 15.0:
            return "10-15%"
        return ">=15%"

    df_out["cumret_bucket"] = df_out["cumret_5d_at_t_minus_1"].map(_cumret_bucket)

    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out):,} rows -> {out_path}")

    sig = df_out[df_out["is_signal"]].copy()
    base = df_out[df_out["is_baseline"]].copy()
    n_sig = len(sig)
    n_base = len(base)

    # -------------------------------------------------------------------------
    # STEP 2 -- Drift delta
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

    print(f"  n_signal:                   {n_sig:,}")
    print(f"  n_baseline:                 {n_base:,}")
    print(f"  Signal   mean ret_to_1330:  {sig_mean:+.4f}%  (median {sig_median:+.4f}%)")
    print(f"  Baseline mean ret_to_1330:  {base_mean:+.4f}%  (median {base_median:+.4f}%)")
    print(f"  DRIFT DELTA:                {delta:+.4f}%   [<= {CONFIG['drift_delta_max']:.2f}% required]")

    # -------------------------------------------------------------------------
    # STEP 3 -- Cohort splits
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 3 -- Cohort splits")
    print("=" * 80)
    print(f"{'split':<32}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")

    def _split_block(label: str, mask_sig: pd.Series, mask_base: pd.Series) -> Tuple[int, float, float, float]:
        s = sig[mask_sig]
        b = base[mask_base]
        ns, nb = len(s), len(b)
        sm = float(s["ret_to_1330"].mean()) if ns else float("nan")
        bm = float(b["ret_to_1330"].mean()) if nb else float("nan")
        dl = sm - bm if (ns and nb) else float("nan")
        print(
            f"{label:<32}"
            f"{ns:>8d}"
            f"{nb:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )
        return ns, sm, bm, dl

    sig_dates = pd.to_datetime(sig["date"])
    base_dates = pd.to_datetime(base["date"])

    _split_block("pre_2024",            sig_dates < cut_2024,  base_dates < cut_2024)
    _split_block("post_2024",           sig_dates >= cut_2024, base_dates >= cut_2024)
    _split_block("pre_sebi_oct2025",    sig_dates < cut_sebi,  base_dates < cut_sebi)
    post_sebi_n, post_sebi_sm, post_sebi_bm, post_sebi_delta = _split_block(
        "post_sebi_oct2025", sig_dates >= cut_sebi, base_dates >= cut_sebi
    )

    for cap_val in ("small_cap", "mid_cap"):
        _split_block(f"cap={cap_val}", sig["cap_segment"] == cap_val, base["cap_segment"] == cap_val)

    print()
    print("  --- 5-day cumret magnitude buckets ---")
    print(f"{'bucket':<32}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    for bucket in ("5-10%", "10-15%", ">=15%"):
        _split_block(
            f"cumret_bucket={bucket}",
            sig["cumret_bucket"] == bucket,
            base["cumret_bucket"] == bucket,
        )

    # OR-overlap diagnostic (signal-only)
    print()
    print("  --- OR-fail proxy overlap (signal rows only) ---")
    if n_sig:
        n_or_overlap = int(sig["or_fail_proxy"].sum())
        frac_or_overlap = n_or_overlap / n_sig
        # Signal-mean within the OR-overlap vs non-overlap
        s_or = sig[sig["or_fail_proxy"]]
        s_nor = sig[~sig["or_fail_proxy"]]
        sm_or = float(s_or["ret_to_1330"].mean()) if len(s_or) else float("nan")
        sm_nor = float(s_nor["ret_to_1330"].mean()) if len(s_nor) else float("nan")
        print(f"  OR-overlap signal rows:        {n_or_overlap:,} / {n_sig:,}  ({frac_or_overlap*100:.2f}%)")
        print(f"  OR-overlap     signal_mean:    {sm_or:+.4f}%   (n={len(s_or):,})")
        print(f"  Non-overlap    signal_mean:    {sm_nor:+.4f}%   (n={len(s_nor):,})")
    else:
        print("  n_signal = 0; skip OR-overlap diagnostic.")

    # -------------------------------------------------------------------------
    # VERDICT
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    drift_ok = (not np.isnan(delta)) and (delta <= float(CONFIG["drift_delta_max"]))
    n_ok = n_sig >= int(CONFIG["n_signal_min"])
    sign_flip = (not np.isnan(post_sebi_delta)) and (post_sebi_delta > 0) and (post_sebi_n >= 50)

    verdict: str
    reason: str
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
    elif falsifier1_state == "CAUTION":
        verdict = "PROCEED-WITH-CAUTION to Phase 3"
        reason = (
            f"Falsifier #1 marginal (50-60%); drift {delta:+.4f}% <= {CONFIG['drift_delta_max']:.2f}%, "
            f"n_signal {n_sig} >= {CONFIG['n_signal_min']}"
        )
    else:
        verdict = "PROCEED to Phase 3"
        reason = (
            f"Falsifier #1 PASS, drift {delta:+.4f}% <= {CONFIG['drift_delta_max']:.2f}%, "
            f"n_signal {n_sig} >= {CONFIG['n_signal_min']}, no post-SEBI sign flip"
        )

    print(f"  {verdict}")
    print(f"  reason: {reason}")
    print(f"  falsifier1_state: {falsifier1_state}")
    print(f"  small_cap stability: {stab['small_cap'][0]:.4f} (n_pairs={stab['small_cap'][1]})")
    print(f"  mid_cap   stability: {stab['mid_cap'][0]:.4f} (n_pairs={stab['mid_cap'][1]})")
    return 0


if __name__ == "__main__":
    sys.exit(run())
