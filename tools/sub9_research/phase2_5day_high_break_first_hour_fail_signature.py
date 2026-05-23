# tools/sub9_research/phase2_5day_high_break_first_hour_fail_signature.py
#
# Phase 2 empirical signature for 5day_high_break_first_hour_fail_short candidate.
# See specs/2026-05-22-brief-5day_high_break_first_hour_fail_short.md
#
# Anti-bias guards (Lesson #5):
#   1. 5-day high computed from daily T-5..T-1 ONLY (no T0 leak)
#   2. intraday_high_prior computed from bars[:i] only -- no look-ahead
#   3. vol_baseline excludes current bar (mean of bars[:i].volume)
#   4. First-fire-per-day latch (sym, date) -- only first qualifying break
#   5. ProductionUniverseGate per-date (Lesson #19)
#   6. large_cap AND unknown EXCLUDED from universe (C-H carry-over)
#
# Pre-registration discipline:
#   - Falsifier #1: break-bar vol_ratio MEDIAN >= 1.5 expected (FOMO)
#     Pre-reg evaluated BEFORE drift-delta computation
#   - Control comparison: intraday-high variant ALWAYS computed and reported
#     (Phase 1 caveat: Indian sources operationalize INTRADAY-high failure)
"""Phase 2 empirical signature -- 5day_high_break_first_hour_fail_short."""
from __future__ import annotations

import sys
from datetime import date, datetime, time as dtime, timedelta
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

    # Universe (cell-lock from brief)
    "accepted_caps": {"small_cap", "mid_cap"},  # large_cap + unknown EXCLUDED (C-H + lunch_lull carry-over)
    "require_mis":   True,

    # Multi-day filter (5-day high momentum gate)
    "lookback_days":          5,            # 5-day window T-5..T-1
    "min_5day_high_ratio":    1.02,         # 5day_high / 5day_T-1_close >= 1.02

    # Signal-scan window (09:30-11:30, first 2 hours after open)
    "sig_window_start":  dtime(9, 30),     # 09:30 inclusive
    "sig_window_end":    dtime(11, 30),    # 11:30 INCLUSIVE -- last considered bar is 11:30

    # Fail tracking (look-ahead within signal-day, NOT into target window)
    "fail_lookahead_bars":   6,             # next 6 bars (30 min)
    "fail_close_threshold":  0.999,         # close < 5day_high * 0.999 -> FAIL bar
    "max_break_bar_time":    dtime(11, 30), # break must occur on or before 11:30

    # Target bar -- close_at_1325 == 13:25 5m bar close
    "target_bar_time":   dtime(13, 25),

    # OR-high overlap (for telemetry)
    "or_bars":           3,                 # ORH = max(high) of first 3 bars (09:15-09:30)

    # Regime / period cuts
    "regime_2024_cut":   date(2024, 1, 1),
    "sebi_oct2025_cut":  date(2025, 10, 1),

    # Acceptance gates (brief)
    "drift_delta_max":          -0.15,      # signal_mean - baseline_mean <= -0.15%
    "n_signal_min":             200,
    "vol_ratio_median_min":     1.5,        # Falsifier #1: break-bar vol_ratio median >= 1.5
    "vol_ratio_lt_1p2_frac_max": 0.40,      # Falsifier #1: <40% of break-bar fires at vol_ratio<1.2

    # Paths
    "monthly_5m_dir":  _REPO_ROOT / "backtest-cache-download" / "monthly",
    "daily_path":      _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather",
    "out_csv":         _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_5day_high_break_first_hour_fail_signature.csv",
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
# Daily prior-window lookup
# -----------------------------------------------------------------------------
def build_daily_lookup(daily_path: Path) -> Dict[str, pd.DataFrame]:
    """Load consolidated_daily and group by symbol. Each value: sorted df with cols [d, high, close]."""
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    df["d"] = df["ts"].dt.date
    df = df[["symbol", "d", "high", "close"]].copy()
    out: Dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("d").reset_index(drop=True)
        out[sym] = g
    return out


def get_prior_window(daily_df: pd.DataFrame, signal_date: date, lookback_days: int) -> Optional[pd.DataFrame]:
    """Return the LAST `lookback_days` daily rows strictly BEFORE signal_date.

    Returns None if fewer than lookback_days rows are available.
    """
    prior = daily_df[daily_df["d"] < signal_date]
    if len(prior) < lookback_days:
        return None
    return prior.tail(lookback_days).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Per-(symbol, day) evaluator
# -----------------------------------------------------------------------------
def evaluate_symbol_day(
    sym_bars: pd.DataFrame,
    sym: str,
    d: date,
    five_day_high: float,
    intraday_high_prior_today: Optional[float],  # max of bars BEFORE sig window starts (for control)
) -> Optional[dict]:
    """Find FIRST bar in [09:30, 11:30] where high breaks the level.

    Returns:
      - 'signal'   if break in window AND any of next 6 bars closes < level * 0.999
      - 'baseline' if break in window AND NO fail within next 30 min
      - None       if no break in window OR target bar missing

    Anti-bias guards:
      - 5-day high passed in (already T-5..T-1 only)
      - vol_baseline = mean(bars[:i].volume) excludes current bar
      - first-fire-per-day -- after first qualifying break, stop scanning

    NOTE: We evaluate TWO variants (5day-high level vs intraday-high-control level).
    Caller dispatches via `which_level` outside.
    """
    pass  # placeholder; per-variant logic inlined below


def scan_variant(
    sym_bars: pd.DataFrame,
    sym: str,
    d: date,
    level: float,
    variant_label: str,
) -> Optional[dict]:
    """Scan sym_bars for a given resistance `level`.

    Returns a record dict or None.
    """
    if level is None or not np.isfinite(level) or level <= 0:
        return None

    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    if sym_bars.empty:
        return None

    # Confine to the regular trading session
    mask_sess = (sym_bars["time"] >= dtime(9, 15)) & (sym_bars["time"] <= dtime(15, 25))
    sym_bars = sym_bars[mask_sess].reset_index(drop=True)
    if sym_bars.empty:
        return None

    # Target bar -- 13:25 5m bar close
    tgt_t = CONFIG["target_bar_time"]
    tgt_rows = sym_bars[sym_bars["time"] == tgt_t]
    if tgt_rows.empty:
        return None
    target_close = float(tgt_rows.iloc[0]["close"])

    # Per-bar arrays
    highs = sym_bars["high"].to_numpy(dtype=np.float64)
    closes = sym_bars["close"].to_numpy(dtype=np.float64)
    vols = sym_bars["volume"].to_numpy(dtype=np.float64)
    times = sym_bars["time"].tolist()
    ts_col = sym_bars["date"].tolist()

    n = len(highs)
    if n < 2:
        return None

    # vol_baseline[i] = mean(vols[0..i-1]) -- excludes current bar
    cum_vol = np.cumsum(vols)
    vol_baseline = np.full(n, np.nan, dtype=np.float64)
    idx_arr = np.arange(1, n)
    vol_baseline[1:] = cum_vol[:-1] / idx_arr

    sw_start = CONFIG["sig_window_start"]
    sw_end_inclusive = CONFIG["sig_window_end"]
    max_break_t = CONFIG["max_break_bar_time"]
    fail_lookahead = int(CONFIG["fail_lookahead_bars"])
    fail_close_thresh = float(CONFIG["fail_close_threshold"]) * level

    # OR-high (for telemetry overlap)
    or_n = int(CONFIG["or_bars"])
    or_high_val: Optional[float] = None
    # find first `or_bars` bars between 09:15 and 09:30 exclusive
    or_mask = (sym_bars["time"] >= dtime(9, 15)) & (sym_bars["time"] < dtime(9, 30))
    or_bars_df = sym_bars[or_mask].head(or_n)
    if len(or_bars_df) >= 1:
        or_high_val = float(or_bars_df["high"].max())

    # First-fire scan: find FIRST bar in [09:30, 11:30] where high > level
    for i in range(n):
        t = times[i]
        if t < sw_start or t > sw_end_inclusive:
            continue
        if t > max_break_t:
            break
        if np.isnan(vol_baseline[i]) or vol_baseline[i] <= 0:
            continue
        if not (highs[i] > level):
            continue

        # Break detected. Compute break-bar vol_ratio (telemetry/falsifier #1)
        break_vr = float(vols[i] / vol_baseline[i])

        # Look ahead up to fail_lookahead bars (within sym_bars order)
        fail_idx = -1
        fail_vr = float("nan")
        end_idx = min(i + fail_lookahead, n - 1)
        for j in range(i + 1, end_idx + 1):
            if closes[j] < fail_close_thresh:
                fail_idx = j
                if vol_baseline[j] > 0 and not np.isnan(vol_baseline[j]):
                    fail_vr = float(vols[j] / vol_baseline[j])
                break

        is_signal = fail_idx > 0
        # Choose signal-anchor bar
        if is_signal:
            anchor_idx = fail_idx
        else:
            # Baseline anchor: min(break_idx + 6, last_bar_in_1130)
            anchor_idx_a = i + fail_lookahead
            # find last bar with time <= 11:30
            last_in_window_idx = -1
            for k in range(n):
                if times[k] <= sw_end_inclusive:
                    last_in_window_idx = k
                else:
                    break
            anchor_idx = min(anchor_idx_a, last_in_window_idx if last_in_window_idx >= 0 else n - 1)
            anchor_idx = min(anchor_idx, n - 1)

        anchor_close = float(closes[anchor_idx])
        ret = (target_close - anchor_close) / anchor_close * 100.0

        # OR-high overlap: did break bar break OR-high too?
        or_overlap = False
        if or_high_val is not None and np.isfinite(or_high_val) and or_high_val > 0:
            # signal "coincides with OR high break" if the level being broken is at/above OR-high
            # OR if the break bar's high also exceeds OR-high (commonly true for any high after 09:30)
            # Per brief: "signal coincides with OR-high break". Define as: break bar high > or_high_val
            or_overlap = bool(highs[i] > or_high_val)

        return {
            "variant": variant_label,
            "symbol": sym,
            "date": d.isoformat(),
            "break_bar_ts": pd.Timestamp(ts_col[i]).isoformat(),
            "break_bar_time": str(times[i]),
            "break_bar_close": float(closes[i]),
            "break_bar_high": float(highs[i]),
            "break_bar_vol_ratio": break_vr,
            "level": float(level),
            "fail_bar_ts": pd.Timestamp(ts_col[fail_idx]).isoformat() if fail_idx > 0 else "",
            "fail_bar_time": str(times[fail_idx]) if fail_idx > 0 else "",
            "fail_bar_vol_ratio": fail_vr,
            "anchor_idx_offset": int(anchor_idx - i),
            "anchor_close": anchor_close,
            "ret_to_1325": ret,
            "target_close_1325": target_close,
            "or_high": float(or_high_val) if or_high_val is not None else float("nan"),
            "or_overlap": bool(or_overlap),
            "is_signal": bool(is_signal),
            "is_baseline": bool(not is_signal),
        }

    # No break in window
    return None


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature -- 5day_high_break_first_hour_fail_short")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print(f"Caps:   {sorted(CONFIG['accepted_caps'])}  (large_cap + unknown EXCLUDED)")
    print(f"Multi-day filter: 5day_high / 5day_T-1_close >= {CONFIG['min_5day_high_ratio']}")
    print(f"Signal scan:      [{CONFIG['sig_window_start']}, {CONFIG['sig_window_end']}]")
    print(f"Fail look-ahead:  {CONFIG['fail_lookahead_bars']} bars (30 min)")
    print(f"Fail threshold:   close < level * {CONFIG['fail_close_threshold']}")
    print(f"Target bar:       {CONFIG['target_bar_time']} (13:25 5m close)")
    print(f"Control variant:  intraday-high-so-far (max of bars before sig_window_start)")
    print("=" * 80)

    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )

    _log("loading consolidated_daily ...")
    daily_lookup = build_daily_lookup(Path(CONFIG["daily_path"]))
    _log(f"  daily symbols: {len(daily_lookup):,}")

    records: List[dict] = []
    months = _months_between(CONFIG["window_start"], CONFIG["window_end"])
    total_eval_pairs = 0
    total_rejected_universe = 0
    total_no_daily_history = 0
    total_no_momentum = 0
    total_no_break = 0
    total_records_5d = 0
    total_records_ih = 0

    lookback = int(CONFIG["lookback_days"])
    min_ratio = float(CONFIG["min_5day_high_ratio"])
    sw_start = CONFIG["sig_window_start"]

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

        # Per-month: pre-filter symbols to accepted caps (cheap)
        nse_all = gate._load_nse_all()
        accepted = CONFIG["accepted_caps"]
        keep_syms = {s for s, row in nse_all.items() if row.cap_segment in accepted}
        df = df[df["symbol"].isin(keep_syms)]
        if df.empty:
            continue

        gb = df.groupby(["symbol", "day"], sort=False)
        month_eval = 0
        month_reject = 0
        month_no_hist = 0
        month_no_mom = 0
        month_no_break = 0
        month_rec_5d = 0
        month_rec_ih = 0

        for (sym, d), bars in gb:
            month_eval += 1
            total_eval_pairs += 1
            if not gate.is_eligible(sym, d):
                month_reject += 1
                total_rejected_universe += 1
                continue

            # 5-day daily prior window
            ddf = daily_lookup.get(sym)
            if ddf is None:
                month_no_hist += 1
                total_no_daily_history += 1
                continue
            prior_window = get_prior_window(ddf, d, lookback)
            if prior_window is None:
                month_no_hist += 1
                total_no_daily_history += 1
                continue

            five_day_high = float(prior_window["high"].max())
            five_day_tm1_close = float(prior_window.iloc[-1]["close"])
            if five_day_tm1_close <= 0 or not np.isfinite(five_day_tm1_close):
                month_no_mom += 1
                total_no_momentum += 1
                continue

            ratio = five_day_high / five_day_tm1_close
            if ratio < min_ratio:
                month_no_mom += 1
                total_no_momentum += 1
                continue

            # 5-day-high variant
            rec_5d = scan_variant(bars, sym, d, five_day_high, "5day_high")
            # intraday-high-control variant: level = max(high) of bars BEFORE 09:30
            sb_pre = bars[bars["date"].dt.time < sw_start]
            ih_level = float(sb_pre["high"].max()) if not sb_pre.empty else float("nan")
            rec_ih = scan_variant(bars, sym, d, ih_level, "intraday_high") if np.isfinite(ih_level) else None

            had_any = False
            if rec_5d is not None:
                rec_5d["five_day_high"] = five_day_high
                rec_5d["five_day_tm1_close"] = five_day_tm1_close
                rec_5d["five_day_ratio"] = ratio
                rec_5d["intraday_high_level"] = ih_level
                records.append(rec_5d)
                month_rec_5d += 1
                total_records_5d += 1
                had_any = True
            if rec_ih is not None:
                rec_ih["five_day_high"] = five_day_high
                rec_ih["five_day_tm1_close"] = five_day_tm1_close
                rec_ih["five_day_ratio"] = ratio
                rec_ih["intraday_high_level"] = ih_level
                records.append(rec_ih)
                month_rec_ih += 1
                total_records_ih += 1
                had_any = True

            if not had_any:
                month_no_break += 1
                total_no_break += 1

        _log(
            f"  {yy:04d}-{mm:02d}: pairs={month_eval:,} "
            f"rej_univ={month_reject:,} no_hist={month_no_hist:,} "
            f"no_mom={month_no_mom:,} no_break={month_no_break:,} "
            f"rec_5d={month_rec_5d:,} rec_ih={month_rec_ih:,}"
        )

    print()
    print("=== Pipeline tally ===")
    print(f"  total (sym, day) pairs in caps:      {total_eval_pairs:,}")
    print(f"  rejected by universe gate:           {total_rejected_universe:,}")
    print(f"  insufficient daily history:          {total_no_daily_history:,}")
    print(f"  failed 5-day momentum filter:        {total_no_momentum:,}")
    print(f"  no break in [09:30, 11:30]:          {total_no_break:,}")
    print(f"  recorded (5day_high variant):        {total_records_5d:,}")
    print(f"  recorded (intraday_high control):    {total_records_ih:,}")

    if not records:
        print("\nNO RECORDS COLLECTED -- abort.")
        return 1

    df_out = pd.DataFrame.from_records(records)

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

    # vol_ratio buckets (break-bar vol_ratio)
    def _vr_bucket(v: float) -> str:
        if pd.isna(v):
            return "nan"
        if v < 1.0:
            return "<1.0"
        if v < 1.5:
            return "1.0-1.5"
        if v < 2.0:
            return "1.5-2.0"
        return ">=2.0"

    df_out["break_vol_ratio_bucket"] = df_out["break_bar_vol_ratio"].map(_vr_bucket)

    # Save CSV
    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out):,} rows -> {out_path}")

    # Split by variant
    df_5d = df_out[df_out["variant"] == "5day_high"].copy()
    df_ih = df_out[df_out["variant"] == "intraday_high"].copy()

    sig_5d = df_5d[df_5d["is_signal"]].copy()
    base_5d = df_5d[df_5d["is_baseline"]].copy()
    n_sig_5d = len(sig_5d)
    n_base_5d = len(base_5d)

    sig_ih = df_ih[df_ih["is_signal"]].copy()
    base_ih = df_ih[df_ih["is_baseline"]].copy()
    n_sig_ih = len(sig_ih)
    n_base_ih = len(base_ih)

    # -------------------------------------------------------------------------
    # STEP 1 -- Vol-ratio pre-registration (Falsifier #1) on 5day_high variant
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 1 -- Vol-ratio pre-registration (Falsifier #1)  [5day_high variant]")
    print("=" * 80)
    if n_sig_5d == 0:
        print("  n_signal_5d = 0 -- cannot evaluate Falsifier #1. Verdict: KILL (no signal events).")
        return 1

    vr_break = sig_5d["break_bar_vol_ratio"].astype(float)
    q25, q50, q75 = vr_break.quantile([0.25, 0.50, 0.75])
    median_vr = float(q50)
    frac_lt_1p2 = float((vr_break < 1.2).mean())

    print(f"  n_signal_5d:                    {n_sig_5d:,}")
    print(f"  break-bar vol_ratio MEDIAN:     {median_vr:.4f}   [>= {CONFIG['vol_ratio_median_min']} required]")
    print(f"  break-bar vol_ratio 25/50/75:   {float(q25):.4f} / {float(q50):.4f} / {float(q75):.4f}")
    print(f"  Fraction vol_ratio < 1.2:       {frac_lt_1p2*100:.2f}%   [< {CONFIG['vol_ratio_lt_1p2_frac_max']*100:.0f}% required]")

    # Per brief: break-bar median > 1.5 AND fail-bar median >= break-bar median
    fail_vrs = sig_5d["fail_bar_vol_ratio"].astype(float).dropna()
    if len(fail_vrs) > 0:
        fail_median = float(fail_vrs.median())
    else:
        fail_median = float("nan")
    print(f"  fail-bar vol_ratio MEDIAN:      {fail_median:.4f}   [>= break-bar median for confirmation]")

    falsifier1_pass = (
        median_vr >= float(CONFIG["vol_ratio_median_min"]) and
        frac_lt_1p2 < float(CONFIG["vol_ratio_lt_1p2_frac_max"])
    )
    fail_geq_break = (not np.isnan(fail_median)) and (fail_median >= median_vr)
    print(f"\n  FALSIFIER #1 (vol_signature): {'PASS' if falsifier1_pass else 'FAIL'}")
    print(f"  Confirmation (fail_vr >= break_vr): {'PASS' if fail_geq_break else 'FAIL'}")

    if not falsifier1_pass:
        print("\n  STOP -- Falsifier #1 FAILED. Continuing with drift report for transparency.")

    # -------------------------------------------------------------------------
    # STEP 2 -- Drift delta (5day_high variant)
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 2 -- Drift delta  [5day_high variant]")
    print("=" * 80)

    sig_mean_5d = float(sig_5d["ret_to_1325"].mean()) if n_sig_5d else float("nan")
    base_mean_5d = float(base_5d["ret_to_1325"].mean()) if n_base_5d else float("nan")
    sig_med_5d = float(sig_5d["ret_to_1325"].median()) if n_sig_5d else float("nan")
    base_med_5d = float(base_5d["ret_to_1325"].median()) if n_base_5d else float("nan")
    delta_5d = sig_mean_5d - base_mean_5d if (n_sig_5d and n_base_5d) else float("nan")

    print(f"  n_signal:                   {n_sig_5d:,}")
    print(f"  n_baseline:                 {n_base_5d:,}")
    print(f"  Signal   mean ret_to_1325:  {sig_mean_5d:+.4f}%  (median {sig_med_5d:+.4f}%)")
    print(f"  Baseline mean ret_to_1325:  {base_mean_5d:+.4f}%  (median {base_med_5d:+.4f}%)")
    print(f"  DRIFT DELTA:                {delta_5d:+.4f}%   [<= {CONFIG['drift_delta_max']:.2f}% required]")

    # -------------------------------------------------------------------------
    # STEP 3 -- Cohort splits (5day_high variant)
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 3 -- Cohort splits  [5day_high variant]")
    print("=" * 80)
    print(f"{'split':<36}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")

    def _split_block(label: str, mask_sig: pd.Series, mask_base: pd.Series) -> Tuple[int, float, float, float]:
        s = sig_5d[mask_sig]
        b = base_5d[mask_base]
        ns, nb = len(s), len(b)
        sm = float(s["ret_to_1325"].mean()) if ns else float("nan")
        bm = float(b["ret_to_1325"].mean()) if nb else float("nan")
        dl = sm - bm if (ns and nb) else float("nan")
        print(
            f"{label:<36}"
            f"{ns:>8d}"
            f"{nb:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )
        return ns, sm, bm, dl

    sig_dates = pd.to_datetime(sig_5d["date"])
    base_dates = pd.to_datetime(base_5d["date"])

    _split_block("pre_2024",            sig_dates < cut_2024,  base_dates < cut_2024)
    _split_block("post_2024",           sig_dates >= cut_2024, base_dates >= cut_2024)
    _split_block("pre_sebi_oct2025",    sig_dates < cut_sebi,  base_dates < cut_sebi)
    post_sebi_n, post_sebi_sm, post_sebi_bm, post_sebi_delta = _split_block(
        "post_sebi_oct2025", sig_dates >= cut_sebi, base_dates >= cut_sebi
    )

    for cap_val in ["small_cap", "mid_cap"]:
        _split_block(f"cap={cap_val}", sig_5d["cap_segment"] == cap_val, base_5d["cap_segment"] == cap_val)

    # vol_ratio buckets (break-bar)
    print()
    print("  --- break-bar vol_ratio buckets (signal rows; baseline_mean from aggregate) ---")
    print(f"{'bucket':<36}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    bucket_stats: List[Tuple[str, int, float]] = []
    for bucket in ["<1.0", "1.0-1.5", "1.5-2.0", ">=2.0"]:
        mask_s = sig_5d["break_vol_ratio_bucket"] == bucket
        s = sig_5d[mask_s]
        ns = len(s)
        sm = float(s["ret_to_1325"].mean()) if ns else float("nan")
        bm = base_mean_5d  # aggregate baseline
        dl = sm - bm if (ns and not np.isnan(bm)) else float("nan")
        bucket_stats.append((bucket, ns, sm))
        print(
            f"break_vr={bucket:<28}"
            f"{ns:>8d}"
            f"{n_base_5d:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )

    # OR-high overlap
    print()
    print("  --- OR-high overlap (signal coincides with OR-high break or not) ---")
    print(f"{'overlap':<36}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    for ov_label, ov_val in [("or_overlap=True", True), ("or_overlap=False", False)]:
        _split_block(ov_label, sig_5d["or_overlap"] == ov_val, base_5d["or_overlap"] == ov_val)

    or_overlap_frac = float(sig_5d["or_overlap"].mean()) if n_sig_5d else float("nan")
    print(f"\n  OR-high overlap fraction (signal): {or_overlap_frac*100:.2f}%")
    if or_overlap_frac > 0.5:
        print("  WARNING: >50% of signals coincide with OR-high break -- mechanism partially derivative of or_window_failure_fade_short.")

    # -------------------------------------------------------------------------
    # STEP 4 -- Control comparison (intraday_high variant vs 5day_high variant)
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 4 -- Control comparison (intraday-high vs 5-day-high)")
    print("=" * 80)
    if n_sig_ih == 0 or n_base_ih == 0:
        print(f"  Control variant: n_sig={n_sig_ih}, n_base={n_base_ih} -- insufficient.")
        ih_delta = float("nan")
        control_stronger = False
    else:
        sig_mean_ih = float(sig_ih["ret_to_1325"].mean())
        base_mean_ih = float(base_ih["ret_to_1325"].mean())
        ih_delta = sig_mean_ih - base_mean_ih
        print(f"  [5day_high] n_sig={n_sig_5d:,}  n_base={n_base_5d:,}  sig_mean={sig_mean_5d:+.4f}%  base_mean={base_mean_5d:+.4f}%  DRIFT={delta_5d:+.4f}%")
        print(f"  [intraday_high control] n_sig={n_sig_ih:,}  n_base={n_base_ih:,}  sig_mean={sig_mean_ih:+.4f}%  base_mean={base_mean_ih:+.4f}%  DRIFT={ih_delta:+.4f}%")
        # "stronger" for SHORT = more negative drift
        if (not np.isnan(delta_5d)) and (not np.isnan(ih_delta)):
            control_stronger = ih_delta < delta_5d
            if control_stronger:
                print(f"\n  CONTROL STRONGER: intraday-high drift ({ih_delta:+.4f}%) more negative than 5-day-high drift ({delta_5d:+.4f}%).")
                print("  => 5-day extension UNSUPPORTED by Phase 2 evidence; the intraday-trigger does the work.")
            else:
                print(f"\n  5-DAY-HIGH STRONGER OR EQUAL: drift {delta_5d:+.4f}% vs control {ih_delta:+.4f}%.")
                print("  => 5-day multi-day filter is additive/at-parity vs intraday-high baseline.")
        else:
            control_stronger = False

    # -------------------------------------------------------------------------
    # VERDICT
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    drift_ok = (not np.isnan(delta_5d)) and (delta_5d <= float(CONFIG["drift_delta_max"]))
    n_ok = n_sig_5d >= int(CONFIG["n_signal_min"])
    sign_flip = (not np.isnan(post_sebi_delta)) and (post_sebi_delta > 0) and (post_sebi_n >= 50)

    verdict: str
    reasons: List[str] = []
    if not n_ok:
        verdict = "KILL"
        reasons.append(f"n_signal {n_sig_5d} < {CONFIG['n_signal_min']}")
    elif not falsifier1_pass:
        verdict = "KILL"
        reasons.append(
            f"Falsifier #1 FAIL: break-bar vol_ratio median {median_vr:.3f} < {CONFIG['vol_ratio_median_min']} "
            f"OR frac<1.2 = {frac_lt_1p2*100:.1f}% >= {CONFIG['vol_ratio_lt_1p2_frac_max']*100:.0f}%"
        )
    elif not drift_ok:
        verdict = "KILL"
        reasons.append(
            f"drift delta {delta_5d:+.4f}% > {CONFIG['drift_delta_max']:.2f}% (insufficient SHORT footprint)"
        )
    elif control_stronger:
        verdict = "KILL"
        reasons.append(
            f"intraday-high control drift {ih_delta:+.4f}% STRONGER than 5-day variant {delta_5d:+.4f}% "
            f"-- 5-day extension unsupported (Phase 1 caveat materialized)"
        )
    elif sign_flip:
        verdict = "DEFER"
        reasons.append(
            f"post-SEBI-Oct-2025 cohort delta is POSITIVE "
            f"({post_sebi_delta:+.4f}%, n_sig={post_sebi_n}) -- regulatory regime risk"
        )
    else:
        verdict = "PROCEED to Phase 3"
        reasons.append(
            f"Falsifier #1 PASS, drift {delta_5d:+.4f}% <= {CONFIG['drift_delta_max']:.2f}%, "
            f"n_signal {n_sig_5d} >= {CONFIG['n_signal_min']}, 5-day variant >= intraday-high control"
        )

    print(f"  {verdict}")
    for r in reasons:
        print(f"  reason: {r}")

    # Monotonicity note
    sm_seq = [sm for (_, ns, sm) in bucket_stats if ns > 0]
    note_lines: List[str] = []
    if len(sm_seq) >= 2:
        # bucket order: <1.0, 1.0-1.5, 1.5-2.0, >=2.0  (low to high)
        # Thesis: higher break-bar volume (FOMO) => stronger fade. So sm_seq should DECREASE.
        decreasing = all(sm_seq[i] >= sm_seq[i+1] for i in range(len(sm_seq)-1))
        increasing = all(sm_seq[i] <= sm_seq[i+1] for i in range(len(sm_seq)-1))
        if decreasing:
            note_lines.append(
                "  MONOTONICITY: break_vol_ratio bucket signal_mean monotonically DECREASES from <1.0 -> >=2.0 "
                "(more negative as volume rises). Mechanism confirmation (higher FOMO -> stronger fade)."
            )
        elif increasing:
            note_lines.append(
                "  MONOTONICITY: signal_mean INCREASES with break_vol_ratio (less negative as vol RISES). "
                "Mechanism INVERTED -- volume signature does not support thesis. CAUTION."
            )
        else:
            note_lines.append(
                "  MONOTONICITY: signal_mean NON-MONOTONIC across break_vol_ratio buckets."
            )

    if note_lines:
        print()
        print("Notes:")
        for ln in note_lines:
            print(ln)

    return 0


if __name__ == "__main__":
    sys.exit(run())
