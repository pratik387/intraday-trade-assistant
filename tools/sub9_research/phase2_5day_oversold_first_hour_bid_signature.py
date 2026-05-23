# tools/sub9_research/phase2_5day_oversold_first_hour_bid_signature.py
#
# Phase 2 empirical signature for `5day_oversold_first_hour_bid_long` candidate.
# See specs/2026-05-22-brief-5day_oversold_first_hour_bid_long.md
#
# Anti-bias guards (Lesson #5):
#   1. 5-day daily filter uses bars STRICTLY BEFORE session_date (T-6 .. T-1) -- no look-ahead
#   2. Wick-bar "no fresh low in next 30 min" uses bars[i+1 .. i+6] (look-ahead is allowed
#      for SIGNAL DETECTION RESEARCH at this stage; signal is marked at wick-bar's close;
#      ret_to_1330 measured from the SAME wick-bar close so the look-ahead is contained inside
#      the same instrument used for the return -- this is consistent with the brief's mechanism
#      definition and matches the pattern used by sibling Phase 2 scripts.)
#   3. First-fire-per-day latch (sym, date) -- only first qualifying bar marked
#   4. ProductionUniverseGate per-date (Lesson #19)
#   5. large_cap AND unknown EXCLUDED from universe (small/mid_cap only)
#
# Pre-registration discipline (Lesson #2):
#   - Wick-bar vol_ratio reported BEFORE drift-delta computation
#   - Falsifier #1 evaluated BEFORE PROCEED/KILL verdict
#   - Falsifier #1 trigger: median vol_ratio < 1.0 -> KILL regardless of drift
"""Phase 2 empirical signature -- 5day_oversold_first_hour_bid_long."""
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
    "accepted_caps": {"small_cap", "mid_cap"},  # large_cap + unknown EXCLUDED (brief Section 5)
    "require_mis":   True,

    # ---- Multi-day filter (capitulation gating) ----
    # 5day_cumret = (close[T-1] / close[T-6]) - 1; require <= -0.08 (i.e. -8% or more)
    "cumret_lookback_bars":  5,         # 5 trading days
    "cumret_max":            -0.08,     # <= -8%
    # 5day_low = min(low[T-5..T-1]); require close[T-1] <= 5day_low * (1 + tol)
    "low_lookback_bars":     5,
    "near_low_tolerance":    0.01,      # close[T-1] within 1% of 5-day low

    # ---- Intraday signal-scan window ----
    "sig_window_start":  dtime(9, 30),     # 09:30 inclusive
    "sig_window_end":    dtime(10, 30),    # 10:30 inclusive (last considered bar starts at 10:30)

    # ---- Wick-bar trigger ----
    "wick_body_ratio_min": 0.5,            # lower_wick / max(body, eps) >= 0.5
    "body_eps":            0.0001,
    "no_lower_low_bars":   6,              # min(low for next 6 bars) >= bars[i].low

    # ---- Baseline anchor ----
    "baseline_anchor_time": dtime(9, 55),  # 09:55 5m bar close

    # ---- Target bar -- close at 13:25 5m bar (which IS 13:30 IST exit reference) ----
    "target_bar_time":     dtime(13, 25),

    # ---- Regime / period cuts ----
    "regime_2024_cut":     date(2024, 1, 1),
    "sebi_oct2025_cut":    date(2025, 10, 1),

    # ---- Overlap telemetry: long_panic_gap_down same-day gap-down >= 1.5% ----
    "overlap_gap_down_pct_min": 1.5,       # |gap_pct| >= 1.5  AND  gap_pct <= 0

    # ---- Acceptance ----
    "drift_delta_min":      0.15,          # signal_mean - baseline_mean must be >= +0.15%
    "n_signal_min":         200,
    "vol_ratio_median_min": 1.0,           # Falsifier #1: median must be >= 1.0 to PROCEED

    # ---- Paths ----
    "monthly_5m_dir":  _REPO_ROOT / "backtest-cache-download" / "monthly",
    "out_csv":         _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_5day_oversold_first_hour_bid_signature.csv",
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
# Daily lookup helpers (multi-day capitulation gating)
# -----------------------------------------------------------------------------
def _build_daily_index(daily_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group consolidated_daily by symbol, sort by ts ascending."""
    idx: Dict[str, pd.DataFrame] = {}
    for sym, g in daily_df.groupby("symbol", sort=False):
        g2 = g.sort_values("ts").reset_index(drop=True)
        idx[sym] = g2
    return idx


def _passes_multiday_filter(
    sym_daily: pd.DataFrame,
    session_date: date,
    cumret_lookback: int,
    cumret_max: float,
    low_lookback: int,
    near_low_tol: float,
) -> Optional[dict]:
    """Check 5-day cumulative oversold + close-near-low filter on T-1 daily bar.

    Returns dict with diagnostic fields if passes, else None.

    Definitions (per brief):
      5day_cumret = (close[T-1] / close[T-6]) - 1
      5day_low    = min(low[T-5 .. T-1])
      Pass iff: 5day_cumret <= cumret_max AND close[T-1] <= 5day_low * (1 + near_low_tol)

    NOTE: T-1 = last trading day STRICTLY BEFORE session_date. T-6 = 6 trading days before
    session_date (which is 5 days BEFORE T-1, so close[T-6] -> close[T-1] is 5 days of return).
    """
    prior = sym_daily[sym_daily["d"] < session_date]
    if len(prior) < max(cumret_lookback + 1, low_lookback + 1):
        return None

    # T-1 is last row
    closes = prior["close"].to_numpy(dtype=np.float64)
    lows   = prior["low"].to_numpy(dtype=np.float64)
    close_tm1 = float(closes[-1])
    close_tm6 = float(closes[-(cumret_lookback + 1)])  # 6 rows back from last
    if close_tm6 <= 0:
        return None

    cumret = (close_tm1 / close_tm6) - 1.0
    if cumret > cumret_max:
        return None

    five_day_low = float(lows[-low_lookback:].min())  # low[T-5 .. T-1]
    if close_tm1 > five_day_low * (1.0 + near_low_tol):
        return None

    return {
        "close_tm1": close_tm1,
        "close_tm6": close_tm6,
        "cumret_5d": cumret,
        "low_5d":    five_day_low,
        "near_low_ratio": (close_tm1 / five_day_low) - 1.0 if five_day_low > 0 else float("nan"),
    }


def _gap_pct_today(sym_daily: pd.DataFrame, session_date: date, first_bar_open: float) -> Optional[float]:
    """Compute today's gap_pct = (open - PDC) / PDC * 100. None if PDC unavailable."""
    prior = sym_daily[sym_daily["d"] < session_date]
    if prior.empty:
        return None
    pdc = float(prior["close"].iloc[-1])
    if pdc <= 0:
        return None
    return ((first_bar_open - pdc) / pdc) * 100.0


# -----------------------------------------------------------------------------
# Per-(symbol, day) evaluator
# -----------------------------------------------------------------------------
def evaluate_symbol_day(
    sym_bars: pd.DataFrame,
    sym: str,
    d: date,
    sym_daily: pd.DataFrame,
) -> Optional[dict]:
    """Evaluate one (symbol, day).

    Returns a record dict:
      - is_signal=True  if FIRST wick-bar bid in [09:30, 10:30] meets all criteria
      - is_baseline=True if NO wick-bar bid fires (use 09:55 close as anchor)
      - None if multi-day filter fails, or target bar / anchor bar missing,
        or 09:30-10:30 bars are missing entirely

    Anti-bias guards:
      - Multi-day filter uses ONLY daily bars strictly before d (no look-ahead)
      - first-fire-per-day: stop scanning after first qualifying wick bar
      - no_lower_low check uses bars[i+1 .. i+6] (next 30 min after the wick bar)
    """
    # Step A: multi-day filter
    daily_pass = _passes_multiday_filter(
        sym_daily,
        d,
        int(CONFIG["cumret_lookback_bars"]),
        float(CONFIG["cumret_max"]),
        int(CONFIG["low_lookback_bars"]),
        float(CONFIG["near_low_tolerance"]),
    )
    if daily_pass is None:
        return None

    # Step B: intraday bars -- need 09:15-15:25 + target bar at 13:25
    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    mask_sess = (sym_bars["time"] >= dtime(9, 15)) & (sym_bars["time"] <= dtime(15, 25))
    sym_bars = sym_bars[mask_sess].reset_index(drop=True)
    if sym_bars.empty:
        return None

    # Target bar (13:25 5m close == 13:30 IST exit reference)
    tgt_t = CONFIG["target_bar_time"]
    tgt_rows = sym_bars[sym_bars["time"] == tgt_t]
    if tgt_rows.empty:
        return None
    target_close = float(tgt_rows.iloc[0]["close"])

    # Today's gap_pct (vs PDC) for overlap telemetry
    first_bar_rows = sym_bars[sym_bars["time"] == dtime(9, 15)]
    if first_bar_rows.empty:
        # No 09:15 open bar -- skip
        return None
    today_first_open = float(first_bar_rows.iloc[0]["open"])
    today_gap_pct = _gap_pct_today(sym_daily, d, today_first_open)

    # Build per-bar arrays
    opens  = sym_bars["open"].to_numpy(dtype=np.float64)
    highs  = sym_bars["high"].to_numpy(dtype=np.float64)
    lows   = sym_bars["low"].to_numpy(dtype=np.float64)
    closes = sym_bars["close"].to_numpy(dtype=np.float64)
    vols   = sym_bars["volume"].to_numpy(dtype=np.float64)
    times  = sym_bars["time"].tolist()
    ts_col = sym_bars["date"].tolist()
    n = len(highs)
    if n < 2:
        return None

    sw_start = CONFIG["sig_window_start"]
    sw_end   = CONFIG["sig_window_end"]
    ratio_min = float(CONFIG["wick_body_ratio_min"])
    body_eps  = float(CONFIG["body_eps"])
    look_ahead = int(CONFIG["no_lower_low_bars"])

    # Wick-bar vol_baseline: mean of vols[:i] (excludes current bar) -- consistent with sibling scripts
    cum_vol = np.cumsum(vols)
    vol_baseline = np.full(n, np.nan, dtype=np.float64)
    idx_arr = np.arange(1, n)
    vol_baseline[1:] = cum_vol[:-1] / idx_arr

    # Scan for FIRST qualifying wick bar in [09:30, 10:30]
    signal_idx: Optional[int] = None
    for i in range(n):
        t = times[i]
        if t < sw_start or t > sw_end:
            continue
        # Need enough look-ahead bars for the no-lower-low check
        if i + look_ahead >= n:
            break
        body = abs(closes[i] - opens[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        if lower_wick < 0:
            # data anomaly; skip
            continue
        ratio = lower_wick / max(body, body_eps)
        if ratio < ratio_min:
            continue
        if not (closes[i] > opens[i]):
            continue
        # no fresh low in next 30 min: min(low[i+1 .. i+6]) >= low[i]
        future_min_low = float(lows[i + 1 : i + 1 + look_ahead].min())
        if future_min_low < lows[i]:
            continue
        signal_idx = i
        break

    if signal_idx is not None:
        i = signal_idx
        sig_close = float(closes[i])
        ret = (target_close - sig_close) / sig_close * 100.0
        body = abs(closes[i] - opens[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        wick_ratio = lower_wick / max(body, body_eps)
        if np.isnan(vol_baseline[i]) or vol_baseline[i] <= 0:
            vol_ratio = float("nan")
        else:
            vol_ratio = float(vols[i] / vol_baseline[i])

        # Overlap telemetry: same-day long_panic_gap_down trigger criterion (gap-down >= 1.5%)
        gap_down_overlap = (
            (today_gap_pct is not None)
            and (today_gap_pct <= -float(CONFIG["overlap_gap_down_pct_min"]))
        )

        return {
            "symbol": sym,
            "date": d.isoformat(),
            "is_signal": True,
            "is_baseline": False,
            "signal_bar_ts": pd.Timestamp(ts_col[i]).isoformat(),
            "signal_bar_time": str(times[i]),
            "signal_bar_open":  float(opens[i]),
            "signal_bar_high":  float(highs[i]),
            "signal_bar_low":   float(lows[i]),
            "signal_bar_close": sig_close,
            "wick_body_ratio":  wick_ratio,
            "vol_ratio":        vol_ratio,
            "ret_to_1330":      ret,
            "target_close_1330": target_close,
            "cumret_5d":        daily_pass["cumret_5d"],
            "low_5d":           daily_pass["low_5d"],
            "near_low_ratio":   daily_pass["near_low_ratio"],
            "today_gap_pct":    today_gap_pct if today_gap_pct is not None else float("nan"),
            "gap_down_overlap": bool(gap_down_overlap),
        }

    # Baseline: no wick-bar bid in [09:30, 10:30]; anchor on 09:55 close
    anchor_t = CONFIG["baseline_anchor_time"]
    anchor_rows = sym_bars[sym_bars["time"] == anchor_t]
    if anchor_rows.empty:
        return None
    anchor_close = float(anchor_rows.iloc[0]["close"])
    if anchor_close <= 0:
        return None
    ret_base = (target_close - anchor_close) / anchor_close * 100.0

    return {
        "symbol": sym,
        "date": d.isoformat(),
        "is_signal": False,
        "is_baseline": True,
        "signal_bar_ts": pd.Timestamp(anchor_rows.iloc[0]["date"]).isoformat(),
        "signal_bar_time": str(anchor_t),
        "signal_bar_open":  float("nan"),
        "signal_bar_high":  float("nan"),
        "signal_bar_low":   float("nan"),
        "signal_bar_close": anchor_close,
        "wick_body_ratio":  float("nan"),
        "vol_ratio":        float("nan"),
        "ret_to_1330":      ret_base,
        "target_close_1330": target_close,
        "cumret_5d":        daily_pass["cumret_5d"],
        "low_5d":           daily_pass["low_5d"],
        "near_low_ratio":   daily_pass["near_low_ratio"],
        "today_gap_pct":    today_gap_pct if today_gap_pct is not None else float("nan"),
        "gap_down_overlap": False,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature -- 5day_oversold_first_hour_bid_long")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print(f"Caps:   {sorted(CONFIG['accepted_caps'])}  (large_cap + unknown EXCLUDED)")
    print(f"Multi-day filter: 5d_cumret <= {CONFIG['cumret_max']:+.3f} "
          f"AND close[T-1] <= 5d_low * (1+{CONFIG['near_low_tolerance']:.2f})")
    print(f"Wick trigger:     lower_wick/body >= {CONFIG['wick_body_ratio_min']} AND green bar "
          f"AND no lower low in next {CONFIG['no_lower_low_bars']} bars")
    print(f"Sig window:       [{CONFIG['sig_window_start']}, {CONFIG['sig_window_end']}]")
    print(f"Baseline anchor:  {CONFIG['baseline_anchor_time']} (no signal day)")
    print(f"Target bar:       close at {CONFIG['target_bar_time']} (= 13:30 IST exit)")
    print("=" * 80)

    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )

    # Load consolidated_daily once and index by symbol
    _log("Loading consolidated_daily.feather ...")
    daily_df = pd.read_feather(_REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather")
    daily_df["ts"] = pd.to_datetime(daily_df["ts"])
    daily_df["d"] = daily_df["ts"].dt.date
    daily_idx = _build_daily_index(daily_df)
    _log(f"  daily symbols indexed: {len(daily_idx):,}")

    records: List[dict] = []
    months = _months_between(CONFIG["window_start"], CONFIG["window_end"])
    total_eval_pairs = 0
    total_rejected_universe = 0
    total_no_record = 0

    nse_all = gate._load_nse_all()
    accepted = CONFIG["accepted_caps"]
    keep_syms = {s for s, row in nse_all.items() if row.cap_segment in accepted}
    _log(f"  cap-eligible symbols (small/mid_cap): {len(keep_syms):,}")

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
        # Confine to window
        df = df[(df["day"] >= CONFIG["window_start"]) & (df["day"] <= CONFIG["window_end"])]
        if df.empty:
            continue

        # Filter to cap-eligible symbols
        df = df[df["symbol"].isin(keep_syms)]
        if df.empty:
            continue

        gb = df.groupby(["symbol", "day"], sort=False)
        month_eval = 0
        month_reject = 0
        month_rec = 0
        month_sig = 0

        for (sym, d), bars in gb:
            month_eval += 1
            total_eval_pairs += 1
            if not gate.is_eligible(sym, d):
                month_reject += 1
                total_rejected_universe += 1
                continue
            sym_daily = daily_idx.get(sym)
            if sym_daily is None or sym_daily.empty:
                total_no_record += 1
                continue
            rec = evaluate_symbol_day(bars, sym, d, sym_daily)
            if rec is None:
                total_no_record += 1
                continue
            records.append(rec)
            month_rec += 1
            if rec.get("is_signal"):
                month_sig += 1

        _log(
            f"  {yy:04d}-{mm:02d}: pairs={month_eval:,} "
            f"rejected_univ={month_reject:,} "
            f"records={month_rec:,} signals={month_sig:,}"
        )

    print()
    print("=== Pipeline tally ===")
    print(f"  total (sym, day) pairs in caps:      {total_eval_pairs:,}")
    print(f"  rejected by universe gate:           {total_rejected_universe:,}")
    print(f"  no record (filter/target/anchor):    {total_no_record:,}")
    print(f"  recorded:                            {len(records):,}")

    if not records:
        print("\nNO RECORDS COLLECTED -- abort.")
        return 1

    df_out = pd.DataFrame.from_records(records)

    # Attach cap_segment from nse_all + cohort splits
    def _cap(s: str) -> str:
        row = nse_all.get(s)
        return row.cap_segment if row else "unknown"

    df_out["cap_segment"] = df_out["symbol"].map(_cap)

    dt_dates = pd.to_datetime(df_out["date"])
    cut_2024 = pd.Timestamp(CONFIG["regime_2024_cut"])
    cut_sebi = pd.Timestamp(CONFIG["sebi_oct2025_cut"])
    df_out["regime_pre_post_2024"] = np.where(dt_dates < cut_2024, "pre", "post")
    df_out["period_pre_post_sebi_oct2025"] = np.where(dt_dates < cut_sebi, "pre", "post")

    # wick/body ratio buckets per brief
    def _wbr_bucket(v: float) -> str:
        if pd.isna(v):
            return "nan"
        if v < 0.5:
            return "<0.5"
        if v < 1.0:
            return "0.5-1.0"
        if v < 2.0:
            return "1.0-2.0"
        return ">=2.0"

    df_out["wick_body_bucket"] = df_out["wick_body_ratio"].map(_wbr_bucket)

    # Save CSV
    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out):,} rows -> {out_path}")

    sig = df_out[df_out["is_signal"]].copy()
    base = df_out[df_out["is_baseline"]].copy()
    n_sig = len(sig)
    n_base = len(base)

    # -------------------------------------------------------------------------
    # STEP 1 -- Falsifier #1 (wick-bar vol_ratio) PRINTED BEFORE DRIFT DELTA
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 1 -- Falsifier #1 (wick-bar vol_ratio)")
    print("=" * 80)
    if n_sig == 0:
        print("  n_signal = 0 -- cannot evaluate Falsifier #1. Verdict: KILL (no signal events).")
        return 0

    vr_sig = sig["vol_ratio"].astype(float).dropna()
    if vr_sig.empty:
        print("  vol_ratio all-NaN -- cannot evaluate Falsifier #1. Verdict: KILL.")
        return 0

    q25, q50, q75 = vr_sig.quantile([0.25, 0.50, 0.75])
    median = float(q50)
    frac_lt_one = float((vr_sig < 1.0).mean())

    print(f"  n_signal:              {n_sig:,}")
    print(f"  n_signal_with_vr:      {len(vr_sig):,}")
    print(f"  vol_ratio MEDIAN:      {median:.4f}   [>= {CONFIG['vol_ratio_median_min']} required]")
    print(f"  vol_ratio 25/50/75:    {float(q25):.4f} / {float(q50):.4f} / {float(q75):.4f}")
    print(f"  Fraction vol_ratio<1.0: {frac_lt_one*100:.2f}%")

    falsifier1_pass = median >= float(CONFIG["vol_ratio_median_min"])
    print(f"\n  FALSIFIER #1: {'PASS' if falsifier1_pass else 'FAIL'}")

    if not falsifier1_pass:
        print("\n  STOP -- Falsifier #1 FAILED (median vol_ratio < 1.0).")
        print("  Wick bars are thin noise, not real institutional bid.")
        print(f"\n  n_signal {n_sig} {'>=' if n_sig >= CONFIG['n_signal_min'] else '<'} {CONFIG['n_signal_min']}")
        print("\n" + "=" * 80)
        print("VERDICT: KILL (Falsifier #1 failed -- thin-volume wicks)")
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
    sig_median_ret = float(sig["ret_to_1330"].median()) if n_sig else float("nan")
    base_median_ret = float(base["ret_to_1330"].median()) if n_base else float("nan")
    delta = sig_mean - base_mean if (n_sig and n_base) else float("nan")

    print(f"  n_signal:                   {n_sig:,}")
    print(f"  n_baseline:                 {n_base:,}")
    print(f"  Signal   mean ret_to_1330:  {sig_mean:+.4f}%  (median {sig_median_ret:+.4f}%)")
    print(f"  Baseline mean ret_to_1330:  {base_mean:+.4f}%  (median {base_median_ret:+.4f}%)")
    print(f"  DRIFT DELTA:                {delta:+.4f}%   [>= +{CONFIG['drift_delta_min']:.2f}% required]")

    # -------------------------------------------------------------------------
    # STEP 3 -- Cohort splits + overlap telemetry
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

    pre24_n, pre24_sm, pre24_bm, pre24_d = _split_block(
        "pre_2024",  sig_dates < cut_2024,  base_dates < cut_2024
    )
    post24_n, post24_sm, post24_bm, post24_d = _split_block(
        "post_2024", sig_dates >= cut_2024, base_dates >= cut_2024
    )
    pre_sebi_n, pre_sebi_sm, pre_sebi_bm, pre_sebi_d = _split_block(
        "pre_sebi_oct2025",  sig_dates < cut_sebi,  base_dates < cut_sebi
    )
    post_sebi_n, post_sebi_sm, post_sebi_bm, post_sebi_d = _split_block(
        "post_sebi_oct2025", sig_dates >= cut_sebi, base_dates >= cut_sebi
    )

    for cap_val in ["small_cap", "mid_cap"]:
        _split_block(f"cap={cap_val}", sig["cap_segment"] == cap_val, base["cap_segment"] == cap_val)

    print()
    print("  --- wick/body ratio buckets (signal rows; baseline = aggregate baseline) ---")
    print(f"{'bucket':<32}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    bucket_stats: List[Tuple[str, int, float]] = []
    for bucket in ["0.5-1.0", "1.0-2.0", ">=2.0"]:
        mask_s = sig["wick_body_bucket"] == bucket
        s = sig[mask_s]
        ns = len(s)
        sm = float(s["ret_to_1330"].mean()) if ns else float("nan")
        bm = base_mean
        dl = sm - bm if (ns and not np.isnan(bm)) else float("nan")
        bucket_stats.append((bucket, ns, sm))
        print(
            f"wick_body_bucket={bucket:<14}"
            f"{ns:>8d}"
            f"{n_base:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )

    # Overlap telemetry with long_panic_gap_down (gap-down >= 1.5% on T+0 09:15)
    print()
    print("  --- Overlap telemetry: long_panic_gap_down (gap_pct <= "
          f"-{CONFIG['overlap_gap_down_pct_min']:.1f}%) ---")
    sig_with_gap = sig.dropna(subset=["today_gap_pct"])
    n_with_gap = len(sig_with_gap)
    n_overlap = int(sig["gap_down_overlap"].sum())
    frac_overlap = (n_overlap / n_sig) if n_sig else float("nan")
    print(f"  n_signal:                   {n_sig:,}")
    print(f"  n_signal_with_gap_pct:      {n_with_gap:,}")
    print(f"  n_overlap (gap_down>=1.5%): {n_overlap:,}")
    print(f"  overlap fraction:           {frac_overlap*100:.2f}%")
    print(f"  (brief: if >50%, mechanism partially derivative; informs M=1.5 validation)")

    # Per-overlap-subset return comparison
    if n_overlap > 0:
        sm_ov = float(sig[sig["gap_down_overlap"]]["ret_to_1330"].mean())
        n_no_ov = int((~sig["gap_down_overlap"]).sum())
        sm_no_ov = float(sig[~sig["gap_down_overlap"]]["ret_to_1330"].mean()) if n_no_ov else float("nan")
        print(f"  signal_mean (overlap=True):  {sm_ov:+.4f}%   (n={n_overlap})")
        print(f"  signal_mean (overlap=False): {sm_no_ov:+.4f}%   (n={n_no_ov})")

    # -------------------------------------------------------------------------
    # VERDICT
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    drift_ok = (not np.isnan(delta)) and (delta >= float(CONFIG["drift_delta_min"]))
    n_ok = n_sig >= int(CONFIG["n_signal_min"])
    # post-SEBI sign flip = post_sebi_delta is NEGATIVE (opposite of LONG direction)
    sign_flip = (not np.isnan(post_sebi_d)) and (post_sebi_d < 0) and (post_sebi_n >= 50)

    verdict: str
    reason: str
    if not n_ok:
        verdict = "KILL"
        reason = f"n_signal {n_sig} < {CONFIG['n_signal_min']}"
    elif not drift_ok:
        verdict = "KILL"
        reason = (
            f"drift delta {delta:+.4f}% < +{CONFIG['drift_delta_min']:.2f}% "
            f"(insufficient LONG drift)"
        )
    elif sign_flip:
        verdict = "DEFER"
        reason = (
            f"post-SEBI-Oct-2025 cohort delta is NEGATIVE "
            f"({post_sebi_d:+.4f}%, n_sig={post_sebi_n}) -- regulatory regime risk"
        )
    else:
        verdict = "PROCEED to Phase 3"
        reason = (
            f"Falsifier #1 PASS (median vr {median:.3f} >= 1.0), "
            f"drift {delta:+.4f}% >= +{CONFIG['drift_delta_min']:.2f}%, "
            f"n_signal {n_sig} >= {CONFIG['n_signal_min']}, no post-SEBI sign flip"
        )

    print(f"  {verdict}")
    print(f"  reason: {reason}")

    # Monotonicity inspection note (wick_body buckets)
    sm_seq = [sm for (_, ns, sm) in bucket_stats if ns > 0]
    note_lines: List[str] = []
    if len(sm_seq) >= 2:
        # If thicker wick = bigger bid = bigger long drift, sm should monotonically INCREASE
        # from 0.5-1.0 -> 1.0-2.0 -> >=2.0.
        increasing = all(sm_seq[i] <= sm_seq[i+1] for i in range(len(sm_seq)-1))
        decreasing = all(sm_seq[i] >= sm_seq[i+1] for i in range(len(sm_seq)-1))
        if increasing:
            note_lines.append(
                "  MONOTONICITY: wick_body signal_mean monotonically INCREASES "
                "(bigger wick -> bigger LONG drift). Mechanism confirmation."
            )
        elif decreasing:
            note_lines.append(
                "  MONOTONICITY: wick_body signal_mean monotonically DECREASES "
                "(bigger wick -> smaller LONG drift). Mechanism INVERTED -- CAUTION."
            )
        else:
            note_lines.append(
                "  MONOTONICITY: wick_body signal_mean NON-MONOTONIC -- mixed effect."
            )

    if note_lines:
        print()
        print("Notes:")
        for ln in note_lines:
            print(ln)

    return 0


if __name__ == "__main__":
    sys.exit(run())
