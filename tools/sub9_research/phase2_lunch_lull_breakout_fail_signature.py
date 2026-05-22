# tools/sub9_research/phase2_lunch_lull_breakout_fail_signature.py
#
# Phase 2 empirical signature for lunch_lull_breakout_fail_short candidate.
# See specs/2026-05-22-brief-lunch_lull_breakout_fail_short.md
#
# Anti-bias guards (Lesson #5):
#   1. intraday_high_prior computed from bars[:i] only -- no look-ahead
#   2. vol_baseline excludes current bar (mean of bars[:i].volume)
#   3. First-fire-per-day latch (sym, date) -- only first qualifying bar marked
#   4. ProductionUniverseGate per-date (Lesson #19)
#   5. large_cap AND unknown EXCLUDED from universe (C-H carry-over)
#
# Pre-registration discipline (raised bar from Phase 1):
#   - vol_ratio distribution reported BEFORE drift-delta computation
#   - Falsifier #1 evaluated BEFORE PROCEED/KILL verdict
"""Phase 2 empirical signature - lunch_lull_breakout_fail_short."""
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
    "accepted_caps": {"small_cap", "mid_cap"},  # large_cap + unknown EXCLUDED (C-H carry-over)
    "require_mis":   True,

    # Signal-scan window (lunch lull, per brief)
    "sig_window_start":  dtime(11, 30),    # 11:30 inclusive
    "sig_window_end":    dtime(13, 0),     # 13:00 EXCLUSIVE -- last considered bar is 12:55

    # Signal trigger
    "vol_ratio_max":     1.2,              # < 1.2 -> SIGNAL (thin-volume breakout)
                                           # >= 1.2 -> BASELINE (volume-confirmed control)

    # Target bar -- close_at_1425 == 14:25 5m bar close (which IS 14:30 IST)
    "target_bar_time":   dtime(14, 25),

    # Regime / period cuts
    "regime_2024_cut":   date(2024, 1, 1),
    "sebi_oct2025_cut":  date(2025, 10, 1),

    # Acceptance
    "drift_delta_max":   -0.15,            # signal_mean - baseline_mean must be <= -0.15%
    "n_signal_min":      200,
    "vol_ratio_median_max":     1.0,       # Falsifier #1: median must be < 1.0
    "vol_ratio_ge_one_frac_max": 0.40,     # Falsifier #1: <40% of fires at >= 1.0

    # Paths
    "monthly_5m_dir":  _REPO_ROOT / "backtest-cache-download" / "monthly",
    "out_csv":         _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_lunch_lull_breakout_fail_signature.csv",
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
# Per-(symbol, day) evaluator
# -----------------------------------------------------------------------------
def evaluate_symbol_day(sym_bars: pd.DataFrame, sym: str, d: date) -> Optional[dict]:
    """Iterate 5m bars and find the FIRST fresh-intraday-high bar in [11:30, 13:00).

    Returns a SINGLE record per (sym, date):
      - 'signal'   if FIRST fresh-IH bar in window has vol_ratio  < 1.2
      - 'baseline' if FIRST fresh-IH bar in window has vol_ratio >= 1.2
      - None       if no fresh-IH bar fires in the window OR target bar missing

    Anti-bias guards:
      - intraday_high_prior = max(bars[:i].high) -- strictly prior bars
      - vol_baseline        = mean(bars[:i].volume) -- excludes current bar
      - first-fire-per-day  -- after marking, stop scanning that (sym, date)
    """
    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    if sym_bars.empty:
        return None

    # Confine to the regular trading session for sanity
    mask_sess = (sym_bars["time"] >= dtime(9, 15)) & (sym_bars["time"] <= dtime(15, 25))
    sym_bars = sym_bars[mask_sess].reset_index(drop=True)
    if sym_bars.empty:
        return None

    # Target bar (14:25 5m bar close == 14:30 IST exit reference)
    tgt_t = CONFIG["target_bar_time"]
    tgt_rows = sym_bars[sym_bars["time"] == tgt_t]
    if tgt_rows.empty:
        return None
    target_close = float(tgt_rows.iloc[0]["close"])

    # Build per-bar arrays
    highs = sym_bars["high"].to_numpy(dtype=np.float64)
    closes = sym_bars["close"].to_numpy(dtype=np.float64)
    vols = sym_bars["volume"].to_numpy(dtype=np.float64)
    times = sym_bars["time"].tolist()
    ts_col = sym_bars["date"].tolist()

    # intraday_high_prior[i] = max(high[0..i-1])  -- excludes current bar
    n = len(highs)
    if n < 2:
        return None
    cum_high = np.maximum.accumulate(highs)
    intraday_high_prior = np.full(n, np.nan, dtype=np.float64)
    intraday_high_prior[1:] = cum_high[:-1]

    # vol_baseline[i] = mean(vols[0..i-1]) -- excludes current bar
    cum_vol = np.cumsum(vols)
    vol_baseline = np.full(n, np.nan, dtype=np.float64)
    idx_arr = np.arange(1, n)
    vol_baseline[1:] = cum_vol[:-1] / idx_arr

    sw_start = CONFIG["sig_window_start"]
    sw_end   = CONFIG["sig_window_end"]
    vol_ratio_max = float(CONFIG["vol_ratio_max"])

    # First-fire scan: find FIRST bar in window where high > intraday_high_prior (fresh IH)
    for i in range(n):
        t = times[i]
        if t < sw_start or t >= sw_end:
            continue
        if np.isnan(intraday_high_prior[i]) or np.isnan(vol_baseline[i]) or vol_baseline[i] <= 0:
            continue
        # Fresh intraday high?
        if not (highs[i] > intraday_high_prior[i]):
            continue
        # Compute vol_ratio
        vr = float(vols[i] / vol_baseline[i])
        is_signal = vr < vol_ratio_max
        is_baseline = not is_signal

        sig_close = float(closes[i])
        ret = (target_close - sig_close) / sig_close * 100.0

        return {
            "symbol": sym,
            "date": d.isoformat(),
            "signal_bar_ts": pd.Timestamp(ts_col[i]).isoformat(),
            "signal_bar_time": str(times[i]),
            "signal_bar_close": sig_close,
            "intraday_high_at_signal": float(highs[i]),  # bar's own high IS new IH
            "intraday_high_prior": float(intraday_high_prior[i]),
            "vol_ratio": vr,
            "ret_to_1430": ret,
            "target_close_1430": target_close,
            "is_signal": bool(is_signal),
            "is_baseline": bool(is_baseline),
        }

    # No fresh-IH bar in the lunch-lull window today; this (sym, date) drops out.
    return None


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature -- lunch_lull_breakout_fail_short")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print(f"Caps:   {sorted(CONFIG['accepted_caps'])}  (large_cap + unknown EXCLUDED)")
    print(f"Signal trigger: fresh intraday high in [11:30, 13:00) AND vol_ratio < {CONFIG['vol_ratio_max']}")
    print(f"Baseline:       fresh intraday high in [11:30, 13:00) AND vol_ratio >= {CONFIG['vol_ratio_max']}")
    print(f"Target bar:     close at {CONFIG['target_bar_time']} (= 14:30 IST exit reference)")
    print("=" * 80)

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
        # Confine to window
        df = df[(df["day"] >= CONFIG["window_start"]) & (df["day"] <= CONFIG["window_end"])]
        if df.empty:
            continue

        # Per-month: pre-filter symbols to those whose cap is in accepted_caps (cheap)
        nse_all = gate._load_nse_all()
        accepted = CONFIG["accepted_caps"]
        # require_mis is checked in is_eligible() per-symbol
        keep_syms = {s for s, row in nse_all.items() if row.cap_segment in accepted}
        df = df[df["symbol"].isin(keep_syms)]
        if df.empty:
            continue

        gb = df.groupby(["symbol", "day"], sort=False)
        month_eval = 0
        month_reject = 0
        month_rec = 0

        for (sym, d), bars in gb:
            month_eval += 1
            total_eval_pairs += 1
            if not gate.is_eligible(sym, d):
                month_reject += 1
                total_rejected_universe += 1
                continue
            rec = evaluate_symbol_day(bars, sym, d)
            if rec is None:
                total_no_record += 1
                continue
            records.append(rec)
            month_rec += 1

        _log(
            f"  {yy:04d}-{mm:02d}: pairs={month_eval:,} "
            f"rejected_univ={month_reject:,} "
            f"records={month_rec:,}"
        )

    print()
    print("=== Pipeline tally ===")
    print(f"  total (sym, day) pairs in caps:      {total_eval_pairs:,}")
    print(f"  rejected by universe gate:           {total_rejected_universe:,}")
    print(f"  no fresh-IH in window or no target:  {total_no_record:,}")
    print(f"  recorded:                            {len(records):,}")

    if not records:
        print("\nNO RECORDS COLLECTED -- abort.")
        return 1

    df_out = pd.DataFrame.from_records(records)

    # Attach cap_segment from nse_all + cohort splits
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

    # vol_ratio buckets (only meaningful for signal rows, but compute for all and we will filter)
    def _vr_bucket(v: float) -> str:
        if pd.isna(v):
            return "nan"
        if v < 0.6:
            return "<0.6"
        if v < 0.8:
            return "0.6-0.8"
        if v < 1.0:
            return "0.8-1.0"
        if v < 1.2:
            return "1.0-1.2"
        return ">=1.2"

    df_out["vol_ratio_bucket"] = df_out["vol_ratio"].map(_vr_bucket)

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
    # STEP 1 -- Vol-ratio pre-registration (Falsifier #1) PRINTED BEFORE DRIFT
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 1 -- Vol-ratio pre-registration (Falsifier #1)")
    print("=" * 80)
    if n_sig == 0:
        print("  n_signal = 0 -- cannot evaluate Falsifier #1. Verdict: KILL (no signal events).")
        return 1

    vr_sig = sig["vol_ratio"].astype(float)
    q25, q50, q75 = vr_sig.quantile([0.25, 0.50, 0.75])
    median = float(q50)
    frac_ge_one = float((vr_sig >= 1.0).mean())

    print(f"  n_signal:              {n_sig:,}")
    print(f"  vol_ratio MEDIAN:      {median:.4f}   [< {CONFIG['vol_ratio_median_max']} required]")
    print(f"  vol_ratio 25/50/75:    {float(q25):.4f} / {float(q50):.4f} / {float(q75):.4f}")
    print(f"  Fraction vol_ratio>=1.0: {frac_ge_one*100:.2f}%   [< {CONFIG['vol_ratio_ge_one_frac_max']*100:.0f}% required]")

    falsifier1_pass = (
        median < float(CONFIG["vol_ratio_median_max"]) and
        frac_ge_one < float(CONFIG["vol_ratio_ge_one_frac_max"])
    )
    print(f"\n  FALSIFIER #1: {'PASS' if falsifier1_pass else 'FAIL'}")

    if not falsifier1_pass:
        print("\n  STOP -- Falsifier #1 FAILED. Verdict: KILL regardless of drift.")
        print("\n  (Note: by construction vol_ratio < 1.2 for all signal rows; the median test")
        print("   confirms the trigger is genuinely picking THIN-volume breakouts.)")
        # Still print n verification
        print(f"\n  n_signal {n_sig} {'>=' if n_sig >= CONFIG['n_signal_min'] else '<'} {CONFIG['n_signal_min']}")
        print("\n" + "=" * 80)
        print("VERDICT: KILL (Falsifier #1 failed)")
        print("=" * 80)
        return 0

    # -------------------------------------------------------------------------
    # STEP 2 -- Drift delta (only if Falsifier #1 PASSED)
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 2 -- Drift delta")
    print("=" * 80)

    sig_mean = float(sig["ret_to_1430"].mean()) if n_sig else float("nan")
    base_mean = float(base["ret_to_1430"].mean()) if n_base else float("nan")
    sig_median_ret = float(sig["ret_to_1430"].median()) if n_sig else float("nan")
    base_median_ret = float(base["ret_to_1430"].median()) if n_base else float("nan")
    delta = sig_mean - base_mean if (n_sig and n_base) else float("nan")

    print(f"  n_signal:                   {n_sig:,}")
    print(f"  n_baseline:                 {n_base:,}")
    print(f"  Signal   mean ret_to_1430:  {sig_mean:+.4f}%  (median {sig_median_ret:+.4f}%)")
    print(f"  Baseline mean ret_to_1430:  {base_mean:+.4f}%  (median {base_median_ret:+.4f}%)")
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
        sm = float(s["ret_to_1430"].mean()) if ns else float("nan")
        bm = float(b["ret_to_1430"].mean()) if nb else float("nan")
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

    for cap_val in ["small_cap", "mid_cap"]:
        _split_block(f"cap={cap_val}", sig["cap_segment"] == cap_val, base["cap_segment"] == cap_val)

    print()
    print("  --- vol_ratio_bucket (signal rows only) ---")
    print(f"{'bucket':<32}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    bucket_stats: List[Tuple[str, int, float]] = []
    for bucket in ["<0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2"]:
        mask_s = sig["vol_ratio_bucket"] == bucket
        # baseline by construction has vol_ratio >= 1.2, so per-bucket baseline is empty for <1.2 buckets.
        # We still report n_sig and signal_mean for monotonicity inspection; delta vs aggregate-baseline
        # is reported as (signal_mean - aggregate_baseline_mean).
        s = sig[mask_s]
        ns = len(s)
        sm = float(s["ret_to_1430"].mean()) if ns else float("nan")
        bm = base_mean  # use aggregate baseline mean for monotonicity comparison
        dl = sm - bm if (ns and not np.isnan(bm)) else float("nan")
        bucket_stats.append((bucket, ns, sm))
        print(
            f"vol_ratio_bucket={bucket:<14}"
            f"{ns:>8d}"
            f"{n_base:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )

    # -------------------------------------------------------------------------
    # VERDICT
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    drift_ok = (not np.isnan(delta)) and (delta <= float(CONFIG["drift_delta_max"]))
    n_ok = n_sig >= int(CONFIG["n_signal_min"])
    # post-SEBI sign flip = post_sebi_delta is positive (opposite of SHORT direction)
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
    else:
        verdict = "PROCEED to Phase 3"
        reason = (
            f"Falsifier #1 PASS, "
            f"drift {delta:+.4f}% <= {CONFIG['drift_delta_max']:.2f}%, "
            f"n_signal {n_sig} >= {CONFIG['n_signal_min']}, no post-SEBI sign flip"
        )

    print(f"  {verdict}")
    print(f"  reason: {reason}")

    # Monotonicity inspection note
    sm_seq = [sm for (_, ns, sm) in bucket_stats if ns > 0]
    note_lines: List[str] = []
    if len(sm_seq) >= 2:
        # We want: as vol_ratio shrinks (bucket moves <0.6 -> 1.0-1.2), the signal_mean
        # should become MORE NEGATIVE if the thin-volume mechanism is real.
        # bucket order in bucket_stats is <0.6, 0.6-0.8, 0.8-1.0, 1.0-1.2  (low to high)
        # So we want sm_seq to be MONOTONICALLY INCREASING (less negative as vol gets bigger).
        increasing = all(sm_seq[i] <= sm_seq[i+1] for i in range(len(sm_seq)-1))
        decreasing = all(sm_seq[i] >= sm_seq[i+1] for i in range(len(sm_seq)-1))
        if increasing:
            note_lines.append(
                "  MONOTONICITY: vol_ratio_bucket signal_mean monotonically INCREASES from <0.6 -> 1.0-1.2 "
                "(more negative when volume is thinner). Mechanism confirmation."
            )
        elif decreasing:
            note_lines.append(
                "  MONOTONICITY: vol_ratio_bucket signal_mean monotonically DECREASES (more negative as vol RISES). "
                "Mechanism INVERTED -- thin-volume is not the driver. CAUTION."
            )
        else:
            note_lines.append(
                "  MONOTONICITY: vol_ratio_bucket signal_mean NON-MONOTONIC -- thin-volume effect is mixed."
            )

    if note_lines:
        print()
        print("Notes:")
        for ln in note_lines:
            print(ln)

    return 0


if __name__ == "__main__":
    sys.exit(run())
