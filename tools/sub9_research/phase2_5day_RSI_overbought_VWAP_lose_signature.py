# tools/sub9_research/phase2_5day_RSI_overbought_VWAP_lose_signature.py
#
# Phase 2 empirical signature for `5day_RSI_overbought_intraday_VWAP_lose_short`.
# See specs/2026-05-22-brief-5day_RSI_overbought_intraday_VWAP_lose_short.md
#
# Mechanism:
#   - Universe: cap in {small_cap, mid_cap}, MIS-eligible, ProductionUniverseGate
#   - Multi-day filter: daily RSI(14) Wilder smoothing on consolidated_daily.close
#     SUSTAINED: RSI[T-1] >= 75 AND RSI[T-2] >= 75 AND RSI[T-3] >= 75
#   - Intraday SHORT signal (per 5m bar in 09:30-12:00):
#       (a) cumulative intraday VWAP via typical_price * volume through bar i
#       (b) cross-down: bars[i].close < VWAP[i] AND bars[i-1].close >= VWAP[i-1]
#       (c) vol_ratio = bars[i].volume / mean(prior intraday volume) >= 1.2
#       First-fire per (sym, date).
#   - Baseline: same RSI-sustained universe, NO VWAP cross-down in 09:30-12:00.
#       Anchor: 12:00 close.
#   - Target: SHORT, ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100
#
# Anti-bias guards (Lesson #5):
#   1. RSI uses only daily bars STRICTLY PRIOR to T (T-1..T-N)
#   2. cumulative VWAP at bar i uses bars[:i+1] only
#   3. vol_baseline excludes current bar (mean of prior intraday bars)
#   4. First-fire-per-day latch (sym, date)
#   5. ProductionUniverseGate per-date (Lesson #19)
#   6. large_cap AND unknown EXCLUDED from universe
#   7. Exclude symbols with <30 daily bars before T (RSI(14) stability)
#
# Pre-registration discipline (Lesson #2 / brief Falsifier #1):
#   - STEP 1: compare 3-day sustained vs SINGLE-DAY-only RSI cohort drift deltas.
#     If |sustained_delta - single_day_delta| < 0.05%, KILL (sustained adds no value).
#   - Post-SEBI-Oct-2025 sign-flip = DEFER signal (regime risk inherited from
#     retired `mis_unwind_vwap_revert_short`).
"""Phase 2 empirical signature - 5day_RSI_overbought_intraday_VWAP_lose_short."""
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
# CONFIG - NO hardcoded defaults inside the logic; every knob is declared here.
# -----------------------------------------------------------------------------
CONFIG: Dict[str, object] = {
    # Date window
    "window_start": date(2023, 1, 2),
    "window_end":   date(2026, 4, 30),

    # Universe
    "accepted_caps": {"small_cap", "mid_cap"},  # large_cap + unknown EXCLUDED
    "require_mis":   True,
    "min_daily_bars_required": 30,  # for stable RSI(14)

    # RSI
    "rsi_period":           14,         # Wilder smoothing alpha = 1/14
    "rsi_threshold":        75.0,
    "rsi_sustained_days":   3,          # primary cohort
    "rsi_durations_to_test": [3, 5, 7], # STEP 3 split

    # Intraday signal scan window
    "sig_window_start":  dtime(9, 30),
    "sig_window_end":    dtime(12, 0),  # EXCLUSIVE

    # Trigger thresholds
    "vol_ratio_min":     1.2,           # vol confirmation for VWAP cross-down

    # Target
    "target_bar_time":   dtime(13, 25), # 5m bar at 13:25 close == 13:30 IST
    # Baseline anchor (no-VWAP-cross day)
    "baseline_anchor_time": dtime(12, 0),

    # Regime / period cuts
    "regime_2024_cut":   date(2024, 1, 1),
    "sebi_oct2025_cut":  date(2025, 10, 1),

    # Acceptance gates
    "drift_delta_max":           -0.15,  # SHORT: signal_mean - baseline_mean must be <= -0.15%
    "n_signal_min":              200,
    "falsifier1_min_separation": 0.05,   # |sustained - single_day| must be >= 0.05%

    # Paths
    "monthly_5m_dir":  _REPO_ROOT / "backtest-cache-download" / "monthly",
    "daily_path":      _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather",
    "out_csv":         _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_5day_RSI_overbought_VWAP_lose_signature.csv",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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
    if isinstance(ts_col.dtype, pd.DatetimeTZDtype):
        return ts_col.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts_col


# -----------------------------------------------------------------------------
# RSI(14) Wilder smoothing -- vectorized per-symbol on daily closes.
# alpha = 1 / period; uses Wilder's recursive average of gains/losses.
# Returns aligned series (NaN for warmup).
# -----------------------------------------------------------------------------
def _wilder_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return out
    diff = np.diff(closes)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)

    # Seed: simple mean of first `period` gains/losses (Wilder seed)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    if avg_loss == 0.0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder recursive: avg = ((period-1)*prev + new) / period
    for i in range(period + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = ((period - 1) * avg_gain + g) / period
        avg_loss = ((period - 1) * avg_loss + l) / period
        if avg_loss == 0.0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def _build_rsi_lookup(
    daily_df: pd.DataFrame,
    period: int,
    min_bars: int,
) -> Dict[str, pd.DataFrame]:
    """Per-symbol DataFrame indexed by date with columns: close, rsi.

    Only symbols with >= min_bars daily rows are included.
    """
    out: Dict[str, pd.DataFrame] = {}
    for sym, grp in daily_df.groupby("symbol", sort=False):
        if len(grp) < min_bars:
            continue
        g = grp.sort_values("ts").reset_index(drop=True)
        closes = g["close"].to_numpy(dtype=np.float64)
        rsi = _wilder_rsi(closes, period)
        out[sym] = pd.DataFrame({
            "d": g["ts"].dt.date.values,
            "close": closes,
            "rsi": rsi,
        }).set_index("d")
    return out


def _rsi_sustained_dates(
    rsi_df: pd.DataFrame,
    threshold: float,
    n_days: int,
) -> set:
    """Return the set of session_dates T for which RSI[T-1..T-n_days] all >= threshold.

    rsi_df is a per-symbol DataFrame indexed by daily date with column 'rsi'.
    """
    rsi = rsi_df["rsi"].to_numpy(dtype=np.float64)
    dates = rsi_df.index.to_numpy()
    if len(rsi) < n_days + 1:
        return set()
    # Boolean array: ok[i] = RSI[i] >= threshold (and not nan)
    ok = (~np.isnan(rsi)) & (rsi >= threshold)
    # For session date at position i (T), require i-1..i-n_days all ok.
    out = set()
    for i in range(n_days, len(rsi)):
        if all(ok[i - k] for k in range(1, n_days + 1)):
            out.add(dates[i])
    return out


# -----------------------------------------------------------------------------
# Per-(symbol, day) evaluator
#
# Returns ONE record per (sym, date):
#   - is_signal=True if VWAP cross-down + vol_ratio>=1.2 fires in [09:30, 12:00)
#   - is_baseline=True if NO VWAP cross-down in [09:30, 12:00)
#   - None otherwise (e.g., bars missing target)
# -----------------------------------------------------------------------------
def evaluate_symbol_day(sym_bars: pd.DataFrame, sym: str, d: date) -> Optional[dict]:
    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    if sym_bars.empty:
        return None

    # Confine to regular session
    mask_sess = (sym_bars["time"] >= dtime(9, 15)) & (sym_bars["time"] <= dtime(15, 25))
    sym_bars = sym_bars[mask_sess].reset_index(drop=True)
    if sym_bars.empty:
        return None

    # Target bar (13:25 5m close == 13:30 IST)
    tgt_t = CONFIG["target_bar_time"]
    tgt_rows = sym_bars[sym_bars["time"] == tgt_t]
    if tgt_rows.empty:
        return None
    target_close = float(tgt_rows.iloc[0]["close"])

    # Baseline anchor bar (12:00 close)
    base_t = CONFIG["baseline_anchor_time"]
    base_rows = sym_bars[sym_bars["time"] == base_t]
    if base_rows.empty:
        return None
    baseline_close = float(base_rows.iloc[0]["close"])

    # Build per-bar arrays
    highs = sym_bars["high"].to_numpy(dtype=np.float64)
    lows = sym_bars["low"].to_numpy(dtype=np.float64)
    closes = sym_bars["close"].to_numpy(dtype=np.float64)
    vols = sym_bars["volume"].to_numpy(dtype=np.float64)
    times = sym_bars["time"].tolist()
    ts_col = sym_bars["date"].tolist()
    n = len(closes)
    if n < 3:
        return None

    # Cumulative intraday VWAP from session start (09:15) through bar i (inclusive)
    typical = (highs + lows + closes) / 3.0
    pv = typical * vols
    cum_pv = np.cumsum(pv)
    cum_v = np.cumsum(vols)
    vwap = np.full(n, np.nan, dtype=np.float64)
    nz = cum_v > 0
    vwap[nz] = cum_pv[nz] / cum_v[nz]

    # vol_baseline[i] = mean(vols[0..i-1]) -- excludes current bar
    cum_vol_prior = np.cumsum(vols) - vols  # = sum of vols[0..i-1]
    idx_arr = np.arange(n, dtype=np.float64)
    vol_baseline = np.full(n, np.nan, dtype=np.float64)
    pos = idx_arr > 0
    vol_baseline[pos] = cum_vol_prior[pos] / idx_arr[pos]

    sw_start = CONFIG["sig_window_start"]
    sw_end   = CONFIG["sig_window_end"]
    vol_min  = float(CONFIG["vol_ratio_min"])

    # First-fire scan: cross-down within [09:30, 12:00)
    for i in range(1, n):
        t = times[i]
        if t < sw_start or t >= sw_end:
            continue
        if np.isnan(vwap[i]) or np.isnan(vwap[i - 1]):
            continue
        if np.isnan(vol_baseline[i]) or vol_baseline[i] <= 0:
            continue
        # Cross-down: prior bar close >= prior VWAP, current bar close < current VWAP
        if not (closes[i - 1] >= vwap[i - 1] and closes[i] < vwap[i]):
            continue
        vr = float(vols[i] / vol_baseline[i])
        if vr < vol_min:
            continue
        sig_close = float(closes[i])
        ret = (target_close - sig_close) / sig_close * 100.0  # SHORT: negative is win
        return {
            "symbol": sym,
            "date": d.isoformat(),
            "signal_bar_ts": pd.Timestamp(ts_col[i]).isoformat(),
            "signal_bar_time": str(times[i]),
            "signal_bar_close": sig_close,
            "vwap_at_signal": float(vwap[i]),
            "vwap_prior": float(vwap[i - 1]),
            "close_prior": float(closes[i - 1]),
            "vol_ratio": vr,
            "ret_to_1330": ret,
            "target_close_1330": target_close,
            "is_signal": True,
            "is_baseline": False,
        }

    # No cross-down in window -> baseline anchor at 12:00 close
    ret_b = (target_close - baseline_close) / baseline_close * 100.0
    return {
        "symbol": sym,
        "date": d.isoformat(),
        "signal_bar_ts": None,
        "signal_bar_time": None,
        "signal_bar_close": baseline_close,
        "vwap_at_signal": float("nan"),
        "vwap_prior": float("nan"),
        "close_prior": float("nan"),
        "vol_ratio": float("nan"),
        "ret_to_1330": ret_b,
        "target_close_1330": target_close,
        "is_signal": False,
        "is_baseline": True,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature -- 5day_RSI_overbought_intraday_VWAP_lose_short")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print(f"Caps:   {sorted(CONFIG['accepted_caps'])}  (large_cap + unknown EXCLUDED)")
    print(f"RSI:    period={CONFIG['rsi_period']} Wilder, threshold={CONFIG['rsi_threshold']}, "
          f"sustained_days={CONFIG['rsi_sustained_days']}")
    print(f"Signal: cumulative-VWAP cross-down in [{CONFIG['sig_window_start']}, {CONFIG['sig_window_end']}) "
          f"+ vol_ratio >= {CONFIG['vol_ratio_min']}")
    print(f"Target: close at {CONFIG['target_bar_time']} (= 13:30 IST exit)")
    print("=" * 80)

    # 1. Load daily + build per-symbol RSI lookups
    _log("Loading consolidated_daily.feather ...")
    daily_df = pd.read_feather(CONFIG["daily_path"])
    daily_df["ts"] = _ensure_naive_ist(daily_df["ts"])
    # Confine to <= window_end + buffer (RSI needs prior days; full history kept)
    daily_df = daily_df[daily_df["ts"] <= pd.Timestamp(CONFIG["window_end"])]
    _log(f"  daily rows: {len(daily_df):,}  symbols: {daily_df['symbol'].nunique():,}")

    rsi_period = int(CONFIG["rsi_period"])
    min_bars = int(CONFIG["min_daily_bars_required"])
    _log(f"Computing Wilder RSI({rsi_period}) per symbol (min {min_bars} daily bars) ...")
    rsi_lookup = _build_rsi_lookup(daily_df, period=rsi_period, min_bars=min_bars)
    _log(f"  RSI built for {len(rsi_lookup):,} symbols")

    # Pre-compute sustained-date sets per (symbol, duration)
    threshold = float(CONFIG["rsi_threshold"])
    durations = list(CONFIG["rsi_durations_to_test"])
    sustained_by_duration: Dict[int, Dict[str, set]] = {dur: {} for dur in durations}
    # Single-day cohort (RSI[T-1] >= threshold only) for Falsifier #1
    single_day_by_sym: Dict[str, set] = {}

    _log("Building sustained-RSI date sets per symbol ...")
    for sym, rdf in rsi_lookup.items():
        # Single-day: T-1 RSI >= threshold (no requirement on T-2/T-3)
        rsi = rdf["rsi"].to_numpy(dtype=np.float64)
        dates = rdf.index.to_numpy()
        ok = (~np.isnan(rsi)) & (rsi >= threshold)
        # session date i has T-1 at position i-1
        sd_set = set()
        for i in range(1, len(rsi)):
            if ok[i - 1]:
                sd_set.add(dates[i])
        single_day_by_sym[sym] = sd_set
        # Sustained for each duration
        for dur in durations:
            sustained_by_duration[dur][sym] = _rsi_sustained_dates(rdf, threshold, dur)

    primary_dur = int(CONFIG["rsi_sustained_days"])
    primary_set = sustained_by_duration[primary_dur]
    _log(f"  primary cohort (sustained {primary_dur}-day RSI>={threshold}): "
         f"{sum(len(v) for v in primary_set.values()):,} (sym,date) pairs")
    _log(f"  single-day cohort (RSI[T-1]>={threshold}): "
         f"{sum(len(v) for v in single_day_by_sym.values()):,} (sym,date) pairs")

    # 2. ProductionUniverseGate
    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )
    nse_all = gate._load_nse_all()
    accepted = CONFIG["accepted_caps"]
    keep_syms_cap = {s for s, row in nse_all.items() if row.cap_segment in accepted}
    _log(f"Cap-eligible (small_cap+mid_cap) symbols: {len(keep_syms_cap):,}")

    # 3. Per-month 5m scan -- two passes over the same (sym,day) iter:
    #    (a) PRIMARY (sustained N-day) records for the main analysis CSV
    #    (b) SINGLE-DAY cohort records for STEP 1 Falsifier #1
    #    Also: ALT-DURATION cohorts (5-day, 7-day) for STEP 3 splits
    primary_records: List[dict] = []
    single_day_records: List[dict] = []
    alt_dur_records: Dict[int, List[dict]] = {dur: [] for dur in durations if dur != primary_dur}

    months = _months_between(CONFIG["window_start"], CONFIG["window_end"])
    total_pairs_seen = 0
    total_universe_pass = 0
    total_primary_pass = 0

    for (yy, mm) in months:
        path = Path(CONFIG["monthly_5m_dir"]) / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            _log(f"  skip {yy:04d}-{mm:02d} (no 5m feather)")
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

        # Pre-filter by cap + RSI eligibility
        df = df[df["symbol"].isin(keep_syms_cap)]
        if df.empty:
            continue

        month_pairs = 0
        month_univ = 0
        month_prim = 0
        month_sd = 0
        month_alt = {dur: 0 for dur in alt_dur_records.keys()}

        for (sym, d), bars in df.groupby(["symbol", "day"], sort=False):
            month_pairs += 1
            total_pairs_seen += 1
            # ProductionUniverseGate per-date
            if not gate.is_eligible(sym, d):
                continue
            month_univ += 1
            total_universe_pass += 1

            # Skip if symbol not in RSI lookup (insufficient daily bars)
            if sym not in rsi_lookup:
                continue

            # PRIMARY cohort: 3-day sustained
            if d in primary_set.get(sym, ()):
                rec = evaluate_symbol_day(bars, sym, d)
                if rec is not None:
                    primary_records.append(rec)
                    month_prim += 1
                    total_primary_pass += 1

            # SINGLE-DAY cohort (Falsifier #1)
            if d in single_day_by_sym.get(sym, ()):
                rec = evaluate_symbol_day(bars, sym, d)
                if rec is not None:
                    single_day_records.append(rec)
                    month_sd += 1

            # ALT-DURATION cohorts (5-day, 7-day) for STEP 3 splits
            for dur in alt_dur_records.keys():
                if d in sustained_by_duration[dur].get(sym, ()):
                    rec = evaluate_symbol_day(bars, sym, d)
                    if rec is not None:
                        alt_dur_records[dur].append(rec)
                        month_alt[dur] += 1

        alt_str = " ".join(f"alt{dur}={month_alt[dur]:,}" for dur in alt_dur_records.keys())
        _log(
            f"  {yy:04d}-{mm:02d}: pairs={month_pairs:,} univ={month_univ:,} "
            f"primary={month_prim:,} single_day={month_sd:,} {alt_str}"
        )

    print()
    print("=== Pipeline tally ===")
    print(f"  total (sym, day) pairs seen:         {total_pairs_seen:,}")
    print(f"  passed universe gate:                {total_universe_pass:,}")
    print(f"  primary ({primary_dur}-day) recorded:           {total_primary_pass:,}")
    print(f"  single-day recorded:                 {len(single_day_records):,}")
    for dur, recs in alt_dur_records.items():
        print(f"  alt-duration {dur}-day recorded:        {len(recs):,}")

    if not primary_records:
        print("\nNO PRIMARY RECORDS COLLECTED -- abort.")
        return 1

    # 4. Build DataFrames + cohort attribution
    df_primary = pd.DataFrame.from_records(primary_records)

    def _cap(s: str) -> str:
        row = nse_all.get(s)
        return row.cap_segment if row else "unknown"

    df_primary["cap_segment"] = df_primary["symbol"].map(_cap)

    dt_dates = pd.to_datetime(df_primary["date"])
    cut_2024 = pd.Timestamp(CONFIG["regime_2024_cut"])
    cut_sebi = pd.Timestamp(CONFIG["sebi_oct2025_cut"])
    df_primary["regime_pre_post_2024"] = np.where(dt_dates < cut_2024, "pre", "post")
    df_primary["period_pre_post_sebi_oct2025"] = np.where(dt_dates < cut_sebi, "pre", "post")

    def _vr_bucket(v: float) -> str:
        if pd.isna(v):
            return "nan"
        if v < 1.5:
            return "1.2-1.5"
        if v < 2.0:
            return "1.5-2.0"
        if v < 3.0:
            return "2.0-3.0"
        return ">=3.0"

    df_primary["vol_ratio_bucket"] = df_primary["vol_ratio"].map(_vr_bucket)

    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_primary.to_csv(out_path, index=False)
    print(f"\nSaved primary cohort: {len(df_primary):,} rows -> {out_path}")

    sig = df_primary[df_primary["is_signal"]].copy()
    base = df_primary[df_primary["is_baseline"]].copy()
    n_sig = len(sig)
    n_base = len(base)

    # =========================================================================
    # STEP 1 -- Falsifier #1: 3-day SUSTAINED vs SINGLE-DAY drift comparison
    # =========================================================================
    print()
    print("=" * 80)
    print(f"STEP 1 -- Falsifier #1 (sustained-{primary_dur}-day vs single-day RSI)")
    print("=" * 80)

    def _drift_delta(records: List[dict]) -> Tuple[int, int, float]:
        """Return (n_signal, n_baseline, sig_mean - base_mean)."""
        if not records:
            return 0, 0, float("nan")
        ddf = pd.DataFrame.from_records(records)
        s = ddf[ddf["is_signal"]]
        b = ddf[ddf["is_baseline"]]
        ns, nb = len(s), len(b)
        if ns == 0 or nb == 0:
            return ns, nb, float("nan")
        return ns, nb, float(s["ret_to_1330"].mean()) - float(b["ret_to_1330"].mean())

    ns_sus, nb_sus, delta_sus = _drift_delta(primary_records)
    ns_sd,  nb_sd,  delta_sd  = _drift_delta(single_day_records)

    print(f"  {primary_dur}-day SUSTAINED cohort:  n_sig={ns_sus:,}  n_base={nb_sus:,}  drift_delta={delta_sus:+.4f}%")
    print(f"  SINGLE-DAY cohort:        n_sig={ns_sd:,}  n_base={nb_sd:,}  drift_delta={delta_sd:+.4f}%")

    if np.isnan(delta_sus) or np.isnan(delta_sd):
        print("\n  Cannot evaluate Falsifier #1 (NaN drift delta). Verdict: KILL.")
        print("\n" + "=" * 80)
        print("VERDICT: KILL (insufficient data for Falsifier #1)")
        print("=" * 80)
        return 0

    separation = abs(delta_sus - delta_sd)
    min_sep = float(CONFIG["falsifier1_min_separation"])
    print(f"  |sustained - single_day|: {separation:.4f}%   [>= {min_sep:.2f}% required]")

    falsifier1_pass = separation >= min_sep
    print(f"\n  FALSIFIER #1: {'PASS' if falsifier1_pass else 'FAIL'}")

    if not falsifier1_pass:
        print("\n  STOP -- Falsifier #1 FAILED.")
        print(f"  Sustained-{primary_dur}-day cohort drift = single-day drift "
              f"(within {min_sep}% tolerance). 'Sustained = positioning saturation' "
              "thesis NOT supported; RSI is just noise either way.")
        print("\n" + "=" * 80)
        print("VERDICT: KILL (Falsifier #1 failed -- sustained adds no value)")
        print("=" * 80)
        return 0

    # =========================================================================
    # STEP 2 -- Aggregate drift delta (3-day sustained)
    # =========================================================================
    print()
    print("=" * 80)
    print(f"STEP 2 -- Aggregate drift delta ({primary_dur}-day sustained)")
    print("=" * 80)

    sig_mean = float(sig["ret_to_1330"].mean()) if n_sig else float("nan")
    base_mean = float(base["ret_to_1330"].mean()) if n_base else float("nan")
    sig_med = float(sig["ret_to_1330"].median()) if n_sig else float("nan")
    base_med = float(base["ret_to_1330"].median()) if n_base else float("nan")
    delta = (sig_mean - base_mean) if (n_sig and n_base) else float("nan")

    print(f"  n_signal:                   {n_sig:,}")
    print(f"  n_baseline:                 {n_base:,}")
    print(f"  Signal   mean ret_to_1330:  {sig_mean:+.4f}%  (median {sig_med:+.4f}%)")
    print(f"  Baseline mean ret_to_1330:  {base_mean:+.4f}%  (median {base_med:+.4f}%)")
    print(f"  DRIFT DELTA:                {delta:+.4f}%   [<= {CONFIG['drift_delta_max']:.2f}% required]")
    print(f"  n acceptance:               {n_sig} {'>=' if n_sig >= CONFIG['n_signal_min'] else '<'} {CONFIG['n_signal_min']}")

    # =========================================================================
    # STEP 3 -- Cohort splits
    # =========================================================================
    print()
    print("=" * 80)
    print("STEP 3 -- Cohort splits")
    print("=" * 80)
    print(f"{'split':<36}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")

    def _split_block(label: str, mask_sig: pd.Series, mask_base: pd.Series) -> Tuple[int, float, float, float]:
        s = sig[mask_sig]
        b = base[mask_base]
        ns, nb = len(s), len(b)
        sm = float(s["ret_to_1330"].mean()) if ns else float("nan")
        bm = float(b["ret_to_1330"].mean()) if nb else float("nan")
        dl = (sm - bm) if (ns and nb) else float("nan")
        print(
            f"{label:<36}"
            f"{ns:>8d}"
            f"{nb:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )
        return ns, sm, bm, dl

    sig_dates = pd.to_datetime(sig["date"])
    base_dates = pd.to_datetime(base["date"])

    # pre/post 2024
    _split_block("pre_2024",            sig_dates < cut_2024,  base_dates < cut_2024)
    _split_block("post_2024",           sig_dates >= cut_2024, base_dates >= cut_2024)

    # pre/post SEBI Oct 2025 (MANDATORY)
    pre_sebi_n, pre_sebi_sm, pre_sebi_bm, pre_sebi_delta = _split_block(
        "pre_sebi_oct2025",    sig_dates < cut_sebi,  base_dates < cut_sebi
    )
    post_sebi_n, post_sebi_sm, post_sebi_bm, post_sebi_delta = _split_block(
        "post_sebi_oct2025",   sig_dates >= cut_sebi, base_dates >= cut_sebi
    )

    # cap=small vs mid
    for cap_val in ["small_cap", "mid_cap"]:
        _split_block(f"cap={cap_val}",  sig["cap_segment"] == cap_val, base["cap_segment"] == cap_val)

    # RSI-duration sweep (3-day vs 5-day vs 7-day)
    print()
    print("  --- RSI-duration sweep ---")
    print(f"{'split':<36}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    # primary (3-day) row first
    print(
        f"{'rsi_duration=' + str(primary_dur) + '_day':<36}"
        f"{n_sig:>8d}"
        f"{n_base:>8d}"
        f"{sig_mean:>12.4f}"
        f"{base_mean:>12.4f}"
        f"{delta:>12.4f}"
    )
    for dur, recs in alt_dur_records.items():
        if not recs:
            print(f"rsi_duration={dur}_day                       (no records)")
            continue
        ddf = pd.DataFrame.from_records(recs)
        ss = ddf[ddf["is_signal"]]
        bb = ddf[ddf["is_baseline"]]
        nss, nbb = len(ss), len(bb)
        sm = float(ss["ret_to_1330"].mean()) if nss else float("nan")
        bm = float(bb["ret_to_1330"].mean()) if nbb else float("nan")
        dl = (sm - bm) if (nss and nbb) else float("nan")
        print(
            f"{'rsi_duration=' + str(dur) + '_day':<36}"
            f"{nss:>8d}"
            f"{nbb:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )

    # vol_ratio buckets (only meaningful for signal rows; baseline has vol_ratio=nan)
    print()
    print("  --- vol_ratio_bucket (signal rows; baseline = aggregate baseline) ---")
    print(f"{'bucket':<36}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    for bucket in ["1.2-1.5", "1.5-2.0", "2.0-3.0", ">=3.0"]:
        mask_s = sig["vol_ratio_bucket"] == bucket
        s = sig[mask_s]
        ns = len(s)
        sm = float(s["ret_to_1330"].mean()) if ns else float("nan")
        bm = base_mean
        dl = (sm - bm) if (ns and not np.isnan(bm)) else float("nan")
        print(
            f"{'vol_ratio_bucket=' + bucket:<36}"
            f"{ns:>8d}"
            f"{n_base:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )

    # =========================================================================
    # VERDICT
    # =========================================================================
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)

    drift_ok = (not np.isnan(delta)) and (delta <= float(CONFIG["drift_delta_max"]))
    n_ok = n_sig >= int(CONFIG["n_signal_min"])
    # post-SEBI sign-flip = positive drift (opposite of SHORT direction) with sufficient n
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
            f"({post_sebi_delta:+.4f}%, n_sig={post_sebi_n}) -- "
            f"regulatory regime risk (inherited from retired `mis_unwind_vwap_revert_short`)"
        )
    else:
        verdict = "PROCEED to Phase 3"
        reason = (
            f"Falsifier #1 PASS (sep={separation:.4f}%), "
            f"drift {delta:+.4f}% <= {CONFIG['drift_delta_max']:.2f}%, "
            f"n_signal {n_sig} >= {CONFIG['n_signal_min']}, "
            f"post-SEBI delta {post_sebi_delta:+.4f}% (no sign flip)"
        )

    print(f"  {verdict}")
    print(f"  reason: {reason}")

    if sign_flip:
        print()
        print("  REGIME-RISK NOTE: predecessor `mis_unwind_vwap_revert_short` was retired")
        print("  due to post-SEBI-Oct-2025 decay (RSI-family setup). Positive post-SEBI drift")
        print("  here is a sign-flip and must be treated as DEFER, NOT proceed.")

    return 0


if __name__ == "__main__":
    sys.exit(run())
