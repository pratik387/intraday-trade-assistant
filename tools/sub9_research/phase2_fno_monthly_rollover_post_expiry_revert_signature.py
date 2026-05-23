# tools/sub9_research/phase2_fno_monthly_rollover_post_expiry_revert_signature.py
#
# Phase 2 empirical signature for `fno_monthly_rollover_intraday_post_expiry_revert`.
# See specs/2026-05-22-brief-fno_monthly_rollover_intraday_post_expiry_revert.md
#
# Mechanism (restated):
#   On T+1 (the trading session IMMEDIATELY AFTER an NSE monthly F&O expiry),
#   F&O-liquid-200 underlyings exhibit a measurable 09:15-10:30 morning-gap
#   reversal because of (a) overnight institutional position resets, (b) cash
#   settlement clearing flow, and (c) forced retail re-entry into the new
#   front-month -- a transient supply/demand imbalance that mean-reverts within
#   the first hour. Direction is gap-sign-driven: gap-DOWN >= 0.5%  => LONG
#   mean-revert; gap-UP >= 0.5% => SHORT mean-revert.
#
# Critical regime cutover (Falsifier #3):
#   NSE Circular FAOP68747 shifted monthly expiry from "last Thursday" to
#   "last Tuesday" effective 2025-09-01. T+1 day shifted Friday -> Wednesday.
#   Phase 2 MUST split pre-2025-09 vs post-2025-09 cohorts.
#
# Direction-asymmetric signal (per brief Phase 2 specification):
#   - For each (sym, T+1 date), compute gap_pct = (open_at_0915 / PDC[T0]) - 1
#   - LONG entry candidate: gap_pct <= -0.005  (gap-DOWN 0.5%+)
#   - SHORT entry candidate: gap_pct >= +0.005  (gap-UP 0.5%+)
#   - Signal close = close of the 09:15 5m bar (= wall-clock 09:20). This is
#     "the actual gap pierce" per brief.
#
# Target measurements:
#   - ret_to_1030 = (close_at_1025 - signal_close) / signal_close * 100
#       (bar labeled 10:25 closes at wall-clock 10:30 per project convention)
#   - mean_revert_R = -signal_sign * ret_to_1030  (signal_sign = +1 LONG, -1 SHORT)
#       Positive mean_revert_R => mean-revert CONFIRMED.
#   - ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100
#       (informational extended hold; bar 13:25 closes at 13:30)
#
# Baseline (control):
#   All F&O-liquid-200 (sym, date) pairs NOT on T+1 of monthly expiry, with
#   |gap_pct| >= 0.005, same gap-direction split.
#
# Anti-bias guards (Lesson #5 #1, Lesson #5 #2, Lesson #19):
#   1. PDC strictly from consolidated_daily for T-1 expiry day (NOT signal day).
#   2. No volume baseline used; no day-aggregate look-ahead.
#   3. First-fire-per-day-per-stock latch (one row per (sym, date)).
#   4. ProductionUniverseGate per-date (Lesson #19) -- aligned with production
#      universe builder; accepted_caps = {"large_cap", "mid_cap"} for F&O
#      eligibility (F&O underlyings span both bands).
#   5. T+1 dates derived from data/futures_basis/2023_2026_basis.parquet
#      (canonical NSE-settled monthly expiry calendar) + assets/nse_holidays.json
#      (next-trading-session skip rule).
#   6. NO exit walk -- pure signature measurement (Phase 2 discipline).
#
# Pre-registration discipline (per brief Falsifiers):
#   - Falsifier #1 (gap-magnitude signature): signal cohort median |gap_pct|
#     should be MATERIALLY > baseline median |gap_pct| (T+1 wider-gap claim).
#     Reported, not auto-killed.
#   - Falsifier #3 (regime split MANDATORY): pre-2025-09-01 vs post-2025-09-01
#     mean_revert_R must show same sign and similar magnitude. Reported.
#
# Required cohort splits: pre/post 2025-09-01 (regime), pre/post-2024 (standard),
#   LONG vs SHORT (direction asymmetry), gap-magnitude buckets, cap_segment,
#   expiry-day-of-week (Thursday vs Tuesday -- mirrors the 2025-09-01 cut).
"""Phase 2 empirical signature - fno_monthly_rollover_intraday_post_expiry_revert."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402

# -----------------------------------------------------------------------------
# CONFIG - NO hardcoded defaults inside the logic; every knob is declared here.
# -----------------------------------------------------------------------------
CONFIG: Dict[str, object] = {
    # Date window (full history; discovery/oos/holdout slices applied at analysis time)
    "window_start": date(2023, 1, 2),
    "window_end":   date(2026, 4, 30),

    # Discovery / OOS / Holdout slice boundaries (standard project convention)
    "discovery_start": date(2023, 1, 2),
    "discovery_end":   date(2024, 12, 31),
    "oos_start":       date(2025, 1, 1),
    "oos_end":         date(2025, 12, 31),
    "holdout_start":   date(2026, 1, 1),
    "holdout_end":     date(2026, 4, 30),

    # Universe (F&O liquid 200: large_cap + mid_cap, MIS-eligible)
    "accepted_caps":             {"large_cap", "mid_cap"},
    "require_mis":               True,
    "min_trading_days_required": 0,   # Lesson #17: disabled
    "min_daily_avg_volume":      0,   # Lesson #17: disabled

    # Gap signal thresholds
    "gap_pct_abs_min":           0.005,   # |gap| >= 0.5% required for signal-side eligibility

    # Signal / target bars (5m bar timestamp = bar OPEN per project convention)
    # Signal: 09:15 bar's CLOSE (i.e., the bar with time=09:15, closing at 09:20).
    # ret_to_1030: close of bar with time=10:25 (closes at 10:30).
    # ret_to_1330: close of bar with time=13:25 (closes at 13:30).
    "signal_bar_time":           dtime(9, 15),
    "target_bar_time_1030":      dtime(10, 25),
    "target_bar_time_1330":      dtime(13, 25),

    # Regime cuts (Falsifier #3 calendar cutover + standard cuts)
    "regime_fao_circular_cut":   date(2025, 9, 1),    # MANDATORY pre/post split
    "regime_2024_cut":           date(2024, 1, 1),

    # Paths
    "fno_universe_csv":  _REPO_ROOT / "assets" / "fno_liquid_200.csv",
    "nse_holidays_json": _REPO_ROOT / "assets" / "nse_holidays.json",
    "stock_sector_map":  _REPO_ROOT / "assets" / "stock_sector_map.json",
    "futures_basis":     _REPO_ROOT / "data" / "futures_basis" / "2023_2026_basis.parquet",
    "monthly_5m_dir":    _REPO_ROOT / "backtest-cache-download" / "monthly",
    "daily_path":        _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather",
    "out_csv":           _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_fno_monthly_rollover_post_expiry_revert_signature.csv",
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


def _normalize_symbol(s: str) -> str:
    """Strip 'NSE:' prefix and '.NS' suffix to bare symbol form."""
    if ":" in s:
        s = s.split(":")[-1]
    if "." in s:
        s = s.split(".")[0]
    return s


# -----------------------------------------------------------------------------
# Load F&O liquid 200 universe (bare symbols).
# -----------------------------------------------------------------------------
def _load_fno_universe(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise RuntimeError(f"{path} missing 'symbol' column")
    return [_normalize_symbol(str(s)) for s in df["symbol"].tolist() if str(s).strip()]


# -----------------------------------------------------------------------------
# Build NSE-holiday set (date objects), used to skip weekend+holidays when
# deriving T+1 of each monthly expiry.
# -----------------------------------------------------------------------------
def _load_holiday_set(path: Path) -> Set[date]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out: Set[date] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        s = str(r.get("tradingDate", "")).strip()
        if not s:
            continue
        try:
            d = datetime.strptime(s, "%d-%b-%Y").date()
        except ValueError:
            try:
                d = datetime.strptime(s, "%Y-%m-%d").date()
            except ValueError:
                continue
        out.add(d)
    return out


def _next_trading_session(d: date, holidays: Set[date]) -> date:
    """Return next trading session strictly after d (skip weekends + NSE holidays)."""
    cur = d + timedelta(days=1)
    # 0=Mon ... 4=Fri, 5=Sat, 6=Sun
    while cur.weekday() >= 5 or cur in holidays:
        cur += timedelta(days=1)
    return cur


# -----------------------------------------------------------------------------
# Load stock_sector_map (NSE: prefix keys -> sector_id string).
# -----------------------------------------------------------------------------
def _load_sector_map(path: Path) -> Dict[str, str]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        if not isinstance(v, str):
            continue
        out[_normalize_symbol(str(k))] = v
    return out


# -----------------------------------------------------------------------------
# Build PDC lookup from consolidated_daily.feather (per-symbol date->close).
# -----------------------------------------------------------------------------
def _build_pdc_lookup(daily_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym, grp in daily_df.groupby("symbol", sort=False):
        g = grp.sort_values("ts").reset_index(drop=True)
        out[sym] = pd.DataFrame({
            "d": g["ts"].dt.date.values,
            "close": g["close"].to_numpy(dtype=np.float64),
        }).set_index("d")
    return out


def _close_on(pdc_df: pd.DataFrame, d: date) -> Optional[float]:
    """Daily close on exactly date `d`, else the most recent <= d. None if unavailable."""
    arr = np.asarray(pdc_df.index)
    mask = arr <= d
    if not mask.any():
        return None
    last = arr[mask][-1]
    return float(pdc_df.loc[last, "close"])


# -----------------------------------------------------------------------------
# Build (T0_expiry -> T_plus_1) mapping from futures_basis parquet + holidays.
# Returns dict: T_plus_1_date -> T0_expiry_date.
# -----------------------------------------------------------------------------
def _build_tplus1_map(parquet_path: Path, holidays: Set[date]) -> Dict[date, date]:
    df = pd.read_parquet(parquet_path, columns=["expiry_date"])
    expiries = sorted({d for d in df["expiry_date"].tolist() if d is not None})
    out: Dict[date, date] = {}
    for t0 in expiries:
        # `expiry_date` column already stores actual NSE-settled date (Thursday/Tuesday/holiday-adjusted).
        # T+1 = next trading session strictly after t0.
        tp1 = _next_trading_session(t0, holidays)
        out[tp1] = t0
    return out


# -----------------------------------------------------------------------------
# Per-(symbol, date) evaluator.
# Returns ONE row per (sym, d) when:
#   - 09:15 bar exists (gap_pct computable)
#   - 10:25 bar exists (ret_to_1030 computable)
#   - |gap_pct| >= 0.005
# Otherwise None.
# -----------------------------------------------------------------------------
def evaluate_symbol_day(
    sym_bars: pd.DataFrame,
    sym: str,
    d: date,
    pdc: Optional[float],
    cfg: Dict[str, object],
) -> Optional[dict]:
    if sym_bars.empty or pdc is None or pdc <= 0:
        return None

    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)

    sig_t = cfg["signal_bar_time"]          # dtime(9, 15)
    tgt_1030_t = cfg["target_bar_time_1030"]  # dtime(10, 25)
    tgt_1330_t = cfg["target_bar_time_1330"]  # dtime(13, 25)

    sig_row = sym_bars[sym_bars["time"] == sig_t]
    if sig_row.empty:
        return None
    open_0915 = float(sig_row.iloc[0]["open"])
    signal_close = float(sig_row.iloc[0]["close"])
    if not np.isfinite(open_0915) or open_0915 <= 0:
        return None
    if not np.isfinite(signal_close) or signal_close <= 0:
        return None

    tgt_1030_row = sym_bars[sym_bars["time"] == tgt_1030_t]
    if tgt_1030_row.empty:
        return None
    close_at_1025 = float(tgt_1030_row.iloc[0]["close"])
    if not np.isfinite(close_at_1025) or close_at_1025 <= 0:
        return None

    # ret_to_1330 is informational; if 13:25 bar missing we still emit (set NaN).
    tgt_1330_row = sym_bars[sym_bars["time"] == tgt_1330_t]
    if not tgt_1330_row.empty:
        c1325 = float(tgt_1330_row.iloc[0]["close"])
        if np.isfinite(c1325) and c1325 > 0:
            close_at_1325 = c1325
            ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100.0
        else:
            close_at_1325 = float("nan")
            ret_to_1330 = float("nan")
    else:
        close_at_1325 = float("nan")
        ret_to_1330 = float("nan")

    gap_pct = (open_0915 / pdc) - 1.0
    gap_abs_min = float(cfg["gap_pct_abs_min"])
    if abs(gap_pct) < gap_abs_min:
        return None

    # Direction & sign
    if gap_pct > 0:
        signal_direction = "SHORT"
        signal_sign = -1
    else:
        signal_direction = "LONG"
        signal_sign = +1

    ret_to_1030 = (close_at_1025 - signal_close) / signal_close * 100.0
    mean_revert_R = -signal_sign * ret_to_1030  # positive = mean-revert confirmed

    return {
        "symbol": sym,
        "date": d.isoformat(),
        "signal_ts": pd.Timestamp.combine(d, sig_t).isoformat(),
        "prior_day_close": pdc,
        "open_0915": open_0915,
        "gap_pct": gap_pct * 100.0,   # stored in PERCENT for readability
        "signal_direction": signal_direction,
        "signal_sign": signal_sign,
        "signal_close": signal_close,
        "close_at_1025": close_at_1025,
        "ret_to_1030": ret_to_1030,
        "mean_revert_R": mean_revert_R,
        "close_at_1325": close_at_1325,
        "ret_to_1330": ret_to_1330,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    cfg = CONFIG
    print("=" * 80)
    print("Phase 2 empirical signature -- fno_monthly_rollover_intraday_post_expiry_revert")
    print(f"Window: {cfg['window_start']} -> {cfg['window_end']}")
    print(f"Universe: F&O liquid 200 (caps={sorted(cfg['accepted_caps'])}, MIS={cfg['require_mis']})")
    print(f"Signal: 09:15 5m bar close; gap_pct = open_0915 / PDC[T0] - 1")
    print(f"        |gap_pct| >= {float(cfg['gap_pct_abs_min'])*100:.2f}%, gap<0 -> LONG, gap>0 -> SHORT")
    print(f"Targets: ret_to_1030 (bar 10:25 close = 10:30 IST); ret_to_1330 (informational)")
    print(f"Regime cut (Falsifier #3, MANDATORY): pre/post {cfg['regime_fao_circular_cut']}")
    print("=" * 80)

    # 1. Load F&O liquid 200 universe
    fno_syms = _load_fno_universe(cfg["fno_universe_csv"])
    fno_set: Set[str] = set(fno_syms)
    _log(f"F&O liquid 200 universe loaded: n={len(fno_syms)} (bare symbols)")

    # 2. Load holidays + build T+1 -> T0 map from futures_basis parquet
    holidays = _load_holiday_set(cfg["nse_holidays_json"])
    _log(f"NSE holidays loaded: n={len(holidays)}")

    tplus1_map = _build_tplus1_map(cfg["futures_basis"], holidays)
    tplus1_dates = sorted(tplus1_map.keys())
    _log(f"T+1 dates derived from {len(tplus1_map)} monthly expiries (sample first 5): "
         f"{[d.isoformat() for d in tplus1_dates[:5]]}")
    if len(tplus1_dates) >= 10:
        _log(f"  (sample last 5): {[d.isoformat() for d in tplus1_dates[-5:]]}")
    # Restrict to window
    tplus1_dates = [d for d in tplus1_dates if cfg["window_start"] <= d <= cfg["window_end"]]
    _log(f"T+1 dates inside window: {len(tplus1_dates)}")

    # 3. Load sector map (optional metadata for output)
    sector_map = _load_sector_map(cfg["stock_sector_map"])
    _log(f"sector_map loaded: n={len(sector_map)} symbols")

    # 4. Load consolidated_daily + build PDC lookup (restricted to F&O 200 for speed)
    _log("Loading consolidated_daily.feather ...")
    daily_df = pd.read_feather(cfg["daily_path"])
    daily_df["ts"] = _ensure_naive_ist(daily_df["ts"])
    daily_df = daily_df[daily_df["ts"] <= pd.Timestamp(cfg["window_end"])]
    daily_df = daily_df[daily_df["symbol"].isin(fno_set)]
    _log(f"  daily rows (F&O 200): {len(daily_df):,}  symbols: {daily_df['symbol'].nunique()}")
    pdc_lookup = _build_pdc_lookup(daily_df)
    _log(f"  PDC lookup built for {len(pdc_lookup):,} symbols")

    # 5. ProductionUniverseGate
    gate = ProductionUniverseGate(
        accepted_caps=cfg["accepted_caps"],
        require_mis=cfg["require_mis"],
        min_trading_days_required=int(cfg["min_trading_days_required"]),
        min_daily_avg_volume=float(cfg["min_daily_avg_volume"]),
    )
    nse_all = gate._load_nse_all()
    fno_eligible_by_cap = {s for s in fno_set if nse_all.get(s) is not None
                           and nse_all[s].cap_segment in cfg["accepted_caps"]}
    _log(f"F&O 200 cap-eligible (large_cap|mid_cap) symbols: {len(fno_eligible_by_cap):,}")

    # 6. Per-month 5m scan -- emit signal rows (T+1) AND baseline rows (non-T+1)
    signal_records: List[dict] = []
    baseline_records: List[dict] = []

    months = _months_between(cfg["window_start"], cfg["window_end"])
    total_pairs_seen = 0
    total_universe_pass = 0
    total_gap_qualify_signal = 0
    total_gap_qualify_baseline = 0

    tplus1_set = set(tplus1_dates)

    for (yy, mm) in months:
        path = Path(cfg["monthly_5m_dir"]) / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            _log(f"  skip {yy:04d}-{mm:02d} (no 5m feather)")
            continue

        df = pd.read_feather(
            path, columns=["symbol", "date", "open", "high", "low", "close", "volume"],
        )
        df["date"] = _ensure_naive_ist(df["date"])
        df["day"] = df["date"].dt.date
        df["time"] = df["date"].dt.time
        df = df[(df["day"] >= cfg["window_start"]) & (df["day"] <= cfg["window_end"])]
        if df.empty:
            continue

        df = df[df["symbol"].isin(fno_set)]
        if df.empty:
            _log(f"  {yy:04d}-{mm:02d}: 0 F&O-200 rows")
            continue

        month_pairs = 0
        month_univ = 0
        month_sig = 0
        month_base = 0

        for (sym, d), bars in df.groupby(["symbol", "day"], sort=False):
            month_pairs += 1
            total_pairs_seen += 1

            if not gate.is_eligible(sym, d):
                continue
            month_univ += 1
            total_universe_pass += 1

            if sym not in pdc_lookup:
                continue
            # PDC for gap calc: prefer T0 expiry close (= prior trading day's close
            # for a T+1 signal). For baseline (non-T+1 dates) we use the strictly-prior
            # trading-day close. Compute via "close on or before d - 1 day" via
            # _close_on(d - 1 day) approach: easier to compute as "close on most
            # recent date strictly < d".
            arr = np.asarray(pdc_lookup[sym].index)
            mask = arr < d
            if not mask.any():
                continue
            pdc_d = arr[mask][-1]
            pdc = float(pdc_lookup[sym].loc[pdc_d, "close"])

            rec = evaluate_symbol_day(bars, sym, d, pdc, cfg)
            if rec is None:
                continue

            is_tplus1 = d in tplus1_set
            t0_expiry = tplus1_map.get(d, None) if is_tplus1 else None
            rec["is_tplus1"] = is_tplus1
            rec["T0_expiry_date"] = t0_expiry.isoformat() if t0_expiry is not None else ""
            rec["pdc_date"] = pdc_d.isoformat() if hasattr(pdc_d, "isoformat") else str(pdc_d)

            if is_tplus1:
                signal_records.append(rec)
                month_sig += 1
                total_gap_qualify_signal += 1
            else:
                baseline_records.append(rec)
                month_base += 1
                total_gap_qualify_baseline += 1

        _log(
            f"  {yy:04d}-{mm:02d}: pairs={month_pairs:,} univ={month_univ:,} "
            f"signal_rows(T+1)={month_sig:,} baseline_rows(non-T+1)={month_base:,}"
        )

    print()
    print("=== Pipeline tally ===")
    print(f"  total (sym, day) pairs seen:                 {total_pairs_seen:,}")
    print(f"  passed universe gate:                        {total_universe_pass:,}")
    print(f"  signal rows (T+1, |gap|>=0.5%):              {total_gap_qualify_signal:,}")
    print(f"  baseline rows (non-T+1, |gap|>=0.5%):        {total_gap_qualify_baseline:,}")

    if not signal_records:
        print("\nNO SIGNAL RECORDS COLLECTED -- abort.")
        return 1

    df_sig = pd.DataFrame.from_records(signal_records)
    df_base = pd.DataFrame.from_records(baseline_records)
    df_sig["cohort"] = "signal_tplus1"
    df_base["cohort"] = "baseline_non_tplus1"

    df_all = pd.concat([df_sig, df_base], ignore_index=True)

    # Cohort attribution columns
    dt_dates = pd.to_datetime(df_all["date"])
    cut_2024 = pd.Timestamp(cfg["regime_2024_cut"])
    cut_fao = pd.Timestamp(cfg["regime_fao_circular_cut"])
    df_all["regime_pre_post_2025_09"] = np.where(dt_dates < cut_fao, "pre_2025_09", "post_2025_09")
    df_all["regime_pre_post_2024"] = np.where(dt_dates < cut_2024, "pre_2024", "post_2024")

    # cap_segment
    def _cap(s: str) -> str:
        row = nse_all.get(s)
        return row.cap_segment if row else "unknown"
    df_all["cap_segment"] = df_all["symbol"].map(_cap)

    # sector_id
    df_all["sector_id"] = df_all["symbol"].map(lambda s: sector_map.get(s, ""))

    # Expiry-day-of-week (signal cohort only -- mirrors pre/post 2025-09-01 split)
    def _exp_dow(s: str) -> str:
        if not s:
            return ""
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            return ""
        return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday()]
    df_all["expiry_day_of_week"] = df_all["T0_expiry_date"].map(_exp_dow)

    # gap_magnitude_bucket (based on |gap_pct| stored in percent)
    def _gap_bucket(g_pct: float) -> str:
        if pd.isna(g_pct):
            return "nan"
        a = abs(g_pct)
        if a < 0.5:
            return "<0.5"
        if a < 1.0:
            return "0.5-1.0"
        if a < 2.0:
            return "1.0-2.0"
        return ">=2.0"
    df_all["gap_magnitude_bucket"] = df_all["gap_pct"].map(_gap_bucket)

    # Discovery / OOS / Holdout slice
    d_start = pd.Timestamp(cfg["discovery_start"])
    d_end = pd.Timestamp(cfg["discovery_end"])
    o_start = pd.Timestamp(cfg["oos_start"])
    o_end = pd.Timestamp(cfg["oos_end"])
    h_start = pd.Timestamp(cfg["holdout_start"])
    h_end = pd.Timestamp(cfg["holdout_end"])

    def _slice(ts: pd.Timestamp) -> str:
        if d_start <= ts <= d_end:
            return "discovery"
        if o_start <= ts <= o_end:
            return "oos"
        if h_start <= ts <= h_end:
            return "holdout"
        return "other"
    df_all["slice"] = dt_dates.map(_slice)

    # Save the CSV
    out_path = Path(cfg["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "signal_ts", "date", "T0_expiry_date", "symbol", "sector_id", "cap_segment",
        "prior_day_close", "pdc_date", "open_0915",
        "gap_pct", "gap_magnitude_bucket", "signal_direction", "signal_sign",
        "signal_close", "close_at_1025", "ret_to_1030", "mean_revert_R",
        "close_at_1325", "ret_to_1330",
        "cohort", "is_tplus1",
        "regime_pre_post_2025_09", "regime_pre_post_2024",
        "expiry_day_of_week", "slice",
    ]
    df_all[out_cols].to_csv(out_path, index=False)
    print(f"\nSaved cohort rows: {len(df_all):,} -> {out_path}")

    # ========================================================================
    # STEP 1 -- Aggregate mean_revert_R: signal (T+1) vs baseline (non-T+1)
    # Split by direction.
    # ========================================================================
    print()
    print("=" * 80)
    print("STEP 1 -- Aggregate mean_revert_R (signal T+1 vs baseline non-T+1)")
    print("=" * 80)
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")

    def _agg_block(label: str, mask_sig_extra: pd.Series, mask_base_extra: pd.Series) -> Tuple[int, int, float, float, float]:
        s = df_all[(df_all["cohort"] == "signal_tplus1") & mask_sig_extra]
        b = df_all[(df_all["cohort"] == "baseline_non_tplus1") & mask_base_extra]
        ns, nb = len(s), len(b)
        sm = float(s["mean_revert_R"].mean()) if ns else float("nan")
        bm = float(b["mean_revert_R"].mean()) if nb else float("nan")
        dl = (sm - bm) if (ns and nb) else float("nan")
        print(
            f"{label:<40}"
            f"{ns:>8d}"
            f"{nb:>8d}"
            f"{sm:>14.4f}"
            f"{bm:>14.4f}"
            f"{dl:>12.4f}"
        )
        return ns, nb, sm, bm, dl

    all_mask = pd.Series([True] * len(df_all), index=df_all.index)
    n_sig_total, n_base_total, sm_total, bm_total, dl_total = _agg_block("ALL (pooled)", all_mask, all_mask)

    print()
    print("  --- by direction ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")
    long_mask = df_all["signal_direction"] == "LONG"
    short_mask = df_all["signal_direction"] == "SHORT"
    _agg_block("LONG (gap-down >= 0.5%)",  long_mask,  long_mask)
    _agg_block("SHORT (gap-up >= 0.5%)",   short_mask, short_mask)

    # ========================================================================
    # STEP 2 -- Falsifier #1 (gap-magnitude signature)
    # T+1 cohort median |gap_pct| should be materially larger than baseline median |gap_pct|.
    # ========================================================================
    print()
    print("=" * 80)
    print("STEP 2 -- Falsifier #1 (gap-magnitude signature)")
    print("=" * 80)
    sig_abs = df_all.loc[df_all["cohort"] == "signal_tplus1", "gap_pct"].abs()
    base_abs = df_all.loc[df_all["cohort"] == "baseline_non_tplus1", "gap_pct"].abs()
    if len(sig_abs) and len(base_abs):
        sig_med_abs = float(sig_abs.median())
        base_med_abs = float(base_abs.median())
        sig_mean_abs = float(sig_abs.mean())
        base_mean_abs = float(base_abs.mean())
        delta_med = sig_med_abs - base_med_abs
        delta_mean = sig_mean_abs - base_mean_abs
        print(f"  signal   median |gap_pct|:   {sig_med_abs:.4f}%   mean: {sig_mean_abs:.4f}%")
        print(f"  baseline median |gap_pct|:   {base_med_abs:.4f}%   mean: {base_mean_abs:.4f}%")
        print(f"  delta median:                {delta_med:+.4f}% (signal - baseline)")
        print(f"  delta mean:                  {delta_mean:+.4f}% (signal - baseline)")
        if delta_med > 0:
            print(f"  Falsifier #1 directional check: SIGNAL gap-magnitude > baseline (expected).")
        else:
            print(f"  Falsifier #1 directional check: SIGNAL gap-magnitude NOT > baseline -- "
                  f"mechanism may not differ from generic gap-fade. INVESTIGATE.")
    else:
        print(f"  Cannot evaluate gap-magnitude signature (signal n={len(sig_abs)}, baseline n={len(base_abs)}).")

    # ========================================================================
    # STEP 3 -- Falsifier #3 (MANDATORY regime split pre/post 2025-09-01)
    # ========================================================================
    print()
    print("=" * 80)
    print("STEP 3 -- Falsifier #3 (MANDATORY regime split pre/post 2025-09-01)")
    print("=" * 80)
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")

    pre_mask = df_all["regime_pre_post_2025_09"] == "pre_2025_09"
    post_mask = df_all["regime_pre_post_2025_09"] == "post_2025_09"
    pre_n, _, pre_sm, _, _ = _agg_block("pre_2025_09 (Thursday expiry era)", pre_mask, pre_mask)
    post_n, _, post_sm, _, _ = _agg_block("post_2025_09 (Tuesday expiry era)", post_mask, post_mask)

    print()
    print("  --- regime x direction ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")
    _agg_block("pre_2025_09 / LONG",   pre_mask & long_mask,  pre_mask & long_mask)
    _agg_block("pre_2025_09 / SHORT",  pre_mask & short_mask, pre_mask & short_mask)
    _agg_block("post_2025_09 / LONG",  post_mask & long_mask, post_mask & long_mask)
    _agg_block("post_2025_09 / SHORT", post_mask & short_mask, post_mask & short_mask)

    if (not np.isnan(pre_sm)) and (not np.isnan(post_sm)):
        if (pre_sm > 0) != (post_sm > 0):
            print(f"\n  Falsifier #3 NOTE: SIGN FLIPPED across 2025-09-01 cutover "
                  f"(pre={pre_sm:+.4f}%, post={post_sm:+.4f}%). "
                  f"Calendar-shift may have broken mechanism.")
        else:
            print(f"\n  Falsifier #3 NOTE: signs CONSISTENT across cutover "
                  f"(pre={pre_sm:+.4f}%, post={post_sm:+.4f}%).")

    # ========================================================================
    # STEP 4 -- Cohort splits
    # ========================================================================
    print()
    print("=" * 80)
    print("STEP 4 -- Cohort splits")
    print("=" * 80)
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")

    # discovery / oos / holdout
    print()
    print("  --- discovery / OOS / holdout ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")
    for slice_label in ["discovery", "oos", "holdout"]:
        mask = df_all["slice"] == slice_label
        _agg_block(slice_label, mask, mask)

    # pre/post 2024
    print()
    print("  --- pre/post 2024 ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")
    pre24 = df_all["regime_pre_post_2024"] == "pre_2024"
    post24 = df_all["regime_pre_post_2024"] == "post_2024"
    _agg_block("pre_2024",  pre24,  pre24)
    _agg_block("post_2024", post24, post24)

    # gap_magnitude_bucket
    print()
    print("  --- gap_magnitude_bucket ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")
    for bucket in ["0.5-1.0", "1.0-2.0", ">=2.0"]:
        mb = df_all["gap_magnitude_bucket"] == bucket
        _agg_block(f"gap_mag={bucket}", mb, mb)

    # cap_segment
    print()
    print("  --- cap_segment ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")
    for cap_val in ["large_cap", "mid_cap"]:
        mc = df_all["cap_segment"] == cap_val
        _agg_block(f"cap={cap_val}", mc, mc)

    # expiry-day-of-week (signal cohort only; baseline = aggregate baseline)
    print()
    print("  --- expiry-day-of-week (signal cohort only; baseline = aggregate baseline) ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean_R':>14}{'base_mean_R':>14}{'delta':>12}")
    sig_only = df_all[df_all["cohort"] == "signal_tplus1"]
    base_only = df_all[df_all["cohort"] == "baseline_non_tplus1"]
    base_mean_R = float(base_only["mean_revert_R"].mean()) if len(base_only) else float("nan")
    n_base_agg = len(base_only)
    for dow_val in ["Thu", "Tue", "Wed", "Mon", "Fri"]:
        s = sig_only[sig_only["expiry_day_of_week"] == dow_val]
        ns = len(s)
        if ns == 0:
            continue
        sm = float(s["mean_revert_R"].mean())
        bm = base_mean_R
        dl = (sm - bm) if (not np.isnan(bm)) else float("nan")
        print(
            f"{'exp_dow=' + dow_val:<40}"
            f"{ns:>8d}"
            f"{n_base_agg:>8d}"
            f"{sm:>14.4f}"
            f"{bm:>14.4f}"
            f"{dl:>12.4f}"
        )

    # ========================================================================
    # End-of-run summary
    # ========================================================================
    print()
    print("=" * 80)
    print("Phase 2 signature run complete -- review splits above for direction/regime stability")
    print("=" * 80)
    print(f"  pooled    n_sig={n_sig_total:,}  delta={dl_total:+.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(run())
