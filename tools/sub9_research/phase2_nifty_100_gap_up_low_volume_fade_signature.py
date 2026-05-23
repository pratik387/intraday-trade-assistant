# tools/sub9_research/phase2_nifty_100_gap_up_low_volume_fade_signature.py
#
# Phase 2 empirical signature for `nifty_100_gap_up_low_volume_followthrough_fade_short`.
# See specs/2026-05-22-brief-nifty_100_gap_up_low_volume_followthrough_fade_short.md
#
# This is a SHORT-fade setup on NIFTY 100 (top-100 by market cap) large-cap
# stocks that gap UP >= +0.8% at 09:15 open AND have a first-hour (09:15-10:15)
# cumulative volume LESS THAN 0.7x the same-stock prior-5-day same-window mean.
# Mechanism: low first-hour volume signals institutional skip (gap was driven
# by retail / pre-market headline FOMO, not real flow). Institutional supply
# concentrates in 10:30-13:30 to fade the un-absorbed retail gap.
#
# Signal: at the 10:15 5m bar's close (= 10:10-10:15 bar). Entry: 10:15 close.
# Target: ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100.
# (The 13:25 5m bar closes at wall-clock 13:30 IST per project convention.)
#
# Anti-bias guards (Lesson #5 #1+#2, Lesson #19):
#   1. PDC strictly from consolidated_daily for T-1 (NOT today).
#   2. 5-day first-hour-volume baseline uses PRIOR 5 trading days, same window,
#      same symbol; current-day's first-hour volume is NOT included in the
#      baseline.
#   3. First-fire-per-day latch: each (sym, T+0) emits AT MOST ONE row.
#   4. ProductionUniverseGate per-date (Lesson #19) -- aligned with production
#      universe builder; large_cap-only.
#   5. NIFTY 100 list source-of-truth: assets/nifty_100_universe.json (built
#      from nse_all.json: market_cap_cr > 0 AND mis_enabled, top-100).
#   6. NO exit walk -- pure signature measurement (Phase 2 discipline).
#   7. Falsifier #1 metric (post-11:00 to 13:30 volume ratio vs prior-5d
#      same-window average) attached to each row for downstream split analysis.
#
# Pre-registration (per brief):
#   - Acceptance: signal-cohort mean ret_to_1330 drift <= -0.20% AND n_signal >= 200.
#   - Falsifier #1: on FADE days, post-11:00-13:30 volume_ratio >= 1.0 (institutional sell-tell).
#   - Cohort splits computed: gap_pct_bucket (0.8-1.5 / 1.5-2.5 / >=2.5),
#     pre/post-2024, pre/post-SEBI-Oct-2025, vol-ratio-bucket (<0.4 / 0.4-0.55 / 0.55-0.7).
"""Phase 2 empirical signature - nifty_100_gap_up_low_volume_followthrough_fade_short."""
from __future__ import annotations

import json
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
    # Date window: full 2023-01 to 2026-04 (matches brief discovery+OOS+holdout).
    "window_start":   date(2023, 1, 2),
    "window_end":     date(2026, 4, 30),

    # Universe
    "accepted_caps":             {"large_cap"},
    "require_mis":               True,
    "min_trading_days_required": 0,   # Lesson #17
    "min_daily_avg_volume":      0,   # Lesson #17

    # Gap signal
    "gap_pct_min":               0.008,   # +0.8% open vs prior-day-close

    # First-hour volume window (09:15-10:15 inclusive of 10:10-10:15 bar)
    "first_hour_start":          dtime(9, 15),
    "first_hour_end":            dtime(10, 15),   # INCLUSIVE upper (10:10-10:15 bar's time-stamp)
    "first_hour_baseline_days":  5,
    "first_hour_vol_ratio_max":  0.7,     # signal cohort: <= 0.7x prior-5d
    # Baseline (control cohort) uses ratio > this same threshold.

    # Signal & target bars (5m bar time-stamp = bar OPEN time per project convention)
    "signal_bar_time":           dtime(10, 15),   # 10:10-10:15 close
    "target_bar_time":           dtime(13, 25),   # 13:25 5m bar close == 13:30 IST exit

    # Falsifier #1: post-11:00 to 13:30 volume ratio window
    "post_window_start":         dtime(11, 0),
    "post_window_end":           dtime(13, 25),   # inclusive; last 5m bar at 13:25 (close = 13:30)
    "post_window_baseline_days": 5,

    # Discovery / OOS / Holdout splits (per brief Phase 2 §5 and project convention)
    "discovery_start":           date(2023, 1, 2),
    "discovery_end":             date(2024, 12, 31),
    "oos_start":                 date(2025, 1, 1),
    "oos_end":                   date(2025, 12, 31),
    "holdout_start":             date(2026, 1, 1),
    "holdout_end":               date(2026, 4, 30),

    # Regime / period cuts
    "regime_2024_cut":           date(2024, 1, 1),
    "sebi_oct2025_cut":          date(2025, 10, 1),

    # Acceptance gates (SHORT direction)
    "drift_max":                 -0.20,   # SHORT: signal-cohort mean must be <= -0.20%
    "n_signal_min":              200,

    # Universe file & data paths
    "universe_json":  _REPO_ROOT / "assets" / "nifty_100_universe.json",
    "monthly_5m_dir": _REPO_ROOT / "backtest-cache-download" / "monthly",
    "daily_path":     _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather",
    "out_csv":        _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_nifty_100_gap_up_low_volume_fade_signature.csv",
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


def _load_universe(path: Path) -> List[str]:
    """Load NIFTY 100 universe from assets/nifty_100_universe.json.

    Supports two schemas (both produced by HF agents on 2026-05-22):
      Schema A (HF2 simple):   {"symbols": ["RELIANCE", ...]}
      Schema B (HF1 richer):   {"universe": [{"symbol": "RELIANCE", ...}, ...]}
    Returns the bare symbol list either way.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    # Schema A
    if "symbols" in payload and isinstance(payload["symbols"], list) and payload["symbols"]:
        return [str(s) for s in payload["symbols"]]
    # Schema B
    if "universe" in payload and isinstance(payload["universe"], list):
        out: List[str] = []
        for row in payload["universe"]:
            if isinstance(row, dict) and "symbol" in row:
                out.append(str(row["symbol"]))
            elif isinstance(row, str):
                out.append(row)
        if out:
            return out
    raise RuntimeError(
        f"universe_json at {path} has neither 'symbols' list nor 'universe' list-of-dicts"
    )


# -----------------------------------------------------------------------------
# Per-symbol prior-day-close (PDC) lookup, built from consolidated_daily.feather.
# Returns dict: bare_symbol -> DataFrame indexed by date with column 'close'.
# Used to compute gap_pct = (open_at_0915 / PDC) - 1 where PDC = close on
# strictly-prior trading day (T-1) of the SAME symbol.
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


def _prior_close(pdc_df: pd.DataFrame, d: date) -> Optional[float]:
    """Strictly-prior daily close (T-1, same symbol). None if not available."""
    # Use index search: largest d_i < d
    idx = pdc_df.index
    # idx is a numpy array of python date objects
    arr = np.asarray(idx)
    mask = arr < d
    if not mask.any():
        return None
    last = arr[mask][-1]
    return float(pdc_df.loc[last, "close"])


# -----------------------------------------------------------------------------
# Per-symbol first-hour-volume series (one value per trading day).
# Computed once over the full window from the monthly 5m feathers (after they're
# concatenated), then used to build prior-5d baselines per (sym, T+0).
# -----------------------------------------------------------------------------
def _compute_first_hour_volumes(
    df: pd.DataFrame,
    first_hour_start: dtime,
    first_hour_end: dtime,
) -> pd.DataFrame:
    """For each (sym, day), compute first-hour cumulative volume.

    First-hour window: bars whose `time` is in [first_hour_start, first_hour_end].
    The brief specifies inclusive of the 10:10-10:15 bar; since bar `time` is the
    bar's OPEN, the relevant bars have times in {09:15, 09:20, ..., 10:10}.
    Note: project convention has 5m bar timestamp = bar OPEN, so the 10:10-10:15
    bar has time=10:10 and closes at 10:15. We include bar times >= 09:15 and
    <= 10:10 (i.e., strict-less-than first_hour_end=10:15).
    Returns long DataFrame with columns ['symbol', 'day', 'first_hour_vol'].
    """
    m = (df["time"] >= first_hour_start) & (df["time"] < first_hour_end)
    sub = df[m]
    agg = sub.groupby(["symbol", "day"], sort=False)["volume"].sum().reset_index()
    agg.rename(columns={"volume": "first_hour_vol"}, inplace=True)
    return agg


def _compute_post_window_volumes(
    df: pd.DataFrame,
    post_window_start: dtime,
    post_window_end: dtime,
) -> pd.DataFrame:
    """For each (sym, day), compute cumulative volume in [post_window_start, post_window_end].

    Inclusive on both ends (bar `time` is the bar OPEN; 13:25 bar's open = 13:25,
    close = 13:30 -- we INCLUDE that bar).
    """
    m = (df["time"] >= post_window_start) & (df["time"] <= post_window_end)
    sub = df[m]
    agg = sub.groupby(["symbol", "day"], sort=False)["volume"].sum().reset_index()
    agg.rename(columns={"volume": "post_vol"}, inplace=True)
    return agg


def _prior_n_mean(
    sym_day_vol_df: pd.DataFrame,
    sym: str,
    d: date,
    n_days: int,
) -> Optional[float]:
    """Return mean of prior-n-days vol for (sym, d). Strictly prior. None if <n_days bars."""
    sub = sym_day_vol_df.get(sym)
    if sub is None:
        return None
    arr_d = sub.index.to_numpy()
    mask = arr_d < d
    if mask.sum() < n_days:
        return None
    # Take the last n_days entries before d
    prior_days = arr_d[mask][-n_days:]
    vals = sub.loc[prior_days].iloc[:, 0].to_numpy(dtype=np.float64)
    if vals.size == 0:
        return None
    return float(vals.mean())


def _by_sym_indexed(
    df_long: pd.DataFrame,
    value_col: str,
) -> Dict[str, pd.DataFrame]:
    """Convert long ['symbol','day',value_col] DataFrame to dict sym -> df indexed by day."""
    out: Dict[str, pd.DataFrame] = {}
    for sym, grp in df_long.groupby("symbol", sort=False):
        g = grp.sort_values("day").reset_index(drop=True)
        out[sym] = pd.DataFrame(
            {value_col: g[value_col].to_numpy(dtype=np.float64)},
            index=g["day"].to_numpy(),
        )
    return out


# -----------------------------------------------------------------------------
# Per-(symbol, day) evaluator -- emits ONE row per (sym, T+0):
#   - gap_pct (signed)
#   - first_hour_vol_ratio (today / prior-5d mean)
#   - signal_close at 10:15 bar
#   - close_at_1325 (= 13:30 IST exit)
#   - is_signal (True iff gap_pct >= gap_pct_min AND first_hour_vol_ratio <= 0.7)
#   - post_11_to_1330_vol_ratio  (Falsifier #1)
# -----------------------------------------------------------------------------
def evaluate_symbol_day(
    sym_bars: pd.DataFrame,
    sym: str,
    d: date,
    pdc: Optional[float],
    first_hour_vol_today: Optional[float],
    first_hour_baseline: Optional[float],
    post_vol_today: Optional[float],
    post_baseline: Optional[float],
    cfg: Dict[str, object],
) -> Optional[dict]:
    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    if sym_bars.empty or pdc is None or pdc <= 0:
        return None

    # Need OPEN at 09:15 (the bar with time=09:15)
    open_row = sym_bars[sym_bars["time"] == dtime(9, 15)]
    if open_row.empty:
        return None
    open_0915 = float(open_row.iloc[0]["open"])
    if not np.isfinite(open_0915) or open_0915 <= 0:
        return None

    # Gap pct
    gap_pct = (open_0915 / pdc) - 1.0

    # Signal close (10:15 = 10:10-10:15 bar close == bar with time=10:10's close)
    sig_t = cfg["signal_bar_time"]  # dtime(10, 15)
    # The bar with time=10:10 closes at 10:15. Per project convention, signal at
    # "10:15 close" means the bar whose CLOSE is at 10:15. Since bar `time` is
    # the OPEN, this is the bar with time=10:10.
    sig_open_time = dtime(10, 10)
    sig_row = sym_bars[sym_bars["time"] == sig_open_time]
    if sig_row.empty:
        return None
    signal_close = float(sig_row.iloc[0]["close"])

    # Target: close at 13:25 bar (= 13:30 IST close per project convention)
    tgt_t = cfg["target_bar_time"]  # dtime(13, 25)
    tgt_row = sym_bars[sym_bars["time"] == tgt_t]
    if tgt_row.empty:
        return None
    close_at_1325 = float(tgt_row.iloc[0]["close"])

    if not np.isfinite(signal_close) or signal_close <= 0:
        return None
    if not np.isfinite(close_at_1325) or close_at_1325 <= 0:
        return None

    ret_to_1330 = (close_at_1325 - signal_close) / signal_close * 100.0

    # First-hour volume ratio
    if (
        first_hour_vol_today is None
        or first_hour_baseline is None
        or first_hour_baseline <= 0
    ):
        first_hour_vol_ratio = float("nan")
    else:
        first_hour_vol_ratio = float(first_hour_vol_today / first_hour_baseline)

    # Falsifier #1 post-11:00 volume ratio
    if post_vol_today is None or post_baseline is None or post_baseline <= 0:
        post_vol_ratio = float("nan")
    else:
        post_vol_ratio = float(post_vol_today / post_baseline)

    gap_min = float(cfg["gap_pct_min"])
    fh_ratio_max = float(cfg["first_hour_vol_ratio_max"])

    gap_ok = gap_pct >= gap_min
    fh_ok = (not np.isnan(first_hour_vol_ratio)) and (first_hour_vol_ratio <= fh_ratio_max)

    is_signal = bool(gap_ok and fh_ok)
    # is_baseline: same gap-up universe but vol-confirmed (ratio > 0.7)
    is_baseline = bool(
        gap_ok
        and (not np.isnan(first_hour_vol_ratio))
        and (first_hour_vol_ratio > fh_ratio_max)
    )

    return {
        "signal_ts": pd.Timestamp.combine(d, sig_t).isoformat(),
        "symbol": sym,
        "date": d.isoformat(),
        "gap_pct": gap_pct * 100.0,    # store in PERCENT for human reading
        "open_0915": open_0915,
        "prior_day_close": pdc,
        "first_hour_vol_today": (
            float(first_hour_vol_today) if first_hour_vol_today is not None else float("nan")
        ),
        "first_hour_vol_baseline_5d": (
            float(first_hour_baseline) if first_hour_baseline is not None else float("nan")
        ),
        "first_hour_vol_ratio_vs_5d": first_hour_vol_ratio,
        "signal_close": signal_close,
        "close_at_1325": close_at_1325,
        "ret_to_1330": ret_to_1330,
        "post_11_to_1330_vol_today": (
            float(post_vol_today) if post_vol_today is not None else float("nan")
        ),
        "post_11_to_1330_vol_baseline_5d": (
            float(post_baseline) if post_baseline is not None else float("nan")
        ),
        "post_11_to_1330_vol_ratio": post_vol_ratio,
        "is_signal": is_signal,
        "is_baseline": is_baseline,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature -- nifty_100_gap_up_low_volume_followthrough_fade_short")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print(f"Universe: NIFTY 100 (large_cap only, MIS=True) from {CONFIG['universe_json'].name}")
    print(f"Gap: open/PDC - 1 >= {CONFIG['gap_pct_min']*100:.2f}%")
    print(
        f"Vol gate: first-hour (09:15-10:15) cum_vol / prior-5d_same-window_mean "
        f"<= {CONFIG['first_hour_vol_ratio_max']:.2f}"
    )
    print(f"Signal:  {CONFIG['signal_bar_time']} 5m bar's close (= 10:15 IST)")
    print(f"Target:  {CONFIG['target_bar_time']} 5m bar's close (= 13:30 IST exit)")
    print(f"Direction: SHORT  --  Acceptance: signal mean ret <= {CONFIG['drift_max']:.2f}% AND n>={CONFIG['n_signal_min']}")
    print("=" * 80)

    # 1. Load NIFTY 100 universe list
    universe_syms = _load_universe(CONFIG["universe_json"])
    universe_set = set(universe_syms)
    _log(f"NIFTY 100 universe loaded: n={len(universe_syms)}")

    # 2. Load daily and build PDC lookup
    _log("Loading consolidated_daily.feather ...")
    daily_df = pd.read_feather(CONFIG["daily_path"])
    daily_df["ts"] = _ensure_naive_ist(daily_df["ts"])
    daily_df = daily_df[daily_df["ts"] <= pd.Timestamp(CONFIG["window_end"])]
    # Restrict to NIFTY 100 for speed
    daily_df = daily_df[daily_df["symbol"].isin(universe_set)]
    _log(f"  daily rows (NIFTY 100): {len(daily_df):,}  symbols: {daily_df['symbol'].nunique():,}")
    pdc_lookup = _build_pdc_lookup(daily_df)
    _log(f"  PDC lookup built for {len(pdc_lookup):,} symbols")

    # 3. ProductionUniverseGate (large_cap + MIS)
    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=int(CONFIG["min_trading_days_required"]),
        min_daily_avg_volume=float(CONFIG["min_daily_avg_volume"]),
    )

    # 4. Single-pass over monthly 5m feathers:
    #    - filter to NIFTY 100 universe
    #    - per (sym, day) compute first_hour_vol and post_vol
    #    - retain bars for those (sym, day) in memory only during the per-month scan
    months = _months_between(CONFIG["window_start"], CONFIG["window_end"])

    fh_start = CONFIG["first_hour_start"]
    fh_end = CONFIG["first_hour_end"]
    post_start = CONFIG["post_window_start"]
    post_end = CONFIG["post_window_end"]
    n_base_days = int(CONFIG["first_hour_baseline_days"])
    n_post_base_days = int(CONFIG["post_window_baseline_days"])

    # Collect first-hour & post volumes per month, then concat into one long DF
    fh_chunks: List[pd.DataFrame] = []
    post_chunks: List[pd.DataFrame] = []
    bars_by_month: List[Tuple[int, int, pd.DataFrame]] = []

    _log("First pass: loading 5m feathers + computing first-hour / post-11 volumes ...")
    for (yy, mm) in months:
        path = Path(CONFIG["monthly_5m_dir"]) / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            _log(f"  skip {yy:04d}-{mm:02d} (no 5m feather)")
            continue
        df = pd.read_feather(
            path,
            columns=["symbol", "date", "open", "high", "low", "close", "volume"],
        )
        df["date"] = _ensure_naive_ist(df["date"])
        df["day"] = df["date"].dt.date
        df["time"] = df["date"].dt.time
        df = df[(df["day"] >= CONFIG["window_start"]) & (df["day"] <= CONFIG["window_end"])]
        if df.empty:
            continue
        df = df[df["symbol"].isin(universe_set)]
        if df.empty:
            _log(f"  {yy:04d}-{mm:02d}: 0 NIFTY-100 rows")
            continue

        fh_chunks.append(_compute_first_hour_volumes(df, fh_start, fh_end))
        post_chunks.append(_compute_post_window_volumes(df, post_start, post_end))
        bars_by_month.append((yy, mm, df))
        _log(f"  {yy:04d}-{mm:02d}: rows={len(df):,}  symbols={df['symbol'].nunique()}")

    if not fh_chunks:
        print("\nNO MONTHLY 5M DATA FOUND IN WINDOW -- abort.")
        return 1

    fh_long = pd.concat(fh_chunks, ignore_index=True)
    post_long = pd.concat(post_chunks, ignore_index=True)
    _log(f"first-hour vol rows (sym,day): {len(fh_long):,}")
    _log(f"post-11 vol rows (sym,day):    {len(post_long):,}")

    # Sym-indexed lookups for baselines
    fh_by_sym = _by_sym_indexed(fh_long, "first_hour_vol")
    post_by_sym = _by_sym_indexed(post_long, "post_vol")

    # 5. Second pass: per (sym, day), evaluate signature row
    _log("Second pass: per (sym, day) signature evaluation ...")
    records: List[dict] = []
    total_pairs_seen = 0
    total_universe_pass = 0
    total_gap_qualify = 0
    total_signal = 0
    total_baseline = 0

    for (yy, mm, df) in bars_by_month:
        month_pairs = 0
        month_univ = 0
        month_gap = 0
        month_sig = 0
        month_base = 0

        for (sym, d), bars in df.groupby(["symbol", "day"], sort=False):
            month_pairs += 1
            total_pairs_seen += 1

            if not gate.is_eligible(sym, d):
                continue
            month_univ += 1
            total_universe_pass += 1

            pdc = _prior_close(pdc_lookup[sym], d) if sym in pdc_lookup else None
            if pdc is None:
                continue

            # Today's first-hour vol and prior-5d baseline
            sym_fh = fh_by_sym.get(sym)
            fh_today: Optional[float] = None
            if sym_fh is not None and d in sym_fh.index:
                fh_today = float(sym_fh.loc[d, "first_hour_vol"])
            fh_base = _prior_n_mean(fh_by_sym, sym, d, n_base_days)

            # Today's post-11:00 vol and prior-5d baseline (Falsifier #1)
            sym_post = post_by_sym.get(sym)
            post_today: Optional[float] = None
            if sym_post is not None and d in sym_post.index:
                post_today = float(sym_post.loc[d, "post_vol"])
            post_base = _prior_n_mean(post_by_sym, sym, d, n_post_base_days)

            rec = evaluate_symbol_day(
                bars, sym, d, pdc,
                fh_today, fh_base,
                post_today, post_base,
                CONFIG,
            )
            if rec is None:
                continue

            # First-fire-per-day latch is naturally enforced: this evaluator emits
            # exactly one row per (sym, T+0). We never re-emit for the same key.
            if rec["gap_pct"] >= float(CONFIG["gap_pct_min"]) * 100.0:
                month_gap += 1
                total_gap_qualify += 1

            if rec["is_signal"]:
                month_sig += 1
                total_signal += 1
            if rec["is_baseline"]:
                month_base += 1
                total_baseline += 1

            records.append(rec)

        _log(
            f"  {yy:04d}-{mm:02d}: pairs={month_pairs:,} univ={month_univ:,} "
            f"gap_qualify={month_gap:,} signal={month_sig:,} baseline={month_base:,}"
        )

    print()
    print("=== Pipeline tally ===")
    print(f"  total (sym, day) pairs seen:      {total_pairs_seen:,}")
    print(f"  passed universe gate:             {total_universe_pass:,}")
    print(f"  gap >= {CONFIG['gap_pct_min']*100:.2f}%:                  {total_gap_qualify:,}")
    print(f"  is_signal (gap+lowvol cohort):    {total_signal:,}")
    print(f"  is_baseline (gap+vol-confirmed):  {total_baseline:,}")

    if not records:
        print("\nNO RECORDS COLLECTED -- abort.")
        return 1

    df_all = pd.DataFrame.from_records(records)

    # Cohort attribution
    dt_dates = pd.to_datetime(df_all["date"])
    cut_2024 = pd.Timestamp(CONFIG["regime_2024_cut"])
    cut_sebi = pd.Timestamp(CONFIG["sebi_oct2025_cut"])
    df_all["regime_pre_post_2024"] = np.where(dt_dates < cut_2024, "pre", "post")
    df_all["period_pre_post_sebi_oct2025"] = np.where(dt_dates < cut_sebi, "pre", "post")

    # Discovery / OOS / Holdout slice tags
    d_start = pd.Timestamp(CONFIG["discovery_start"])
    d_end = pd.Timestamp(CONFIG["discovery_end"])
    o_start = pd.Timestamp(CONFIG["oos_start"])
    o_end = pd.Timestamp(CONFIG["oos_end"])
    h_start = pd.Timestamp(CONFIG["holdout_start"])
    h_end = pd.Timestamp(CONFIG["holdout_end"])

    def _slice(ts: pd.Timestamp) -> str:
        if d_start <= ts <= d_end:
            return "discovery"
        if o_start <= ts <= o_end:
            return "oos"
        if h_start <= ts <= h_end:
            return "holdout"
        return "other"

    df_all["slice"] = dt_dates.map(_slice)

    def _gap_bucket(g_pct: float) -> str:
        # g_pct stored in PERCENT
        if pd.isna(g_pct):
            return "nan"
        if g_pct < 0.8:
            return "<0.8"
        if g_pct < 1.5:
            return "0.8-1.5"
        if g_pct < 2.5:
            return "1.5-2.5"
        return ">=2.5"

    df_all["gap_pct_bucket"] = df_all["gap_pct"].map(_gap_bucket)

    def _vr_bucket(v: float) -> str:
        if pd.isna(v):
            return "nan"
        if v < 0.4:
            return "<0.4"
        if v < 0.55:
            return "0.4-0.55"
        if v < 0.7:
            return "0.55-0.7"
        if v <= 1.0:
            return "0.7-1.0"
        return ">1.0"

    df_all["vol_ratio_bucket"] = df_all["first_hour_vol_ratio_vs_5d"].map(_vr_bucket)

    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"\nSaved cohort rows: {len(df_all):,} -> {out_path}")

    # -----------------------------------------------------------------
    # STEP 1 -- Aggregate drift (signal vs baseline), SHORT direction
    # -----------------------------------------------------------------
    sig = df_all[df_all["is_signal"]].copy()
    base = df_all[df_all["is_baseline"]].copy()
    n_sig = len(sig)
    n_base = len(base)
    sig_mean = float(sig["ret_to_1330"].mean()) if n_sig else float("nan")
    base_mean = float(base["ret_to_1330"].mean()) if n_base else float("nan")
    sig_med = float(sig["ret_to_1330"].median()) if n_sig else float("nan")
    base_med = float(base["ret_to_1330"].median()) if n_base else float("nan")
    delta = (sig_mean - base_mean) if (n_sig and n_base) else float("nan")

    print()
    print("=" * 80)
    print("STEP 1 -- Aggregate drift (SHORT direction; signal_close -> 13:30)")
    print("=" * 80)
    print(f"  n_signal:                    {n_sig:,}")
    print(f"  n_baseline (vol-confirmed):  {n_base:,}")
    print(f"  Signal   mean ret_to_1330:   {sig_mean:+.4f}%  (median {sig_med:+.4f}%)")
    print(f"  Baseline mean ret_to_1330:   {base_mean:+.4f}%  (median {base_med:+.4f}%)")
    print(f"  Delta (signal - baseline):   {delta:+.4f}%")
    print(f"  Acceptance: signal_mean <= {CONFIG['drift_max']:.2f}% AND n_signal >= {CONFIG['n_signal_min']}")

    # -----------------------------------------------------------------
    # STEP 2 -- Falsifier #1: post-11:00 to 13:30 volume signature
    # -----------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 2 -- Falsifier #1 (post-11:00 to 13:30 volume ratio vs 5d baseline)")
    print("=" * 80)
    sig_v = sig.dropna(subset=["post_11_to_1330_vol_ratio"]).copy()
    base_v = base.dropna(subset=["post_11_to_1330_vol_ratio"]).copy()
    if len(sig_v):
        sig_post_med = float(sig_v["post_11_to_1330_vol_ratio"].median())
        sig_post_mean = float(sig_v["post_11_to_1330_vol_ratio"].mean())
        frac_above_1 = float((sig_v["post_11_to_1330_vol_ratio"] >= 1.0).mean())
        print(f"  signal cohort n_valid:           {len(sig_v):,}")
        print(f"  signal post_vol_ratio median:    {sig_post_med:.4f}")
        print(f"  signal post_vol_ratio mean:      {sig_post_mean:.4f}")
        print(f"  signal frac (ratio >= 1.0):      {frac_above_1*100:.2f}%   [expect >= 50% on FADE days]")
    else:
        print("  signal cohort: no valid rows for post_vol_ratio.")
    if len(base_v):
        base_post_med = float(base_v["post_11_to_1330_vol_ratio"].median())
        base_post_mean = float(base_v["post_11_to_1330_vol_ratio"].mean())
        print(f"  baseline cohort n_valid:         {len(base_v):,}")
        print(f"  baseline post_vol_ratio median:  {base_post_med:.4f}")
        print(f"  baseline post_vol_ratio mean:    {base_post_mean:.4f}")

    # -----------------------------------------------------------------
    # STEP 3 -- Cohort splits
    # -----------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 3 -- Cohort splits")
    print("=" * 80)
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")

    def _split_block(label: str, mask_sig: pd.Series, mask_base: pd.Series) -> None:
        s = sig[mask_sig]
        b = base[mask_base]
        ns, nb = len(s), len(b)
        sm = float(s["ret_to_1330"].mean()) if ns else float("nan")
        bm = float(b["ret_to_1330"].mean()) if nb else float("nan")
        dl = (sm - bm) if (ns and nb) else float("nan")
        print(
            f"{label:<40}"
            f"{ns:>8d}"
            f"{nb:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )

    sig_dates = pd.to_datetime(sig["date"]) if n_sig else pd.Series([], dtype="datetime64[ns]")
    base_dates = pd.to_datetime(base["date"]) if n_base else pd.Series([], dtype="datetime64[ns]")

    # discovery / oos / holdout
    print()
    print("  --- discovery / OOS / holdout ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    for slice_label, (lo, hi) in [
        ("discovery (2023-2024)", (d_start, d_end)),
        ("oos (2025)",            (o_start, o_end)),
        ("holdout (2026 H1)",     (h_start, h_end)),
    ]:
        ms = (sig_dates >= lo) & (sig_dates <= hi) if n_sig else pd.Series([], dtype=bool)
        mb = (base_dates >= lo) & (base_dates <= hi) if n_base else pd.Series([], dtype=bool)
        _split_block(slice_label, ms, mb)

    # pre/post 2024
    print()
    print("  --- pre/post 2024 ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    if n_sig:
        _split_block("pre_2024",  sig_dates <  cut_2024, base_dates <  cut_2024)
        _split_block("post_2024", sig_dates >= cut_2024, base_dates >= cut_2024)

    # pre/post SEBI Oct 2025
    print()
    print("  --- pre/post SEBI Oct 2025 ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    if n_sig:
        _split_block("pre_sebi_oct2025",  sig_dates <  cut_sebi, base_dates <  cut_sebi)
        _split_block("post_sebi_oct2025", sig_dates >= cut_sebi, base_dates >= cut_sebi)

    # gap_pct_bucket
    print()
    print("  --- gap_pct_bucket ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    for bucket in ["0.8-1.5", "1.5-2.5", ">=2.5"]:
        _split_block(
            f"gap_pct={bucket}",
            sig["gap_pct_bucket"] == bucket,
            base["gap_pct_bucket"] == bucket,
        )

    # vol_ratio_bucket (signal cohort only; baseline ratio is by definition >0.7)
    print()
    print("  --- first_hour_vol_ratio_bucket (signal cohort only; baseline = aggregate baseline) ---")
    print(f"{'split':<40}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    for bucket in ["<0.4", "0.4-0.55", "0.55-0.7"]:
        s = sig[sig["vol_ratio_bucket"] == bucket]
        ns = len(s)
        sm = float(s["ret_to_1330"].mean()) if ns else float("nan")
        bm = base_mean
        dl = (sm - bm) if (ns and not np.isnan(bm)) else float("nan")
        print(
            f"{'vol_ratio=' + bucket:<40}"
            f"{ns:>8d}"
            f"{n_base:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )

    # -----------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------
    print()
    print("=" * 80)
    print("VERDICT (Phase 2 signature, pre-registered acceptance)")
    print("=" * 80)
    drift_ok = (not np.isnan(sig_mean)) and (sig_mean <= float(CONFIG["drift_max"]))
    n_ok = n_sig >= int(CONFIG["n_signal_min"])
    if not n_ok:
        verdict = "KILL"
        reason = f"n_signal {n_sig} < {CONFIG['n_signal_min']}"
    elif not drift_ok:
        verdict = "KILL"
        reason = (
            f"signal mean ret_to_1330 {sig_mean:+.4f}% > {CONFIG['drift_max']:+.2f}% "
            f"(SHORT drift insufficient)"
        )
    else:
        verdict = "PROCEED to Phase 3"
        reason = (
            f"signal mean ret_to_1330 {sig_mean:+.4f}% <= {CONFIG['drift_max']:+.2f}% "
            f"AND n_signal {n_sig} >= {CONFIG['n_signal_min']}"
        )

    print(f"  {verdict}")
    print(f"  reason: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
