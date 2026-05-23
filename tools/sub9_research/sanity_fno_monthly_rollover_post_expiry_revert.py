"""Phase 4 sanity: fno_monthly_rollover_intraday_post_expiry_revert.

Brief: specs/2026-05-22-brief-fno_monthly_rollover_intraday_post_expiry_revert.md
Predecessor Phase 2: tools/sub9_research/phase2_fno_monthly_rollover_post_expiry_revert_signature.py

# Mechanism (restated)

On T+1 (the trading session IMMEDIATELY AFTER NSE monthly F&O expiry), F&O-liquid-200
single-stock-futures underlyings exhibit a measurable 09:15-10:30 morning-gap reversal
because of (a) overnight institutional position resets, (b) cash-settlement clearing
flow, and (c) forced retail re-entry into the new front-month. Direction is gap-sign
driven:
  - LONG entry: gap_pct <= -0.005 (gap-DOWN >= 0.5%, expect mean-revert UP)
  - SHORT entry: gap_pct >= +0.005 (gap-UP >= 0.5%, expect mean-revert DOWN)

Signal bar = 09:15 5m bar (closes at 09:20 IST). gap_pct computed against T0 expiry
close (PDC for signal date). Mode B entry at bars[i+1].open = 09:20 5m bar's open.
Time stop = 10:30 IST (5m bar labeled 10:25 closes at 10:30).

# Anti-bias guards (Lesson #5 6-failure-mode checklist)

  1. No day-aggregate look-ahead. PDC is strictly the close of the PRIOR trading day
     (T0 expiry day) for the T+1 signal — looked up via consolidated_daily.feather
     with strict "<" date comparison.
  2. No volume baseline used (this is a gap-magnitude / direction signal, not a
     volume-spike signal). No look-ahead from intraday aggregates.
  3. Mode B entry at bars[i+1].open. Path walk starts AT the entry bar. Per Lesson #5
     #3: walk starts AT bars[i+1], NOT bars[i+2] — the entry bar's intra-bar OHLC
     range happens AFTER entry-at-open and is fully eligible for SL/T1/T2 hits.
  4. Same-bar SL+T1/T2 picks SL (pessimistic for both LONG and SHORT). Per Lesson #5
     #4. For LONG: if low<=hard_sl AND (high>=t1 OR high>=t2) in the same bar, exit
     at hard_sl. Mirror for SHORT.
  5. Filters at signal time only. ALL parameters pre-registered in brief §5 and frozen
     in the constants block below. NO post-hoc tuning at Phase 4. Phase 5 R-sweep
     will optimise; Phase 4 records full ledger only.
  6. Output canonical schema validated downstream via tools.methodology.sanity_csv_schema
     when consumed by cell_sweep / walk_forward. First-fire-per-day-per-stock latch
     ensures exactly one trade row per (sym, signal_date).

# Regime gate (Falsifier #3 — calendar cutover)

NSE Circular FAOP68747 shifted monthly expiry "last Thursday" -> "last Tuesday" effective
2025-09-01. T+1 day therefore shifted Friday -> Wednesday. The output CSV stamps
`regime_pre_post_2025_09` and `expiry_day_of_week` for downstream split analysis.
Phase 2 confirmed regime-stable mean_revert_R across the cutover (Falsifier #3 PASS).

# Universe (Lesson #19)

F&O liquid 200 (assets/fno_liquid_200.csv, ~153 NSE symbols). cap_segment in
{large_cap, mid_cap} (F&O underlyings span both bands). MIS-eligible required. Per-date
filter via ProductionUniverseGate, NOT window-level coverage (which has survivorship
bias per Lesson #16). min_trading_days_required=0, min_daily_avg_volume=0 (Lesson #17:
zero legacy intraday-MIS filters for cell-locked setups).

# SL convention (LONG/SHORT, mean-revert mechanic)

LONG (after gap-DOWN, mean-revert UP):
  hard_sl = MIN(signal_bar.low * (1 - 0.002), entry_price * (1 - 0.01))   # widest below entry
  R = entry_price - hard_sl                                                # R > 0
  t1 = entry_price + 1.0 * R                                               # mean-revert target up
  t2 = entry_price + 2.0 * R

SHORT (after gap-UP, mean-revert DOWN):
  hard_sl = MAX(signal_bar.high * (1 + 0.002), entry_price * (1 + 0.01))   # widest above entry
  R = hard_sl - entry_price                                                # R > 0
  t1 = entry_price - 1.0 * R                                               # mean-revert target down
  t2 = entry_price - 2.0 * R

Brief §5 had LONG/SHORT SL formulas placed inconsistently with the mean-revert thesis
(LONG SL above entry, SHORT SL below entry); above we use the standard convention that
matches the project's surviving LONG reference (sanity_5day_RSI_VWAP_absorb_continuation
_long.py) and produces a positive R for mean-revert exits. SL is on the OPPOSITE side of
entry from T1/T2 in both directions — the only convention that makes "R = SL-distance"
finite and positive and lets Phase 5 R-sweep optimise sanely.

Risk per trade: Rs 1,000 (matches below_vwap / 5day_RSI convention).
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"
_DAILY_PATH = _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather"

sys.path.insert(0, str(_REPO_ROOT))
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402
from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402


# ---------------------------------------------------------------------------
# Pre-registered constants (brief §5; LOCKED, no post-hoc tuning at Phase 4)
# ---------------------------------------------------------------------------

# Window definitions (standard discovery / OOS / holdout slices)
WINDOWS: Dict[str, Tuple[date, date]] = {
    "discovery": (date(2023, 1, 1), date(2024, 12, 31)),
    "oos":       (date(2025, 1, 1), date(2025, 12, 31)),
    "holdout":   (date(2026, 1, 1), date(2026, 4, 30)),
}

# Universe (Lesson #19, F&O liquid 200)
ALLOWED_CAP_SEGMENTS: Tuple[str, ...] = ("large_cap", "mid_cap")  # F&O underlyings span both
REQUIRE_MIS = True
MIN_TRADING_DAYS_REQUIRED = 0   # Lesson #17 — zero legacy filters
MIN_DAILY_AVG_VOLUME = 0        # Lesson #17 — zero legacy filters

# Gap signal threshold (|gap_pct| >= 0.5% required)
GAP_PCT_ABS_MIN = 0.005

# Signal / entry / time-stop bars (5m bar labels are START time per project convention)
# 09:15 bar = opening bar of the session; closes at wall-clock 09:20.
# 09:20 bar = next bar; entry at its OPEN (Mode B).
# 10:25 bar = bar that closes at wall-clock 10:30 (time stop).
SIGNAL_BAR_HHMM = "09:15"
ENTRY_BAR_HHMM = "09:20"
TIME_STOP_HHMM_INT = 1025   # bar labeled 10:25 closes at 10:30 wall-clock

# Exit parameters (brief §5)
SL_PCT_FROM_EXTREME = 0.002     # 0.2% beyond signal-bar low (LONG) / high (SHORT)
SL_PCT_FROM_ENTRY = 0.01        # 1% beyond entry (floor on SL distance)
T1_R_MULT = 1.0
T2_R_MULT = 2.0

# Position sizing
RISK_PER_TRADE_RUPEES = 1000

# Regime cutover (Falsifier #3, MANDATORY)
REGIME_FAO_CIRCULAR_CUT = date(2025, 9, 1)

# Data paths
_FNO_UNIVERSE_CSV = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
_NSE_HOLIDAYS_JSON = _REPO_ROOT / "assets" / "nse_holidays.json"
_FUTURES_BASIS = _REPO_ROOT / "data" / "futures_basis" / "2023_2026_basis.parquet"

_DOW_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            m, y = 1, y + 1
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


def _hhmm_add_5(hhmm: str) -> str:
    """Add 5 minutes to an HH:MM string."""
    h, m = hhmm.split(":")
    total = int(h) * 60 + int(m) + 5
    return f"{total // 60:02d}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Universe / calendar loaders
# ---------------------------------------------------------------------------

def _load_fno_universe(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise RuntimeError(f"{path} missing 'symbol' column")
    return [_normalize_symbol(str(s)) for s in df["symbol"].tolist() if str(s).strip()]


def _load_holiday_set(path: Path) -> Set[date]:
    """NSE holiday set from assets/nse_holidays.json (tradingDate field)."""
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
    """Next trading session strictly AFTER d (skips weekends + NSE holidays)."""
    cur = d + timedelta(days=1)
    while cur.weekday() >= 5 or cur in holidays:
        cur += timedelta(days=1)
    return cur


def _build_tplus1_map(parquet_path: Path, holidays: Set[date]) -> Dict[date, date]:
    """Return dict: T+1 date -> T0 (expiry) date for each monthly expiry.

    Source of truth: data/futures_basis/2023_2026_basis.parquet `expiry_date` column
    (41 monthly expiries, holiday-adjusted to actual NSE-settled dates). T+1 = next
    trading session strictly after the expiry date.
    """
    df = pd.read_parquet(parquet_path, columns=["expiry_date"])
    expiries = sorted({d for d in df["expiry_date"].tolist() if d is not None})
    out: Dict[date, date] = {}
    for t0 in expiries:
        tp1 = _next_trading_session(t0, holidays)
        out[tp1] = t0
    return out


# ---------------------------------------------------------------------------
# Daily PDC lookup (consolidated_daily.feather restricted to F&O universe)
# ---------------------------------------------------------------------------

def _build_pdc_lookup(daily_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per-symbol date->close DataFrame, indexed by `d` (datetime.date).

    Each value is a DataFrame with a single 'close' column, index = sorted dates.
    Used downstream to find "close strictly before signal date" via mask < d.
    """
    out: Dict[str, pd.DataFrame] = {}
    for sym, grp in daily_df.groupby("symbol", sort=False):
        g = grp.sort_values("ts").reset_index(drop=True)
        out[str(sym)] = pd.DataFrame({
            "d": g["ts"].dt.date.values,
            "close": g["close"].to_numpy(dtype=np.float64),
        }).set_index("d")
    return out


def _pdc_strictly_before(pdc_df: pd.DataFrame, d: date) -> Tuple[Optional[float], Optional[date]]:
    """Daily close on the MOST RECENT date strictly < d. Anti-bias guard #1."""
    arr = np.asarray(pdc_df.index)
    mask = arr < d
    if not mask.any():
        return (None, None)
    last = arr[mask][-1]
    return (float(pdc_df.loc[last, "close"]), last)


# ---------------------------------------------------------------------------
# 5m bars loading
# ---------------------------------------------------------------------------

def _load_window_5m(d0: date, d1: date) -> pd.DataFrame:
    """Concat monthly 5m enriched feathers spanning [d0, d1] inclusive."""
    chunks: List[pd.DataFrame] = []
    for (yy, mm) in _months_between(d0, d1):
        p = _MONTHLY_DIR / f"{yy:04d}_{mm:02d}_5m_enriched.feather"
        if not p.exists():
            continue
        df = pd.read_feather(
            p, columns=["date", "symbol", "open", "high", "low", "close", "volume"],
        )
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype("float32")
        chunks.append(df)
    if not chunks:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close",
                                     "volume", "d", "hhmm"])
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    df["symbol"] = df["symbol"].astype(str)
    df = df.drop(columns=["date"])
    mask = (df["d"] >= d0) & (df["d"] <= d1)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Session bar index for fast per-(sym, date) walks
# ---------------------------------------------------------------------------

def _build_session_bars_index(
    df: pd.DataFrame,
) -> Dict[Tuple[str, date], List[Tuple[str, float, float, float, float, float]]]:
    """Build (symbol, date) -> [(hhmm, open, high, low, close, volume), ...]."""
    df_sorted = df.sort_values(["symbol", "d", "hhmm"])
    idx: Dict[Tuple[str, date], List[Tuple[str, float, float, float, float, float]]] = {}
    for r in df_sorted[
        ["symbol", "d", "hhmm", "open", "high", "low", "close", "volume"]
    ].itertuples(index=False):
        sym = str(r.symbol)
        key = (sym, r.d)
        if key not in idx:
            idx[key] = []
        idx[key].append((
            r.hhmm, float(r.open), float(r.high), float(r.low),
            float(r.close), float(r.volume),
        ))
    return idx


# ---------------------------------------------------------------------------
# Exit walk (direction-aware mean-revert)
# ---------------------------------------------------------------------------

def _walk_to_exit(
    session_bars: List[Tuple[str, float, float, float, float, float]],
    side: str,                     # "LONG" or "SHORT"
    entry_hhmm: str,
    entry_price: float,
    hard_sl: float,
    t1: float,
    t2: float,
    time_stop_hhmm_int: int,
) -> Tuple[float, str, str, float, float]:
    """Walk forward through pre-indexed session bars to find the exit.

    Anti-bias guard #3 (Lesson #5 #3): walk starts AT the entry bar (bars[i+1] in
    Mode B). entry was at its OPEN, so the bar's full intra-bar range happens AFTER
    entry and is eligible for SL/T1/T2.

    Anti-bias guard #4 (Lesson #5 #4): same-bar SL+T1 or SL+T2 picks STOP
    (pessimistic for both LONG and SHORT).

    Returns: (exit_price, exit_reason, exit_hhmm, mfe_r, mae_r)
      mfe_r / mae_r = max-favorable / max-adverse excursion in R-multiples (always >= 0).

    For LONG (mean-revert UP after gap-DOWN):
      - SL is BELOW entry, T1/T2 ABOVE entry.
      - R = entry - hard_sl (> 0).
      - MFE = max(high - entry, 0) / R, MAE = max(entry - low, 0) / R.
      - SL hit: low <= hard_sl. T1 hit: high >= t1. T2 hit: high >= t2.

    For SHORT (mean-revert DOWN after gap-UP):
      - SL is ABOVE entry, T1/T2 BELOW entry.
      - R = hard_sl - entry (> 0).
      - MFE = max(entry - low, 0) / R, MAE = max(high - entry, 0) / R.
      - SL hit: high >= hard_sl. T1 hit: low <= t1. T2 hit: low <= t2.
    """
    if side == "LONG":
        R = entry_price - hard_sl
    else:  # SHORT
        R = hard_sl - entry_price
    if R <= 0:
        return (entry_price, "invalid_R", entry_hhmm, 0.0, 0.0)

    mfe = 0.0
    mae = 0.0
    exit_price = entry_price
    exit_reason = "time_stop"
    exit_hhmm = entry_hhmm
    walked = False

    for (hhmm, op, hi, lo, cl, _vol) in session_bars:
        if hhmm < entry_hhmm:
            continue
        walked = True

        # MFE / MAE accounting (direction-aware)
        if side == "LONG":
            if hi > entry_price:
                mfe = max(mfe, hi - entry_price)
            if lo < entry_price:
                mae = max(mae, entry_price - lo)
        else:  # SHORT
            if lo < entry_price:
                mfe = max(mfe, entry_price - lo)
            if hi > entry_price:
                mae = max(mae, hi - entry_price)

        cur_hhmm_int = int(hhmm.replace(":", ""))
        # Time stop FIRST (10:30 wall-clock = bar 10:25; close at that bar's close).
        # Per project convention: time stop fires at the bar whose label >= TIME_STOP
        # — close the trade at the bar's close before evaluating SL/T1/T2 on the
        # SAME bar (matches reference sanity_5day_RSI_VWAP_absorb_continuation_long).
        if cur_hhmm_int >= time_stop_hhmm_int:
            exit_price = cl
            exit_reason = "time_stop"
            exit_hhmm = hhmm
            break

        if side == "LONG":
            sl_hit = lo <= hard_sl
            t2_hit = hi >= t2
            t1_hit = hi >= t1
        else:  # SHORT
            sl_hit = hi >= hard_sl
            t2_hit = lo <= t2
            t1_hit = lo <= t1

        # Pessimism on same-bar ambiguity (Lesson #5 #4)
        if sl_hit and (t1_hit or t2_hit):
            exit_price = hard_sl
            exit_reason = "same_bar_sl"
            exit_hhmm = hhmm
            break
        if sl_hit:
            exit_price = hard_sl
            exit_reason = "sl"
            exit_hhmm = hhmm
            break
        if t2_hit:
            exit_price = t2
            exit_reason = "t2"
            exit_hhmm = hhmm
            break
        if t1_hit:
            exit_price = t1
            exit_reason = "t1"
            exit_hhmm = hhmm
            break

    if not walked:
        return (entry_price, "no_data", entry_hhmm, 0.0, 0.0)
    mfe_r = mfe / R if R > 0 else 0.0
    mae_r = mae / R if R > 0 else 0.0
    return (exit_price, exit_reason, exit_hhmm, mfe_r, mae_r)


def _close_at(
    session_bars: List[Tuple[str, float, float, float, float, float]],
    target_hhmm: str,
) -> float:
    for (hhmm, _op, _hi, _lo, cl, _vol) in session_bars:
        if hhmm == target_hhmm:
            return cl
    return float("nan")


# ---------------------------------------------------------------------------
# Signal identification (per T+1 date, per F&O symbol)
# ---------------------------------------------------------------------------

def _identify_signals(
    df_window: pd.DataFrame,
    tplus1_set: Set[date],
    tplus1_map: Dict[date, date],
    pdc_lookup: Dict[str, pd.DataFrame],
    fno_set: Set[str],
) -> pd.DataFrame:
    """Return one row per (sym, T+1 date) when |gap_pct| >= GAP_PCT_ABS_MIN.

    Signal-time fields only — no exit walk yet. First-fire-per-day-per-stock latch
    is automatic since gap_pct is a 09:15-bar quantity.

    Columns: symbol, d, T0_expiry_date, signal_hhmm, signal_open, signal_high,
             signal_low, signal_close, prior_day_close, pdc_date, gap_pct,
             signal_direction, signal_sign.
    """
    if df_window.empty:
        return pd.DataFrame()

    # Restrict to T+1 dates and F&O universe to minimise work
    df = df_window[df_window["d"].isin(tplus1_set) & df_window["symbol"].isin(fno_set)]
    if df.empty:
        return pd.DataFrame()

    # Pull the 09:15 bar for each (sym, T+1 date)
    sig = df[df["hhmm"] == SIGNAL_BAR_HHMM].copy()
    if sig.empty:
        return pd.DataFrame()

    sig = sig.rename(columns={
        "open": "signal_open",
        "high": "signal_high",
        "low": "signal_low",
        "close": "signal_close",
    })

    # PDC lookup (strictly < signal_date) per row
    pdcs: List[float] = []
    pdc_dates: List[Optional[date]] = []
    for sym, d in zip(sig["symbol"].to_numpy(), sig["d"].to_numpy()):
        sym_str = str(sym)
        if sym_str not in pdc_lookup:
            pdcs.append(float("nan"))
            pdc_dates.append(None)
            continue
        pdc_val, pdc_d = _pdc_strictly_before(pdc_lookup[sym_str], d)
        if pdc_val is None or pdc_val <= 0:
            pdcs.append(float("nan"))
            pdc_dates.append(None)
        else:
            pdcs.append(pdc_val)
            pdc_dates.append(pdc_d)
    sig["prior_day_close"] = pdcs
    sig["pdc_date"] = pdc_dates

    sig = sig[sig["prior_day_close"].notna() & (sig["signal_open"] > 0)]
    if sig.empty:
        return pd.DataFrame()

    sig["gap_pct"] = (sig["signal_open"].astype(float) / sig["prior_day_close"].astype(float)) - 1.0
    sig = sig[sig["gap_pct"].abs() >= GAP_PCT_ABS_MIN].copy()
    if sig.empty:
        return pd.DataFrame()

    sig["signal_direction"] = np.where(sig["gap_pct"] > 0, "SHORT", "LONG")
    sig["signal_sign"] = np.where(sig["gap_pct"] > 0, -1, +1)
    sig["T0_expiry_date"] = sig["d"].map(lambda dd: tplus1_map.get(dd))
    sig["signal_hhmm"] = SIGNAL_BAR_HHMM
    sig = sig[[
        "symbol", "d", "T0_expiry_date", "signal_hhmm",
        "signal_open", "signal_high", "signal_low", "signal_close",
        "prior_day_close", "pdc_date", "gap_pct", "signal_direction", "signal_sign",
    ]].reset_index(drop=True)
    return sig


# ---------------------------------------------------------------------------
# Per-window driver
# ---------------------------------------------------------------------------

def run_window(
    window_label: str,
    fno_set: Set[str],
    tplus1_set: Set[date],
    tplus1_map: Dict[date, date],
    pdc_lookup: Dict[str, pd.DataFrame],
    gate: ProductionUniverseGate,
) -> pd.DataFrame:
    d0, d1 = WINDOWS[window_label]
    print(f"\n=== Window: {window_label} ({d0} to {d1}) ===", flush=True)

    # T+1 dates that fall inside the window
    window_tp1_dates = sorted(d for d in tplus1_set if d0 <= d <= d1)
    print(f"  T+1 dates in window: {len(window_tp1_dates)}", flush=True)
    if not window_tp1_dates:
        return pd.DataFrame()

    print(f"  Loading 5m bars for window...", flush=True)
    df = _load_window_5m(d0, d1)
    print(f"  Loaded {len(df):,} bars", flush=True)
    if df.empty:
        return pd.DataFrame()

    # Identify signal candidates (T+1 only, gap qualifying)
    signals = _identify_signals(df, tplus1_set, tplus1_map, pdc_lookup, fno_set)
    print(f"  {len(signals):,} signal candidates (|gap_pct| >= "
          f"{GAP_PCT_ABS_MIN*100:.2f}%, T+1 only, F&O-200 only)", flush=True)
    if signals.empty:
        return pd.DataFrame()

    # ProductionUniverseGate per (sym, signal_date) — Lesson #19
    print("  Applying ProductionUniverseGate per (signal_symbol, signal_date)...",
          flush=True)
    eligible_mask = signals.apply(
        lambda r: gate.is_eligible(str(r["symbol"]), r["d"]), axis=1
    )
    n_before = len(signals)
    signals = signals[eligible_mask].reset_index(drop=True)
    print(f"  Universe gate kept {len(signals):,} / {n_before:,} "
          f"({n_before - len(signals):,} rejected by cap/MIS).", flush=True)
    if signals.empty:
        return pd.DataFrame()

    # Filter bars to the (signal_symbol, signal_date) pairs we actually need
    keep_df = pd.DataFrame(
        list(set(zip(signals["symbol"].astype(str), signals["d"]))),
        columns=["symbol", "d"],
    )
    df_for_idx = df.merge(keep_df, on=["symbol", "d"], how="inner")
    del df
    print(f"  Bars after merge: {len(df_for_idx):,}", flush=True)
    session_idx = _build_session_bars_index(df_for_idx)
    del df_for_idx

    # Walk each signal to its exit
    trades: List[dict] = []
    cap_cache: Dict[str, Optional[str]] = {}
    n_total = len(signals)
    for i, sig in enumerate(signals.itertuples()):
        if i % 200 == 0 and i > 0:
            print(f"    processed {i:,} / {n_total:,}", flush=True)
        sym = str(sig.symbol)
        d_val = sig.d
        signal_hhmm = sig.signal_hhmm
        side = sig.signal_direction  # "LONG" or "SHORT"
        signal_high = float(sig.signal_high)
        signal_low = float(sig.signal_low)
        signal_close = float(sig.signal_close)
        gap_pct = float(sig.gap_pct)

        session_bars = session_idx.get((sym, d_val))
        if not session_bars:
            continue

        # Mode B entry: bars[i+1].open == 09:20 bar's open
        entry_hhmm = _hhmm_add_5(signal_hhmm)  # "09:20"
        entry_bar = next(
            ((h, o, hi, lo, cl, v) for (h, o, hi, lo, cl, v) in session_bars
             if h == entry_hhmm),
            None,
        )
        if entry_bar is None:
            continue  # 09:20 bar missing (session-truncated, skip)
        entry_price = float(entry_bar[1])
        if entry_price <= 0:
            continue

        # SL convention (see header docstring): widest of (extreme-based, entry-based)
        if side == "LONG":
            sl_from_extreme = signal_low * (1.0 - SL_PCT_FROM_EXTREME)
            sl_from_entry = entry_price * (1.0 - SL_PCT_FROM_ENTRY)
            hard_sl = min(sl_from_extreme, sl_from_entry)  # widest = furthest BELOW entry
            if hard_sl >= entry_price:
                continue  # degenerate (entry below signal low — gap reversal already done)
            R = entry_price - hard_sl
            t1 = entry_price + T1_R_MULT * R
            t2 = entry_price + T2_R_MULT * R
            broker_side = "BUY"
        else:  # SHORT
            sl_from_extreme = signal_high * (1.0 + SL_PCT_FROM_EXTREME)
            sl_from_entry = entry_price * (1.0 + SL_PCT_FROM_ENTRY)
            hard_sl = max(sl_from_extreme, sl_from_entry)  # widest = furthest ABOVE entry
            if hard_sl <= entry_price:
                continue  # degenerate
            R = hard_sl - entry_price
            t1 = entry_price - T1_R_MULT * R
            t2 = entry_price - T2_R_MULT * R
            broker_side = "SELL"

        exit_price, exit_reason, exit_hhmm, mfe_r, mae_r = _walk_to_exit(
            session_bars, side, entry_hhmm,
            entry_price, hard_sl, t1, t2,
            TIME_STOP_HHMM_INT,
        )
        if exit_reason in ("no_data", "invalid_R"):
            continue

        # Cap segment lookup (already passed universe gate)
        if sym not in cap_cache:
            cap_cache[sym] = gate._cap_segment(sym)  # noqa: SLF001 (internal cache)
        cap_seg = cap_cache[sym] or "unknown"

        qty = max(1, int(RISK_PER_TRADE_RUPEES / R))
        if side == "LONG":
            gross = (exit_price - entry_price) * qty
        else:  # SHORT
            gross = (entry_price - exit_price) * qty
        fee = calc_fee(entry_price, exit_price, qty, broker_side)
        net_pnl = gross - fee

        # pnl_pct: per-share % return, side-aware, NO fees, NO leverage (raw signal edge)
        if side == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100.0
        # r_multiple is realized R (positive for win)
        r_multiple = ((pnl_pct / 100.0) * entry_price) / R if R > 0 else 0.0

        # Regime stamping (Falsifier #3)
        regime = "pre_2025_09" if d_val < REGIME_FAO_CIRCULAR_CUT else "post_2025_09"
        t0_exp = sig.T0_expiry_date
        exp_dow = _DOW_NAMES[t0_exp.weekday()] if t0_exp is not None else ""
        pdc_date = sig.pdc_date

        # close_at_1025 = close of the 10:25-10:30 bar = wall-clock 10:30
        # (matches brief Phase 2 target measurement)
        close_at_1025 = _close_at(session_bars, "10:25")

        trades.append({
            "signal_date": d_val,
            "T0_expiry_date": t0_exp.isoformat() if t0_exp is not None else "",
            "symbol": f"NSE:{sym}",
            "side": side,
            "gap_pct": float(gap_pct) * 100.0,   # stored in PERCENT for downstream readability
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "qty": int(qty),
            "pnl_pct": float(pnl_pct),
            "exit_reason": exit_reason,
            "same_bar": exit_reason == "same_bar_sl",
            "cap_segment": cap_seg,
            "signal_ts": f"{d_val}T{signal_hhmm}:00",
            "entry_ts": f"{d_val}T{entry_hhmm}:00",
            "exit_ts": f"{d_val}T{exit_hhmm}:00",
            "realized_pnl_inr": float(gross),
            "fee_inr": float(fee),
            "net_pnl_inr": float(net_pnl),
            "r_multiple": float(r_multiple),
            "t1_target": float(t1),
            "t2_target": float(t2),
            "hard_sl": float(hard_sl),
            "t1_partial_booked": False,
            "mfe_r": float(mfe_r),
            "mae_r": float(mae_r),
            "R_per_share": float(R),
            "regime_pre_post_2025_09": regime,
            "expiry_day_of_week": exp_dow,
            "close_at_1025": float(close_at_1025),
            "signal_open": float(sig.signal_open),
            "signal_high": float(signal_high),
            "signal_low": float(signal_low),
            "signal_close": float(signal_close),
            "prior_day_close": float(sig.prior_day_close),
            "pdc_date": pdc_date.isoformat() if pdc_date is not None else "",
        })

    trades_df = pd.DataFrame(trades)
    print(f"  Generated {len(trades_df):,} trades", flush=True)
    if not trades_df.empty:
        n_long = (trades_df["side"] == "LONG").sum()
        n_short = (trades_df["side"] == "SHORT").sum()
        n_win = (trades_df["net_pnl_inr"] > 0).sum()
        n_loss = (trades_df["net_pnl_inr"] < 0).sum()
        gross_pnl = trades_df["realized_pnl_inr"].sum()
        total_net = trades_df["net_pnl_inr"].sum()
        wins_inr = trades_df.loc[trades_df["net_pnl_inr"] > 0, "net_pnl_inr"].sum()
        losses_inr = trades_df.loc[trades_df["net_pnl_inr"] < 0, "net_pnl_inr"].abs().sum()
        pf = (wins_inr / losses_inr) if losses_inr > 0 else float("inf")
        exit_mix = trades_df["exit_reason"].value_counts(normalize=True).round(3).to_dict()
        print(f"  Side mix:        LONG={n_long}  SHORT={n_short}", flush=True)
        print(f"  Trades:          {n_win} winners, {n_loss} losers", flush=True)
        print(f"  Gross PnL:       Rs {gross_pnl:+,.0f}", flush=True)
        print(f"  NET PnL:         Rs {total_net:+,.0f}", flush=True)
        print(f"  NET PF:          {pf:.3f}", flush=True)
        print(f"  Exit-reason mix: {exit_mix}", flush=True)
    return trades_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 80, flush=True)
    print("Phase 4 sanity -- fno_monthly_rollover_intraday_post_expiry_revert", flush=True)
    print(f"  Universe:        F&O liquid 200 (caps={list(ALLOWED_CAP_SEGMENTS)}, "
          f"MIS={REQUIRE_MIS})", flush=True)
    print(f"  Gap threshold:   |gap_pct| >= {GAP_PCT_ABS_MIN*100:.2f}% "
          f"(gap<0 -> LONG, gap>0 -> SHORT)", flush=True)
    print(f"  Signal bar:      {SIGNAL_BAR_HHMM} 5m (Mode B entry at "
          f"{ENTRY_BAR_HHMM} bar's open)", flush=True)
    print(f"  Time stop:       bar label {TIME_STOP_HHMM_INT} (closes at 10:30 IST)", flush=True)
    print(f"  R-multiples:     T1=+{T1_R_MULT}R, T2=+{T2_R_MULT}R "
          f"(direction-aware, mean-revert)", flush=True)
    print(f"  Risk per trade:  Rs {RISK_PER_TRADE_RUPEES}", flush=True)
    print(f"  Regime cut:      pre/post {REGIME_FAO_CIRCULAR_CUT.isoformat()} "
          f"(FAOP68747 Thursday->Tuesday expiry shift)", flush=True)
    print("=" * 80, flush=True)

    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. F&O liquid 200 universe
    fno_syms = _load_fno_universe(_FNO_UNIVERSE_CSV)
    fno_set: Set[str] = set(fno_syms)
    _log(f"F&O liquid 200 universe loaded: n={len(fno_syms)} (bare symbols)")

    # 2. NSE holidays + T+1 -> T0 map
    holidays = _load_holiday_set(_NSE_HOLIDAYS_JSON)
    _log(f"NSE holidays loaded: n={len(holidays)}")

    tplus1_map = _build_tplus1_map(_FUTURES_BASIS, holidays)
    tplus1_set: Set[date] = set(tplus1_map.keys())
    _log(f"T+1 dates derived from {len(tplus1_map)} monthly expiries "
         f"(sample first 5): {[d.isoformat() for d in sorted(tplus1_set)[:5]]}")
    if len(tplus1_set) >= 10:
        _log(f"  (sample last 5): "
             f"{[d.isoformat() for d in sorted(tplus1_set)[-5:]]}")

    # 3. PDC lookup (consolidated_daily restricted to F&O universe for speed)
    _log("Loading consolidated_daily.feather for PDC lookup...")
    daily_df = pd.read_feather(_DAILY_PATH)
    daily_df["ts"] = _ensure_naive_ist(daily_df["ts"])
    daily_df = daily_df[daily_df["symbol"].isin(fno_set)]
    _log(f"  daily rows (F&O 200): {len(daily_df):,}  "
         f"symbols: {daily_df['symbol'].nunique()}")
    pdc_lookup = _build_pdc_lookup(daily_df)
    _log(f"  PDC lookup built for {len(pdc_lookup):,} symbols")
    del daily_df

    # 4. ProductionUniverseGate (Lesson #19)
    gate = ProductionUniverseGate(
        accepted_caps=set(ALLOWED_CAP_SEGMENTS),
        require_mis=REQUIRE_MIS,
        min_trading_days_required=int(MIN_TRADING_DAYS_REQUIRED),
        min_daily_avg_volume=float(MIN_DAILY_AVG_VOLUME),
    )

    # 5. Per-window run + CSV output
    for window_label in ("discovery", "oos", "holdout"):
        trades_df = run_window(
            window_label, fno_set, tplus1_set, tplus1_map, pdc_lookup, gate,
        )
        out_path = (
            out_dir
            / f"_fno_monthly_rollover_post_expiry_revert_trades_{window_label}.csv"
        )
        trades_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}", flush=True)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
