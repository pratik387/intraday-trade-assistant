"""Phase 4 sanity: 5day_RSI_VWAP_absorb_continuation_long.

Brief: specs/2026-05-22-brief-5day_RSI_VWAP_absorb_continuation_long.md
Predecessor Phase 2: tools/sub9_research/phase2_5day_RSI_VWAP_absorb_continuation_signature.py
Phase 3 pre-registration commit: 605f1d7 (chore(config): pre-register mechanism for ...)

# Anti-bias guards (Lesson #5 6-failure-mode checklist)

  1. No day-aggregate look-ahead. Intraday VWAP is cumulative cum_pv/cum_vol up
     to and INCLUDING the signal bar (real-time computable). Daily RSI(14) uses
     ONLY days T-1 and earlier via .shift(1)/.shift(2)/.shift(3) on the per-symbol
     daily series; NO same-day daily data leaks into the signal.
  2. Volume baseline EXCLUDES current bar. vol_baseline_at_i = mean of intraday
     bars[0..i-1].volume (cumulative prior bars within session, excluding bar i).
     vol_ratio_at_i = bars[i].volume / vol_baseline. Per brief section 5.
  3. Mode B entry at bars[i+1].open. Path walk starts at bars[i+1].
  4. Same-bar SL+target hits: SL wins (pessimistic for LONG = exit at SL,
     ignoring T1/T2 even if both touched in the same bar).
  5. Filters at signal time only. ALL parameters pre-registered in brief
     section 5 and frozen in this script's constants block. NO post-hoc tuning.
  6. Output canonical schema validated downstream via tools.methodology.sanity_csv_schema
     when consumed by cell_sweep / walk_forward.

# Regime gate (candidate-specific)

Mechanism is regime-conditional: works only post-2024 (retail expansion +
passive AUM concentration). Phase 2 evidence: pre-2024 cohort delta -0.006%
(correctly fails), post-2024 +0.162% (passes). The Discovery window is therefore
2024-01-01 to 2024-12-31 (one year, not the conventional two). Phase 5
OOS = 2025-01-01 to 2025-12-31; Holdout = 2026-01-01 to 2026-04-30 (~4 months).

# Universe (Lesson #19)

cap_segment in {small_cap, mid_cap}. EXCLUDES large_cap (C-H 2026-05-22:
large_cap drifts UP in 11-13 window, inverted from this candidate's mechanism).
EXCLUDES unknown (lunch_lull 2026-05-22: ~42% of consolidated_daily is unknown
cap, too noisy). MIS-eligible required. Uses ProductionUniverseGate per-date
(Lesson #19), NOT window-level coverage (which has survivorship bias).

# Mechanic (from brief section 5)

Multi-day signal:  RSI(14)[T-1], RSI(14)[T-2], RSI(14)[T-3] ALL >= 75
                   (sustained 3-day overbought)
Intraday signal:   first VWAP cross-DOWN in 09:30-12:00 IST, where:
                     - bars[i].close < intraday_VWAP_at_i (cumulative since 09:15)
                     - bars[i-1].close >= intraday_VWAP_at_i-1
                     - bars[i].volume / vol_baseline >= 1.2
                       (vol_baseline = mean of intraday prior bars, EXCLUDING bar i)
Entry:             bars[i+1].open (Mode B)
Hard SL (LONG):    min(signal_bar.low * 0.998, entry_price * 0.99)
                   (wider of the two; ensures meaningful R)
T1:                entry + 1.0R
T2:                entry + 2.0R
                   (1R = entry_price - hard_sl)
Time stop:         13:30 IST (5m bar closing at 13:30)
Risk per trade:    Rs 1,000 (matches below_vwap convention)

Absorption (informational; recorded for Phase 5 stratification, NOT a hard gate):
  cross_bar_vol_ratio    = signal-bar vol_ratio
  next_bar_vol_ratio     = (bar[i+1].volume) / (vol_baseline at bar[i+1])
                           where vol_baseline at i+1 = mean of bars[0..i].volume
  absorption_confirmed   = (next_bar_vol_ratio < cross_bar_vol_ratio)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"
_DAILY_PATH = _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather"

sys.path.insert(0, str(_REPO_ROOT))
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402
from tools.sub9_research.production_universe import ProductionUniverseGate  # noqa: E402


# ---------------------------------------------------------------------------
# Pre-registered constants (brief section 5; LOCKED, no post-hoc tuning)
# ---------------------------------------------------------------------------

# Window definitions (regime-gated to post-2024)
WINDOWS = {
    "discovery": (date(2024, 1, 1), date(2024, 12, 31)),
    "oos":       (date(2025, 1, 1), date(2025, 12, 31)),
    "holdout":   (date(2026, 1, 1), date(2026, 4, 30)),
}

# Daily RSI computation (Wilder smoothing)
RSI_PERIOD = 14
RSI_THRESHOLD = 75.0
RSI_SUSTAINED_DAYS = 3   # require RSI[T-1], RSI[T-2], RSI[T-3] all >= threshold

# Intraday signal
SIGNAL_HHMM_MIN = "09:30"
SIGNAL_HHMM_MAX = "12:00"
VOL_RATIO_MIN = 1.2

# Exit parameters
SL_PCT_FROM_LOW = 0.002      # 0.2% below signal-bar low (LONG)
SL_PCT_FROM_ENTRY = 0.01     # 1% below entry (LONG); floor on SL distance
T1_R_MULT = 1.0
T2_R_MULT = 2.0
TIME_STOP_HHMM_INT = 1325    # 5m bar starting 13:25 closes at 13:30 wall-clock
                              # (matches brief target: close_at_1325). Bars are
                              # labeled by START time so 1325-labeled bar = 13:25-13:30.

# Position sizing
RISK_PER_TRADE_RUPEES = 1000

# Universe (Lesson #19)
ALLOWED_CAP_SEGMENTS = ("small_cap", "mid_cap")  # EXCLUDES large_cap AND unknown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _months_between(d0: date, d1: date) -> List[Tuple[int, int]]:
    """Return (year, month) tuples covering [d0, d1] inclusive."""
    out = []
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _compute_rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI(period) via Wilder smoothing (alpha = 1/period).

    Returns float Series aligned with input index. Values are NaN for the first
    `period` observations (insufficient warmup). NO look-ahead — operates only
    on the series and its lagged diffs.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------------------------------------------------------------------------
# Daily RSI table (computed once across all symbols, all dates)
# ---------------------------------------------------------------------------

def _load_daily_rsi_table() -> pd.DataFrame:
    """Load consolidated_daily.feather and compute per-(symbol, date) RSI(14)
    + lagged versions for sustained-RSI gating.

    Returns DataFrame with columns: symbol, d, rsi14, rsi_t1, rsi_t2, rsi_t3,
    rsi_sustained_3d (bool — True iff rsi_t1, rsi_t2, rsi_t3 all >= threshold).

    NO look-ahead: rsi_t{k} = rsi14 shifted by k days within each symbol, so
    when looked up at signal_date D, rsi_t1 corresponds to RSI as of D-1's close.
    """
    print(f"  Loading consolidated_daily.feather for RSI computation...", flush=True)
    df = pd.read_feather(_DAILY_PATH)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df.sort_values(["symbol", "d"]).reset_index(drop=True)
    df["symbol"] = df["symbol"].astype(str)
    df["close"] = df["close"].astype("float64")

    print(f"  Computing RSI({RSI_PERIOD}) Wilder per symbol...", flush=True)
    df["rsi14"] = df.groupby("symbol", observed=True)["close"].transform(
        lambda s: _compute_rsi_wilder(s, RSI_PERIOD)
    )
    df["rsi_t1"] = df.groupby("symbol", observed=True)["rsi14"].shift(1)
    df["rsi_t2"] = df.groupby("symbol", observed=True)["rsi14"].shift(2)
    df["rsi_t3"] = df.groupby("symbol", observed=True)["rsi14"].shift(3)

    df["rsi_sustained_3d"] = (
        (df["rsi_t1"] >= RSI_THRESHOLD)
        & (df["rsi_t2"] >= RSI_THRESHOLD)
        & (df["rsi_t3"] >= RSI_THRESHOLD)
    )
    out = df[["symbol", "d", "rsi_t1", "rsi_t2", "rsi_t3", "rsi_sustained_3d"]].copy()
    print(f"  RSI table built: {len(out):,} (symbol, date) rows; "
          f"{int(out['rsi_sustained_3d'].sum()):,} pass sustained-3-day gate.",
          flush=True)
    return out


# ---------------------------------------------------------------------------
# 5m bars loading + intraday feature computation
# ---------------------------------------------------------------------------

def _load_window(d0: date, d1: date) -> pd.DataFrame:
    """Load 5m bars from monthly feathers covering [d0, d1]."""
    chunks = []
    for (yyyy, mm) in _months_between(d0, d1):
        p = _MONTHLY_DIR / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
        if not p.exists():
            continue
        df = pd.read_feather(
            p,
            columns=["date", "symbol", "open", "high", "low", "close", "volume"],
        )
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype("float32")
        chunks.append(df)
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    df["symbol"] = df["symbol"].astype("category")
    df = df.drop(columns=["date"])
    mask = (df["d"] >= d0) & (df["d"] <= d1)
    return df[mask].reset_index(drop=True)


def _add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(symbol, date) intraday features:
      - cumulative VWAP since 09:15
      - vol_baseline (cumulative prior bars within session, EXCLUDING current)
      - vol_ratio_intraday = bars[i].volume / vol_baseline
      - prev_close, prev_vwap (for VWAP-cross detection)

    Anti-bias guard #1 (no look-ahead): all cumulative quantities use only
    bars up to and INCLUDING the current bar. Cross-detection uses prev_close
    vs prev_vwap (both at bar i-1, no look-ahead).

    Anti-bias guard #2 (vol baseline excludes current): vol_baseline at bar i
    is the mean of bars[0..i-1].volume — current bar excluded.
    """
    df = df.sort_values(["symbol", "d", "hhmm"]).reset_index(drop=True)
    grp = df.groupby(["symbol", "d"], observed=True)

    # Cumulative VWAP (price-volume / volume, cumulative since 09:15)
    df["pv"] = df["close"].astype("float64") * df["volume"].astype("float64")
    df["vol_f64"] = df["volume"].astype("float64")
    df["cum_pv"] = grp["pv"].cumsum()
    df["cum_vol"] = grp["vol_f64"].cumsum()
    df["vwap"] = (df["cum_pv"] / df["cum_vol"].replace(0, np.nan)).astype("float64")

    # Volume baseline (cumulative prior bars within session, EXCLUDING current)
    df["cum_vol_count"] = grp.cumcount() + 1
    # Prior-bars sum = cum_vol (including current) - current bar's volume
    df["vol_prior_sum"] = df["cum_vol"] - df["vol_f64"]
    df["vol_prior_count"] = df["cum_vol_count"] - 1
    df["vol_baseline"] = np.where(
        df["vol_prior_count"] > 0,
        df["vol_prior_sum"] / df["vol_prior_count"],
        np.nan,
    )
    df["vol_ratio_intraday"] = (df["vol_f64"] / df["vol_baseline"]).astype("float64")

    # Previous bar's close / vwap (within session) for cross detection
    df["prev_close"] = grp["close"].shift(1)
    df["prev_vwap"] = grp["vwap"].shift(1)

    # VWAP cross-DOWN detection (transition from >=VWAP to <VWAP)
    df["vwap_cross_down"] = (
        df["prev_close"].notna()
        & df["prev_vwap"].notna()
        & (df["prev_close"] >= df["prev_vwap"])
        & (df["close"] < df["vwap"])
    )

    # Drop intermediates to limit memory
    df = df.drop(columns=["pv", "vol_f64", "cum_pv", "cum_vol", "cum_vol_count",
                          "vol_prior_sum", "vol_prior_count"])
    return df


def _identify_signals(df: pd.DataFrame, daily_rsi: pd.DataFrame) -> pd.DataFrame:
    """Apply locked filter rules to identify LONG-entry candidates.

    Filters (all evaluated at signal-bar's close; no look-ahead beyond i):
      1. VWAP cross-DOWN (computed in _add_intraday_features)
      2. vol_ratio_intraday >= VOL_RATIO_MIN
      3. signal HHMM in [SIGNAL_HHMM_MIN, SIGNAL_HHMM_MAX]
      4. Daily RSI sustained 3-day (joined from daily_rsi on (symbol, d))

    First-fire-per-day latch: take only the FIRST qualifying bar per (sym, d).
    """
    mask = (
        df["vwap_cross_down"]
        & df["vol_ratio_intraday"].notna()
        & (df["vol_ratio_intraday"] >= VOL_RATIO_MIN)
        & (df["hhmm"] >= SIGNAL_HHMM_MIN)
        & (df["hhmm"] <= SIGNAL_HHMM_MAX)
    )
    sig = df.loc[mask, [
        "symbol", "d", "hhmm", "open", "high", "low", "close",
        "volume", "vwap", "vol_baseline", "vol_ratio_intraday",
    ]].copy()
    # Cast symbol back to str for merge
    sig["symbol"] = sig["symbol"].astype(str)
    # Join sustained-RSI flag on (symbol, d)
    sig = sig.merge(daily_rsi, on=["symbol", "d"], how="left")
    sig = sig[sig["rsi_sustained_3d"].fillna(False)].copy()
    # First-fire-per-day latch
    sig = sig.sort_values(["symbol", "d", "hhmm"]).drop_duplicates(
        subset=["symbol", "d"], keep="first"
    ).reset_index(drop=True)
    return sig


# ---------------------------------------------------------------------------
# Session bar index + exit walk
# ---------------------------------------------------------------------------

def _build_session_bars_index(
    df: pd.DataFrame,
) -> Dict[Tuple[str, date], List[Tuple[str, float, float, float, float, float]]]:
    """Build (symbol, date) -> list of (hhmm, open, high, low, close, volume).

    Stored as plain tuples to avoid pandas overhead during the per-signal walk.
    """
    print("  Building per-session bars index...", flush=True)
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
    print(f"  Index built: {len(idx):,} (symbol, date) keys", flush=True)
    return idx


def _hhmm_add_5(hhmm: str) -> str:
    """Add 5 minutes to an HH:MM string."""
    h, m = hhmm.split(":")
    total = int(h) * 60 + int(m) + 5
    return f"{total // 60:02d}:{total % 60:02d}"


def _walk_to_exit_indexed(
    session_bars: List[Tuple[str, float, float, float, float, float]],
    entry_hhmm: str,
    entry_price: float,
    hard_sl: float,
    t1: float,
    t2: float,
    time_stop_hhmm_int: int,
) -> Tuple[float, str, str, float, float]:
    """Walk forward through pre-indexed session bars to find exit (LONG side).

    Anti-bias guard #3: walk starts AT the entry bar (bars[i+1]) — entry was at
    its OPEN, so the bar's full intra-bar range happens AFTER entry.

    Anti-bias guard #4: same-bar SL+T1/T2 picks STOP (pessimistic for LONG).

    Returns: (exit_price, exit_reason, exit_hhmm, mfe_r, mae_r)
    """
    R = entry_price - hard_sl
    if R <= 0:
        return (entry_price, "invalid_R", entry_hhmm, 0.0, 0.0)

    mfe = 0.0
    mae = 0.0
    exit_price = entry_price
    exit_reason = "time_stop"
    exit_hhmm = entry_hhmm
    walked = False

    for (hhmm, op, hi, lo, cl, vol) in session_bars:
        if hhmm < entry_hhmm:
            continue
        walked = True
        if hi > entry_price:
            mfe = max(mfe, hi - entry_price)
        if lo < entry_price:
            mae = max(mae, entry_price - lo)
        cur_hhmm_int = int(hhmm.replace(":", ""))
        if cur_hhmm_int >= time_stop_hhmm_int:
            exit_price = cl
            exit_reason = "time_stop"
            exit_hhmm = hhmm
            break
        sl_hit = lo <= hard_sl
        t2_hit = hi >= t2
        t1_hit = hi >= t1
        if sl_hit and (t1_hit or t2_hit):
            # Same-bar ambiguity: SL wins (pessimistic for LONG)
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


def _next_bar_vol_ratio(
    session_bars: List[Tuple[str, float, float, float, float, float]],
    signal_hhmm: str,
) -> Tuple[float, float]:
    """Return (next_bar_vol_ratio, next_bar_volume) for absorption tracking.

    next_bar_vol_ratio = bars[i+1].volume / (mean of bars[0..i].volume)
    The denominator is the vol baseline AT bar i+1 (cumulative prior bars
    EXCLUDING bar i+1 = all bars up to and including signal bar i).

    Returns (NaN, NaN) if next bar doesn't exist.
    """
    next_hhmm = _hhmm_add_5(signal_hhmm)
    prior_vol_sum = 0.0
    prior_vol_count = 0
    next_volume = float("nan")
    for (hhmm, op, hi, lo, cl, vol) in session_bars:
        if hhmm < next_hhmm:
            prior_vol_sum += vol
            prior_vol_count += 1
        elif hhmm == next_hhmm:
            next_volume = vol
            break
        else:
            break  # next bar missing
    if prior_vol_count <= 0 or not np.isfinite(next_volume):
        return (float("nan"), float("nan"))
    baseline = prior_vol_sum / prior_vol_count
    if baseline <= 0:
        return (float("nan"), next_volume)
    return (next_volume / baseline, next_volume)


def _close_at_indexed(
    session_bars: List[Tuple[str, float, float, float, float, float]],
    target_hhmm: str,
) -> float:
    for (hhmm, op, hi, lo, cl, vol) in session_bars:
        if hhmm == target_hhmm:
            return cl
    return float("nan")


# ---------------------------------------------------------------------------
# Window orchestration
# ---------------------------------------------------------------------------

def run_window(window_label: str, daily_rsi: pd.DataFrame) -> pd.DataFrame:
    d0, d1 = WINDOWS[window_label]
    print(f"\n=== Window: {window_label} ({d0} to {d1}) ===", flush=True)

    df = _load_window(d0, d1)
    print(f"  Loaded {len(df):,} bars", flush=True)
    if df.empty:
        return pd.DataFrame()

    # Universe gate (Lesson #19): per-date filter via ProductionUniverseGate
    gate = ProductionUniverseGate(
        accepted_caps=set(ALLOWED_CAP_SEGMENTS),
        require_mis=True,
        min_trading_days_required=0,   # Lesson #17 — zero legacy filters
        min_daily_avg_volume=0,
    )

    # Pre-compute features on the full window data (cheap; intraday VWAP +
    # vol_baseline + cross detection are per-(sym, d) operations).
    df = _add_intraday_features(df)
    print("  Intraday features computed.", flush=True)

    signals = _identify_signals(df, daily_rsi)
    print(f"  {len(signals):,} signal candidates (after sustained-RSI + first-fire latch)",
          flush=True)
    if signals.empty:
        return pd.DataFrame()

    # Apply per-date universe gate to SIGNAL list (not full data — much faster)
    print("  Applying ProductionUniverseGate per (signal_symbol, signal_date)...",
          flush=True)
    eligible_mask = signals.apply(
        lambda r: gate.is_eligible(str(r["symbol"]), r["d"]), axis=1
    )
    n_before = len(signals)
    signals = signals[eligible_mask].reset_index(drop=True)
    n_rej = n_before - len(signals)
    print(f"  Universe gate kept {len(signals):,} / {n_before:,} "
          f"({n_rej:,} rejected by cap/MIS).", flush=True)
    if signals.empty:
        return pd.DataFrame()

    # Build session bar index for the universe-passing subset.
    # Use merge-based filter (vectorized) — apply-over-24M-rows blows memory.
    keep_df = pd.DataFrame(
        list(set(zip(signals["symbol"].astype(str), signals["d"]))),
        columns=["symbol_str", "d"],
    )
    print(f"  Filtering bars to {len(keep_df):,} (signal_symbol, signal_date) keys "
          f"via merge...", flush=True)
    df["symbol_str"] = df["symbol"].astype(str)
    df_for_idx = df.merge(keep_df, on=["symbol_str", "d"], how="inner")
    df_for_idx = df_for_idx.drop(columns=["symbol_str"])
    del df  # free 24M-row frame before building per-session index
    print(f"  Bars after merge: {len(df_for_idx):,}", flush=True)
    session_idx = _build_session_bars_index(df_for_idx)
    del df_for_idx  # free before per-signal loop

    trades = []
    n_total = len(signals)
    cap_cache: Dict[str, Optional[str]] = {}
    for i, sig in enumerate(signals.itertuples()):
        if i % 500 == 0 and i > 0:
            print(f"    processed {i:,} / {n_total:,}", flush=True)
        sym = str(sig.symbol)
        d_val = sig.d
        signal_hhmm = sig.hhmm

        session_bars = session_idx.get((sym, d_val))
        if not session_bars:
            continue

        # Mode B entry: next 5m bar's OPEN
        entry_hhmm = _hhmm_add_5(signal_hhmm)
        entry_bar = next(
            ((h, o, hi, lo, cl, v) for (h, o, hi, lo, cl, v) in session_bars
             if h == entry_hhmm),
            None,
        )
        if entry_bar is None:
            continue  # no next bar (e.g., signal at session-last bar)
        entry_price = entry_bar[1]
        if entry_price <= 0:
            continue

        # SL (LONG): widest of (signal_low - 0.2%, entry - 1%)
        signal_low = float(sig.low)
        sl_from_low = signal_low * (1.0 - SL_PCT_FROM_LOW)
        sl_from_entry = entry_price * (1.0 - SL_PCT_FROM_ENTRY)
        hard_sl = min(sl_from_low, sl_from_entry)
        if hard_sl >= entry_price:
            continue  # degenerate; skip
        R = entry_price - hard_sl
        t1 = entry_price + T1_R_MULT * R
        t2 = entry_price + T2_R_MULT * R

        exit_price, exit_reason, exit_hhmm, mfe_r, mae_r = _walk_to_exit_indexed(
            session_bars, entry_hhmm,
            entry_price, hard_sl, t1, t2,
            TIME_STOP_HHMM_INT,
        )
        if exit_reason in ("no_data", "invalid_R"):
            continue

        # Absorption signature (informational only)
        cross_bar_vol_ratio = float(sig.vol_ratio_intraday)
        next_bar_vr, _next_vol = _next_bar_vol_ratio(session_bars, signal_hhmm)
        absorption_confirmed = bool(
            np.isfinite(next_bar_vr) and (next_bar_vr < cross_bar_vol_ratio)
        )

        # Cap segment lookup for output (already passed universe gate)
        if sym not in cap_cache:
            cap_cache[sym] = gate._cap_segment(sym)  # noqa: SLF001 (internal cache)
        cap_seg = cap_cache[sym] or "unknown"

        qty = max(1, int(RISK_PER_TRADE_RUPEES / R))
        gross = (exit_price - entry_price) * qty
        fee = calc_fee(entry_price, exit_price, qty, "BUY")
        net_pnl = gross - fee
        # pnl_pct: RAW per-share % return, side-aware, NO fees, NO leverage
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0

        trades.append({
            "signal_date": d_val,
            "symbol": f"NSE:{sym}",
            "side": "LONG",
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
            "r_multiple": (pnl_pct / 100.0 * entry_price) / R if R > 0 else 0.0,
            "t1_target": float(t1),
            "t2_target": float(t2),
            "hard_sl": float(hard_sl),
            "t1_partial_booked": False,
            "mfe_r": float(mfe_r),
            "mae_r": float(mae_r),
            "R_per_share": float(R),
            "cross_bar_vol_ratio": float(cross_bar_vol_ratio),
            "next_bar_vol_ratio": float(next_bar_vr),
            "absorption_confirmed": absorption_confirmed,
            "rsi_t1": float(sig.rsi_t1),
            "rsi_t2": float(sig.rsi_t2),
            "rsi_t3": float(sig.rsi_t3),
            # close_at_1325 = close of the 13:25-13:30 bar = wall-clock 13:30
            # (matches brief Phase 2 target time)
            "close_at_1325": _close_at_indexed(session_bars, "13:25"),
        })

    trades_df = pd.DataFrame(trades)
    print(f"  Generated {len(trades_df):,} trades", flush=True)
    if not trades_df.empty:
        n_win = (trades_df["pnl_pct"] > 0).sum()
        n_loss = (trades_df["pnl_pct"] < 0).sum()
        total_net = trades_df["net_pnl_inr"].sum()
        abs_rate = trades_df["absorption_confirmed"].mean()
        exit_mix = trades_df["exit_reason"].value_counts(normalize=True).round(3).to_dict()
        print(f"  Trades: {n_win} winners, {n_loss} losers, NET Rs {total_net:+,.0f}",
              flush=True)
        print(f"  Absorption-confirmed rate: {abs_rate:.1%}", flush=True)
        print(f"  Exit-reason mix: {exit_mix}", flush=True)
    return trades_df


def main():
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Daily RSI is computed once across full date range; per-window we just filter.
    daily_rsi = _load_daily_rsi_table()

    for window_label in ("discovery", "oos", "holdout"):
        trades_df = run_window(window_label, daily_rsi)
        out_path = (
            out_dir
            / f"_5day_RSI_VWAP_absorb_continuation_long_trades_{window_label}.csv"
        )
        trades_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
