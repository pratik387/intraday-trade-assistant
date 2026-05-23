# tools/sub9_research/phase2_5day_capitulation_thin_wick_squeeze_signature.py
#
# Phase 2 empirical signature for `5day_capitulation_thin_wick_squeeze_long` candidate.
# See specs/2026-05-22-brief-5day_capitulation_thin_wick_squeeze_long.md
#
# Mechanism-rephrase of KILLED `5day_oversold_first_hour_bid_long`. Same SIGNAL
# definition (multi-day capitulation + first-hour lower-wick green bar + 30-min
# low-hold + LONG to 13:30), but the mechanism story is reframed from
# "institutional bid on the wick" (falsified by thin-volume wicks) to
# "thin-tape short-cover squeeze on a primed multi-day-decline tape".
#
# Two NEW falsifiers (replacing the failed wick-vol institutional-bid falsifier):
#
#   Falsifier #1 (delivery% asymmetry, squeeze proxy):
#       For each signal/baseline row, compute 5day_delivery_median = median of
#       delivery_pct[T-5..T-1] (left-join data/delivery_pct/delivery_history.parquet
#       on (symbol, date), prefer 'EQ' series, fall back to other series -- per
#       #4 Phase 2 working code).
#       ACCEPTANCE: signal_median_5day_delivery < baseline_median_5day_delivery - 2pp.
#       (Signal cohort having materially lower delivery% = proxy for elevated
#       short positioning -- the squeeze fuel.)
#
#   Falsifier #2 (next-day gap-up rate, squeeze continuation):
#       For each row, look up T+1 daily open from consolidated_daily.feather.
#       Compute gap = (T+1_open / T_close) - 1, where T_close = signal_close
#       (wick-bar close for signal rows; 09:55 close for baseline rows).
#       Gap-up flag = (gap >= +0.5%).
#       ACCEPTANCE: signal_gap_up_rate > baseline_gap_up_rate + 5pp.
#
# Anti-bias guards (Lesson #5):
#   1. 5-day daily filter uses bars STRICTLY BEFORE session_date (T-6..T-1) -- no look-ahead
#   2. Wick-bar "no fresh low in next 30 min" uses bars[i+1..i+6] (look-ahead is
#      allowed for SIGNAL DETECTION at this research stage; signal is marked at
#      wick-bar close; ret_to_1330 measured from the SAME bar close so the
#      look-ahead is contained inside the same instrument used for the return).
#   3. First-fire-per-day latch (sym, date) -- only first qualifying bar marked
#   4. ProductionUniverseGate per-date (Lesson #19)
#   5. large_cap AND unknown EXCLUDED from universe (small_cap + mid_cap only)
#   6. T+1 daily open is the NEXT trading day's open from consolidated_daily --
#      strict-after-session_date selection.
#   7. 5day_delivery_median uses delivery_pct[T-5..T-1] -- strict-before-session_date.
#
# Pre-registration discipline (Lesson #2):
#   Both Falsifier #1 (delivery%) and Falsifier #2 (gap-up) are evaluated and
#   printed BEFORE the drift-delta step. Either failing -> KILL regardless of drift.
"""Phase 2 empirical signature -- 5day_capitulation_thin_wick_squeeze_long."""
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
    "cumret_lookback_bars":  5,         # 5 trading days
    "cumret_max":            -0.08,     # <= -8%
    "low_lookback_bars":     5,
    "near_low_tolerance":    0.01,      # close[T-1] within 1% of 5-day low

    # ---- Intraday signal-scan window ----
    "sig_window_start":  dtime(9, 30),
    "sig_window_end":    dtime(10, 30),

    # ---- Wick-bar trigger ----
    "wick_body_ratio_min": 0.5,
    "body_eps":            0.0001,
    "no_lower_low_bars":   6,

    # ---- Baseline anchor ----
    "baseline_anchor_time": dtime(9, 55),

    # ---- Target bar -- close at 13:25 5m bar (which IS 13:30 IST exit reference) ----
    "target_bar_time":     dtime(13, 25),

    # ---- Regime / period cuts ----
    "regime_2024_cut":     date(2024, 1, 1),
    "sebi_oct2025_cut":    date(2025, 10, 1),

    # ---- Overlap telemetry: long_panic_gap_down same-day gap-down >= 1.5% ----
    "overlap_gap_down_pct_min": 1.5,

    # ---- Acceptance ----
    "drift_delta_min":            0.15,   # signal_mean - baseline_mean must be >= +0.15%
    "n_signal_min":               200,
    # NEW Falsifier #1: delivery% asymmetry (signal cohort lower delivery%, proxy for short positioning)
    "delivery_median_delta_max":  -2.0,   # (signal_med - baseline_med) MUST be <= -2.0pp
    # NEW Falsifier #2: next-day gap-up corroboration
    "gap_up_threshold":           0.5,    # +0.5% gap-up cutoff
    "gap_up_rate_delta_min":      5.0,    # (signal_rate - baseline_rate) MUST be >= +5.0pp

    # ---- Paths ----
    "monthly_5m_dir":     _REPO_ROOT / "backtest-cache-download" / "monthly",
    "daily_feather_path": _REPO_ROOT / "backtest-cache-download" / "consolidated_daily.feather",
    "delivery_parquet":   _REPO_ROOT / "data" / "delivery_pct" / "delivery_history.parquet",
    "out_csv":            _REPO_ROOT / "reports" / "sub9_sanity" / "_phase2_5day_capitulation_thin_wick_squeeze_signature.csv",
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
# Daily lookup helpers (multi-day capitulation gating + T+1 open + delivery%)
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
    """Check 5-day cumulative oversold + close-near-low filter on T-1 daily bar."""
    prior = sym_daily[sym_daily["d"] < session_date]
    if len(prior) < max(cumret_lookback + 1, low_lookback + 1):
        return None

    closes = prior["close"].to_numpy(dtype=np.float64)
    lows   = prior["low"].to_numpy(dtype=np.float64)
    close_tm1 = float(closes[-1])
    close_tm6 = float(closes[-(cumret_lookback + 1)])
    if close_tm6 <= 0:
        return None

    cumret = (close_tm1 / close_tm6) - 1.0
    if cumret > cumret_max:
        return None

    five_day_low = float(lows[-low_lookback:].min())
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


def _next_day_open(sym_daily: pd.DataFrame, session_date: date) -> Optional[float]:
    """Look up T+1 daily open: FIRST daily bar strictly AFTER session_date."""
    after = sym_daily[sym_daily["d"] > session_date]
    if after.empty:
        return None
    op = float(after["open"].iloc[0])
    if op <= 0:
        return None
    return op


def _5day_delivery_median(
    sym: str,
    session_date: date,
    deliv_by_sym: Dict[str, pd.DataFrame],
    lookback: int,
) -> float:
    """Median of delivery_pct over the last `lookback` trading days strictly
    before session_date. NaN if no rows / fewer than 1 row.

    Note: We use whatever rows exist within the lookback in `deliv_by_sym[sym]`,
    so even partial coverage yields a median (consistent with #4 Phase 2 wiring).
    """
    g = deliv_by_sym.get(sym)
    if g is None or g.empty:
        return float("nan")
    prior = g[g["d"] < session_date]
    if prior.empty:
        return float("nan")
    tail = prior.tail(lookback)
    if tail.empty:
        return float("nan")
    return float(tail["delivery_pct"].median())


# -----------------------------------------------------------------------------
# Per-(symbol, day) evaluator
# -----------------------------------------------------------------------------
def evaluate_symbol_day(
    sym_bars: pd.DataFrame,
    sym: str,
    d: date,
    sym_daily: pd.DataFrame,
    deliv_by_sym: Dict[str, pd.DataFrame],
) -> Optional[dict]:
    """Evaluate one (symbol, day). Returns a record dict OR None.

    Adds three NEW fields (vs predecessor #2):
      - delivery_5d_median  (Falsifier #1)
      - next_day_open       (Falsifier #2 raw input)
      - next_day_gap_pct    (Falsifier #2 derived; gap = T+1_open / T_close - 1)
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

    sym_bars = sym_bars.sort_values("date").reset_index(drop=True)
    mask_sess = (sym_bars["time"] >= dtime(9, 15)) & (sym_bars["time"] <= dtime(15, 25))
    sym_bars = sym_bars[mask_sess].reset_index(drop=True)
    if sym_bars.empty:
        return None

    tgt_t = CONFIG["target_bar_time"]
    tgt_rows = sym_bars[sym_bars["time"] == tgt_t]
    if tgt_rows.empty:
        return None
    target_close = float(tgt_rows.iloc[0]["close"])

    first_bar_rows = sym_bars[sym_bars["time"] == dtime(9, 15)]
    if first_bar_rows.empty:
        return None
    today_first_open = float(first_bar_rows.iloc[0]["open"])
    today_gap_pct = _gap_pct_today(sym_daily, d, today_first_open)

    # ---- T+1 next-day open (Falsifier #2 raw) ----
    nx_open = _next_day_open(sym_daily, d)

    # ---- 5day_delivery_median (Falsifier #1 raw) ----
    deliv_5d = _5day_delivery_median(
        sym, d, deliv_by_sym, int(CONFIG["low_lookback_bars"])
    )

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

    cum_vol = np.cumsum(vols)
    vol_baseline = np.full(n, np.nan, dtype=np.float64)
    idx_arr = np.arange(1, n)
    vol_baseline[1:] = cum_vol[:-1] / idx_arr

    signal_idx: Optional[int] = None
    for i in range(n):
        t = times[i]
        if t < sw_start or t > sw_end:
            continue
        if i + look_ahead >= n:
            break
        body = abs(closes[i] - opens[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        if lower_wick < 0:
            continue
        ratio = lower_wick / max(body, body_eps)
        if ratio < ratio_min:
            continue
        if not (closes[i] > opens[i]):
            continue
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

        gap_down_overlap = (
            (today_gap_pct is not None)
            and (today_gap_pct <= -float(CONFIG["overlap_gap_down_pct_min"]))
        )

        # Falsifier #2 raw gap: T+1_open / sig_close - 1   (signal reference close)
        if nx_open is None or sig_close <= 0:
            next_day_gap_pct = float("nan")
        else:
            next_day_gap_pct = ((nx_open / sig_close) - 1.0) * 100.0

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
            "delivery_5d_median": deliv_5d,
            "next_day_open":     nx_open if nx_open is not None else float("nan"),
            "next_day_gap_pct":  next_day_gap_pct,
        }

    anchor_t = CONFIG["baseline_anchor_time"]
    anchor_rows = sym_bars[sym_bars["time"] == anchor_t]
    if anchor_rows.empty:
        return None
    anchor_close = float(anchor_rows.iloc[0]["close"])
    if anchor_close <= 0:
        return None
    ret_base = (target_close - anchor_close) / anchor_close * 100.0

    # Falsifier #2 gap for baseline: T+1_open / anchor_close - 1
    # (Brief specifies using signal_close consistently as reference.
    # For baseline rows, the "signal close" IS the 09:55 anchor close --
    # that's the closest cash-bar reference price for a non-signal day.)
    if nx_open is None or anchor_close <= 0:
        next_day_gap_pct = float("nan")
    else:
        next_day_gap_pct = ((nx_open / anchor_close) - 1.0) * 100.0

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
        "delivery_5d_median": deliv_5d,
        "next_day_open":     nx_open if nx_open is not None else float("nan"),
        "next_day_gap_pct":  next_day_gap_pct,
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run() -> int:
    print("=" * 80)
    print("Phase 2 empirical signature -- 5day_capitulation_thin_wick_squeeze_long")
    print(f"Window: {CONFIG['window_start']} -> {CONFIG['window_end']}")
    print(f"Caps:   {sorted(CONFIG['accepted_caps'])}  (large_cap + unknown EXCLUDED)")
    print(f"Multi-day filter: 5d_cumret <= {CONFIG['cumret_max']:+.3f} "
          f"AND close[T-1] <= 5d_low * (1+{CONFIG['near_low_tolerance']:.2f})")
    print(f"Wick trigger:     lower_wick/body >= {CONFIG['wick_body_ratio_min']} AND green bar "
          f"AND no lower low in next {CONFIG['no_lower_low_bars']} bars")
    print(f"Sig window:       [{CONFIG['sig_window_start']}, {CONFIG['sig_window_end']}]")
    print(f"Baseline anchor:  {CONFIG['baseline_anchor_time']} (no signal day)")
    print(f"Target bar:       close at {CONFIG['target_bar_time']} (= 13:30 IST exit)")
    print(f"Falsifier #1:     signal_med_5d_delivery < baseline_med_5d_delivery - "
          f"{abs(float(CONFIG['delivery_median_delta_max'])):.1f}pp")
    print(f"Falsifier #2:     signal_gap_up_rate > baseline_gap_up_rate + "
          f"{float(CONFIG['gap_up_rate_delta_min']):.1f}pp  (gap >= +{CONFIG['gap_up_threshold']}%)")
    print("=" * 80)

    gate = ProductionUniverseGate(
        accepted_caps=CONFIG["accepted_caps"],
        require_mis=CONFIG["require_mis"],
        min_trading_days_required=0,
        min_daily_avg_volume=0,
    )

    # ---- consolidated_daily (for cumret/low/T+1 open) ----
    _log("Loading consolidated_daily.feather ...")
    daily_df = pd.read_feather(CONFIG["daily_feather_path"])
    daily_df["ts"] = pd.to_datetime(daily_df["ts"])
    daily_df["d"] = daily_df["ts"].dt.date
    daily_idx = _build_daily_index(daily_df)
    _log(f"  daily symbols indexed: {len(daily_idx):,}")

    # ---- Delivery% lookup table (Lesson #19 wiring per #4 Phase 2) ----
    _log(f"loading delivery%: {CONFIG['delivery_parquet']}")
    deliv = pd.read_parquet(CONFIG["delivery_parquet"])
    deliv["date"] = pd.to_datetime(deliv["date"])
    deliv["d"] = deliv["date"].dt.date
    # Prefer EQ, fall back to whatever exists per (symbol, date).
    deliv["_series_rank"] = (deliv["series"] != "EQ").astype(int)
    deliv = deliv.sort_values(["symbol", "d", "_series_rank"])
    deliv = deliv.drop_duplicates(subset=["symbol", "d"], keep="first")
    # Group by symbol so we can window-slice T-5..T-1 in evaluator
    deliv_by_sym: Dict[str, pd.DataFrame] = {}
    for sym, g in deliv.groupby("symbol", sort=False):
        g2 = g[["d", "delivery_pct"]].sort_values("d").reset_index(drop=True)
        deliv_by_sym[sym] = g2
    _log(f"  delivery% per-symbol groups: {len(deliv_by_sym):,}")

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
        df = df[(df["day"] >= CONFIG["window_start"]) & (df["day"] <= CONFIG["window_end"])]
        if df.empty:
            continue

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
            rec = evaluate_symbol_day(bars, sym, d, sym_daily, deliv_by_sym)
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

    def _cap(s: str) -> str:
        row = nse_all.get(s)
        return row.cap_segment if row else "unknown"

    df_out["cap_segment"] = df_out["symbol"].map(_cap)

    dt_dates = pd.to_datetime(df_out["date"])
    cut_2024 = pd.Timestamp(CONFIG["regime_2024_cut"])
    cut_sebi = pd.Timestamp(CONFIG["sebi_oct2025_cut"])
    df_out["regime_pre_post_2024"] = np.where(dt_dates < cut_2024, "pre", "post")
    df_out["period_pre_post_sebi_oct2025"] = np.where(dt_dates < cut_sebi, "pre", "post")

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

    def _deliv_bucket(v: float) -> str:
        if pd.isna(v):
            return "nan"
        if v < 30.0:
            return "<30%"
        if v < 40.0:
            return "30-40%"
        if v < 50.0:
            return "40-50%"
        return ">=50%"

    df_out["deliv_5d_bucket"] = df_out["delivery_5d_median"].map(_deliv_bucket)

    out_path = Path(CONFIG["out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out):,} rows -> {out_path}")

    sig = df_out[df_out["is_signal"]].copy()
    base = df_out[df_out["is_baseline"]].copy()
    n_sig = len(sig)
    n_base = len(base)

    # -------------------------------------------------------------------------
    # STEP 1 -- Falsifier #1 (delivery% asymmetry)   --   BEFORE DRIFT
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 1 -- Falsifier #1 (delivery% asymmetry)")
    print("=" * 80)

    sig_dp = sig["delivery_5d_median"].dropna()
    base_dp = base["delivery_5d_median"].dropna()
    sig_dp_cov = float(len(sig_dp)) / max(1, n_sig)
    base_dp_cov = float(len(base_dp)) / max(1, n_base)

    print(f"  n_signal:                      {n_sig:,}")
    print(f"  n_baseline:                    {n_base:,}")
    print(f"  5d-delivery% coverage:         signal={sig_dp_cov*100:.1f}%, baseline={base_dp_cov*100:.1f}%")

    if len(sig_dp) == 0 or len(base_dp) == 0:
        print("  delivery% data missing -- cannot evaluate Falsifier #1.")
        print("\n" + "=" * 80)
        print("VERDICT: KILL (Falsifier #1 not evaluable -- delivery% join failed)")
        print("=" * 80)
        return 0

    sig_dp_med  = float(sig_dp.median())
    base_dp_med = float(base_dp.median())
    sig_dp_mean  = float(sig_dp.mean())
    base_dp_mean = float(base_dp.mean())
    deliv_delta = sig_dp_med - base_dp_med

    delta_max = float(CONFIG["delivery_median_delta_max"])  # negative threshold (e.g. -2.0)
    print(f"  signal   5d-delivery% median:  {sig_dp_med:.3f}%   (mean {sig_dp_mean:.3f}%)")
    print(f"  baseline 5d-delivery% median:  {base_dp_med:.3f}%   (mean {base_dp_mean:.3f}%)")
    print(f"  delta (sig - base) MEDIAN:     {deliv_delta:+.3f}pp   [<= {delta_max:+.2f}pp required]")

    falsifier1_pass = deliv_delta <= delta_max
    print(f"\n  FALSIFIER #1: {'PASS' if falsifier1_pass else 'FAIL'}")

    if not falsifier1_pass:
        print("\n  Signal cohort 5-day delivery%% is NOT materially below baseline.")
        print("  Short-positioning proxy fails -> squeeze fuel absent.")

    # -------------------------------------------------------------------------
    # STEP 1.5 -- Falsifier #2 (next-day gap-up rate)   --   BEFORE DRIFT
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 1.5 -- Falsifier #2 (next-day gap-up rate)")
    print("=" * 80)

    gap_thr = float(CONFIG["gap_up_threshold"])
    gap_rate_delta_min = float(CONFIG["gap_up_rate_delta_min"])

    sig_gap = sig["next_day_gap_pct"].dropna()
    base_gap = base["next_day_gap_pct"].dropna()
    sig_gap_cov = float(len(sig_gap)) / max(1, n_sig)
    base_gap_cov = float(len(base_gap)) / max(1, n_base)

    print(f"  T+1 gap coverage:              signal={sig_gap_cov*100:.1f}%, baseline={base_gap_cov*100:.1f}%")

    if len(sig_gap) == 0 or len(base_gap) == 0:
        print("  T+1 open data missing -- cannot evaluate Falsifier #2.")
        print("\n" + "=" * 80)
        print("VERDICT: KILL (Falsifier #2 not evaluable -- T+1 open lookup failed)")
        print("=" * 80)
        return 0

    sig_gap_up_rate  = float((sig_gap >= gap_thr).mean()) * 100.0
    base_gap_up_rate = float((base_gap >= gap_thr).mean()) * 100.0
    gap_rate_delta = sig_gap_up_rate - base_gap_up_rate

    sig_gap_med  = float(sig_gap.median())
    base_gap_med = float(base_gap.median())
    sig_gap_mean  = float(sig_gap.mean())
    base_gap_mean = float(base_gap.mean())

    print(f"  signal   T+1 gap median:       {sig_gap_med:+.3f}%   (mean {sig_gap_mean:+.3f}%)  n={len(sig_gap):,}")
    print(f"  baseline T+1 gap median:       {base_gap_med:+.3f}%   (mean {base_gap_mean:+.3f}%)  n={len(base_gap):,}")
    print(f"  signal   gap-up rate (>= +{gap_thr}%):    {sig_gap_up_rate:.2f}%")
    print(f"  baseline gap-up rate (>= +{gap_thr}%):    {base_gap_up_rate:.2f}%")
    print(f"  delta (sig - base) gap-up rate: {gap_rate_delta:+.2f}pp   [>= +{gap_rate_delta_min:.1f}pp required]")

    falsifier2_pass = gap_rate_delta >= gap_rate_delta_min
    print(f"\n  FALSIFIER #2: {'PASS' if falsifier2_pass else 'FAIL'}")

    if not falsifier2_pass:
        print("\n  Signal cohort T+1 gap-up rate is NOT materially above baseline.")
        print("  Squeeze continuation into next session not corroborated.")

    # -------------------------------------------------------------------------
    # STEP 2 -- Drift delta (always reported for diagnostic value)
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("STEP 2 -- Drift delta (LONG)")
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
    # STEP 3 -- Cohort splits + telemetry
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
        print(f"{label:<32}{ns:>8d}{nb:>8d}{sm:>12.4f}{bm:>12.4f}{dl:>12.4f}")
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
    print("  --- 5-day delivery% buckets (signal vs baseline by bucket) ---")
    print(f"{'split':<32}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    for bucket in ["<30%", "30-40%", "40-50%", ">=50%"]:
        _split_block(
            f"deliv_5d={bucket}",
            sig["deliv_5d_bucket"] == bucket,
            base["deliv_5d_bucket"] == bucket,
        )

    print()
    print("  --- wick/body ratio buckets (signal rows; baseline = aggregate baseline) ---")
    print(f"{'bucket':<32}{'n_sig':>8}{'n_base':>8}{'sig_mean':>12}{'base_mean':>12}{'delta':>12}")
    for bucket in ["0.5-1.0", "1.0-2.0", ">=2.0"]:
        mask_s = sig["wick_body_bucket"] == bucket
        s = sig[mask_s]
        ns = len(s)
        sm = float(s["ret_to_1330"].mean()) if ns else float("nan")
        bm = base_mean
        dl = sm - bm if (ns and not np.isnan(bm)) else float("nan")
        print(
            f"wick_body_bucket={bucket:<14}"
            f"{ns:>8d}"
            f"{n_base:>8d}"
            f"{sm:>12.4f}"
            f"{bm:>12.4f}"
            f"{dl:>12.4f}"
        )

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
    sign_flip = (not np.isnan(post_sebi_d)) and (post_sebi_d < 0) and (post_sebi_n >= 50)

    if not n_ok:
        verdict = "KILL"
        reason = f"n_signal {n_sig} < {CONFIG['n_signal_min']}"
    elif not falsifier1_pass:
        verdict = "KILL"
        reason = (
            f"Falsifier #1 FAIL (delivery delta {deliv_delta:+.3f}pp > {delta_max:+.2f}pp "
            f"-- squeeze fuel absent)"
        )
    elif not falsifier2_pass:
        verdict = "KILL"
        reason = (
            f"Falsifier #2 FAIL (gap-up rate delta {gap_rate_delta:+.2f}pp < "
            f"+{gap_rate_delta_min:.1f}pp -- squeeze continuation absent)"
        )
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
            f"F1 PASS (delivery {deliv_delta:+.3f}pp <= {delta_max:+.2f}pp), "
            f"F2 PASS (gap-up {gap_rate_delta:+.2f}pp >= +{gap_rate_delta_min:.1f}pp), "
            f"drift {delta:+.4f}% >= +{CONFIG['drift_delta_min']:.2f}%, "
            f"n_signal {n_sig} >= {CONFIG['n_signal_min']}, no post-SEBI sign flip"
        )

    print(f"  {verdict}")
    print(f"  reason: {reason}")

    return 0


if __name__ == "__main__":
    sys.exit(run())
