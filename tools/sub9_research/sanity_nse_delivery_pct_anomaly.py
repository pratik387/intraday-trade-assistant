"""sanity_nse_delivery_pct_anomaly.py — Sub-9 Lane 1 sanity.

Per brief: specs/2026-05-08-sub-project-9-brief-nse_delivery_pct_anomaly.md

Mechanic:
  T-day signal (post-EOD bhavcopy):
    pump:        delivery_pct < 20 AND daily_return > +3.0%
    accumulation: delivery_pct > 60 AND daily_return > +2.0%

  T+1 09:30-11:00 confirmation candle entry:
    pump_fade SHORT:  close < open AND close < session_vwap
                      AND bar cross-day RVOL >= 1.0
    accumulation LONG: close > open AND close > session_vwap
                       AND bar cross-day RVOL >= 1.2
                       AND pullback-from-day-high < 1%
    Latch: one fire per (symbol, T+1, side)

  Stops:
    SHORT: min(09:15 bar high × 1.005, T-day close × 1.012)
    LONG:  09:15 bar low × 0.995

  Targets: T1=1R (50% qty), T2=2R (50% qty)
  Time stop: 13:00 IST hard exit

  Universe: cap-broad; mis_leverage>=1.0; 20d ADV*close >= ₹2 Cr.

Usage (per user directive: Discovery ONLY first):
    python tools/sub9_research/sanity_nse_delivery_pct_anomaly.py
"""
from __future__ import annotations

import gc
import os
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from services.state.zerodha_mis_fetcher import ZerodhaMISFetcher  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---------------- MIS filter (production parity) ----------------
# Production's early_mis_universe_filter trims the universe to MIS-allowed
# symbols (~1668 of ~2440 NSE EQs). Sanity must mirror this — counting
# trades on non-MIS-tradeable symbols (e.g. BHAGERIA, SUMMITSEC) inflates
# PF/n with un-executable signals. Caveat: this is the CURRENT MIS list
# applied as a proxy for the whole period; historical drift is unmodeled.
_MIS_FETCHER: Optional[ZerodhaMISFetcher] = None


def _get_mis_allowed_set() -> set:
    """Return set of bare NSE symbols allowed for MIS (intraday short)."""
    global _MIS_FETCHER
    if _MIS_FETCHER is None:
        _MIS_FETCHER = ZerodhaMISFetcher()
        ok = _MIS_FETCHER.load_from_zerodha()
        if not ok:
            print("  WARN: MIS list load failed — proceeding without MIS filter")
            return set()
        print(f"  MIS list loaded: {_MIS_FETCHER.count()} symbols")
    return set(_MIS_FETCHER._mis_symbols.keys())

# ---------------- Period env override ----------------
_PERIOD = os.environ.get("SANITY_PERIOD", "discovery").lower()
if _PERIOD == "oos":
    DISCOVERY_START, DISCOVERY_END = date(2025, 1, 1), date(2025, 9, 30)
    _Y_RANGE = (2025,)
    _M_RANGE_FN = lambda y: range(1, 10)
    _OUT_SUFFIX = "_oos"
elif _PERIOD == "holdout":
    DISCOVERY_START, DISCOVERY_END = date(2025, 10, 1), date(2026, 4, 30)
    _Y_RANGE = (2025, 2026)
    _M_RANGE_FN = lambda y: range(10, 13) if y == 2025 else range(1, 5)
    _OUT_SUFFIX = "_holdout"
else:
    DISCOVERY_START, DISCOVERY_END = date(2023, 1, 1), date(2024, 12, 30)
    _Y_RANGE = (2023, 2024)
    _M_RANGE_FN = lambda y: range(1, 13)
    _OUT_SUFFIX = ""

# ---------------- Brief §6 locked params ----------------
PUMP_DELIVERY_PCT_MAX = 20.0
PUMP_PRIOR_RETURN_MIN = 3.0
ACCUM_DELIVERY_PCT_MIN = 60.0
ACCUM_PRIOR_RETURN_MIN = 2.0

GAP_MIN_SHORT, GAP_MAX_SHORT = -2.0, 3.0
GAP_MIN_LONG, GAP_MAX_LONG = -1.0, 2.0

ACTIVE_START_HHMM = 930
ACTIVE_END_HHMM = 1100
TIME_STOP_HHMM = 1300
OPEN_BAR_HHMM = 915

VOL_RATIO_SHORT_MIN = 1.0
VOL_RATIO_LONG_MIN = 1.2
PULLBACK_LONG_MAX_PCT = 1.0

STOP_OPEN_HIGH_BUFFER = 0.005   # 0.5%
STOP_TDAY_CLOSE_BUFFER = 0.012  # 1.2%
LONG_STOP_OPEN_LOW_BUFFER = 0.005

T1_R_MULT, T2_R_MULT = 1.0, 2.0
T1_QTY_PCT = 0.5

MIN_LIQUIDITY_CR = 2.0
RISK_PER_TRADE = 1000.0

RVOL_LOOKBACK_DAYS = 20
RVOL_MIN_PRIOR = 5

PF_FLOOR = 1.10
N_FLOOR = 500
WR_BAND_PP = 10.0


# ---------------- Data loading ----------------
def load_delivery_pct() -> pd.DataFrame:
    """Load 1.82M-row delivery_history.parquet, filter to EQ, return per (symbol, date)."""
    print("  loading delivery_history.parquet ...")
    df = pd.read_parquet(_REPO / "data" / "delivery_pct" / "delivery_history.parquet")
    df = df[df["series"] == "EQ"].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["symbol", "date", "delivery_pct", "total_traded_qty",
             "total_traded_value", "close_price"]].copy()
    print(f"  delivery rows (EQ only): {len(df):,}")
    return df


def load_consolidated_daily() -> pd.DataFrame:
    print("  loading consolidated_daily.feather ...")
    df = pd.read_feather(_REPO / "cache" / "preaggregate" / "consolidated_daily.feather")
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["date"] = df["ts"].dt.date
    df = df[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    return df


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    p = _REPO / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_feather(p, columns=["date", "symbol", "open", "high", "low",
                                      "close", "volume", "vwap"])
    for c in ("open", "high", "low", "close", "vwap"):
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("float32")
    return df


# ---------------- Signal computation ----------------
def compute_t_day_signals(delivery: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Merge delivery + daily, compute pump + accumulation signals.

    Returns events DataFrame with one row per (symbol, T_date) where signal fires.
    """
    print("  merging delivery_pct with consolidated_daily ...")
    daily = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    daily["pdc"] = daily.groupby("symbol")["close"].shift(1)
    daily["daily_return_pct"] = (daily["close"] - daily["pdc"]) / daily["pdc"] * 100.0
    daily["adv_20d_cr"] = daily.groupby("symbol")["volume"].transform(
        lambda v: v.shift(1).rolling(20).mean()
    ) * daily["close"] / 1e7

    m = daily.merge(delivery, on=["symbol", "date"], how="inner")
    m = m.dropna(subset=["delivery_pct", "daily_return_pct", "adv_20d_cr"])

    # Liquidity gate
    m = m[m["adv_20d_cr"] >= MIN_LIQUIDITY_CR].copy()
    print(f"  after liquidity gate (adv*close>={MIN_LIQUIDITY_CR}cr): {len(m):,}")

    # MIS-eligibility gate (production parity — see header comment).
    mis_set = _get_mis_allowed_set()
    if mis_set:
        before = len(m)
        m = m[m["symbol"].isin(mis_set)].copy()
        print(f"  after MIS filter (Zerodha current list): {len(m):,} (-{before-len(m):,})")

    # Signal flags
    m["pump_signal"] = (m["delivery_pct"] < PUMP_DELIVERY_PCT_MAX) & \
                      (m["daily_return_pct"] > PUMP_PRIOR_RETURN_MIN)
    m["accum_signal"] = (m["delivery_pct"] >= ACCUM_DELIVERY_PCT_MIN) & \
                       (m["daily_return_pct"] > ACCUM_PRIOR_RETURN_MIN)
    m = m[m["pump_signal"] | m["accum_signal"]].copy()
    print(f"  signal events (pump | accum): {len(m):,}")
    print(f"    pump only:  {(m['pump_signal'] & ~m['accum_signal']).sum():,}")
    print(f"    accum only: {(m['accum_signal'] & ~m['pump_signal']).sum():,}")

    # Discovery window for T+1 entry: T_date+1 must fall in [DISCOVERY_START, DISCOVERY_END]
    # Keep t_day as datetime.date so it compares cleanly with big5m["d"] later.
    m["t_day"] = m["date"]   # datetime.date
    # Date filter using date objects directly (avoid Timestamp/date comparison issue)
    _start_d = DISCOVERY_START
    _end_d_plus = DISCOVERY_END + pd.Timedelta(days=5).to_pytimedelta()
    m = m[(m["t_day"] >= _start_d) & (m["t_day"] <= _end_d_plus)].copy()
    print(f"  signals with T_day in period (incl 5d weekend buffer): {len(m):,}")
    return m


# ---------------- 5m bar window utilities ----------------
def derive_hhmm(ts: pd.Series) -> pd.Series:
    return (ts.dt.hour.astype("int16") * 100 + ts.dt.minute.astype("int16")).astype("int16")


def attach_cross_day_rvol(big5m: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, hhmm): rolling 20-prior-session mean volume → cross_day_rvol."""
    print("  computing cross-day RVOL per (symbol, hhmm) ...")
    big5m = big5m.sort_values(["symbol", "hhmm", "d"]).reset_index(drop=True)
    grp = big5m.groupby(["symbol", "hhmm"], sort=False, observed=True)
    big5m["_vol_shift"] = grp["volume"].shift(1)
    big5m["_vol_mean20"] = grp["_vol_shift"].rolling(
        RVOL_LOOKBACK_DAYS, min_periods=RVOL_MIN_PRIOR
    ).mean().reset_index(level=[0, 1], drop=True)
    big5m["cross_day_rvol"] = big5m["volume"] / big5m["_vol_mean20"].replace(0, np.nan)
    big5m = big5m.drop(columns=["_vol_shift", "_vol_mean20"])
    print(f"  bars with valid cross-day RVOL: {big5m['cross_day_rvol'].notna().sum():,}")
    return big5m


# ---------------- Trade simulation ----------------
def simulate_one(symbol: str, t1_date: pd.Timestamp, side: str,
                 t_day_close: float, day_bars: pd.DataFrame) -> Optional[Dict]:
    """Simulate a single signal's T+1 trade. Returns trade dict or None."""
    if day_bars.empty:
        return None
    open_bar = day_bars[day_bars["hhmm"] == OPEN_BAR_HHMM]
    if open_bar.empty:
        return None
    open_bar = open_bar.iloc[0]
    open_09_15 = float(open_bar["open"])
    high_09_15 = float(open_bar["high"])
    low_09_15 = float(open_bar["low"])

    # Gap filter
    gap_pct = (open_09_15 - t_day_close) / t_day_close * 100.0
    if side == "SHORT" and not (GAP_MIN_SHORT <= gap_pct <= GAP_MAX_SHORT):
        return None
    if side == "LONG" and not (GAP_MIN_LONG <= gap_pct <= GAP_MAX_LONG):
        return None

    # Active window bars
    active = day_bars[(day_bars["hhmm"] >= ACTIVE_START_HHMM) &
                     (day_bars["hhmm"] <= ACTIVE_END_HHMM)].sort_values("hhmm")
    if active.empty:
        return None

    # Find first confirmation candle
    confirm_bar = None
    if side == "SHORT":
        # close < open AND close < vwap AND rvol >= VOL_RATIO_SHORT_MIN
        cand = active[(active["close"] < active["open"]) &
                      (active["close"] < active["vwap"]) &
                      (active["cross_day_rvol"] >= VOL_RATIO_SHORT_MIN)]
        if not cand.empty:
            confirm_bar = cand.iloc[0]
    else:  # LONG
        # close > open AND close > vwap AND rvol >= VOL_RATIO_LONG_MIN AND
        # pullback < 1% (close vs running session high so far)
        cumhigh = active["high"].cummax()
        pullback_pct = (cumhigh - active["close"]) / cumhigh * 100.0
        cand = active[(active["close"] > active["open"]) &
                      (active["close"] > active["vwap"]) &
                      (active["cross_day_rvol"] >= VOL_RATIO_LONG_MIN) &
                      (pullback_pct < PULLBACK_LONG_MAX_PCT)]
        if not cand.empty:
            confirm_bar = cand.iloc[0]
    if confirm_bar is None:
        return None

    entry_price = float(confirm_bar["close"])
    entry_ts = confirm_bar["date"]

    # Stop
    if side == "SHORT":
        stop_a = high_09_15 * (1.0 + STOP_OPEN_HIGH_BUFFER)
        stop_b = t_day_close * (1.0 + STOP_TDAY_CLOSE_BUFFER)
        hard_sl = min(stop_a, stop_b)
        if hard_sl <= entry_price:
            return None  # Geometry invalid
    else:  # LONG
        hard_sl = low_09_15 * (1.0 - LONG_STOP_OPEN_LOW_BUFFER)
        if hard_sl >= entry_price:
            return None
    R = abs(entry_price - hard_sl)
    if R <= 0:
        return None

    if side == "SHORT":
        t1_target = entry_price - R * T1_R_MULT
        t2_target = entry_price - R * T2_R_MULT
    else:
        t1_target = entry_price + R * T1_R_MULT
        t2_target = entry_price + R * T2_R_MULT

    # Forward bars after entry, until time_stop or exhaustion
    forward = day_bars[day_bars["date"] > entry_ts].sort_values("date")
    if forward.empty:
        return None
    forward = forward[forward["hhmm"] <= TIME_STOP_HHMM]

    qty = max(int(RISK_PER_TRADE / max(R, 1e-6)), 1)
    qty = min(qty, 100000)  # cap

    hit_t1 = False
    t1_exit_price = None
    exit_price = None
    exit_ts = None
    exit_reason = None
    actual_sl = hard_sl

    # MFE/MAE tracking (research-grounded target sweep)
    mfe_price = entry_price   # most-favorable price reached
    mae_price = entry_price   # most-adverse price reached
    # Close prices at potential time-stop checkpoints (for sweep)
    closes_at_hhmm = {}        # hhmm -> close price

    for _, bar in forward.iterrows():
        hi, lo, cl = float(bar["high"]), float(bar["low"]), float(bar["close"])
        bar_ts = bar["date"]
        bar_hhmm = int(bar["hhmm"])

        # Track MFE/MAE BEFORE checking exits (so we capture the full path)
        if side == "SHORT":
            mfe_price = min(mfe_price, lo)
            mae_price = max(mae_price, hi)
        else:
            mfe_price = max(mfe_price, hi)
            mae_price = min(mae_price, lo)

        # Snapshot close at end-of-bar for each "checkpoint hhmm"
        # We track close at every bar; the "what-if time stop" sweep can pick any
        closes_at_hhmm[bar_hhmm] = cl

        if side == "SHORT":
            if hi >= actual_sl:
                exit_price = actual_sl
                exit_ts = bar_ts
                exit_reason = "stop"
                break
            if not hit_t1 and lo <= t1_target:
                hit_t1 = True
                t1_exit_price = t1_target
                actual_sl = entry_price
            if hit_t1 and lo <= t2_target:
                exit_price = t2_target
                exit_ts = bar_ts
                exit_reason = "t2"
                break
        else:  # LONG
            if lo <= actual_sl:
                exit_price = actual_sl
                exit_ts = bar_ts
                exit_reason = "stop"
                break
            if not hit_t1 and hi >= t1_target:
                hit_t1 = True
                t1_exit_price = t1_target
                actual_sl = entry_price
            if hit_t1 and hi >= t2_target:
                exit_price = t2_target
                exit_ts = bar_ts
                exit_reason = "t2"
                break
        if bar_hhmm >= TIME_STOP_HHMM:
            exit_price = cl
            exit_ts = bar_ts
            exit_reason = "time_stop"
            break

    if exit_price is None:
        last = forward.iloc[-1]
        exit_price = float(last["close"])
        exit_ts = last["date"]
        exit_reason = "eod"

    # Compute MFE/MAE in R-units (R = stop_distance)
    if side == "SHORT":
        mfe_r = (entry_price - mfe_price) / R if R > 0 else 0.0
        mae_r = (mae_price - entry_price) / R if R > 0 else 0.0
    else:
        mfe_r = (mfe_price - entry_price) / R if R > 0 else 0.0
        mae_r = (entry_price - mae_price) / R if R > 0 else 0.0

    # Pick close at standard checkpoints for time-stop sweep
    def _close_at(hhmm: int) -> float:
        # Closest available bar close at-or-before hhmm
        eligible = [v for k, v in closes_at_hhmm.items() if k <= hhmm]
        return float(eligible[-1]) if eligible else float("nan")

    close_at_1100 = _close_at(1100)
    close_at_1200 = _close_at(1200)
    close_at_1300 = _close_at(1300)

    # PnL — calc_fee(entry_price, exit_price, qty, side) returns round-trip fees.
    # side: "BUY" for LONG (opened BUY), anything else (e.g. "SELL") for SHORT.
    fee_side = "BUY" if side == "LONG" else "SELL"
    if side == "SHORT":
        if hit_t1:
            partial_q = max(int(qty * T1_QTY_PCT), 1)
            remain_q = qty - partial_q
            gross_partial = (entry_price - t1_exit_price) * partial_q
            gross_remain = (entry_price - exit_price) * remain_q
            gross = gross_partial + gross_remain
            fee = (calc_fee(entry_price, t1_exit_price, partial_q, fee_side) +
                   calc_fee(entry_price, exit_price, remain_q, fee_side))
        else:
            gross = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, fee_side)
    else:  # LONG
        if hit_t1:
            partial_q = max(int(qty * T1_QTY_PCT), 1)
            remain_q = qty - partial_q
            gross_partial = (t1_exit_price - entry_price) * partial_q
            gross_remain = (exit_price - entry_price) * remain_q
            gross = gross_partial + gross_remain
            fee = (calc_fee(entry_price, t1_exit_price, partial_q, fee_side) +
                   calc_fee(entry_price, exit_price, remain_q, fee_side))
        else:
            gross = (exit_price - entry_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, fee_side)
    net_pnl = gross - fee

    return {
        "symbol": symbol,
        "t_day": t1_date - pd.Timedelta(days=1),  # ish; actual T+1 entry day is t1_date
        "t1_date": t1_date,
        "side": side,
        "t_day_close": t_day_close,
        "open_09_15": open_09_15,
        "high_09_15": high_09_15,
        "low_09_15": low_09_15,
        "gap_pct": gap_pct,
        "entry_ts": entry_ts,
        "entry_price": entry_price,
        "hard_sl": hard_sl,
        "stop_distance": R,
        "t1_target": t1_target,
        "t2_target": t2_target,
        "qty": qty,
        "hit_t1": hit_t1,
        "t1_exit_price": t1_exit_price,
        "exit_ts": exit_ts,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_pnl": gross,
        "fee": fee,
        "net_pnl": net_pnl,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "close_at_1100": close_at_1100,
        "close_at_1200": close_at_1200,
        "close_at_1300": close_at_1300,
    }


# ---------------- Main per-month loop ----------------
def _resolve_t1_dates(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Find EXACT next trading date after t_day per (symbol, t_day) using the
    global daily calendar. Each event resolves to exactly one entry_d.
    """
    print("  resolving T+1 dates per (symbol, t_day) ...")
    daily_sorted = daily.sort_values(["symbol", "date"]).copy()
    daily_sorted["next_d"] = daily_sorted.groupby("symbol")["date"].shift(-1)
    lookup = daily_sorted.set_index(["symbol", "date"])["next_d"]
    events = events.copy()
    keys = list(zip(events["symbol"], events["t_day"]))
    events["entry_d"] = [lookup.get(k, None) for k in keys]
    events = events.dropna(subset=["entry_d"])
    events = events[(events["entry_d"] >= DISCOVERY_START) &
                    (events["entry_d"] <= DISCOVERY_END)].copy()
    print(f"  events with valid T+1 in period: {len(events):,}")
    return events


def run_period(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    events = _resolve_t1_dates(events, daily)
    if events.empty:
        return pd.DataFrame()

    trades: List[Dict] = []
    for yyyy in _Y_RANGE:
        for mm in _M_RANGE_FN(yyyy):
            big5m = _load_5m_for_month(yyyy, mm)
            if big5m.empty:
                continue
            big5m["d"] = big5m["date"].dt.date
            big5m["hhmm"] = derive_hhmm(big5m["date"])
            big5m = attach_cross_day_rvol(big5m)
            month_d_min = big5m["d"].min()
            month_d_max = big5m["d"].max()
            print(f"  [{yyyy}-{mm:02d}] 5m bars: {len(big5m):,} "
                  f"({month_d_min}..{month_d_max})")

            sub_evts = events[(events["entry_d"] >= month_d_min) &
                              (events["entry_d"] <= month_d_max)].copy()
            if sub_evts.empty:
                del big5m
                gc.collect()
                continue
            print(f"    events firing in this month: {len(sub_evts):,}")

            # Group bars by (symbol, d) for fast lookup
            big5m_idx = big5m.set_index(["symbol", "d"]).sort_index()

            for _, ev in sub_evts.iterrows():
                sym = ev["symbol"]
                edate = ev["entry_d"]
                key = (sym, edate)
                if key not in big5m_idx.index:
                    continue
                day_bars = big5m_idx.loc[[key]].reset_index() if isinstance(big5m_idx.loc[key], pd.DataFrame) \
                    else big5m_idx.loc[[key]].reset_index()
                # Day bars
                t_day_close = float(ev["close"])
                # SHORT (pump fade) attempt
                if bool(ev["pump_signal"]):
                    tr = simulate_one(sym, pd.Timestamp(edate), "SHORT",
                                     t_day_close, day_bars)
                    if tr is not None:
                        tr["signal_type"] = "pump_fade"
                        tr["delivery_pct"] = float(ev["delivery_pct"])
                        tr["daily_return_pct"] = float(ev["daily_return_pct"])
                        tr["adv_20d_cr"] = float(ev["adv_20d_cr"])
                        trades.append(tr)
                # LONG (accumulation) attempt
                if bool(ev["accum_signal"]):
                    tr = simulate_one(sym, pd.Timestamp(edate), "LONG",
                                     t_day_close, day_bars)
                    if tr is not None:
                        tr["signal_type"] = "accumulation"
                        tr["delivery_pct"] = float(ev["delivery_pct"])
                        tr["daily_return_pct"] = float(ev["daily_return_pct"])
                        tr["adv_20d_cr"] = float(ev["adv_20d_cr"])
                        trades.append(tr)

            del big5m, big5m_idx
            gc.collect()
    return pd.DataFrame(trades)


# ---------------- Reporting ----------------
def pf_of(s: pd.Series) -> float:
    g = s[s > 0].sum()
    l = -s[s < 0].sum()
    return float(g / l) if l > 0 else float("inf")


def report(trades: pd.DataFrame):
    if trades.empty:
        print("\n[NO TRADES]")
        return

    print(f"\n{'='*70}\nAGGREGATE\n{'='*70}")
    n = len(trades)
    pf = pf_of(trades["net_pnl"])
    wr = 100 * (trades["net_pnl"] > 0).mean()
    net = trades["net_pnl"].sum()
    daily = trades.set_index(pd.to_datetime(trades["entry_ts"]).dt.date)["net_pnl"].groupby(level=0).sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else float("nan")
    print(f"  n={n:,}  PF={pf:.3f}  WR={wr:.1f}%  Sharpe(d)={sharpe:.3f}  NET=Rs.{net:,.0f}")

    print(f"\nPer side:")
    for side, g in trades.groupby("side"):
        p = pf_of(g["net_pnl"])
        w = 100*(g["net_pnl"]>0).mean()
        print(f"  {side}: n={len(g):,} PF={p:.3f} WR={w:.1f}% NET=Rs.{g['net_pnl'].sum():,.0f}")

    print(f"\nPer signal_type:")
    for st, g in trades.groupby("signal_type"):
        p = pf_of(g["net_pnl"])
        w = 100*(g["net_pnl"]>0).mean()
        print(f"  {st}: n={len(g):,} PF={p:.3f} WR={w:.1f}% NET=Rs.{g['net_pnl'].sum():,.0f}")

    print(f"\nPer year:")
    trades["_yr"] = pd.to_datetime(trades["entry_ts"]).dt.year
    for y, g in trades.groupby("_yr"):
        p = pf_of(g["net_pnl"])
        print(f"  {y}: n={len(g):,} PF={p:.3f} NET=Rs.{g['net_pnl'].sum():,.0f}")

    # 2024_PF / 2023_PF saturation check (per brief §9)
    if 2023 in trades["_yr"].unique() and 2024 in trades["_yr"].unique():
        pf23 = pf_of(trades[trades["_yr"]==2023]["net_pnl"])
        pf24 = pf_of(trades[trades["_yr"]==2024]["net_pnl"])
        if pf23 > 0:
            ratio = pf24 / pf23
            print(f"\nSATURATION CHECK: 2024_PF / 2023_PF = {ratio:.3f}  "
                  f"(brief §9: ≤ 0.7 → RETIRE)")

    # Per cap_segment × signal_type
    print(f"\nPer cap_segment × signal_type:")
    trades["cap_segment"] = trades["symbol"].apply(lambda s: get_cap_segment("NSE:" + s) or "unknown")
    for (cap, st), g in trades.groupby(["cap_segment", "signal_type"]):
        if len(g) < 30:
            continue
        p = pf_of(g["net_pnl"])
        w = 100*(g["net_pnl"]>0).mean()
        print(f"  {cap:<12} {st:<14} n={len(g):>5,} PF={p:.3f} WR={w:.1f}% NET=Rs.{g['net_pnl'].sum():,.0f}")

    print(f"\nExit reasons:")
    for r, g in trades.groupby("exit_reason"):
        avg = g["net_pnl"].mean()
        print(f"  {r:<12} n={len(g):>5,} avg=Rs.{avg:,.0f}")

    # Pass-gates
    pass_pf = pf >= PF_FLOOR
    pass_n = n >= N_FLOOR
    pass_wr = abs(wr - 50.0) <= WR_BAND_PP
    pass_sh = sharpe > 0 if not np.isnan(sharpe) else False
    print(f"\nPASS-GATE: PF>={PF_FLOOR}={pass_pf}  n>={N_FLOOR}={pass_n}  "
          f"|WR-50|<={WR_BAND_PP}={pass_wr}  Sharpe>0={pass_sh}")


def main():
    print(f"=== sanity_nse_delivery_pct_anomaly  (period={_PERIOD}) ===")
    print(f"Window: {DISCOVERY_START} .. {DISCOVERY_END}")

    delivery = load_delivery_pct()
    delivery["symbol_with_prefix"] = delivery["symbol"]  # already bare
    daily = load_consolidated_daily()
    print(f"  daily rows: {len(daily):,}")

    events = compute_t_day_signals(delivery, daily)
    if events.empty:
        print("[NO EVENTS]")
        return

    print(f"\nSimulating {len(events):,} signal events ...")
    trades = run_period(events, daily)
    print(f"\nTotal trades: {len(trades):,}")

    if not trades.empty:
        out = _REPO / "reports" / "sub9_sanity" / f"nse_delivery_pct_anomaly_trades{_OUT_SUFFIX}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(out, index=False)
        print(f"Trades CSV: {out}")

    report(trades)


if __name__ == "__main__":
    main()
