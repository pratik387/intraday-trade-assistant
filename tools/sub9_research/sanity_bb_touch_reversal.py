"""Pure-math sanity check for `bb_touch_reversal` candidate.

NOT a §3.3 brief-gated test — pure mathematical-pattern probe on post-SEBI
Discovery window 2024-09-01 .. 2025-09-30. Let cell mining find which cells
have edge.

Mathematical pattern (15-minute Bollinger Bands):
  - Aggregate 5m bars to 15m on-the-fly (open=first, high=max, low=min,
    close=last, volume=sum).
  - Compute 20-period rolling SMA + stdev of 15m close.
  - Upper band = SMA + 2.0 * sigma; Lower band = SMA - 2.0 * sigma.
  - SHORT trigger: 15m bar HIGH >= upper band.
  - LONG trigger:  15m bar LOW  <= lower band.
  - Confirmation: NEXT 15m bar closes back inside both bands (rejecting
    the extreme).
  - Entry: rejection (confirmation) 15m bar's close. In 5m bars this is
    the 5m bar ending at the rejection-15m's close timestamp.

Entry / Trade mechanic (per task):
  - Entry: rejection bar's close (= 5m bar ending at confirmation 15m close)
  - Hard SL:
      SHORT -> max(prior 15m bar high, rejection 15m bar high) * 1.005
      LONG  -> min(prior 15m bar low,  rejection 15m bar low)  * 0.995
      min stop floor 0.5% of entry
  - R = |entry - hard_sl|
  - T1 (50% qty): middle band (SMA at rejection bar)
  - T2 (50% qty): opposite band (full reversion)
  - Time stop: 15:10 IST
  - BE trail: active_sl = entry_price after T1 fills
  - Latch: one fire per (symbol, session_date)
  - Bar-walk on UNDERLYING 5m bars (more granular SL/T1/T2 detection)

Universe:
  - Full NSE × MIS-enabled: nse_all.json with mis_leverage >= 1.0 AND
    cap_segment in {large_cap, mid_cap, small_cap}
  - Union with `assets/fno_liquid_200.csv` and `assets/ind_nifty*.csv`
    (these are subsets of nse_all eligible — included for completeness).
  - Intersected with 5m feather availability.

Regime-break pre-flight: MIS_leverage + STT_drag tags, Discovery window,
high severity, NON-raising (research probe).

Period:
  Discovery: 2024-09-01 .. 2025-09-30
  OOS:       2025-10-01 .. 2026-04-30  (only if Discovery passes/marginal)

Output:
  - reports/sub9_sanity/bb_touch_reversal_trades.csv

Usage:
    .venv/Scripts/python tools/sub9_research/sanity_bb_touch_reversal.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from services.regime_break_detector import check_window        # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# =====================================================================
# Locked params (pure-math probe — no parameter sweep here)
# =====================================================================

DISCOVERY_START = date(2024, 9, 1)
DISCOVERY_END   = date(2025, 9, 30)
OOS_START       = date(2025, 10, 1)
OOS_END         = date(2026, 4, 30)

# Bollinger Band on 15m close
BB_PERIOD = 20            # 20-period SMA
BB_STDEV_MULT = 2.0       # +/- 2.0 sigma

# Entry / trade mechanic
HARD_SL_BUFFER_MULT_SHORT = 1.005  # max(prior_high, cur_high) * 1.005
HARD_SL_BUFFER_MULT_LONG  = 0.995  # min(prior_low,  cur_low)  * 0.995
MIN_STOP_PCT = 0.5         # min stop 0.5% of entry
T1_PARTIAL_FRACTION = 0.5  # 50% qty at T1 (middle band)
T2_PARTIAL_FRACTION = 0.5  # 50% qty at T2 (opposite band)
USE_BREAKEVEN_TRAIL_AFTER_T1 = True
TIME_STOP_HHMM = "15:10"   # absolute hard time stop

# Universe filters
ALLOWED_CAP_SEGMENTS = {"large_cap", "mid_cap", "small_cap"}
MIN_MIS_LEVERAGE = 1.0

# Sizing
RISK_PER_TRADE_RUPEES = 1000

# Allowable trigger window (let cell mining decide best time-of-day band)
TRIGGER_WINDOW_START_HHMM = "09:30"
TRIGGER_WINDOW_END_HHMM   = "15:00"  # ensure confirmation + exit room before 15:10

# Cell mining thresholds (Gauntlet v2)
SHIP_MIN_N = 125
SHIP_PF = 1.30
SHIP_SHARPE = 0.5
SHIP_WIN_MO_PCT = 55.0
SHIP_TOP_MO_PCT = 40.0
SURVIVOR_MIN_N = 100
SURVIVOR_PF = 1.20
SURVIVOR_SHARPE = 0.0


# =====================================================================
# Universe construction
# =====================================================================

def load_nse_all_eligible() -> Dict[str, dict]:
    """Return {bare_symbol: {mis_leverage, cap_segment}} for MIS>=1 + L/M/S."""
    path = _REPO_ROOT / "nse_all.json"
    with path.open() as f:
        rows = json.load(f)
    out: Dict[str, dict] = {}
    for r in rows:
        raw = str(r["symbol"]).replace(".NS", "").split(":")[-1]
        mis = float(r.get("mis_leverage") or 0.0)
        cap = r.get("cap_segment", "unknown")
        if mis >= MIN_MIS_LEVERAGE and cap in ALLOWED_CAP_SEGMENTS:
            out[raw] = {"mis_leverage": mis, "cap_segment": cap}
    return out


def load_universe() -> Dict[str, dict]:
    """Combined eligible universe: nse_all + fno_liquid_200 + ind_nifty*.

    Returns {raw_symbol: {cap_segment, mis_leverage}}.
    """
    eligible = load_nse_all_eligible()

    # Add F&O 200 (intersect with eligibility — F&O symbols are typically
    # large/mid cap with MIS, so this is mostly redundant but safe).
    extras: set = set()
    fno_path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    if fno_path.exists():
        df = pd.read_csv(fno_path)
        for s in df["symbol"].astype(str):
            extras.add(s.replace("NSE:", ""))

    # ind_nifty*.csv (sector indices)
    for p in (_REPO_ROOT / "assets").glob("ind_nifty*.csv"):
        try:
            df = pd.read_csv(p)
            col = "Symbol" if "Symbol" in df.columns else df.columns[2]
            for s in df[col].dropna().astype(str):
                extras.add(s.replace("NSE:", ""))
        except Exception:
            continue

    # Cross-filter extras against nse_all metadata to enforce MIS / cap
    for raw in extras:
        if raw in eligible:
            continue
        # caller may want extras even if missing from nse_all — fetch cap via
        # services helper; default mis_leverage 1.0 (most NSE equities MIS 5x).
        cap = get_cap_segment("NSE:" + raw)
        if cap in ALLOWED_CAP_SEGMENTS:
            eligible[raw] = {"mis_leverage": 1.0, "cap_segment": cap}

    return eligible


# =====================================================================
# 5m feather loaders + 15m aggregation
# =====================================================================

def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_5m(start: date, end: date) -> pd.DataFrame:
    """Concat monthly 5m feathers covering [start, end]."""
    parts: List[pd.DataFrame] = []
    yyyy, mm = start.year, start.month
    print(f"  loading 5m feathers {yyyy:04d}-{mm:02d} .. {end.year:04d}-{end.month:02d} ...")
    while (yyyy, mm) <= (end.year, end.month):
        mdf = _load_5m_for_month(yyyy, mm)
        if not mdf.empty:
            parts.append(mdf)
        # next month
        if mm == 12:
            yyyy += 1; mm = 1
        else:
            mm += 1
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"])
    if big["date"].dt.tz is not None:
        big["date"] = big["date"].dt.tz_localize(None)
    # OOM-safe session_date: keep as datetime64 via dt.floor("D"), avoid
    # .dt.date which forced Python-object conversion + complex128 inference
    # (OOM on 23.87M rows at 364 MiB allocation).
    # Downstream code uses big["d"] for filtering — store as datetime64.
    big["d"] = big["date"].dt.floor("D")
    # Trim to exact window
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    big = big[(big["d"] >= start_ts) & (big["d"] <= end_ts)].copy()
    big = big.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"    total 5m bars in window: {len(big):,}")
    return big


def aggregate_to_15m(day5m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one (symbol, day) of 5m bars into 15m bars.

    15m boundary anchored to NSE session open 09:15. So 15m bars end at
    09:30, 09:45, 10:00, ... 15:30. We label each 15m bar by its END time
    (i.e. 09:30 bar = 09:15..09:30 5m bars combined). The 5m bar whose
    `date` field is the END time of a 15m bar is the "closing 5m" of that
    15m bar — its 5m timestamp == the 15m end timestamp.

    5m `date` convention: bar timestamp = bar END (NSE convention used
    throughout this codebase — verified vs vwap_deviation_meanrevert
    sanity which uses df['date'].dt.strftime as bar END HHMM).
    """
    if day5m.empty:
        return pd.DataFrame()
    s = day5m.sort_values("date").reset_index(drop=True)
    # 15m bin key: floor((minute_of_session - 0) / 15) groups consecutive 3
    # 5m bars together (09:15 open). With 5m bar END timestamps:
    #   bars ending 09:20, 09:25, 09:30 -> 15m ending 09:30
    #   bars ending 09:35, 09:40, 09:45 -> 15m ending 09:45
    # 15m_end_ts = ceil(5m_end_ts to next 15m boundary) using 09:30 anchor.
    # Simplest: compute minutes-since-09:15 of 5m bar end, then bin to 15m.
    ts = s["date"]
    minutes_from_open = (ts.dt.hour * 60 + ts.dt.minute) - (9 * 60 + 15)
    # 15m bin index (0 = 09:15-09:30, 1 = 09:30-09:45, ...). 5m bar END at
    # minutes_from_open of 5, 10, 15 belong to bin 0 (15m end = 09:30).
    # bin = ceil(minutes_from_open / 15) - 1, clamped.
    bin_idx = ((minutes_from_open + 14) // 15) - 1
    bin_idx = bin_idx.clip(lower=0)
    s["_bin"] = bin_idx.astype(int)

    agg = s.groupby("_bin", as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        end_ts=("date", "last"),       # END timestamp of the 15m bar (= last 5m's date)
        start_ts=("date", "first"),
    )
    return agg


# =====================================================================
# Trigger detection on 15m bars
# =====================================================================

def find_triggers_for_day(
    sym: str,
    sd: date,
    cap_segment: str,
    bars5m_day: pd.DataFrame,
    bars5m_prior_days: pd.DataFrame,
) -> List[dict]:
    """For one (symbol, day) build 15m bars (with BB warmup from prior days),
    detect BB-touch -> next-bar-rejection events, return one row per event.

    Returns rows with side, trigger_15m_end, confirmation_15m_end,
    trigger_high/low (for SL), middle_band (T1), opposite_band (T2),
    bb_width regime (narrow/mid/wide via daily-cell terciles).
    """
    # Build 15m for the current day + sufficient prior days to satisfy BB
    # warmup of 20 periods. Use combined prior-day 5m bars.
    if bars5m_day.empty:
        return []

    # Build 15m current day
    cur15 = aggregate_to_15m(bars5m_day)
    if cur15.empty:
        return []
    cur15["symbol"] = sym
    cur15["d"] = sd

    # Build 15m for prior days; we'll concat to provide warmup.
    if not bars5m_prior_days.empty:
        prior_parts: List[pd.DataFrame] = []
        for pd_d, g in bars5m_prior_days.groupby(bars5m_prior_days["date"].dt.date):
            agg = aggregate_to_15m(g)
            if not agg.empty:
                agg["symbol"] = sym
                agg["d"] = pd_d
                prior_parts.append(agg)
        prior15 = pd.concat(prior_parts, ignore_index=True) if prior_parts else pd.DataFrame()
    else:
        prior15 = pd.DataFrame()

    combo = pd.concat([prior15, cur15], ignore_index=True) if not prior15.empty else cur15.copy()
    combo = combo.sort_values("end_ts").reset_index(drop=True)

    # Rolling 20-period SMA + stdev of close (population stdev via ddof=0 is
    # the textbook BB definition; pandas default is sample stdev (ddof=1).
    # We use ddof=0 to match canonical Bollinger Band formula.)
    combo["sma20"] = combo["close"].rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    combo["std20"] = combo["close"].rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    combo["upper"] = combo["sma20"] + BB_STDEV_MULT * combo["std20"]
    combo["lower"] = combo["sma20"] - BB_STDEV_MULT * combo["std20"]

    # Limit to current day bars (we have warmup from prior days)
    today = combo[combo["d"] == sd].reset_index(drop=True)
    if today.empty or today["sma20"].isna().all():
        return []

    # Need prior 15m bar (for SL calc) — keep index of full combo
    today_idx_in_combo = combo.index[combo["d"] == sd].tolist()

    rows: List[dict] = []
    for local_i, ci in enumerate(today_idx_in_combo):
        # Trigger bar = combo.iloc[ci]
        if ci + 1 >= len(combo):
            continue  # need confirmation bar
        trig = combo.iloc[ci]
        conf = combo.iloc[ci + 1]

        # Need fully-warmed BB on trigger bar
        if pd.isna(trig["sma20"]) or pd.isna(trig["upper"]) or pd.isna(trig["lower"]):
            continue
        # Confirmation bar must also be SAME day (don't cross day boundary)
        if pd.Timestamp(trig["end_ts"]).date() != sd or pd.Timestamp(conf["end_ts"]).date() != sd:
            continue

        # Trigger-window filter (15m end time)
        hhmm = pd.Timestamp(trig["end_ts"]).strftime("%H:%M")
        if not (TRIGGER_WINDOW_START_HHMM <= hhmm <= TRIGGER_WINDOW_END_HHMM):
            continue

        trig_high = float(trig["high"]); trig_low = float(trig["low"])
        upper = float(trig["upper"]); lower = float(trig["lower"]); sma = float(trig["sma20"])
        std = float(trig["std20"])

        side = None
        if trig_high >= upper:
            side = "SHORT"
        elif trig_low <= lower:
            side = "LONG"
        else:
            continue

        # Confirmation: next 15m closes BACK INSIDE bands.
        # Use the confirmation bar's CURRENT bands (next bar SMA/upper/lower).
        c_close = float(conf["close"])
        c_upper = float(conf["upper"]) if not pd.isna(conf["upper"]) else upper
        c_lower = float(conf["lower"]) if not pd.isna(conf["lower"]) else lower
        if not (c_lower < c_close < c_upper):
            continue

        # Prior 15m bar for SL
        if ci - 1 < 0:
            prior = None
        else:
            p = combo.iloc[ci - 1]
            # Use prior bar even if it's on prior session (it's the
            # immediately preceding 15m, still valid for structural SL).
            prior = p

        if side == "SHORT":
            prior_high = float(prior["high"]) if prior is not None else trig_high
            sl_struct = max(trig_high, prior_high) * HARD_SL_BUFFER_MULT_SHORT
        else:
            prior_low = float(prior["low"]) if prior is not None else trig_low
            sl_struct = min(trig_low, prior_low) * HARD_SL_BUFFER_MULT_LONG

        rows.append({
            "symbol": sym,
            "session_date": sd,
            "cap_segment": cap_segment,
            "side": side,
            "trigger_15m_end": pd.Timestamp(trig["end_ts"]),
            "confirmation_15m_end": pd.Timestamp(conf["end_ts"]),
            "trigger_high": trig_high,
            "trigger_low": trig_low,
            "sl_struct": sl_struct,           # before min-stop floor
            "middle_band": sma,               # T1
            "opposite_band": lower if side == "SHORT" else upper,  # T2
            "upper_band": upper,
            "lower_band": lower,
            "bb_std": std,
            "confirmation_close": c_close,
        })

    return rows


def detect_all_triggers(
    big5m: pd.DataFrame,
    universe_map: Dict[str, dict],
) -> pd.DataFrame:
    """Iterate symbols/days; produce one row per (sym, day) trigger event.

    Latch is applied AFTER triggers are detected — first event per
    (symbol, session_date) regardless of side.
    """
    print("  filtering 5m to universe + ALLOWED_CAP_SEGMENTS ...")
    df = big5m[big5m["symbol"].isin(universe_map.keys())].copy()
    print(f"    universe-filtered 5m bars: {len(df):,}")

    # group by symbol for fast per-symbol processing
    out_rows: List[dict] = []
    n_sym = 0
    n_events = 0
    for sym, sym_df in df.groupby("symbol", sort=False):
        n_sym += 1
        meta = universe_map.get(sym)
        if meta is None:
            continue
        cap = meta["cap_segment"]
        # process day by day; provide prior 1-2 trading days for BB warmup
        sym_df = sym_df.sort_values("date").reset_index(drop=True)
        days = sorted(sym_df["d"].unique())
        # index by day for quick slicing
        for di, sd in enumerate(days):
            day_bars = sym_df[sym_df["d"] == sd]
            if len(day_bars) < 6:  # at least 1 full 15m bar (3 5m) + room
                continue
            # 2 prior trading days for BB warmup (20 periods × 15m ≈ 5 hours
            # = 1.3 trading days; 2 prior days is plenty).
            prior_days = days[max(0, di - 2):di]
            prior_bars = sym_df[sym_df["d"].isin(prior_days)] if prior_days else pd.DataFrame()
            rows = find_triggers_for_day(sym, sd, cap, day_bars, prior_bars)
            out_rows.extend(rows)
            n_events += len(rows)
        if n_sym % 200 == 0:
            print(f"    progress: scanned {n_sym} symbols, {n_events} triggers so far")
    print(f"    final: {n_sym} symbols, {n_events} raw trigger events")
    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)


# =====================================================================
# Trade simulation: bar-walk on 5m bars
# =====================================================================

def simulate(
    triggers: pd.DataFrame,
    big5m: pd.DataFrame,
) -> pd.DataFrame:
    if triggers.empty:
        return pd.DataFrame()

    print(f"\n  simulating {len(triggers):,} triggers (post-latch) ...")
    # Latch: one fire per (symbol, session_date), keep earliest by confirmation_15m_end
    triggers = triggers.sort_values(["symbol", "session_date", "confirmation_15m_end"])
    triggers = triggers.drop_duplicates(subset=["symbol", "session_date"], keep="first")
    print(f"    after (symbol, day) latch: {len(triggers):,}")

    # Build {(symbol, day): 5m_df_for_day}
    big5m_by_sym = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    trades: List[dict] = []
    n_no_entry = n_traded = 0

    for _, t in triggers.iterrows():
        sym = t["symbol"]; sd = t["session_date"]; side = t["side"]
        conf_end = pd.Timestamp(t["confirmation_15m_end"])
        sma = float(t["middle_band"])
        opp = float(t["opposite_band"])
        sl_struct = float(t["sl_struct"])

        sym_df = big5m_by_sym.get(sym)
        if sym_df is None:
            n_no_entry += 1; continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty:
            n_no_entry += 1; continue

        # Entry 5m bar = the 5m bar whose `date` == confirmation_15m_end
        entry_mask = day_df["date"] == conf_end
        if not entry_mask.any():
            n_no_entry += 1; continue
        entry_idx = int(day_df.index[entry_mask][0])
        entry_bar = day_df.iloc[entry_idx]
        entry_price = float(entry_bar["close"])  # rejection 15m's close = 5m bar's close

        # Apply min-stop floor (0.5% of entry)
        if side == "SHORT":
            sl_floor = entry_price * (1.0 + MIN_STOP_PCT / 100.0)
            hard_sl = max(sl_struct, sl_floor)
            stop_distance = hard_sl - entry_price
        else:
            sl_floor = entry_price * (1.0 - MIN_STOP_PCT / 100.0)
            hard_sl = min(sl_struct, sl_floor)
            stop_distance = entry_price - hard_sl

        if stop_distance <= 0:
            n_no_entry += 1; continue

        # T1 = middle band; T2 = opposite band
        t1_target = sma
        t2_target = opp

        # For LONG, T1 > entry, T2 > entry; for SHORT, T1 < entry, T2 < entry.
        # Sanity-check: T1 must be in the right direction.
        if side == "SHORT" and (t1_target >= entry_price or t2_target >= entry_price):
            # Sometimes mean reversion already happened (rejection bar over-
            # reverted). Skip — math says no edge.
            n_no_entry += 1; continue
        if side == "LONG" and (t1_target <= entry_price or t2_target <= entry_price):
            n_no_entry += 1; continue

        # Bar walk on 5m bars after entry bar, up to 15:10 IST
        forward = day_df.iloc[entry_idx + 1:].reset_index(drop=True)
        forward = forward[forward["date"].dt.strftime("%H:%M") <= TIME_STOP_HHMM]
        if forward.empty:
            n_no_entry += 1; continue

        exit_ts = None; exit_price = None; exit_reason = None
        hit_t1 = False; t1_exit_price = None; t1_exit_ts = None

        for _, bar in forward.iterrows():
            ts = bar["date"]
            high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])
            active_sl = (entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl)

            if side == "SHORT":
                if high >= active_sl:
                    exit_ts = ts; exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and low <= t1_target:
                    hit_t1 = True; t1_exit_price = t1_target; t1_exit_ts = ts
                if hit_t1 and low <= t2_target:
                    exit_ts = ts; exit_price = t2_target; exit_reason = "t2"
                    break
            else:
                if low <= active_sl:
                    exit_ts = ts; exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and high >= t1_target:
                    hit_t1 = True; t1_exit_price = t1_target; t1_exit_ts = ts
                if hit_t1 and high >= t2_target:
                    exit_ts = ts; exit_price = t2_target; exit_reason = "t2"
                    break

            if ts.strftime("%H:%M") >= TIME_STOP_HHMM:
                exit_ts = ts; exit_price = close; exit_reason = "time_stop_hard"
                break

        if exit_price is None:
            # walked to end of forward window
            last = forward.iloc[-1]
            exit_ts = last["date"]; exit_price = float(last["close"])
            exit_reason = "time_stop_bars"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

        if hit_t1:
            qty_t1 = max(int(qty * T1_PARTIAL_FRACTION), 1)
            qty_t2 = max(qty - qty_t1, 0)
            if side == "SHORT":
                pnl_t1 = (entry_price - t1_exit_price) * qty_t1
                pnl_t2 = (entry_price - exit_price) * qty_t2 if qty_t2 > 0 else 0.0
            else:
                pnl_t1 = (t1_exit_price - entry_price) * qty_t1
                pnl_t2 = (exit_price - entry_price) * qty_t2 if qty_t2 > 0 else 0.0
            realized_pnl = pnl_t1 + pnl_t2
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL" if side == "SHORT" else "BUY")
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, "SELL" if side == "SHORT" else "BUY") if qty_t2 > 0 else 0.0
            fee = fee_t1 + fee_t2
            blended_exit = (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1) if qty_t2 > 0 else t1_exit_price
        else:
            if side == "SHORT":
                realized_pnl = (entry_price - exit_price) * qty
            else:
                realized_pnl = (exit_price - entry_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL" if side == "SHORT" else "BUY")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "T1_entry_date": sd,
            "symbol": "NSE:" + sym,
            "cap_segment": t["cap_segment"],
            "side": side,
            "trigger_15m_end": pd.Timestamp(t["trigger_15m_end"]),
            "confirmation_15m_end": conf_end,
            "entry_ts": entry_bar["date"],
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "t2_target": t2_target,
            "bb_std": float(t["bb_std"]),
            "upper_band": float(t["upper_band"]),
            "lower_band": float(t["lower_band"]),
            "middle_band": sma,
            "hit_t1": hit_t1,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "stop_distance": stop_distance,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
        n_traded += 1

    print(f"    no entry / invalid:    {n_no_entry}")
    print(f"    traded:                {n_traded}")
    return pd.DataFrame(trades)


# =====================================================================
# Reporting + cell mining
# =====================================================================

def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return g / l if l > 0 else float("inf")


def _sharpe(daily: pd.Series) -> float:
    if daily.empty or daily.std() == 0 or daily.size < 2:
        return 0.0
    return float(daily.mean() / daily.std())


def add_cells(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades.copy()
    # Time-of-day on entry_ts
    ts = pd.to_datetime(df["entry_ts"])
    hm = ts.dt.hour * 60 + ts.dt.minute
    def _tod_bucket(m):
        if m < 11 * 60 + 30:
            return "09:30-11:30"
        if m < 13 * 60 + 30:
            return "11:30-13:30"
        return "13:30-15:10"
    df["tod_bucket"] = hm.apply(_tod_bucket)

    # BB-width regime via daily session-wide tercile of bb_std across ALL
    # trades (research probe — let cell mining tell us which works).
    if len(df) >= 30:
        q1 = df["bb_std"].quantile(1.0 / 3.0)
        q2 = df["bb_std"].quantile(2.0 / 3.0)
        def _w(v):
            if v <= q1: return "narrow"
            if v <= q2: return "mid"
            return "wide"
        df["bb_width_regime"] = df["bb_std"].apply(_w)
    else:
        df["bb_width_regime"] = "mid"

    df["_month"] = pd.to_datetime(df["T1_entry_date"]).dt.strftime("%Y-%m")
    df["_session_date"] = pd.to_datetime(df["T1_entry_date"]).dt.date
    return df


def cell_mine(trades: pd.DataFrame, dims: List[str], pnl_col: str = "net_pnl",
              max_k: int = 3) -> pd.DataFrame:
    rows = []
    for k in range(1, max_k + 1):
        for combo in combinations(dims, k):
            sub = trades.dropna(subset=list(combo) + [pnl_col])
            if sub.empty:
                continue
            for cell_vals, cell_sub in sub.groupby(list(combo), observed=True):
                pnl = cell_sub[pnl_col]
                n = int(len(pnl))
                pf = _pf(pnl)
                wr = 100.0 * float((pnl > 0).mean()) if n > 0 else 0.0
                net = float(pnl.sum())
                daily = cell_sub.groupby("_session_date")[pnl_col].sum()
                sharpe = _sharpe(daily)
                monthly = cell_sub.groupby("_month")[pnl_col].sum()
                n_mo = int(monthly.size)
                win_mo_pct = 100.0 * float((monthly > 0).mean()) if n_mo else 0.0
                total_abs = abs(net)
                top_mo_pct = (100.0 * float(monthly.abs().max()) / total_abs) if total_abs > 1e-6 else 0.0

                if not isinstance(cell_vals, tuple):
                    cell_vals = (cell_vals,)
                cell_label = " | ".join(f"{c}={v}" for c, v in zip(combo, cell_vals))
                rows.append({
                    "dims": ",".join(combo), "k": k, "cell": cell_label,
                    "n": n, "pf": float(pf), "wr": wr, "net": net,
                    "sharpe": sharpe, "n_months": n_mo,
                    "win_mo_pct": round(win_mo_pct, 1),
                    "top_mo_pct": round(top_mo_pct, 1),
                })
    return pd.DataFrame(rows)


def report(trades: pd.DataFrame) -> Dict[str, int]:
    """Print full report + return summary dict for return-message."""
    summary: Dict[str, int] = {
        "n": 0, "ship_eligible": 0, "survivors": 0,
    }
    if trades.empty:
        print("\n[NO TRADES]")
        return summary
    df = add_cells(trades)
    npnl = df["net_pnl"]
    n = len(df)
    pf = _pf(npnl)
    wr = float((npnl > 0).mean()) * 100
    daily = df.groupby("_session_date")["net_pnl"].sum()
    sharpe = _sharpe(daily)
    monthly = df.groupby("_month")["net_pnl"].sum()
    n_mo = int(monthly.size)
    win_mo_pct = 100.0 * float((monthly > 0).mean()) if n_mo else 0.0
    total_abs = abs(float(npnl.sum()))
    top_mo_pct = (100.0 * float(monthly.abs().max()) / total_abs) if total_abs > 1e-6 else 0.0

    print("\n" + "=" * 72)
    print("bb_touch_reversal -- pure-math sanity check")
    print("=" * 72)
    print(f"Period: {df['T1_entry_date'].min()} .. {df['T1_entry_date'].max()}")
    print(f"\nAGGREGATE:")
    print(f"  n = {n:,}")
    print(f"  WR = {wr:.1f}%")
    print(f"  Gross PnL: Rs.{int(df['realized_pnl'].sum()):,}")
    print(f"  Fees:      Rs.{int(df['fee'].sum()):,}")
    print(f"  NET PnL:   Rs.{int(npnl.sum()):,}")
    print(f"  NET PF:    {pf:.3f}")
    print(f"  Sharpe (daily): {sharpe:.3f}")
    print(f"  Monthly: n_mo={n_mo} win_mo={win_mo_pct:.1f}% top_mo={top_mo_pct:.1f}%")

    print("\nPer side:")
    for sd, grp in df.groupby("side"):
        pnl = grp["net_pnl"]
        print(f"  {sd:<6} n={len(grp):>5} PF={_pf(pnl):.3f} WR={float((pnl>0).mean())*100:.1f}% NET=Rs.{int(pnl.sum()):>11,}")

    print("\nPer cap_segment:")
    for cap, grp in df.groupby("cap_segment"):
        pnl = grp["net_pnl"]
        print(f"  {cap:<12} n={len(grp):>5} PF={_pf(pnl):.3f} WR={float((pnl>0).mean())*100:.1f}% NET=Rs.{int(pnl.sum()):>11,}")

    print("\nPer time-of-day:")
    for tod, grp in df.groupby("tod_bucket"):
        pnl = grp["net_pnl"]
        print(f"  {tod:<12} n={len(grp):>5} PF={_pf(pnl):.3f} WR={float((pnl>0).mean())*100:.1f}% NET=Rs.{int(pnl.sum()):>11,}")

    print("\nPer bb_width_regime:")
    for w, grp in df.groupby("bb_width_regime"):
        pnl = grp["net_pnl"]
        print(f"  {w:<6} n={len(grp):>5} PF={_pf(pnl):.3f} WR={float((pnl>0).mean())*100:.1f}% NET=Rs.{int(pnl.sum()):>11,}")

    print("\nMonthly NET:")
    for m, v in monthly.items():
        print(f"  {m}  Rs.{int(v):>10,}")

    print("\nExit reasons:")
    for r, grp in df.groupby("exit_reason"):
        print(f"  {r:<22} n={len(grp):>5} avg_net=Rs.{int(grp['net_pnl'].mean()):>6,}")

    # ---- Cell mining ----
    dims = ["side", "cap_segment", "tod_bucket", "bb_width_regime"]
    print("\n" + "-" * 72)
    print("CELL MINING (1D / 2D / 3D combinations)")
    print("-" * 72)
    cells = cell_mine(df, dims, "net_pnl", max_k=3)
    if cells.empty:
        print("  no cells produced")
        summary["n"] = n
        return summary

    # Ship-eligible
    ship_mask = (
        (cells["n"] >= SHIP_MIN_N)
        & (cells["pf"] >= SHIP_PF)
        & (cells["sharpe"] >= SHIP_SHARPE)
        & (cells["win_mo_pct"] >= SHIP_WIN_MO_PCT)
        & (cells["top_mo_pct"] < SHIP_TOP_MO_PCT)
    )
    survivor_mask = (
        (cells["n"] >= SURVIVOR_MIN_N)
        & (cells["pf"] >= SURVIVOR_PF)
        & (cells["sharpe"] >= SURVIVOR_SHARPE)
    )
    ship = cells[ship_mask].sort_values(["pf", "n"], ascending=[False, False])
    survivors = cells[survivor_mask].sort_values(["pf", "n"], ascending=[False, False])

    print(f"\nSURVIVORS (n>={SURVIVOR_MIN_N}, PF>={SURVIVOR_PF}, Sharpe>={SURVIVOR_SHARPE}): {len(survivors)}")
    for _, r in survivors.head(20).iterrows():
        print(f"  [{r['dims']}] {r['cell']}  n={r['n']} PF={r['pf']:.3f} WR={r['wr']:.1f}% "
              f"Sh={r['sharpe']:.2f} win_mo={r['win_mo_pct']:.1f}% top_mo={r['top_mo_pct']:.1f}% "
              f"NET=Rs.{r['net']:,.0f}")

    print(f"\nSHIP-ELIGIBLE (gauntlet-v2: n>={SHIP_MIN_N}, PF>={SHIP_PF}, Sh>={SHIP_SHARPE}, "
          f"win_mo>={SHIP_WIN_MO_PCT}%, top_mo<{SHIP_TOP_MO_PCT}%): {len(ship)}")
    for _, r in ship.head(20).iterrows():
        print(f"  [{r['dims']}] {r['cell']}  n={r['n']} PF={r['pf']:.3f} Sh={r['sharpe']:.2f} "
              f"win_mo={r['win_mo_pct']:.1f}% top_mo={r['top_mo_pct']:.1f}% NET=Rs.{r['net']:,.0f}")

    # ---- Verdict ----
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"Aggregate PF = {pf:.3f}, n = {n:,}")
    if pf >= SHIP_PF and n >= SHIP_MIN_N and sharpe >= SHIP_SHARPE:
        print(f"  AGGREGATE passes ship gate (PF>={SHIP_PF}, n>={SHIP_MIN_N}, Sh>={SHIP_SHARPE}).")
        ship_via = "aggregate"
    elif pf >= SURVIVOR_PF and n >= SURVIVOR_MIN_N:
        print(f"  AGGREGATE marginal (PF>={SURVIVOR_PF}, n>={SURVIVOR_MIN_N}). Check cells.")
        ship_via = "aggregate-marginal"
    else:
        print(f"  AGGREGATE fails ship/survivor gates.")
        ship_via = "aggregate-fail"

    if len(ship) > 0:
        print(f"  CELLS: {len(ship)} ship-eligible cell(s) found.")
        oos_decision = "PROCEED to OOS"
    elif len(survivors) > 0:
        print(f"  CELLS: 0 ship-eligible but {len(survivors)} survivor(s). MARGINAL.")
        oos_decision = "MARGINAL — OOS optional for survivor exploration"
    else:
        print(f"  CELLS: 0 ship-eligible, 0 survivors.")
        oos_decision = "RETIRE — no OOS test warranted"
    print(f"  OOS decision: {oos_decision}")

    summary.update({"n": n, "ship_eligible": int(len(ship)), "survivors": int(len(survivors))})
    return summary


# =====================================================================
# Main
# =====================================================================

def main():
    # Force UTF-8 stdout on Windows (rule_changes.csv contains "→")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 72)
    print("bb_touch_reversal -- pure-math sanity probe")
    print("=" * 72)

    # ---- Pre-flight regime check (non-raising research mode) ----
    print("\nRegime-break pre-flight (Discovery 2024-09-01..2025-09-30):")
    hits = check_window(
        "bb_touch_reversal", ["MIS_leverage", "STT_drag"],
        "Discovery", DISCOVERY_START, DISCOVERY_END,
        min_severity="high", raise_on_break=False,
    )
    if hits:
        print(f"  WARNING: {len(hits)} rule change(s) in window:")
        for r in hits:
            print(f"    {r.effective_date} [{r.severity.upper()}] {r.description}")
        print("  Proceeding as research probe (raise_on_break=False).")
    else:
        print("  no high+ rule changes")

    # ---- Universe ----
    print("\nUniverse:")
    univ_map = load_universe()
    print(f"  eligible universe (mis>=1 + L/M/S, post-dedup): {len(univ_map)}")

    # ---- Load 5m bars ----
    print("\nLoading 5m feathers (Discovery):")
    big5m = build_5m(DISCOVERY_START, DISCOVERY_END)
    if big5m.empty:
        print("  no 5m data in window — aborting.")
        return
    syms_with_data = set(big5m["symbol"].unique().tolist())
    eff_universe = {s: m for s, m in univ_map.items() if s in syms_with_data}
    print(f"  universe intersected with 5m data:  {len(eff_universe)}")

    # ---- Detect triggers ----
    print("\nDetecting 15m BB-touch -> rejection events:")
    triggers = detect_all_triggers(big5m, eff_universe)
    if triggers.empty:
        print("  no triggers produced — aborting.")
        return
    print(f"  total trigger events (pre-latch): {len(triggers):,}")

    # ---- Simulate ----
    print("\nSimulating trades (5m bar walk):")
    trades = simulate(triggers, big5m)

    # ---- Write CSV ----
    out = _REPO_ROOT / "reports" / "sub9_sanity" / "bb_touch_reversal_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nWrote {len(trades):,} trades -> {out}")

    # ---- Report ----
    report(trades)


if __name__ == "__main__":
    main()
