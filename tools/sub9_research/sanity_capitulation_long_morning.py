"""Pre-coding sanity check for capitulation_long_morning candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-07-sub-project-9-brief-
capitulation_long_morning.md): BEFORE writing detector code, simulate
the morning gap-down panic-capitulation LONG fade on 24 months of
2023-2024 5m enriched feathers across the broad NSE equity universe.

Decision criteria (from brief §9):
  - HARD n>=500 floor — n<500 -> STRUCTURAL RETIRE (not just PF retire)
  - NET PF >= 1.10 -> proceed
  - PF < 1.10 -> THESIS RETIRE
  - No-news filter must contribute >= 0.10 absolute PF lift vs unfiltered

Mechanic (mirror of gap_fade_short, sign-flipped):
  - T+0 09:15 5m bar: gap_pct = (open_09:15 - pdc) / pdc * 100 in [-8, -1.5]
  - No-news filter: no earnings event for symbol within +/-2 trading days
    (earnings_calendar/earnings_events.parquet covers ~117 F&O symbols;
    for symbols outside coverage, treat as "unknown news" — INCLUDE with
    flag for transparency, since exclusion would gut the universe).
  - Exhaustion candle (one of 09:15 / 09:20 / 09:25 5m bars):
      lower_wick / body >= 0.5 AND body_size_pct <= 30% AND green
      (close > open). Mirror of gap_fade_short upper-wick + body filters.
  - Volume decline filter (if not opening bar): vol(current) < vol(09:15).
  - Entry: NEXT bar's open (Streak/AlgoTest signal=trade-1 convention,
    matches sanity_volume_spike_exhaustion_reversal Q1 decision).
  - Stop (LONG): stop_a = gap_low * (1 - cap_buf); cap_buf = 0.005 micro
    else 0.0025; stop_b = entry - ATR*1.5 (per brief).
    hard_sl = min(stop_a, stop_b). min_stop_distance_pct = 0.3%.
  - T1 (50% qty): 50% gap fill = (entry + pdc) / 2.
  - T2 (50% qty): full PDC.
  - Time stop: 10:15 IST hard exit (mirror of gap_fade_short).
  - Breakeven trail after T1 fills (Tradejini/Share India retail-pro
    convention; matches volume_spike_exhaustion_reversal Q8 decision).
  - Latch: one fire per (symbol, day, side=LONG).

Universe: data-broad (all NSE liquid stocks). No cap-segment pre-lock,
no F&O 200 pre-lock — cell selection is the gauntlet's job (Stage 3).
Liquidity gate: 20-day ADV * close >= Rs 2 Cr (data-quality only).

Usage:
    python tools/sub9_research/sanity_capitulation_long_morning.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment              # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee     # noqa: E402


# ---- Locked params (per brief §6 — direct mirror of gap_fade_short) ----
# Discovery period
DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END   = date(2024, 12, 31)

# Gap thresholds — mirror of gap_fade_short.min_gap_pct_above_pdc=1.5,
# max=8.0 (sign-flipped for gap-down).
MIN_GAP_DOWN_PCT = 1.5    # |gap| >= 1.5%
MAX_GAP_DOWN_PCT = 8.0    # |gap| <= 8.0%

# Exhaustion-candle params — direct mirror of gap_fade_short
MIN_LOWER_WICK_RATIO = 0.5     # lower_wick / body >= 0.5
MAX_BODY_SIZE_PCT = 30.0       # body_size_pct <= 30%

# Active window: 09:15 / 09:20 / 09:25 (entry-bar at idx+1 of confirmation)
ACTIVE_WINDOW_HHMM = ["09:15", "09:20", "09:25"]
TIME_STOP_HHMM = "10:15"        # mirror of gap_fade_short.time_stop_at

# Stop buffer (cap-aware) — per brief §6 step 6
STOP_BUF_MICRO = 0.005           # 0.5% below gap_low for micro_cap
STOP_BUF_OTHER = 0.0025          # 0.25% below gap_low otherwise
ATR_STOP_MULT = 1.5              # entry - 1.5*ATR
MIN_STOP_PCT = 0.3               # 0.3% min stop distance (% of entry)

# T1/T2 = structural targets (per brief §6 step 7)
# T1 = 50% gap fill = (entry + pdc) / 2
# T2 = full PDC
T1_PARTIAL_QTY_PCT = 0.5
USE_BREAKEVEN_TRAIL_AFTER_T1 = True

# No-news filter
EARNINGS_BLACKOUT_DAYS = 2       # +/-2 calendar days around announce_date
NO_NEWS_LOOKBACK_DAYS = 5        # 5-day lookback (used if announcement feed available)

# Volume decline filter
REQUIRE_VOL_DECLINE = True

# Liquidity gate
MIN_ADV_INR_CR = 2.0             # 20-day ADV * close >= Rs 2 Cr (data-quality)

RISK_PER_TRADE_RUPEES = 1000


_NEEDED_5M_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_feather(path, columns=_NEEDED_5M_COLS)
    # Memory: shrink dtypes (float64 -> float32, int64 -> int32 for volume)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("int64")  # keep int64 - some volumes large
    return df


def build_full_period_5m() -> pd.DataFrame:
    """Concat 24 months of 5m bars. Skips global sort to avoid OOM.

    Downstream code groups by (symbol, d) and sorts inside groups, so a
    global sort is not required. Memory budget: ~2.8GB for the unsorted
    DataFrame; the prior global-sort path allocated another 2.8GB on top
    which OOM-ed on a 16GB machine.
    """
    print("  loading 24 monthly 5m feathers (2023-01 .. 2024-12) ...")
    parts: List[pd.DataFrame] = []
    for yyyy in (2023, 2024):
        for m in range(1, 13):
            mdf = _load_5m_for_month(yyyy, m)
            if not mdf.empty:
                parts.append(mdf)
    big = pd.concat(parts, ignore_index=True, copy=False)
    big["d"] = big["date"].dt.date
    print(f"  total 5m bars: {len(big):,}")
    print(f"  unique symbols: {big['symbol'].nunique():,}")
    return big


def load_consolidated_daily() -> pd.DataFrame:
    """Load 1day OHLCV from consolidated_daily.feather — used for PDC and ADV."""
    print("  loading consolidated_daily.feather ...")
    path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    df = pd.read_feather(path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    # Need history back to ~late 2022 for 20-day ADV warmup at 2023-01-01
    df = df[(df["d"] >= date(2022, 11, 1)) & (df["d"] <= DISCOVERY_END)].copy()
    df = df[["symbol", "d", "close", "volume"]].copy()
    df["traded_value"] = df["close"] * df["volume"]
    df = df.sort_values(["symbol", "d"]).reset_index(drop=True)
    df["adv_20d_cr"] = df.groupby("symbol")["traded_value"].transform(
        lambda v: v.shift(1).rolling(20).mean()
    ) / 1e7
    df["pdc"] = df.groupby("symbol")["close"].shift(1)
    print(f"  daily rows: {len(df):,} | symbols: {df['symbol'].nunique():,}")
    return df[["symbol", "d", "close", "pdc", "adv_20d_cr"]]


def load_earnings_blackout() -> Set[Tuple[str, date]]:
    """Build set of (bare_symbol, blackout_date) pairs.

    For each earnings event, blackout +/-2 calendar days around the
    announce_date. Returns set of (bare_symbol, date) tuples to exclude.

    Coverage: ~117 F&O symbols per the parquet. Symbols OUTSIDE coverage
    are treated as "unknown news" — INCLUDED in the sanity sample with a
    coverage flag (so we don't gut the universe). Brief §11 explicitly
    accepts this proxy at the sanity stage.
    """
    print("  loading earnings_calendar/earnings_events.parquet ...")
    path = _REPO_ROOT / "data" / "earnings_calendar" / "earnings_events.parquet"
    df = pd.read_parquet(path)
    df["bare_symbol"] = df["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    df["announce_date"] = pd.to_datetime(df["announce_date"]).dt.date
    df = df.dropna(subset=["announce_date"])
    print(f"  earnings events: {len(df)} rows | covered symbols: {df['bare_symbol'].nunique()}")

    blackout: Set[Tuple[str, date]] = set()
    for _, r in df.iterrows():
        sym = r["bare_symbol"]
        d0 = r["announce_date"]
        for offset in range(-EARNINGS_BLACKOUT_DAYS, EARNINGS_BLACKOUT_DAYS + 1):
            blackout.add((sym, d0 + timedelta(days=offset)))
    covered_syms = set(df["bare_symbol"].unique())
    print(f"  earnings blackout pairs (+/-{EARNINGS_BLACKOUT_DAYS}d): {len(blackout):,}")
    return blackout, covered_syms


def find_gap_events(big5m: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Find all (symbol, d) pairs with gap-down >= MIN_GAP_DOWN_PCT.

    Also returns gap-UP pairs (for mutual-exclusivity overlap check).
    """
    print("  computing 09:15 open per (symbol, day) and joining PDC ...")
    open_bars = big5m[big5m["date"].dt.strftime("%H:%M") == "09:15"].copy()
    open_bars = open_bars[["symbol", "d", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "open_09_15",
            "high": "high_09_15",
            "low":  "low_09_15",
            "close": "close_09_15",
            "volume": "vol_09_15",
        }
    )

    daily_idx = daily.set_index(["symbol", "d"])[["pdc", "adv_20d_cr"]]
    open_bars = open_bars.merge(daily_idx, left_on=["symbol", "d"], right_index=True, how="left")
    open_bars = open_bars.dropna(subset=["pdc", "adv_20d_cr"])
    open_bars = open_bars[open_bars["pdc"] > 0]
    open_bars["gap_pct"] = (open_bars["open_09_15"] - open_bars["pdc"]) / open_bars["pdc"] * 100.0
    print(f"  symbol-days with PDC + ADV: {len(open_bars):,}")

    # Liquidity gate
    pre_liq = len(open_bars)
    open_bars = open_bars[open_bars["adv_20d_cr"] >= MIN_ADV_INR_CR].copy()
    print(f"  after ADV >= Rs {MIN_ADV_INR_CR}Cr: {len(open_bars):,} (dropped {pre_liq - len(open_bars):,})")

    # Cap segment annotation (NOT a pre-filter — broad universe)
    open_bars["nse_symbol"] = "NSE:" + open_bars["symbol"].astype(str)
    open_bars["cap_segment"] = open_bars["nse_symbol"].apply(get_cap_segment)
    cap_counts = open_bars["cap_segment"].value_counts().to_dict()
    print(f"  cap_segment distribution (no pre-filter): {cap_counts}")

    # Split into gap-down and gap-up populations for overlap diagnostic
    gap_down = open_bars[
        (open_bars["gap_pct"] <= -MIN_GAP_DOWN_PCT)
        & (open_bars["gap_pct"] >= -MAX_GAP_DOWN_PCT)
    ].copy()
    gap_up = open_bars[
        (open_bars["gap_pct"] >= MIN_GAP_DOWN_PCT)
        & (open_bars["gap_pct"] <= MAX_GAP_DOWN_PCT)
    ].copy()
    print(f"  gap-DOWN events ({-MAX_GAP_DOWN_PCT}% to {-MIN_GAP_DOWN_PCT}%): {len(gap_down):,}")
    print(f"  gap-UP   events ({MIN_GAP_DOWN_PCT}% to {MAX_GAP_DOWN_PCT}%): {len(gap_up):,}")

    return gap_down, gap_up


def apply_no_news_filter(
    gap_events: pd.DataFrame,
    blackout: Set[Tuple[str, date]],
    covered_syms: Set[str],
) -> pd.DataFrame:
    """Drop (symbol, day) pairs that fall in earnings +/-2d blackout.

    Symbols outside earnings_calendar coverage are NOT dropped — they're
    flagged as 'news_status_unknown' so we can report split metrics.
    """
    print(f"  applying earnings +/-{EARNINGS_BLACKOUT_DAYS}d blackout ...")
    pre = len(gap_events)
    gap_events = gap_events.copy()

    def status(row) -> str:
        sym, d_ = row["symbol"], row["d"]
        if sym not in covered_syms:
            return "uncovered"
        if (sym, d_) in blackout:
            return "earnings_blackout"
        return "no_news"

    gap_events["news_status"] = gap_events.apply(status, axis=1)
    counts = gap_events["news_status"].value_counts().to_dict()
    print(f"  news_status counts: {counts}")

    # Strict no-news = drop earnings_blackout (keep no_news + uncovered).
    filtered = gap_events[gap_events["news_status"] != "earnings_blackout"].copy()
    print(f"  after strict no-news filter (drop blackout): {len(filtered):,} (dropped {pre - len(filtered):,})")
    return filtered


def find_confirmation_bars(
    gap_events: pd.DataFrame,
    big5m: pd.DataFrame,
) -> pd.DataFrame:
    """For each gap-down (symbol, d), scan 09:15/09:20/09:25 for an exhaustion
    candle (lower_wick/body >= 0.5 AND body_size_pct <= 30% AND green).

    Apply volume-decline filter for non-opening bars.
    Returns the confirmation bar's row (with gap-event metadata joined).
    """
    print("  scanning 09:15 / 09:20 / 09:25 bars for exhaustion candles ...")
    keys = set(zip(gap_events["symbol"], gap_events["d"]))
    morn = big5m[big5m["date"].dt.strftime("%H:%M").isin(ACTIVE_WINDOW_HHMM)].copy()
    morn["key"] = list(zip(morn["symbol"], morn["d"]))
    morn = morn[morn["key"].apply(lambda k: k in keys)].copy()
    morn["hhmm"] = morn["date"].dt.strftime("%H:%M")
    morn = morn.sort_values(["symbol", "d", "date"]).reset_index(drop=True)
    print(f"  morning-window bars in gap-down universe: {len(morn):,}")

    # Compute candle features
    morn["body"] = (morn["close"] - morn["open"]).abs()
    candle_bot = morn[["open", "close"]].min(axis=1)
    morn["lower_wick"] = candle_bot - morn["low"]
    # Avoid div-by-zero: when body is negligible treat ratio as inf so it
    # passes the "wick >= 0.5x body" floor (extreme exhaustion = doji bottom)
    morn["lower_wick_ratio"] = np.where(
        morn["body"] < 1e-8, np.inf, morn["lower_wick"] / morn["body"].replace(0, np.nan)
    )
    morn["body_size_pct"] = np.where(
        morn["open"] > 0, (morn["body"] / morn["open"]) * 100.0, np.nan
    )
    morn["is_green"] = morn["close"] > morn["open"]

    # Volume of 09:15 bar per (symbol, d)
    vol_915 = (
        morn[morn["hhmm"] == "09:15"][["symbol", "d", "volume"]]
        .rename(columns={"volume": "vol_09_15"})
    )
    morn = morn.merge(vol_915, on=["symbol", "d"], how="left")

    # Exhaustion mask
    exh_mask = (
        (morn["lower_wick_ratio"] >= MIN_LOWER_WICK_RATIO)
        & (morn["body_size_pct"] <= MAX_BODY_SIZE_PCT)
        & (morn["is_green"])
    )
    print(f"  bars passing exhaustion-candle (wick+body+green): {int(exh_mask.sum()):,}")

    # Volume decline (skip the 09:15 bar itself)
    if REQUIRE_VOL_DECLINE:
        vol_ok = (morn["hhmm"] == "09:15") | (morn["volume"] < morn["vol_09_15"])
        exh_mask = exh_mask & vol_ok
        print(f"  after volume-decline filter:                  {int(exh_mask.sum()):,}")

    confirms = morn[exh_mask].copy()
    # Latch first qualifying confirmation per (symbol, d)
    confirms = confirms.sort_values(["symbol", "d", "date"]).drop_duplicates(
        subset=["symbol", "d"], keep="first"
    )
    print(f"  unique (symbol, day) confirmations:            {len(confirms):,}")

    # Join gap-event metadata
    gap_meta = gap_events.set_index(["symbol", "d"])[
        ["pdc", "open_09_15", "high_09_15", "low_09_15", "vol_09_15",
         "gap_pct", "cap_segment", "adv_20d_cr", "news_status"]
    ].rename(columns={"vol_09_15": "vol_09_15_meta"})
    confirms = confirms.merge(gap_meta, left_on=["symbol", "d"], right_index=True,
                              how="left", suffixes=("", "_meta"))
    return confirms


def compute_atr_5m_per_day(big5m: pd.DataFrame) -> pd.Series:
    """Compute simple intraday ATR proxy: per-(symbol, d) mean(high-low).

    Brief is silent on ATR window — using full-day mean is a defensible
    proxy at sanity stage. Without modifying big5m (memory-conscious).
    """
    print("  computing per-day ATR proxy (mean bar range) ...")
    rng = (big5m["high"].values - big5m["low"].values)
    tmp = pd.DataFrame({
        "symbol": big5m["symbol"].values,
        "d": big5m["d"].values,
        "rng": rng,
    })
    atr = tmp.groupby(["symbol", "d"])["rng"].mean()
    return atr


def simulate(
    confirms: pd.DataFrame,
    big5m: pd.DataFrame,
    atr_table: pd.Series,
) -> pd.DataFrame:
    """For each confirmation bar, enter at NEXT bar's open (Streak/AlgoTest
    convention; matches volume_spike_exhaustion_reversal Q1 decision).

    Walk forward through 5m bars until exit:
      - hard SL hit (or breakeven if T1 already filled)
      - T2 hit
      - 10:15 IST time stop
    """
    print("  simulating LONG entries -> exits ...")
    # Only build day_dfs for symbols that have at least one confirmation —
    # avoids materializing 1500+ per-symbol frames when most won't fire.
    needed_syms = set(confirms["symbol"].unique())
    sub5m = big5m[big5m["symbol"].isin(needed_syms)]
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in sub5m.groupby("symbol")
    }
    del sub5m

    trades: List[dict] = []
    n_no_entry = n_no_bars = n_stop_zero = n_traded = 0

    for _, c in confirms.iterrows():
        sym = c["symbol"]; sd = c["d"]
        conf_ts = c["date"]
        gap_low = float(c["low_09_15"])
        pdc = float(c["pdc"])
        cap_seg = c["cap_segment"]

        sym_df = days_per_sym.get(sym)
        if sym_df is None:
            n_no_entry += 1; continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty:
            n_no_entry += 1; continue
        idx_arr = day_df.index[day_df["date"] == conf_ts].tolist()
        if not idx_arr or idx_arr[0] + 1 >= len(day_df):
            n_no_entry += 1; continue
        conf_idx = idx_arr[0]

        # Entry at next bar's open
        entry_bar = day_df.iloc[conf_idx + 1]
        entry_price = float(entry_bar["open"])
        entry_ts = entry_bar["date"]

        # ATR proxy for this (symbol, d)
        atr_val = float(atr_table.get((sym, sd), entry_price * 0.01))

        # Stop construction (LONG)
        cap_buf = STOP_BUF_MICRO if cap_seg == "micro_cap" else STOP_BUF_OTHER
        stop_a = gap_low * (1.0 - cap_buf)
        stop_b = entry_price - atr_val * ATR_STOP_MULT
        hard_sl = min(stop_a, stop_b)

        # min_stop_distance_pct enforcement
        sl_min_floor = entry_price * (1.0 - MIN_STOP_PCT / 100.0)
        if hard_sl > sl_min_floor:
            hard_sl = sl_min_floor

        stop_distance = entry_price - hard_sl
        if stop_distance <= 0:
            n_stop_zero += 1; continue

        # Targets (structural, anchored to PDC)
        t1_target = (entry_price + pdc) / 2.0  # 50% gap fill
        t2_target = pdc                         # full PDC
        # Sanity: if entry already above PDC (rare; gap could close fast),
        # skip — no reversion edge to capture.
        if t1_target <= entry_price or t2_target <= entry_price:
            n_stop_zero += 1; continue

        # Walk forward: from entry_bar until 10:15 hard time stop
        forward = day_df.iloc[conf_idx + 1:].copy()
        forward["hhmm"] = forward["date"].dt.strftime("%H:%M")
        if forward.empty:
            n_no_bars += 1; continue

        exit_ts = None; exit_price = None; exit_reason = None
        hit_t1 = False; t1_exit_price = None

        for _, bar in forward.iterrows():
            ts = bar["date"]; high = float(bar["high"]); low = float(bar["low"])
            close = float(bar["close"]); hhmm = bar["hhmm"]

            active_sl = (entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl)
            # LONG: stop is below
            if low <= active_sl:
                exit_ts = ts; exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break
            if not hit_t1 and high >= t1_target:
                hit_t1 = True; t1_exit_price = t1_target
            if hit_t1 and high >= t2_target:
                exit_ts = ts; exit_price = t2_target; exit_reason = "t2"
                break
            if hhmm >= TIME_STOP_HHMM:
                exit_ts = ts; exit_price = close; exit_reason = "time_stop"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]; exit_price = float(last["close"])
            exit_reason = "time_stop"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        if hit_t1:
            qty_t1 = qty // 2; qty_t2 = qty - qty_t1
            pnl_t1 = (t1_exit_price - entry_price) * qty_t1
            pnl_t2 = (exit_price - entry_price) * qty_t2
            realized_pnl = pnl_t1 + pnl_t2
            fee = (calc_fee(entry_price, t1_exit_price, qty_t1, "BUY")
                   + calc_fee(entry_price, exit_price, qty_t2, "BUY"))
            blended_exit = (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
        else:
            realized_pnl = (exit_price - entry_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "BUY")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "T0_signal_date": sd,
            "symbol": "NSE:" + sym,
            "bare_symbol": sym,
            "cap_segment": cap_seg,
            "side": "LONG",
            "news_status": c["news_status"],
            "gap_pct": float(c["gap_pct"]),
            "pdc": pdc,
            "open_09_15": float(c["open_09_15"]),
            "low_09_15": float(c["low_09_15"]),
            "vol_09_15": float(c["vol_09_15_meta"]),
            "adv_20d_cr": float(c["adv_20d_cr"]),
            "confirmation_ts": conf_ts,
            "confirmation_hhmm": c["hhmm"],
            "lower_wick_ratio": float(c["lower_wick_ratio"]),
            "body_size_pct": float(c["body_size_pct"]),
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "atr_proxy": atr_val,
            "t1_target": t1_target,
            "t2_target": t2_target,
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

    print(f"\n  no entry bar:        {n_no_entry}")
    print(f"  no bars after:       {n_no_bars}")
    print(f"  zero/negative stop:  {n_stop_zero}")
    print(f"  traded:              {n_traded}")
    return pd.DataFrame(trades)


def _agg_metrics(grp: pd.DataFrame) -> Tuple[int, float, float, int]:
    """Return (n, pf, wr, net) for a group."""
    n = len(grp)
    if n == 0:
        return 0, float("nan"), float("nan"), 0
    npnl = grp["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    wr = round(float((npnl > 0).mean()) * 100, 1)
    net = int(npnl.sum())
    return n, pf, wr, net


def report(trades: pd.DataFrame, gap_up_keys: Set[Tuple[str, date]]) -> None:
    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades")
        print("\n--- VERDICT ---")
        print("n=0 < 500 -> STRUCTURAL RETIRE (data + filter combo yields no signal).")
        return

    n, pf, wr, net = _agg_metrics(trades)
    npnl = trades["net_pnl"]
    daily = trades.groupby("T0_signal_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0

    print("\n=== capitulation_long_morning -- pre-coding sanity check ===")
    print(f"Period: {trades['T0_signal_date'].min()} .. {trades['T0_signal_date'].max()}")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   Rs.{net:,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    print("\nPer cap_segment (cell selection diagnostic — gauntlet's job at Stage 3):")
    for cap, grp in trades.groupby("cap_segment"):
        n2, pf2, wr2, net2 = _agg_metrics(grp)
        print(f"  {str(cap):<14} n={n2:>5} PF={pf2:>6} WR={wr2:>5}% netPnL=Rs.{net2:>11,}")

    print("\nPer news_status (no-news filter diagnostic):")
    pf_by_status: Dict[str, float] = {}
    for st, grp in trades.groupby("news_status"):
        n2, pf2, wr2, net2 = _agg_metrics(grp)
        pf_by_status[st] = pf2
        print(f"  {st:<22} n={n2:>5} PF={pf2:>6} WR={wr2:>5}% netPnL=Rs.{net2:>11,}")

    # Per-month
    print("\nPer month (n / PF / netPnL):")
    trades["_month"] = pd.to_datetime(trades["T0_signal_date"]).dt.strftime("%Y-%m")
    for mth, grp in trades.groupby("_month"):
        n2, pf2, wr2, net2 = _agg_metrics(grp)
        print(f"  {mth:<8} n={n2:>4} PF={pf2:>6} netPnL=Rs.{net2:>11,}")

    # Exit-reason
    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        n2 = len(grp)
        avg_net = int(grp["net_pnl"].mean())
        print(f"  {rsn:<22} n={n2:>5} avg_net=Rs.{avg_net:>7,}")

    # Independence check vs gap_fade_short triggers (mutual exclusivity by construction)
    fired_keys = set(zip(trades["bare_symbol"], trades["T0_signal_date"]))
    overlap = fired_keys & gap_up_keys
    overlap_pct = (len(overlap) / max(len(fired_keys), 1)) * 100.0
    print(f"\nIndependence check vs gap_fade_short (gap-up trigger universe):")
    print(f"  capitulation_long fires:                {len(fired_keys)}")
    print(f"  gap-up trigger (symbol, day) universe:  {len(gap_up_keys)}")
    print(f"  overlap (same symbol-day):              {len(overlap)}")
    print(f"  overlap pct:                            {overlap_pct:.2f}%")
    if overlap_pct > 0.5:
        print("  -> WARNING: overlap > 0.5% violates mechanical independence claim.")
    else:
        print("  -> PASS: gap-up vs gap-down are mutually exclusive on same symbol-day.")

    # Sample-size diagnostic
    print(f"\nSample-size diagnostic: n={n} (HARD floor: 500)")
    if n < 500:
        print("  -> n < 500: STRUCTURAL RETIRE (filter combo too restrictive on broad universe).")
    else:
        print(f"  -> n >= 500 cleared (margin: {n - 500}, ratio: {n / 500:.2f}x).")

    # Strict no-news vs uncovered PF lift diagnostic
    pf_no_news = pf_by_status.get("no_news", float("nan"))
    pf_uncov = pf_by_status.get("uncovered", float("nan"))
    if not np.isnan(pf_no_news) and not np.isnan(pf_uncov):
        lift = pf_no_news - pf_uncov
        print(f"\nNo-news filter PF-lift proxy (no_news vs uncovered): {lift:+.3f}")
        print(f"  (brief §9 falsification: lift >= 0.10 needed to confirm filter is load-bearing)")

    # Verdict
    print("\n--- VERDICT ---")
    if n < 500:
        print(f"VERDICT: STRUCTURAL RETIRE (n={n} < 500 hard floor; brief §9 criterion).")
    elif pf < 1.10:
        print(f"VERDICT: THESIS RETIRE (n={n} OK but PF={pf} < 1.10; brief §9 criterion).")
    else:
        print(f"VERDICT: STRONG PROCEED (n={n} >= 500 AND PF={pf} >= 1.10).")


def main():
    big5m = build_full_period_5m()
    daily = load_consolidated_daily()
    blackout, covered_syms = load_earnings_blackout()

    print("\nFinding gap events ...")
    gap_down, gap_up = find_gap_events(big5m, daily)
    gap_up_keys: Set[Tuple[str, date]] = set(zip(gap_up["symbol"], gap_up["d"]))

    if gap_down.empty:
        print("[NO GAP-DOWN EVENTS] aborting.")
        return

    print("\nApplying no-news filter ...")
    gap_filtered = apply_no_news_filter(gap_down, blackout, covered_syms)

    if gap_filtered.empty:
        print("[NO POST-FILTER EVENTS] aborting.")
        return

    print("\nFinding confirmation bars ...")
    confirms = find_confirmation_bars(gap_filtered, big5m)

    if confirms.empty:
        print("[NO CONFIRMATIONS] aborting.")
        return

    atr_table = compute_atr_5m_per_day(big5m)

    print("\nSimulating entries -> exits:")
    trades = simulate(confirms, big5m, atr_table)

    # Funnel summary (concise)
    print("\n=== FUNNEL ===")
    print(f"  unique stocks (5m universe):    {big5m['symbol'].nunique():,}")
    print(f"  gap-down events (broad):        {len(gap_down):,}")
    print(f"  after no-news filter:           {len(gap_filtered):,}")
    print(f"  after confirmation candle:      {len(confirms):,}")
    print(f"  fired (final trades):           {len(trades):,}")

    report(trades, gap_up_keys)

    out = _REPO_ROOT / "reports" / "sub9_sanity" / "capitulation_long_morning_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
