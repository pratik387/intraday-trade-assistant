"""Pre-coding sanity check for mwpl_recalc_forced_rebalance_fade candidate (C1).

Per brief: specs/2026-05-14-brief-mwpl_recalc_forced_rebalance_fade.md
Initial verdict in brief: MARGINAL ("sample ~40-80 trades at/below stat-power floor").
This script actually runs the sanity that was deferred — uses the quarterly
recalc dates as event signals and tests whether mid/small-cap F&O stocks drift
DOWN in T+1..T+5/T+10 post-recalc.

============================================================
SIMPLIFIED METHODOLOGY (per controller brief 2026-05-14)
============================================================

Because we don't yet have the per-stock MWPL parquet + UDiFF FutEq-OI parquet
(data-engineering work deferred until C1 sanity passes), this sanity uses the
QUARTERLY RECALC DATES THEMSELVES as the event trigger — no OI/MWPL ratio
threshold. Universe is the full F&O 200 mid/small-cap subset. This is more
conservative for the mechanism (no qualification filter = harder to find edge)
but more sample-friendly. If the mechanism prints here, it almost certainly
prints harder when filtered to the >=0.85 OI/MWPL cohort.

Event dates tested (12 quarterly cutovers 2024-01 .. 2026-04):
  Pre-rule (old MWPL formula):
    2024-01-01, 2024-04-01, 2024-07-01, 2024-10-01
    2025-01-01, 2025-04-01, 2025-07-01
  Rule-launch quarter:
    2025-10-01   ** also itself a regime break (critical SEBI MWPL rule) **
  Post-rule (new MWPL formula):
    2026-01-01, 2026-04-01

Two variants per event:
  - INTRADAY: T+1 09:30 SHORT, exit T+1 15:10 (single MIS session)
  - SWING:    T+1 open SHORT (full qty), exit T+5 close (CNC hypothesis)

Both use Rs 1000 risk-per-trade sizing for fair PF comparison.

============================================================
Gauntlet-v2 ship gates
============================================================
  n >= 100
  PF >= 1.20
  Sharpe > 0
  per-month stability (best month / worst month <= 3x; no single month carries)

============================================================
regime_break_detector pre-flight
============================================================
depends_on = [MWPL, single_stock_FO, F&O_speculation]
The 2025-10-01 critical MWPL row IS the rule-launch event — this is the brief's
"rule_creating sensitivity means rule launch IS the start of Discovery, not a
break within it" exception. We surface it; we do NOT abort.

============================================================
Outputs
============================================================
  reports/sub9_sanity/mwpl_recalc_drift_short_trades.csv

Usage:
    .venv/Scripts/python -m tools.sub9_research.sanity_mwpl_recalc_drift_short
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from services.state.zerodha_mis_fetcher import ZerodhaMISFetcher  # noqa: E402
from services.regime_break_detector import (                  # noqa: E402
    check_window,
)
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ============================================================================
# Config (brief-aligned; quarter cutover dates are LOCKED)
# ============================================================================

# Quarterly MWPL recalc dates per brief §"Mechanism" — Apr/Jul/Oct/Jan
# We test the 12 quarter cutovers in 2024-01 .. 2026-04 window.
RECALC_DATES: List[Tuple[date, str, str]] = [
    # (recalc_effective_date, regime_label, quarter_label)
    (date(2024, 1, 1),  "pre_rule",     "Q4_2023"),  # Q1 cutover refers to *previous* year's Q4 end-of-quarter? No — quarterly recalc names per brief: Apr=Q1, Jul=Q2, Oct=Q3, Jan=Q4
    (date(2024, 4, 1),  "pre_rule",     "Q1_2024"),
    (date(2024, 7, 1),  "pre_rule",     "Q2_2024"),
    (date(2024, 10, 1), "pre_rule",     "Q3_2024"),
    (date(2025, 1, 1),  "pre_rule",     "Q4_2024"),
    (date(2025, 4, 1),  "pre_rule",     "Q1_2025"),
    (date(2025, 7, 1),  "pre_rule",     "Q2_2025"),
    (date(2025, 10, 1), "rule_launch",  "Q3_2025"),   # the SEBI MWPL re-formula effective date
    (date(2026, 1, 1),  "post_rule",    "Q4_2025"),
    (date(2026, 4, 1),  "post_rule",    "Q1_2026"),   # also STT hike
]

ALLOWED_CAPS = {"mid_cap", "small_cap"}   # per brief — large-cap rarely OI-binding; micro not F&O
# Fallback: F&O 200 list is heavily large-cap-skewed by construction. Only 4 of
# 153 F&O 200 symbols are tagged mid_cap in nse_all.json, and 0 small_cap.
# This is per-row sample of 4 stocks/event x 10 events = 40 trades, BELOW the
# brief's own estimate (40-80) and well below the n>=100 gauntlet floor.
# Brief §"Sample availability estimate" lines 92-100 acknowledged this risk.
# To produce a meaningful mechanism check we ALSO run a "wider" variant that
# admits large_cap and unknown (still excludes micro_cap). Documented in
# output as the secondary lane — the primary mid/small-cap lane is reported
# first per brief lock-in.
ALLOWED_CAPS_WIDE = {"mid_cap", "small_cap", "large_cap", "unknown"}
ENTRY_HHMM = "09:30"
EXIT_HHMM_INTRADAY = "15:10"
SWING_HOLD_DAYS = 5   # T+1 entry -> T+5 close exit (brief upper bound; brief says 5-10)

# Stop / target (mirroring fno_removal_drift_short Cell A)
SL_ENTRY_CAP_PCT = 1.5     # entry * 1.015 — upper cap
SL_DAYHIGH_BUFFER_PCT = 0.5  # day-high-so-far * 1.005
SL_FLOOR_PCT = 1.0          # entry * 1.01 — lower floor

T1_TARGET_PCT = 1.0         # entry * 0.99
T2_TARGET_PCT = 2.0         # entry * 0.98
T1_QTY_FRAC = 0.5
USE_BE_TRAIL_AFTER_T1 = True

RISK_PER_TRADE_RUPEES = 1000

# Ship gates (per controller — gauntlet-v2)
PF_MIN = 1.20
N_MIN = 100
SHARPE_MIN = 0.0
PER_MONTH_STABILITY_MAX_RATIO = 3.0   # best_month_PF / worst_month_PF <= 3

# Regime break deps (per brief)
DEPENDS_ON = ["MWPL", "single_stock_FO", "F&O_speculation"]


# ============================================================================
# Universe: F&O liquid 200
# ============================================================================

_FNO_200_PATH = _REPO_ROOT / "assets" / "fno_liquid_200.csv"


def load_fno_universe() -> set:
    """Return set of bare symbols (no NSE: prefix) for F&O 200 universe."""
    if not _FNO_200_PATH.exists():
        raise FileNotFoundError(f"F&O 200 universe file missing: {_FNO_200_PATH}")
    df = pd.read_csv(_FNO_200_PATH)
    symbols = set()
    for s in df["symbol"]:
        s = str(s).strip()
        if s.startswith("NSE:"):
            s = s[4:]
        symbols.add(s)
    print(f"  F&O 200 universe loaded: {len(symbols)} symbols")
    return symbols


# ============================================================================
# MIS-eligibility (for intraday variant, production parity)
# ============================================================================

_MIS_FETCHER = None


def _get_mis_allowed_set() -> set:
    global _MIS_FETCHER
    if _MIS_FETCHER is None:
        _MIS_FETCHER = ZerodhaMISFetcher()
        if not _MIS_FETCHER.load_from_zerodha():
            print("  WARN: MIS list load failed — proceeding without MIS filter")
            return set()
        print(f"  MIS list loaded: {_MIS_FETCHER.count()} symbols")
    return set(_MIS_FETCHER._mis_symbols.keys())


# ============================================================================
# 5m + daily loaders
# ============================================================================

_NEEDED_5M_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = (_REPO_ROOT / "backtest-cache-download" / "monthly"
            / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_feather(path, columns=_NEEDED_5M_COLS)
    except Exception:
        return pd.read_feather(path)


def _months_to_load() -> List[Tuple[int, int]]:
    """Load the recalc month + next month (for swing T+5 lookahead).

    Each recalc is on the 1st of a month; T+1 entry is usually the same month.
    Swing exit on T+5 may straddle into the next month for early-month recalcs.
    """
    months: set = set()
    for (d, _, _) in RECALC_DATES:
        months.add((d.year, d.month))
        # next month (T+5 lookahead)
        nxt_m = (d.month % 12) + 1
        nxt_y = d.year + (1 if d.month == 12 else 0)
        months.add((nxt_y, nxt_m))
    return sorted(months)


def build_window_5m() -> pd.DataFrame:
    months = _months_to_load()
    print(f"  loading {len(months)} monthly 5m feathers ...")
    parts: List[pd.DataFrame] = []
    for (y, m) in months:
        mdf = _load_5m_for_month(y, m)
        if not mdf.empty:
            parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = (pd.concat(parts, ignore_index=True)
           .sort_values(["symbol", "date"])
           .reset_index(drop=True))
    big["d"] = big["date"].dt.date
    big["hhmm"] = big["date"].dt.strftime("%H:%M")
    print(f"  total 5m bars: {len(big):,}")
    return big


def load_production_daily() -> pd.DataFrame:
    """Load 1day OHLCV (for swing T+5 close exit + prior-day close reference)."""
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        print(f"  WARN: {daily_path} missing; swing variant will be skipped.")
        return pd.DataFrame()
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[["symbol", "d", "open", "high", "low", "close", "volume"]].copy()
    print(f"  daily rows: {len(df):,} | symbols: {df['symbol'].nunique()}")
    return df


# ============================================================================
# Helper: find T+N trading day for a symbol
# ============================================================================


def _next_trading_day(distinct_days_sorted: List[date], after: date) -> Optional[date]:
    """First trading day strictly AFTER `after` in the sorted list."""
    for d in distinct_days_sorted:
        if d > after:
            return d
    return None


def _trading_day_offset(
    distinct_days_sorted: List[date],
    base: date,
    offset_n: int,
) -> Optional[date]:
    """Nth trading day at-or-after `base` (offset_n=0 → first session at-or-after base;
    offset_n=1 → next; etc.)."""
    candidates = [d for d in distinct_days_sorted if d >= base]
    if len(candidates) <= offset_n:
        return None
    return candidates[offset_n]


# ============================================================================
# Pre-flight: regime-break detector
# ============================================================================


def regime_preflight() -> None:
    """Surface all rule rows in 2024-01-01..2026-04-30 for [MWPL, single_stock_FO,
    F&O_speculation]. We expect 2025-10-01 critical MWPL row to be flagged —
    that's the rule_creating launch, not a break (per brief §Pre-flight bypass).
    """
    print("\n=== regime_break_detector pre-flight ===")
    print(f"  strategy:    mwpl_recalc_forced_rebalance_fade")
    print(f"  depends_on:  {DEPENDS_ON}")

    full_start = RECALC_DATES[0][0]
    full_end = date(2026, 4, 30)
    print(f"\n  full sanity window: {full_start} .. {full_end}")

    hits = check_window(
        strategy_name="mwpl_recalc_forced_rebalance_fade",
        depends_on=DEPENDS_ON,
        window_label="sanity_full",
        start=full_start,
        end=full_end,
        min_severity="medium",   # surface medium+ for visibility
        raise_on_break=False,
    )
    if not hits:
        print("    no rule rows in window (clean).")
        return
    for r in hits:
        desc = (r.description[:90]
                .encode("ascii", errors="replace").decode("ascii"))
        print(f"    {r.effective_date} [{r.severity.upper():<8}] {desc}")

    crit = [r for r in hits if r.severity == "critical"]
    if crit:
        print(f"\n  note: {len(crit)} critical row(s) — per brief §Pre-flight: "
              f"the 2025-10-01 MWPL rule-launch IS one of the events under "
              f"test (rule_creating sensitivity). NOT aborting.")


# ============================================================================
# Intraday simulator (T+1 09:30 SHORT → 15:10 EOD or stop/target)
# ============================================================================


def simulate_intraday(
    events: pd.DataFrame,   # rows: recalc_date, regime, quarter, symbol, cap_segment, entry_date
    big5m: pd.DataFrame,
) -> pd.DataFrame:
    """Bar-walk SHORT entry simulator with T1 partial + BE trail (mirror Cell A
    pattern from fno_removal_drift_short)."""
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date") for sym, g in big5m.groupby("symbol")
    }
    trades: List[dict] = []
    n_no_data = n_no_entry_bar = n_zero_stop = 0
    n_traded = 0

    for _, ev in events.iterrows():
        sym = ev["symbol"]
        entry_d = ev["entry_date"]
        sym_df = days_per_sym.get(sym)
        if sym_df is None or sym_df.empty:
            n_no_data += 1
            continue
        day_df = (sym_df[sym_df["d"] == entry_d]
                  .sort_values("date").reset_index(drop=True))
        if day_df.empty:
            n_no_data += 1
            continue

        entry_rows = day_df[day_df["hhmm"] == ENTRY_HHMM]
        if entry_rows.empty:
            n_no_entry_bar += 1
            continue
        entry_bar = entry_rows.iloc[0]
        entry_idx = entry_bar.name
        entry_ts = entry_bar["date"]
        entry_price = float(entry_bar["close"])

        # Day-high-so-far at entry = max(high) across 09:15-09:30 bars
        window_so_far = day_df[day_df["hhmm"] <= ENTRY_HHMM]
        day_high_so_far = float(window_so_far["high"].max())

        sl_cap = entry_price * (1.0 + SL_ENTRY_CAP_PCT / 100.0)
        sl_daily = day_high_so_far * (1.0 + SL_DAYHIGH_BUFFER_PCT / 100.0)
        sl_floor = entry_price * (1.0 + SL_FLOOR_PCT / 100.0)
        # min(cap, daily), THEN floor at entry*1.01
        candidate = min(sl_cap, sl_daily)
        hard_sl = max(candidate, sl_floor)
        stop_distance = hard_sl - entry_price
        if stop_distance <= 0:
            n_zero_stop += 1
            continue

        t1_target = entry_price * (1.0 - T1_TARGET_PCT / 100.0)
        t2_target = entry_price * (1.0 - T2_TARGET_PCT / 100.0)

        forward = (day_df.iloc[entry_idx + 1:].copy()
                   if entry_idx + 1 < len(day_df) else pd.DataFrame())
        if forward.empty:
            n_no_entry_bar += 1
            continue

        exit_ts = None
        exit_price: Optional[float] = None
        exit_reason: Optional[str] = None
        hit_t1 = False
        t1_exit_price: Optional[float] = None

        for _, bar in forward.iterrows():
            ts = bar["date"]
            hi = float(bar["high"])
            lo = float(bar["low"])
            cl = float(bar["close"])
            hhmm = bar["hhmm"]

            active_sl = entry_price if (hit_t1 and USE_BE_TRAIL_AFTER_T1) else hard_sl

            # SHORT: SL is ABOVE entry; check first
            if hi >= active_sl:
                exit_ts = ts
                exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break

            if (not hit_t1) and (lo <= t1_target):
                hit_t1 = True
                t1_exit_price = t1_target

            if lo <= t2_target:
                exit_ts = ts
                exit_price = t2_target
                exit_reason = "t2"
                break

            if hhmm >= EXIT_HHMM_INTRADAY:
                exit_ts = ts
                exit_price = cl
                exit_reason = "time_stop"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "last_bar"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        if hit_t1:
            qty_t1 = max(int(qty * T1_QTY_FRAC), 1)
            qty_t2 = qty - qty_t1
            if qty_t2 < 1:
                qty_t2 = 0
            pnl_t1 = (entry_price - t1_exit_price) * qty_t1
            pnl_runner = ((entry_price - exit_price) * qty_t2
                          if qty_t2 > 0 else 0.0)
            realized_pnl = pnl_t1 + pnl_runner
            fee = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL")
            if qty_t2 > 0:
                fee += calc_fee(entry_price, exit_price, qty_t2, "SELL")
            blended_exit = (
                (t1_exit_price * qty_t1 + exit_price * qty_t2) / qty
                if qty_t2 > 0 else t1_exit_price
            )
            if exit_reason == "t2":
                exit_reason = "t1_partial+t2_full"
            elif exit_reason == "time_stop":
                exit_reason = "t1_partial+time_stop"
            elif exit_reason == "breakeven_trail":
                exit_reason = "t1_partial+be_trail"
            elif exit_reason == "last_bar":
                exit_reason = "t1_partial+last_bar"
        else:
            realized_pnl = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "variant":         "intraday",
            "recalc_date":     ev["recalc_date"],
            "regime":          ev["regime"],
            "quarter":         ev["quarter"],
            "session_date":    entry_d,
            "T1_entry_date":   entry_d,   # compat alias
            "symbol":          "NSE:" + sym,
            "bare_symbol":     sym,
            "cap_segment":     ev["cap_segment"],
            "side":            "SHORT",
            "entry_ts":        entry_ts,
            "entry_price":     entry_price,
            "day_high_so_far": day_high_so_far,
            "hard_sl":         hard_sl,
            "t1_target":       t1_target,
            "t2_target":       t2_target,
            "stop_distance":   stop_distance,
            "hit_t1":          hit_t1,
            "exit_ts":         exit_ts,
            "exit_price":      blended_exit,
            "exit_reason":     exit_reason,
            "qty":             qty,
            "realized_pnl":    realized_pnl,
            "fee":             fee,
            "net_pnl":         net_pnl,
        })
        n_traded += 1

    print(f"\n  [INTRADAY] simulator counters:")
    print(f"    no 5m data:          {n_no_data}")
    print(f"    no 09:30 entry bar:  {n_no_entry_bar}")
    print(f"    zero/neg stop:       {n_zero_stop}")
    print(f"    traded:              {n_traded}")

    return pd.DataFrame(trades)


# ============================================================================
# Swing simulator (T+1 open SHORT, exit T+5 close — CNC)
# ============================================================================


def simulate_swing(
    events: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """T+1 open SHORT → T+5 close exit. Uses Rs 1000 risk sizing with a stop
    distance computed from prior-day close to entry to keep sizing comparable
    to intraday variant. If T+5 trading day not available, skip.

    Note: 'swing' uses CNC (no MIS leverage). For sanity comparison we use
    the same calc_fee — STT delivery would be ~0.1% per side but our calc_fee
    is the intraday model. We accept this is OPTIMISTIC fee model for swing
    (CNC would have higher STT); the gross PnL is what really matters here
    for the mechanism check.
    """
    trades: List[dict] = []
    n_no_data = n_no_t5 = n_traded = 0

    daily_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("d").reset_index(drop=True)
        for sym, g in daily.groupby("symbol")
    }

    for _, ev in events.iterrows():
        sym = ev["symbol"]
        entry_d = ev["entry_date"]
        d_sym = daily_per_sym.get(sym)
        if d_sym is None or d_sym.empty:
            n_no_data += 1
            continue

        # entry day's row (T+1 trading day after recalc)
        d_sym_dates = list(d_sym["d"])
        if entry_d not in d_sym_dates:
            n_no_data += 1
            continue
        entry_row = d_sym[d_sym["d"] == entry_d].iloc[0]
        entry_price = float(entry_row["open"])
        prior_close = None
        # prior day close (for stop calc)
        idx_entry = d_sym_dates.index(entry_d)
        if idx_entry > 0:
            prior_close = float(d_sym.iloc[idx_entry - 1]["close"])

        # Exit on T+SWING_HOLD_DAYS trading day close (0-indexed from entry day:
        # entry_d is day 0; exit on day 5 = 5 trading days after entry).
        exit_idx = idx_entry + SWING_HOLD_DAYS
        if exit_idx >= len(d_sym):
            n_no_t5 += 1
            continue
        exit_row = d_sym.iloc[exit_idx]
        exit_d = exit_row["d"]
        exit_price = float(exit_row["close"])

        # Stop distance for sizing: use a notional 2% of entry (typical swing
        # short stop). This is purely for qty sizing to keep Rs-risk comparable.
        notional_stop_distance = entry_price * 0.02
        qty = max(int(RISK_PER_TRADE_RUPEES / max(notional_stop_distance, 1e-6)), 1)

        # SHORT: pnl = (entry - exit) * qty
        realized_pnl = (entry_price - exit_price) * qty
        fee = calc_fee(entry_price, exit_price, qty, "SELL")
        net_pnl = realized_pnl - fee
        pct = (entry_price / exit_price - 1.0) * 100.0

        trades.append({
            "variant":         "swing",
            "recalc_date":     ev["recalc_date"],
            "regime":          ev["regime"],
            "quarter":         ev["quarter"],
            "session_date":    entry_d,
            "T1_entry_date":   entry_d,
            "symbol":          "NSE:" + sym,
            "bare_symbol":     sym,
            "cap_segment":     ev["cap_segment"],
            "side":            "SHORT",
            "entry_price":     entry_price,
            "exit_price":      exit_price,
            "exit_date":       exit_d,
            "exit_reason":     f"t+{SWING_HOLD_DAYS}_close",
            "stop_distance":   notional_stop_distance,
            "qty":             qty,
            "realized_pnl":    realized_pnl,
            "fee":             fee,
            "net_pnl":         net_pnl,
            "pct_drift":       pct,
            "hit_t1":          False,
            "day_high_so_far": float("nan"),
            "hard_sl":         float("nan"),
            "t1_target":       float("nan"),
            "t2_target":       float("nan"),
            "entry_ts":        pd.NaT,
            "exit_ts":         pd.NaT,
        })
        n_traded += 1

    print(f"\n  [SWING] simulator counters:")
    print(f"    no daily data:    {n_no_data}")
    print(f"    no T+{SWING_HOLD_DAYS} day:        {n_no_t5}")
    print(f"    traded:           {n_traded}")

    return pd.DataFrame(trades)


# ============================================================================
# Event builder: cross-join (recalc dates) x (F&O 200 mid/small-cap universe)
# ============================================================================


def build_events(
    big5m: pd.DataFrame,
    daily: pd.DataFrame,
    fno_universe: set,
    mis_allowed: set,
    apply_mis_filter: bool,
    allowed_caps: set = None,
) -> pd.DataFrame:
    """For each recalc date, find T+1 trading day and emit one row per qualifying
    symbol. Universe filter: F&O 200 + cap_segment in `allowed_caps` (+ MIS for
    intraday variant)."""
    if allowed_caps is None:
        allowed_caps = ALLOWED_CAPS
    # Trading-day index from 5m feathers
    distinct_5m_days = sorted(big5m["d"].unique()) if not big5m.empty else []
    distinct_daily_days = sorted(daily["d"].unique()) if not daily.empty else []
    # Use 5m days for intraday entry resolution (must have 09:30 bar)
    # but fall back to daily days if 5m sparse

    rows = []
    for (recalc_d, regime, quarter) in RECALC_DATES:
        # T+1 = first trading day strictly AFTER recalc_d
        t1_5m = _next_trading_day(distinct_5m_days, recalc_d)
        t1_daily = _next_trading_day(distinct_daily_days, recalc_d)
        # We'll use the 5m T+1 for intraday and the daily T+1 for swing; they
        # should normally coincide. If they differ, log a warning.
        if t1_5m is not None and t1_daily is not None and t1_5m != t1_daily:
            print(f"  WARN: recalc {recalc_d} T+1 mismatch 5m={t1_5m} daily={t1_daily}")
        # Prefer 5m for entry_date (intraday entry needs 5m); swing also uses
        # this date as entry (the open price comes from daily that matches).
        entry_d = t1_5m if t1_5m is not None else t1_daily
        if entry_d is None:
            print(f"  WARN: no T+1 trading day after {recalc_d}; skipping recalc.")
            continue

        # Per recalc, walk the universe
        for sym in fno_universe:
            cap = get_cap_segment("NSE:" + sym)
            if cap not in allowed_caps:
                continue
            if apply_mis_filter and mis_allowed and sym not in mis_allowed:
                continue
            rows.append({
                "recalc_date": recalc_d,
                "regime":      regime,
                "quarter":     quarter,
                "symbol":      sym,
                "cap_segment": cap,
                "entry_date":  entry_d,
            })

    return pd.DataFrame(rows)


# ============================================================================
# Reporting
# ============================================================================


def _metrics(trades: pd.DataFrame, pnl_col: str = "net_pnl") -> dict:
    if trades.empty:
        return dict(n=0, pf=float("nan"), wr=float("nan"),
                    sharpe=float("nan"), gross=0.0, fee=0.0, net=0.0)
    npnl = trades[pnl_col].dropna()
    if npnl.empty:
        return dict(n=0, pf=float("nan"), wr=float("nan"),
                    sharpe=float("nan"), gross=0.0, fee=0.0, net=0.0)
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    wr = float((npnl > 0).mean()) * 100.0
    if "session_date" in trades.columns:
        daily = trades.groupby("session_date")[pnl_col].sum()
    else:
        daily = npnl
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    gross = float(trades["realized_pnl"].sum()) if "realized_pnl" in trades.columns else 0.0
    fee = float(trades["fee"].sum()) if "fee" in trades.columns else 0.0
    return dict(
        n=int(len(npnl)),
        pf=pf,
        wr=wr,
        sharpe=sharpe,
        gross=gross,
        fee=fee,
        net=float(npnl.sum()),
    )


def _print_block(label: str, m: dict) -> None:
    print(f"\n--- {label} ---")
    if m["n"] == 0:
        print("  n=0 (no trades)")
        return
    print(f"  n          = {m['n']}")
    print(f"  WR         = {m['wr']:.1f}%")
    print(f"  Gross PnL  = Rs.{int(m['gross']):,}")
    print(f"  Fees       = Rs.{int(m['fee']):,}")
    print(f"  NET PnL    = Rs.{int(m['net']):,}")
    print(f"  NET PF     = {m['pf']:.3f}")
    print(f"  NET Sharpe = {m['sharpe']:.3f}")


def _verdict_gauntlet(m: dict, stability_ratio: Optional[float]) -> str:
    if m["n"] == 0:
        return "RETIRE: n=0"
    if m["n"] < N_MIN:
        return f"RETIRE: n={m['n']} < {N_MIN} (sample-size floor)"
    if m["pf"] < PF_MIN:
        return f"RETIRE: PF={m['pf']:.3f} < {PF_MIN}"
    if m["sharpe"] <= SHARPE_MIN:
        return f"RETIRE: Sharpe={m['sharpe']:.3f} <= {SHARPE_MIN}"
    if stability_ratio is not None and stability_ratio > PER_MONTH_STABILITY_MAX_RATIO:
        return (f"RETIRE: per-month stability ratio={stability_ratio:.2f} "
                f"> {PER_MONTH_STABILITY_MAX_RATIO} (one month carries)")
    return (f"STRONG PROCEED: n={m['n']} PF={m['pf']:.3f} "
            f"Sharpe={m['sharpe']:.3f}")


def _stability_ratio(trades: pd.DataFrame) -> Optional[float]:
    """Best-month PF / worst-month PF. Returns None if <2 months with both
    wins and losses present."""
    if trades.empty:
        return None
    t = trades.copy()
    t["month"] = pd.to_datetime(t["session_date"]).dt.to_period("M").astype(str)
    months_pf = {}
    for month, grp in t.groupby("month"):
        npnl = grp["net_pnl"]
        wins = npnl[npnl > 0].sum()
        losses = npnl[npnl < 0].abs().sum()
        if losses > 0 and wins > 0:
            months_pf[month] = wins / losses
    if len(months_pf) < 2:
        return None
    vals = sorted(months_pf.values())
    if vals[0] <= 0:
        return None
    return vals[-1] / vals[0]


def _report_variant(label: str, trades: pd.DataFrame) -> dict:
    print("\n" + "#" * 78)
    print(f"#  VARIANT: {label}")
    print("#" * 78)

    m = _metrics(trades)
    _print_block(f"{label} OVERALL", m)
    stability = _stability_ratio(trades)
    if stability is not None:
        print(f"  per-month stability ratio (best/worst PF): {stability:.2f}")
    else:
        print(f"  per-month stability ratio: n/a (insufficient month coverage)")

    if trades.empty:
        v = _verdict_gauntlet(m, stability)
        print(f"\n  GAUNTLET VERDICT: {v}")
        return {"metrics": m, "verdict": v, "stability": stability}

    # Per cap_segment
    print(f"\n  Per cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        mc = _metrics(grp)
        print(f"    {cap:<12} n={mc['n']:>4} PF={mc['pf']:>6.3f} "
              f"WR={mc['wr']:>5.1f}% netPnL=Rs.{int(mc['net']):>10,}")

    # Per regime (pre_rule / rule_launch / post_rule)
    print(f"\n  Per regime:")
    for reg, grp in trades.groupby("regime"):
        mr = _metrics(grp)
        print(f"    {reg:<14} n={mr['n']:>4} PF={mr['pf']:>6.3f} "
              f"WR={mr['wr']:>5.1f}% netPnL=Rs.{int(mr['net']):>10,}")

    # Per quarter (recalc-cohort)
    print(f"\n  Per recalc quarter (sample size + PF):")
    for q, grp in trades.groupby("quarter"):
        mq = _metrics(grp)
        print(f"    {q:<10} n={mq['n']:>4} PF={mq['pf']:>6.3f} "
              f"WR={mq['wr']:>5.1f}% net=Rs.{int(mq['net']):>9,}")

    # Per month stability table
    print(f"\n  Per-entry-month PF (stability check):")
    tt = trades.copy()
    tt["month"] = pd.to_datetime(tt["session_date"]).dt.to_period("M").astype(str)
    for mo, grp in tt.groupby("month"):
        mm = _metrics(grp)
        print(f"    {mo}  n={mm['n']:>4}  PF={mm['pf']:>6.3f}  "
              f"WR={mm['wr']:>5.1f}%  net=Rs.{int(mm['net']):>9,}")

    # Exit-reason breakdown (intraday only — swing has uniform exit_reason)
    print(f"\n  Exit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        avg = int(grp["net_pnl"].mean())
        print(f"    {rsn:<25} n={len(grp):>4} avg_net=Rs.{avg:>6,}")

    # Oct-1-2025 cliff specifically (regime change recalc)
    oct_1 = trades[trades["recalc_date"] == date(2025, 10, 1)]
    if not oct_1.empty:
        m_oct = _metrics(oct_1)
        print(f"\n  Oct-1-2025 cliff (rule-launch event) ONLY:")
        print(f"    n={m_oct['n']}  PF={m_oct['pf']:.3f}  WR={m_oct['wr']:.1f}%  "
              f"net=Rs.{int(m_oct['net']):,}")
        # Compare against pre_rule + post_rule combined PF
        not_oct = trades[trades["recalc_date"] != date(2025, 10, 1)]
        m_other = _metrics(not_oct)
        print(f"    rest (excl Oct-1-2025) n={m_other['n']}  PF={m_other['pf']:.3f}")
        if m_oct["n"] > 0 and m_other["n"] > 0:
            if m_oct["pf"] > 1.20 and m_other["pf"] < 1.05:
                print(f"    NOTE: Oct-1-2025 cliff PF carries the result; "
                      f"caveat #1 (pre-pricing) appears FALSIFIED — there IS "
                      f"a post-launch reaction.")
            elif m_oct["pf"] < 1.05 and m_other["pf"] < 1.05:
                print(f"    NOTE: Oct-1-2025 cliff shows NO edge — "
                      f"consistent with caveat #1 (pre-pricing by announcement).")
            else:
                print(f"    NOTE: Oct-1-2025 cliff in line with rest — no clear "
                      f"pre-pricing signal.")

    v = _verdict_gauntlet(m, stability)
    print(f"\n  GAUNTLET VERDICT ({label}): {v}")
    return {"metrics": m, "verdict": v, "stability": stability}


# ============================================================================
# Main
# ============================================================================


def main():
    # 1. Pre-flight: surface SEBI rule changes affecting MWPL / FO_speculation
    regime_preflight()

    # 2. Universe + MIS
    print("\n=== universe load ===")
    fno_universe = load_fno_universe()
    mis_allowed = _get_mis_allowed_set()

    # 3. Load 5m bars across all relevant months
    print("\n=== 5m bar load ===")
    big5m = build_window_5m()
    if big5m.empty:
        print("[ABORT] no 5m feathers available for any recalc month")
        return

    # 4. Load daily for swing variant
    print("\n=== daily bar load ===")
    daily = load_production_daily()

    # 5. Build event lists for PRIMARY (mid/small) and SECONDARY (wide) lanes.
    print("\n=== build event list (PRIMARY: mid/small cap per brief) ===")
    events_intraday_pri = build_events(
        big5m, daily, fno_universe, mis_allowed,
        apply_mis_filter=True, allowed_caps=ALLOWED_CAPS,
    )
    events_swing_pri = build_events(
        big5m, daily, fno_universe, mis_allowed,
        apply_mis_filter=False, allowed_caps=ALLOWED_CAPS,
    )
    print(f"  PRIMARY intraday events: {len(events_intraday_pri)}")
    print(f"  PRIMARY swing events:    {len(events_swing_pri)}")

    print("\n=== build event list (SECONDARY: wide — incl. large/unknown) ===")
    events_intraday_wide = build_events(
        big5m, daily, fno_universe, mis_allowed,
        apply_mis_filter=True, allowed_caps=ALLOWED_CAPS_WIDE,
    )
    events_swing_wide = build_events(
        big5m, daily, fno_universe, mis_allowed,
        apply_mis_filter=False, allowed_caps=ALLOWED_CAPS_WIDE,
    )
    print(f"  SECONDARY intraday events: {len(events_intraday_wide)}")
    print(f"  SECONDARY swing events:    {len(events_swing_wide)}")

    # Per-recalc sample size (PRIMARY lane)
    print(f"\n  Per-recalc PRIMARY (mid/small) sample size:")
    for (recalc_d, regime, q) in RECALC_DATES:
        sub = events_intraday_pri[events_intraday_pri["recalc_date"] == recalc_d]
        by_cap = sub["cap_segment"].value_counts().to_dict() if not sub.empty else {}
        print(f"    {recalc_d}  {regime:<12} {q:<10}  n={len(sub):>4}  by_cap={by_cap}")

    # Per-recalc sample size (SECONDARY lane)
    print(f"\n  Per-recalc SECONDARY (wide) sample size:")
    for (recalc_d, regime, q) in RECALC_DATES:
        sub = events_intraday_wide[events_intraday_wide["recalc_date"] == recalc_d]
        by_cap = sub["cap_segment"].value_counts().to_dict() if not sub.empty else {}
        print(f"    {recalc_d}  {regime:<12} {q:<10}  n={len(sub):>4}  by_cap={by_cap}")

    if events_intraday_pri.empty and events_intraday_wide.empty:
        print("[ABORT] no events to simulate")
        return

    # 6. Run simulators for both lanes
    print("\n=== INTRADAY simulator — PRIMARY (mid/small) ===")
    trades_intraday_pri = simulate_intraday(events_intraday_pri, big5m)
    print("\n=== INTRADAY simulator — SECONDARY (wide) ===")
    trades_intraday_wide = simulate_intraday(events_intraday_wide, big5m)

    if daily.empty:
        print("\n[WARN] swing variant skipped (no daily feather)")
        trades_swing_pri = pd.DataFrame()
        trades_swing_wide = pd.DataFrame()
    else:
        print("\n=== SWING simulator — PRIMARY (mid/small) ===")
        trades_swing_pri = simulate_swing(events_swing_pri, daily)
        print("\n=== SWING simulator — SECONDARY (wide) ===")
        trades_swing_wide = simulate_swing(events_swing_wide, daily)

    # 7. Report PRIMARY lane (brief-locked: mid/small-cap)
    print("\n\n" + "*" * 78)
    print("*  PRIMARY LANE — mid/small-cap only (per brief §Cells)")
    print("*" * 78)
    intraday_pri_result = _report_variant("INTRADAY-PRIMARY", trades_intraday_pri)
    swing_pri_result = _report_variant("SWING-PRIMARY", trades_swing_pri)

    # 8. Report SECONDARY lane (wide — to escape sample-floor)
    print("\n\n" + "*" * 78)
    print("*  SECONDARY LANE — wide universe (large+mid+small+unknown)")
    print("*  (escapes the F&O-200-large-cap-skew sample-size trap; reports")
    print("*   what we'd see if the brief's strict mid/small filter weren't applied)")
    print("*" * 78)
    intraday_wide_result = _report_variant("INTRADAY-WIDE", trades_intraday_wide)
    swing_wide_result = _report_variant("SWING-WIDE", trades_swing_wide)

    # 9. Final summary
    print("\n\n" + "=" * 78)
    print("FINAL SUMMARY — mwpl_recalc_forced_rebalance_fade (C1)")
    print("=" * 78)
    print(f"\n  Ship gates: n >= {N_MIN}, PF >= {PF_MIN}, Sharpe > {SHARPE_MIN}, "
          f"per-month stability <= {PER_MONTH_STABILITY_MAX_RATIO}x\n")
    rows = [
        ("INTRADAY-PRIMARY", intraday_pri_result),
        ("SWING-PRIMARY",    swing_pri_result),
        ("INTRADAY-WIDE",    intraday_wide_result),
        ("SWING-WIDE",       swing_wide_result),
    ]
    for name, r in rows:
        m = r["metrics"]
        s = r["stability"]
        s_str = f"{s:.2f}" if s is not None else "n/a"
        print(f"  {name:<18}  n={m['n']:>5}  PF={m['pf']:>6.3f}  "
              f"WR={m['wr']:>5.1f}%  Sharpe={m['sharpe']:>6.3f}  "
              f"stability={s_str:<6}  => {r['verdict']}")

    # 10. Persist trades CSV (combined; tag lane via cap_segment naturally)
    all_trades_parts = []
    for df, lane in (
        (trades_intraday_pri, "primary"),
        (trades_swing_pri, "primary"),
        (trades_intraday_wide, "wide"),
        (trades_swing_wide, "wide"),
    ):
        if not df.empty:
            df = df.copy()
            df["lane"] = lane
            all_trades_parts.append(df)
    all_trades = (pd.concat(all_trades_parts, ignore_index=True)
                  if all_trades_parts else pd.DataFrame())
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mwpl_recalc_drift_short_trades.csv"
    if not all_trades.empty:
        all_trades.to_csv(out_path, index=False)
        print(f"\nFull trade log: {out_path}  ({len(all_trades)} rows)")
    else:
        print("\n[no trades persisted]")


if __name__ == "__main__":
    main()
