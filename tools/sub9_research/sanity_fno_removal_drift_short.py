"""Pre-coding sanity check for fno_removal_drift_short candidate (Sub-9 Candidate 3).

Per brief: specs/2026-05-14-brief-fno_removal_drift_short.md

Mechanism (one sentence, brief §"Mechanism" line 9):
  NSE FAOP* circulars naming single stocks failing SEBI Aug-30-2024 entry/exit
  criteria trigger forced cash-side selling from arb desks / option MMs / F&O-
  mandate funds; this absorbs concentrated supply on T+0 (announcement day),
  producing a predictable negative drift between 09:30 open and 15:10 close.
  SHORT-only by mechanism (no symmetric "buy on removal" institutional flow).

Cells (brief §"Cells (pre-registered)" lines 89-130):

  Cell A — Primary (single-entry MIS):
    - Universe: any cap_segment EXCEPT micro_cap; T-30 ADV >= Rs.5 cr
    - Entry: T+0 09:30 IST 5m bar close (first stable bar post-opening-auction)
    - Direction: SHORT
    - Confirmation gates at 09:30:
        * 5m bar close < open (red bar)
        * Vol on 09:15-09:30 bars > 1.5x avg 30-day 5m volume
    - Hard SL: min(entry * 1.015, T+0 day-high-so-far * 1.005), floor entry * 1.01
    - T1 (50% qty): entry * 0.99
    - T2 (50% qty): entry * 0.98
    - Time stop: 15:10 IST
    - BE trail: after T1 fills, runner SL -> entry_price
    - Latch: one fire per (symbol, circular_date)

  Cell B — Secondary (T-1 pre-announcement, exploratory):
    - Same as Cell A but entry on trading day BEFORE circular_date
    - Reports independently; informational only (public-computability research)

Option A / B parallel run (brief §"Risks / open questions" #1, lines 244-275):
  Option A: 2024-09-13 -> 2026-05-14 (wide post-Aug-30-2024-norms window, ~49 events)
  Option B: 2025-10-01 -> 2026-05-14 (strict post-Oct-1-2025 MWPL window, ~9-13 events)
  Both options run AND report independently. Decision per §"Sanity decision".

Falsification thresholds (brief §"Sanity decision" lines 226-230):
  PF >= 1.30 AND n >= 20 -> STRONG PROCEED
  PF in [1.10, 1.30) AND n >= 20 -> MARGINAL (extend to swing variant as side-research)
  PF < 1.10 OR n < 20 -> RETIRE

Side-research multi-day swing (brief §"Risks" #2):
  Hypothetical T+0 entry -> T+5 close exit, no extra fees beyond entry+exit
  (assume CNC). Reported informationally only — DOES NOT ship under this brief
  per falsifier #7 (real-edge-wrong-infra).

Regime-break pre-flight:
  Calls services.regime_break_detector.check_window() with depends_on tags
  ["single_stock_FO", "F&O_speculation"]. Option A (2024-09-13 start) is post-
  Aug-30-2024 norms by construction, so the norm-change row at 2024-09-13 itself
  is the earliest comparable sample point (not a regime break for this setup).
  Critical rows that DO fall inside Option A's window (2025-10-01 MWPL tightening
  and 2026-04-01 STT hike) are surfaced informationally — the brief explicitly
  treats them as sub-window splits, not aborts. We do NOT raise on these.

STUB MODE:
  If data/fno_eligibility/removals_2024_2026.csv does not exist, emits a
  hardcoded sample event list (Dec 2024 + 2 post-Oct-2025) drawn from the
  brief's "Total verifiable removal events" table. Replace with the curated
  CSV before any actual Discovery run.

Outputs:
  - reports/sub9_sanity/fno_removal_drift_short_trades.csv (one row per trade,
    cell-mine compatible: net_pnl / cap_segment / side / session_date)
  - stdout summary: Option A + Option B + per-STT sub-window + per-cap-segment
    + per-gap-days bucket + side-research multi-day PF
  - Per-cell verdict against pre-registered thresholds

Usage:
    python -m tools.sub9_research.sanity_fno_removal_drift_short
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

from services.symbol_metadata import get_cap_segment                  # noqa: E402
from services.regime_break_detector import (                          # noqa: E402
    check_window, GauntletRegimeBreak,
)
from tools.sub7_validation.build_per_setup_pnl import calc_fee        # noqa: E402


# ============================================================================
# Brief-locked params (§"Cells (pre-registered)")
# ============================================================================

# Cell A primary — universe
EXCLUDED_CAPS = {"micro_cap"}              # all OTHER caps admissible
MIN_T30_ADV_RUPEES = 5e7                   # Rs.5 cr cash-market T-30 ADV
ADV_LOOKBACK_DAYS = 30

# Entry / exits
ENTRY_HHMM = "09:30"                       # bar that closes at 09:30
VOL_WINDOW_START_HHMM = "09:15"
VOL_WINDOW_END_HHMM = "09:30"              # inclusive (09:15-09:20-09:25-09:30)
MIN_VOL_RATIO_30D_5M = 1.5                 # vol on 09:15-09:30 > 1.5x 30d-avg 5m vol

# Hard SL: min(entry * 1.015, T+0 day-high-so-far * 1.005), floor entry * 1.01
SL_ENTRY_CAP_PCT = 1.5                     # entry * 1.015 — UPPER cap
SL_DAYHIGH_BUFFER_PCT = 0.5                # day-high-so-far * 1.005
SL_FLOOR_PCT = 1.0                         # entry * 1.01 — LOWER floor (min stop dist)

T1_TARGET_PCT = 1.0                        # entry * 0.99
T2_TARGET_PCT = 2.0                        # entry * 0.98
T1_QTY_FRAC = 0.5                          # 50% qty at T1
USE_BE_TRAIL_AFTER_T1 = True               # runner SL -> entry after T1 fills

TIME_STOP_HHMM = "15:10"
RISK_PER_TRADE_RUPEES = 1000

# Side-research multi-day exit (brief §"Risks" #2)
MULTI_DAY_HOLD_DAYS = 5                    # T+0 entry -> T+5 close exit

# ============================================================================
# Option A / B windows (brief §"Risks" #1)
# ============================================================================

OPTION_A_START = date(2024, 9, 13)         # first post-Aug-30-2024-norms removal
OPTION_A_END   = date(2026, 5, 14)         # today

OPTION_B_START = date(2025, 10, 1)         # post-MWPL tightening
OPTION_B_END   = date(2026, 5, 14)

# STT-hike sub-window boundary (brief §"Falsification" #8)
STT_HIKE_DATE  = date(2026, 4, 1)          # Budget 2026 STT hike effective

# ============================================================================
# Decision thresholds (brief §"Sanity decision")
# ============================================================================

PF_STRONG_PROCEED = 1.30
PF_MARGINAL_FLOOR = 1.10
N_MIN_FLOOR = 20

# ============================================================================
# Regime-break pre-flight
# ============================================================================

DEPENDS_ON = ["single_stock_FO", "F&O_speculation"]


def regime_preflight() -> None:
    """Surface regime-break rows inside both sanity windows; do NOT raise.

    Per brief §"Regulatory dependencies" lines 56-59 + §"Risks" #1 lines
    254-258: critical rule rows inside Option A's window (2025-10-01 MWPL,
    2026-04-01 STT) are NOT regime breaks for THIS setup — they're sub-
    window split boundaries reported separately. We surface them at
    info level, then proceed.
    """
    print("\n=== regime_break_detector pre-flight ===")
    print(f"  strategy:    fno_removal_drift_short")
    print(f"  depends_on:  {DEPENDS_ON}")

    for label, start, end in [
        ("Option_A_wide",   OPTION_A_START, OPTION_A_END),
        ("Option_B_strict", OPTION_B_START, OPTION_B_END),
    ]:
        print(f"\n  window: {label}  {start} .. {end}")
        hits = check_window(
            strategy_name="fno_removal_drift_short",
            depends_on=DEPENDS_ON,
            window_label=label,
            start=start,
            end=end,
            min_severity="low",
            raise_on_break=False,
        )
        if not hits:
            print("    no rule rows in window (clean).")
            continue
        for r in hits:
            # Description may contain non-CP1252 chars (arrows etc.); sanitize
            # for Windows console which defaults to cp1252.
            desc = (r.description[:90]
                    .encode("ascii", errors="replace").decode("ascii"))
            print(f"    {r.effective_date} [{r.severity.upper():<8}] {desc}")
        # Per brief: the 2025-10-01 / 2026-04-01 criticals are sub-window
        # split boundaries, not regime breaks. Surface; do not raise.
        crit = [r for r in hits if r.severity == "critical"]
        if crit:
            print(f"    note: {len(crit)} critical row(s) — handled via "
                  f"pre/post-STT and pre/post-Oct-2025 sub-window splits "
                  f"in the report, not by aborting.")


# ============================================================================
# Event-source loader (real CSV or STUB)
# ============================================================================

_REMOVALS_CSV = _REPO_ROOT / "data" / "fno_eligibility" / "removals_2024_2026.csv"


def _stub_events() -> pd.DataFrame:
    """Hardcoded sample events for STUB MODE (CSV being curated in parallel).

    Picks ONE early-Option-A event (Dec 2024 circular for Feb-2025 removal)
    plus TWO post-Oct-2025 events (Option B). Symbols from brief lines
    166-184 verifiable removals table. Just enough to exercise the
    simulator end-to-end without the curated CSV.
    """
    print("  STUB MODE: removals CSV not yet curated — using 5 sample events.")
    print(f"    expected real CSV at: {_REMOVALS_CSV}")
    rows = [
        # (circular_date, effective_date, symbol, circular_ref, gap_days, is_post_oct_1_2025)
        # Dec-2024 batch (pre-Oct-2025; Option A only)
        ("2024-12-23", "2025-02-28", "INDIAMART",  "FAOP65702", 67, False),
        ("2024-12-23", "2025-02-28", "PVRINOX",    "FAOP65702", 67, False),
        ("2024-12-23", "2025-02-28", "SUNTV",      "FAOP65702", 67, False),
        # Post-Oct-2025 events (Option A AND Option B)
        ("2025-10-24", "2026-01-30", "HFCL",       "FAOP-NSE-2025-10-24", 98, True),
        ("2025-10-24", "2026-01-30", "CYIENT",     "FAOP-NSE-2025-10-24", 98, True),
    ]
    df = pd.DataFrame(
        rows,
        columns=["circular_date", "effective_date", "symbol",
                 "circular_ref", "gap_days", "is_post_oct_1_2025"],
    )
    df["circular_date"] = pd.to_datetime(df["circular_date"]).dt.date
    df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
    return df


def load_removals_events() -> pd.DataFrame:
    """Load the curated removals CSV if it exists; otherwise emit stub events.

    Schema: [circular_date, effective_date, symbol, circular_ref, gap_days,
             is_post_oct_1_2025].
    """
    if _REMOVALS_CSV.exists():
        print(f"  loading removals CSV: {_REMOVALS_CSV}")
        df = pd.read_csv(_REMOVALS_CSV)
        df["circular_date"] = pd.to_datetime(df["circular_date"]).dt.date
        df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
        if "is_post_oct_1_2025" in df.columns:
            df["is_post_oct_1_2025"] = df["is_post_oct_1_2025"].astype(bool)
        else:
            df["is_post_oct_1_2025"] = df["circular_date"] >= date(2025, 10, 1)
        if "gap_days" not in df.columns:
            df["gap_days"] = (
                pd.to_datetime(df["effective_date"]) -
                pd.to_datetime(df["circular_date"])
            ).dt.days
        return df.reset_index(drop=True)
    return _stub_events()


# ============================================================================
# 5m + daily feather loading
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


def _months_to_load(events: pd.DataFrame) -> List[Tuple[int, int]]:
    """Load the circular month, T-1's month (for Cell B), and a 1-month
    forward lookahead (for multi-day side-research at T+5)."""
    months: set = set()
    for d in events["circular_date"]:
        if isinstance(d, (datetime, pd.Timestamp)):
            d = d.date() if hasattr(d, "date") else d
        # circular month
        months.add((d.year, d.month))
        # previous month (Cell B T-1 might straddle month boundary)
        prev_y = d.year - (1 if d.month == 1 else 0)
        prev_m = 12 if d.month == 1 else d.month - 1
        months.add((prev_y, prev_m))
        # next month (T+5 lookahead for side-research)
        nxt_m = (d.month % 12) + 1
        nxt_y = d.year + (1 if d.month == 12 else 0)
        months.add((nxt_y, nxt_m))
    return sorted(months)


def build_window_5m(events: pd.DataFrame) -> pd.DataFrame:
    months = _months_to_load(events)
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
    """Load 1day OHLCV (for T-30 ADV + multi-day side-research T+5 close)."""
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        print(f"  WARN: {daily_path} missing; ADV gate + multi-day will skip.")
        return pd.DataFrame()
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[["symbol", "d", "open", "high", "low", "close", "volume"]].copy()
    return df


# ============================================================================
# Context: cap_segment, ADV, 30-day 5m volume average
# ============================================================================


def _compute_adv_t30(daily: pd.DataFrame, sym: str, ref_date: date) -> float:
    """Cash-market T-30 ADV in rupees (close * volume avg, lookback 30 trading days
    BEFORE ref_date). NaN if insufficient history."""
    g = daily[(daily["symbol"] == sym) & (daily["d"] < ref_date)]
    if len(g) < ADV_LOOKBACK_DAYS:
        return float("nan")
    g = g.sort_values("d").tail(ADV_LOOKBACK_DAYS)
    return float((g["close"] * g["volume"]).mean())


def _compute_5m_vol_30d_avg(
    big5m: pd.DataFrame,
    sym: str,
    ref_date: date,
) -> float:
    """Average 5m bar volume across 30 trading days BEFORE ref_date."""
    g = big5m[(big5m["symbol"] == sym) & (big5m["d"] < ref_date)]
    if g.empty:
        return float("nan")
    # Use the 30 most-recent trading days BEFORE ref_date
    distinct_days = sorted(g["d"].unique())
    if len(distinct_days) < 5:
        return float("nan")
    lookback = distinct_days[-ADV_LOOKBACK_DAYS:]
    sub = g[g["d"].isin(lookback)]
    if sub.empty:
        return float("nan")
    return float(sub["volume"].mean())


def attach_context(
    events: pd.DataFrame,
    daily: pd.DataFrame,
    big5m: pd.DataFrame,
) -> pd.DataFrame:
    """Attach cap_segment, T-30 ADV (rupees), and 30-day avg 5m volume.

    Cell B (T-1 entry) uses the same ADV / 30d-vol — they're stock-level
    properties, not date-specific. We compute relative to circular_date.
    """
    out = events.copy()
    out["cap_segment"] = out["symbol"].apply(
        lambda s: get_cap_segment("NSE:" + s)
    )
    advs: List[float] = []
    vol30s: List[float] = []
    for _, r in out.iterrows():
        sym = r["symbol"]
        cdate = r["circular_date"]
        if not daily.empty:
            advs.append(_compute_adv_t30(daily, sym, cdate))
        else:
            advs.append(float("nan"))
        vol30s.append(_compute_5m_vol_30d_avg(big5m, sym, cdate))
    out["adv_t30_rupees"] = advs
    out["vol_30d_avg_5m"] = vol30s
    return out


# ============================================================================
# Cell A universe filter
# ============================================================================


def apply_universe_filter(events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """cap_segment NOT in EXCLUDED_CAPS + T-30 ADV >= 5cr."""
    funnel = {"in_events": len(events)}
    df = events.copy()

    df = df[~df["cap_segment"].isin(EXCLUDED_CAPS)]
    funnel["cap_admissible"] = len(df)

    df = df[(df["adv_t30_rupees"].isna()) |
            (df["adv_t30_rupees"] >= MIN_T30_ADV_RUPEES)]
    funnel["adv_admissible"] = len(df)

    return {"kept": df.reset_index(drop=True),
            "funnel": pd.DataFrame([funnel])}


# ============================================================================
# Bar-walk simulator (mirrors C2 fno_ban_entry_t1_fade exactly)
# ============================================================================


def _resolve_entry_day(
    sym_5m: pd.DataFrame,
    target_d: date,
    direction: str,  # 'on' (Cell A: T+0 = circular_date) | 'prev' (Cell B: T-1)
) -> Optional[date]:
    """Find actual trading day matching target. For Cell A, use circular_date
    if it has bars; else next trading day with bars. For Cell B, use the
    LAST trading day BEFORE circular_date.
    """
    if sym_5m.empty:
        return None
    distinct_days = sorted(sym_5m["d"].unique())
    if direction == "on":
        # circular_date itself (if NSE was open) else next trading day
        future = [d for d in distinct_days if d >= target_d]
        return future[0] if future else None
    else:  # 'prev'
        past = [d for d in distinct_days if d < target_d]
        return past[-1] if past else None


def simulate_short(
    events: pd.DataFrame,
    big5m: pd.DataFrame,
    daily: pd.DataFrame,
    cell_label: str,        # 'A' (T+0) or 'B' (T-1)
) -> Dict:
    """Run the bar-walk SHORT entry simulator over `events`.

    Mirrors sanity_fno_ban_entry_t1_fade simulator structure:
      active_sl = entry_price if t1_hit else hard_sl
      T1 partial -> runner with BE trail
      Exit reasons: stop / breakeven_trail / t2 / time_stop / eod_fallback
    """
    trades: List[dict] = []
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date") for sym, g in big5m.groupby("symbol")
    }

    n_no_data = n_no_entry_bar = n_not_red = n_vol_fail = n_zero_stop = 0
    n_traded = 0

    for _, ev in events.iterrows():
        sym = ev["symbol"]
        c_date = ev["circular_date"]
        sym_df = days_per_sym.get(sym)
        if sym_df is None or sym_df.empty:
            n_no_data += 1
            continue

        if cell_label == "A":
            entry_d = _resolve_entry_day(sym_df, c_date, "on")
        else:  # 'B'
            entry_d = _resolve_entry_day(sym_df, c_date, "prev")
        if entry_d is None:
            n_no_data += 1
            continue

        day_df = (sym_df[sym_df["d"] == entry_d]
                  .sort_values("date").reset_index(drop=True))
        if day_df.empty:
            n_no_data += 1
            continue

        # Entry bar = 09:30 bar close
        entry_rows = day_df[day_df["hhmm"] == ENTRY_HHMM]
        if entry_rows.empty:
            n_no_entry_bar += 1
            continue
        entry_bar = entry_rows.iloc[0]
        entry_idx = entry_bar.name
        entry_ts = entry_bar["date"]
        entry_price = float(entry_bar["close"])
        entry_open = float(entry_bar["open"])

        # Gate 1: red bar (close < open)
        if entry_price >= entry_open:
            n_not_red += 1
            continue

        # Gate 2: vol(09:15-09:30) > 1.5x 30d-avg 5m volume
        vol_window = day_df[(day_df["hhmm"] >= VOL_WINDOW_START_HHMM) &
                            (day_df["hhmm"] <= VOL_WINDOW_END_HHMM)]
        if vol_window.empty:
            n_no_entry_bar += 1
            continue
        vol_window_total = float(vol_window["volume"].sum())
        vol_window_avg = vol_window_total / max(len(vol_window), 1)
        vol_30d_avg = float(ev.get("vol_30d_avg_5m", float("nan")))
        if pd.isna(vol_30d_avg) or vol_30d_avg <= 0:
            # Cannot verify — fail closed (skip).
            n_vol_fail += 1
            continue
        vol_ratio = vol_window_avg / vol_30d_avg
        if vol_ratio < MIN_VOL_RATIO_30D_5M:
            n_vol_fail += 1
            continue

        # Hard SL: min(entry * 1.015, day-high-so-far * 1.005), floor entry * 1.01
        # day-high-so-far at 09:30 = max(high) across 09:15-09:30 bars (inclusive)
        day_high_so_far = float(vol_window["high"].max())
        sl_cap = entry_price * (1.0 + SL_ENTRY_CAP_PCT / 100.0)
        sl_daily = day_high_so_far * (1.0 + SL_DAYHIGH_BUFFER_PCT / 100.0)
        sl_floor = entry_price * (1.0 + SL_FLOOR_PCT / 100.0)
        # Brief: min(entry*1.015, day-high*1.005), THEN floor at entry*1.01
        candidate = min(sl_cap, sl_daily)
        hard_sl = max(candidate, sl_floor)
        stop_distance = hard_sl - entry_price
        if stop_distance <= 0:
            n_zero_stop += 1
            continue

        t1_target = entry_price * (1.0 - T1_TARGET_PCT / 100.0)
        t2_target = entry_price * (1.0 - T2_TARGET_PCT / 100.0)

        # Walk forward from bar AFTER entry to 15:10 time stop
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

            # BE trail after T1 fills (SHORT: SL -> entry_price)
            active_sl = entry_price if (hit_t1 and USE_BE_TRAIL_AFTER_T1) else hard_sl

            # SHORT: SL is ABOVE entry; check first (worst-case)
            if hi >= active_sl:
                exit_ts = ts
                exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break

            # T1 partial (50% qty)
            if (not hit_t1) and (lo <= t1_target):
                hit_t1 = True
                t1_exit_price = t1_target

            # T2 full close on runner
            if lo <= t2_target:
                exit_ts = ts
                exit_price = t2_target
                exit_reason = "t2"
                break

            # Time stop 15:10
            if hhmm >= TIME_STOP_HHMM:
                exit_ts = ts
                exit_price = cl
                exit_reason = "time_stop"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "last_bar"

        # Position sizing + PnL
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
            # Refine exit reason to surface T1 partial story
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

        # Multi-day side-research (T+0 entry -> T+MULTI_DAY_HOLD_DAYS close,
        # CNC, no extra fees beyond entry+exit — same legs we already paid).
        # Use the daily feather close on the Nth-future-trading-day.
        multi_day_net = float("nan")
        multi_day_pct = float("nan")
        if not daily.empty:
            d_sym = daily[daily["symbol"] == sym].sort_values("d")
            future_d = d_sym[d_sym["d"] > entry_d]
            if len(future_d) >= MULTI_DAY_HOLD_DAYS:
                exit_daily = future_d.iloc[MULTI_DAY_HOLD_DAYS - 1]
                md_exit = float(exit_daily["close"])
                # Multi-day uses FULL qty (no T1 partial — single CNC short hypothesis)
                md_realized = (entry_price - md_exit) * qty
                md_fee = calc_fee(entry_price, md_exit, qty, "SELL")
                multi_day_net = md_realized - md_fee
                multi_day_pct = (entry_price / md_exit - 1.0) * 100.0

        trades.append({
            "cell":              cell_label,
            "circular_date":     c_date,
            "effective_date":    ev["effective_date"],
            "session_date":      entry_d,
            "T1_entry_date":     entry_d,   # compat alias for cell-mine tools
            "gap_days":          int(ev.get("gap_days", -1) or -1),
            "is_post_oct_1_2025": bool(ev.get("is_post_oct_1_2025", False)),
            "symbol":            "NSE:" + sym,
            "bare_symbol":       sym,
            "cap_segment":       ev["cap_segment"],
            "adv_t30_rupees":    float(ev.get("adv_t30_rupees") or float("nan")),
            "vol_30d_avg_5m":    vol_30d_avg,
            "vol_window_avg_5m": vol_window_avg,
            "vol_ratio":         float(vol_ratio),
            "side":              "SHORT",
            "entry_ts":          entry_ts,
            "entry_price":       entry_price,
            "entry_open":        entry_open,
            "day_high_so_far":   day_high_so_far,
            "hard_sl":           hard_sl,
            "t1_target":         t1_target,
            "t2_target":         t2_target,
            "stop_distance":     stop_distance,
            "hit_t1":            hit_t1,
            "exit_ts":           exit_ts,
            "exit_price":        blended_exit,
            "exit_reason":       exit_reason,
            "qty":               qty,
            "realized_pnl":      realized_pnl,
            "fee":               fee,
            "net_pnl":           net_pnl,
            "multi_day_net":     multi_day_net,
            "multi_day_pct":     multi_day_pct,
            "circular_ref":      ev.get("circular_ref", ""),
        })
        n_traded += 1

    print(f"\n  [Cell {cell_label}] simulator counters:")
    print(f"    no 5m data:                  {n_no_data}")
    print(f"    no 09:30 entry bar:          {n_no_entry_bar}")
    print(f"    09:30 bar not RED:           {n_not_red}")
    print(f"    vol < 1.5x 30d avg:          {n_vol_fail}")
    print(f"    zero/neg stop distance:      {n_zero_stop}")
    print(f"    traded:                      {n_traded}")

    return {
        "trades": pd.DataFrame(trades),
        "n_no_data": n_no_data,
        "n_no_entry_bar": n_no_entry_bar,
        "n_not_red": n_not_red,
        "n_vol_fail": n_vol_fail,
        "n_zero_stop": n_zero_stop,
        "n_traded": n_traded,
    }


# ============================================================================
# Metrics + reporting
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
    sharpe = (float(daily.mean() / daily.std())
              if daily.std() > 0 else 0.0)
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


def _verdict(m: dict) -> str:
    """Per brief §"Sanity decision" lines 226-230."""
    if m["n"] == 0:
        return "RETIRE: n=0 (no trades produced)"
    if m["n"] < N_MIN_FLOOR:
        return f"RETIRE: n={m['n']} < {N_MIN_FLOOR} sample-size floor"
    if m["pf"] < PF_MARGINAL_FLOOR:
        return f"RETIRE: PF={m['pf']:.3f} < {PF_MARGINAL_FLOOR}"
    if m["pf"] >= PF_STRONG_PROCEED:
        return (f"STRONG PROCEED: PF={m['pf']:.3f} >= {PF_STRONG_PROCEED} "
                f"AND n={m['n']} >= {N_MIN_FLOOR}")
    return (f"MARGINAL: PF={m['pf']:.3f} in [{PF_MARGINAL_FLOOR},"
            f" {PF_STRONG_PROCEED}); extend to swing variant as side-research")


def _report_option(
    label: str,
    trades: pd.DataFrame,
    win_start: date,
    win_end: date,
) -> dict:
    """Report a single option-window cell. Returns metrics + verdict."""
    print("\n" + "=" * 78)
    print(f"OPTION {label}  window {win_start} .. {win_end}")
    print("=" * 78)
    if trades.empty:
        print("  n=0 (no trades in window)")
        m = _metrics(trades)
        v = _verdict(m)
        print(f"  VERDICT: {v}")
        return {"metrics": m, "verdict": v, "trades": trades}

    sub = trades[(trades["session_date"] >= win_start) &
                 (trades["session_date"] <= win_end)]
    m = _metrics(sub)
    _print_block(f"Option {label} (n={m['n']})", m)
    v = _verdict(m)
    print(f"  VERDICT: {v}")

    # Pre-STT vs post-STT sub-windows
    pre  = sub[sub["session_date"] <  STT_HIKE_DATE]
    post = sub[sub["session_date"] >= STT_HIKE_DATE]
    _print_block(f"  pre-Apr-2026-STT  ({win_start}..{STT_HIKE_DATE - timedelta(days=1)})",
                 _metrics(pre))
    _print_block(f"  post-Apr-2026-STT ({STT_HIKE_DATE}..{win_end})",
                 _metrics(post))

    # Per cap_segment
    if not sub.empty:
        print("\n  Per cap_segment:")
        for cap, grp in sub.groupby("cap_segment"):
            mc = _metrics(grp)
            print(f"    {cap:<12} n={mc['n']:>4} PF={mc['pf']:>6.3f} "
                  f"WR={mc['wr']:>5.1f}% netPnL=Rs.{int(mc['net']):>10,}")

    # Per gap_days bucket
    if "gap_days" in sub.columns and not sub.empty:
        print("\n  Per gap_days bucket:")
        # Buckets: <=30, 31-60, 61-90, >90
        def _bucket(g):
            if g <= 30: return "0-30d"
            if g <= 60: return "31-60d"
            if g <= 90: return "61-90d"
            return ">90d"
        gb = sub.copy()
        gb["gap_bucket"] = gb["gap_days"].apply(_bucket)
        for bk, grp in gb.groupby("gap_bucket"):
            mb = _metrics(grp)
            print(f"    {bk:<8} n={mb['n']:>4} PF={mb['pf']:>6.3f} "
                  f"WR={mb['wr']:>5.1f}% netPnL=Rs.{int(mb['net']):>10,}")

    # Exit-reason breakdown
    if not sub.empty:
        print("\n  Exit-reason breakdown:")
        for rsn, grp in sub.groupby("exit_reason"):
            avg = int(grp["net_pnl"].mean())
            print(f"    {rsn:<25} n={len(grp):>4} avg_net=Rs.{avg:>6,}")

    # Side-research multi-day PF (informational only)
    if "multi_day_net" in sub.columns:
        md = sub.dropna(subset=["multi_day_net"])
        if not md.empty:
            mm = _metrics(md, pnl_col="multi_day_net")
            print(f"\n  SIDE-RESEARCH (multi-day T+0 -> T+{MULTI_DAY_HOLD_DAYS} close, CNC):")
            print(f"    n={mm['n']}  PF={mm['pf']:.3f}  WR={mm['wr']:.1f}%  "
                  f"net=Rs.{int(mm['net']):,}")
            print(f"    (informational only; does NOT ship under this brief — falsifier #7)")

    return {"metrics": m, "verdict": v, "trades": sub}


# ============================================================================
# Main
# ============================================================================


def main():
    # 1. Regime-break pre-flight (surface, do not abort — sub-window splits per brief).
    regime_preflight()

    # 2. Load removal events (real curated CSV or STUB MODE).
    print("\n=== removal events ===")
    events_all = load_removals_events()
    print(f"  loaded {len(events_all)} events")
    if events_all.empty:
        print("[ABORT] no events to process")
        return

    # 3. Load daily for ADV + multi-day side-research.
    print("\n=== daily context ===")
    daily = load_production_daily()
    if daily.empty:
        print("  WARN: ADV gate disabled (no daily feather); multi-day skipped.")

    # 4. Load 5m bars covering all event months + lookback + lookahead.
    print("\n=== 5m bars ===")
    big5m = build_window_5m(events_all)
    if big5m.empty:
        print("[ABORT] no 5m feathers available for event months")
        return

    # 5. Attach cap / ADV / 30d-vol context.
    print("\n=== context (cap + T-30 ADV + 30d 5m vol) ===")
    events_ctx = attach_context(events_all, daily, big5m)

    # 6. Apply universe filter (cap + ADV).
    print("\n=== universe filter (cap not micro + ADV >= 5cr) ===")
    res = apply_universe_filter(events_ctx)
    kept = res["kept"]
    funnel_df = res["funnel"]
    print(f"  events after universe filter: {len(kept)}")
    if not funnel_df.empty:
        f = funnel_df.iloc[0]
        for k, v in f.items():
            print(f"    {k:<22} {int(v):,}")
    if kept.empty:
        print("[ABORT] no events pass universe filter")
        return

    # 7. Run Cell A (T+0) + Cell B (T-1) simulators.
    print("\n=== Cell A (T+0 SHORT) simulator ===")
    sim_a = simulate_short(kept, big5m, daily, cell_label="A")
    trades_a = sim_a["trades"]

    print("\n=== Cell B (T-1 SHORT, exploratory) simulator ===")
    sim_b = simulate_short(kept, big5m, daily, cell_label="B")
    trades_b = sim_b["trades"]

    # 8. Report Option A + Option B for Cell A (primary).
    print("\n\n" + "#" * 78)
    print("#  CELL A — PRIMARY (T+0 09:30 SHORT) — by option-window")
    print("#" * 78)

    cell_a_results: Dict[str, dict] = {}
    cell_a_results["A_wide"] = _report_option(
        "A (wide: 2024-09-13 -> 2026-05-14)",
        trades_a, OPTION_A_START, OPTION_A_END,
    )
    cell_a_results["B_strict"] = _report_option(
        "B (strict: 2025-10-01 -> 2026-05-14)",
        trades_a, OPTION_B_START, OPTION_B_END,
    )

    # 9. Report Cell B (exploratory).
    print("\n\n" + "#" * 78)
    print("#  CELL B — SECONDARY (T-1 SHORT, exploratory only) — by option-window")
    print("#" * 78)
    cell_b_results: Dict[str, dict] = {}
    cell_b_results["A_wide"] = _report_option(
        "A (wide) — Cell B",
        trades_b, OPTION_A_START, OPTION_A_END,
    )
    cell_b_results["B_strict"] = _report_option(
        "B (strict) — Cell B",
        trades_b, OPTION_B_START, OPTION_B_END,
    )

    # 10. Final summary table.
    print("\n\n" + "=" * 78)
    print("SUMMARY — pre-registered thresholds (PF>=1.30 & n>=20 = STRONG PROCEED)")
    print("=" * 78)
    for cell_label, results in (("CELL_A_primary", cell_a_results),
                                ("CELL_B_secondary", cell_b_results)):
        print(f"\n  {cell_label}:")
        for opt_label, r in results.items():
            m = r["metrics"]
            print(f"    {opt_label:<10}  n={m['n']:>4}  PF={m['pf']:>6.3f}  "
                  f"WR={m['wr']:>5.1f}%  Sharpe={m['sharpe']:>6.3f}  "
                  f"=> {r['verdict']}")

    # 11. Persist trades CSV (compat with _cell_mine_tier_a.py).
    all_trades = pd.concat([trades_a, trades_b], ignore_index=True) if (
        not trades_a.empty or not trades_b.empty
    ) else pd.DataFrame()
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fno_removal_drift_short_trades.csv"
    if not all_trades.empty:
        all_trades.to_csv(out_path, index=False)
        print(f"\nFull trade log: {out_path}  ({len(all_trades)} rows)")
    else:
        # Write empty header so downstream tools don't crash
        empty = pd.DataFrame(columns=[
            "cell", "circular_date", "effective_date", "session_date",
            "T1_entry_date", "gap_days", "is_post_oct_1_2025",
            "symbol", "bare_symbol", "cap_segment", "side",
            "entry_ts", "entry_price", "hard_sl", "t1_target", "t2_target",
            "exit_ts", "exit_price", "exit_reason", "qty",
            "realized_pnl", "fee", "net_pnl", "multi_day_net", "multi_day_pct",
        ])
        empty.to_csv(out_path, index=False)
        print(f"\nFull trade log (empty): {out_path}")


if __name__ == "__main__":
    main()
