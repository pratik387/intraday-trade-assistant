"""Pre-coding sanity check for fno_ban_entry_t1_fade candidate (Sub-9 Candidate 2).

Per brief: specs/2026-05-14-brief-fno_ban_entry_t1_fade.md

Mechanism (one sentence):
  Under the Nov 3, 2025 intraday FutEq OI monitoring regime, a stock entering
  the F&O ban list creates a structural one-way exit flow for the next
  session — fresh longs are blocked, existing longs unwind into a buyer-thin
  tape, producing a predictable T+1 negative drift. SHORT-only by rule
  design (no symmetric mechanic blocks fresh shorts).

Setup (brief §"Implementation sketch", lines 250-260):
  1. Load ban-event parquet (data/fno_ban_history/fno_ban_events.parquet).
  2. For each event, compute T+1 = next trading day after ban_date.
  3. Cell A primary filter:
       - cap_segment in {mid_cap, small_cap}
       - mis_leverage >= 1.0 (must be MIS-short-eligible)
       - prior_day_return >= +1.5% (T0 close vs T-1 close)
       - T+1 09:15 gap in [-2%, +2%] (vs T0 close = PDC)
  4. Entry: T+1 10:00 IST 5m bar close. Bar must be RED (close < open).
  5. SL: max(T+1 09:30-10:00 high) × 1.005; floor entry × 1.01 (1% min).
  6. T1 (50% qty): T0 close (PDC).
  7. T2 (50% qty): PDC × (1 - 0.5 × prior_day_return_pct/100).
  8. Time stop: 14:30 IST.
  9. BE trail: after T1 fills, runner SL moves to entry (active_sl pattern).

Pre-coding sanity decision criteria (brief §"Pre-coding sanity check"):
  Window: Nov 3, 2025 -> Dec 31, 2025 (~40 trading days)
  Cell A pass:  NET PF >= 1.20  AND  n >= 50  AND  NET Sharpe > 0  AND  WR >= 30%
  Cell B pass:  NET PF >= 1.15  AND  n >= 25
  If Cell A misses -> brief retires immediately (no cell-mining rescue).

Regime-break pre-flight:
  Calls services.regime_break_detector.check_window() with depends_on tags
  ["MWPL", "intraday_ban", "single_stock_FO", "F&O_speculation"]. The
  sanity window (Nov 3 - Dec 31, 2025) is post-Nov-3-2025 by construction;
  pre-flight should pass at min_severity="critical" (note: rule_changes.csv
  contains the Nov 3 row itself at MEDIUM severity, so we use 'critical'
  to admit the canonical post-rule window; HIGH would trip on Apr-1-2026
  STT hike were we to expand the window post-mar-2026).

STUB MODE:
  If data/fno_ban_history/fno_ban_events.parquet does not exist, emits a
  hardcoded sample event list covering Oct 2025 - Dec 2025 (small +
  mid-cap names) for skeleton verification. Replace with real scraper
  output before any actual Discovery run.

Usage:
    python -m tools.sub9_research.sanity_fno_ban_entry_t1_fade
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment, get_mis_info       # noqa: E402
from services.regime_break_detector import check_window, GauntletRegimeBreak  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee           # noqa: E402


# ============================================================================
# Brief-locked params (§"Cells (pre-registered)", §"Implementation sketch")
# ============================================================================

# Cell A primary
ALLOWED_CAPS_A = {"mid_cap", "small_cap"}
MIN_MIS_LEVERAGE = 1.0
MIN_PRIOR_DAY_RETURN_PCT = 1.5
GAP_PCT_MIN = -2.0
GAP_PCT_MAX = 2.0

# Cell B (intraday-entry only) — secondary
# Mirror Cell A's cap + MIS gates but does not require prior-day-return
# qualifier or gap-band (the intraday-entry surprise is its own trigger).
# For sanity v1 (T+1 simulation regardless of event_type), Cell B uses
# the same T+1 simulation as Cell A but filtered on event_type == "intraday".

# Entry / exits
ENTRY_HHMM = "10:00"               # T+1 10:00 IST 5m bar close
SL_WINDOW_START_HHMM = "09:30"
SL_WINDOW_END_HHMM = "10:00"       # inclusive of the entry bar
SL_BUFFER_PCT = 0.5                # 0.5% above the 09:30-10:00 high
SL_MIN_PCT = 1.0                   # entry x 1.01 floor (1% min stop distance)

TIME_STOP_HHMM = "14:30"           # hard time stop

T1_QTY_FRAC = 0.5                  # 50% qty at T1
USE_BE_TRAIL_AFTER_T1 = True       # runner SL -> entry after T1 fills

RISK_PER_TRADE_RUPEES = 1000

# Sanity window (brief §"Pre-coding sanity check")
SANITY_START = date(2025, 11, 3)
SANITY_END   = date(2025, 12, 31)

# Decision thresholds (brief §"Pre-coding sanity check")
CELL_A_PF_MIN = 1.20
CELL_A_N_MIN = 50
CELL_A_WR_MIN = 30.0
CELL_A_SHARPE_MIN = 0.0   # strictly > 0

CELL_B_PF_MIN = 1.15
CELL_B_N_MIN = 25

# ============================================================================
# Regime-break pre-flight
# ============================================================================

DEPENDS_ON = ["MWPL", "intraday_ban", "single_stock_FO", "F&O_speculation"]


def regime_preflight() -> None:
    """Call regime_break_detector for the sanity window. Fails fast on
    high/critical rule changes that affect this setup's mechanism.

    Brief §"Regulatory dependencies": Discovery-B and pre-coding sanity
    windows are post-Nov-3-2025 by construction. The Nov 3 row itself is
    severity=medium so it does NOT trip the default `high` gate — but we
    log it at info level so the human reviewer sees it. The next critical
    row (Apr 1 2026 STT) is outside the sanity window so no trip there.
    """
    print("\n=== regime_break_detector pre-flight ===")
    print(f"  strategy:    fno_ban_entry_t1_fade")
    print(f"  depends_on:  {DEPENDS_ON}")
    print(f"  window:      {SANITY_START} .. {SANITY_END}  (label: pre_coding_sanity)")
    try:
        hits_high = check_window(
            strategy_name="fno_ban_entry_t1_fade",
            depends_on=DEPENDS_ON,
            window_label="pre_coding_sanity",
            start=SANITY_START,
            end=SANITY_END,
            min_severity="high",
            raise_on_break=True,
        )
        # Also surface medium-severity rows for human awareness (no raise).
        med_hits = check_window(
            strategy_name="fno_ban_entry_t1_fade",
            depends_on=DEPENDS_ON,
            window_label="pre_coding_sanity",
            start=SANITY_START,
            end=SANITY_END,
            min_severity="medium",
            raise_on_break=False,
        )
        print(f"  PASS (no high/critical rule changes in window).")
        if med_hits:
            print(f"  Note: {len(med_hits)} medium-severity row(s) in window (informational):")
            for r in med_hits:
                print(f"    - {r.effective_date} [{r.severity}] {r.description[:90]}")
    except GauntletRegimeBreak as e:
        print(f"  FAIL: {e}")
        raise


# ============================================================================
# Ban-event source (real parquet OR stub)
# ============================================================================

_BAN_EVENTS_PARQUET = _REPO_ROOT / "data" / "fno_ban_history" / "fno_ban_events.parquet"


def _stub_events() -> pd.DataFrame:
    """Hardcoded sample ban events covering Oct-Dec 2025 for skeleton
    verification only. Symbols chosen from common F&O small/mid-cap
    universe known to hit ban under the new MWPL regime (per market
    chatter). DO NOT use for any real validation — replace with real
    scraper output before Discovery."""
    print("  STUB MODE: ban events parquet not yet populated.")
    print("  Generating 10 synthetic events for skeleton verification.")
    print("  Run scraper first for real Discovery.")
    # Stub picks: mid/small-cap symbols whose ban_date lands on a day with
    # actual prior_day_return >= 1.5% in the sanity window (Nov-Dec 2025).
    # Hand-picked from the daily feather so skeleton run produces some
    # trades to exercise the SL/T1/T2/time-stop simulator end-to-end.
    rows = [
        # (symbol, ban_date, event_type, entry_snapshot_index)
        ("AARTIIND",   date(2025, 11, 3),  "eod",      None),
        ("ABFRL",      date(2025, 11, 3),  "intraday", 2),
        ("APLLTD",     date(2025, 11, 3),  "eod",      None),
        ("ATUL",       date(2025, 11, 3),  "intraday", 3),
        ("ALKYLAMINE", date(2025, 11, 3),  "eod",      None),
        ("ALLCARGO",   date(2025, 11, 3),  "intraday", 1),
        ("ASHOKA",     date(2025, 11, 3),  "eod",      None),
        ("ASTRAMICRO", date(2025, 11, 3),  "intraday", 4),
        ("BAJAJCON",   date(2025, 11, 3),  "eod",      None),
        ("BANCOINDIA", date(2025, 11, 3),  "intraday", 2),
    ]
    df = pd.DataFrame(rows, columns=["symbol", "ban_date", "event_type", "entry_snapshot_index"])
    df["ban_entry_time"] = pd.NaT
    df["ban_exit_time"] = pd.NaT
    df["mwpl_pct_at_entry"] = np.nan
    return df


def load_ban_events() -> pd.DataFrame:
    """Load ban-event parquet if it exists; otherwise emit stub events.

    Schema: [symbol, ban_date, ban_entry_time, ban_exit_time,
             mwpl_pct_at_entry, event_type, entry_snapshot_index]
    """
    if _BAN_EVENTS_PARQUET.exists():
        print(f"  loading ban events: {_BAN_EVENTS_PARQUET}")
        df = pd.read_parquet(_BAN_EVENTS_PARQUET)
        # Normalize ban_date to date (parquet may carry datetime/Timestamp)
        if pd.api.types.is_datetime64_any_dtype(df["ban_date"]):
            df["ban_date"] = df["ban_date"].dt.date
        return df.reset_index(drop=True)
    return _stub_events()


# ============================================================================
# 5m feather loading
# ============================================================================

_NEEDED_5M_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_feather(path, columns=_NEEDED_5M_COLS)
    except Exception:
        return pd.read_feather(path)


def build_window_5m(events: pd.DataFrame) -> pd.DataFrame:
    """Load monthly 5m feathers spanning all distinct ban-date months PLUS
    the following month (for T+1 which may straddle a month boundary).
    """
    months: set = set()
    for d in events["ban_date"]:
        if isinstance(d, (datetime, pd.Timestamp)):
            d = d.date() if hasattr(d, "date") else d
        months.add((d.year, d.month))
        # Following month (T+1 month boundary)
        nxt = (d.month % 12) + 1
        nxt_y = d.year + (1 if d.month == 12 else 0)
        months.add((nxt_y, nxt))
    print(f"  loading {len(months)} monthly 5m feathers ...")
    parts: List[pd.DataFrame] = []
    for (y, m) in sorted(months):
        mdf = _load_5m_for_month(y, m)
        if not mdf.empty:
            parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total 5m bars: {len(big):,}")
    return big


def load_production_daily() -> pd.DataFrame:
    """Load 1day OHLCV from consolidated_daily.feather for prior-day-return
    and T0 close (PDC) computation. Same source mock_broker.get_daily
    reads at runtime — production parity."""
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        print(f"  WARN: {daily_path} missing; daily-context (prior-day return) will be unavailable.")
        return pd.DataFrame()
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[["symbol", "d", "open", "high", "low", "close", "volume"]].copy()
    return df


# ============================================================================
# Filter + simulator
# ============================================================================


def attach_context(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Attach cap_segment, mis_leverage, T0 close (=PDC), prior_day_return.

    The brief defines:
      - T0 = ban_date (the close BEFORE T+1)
      - PDC = T0 close
      - prior_day_return = (T0 close / T-1 close - 1) * 100
    """
    out = events.copy()
    # Metadata
    out["cap_segment"] = out["symbol"].apply(get_cap_segment)
    out["mis_leverage"] = out["symbol"].apply(
        lambda s: (get_mis_info(s).get("mis_leverage") or 0.0)
    )

    if daily.empty:
        out["t0_close"] = np.nan
        out["tm1_close"] = np.nan
        out["prior_day_return_pct"] = np.nan
        return out

    # Index daily by (symbol, d) for fast lookup of T0 and T-1 closes.
    daily_sorted = daily.sort_values(["symbol", "d"]).reset_index(drop=True)
    daily_sorted["prev_close"] = daily_sorted.groupby("symbol")["close"].shift(1)
    key = daily_sorted.set_index(["symbol", "d"])[["close", "prev_close"]]

    t0_closes, tm1_closes = [], []
    for _, r in out.iterrows():
        k = (r["symbol"], r["ban_date"])
        if k in key.index:
            row = key.loc[k]
            # If duplicated, take the last (single-row expected though)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            t0_closes.append(float(row["close"]) if pd.notna(row["close"]) else np.nan)
            tm1_closes.append(float(row["prev_close"]) if pd.notna(row["prev_close"]) else np.nan)
        else:
            t0_closes.append(np.nan)
            tm1_closes.append(np.nan)

    out["t0_close"] = t0_closes
    out["tm1_close"] = tm1_closes
    with np.errstate(invalid="ignore", divide="ignore"):
        out["prior_day_return_pct"] = (out["t0_close"] / out["tm1_close"] - 1.0) * 100.0
    return out


def apply_cell_a_filter(events: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Apply Cell A primary filter (cap + MIS + prior-day return). Gap-band
    is applied later in the simulator (needs T+1 09:15 open).

    Returns dict with 'kept' and 'funnel'.
    """
    funnel = {"in_events": len(events)}
    df = events.copy()

    df = df[df["cap_segment"].isin(ALLOWED_CAPS_A)]
    funnel["cap_admissible"] = len(df)

    df = df[df["mis_leverage"] >= MIN_MIS_LEVERAGE]
    funnel["mis_eligible"] = len(df)

    df = df[df["prior_day_return_pct"] >= MIN_PRIOR_DAY_RETURN_PCT]
    funnel["prior_return_qualified"] = len(df)

    # Restrict to sanity window
    df = df[(df["ban_date"] >= SANITY_START) & (df["ban_date"] <= SANITY_END)]
    funnel["in_sanity_window"] = len(df)

    funnel_df = pd.DataFrame([funnel])
    return {"kept": df.reset_index(drop=True), "funnel": funnel_df}


def simulate_t1_short(events: pd.DataFrame, big5m: pd.DataFrame) -> Dict:
    """For each ban event, find T+1 (next trading day in 5m data), apply
    gap-band, enter SHORT at 10:00 5m close (if red), exit on
    SL / T1 partial / T2 / 14:30 time stop with BE trail after T1.

    Returns dict with `trades` DataFrame and counters.
    """
    trades: List[dict] = []
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date") for sym, g in big5m.groupby("symbol")
    }

    n_no_t1 = n_gap_fail = n_no_entry_bar = n_not_red = n_zero_stop = n_traded = 0

    for _, ev in events.iterrows():
        sym = ev["symbol"]
        ban_d = ev["ban_date"]
        t0_close = ev["t0_close"]
        prior_ret = ev["prior_day_return_pct"]
        if pd.isna(t0_close) or pd.isna(prior_ret):
            # No daily context (likely stub mode without daily feather) -> skip.
            n_no_t1 += 1
            continue

        sym_df = days_per_sym.get(sym)
        if sym_df is None or sym_df.empty:
            n_no_t1 += 1
            continue

        future = sym_df[sym_df["d"] > ban_d]
        if future.empty:
            n_no_t1 += 1
            continue

        t1_date = future["d"].iloc[0]
        t1_df = future[future["d"] == t1_date].sort_values("date").reset_index(drop=True)
        if t1_df.empty:
            n_no_t1 += 1
            continue

        # T+1 09:15 gap (open of the first 5m bar) vs PDC (T0 close)
        t1_open = float(t1_df.iloc[0]["open"])
        gap_pct = (t1_open / t0_close - 1.0) * 100.0
        if gap_pct < GAP_PCT_MIN or gap_pct > GAP_PCT_MAX:
            n_gap_fail += 1
            continue

        # Entry bar: 10:00
        t1_df["hhmm"] = t1_df["date"].dt.strftime("%H:%M")
        entry_rows = t1_df[t1_df["hhmm"] == ENTRY_HHMM]
        if entry_rows.empty:
            n_no_entry_bar += 1
            continue
        entry_bar = entry_rows.iloc[0]
        entry_idx = entry_bar.name
        entry_ts = entry_bar["date"]
        entry_price = float(entry_bar["close"])
        entry_open = float(entry_bar["open"])

        # Must be RED (close < open)
        if entry_price >= entry_open:
            n_not_red += 1
            continue

        # SL: max(09:30-10:00 high) * 1.005, floor entry * 1.01
        sl_window = t1_df[(t1_df["hhmm"] >= SL_WINDOW_START_HHMM)
                          & (t1_df["hhmm"] <= SL_WINDOW_END_HHMM)]
        if sl_window.empty:
            n_no_entry_bar += 1
            continue
        sl_window_high = float(sl_window["high"].max())
        sl_struct = sl_window_high * (1.0 + SL_BUFFER_PCT / 100.0)
        sl_floor = entry_price * (1.0 + SL_MIN_PCT / 100.0)
        hard_sl = max(sl_struct, sl_floor)
        stop_distance = hard_sl - entry_price
        if stop_distance <= 0:
            n_zero_stop += 1
            continue

        # T1 = PDC; T2 = PDC * (1 - 0.5 * prior_ret/100)
        t1_target = float(t0_close)
        t2_target = float(t0_close) * (1.0 - 0.5 * (prior_ret / 100.0))

        # Walk forward from the bar AFTER entry
        forward = t1_df.iloc[entry_idx + 1:].copy() if entry_idx + 1 < len(t1_df) else pd.DataFrame()
        if forward.empty:
            # No bars after entry — skip (would need EOD fallback but pointless)
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

            # BE trail after T1 fills
            active_sl = entry_price if (hit_t1 and USE_BE_TRAIL_AFTER_T1) else hard_sl

            # SHORT: SL is above; check first (worst-case bar fill)
            if hi >= active_sl:
                exit_ts = ts
                exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break

            # T1 partial (50% qty)
            if (not hit_t1) and (lo <= t1_target):
                hit_t1 = True
                t1_exit_price = t1_target

            # T2 full close (after T1 — only if T1 already filled, else T2 only on this bar)
            if lo <= t2_target:
                exit_ts = ts
                exit_price = t2_target
                exit_reason = "t2"
                break

            # Time stop 14:30
            if hhmm >= TIME_STOP_HHMM:
                exit_ts = ts
                exit_price = cl
                exit_reason = "time_stop_1430"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "eod_fallback"

        # Position sizing + PnL (T1 = 50% qty partial, T2/SL/time = runner)
        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        if hit_t1:
            qty_t1 = max(int(qty * T1_QTY_FRAC), 1)
            qty_t2 = qty - qty_t1
            if qty_t2 < 1:
                qty_t2 = 0
            pnl_t1 = (entry_price - t1_exit_price) * qty_t1
            pnl_runner = (entry_price - exit_price) * qty_t2 if qty_t2 > 0 else 0.0
            realized_pnl = pnl_t1 + pnl_runner
            fee = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL")
            if qty_t2 > 0:
                fee += calc_fee(entry_price, exit_price, qty_t2, "SELL")
            blended_exit = (
                (t1_exit_price * qty_t1 + exit_price * qty_t2) / qty
                if qty_t2 > 0 else t1_exit_price
            )
        else:
            realized_pnl = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "T0_signal_date": ban_d,
            "T1_entry_date": t1_date,
            "session_date": t1_date,
            "symbol": "NSE:" + sym,
            "bare_symbol": sym,
            "cap_segment": ev["cap_segment"],
            "mis_leverage": float(ev["mis_leverage"]),
            "event_type": ev["event_type"],
            "entry_snapshot_index": ev.get("entry_snapshot_index"),
            "side": "SHORT",
            "tm1_close": ev["tm1_close"],
            "t0_close": t0_close,
            "prior_day_return_pct": float(prior_ret),
            "t1_open": t1_open,
            "gap_pct": float(gap_pct),
            "sl_window_high": sl_window_high,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "t2_target": t2_target,
            "stop_distance": stop_distance,
            "hit_t1": hit_t1,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
        n_traded += 1

    print(f"\n  no T+1 data:                          {n_no_t1}")
    print(f"  gap fail (outside [{GAP_PCT_MIN}%, {GAP_PCT_MAX}%]): {n_gap_fail}")
    print(f"  no {ENTRY_HHMM} entry bar:                  {n_no_entry_bar}")
    print(f"  10:00 bar not RED:                    {n_not_red}")
    print(f"  zero/neg stop_distance:               {n_zero_stop}")
    print(f"  traded:                               {n_traded}")

    return {
        "trades": pd.DataFrame(trades),
        "n_no_t1": n_no_t1,
        "n_gap_fail": n_gap_fail,
        "n_no_entry_bar": n_no_entry_bar,
        "n_not_red": n_not_red,
        "n_zero_stop": n_zero_stop,
        "n_traded": n_traded,
    }


# ============================================================================
# Reporting + verdict
# ============================================================================


def _cell_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return dict(n=0, pf=float("nan"), wr=float("nan"),
                    sharpe=float("nan"), gross=0.0, fee=0.0, net=0.0)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = wins / losses if losses > 0 else float("inf")
    wr = float((npnl > 0).mean()) * 100.0
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    return dict(
        n=len(trades),
        pf=float(pf),
        wr=float(wr),
        sharpe=float(sharpe),
        gross=float(trades["realized_pnl"].sum()),
        fee=float(trades["fee"].sum()),
        net=float(npnl.sum()),
    )


def _print_cell_block(label: str, m: dict) -> None:
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


def _verdict_cell_a(m: dict) -> str:
    """Return PASS / FAIL string per brief pre-coding sanity thresholds."""
    if m["n"] == 0:
        return f"FAIL: n=0 (no trades produced)"
    fails = []
    if m["pf"] < CELL_A_PF_MIN:
        fails.append(f"PF={m['pf']:.3f} < {CELL_A_PF_MIN}")
    if m["n"] < CELL_A_N_MIN:
        fails.append(f"n={m['n']} < {CELL_A_N_MIN}")
    if m["sharpe"] <= CELL_A_SHARPE_MIN:
        fails.append(f"Sharpe={m['sharpe']:.3f} <= {CELL_A_SHARPE_MIN}")
    if m["wr"] < CELL_A_WR_MIN:
        fails.append(f"WR={m['wr']:.1f}% < {CELL_A_WR_MIN}%")
    if fails:
        return "FAIL (" + "; ".join(fails) + ")"
    return "PASS — proceed to detector implementation"


def _verdict_cell_b(m: dict) -> str:
    if m["n"] == 0:
        return f"FAIL: n=0 (no intraday-entry events traded)"
    fails = []
    if m["pf"] < CELL_B_PF_MIN:
        fails.append(f"PF={m['pf']:.3f} < {CELL_B_PF_MIN}")
    if m["n"] < CELL_B_N_MIN:
        fails.append(f"n={m['n']} < {CELL_B_N_MIN}")
    if fails:
        return "FAIL (" + "; ".join(fails) + ")"
    return "PASS — Cell B exploratory threshold met"


def report(trades: pd.DataFrame, funnel_df: pd.DataFrame, sim_summary: dict) -> dict:
    print("\n" + "=" * 78)
    print("fno_ban_entry_t1_fade — pre-coding sanity check")
    print("=" * 78)

    if not trades.empty:
        print(f"\nPeriod: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")

    # Funnel
    print("\n=== FUNNEL (events -> filters -> entries) ===")
    if not funnel_df.empty:
        f = funnel_df.iloc[0]
        for k, v in f.items():
            print(f"  {k:<28} {int(v):,}")
    print(f"  no T+1 data:                 {sim_summary['n_no_t1']}")
    print(f"  gap-band fail:               {sim_summary['n_gap_fail']}")
    print(f"  no 10:00 entry bar:          {sim_summary['n_no_entry_bar']}")
    print(f"  10:00 bar not RED:           {sim_summary['n_not_red']}")
    print(f"  zero stop distance:          {sim_summary['n_zero_stop']}")
    print(f"  traded:                      {sim_summary['n_traded']}")

    # Cell A = all trades passing filter + simulation (Cell A is the primary)
    m_a = _cell_metrics(trades)
    _print_cell_block("Cell A (primary: mid+small cap, prior_ret>=1.5%, gap [-2,+2])", m_a)
    verdict_a = _verdict_cell_a(m_a)
    print(f"  VERDICT Cell A: {verdict_a}")

    # Cell B = subset where event_type == 'intraday'
    if not trades.empty and "event_type" in trades.columns:
        cell_b_trades = trades[trades["event_type"] == "intraday"]
    else:
        cell_b_trades = pd.DataFrame()
    m_b = _cell_metrics(cell_b_trades)
    _print_cell_block("Cell B (intraday-entry only)", m_b)
    verdict_b = _verdict_cell_b(m_b)
    print(f"  VERDICT Cell B: {verdict_b}")

    # Per cap_segment
    if not trades.empty:
        print("\nPer cap_segment:")
        for cap, grp in trades.groupby("cap_segment"):
            m = _cell_metrics(grp)
            print(f"  {cap:<12} n={m['n']:>4} PF={m['pf']:>6.3f} "
                  f"WR={m['wr']:>5.1f}% Sh={m['sharpe']:>6.2f} netPnL=Rs.{int(m['net']):>10,}")

    # Per month
    if not trades.empty:
        print("\nPer month:")
        tm = trades.copy()
        tm["month"] = pd.to_datetime(tm["T1_entry_date"]).dt.strftime("%Y-%m")
        for mth, grp in tm.groupby("month"):
            m = _cell_metrics(grp)
            print(f"  {mth} n={m['n']:>3} PF={m['pf']:>6.3f} "
                  f"WR={m['wr']:>5.1f}% netPnL=Rs.{int(m['net']):>10,}")

    # Exit-reason breakdown
    if not trades.empty:
        print("\nExit-reason breakdown:")
        for rsn, grp in trades.groupby("exit_reason"):
            avg = int(grp["net_pnl"].mean())
            print(f"  {rsn:<22} n={len(grp):>4} avg_net=Rs.{avg:>6,}")

    return {"cell_a": m_a, "cell_b": m_b,
            "verdict_a": verdict_a, "verdict_b": verdict_b}


# ============================================================================
# Main
# ============================================================================


def main():
    # 1. Regime-break pre-flight (fail fast if window straddles high/critical rule).
    regime_preflight()

    # 2. Load ban events (real parquet or STUB MODE fallback).
    print("\n=== ban events ===")
    events = load_ban_events()
    print(f"  loaded {len(events)} events")
    if events.empty:
        print("[ABORT] no ban events to process")
        return

    # 3. Load daily for prior-day-return + T0 close (PDC).
    print("\n=== daily context ===")
    daily = load_production_daily()
    if daily.empty:
        print("  WARN: no daily feather; sanity will skip events without context.")

    # 4. Load 5m bars covering all event months + 1 lookforward month.
    print("\n=== 5m bars ===")
    big5m = build_window_5m(events)
    if big5m.empty:
        print("[ABORT] no 5m feathers available for event months")
        return

    # 5. Attach cap/mis/T0/prior-ret context.
    events = attach_context(events, daily)

    # 6. Apply Cell A filter (cap + MIS + prior-ret + window).
    res = apply_cell_a_filter(events)
    kept = res["kept"]
    funnel_df = res["funnel"]
    print(f"\nCell A filter -> {len(kept)} events admissible (pre-gap-band).")
    if kept.empty:
        # Still run reporting so the verdict prints.
        report(pd.DataFrame(), funnel_df, dict(
            n_no_t1=0, n_gap_fail=0, n_no_entry_bar=0,
            n_not_red=0, n_zero_stop=0, n_traded=0))
        return

    # 7. Simulate T+1 SHORT entries.
    print("\n=== T+1 simulation (10:00 SHORT -> SL / T1 / T2 / 14:30) ===")
    sim = simulate_t1_short(kept, big5m)
    trades = sim["trades"]

    # 8. Report + verdict.
    report(trades, funnel_df, sim)

    # 9. Persist trade log.
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fno_ban_entry_t1_fade_trades.csv"
    trades.to_csv(out_path, index=False)
    print(f"\nFull trade log: {out_path}")


if __name__ == "__main__":
    main()
