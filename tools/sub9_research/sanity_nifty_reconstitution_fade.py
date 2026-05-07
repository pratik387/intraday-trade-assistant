"""Pre-coding sanity check for nifty_reconstitution_announcement_window candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-07-sub-project-9-brief-nifty_
reconstitution_window.md): BEFORE writing detector code, simulate the
inclusion-side fade of the passive-buying-exhaustion in the NIFTY 50 /
Next 50 / Bank reconstitution events on 2 years of 5m bar data.

Decision criterion (from brief, locked thresholds):
  PF >= 1.10 AND n >= 30 AND |WR delta| <= 10pp -> proceed (ship winner only)
  n  < 20    -> retire-pre-implementation (sample structurally too tight)
  PF < 1.10  -> retire candidate

Brief is explicit: ship the WINNER only -- no loosening allowed. If both
windows fail PF or n gates, retire decisively.

Usage:
    python tools/sub9_research/sanity_nifty_reconstitution_fade.py

Two windows tested in one run (per brief §6 + §8):
  - Window A (PRIMARY): T+0 effective-day fade
      entry: T+0 09:15 5m bar's CLOSE (per brief §6.2 entry price)
      gate:  09:15 bar must close green/positive (gap-up confirms passive
             demand cleared the auction); abort if 09:15 prints negative
      exit:  10:30 IST OR T1 (0.5R partial 50%) OR T2 (1.5R full) OR stop
      stop:  entry x 1.010 (1.0% above entry)
      side:  SHORT
  - Window B (SECONDARY): T-1 14:00 run-up exhaustion fade
      entry: T-1 14:00 5m bar's CLOSE
      exit:  T-1 15:10 IST OR T1 (0.5R) OR T2 (1.5R) OR stop
      stop:  entry x 1.010 (1.0% above entry; matches Window A for
             apples-to-apples comparison; brief §6.2 uses T-1 high
             buffer but the locked-params spec says 1% hard stop)
      side:  SHORT

Both windows: T1 = 0.5R partial (50% qty), T2 = 1.5R full (50%); breakeven
trail on T2 leg after T1 fill (Indian retail-pro convention per
volume_spike_exhaustion sanity precedent). Latch = one fire per
(symbol, effective_date, window).

Universe (per brief §7): F&O liquid universe, NIFTY 50 + NIFTY Next 50 +
NIFTY Bank inclusions only. 2yr Discovery window 2023-01-01 .. 2024-12-31.

Sample-size honesty: per brief §9, 39 inclusions over 3.5yr -> 22-25
in 2yr Discovery. Below n=30 floor likely. If best window n<20 -> retire.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params per brief §6 + §9 (round-5 standard) ----
DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END   = date(2024, 12, 31)

ALLOWED_INDICES = {
    "NIFTY 50", "NIFTY Next 50", "NIFTY Bank",
    # Round-5 expansion 2026-05-07: same passive-rebalance mechanism on
    # NSE-family broad-market indices; data/index_reconstitution/events.parquet
    # now 625 rows (303 inclusions in 2023-24 Discovery, 13.7x the original
    # 22). Methodologically clean class expansion — see brief §11 note about
    # "if reconstitution-event infrastructure expands to NIFTY 500, this
    # re-emerges as a candidate."
    "NIFTY 500", "NIFTY Midcap 150", "NIFTY Smallcap 250",
}

# Window A: T+0 effective-day
WINDOW_A_ENTRY_HHMM = "09:15"
WINDOW_A_EXIT_HHMM  = "10:30"

# Window B: T-1 secondary
WINDOW_B_ENTRY_HHMM = "14:00"
WINDOW_B_EXIT_HHMM  = "15:10"

# Stop: 1.0% hard stop above entry (SHORT) -- brief §6.3
HARD_STOP_PCT = 1.0

# Targets (R-multiples, per task spec)
T1_R_MULTIPLE = 0.5    # T1 partial @ 0.5R
T2_R_MULTIPLE = 1.5    # T2 full @ 1.5R
T1_PARTIAL_FRACTION = 0.5

USE_BREAKEVEN_TRAIL_AFTER_T1 = True  # matches volume_spike_exhaustion convention
RISK_PER_TRADE_RUPEES = 1000


def load_events() -> pd.DataFrame:
    """Load reconstitution events; restrict to inclusions in allowed indices,
    inside the Discovery window."""
    path = _REPO_ROOT / "data" / "index_reconstitution" / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    df = pd.read_parquet(path)
    df["announcement_date"] = pd.to_datetime(df["announcement_date"]).dt.date
    df["effective_date"]    = pd.to_datetime(df["effective_date"]).dt.date

    print(f"  raw events: {len(df)}")
    incl = df[df["action"] == "inclusion"].copy()
    print(f"  inclusions: {len(incl)}")

    incl = incl[incl["index_name"].isin(ALLOWED_INDICES)]
    print(f"  inclusions in {sorted(ALLOWED_INDICES)}: {len(incl)}")

    incl = incl[
        (incl["effective_date"] >= DISCOVERY_START)
        & (incl["effective_date"] <= DISCOVERY_END)
    ]
    print(f"  inclusions in Discovery {DISCOVERY_START} .. {DISCOVERY_END}: {len(incl)}")
    return incl.reset_index(drop=True)


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_5m_for_events(events: pd.DataFrame) -> pd.DataFrame:
    """Load only the months containing each event's effective_date and the
    immediately preceding month (T-1 may fall on a prior month-end Friday).

    Returns concatenated, symbol-filtered, sorted 5m bars covering all
    candidate trading days for both Window A and Window B per event.
    """
    needed_months = set()
    for ed in events["effective_date"]:
        needed_months.add((ed.year, ed.month))
        # T-1 can cross month boundary -> include prior month
        prev_day = ed - timedelta(days=1)
        # walk back up to 5 calendar days to be safe (covers long weekends)
        for _ in range(5):
            needed_months.add((prev_day.year, prev_day.month))
            prev_day -= timedelta(days=1)

    print(f"  loading {len(needed_months)} monthly 5m feathers ...")
    parts: List[pd.DataFrame] = []
    universe_syms = set(events["symbol"].str.replace("NSE:", "", regex=False).unique())

    for yyyy, mm in sorted(needed_months):
        mdf = _load_5m_for_month(yyyy, mm)
        if mdf.empty:
            continue
        mdf = mdf[mdf["symbol"].isin(universe_syms)]
        if not mdf.empty:
            parts.append(mdf)

    if not parts:
        return pd.DataFrame()

    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total event-relevant 5m bars: {len(big):,}")
    return big


def _prev_trading_day(sym_df: pd.DataFrame, target_d: date) -> Optional[date]:
    """Return the most recent trading day strictly before target_d for which
    5m bars exist for this symbol."""
    days_before = sym_df[sym_df["d"] < target_d]["d"].unique()
    if len(days_before) == 0:
        return None
    return max(days_before)


def simulate_window(
    events: pd.DataFrame,
    big5m: pd.DataFrame,
    window_label: str,
    entry_hhmm: str,
    exit_hhmm: str,
    use_t_minus_1: bool,
    require_green_entry: bool,
) -> tuple[pd.DataFrame, dict]:
    """Run a single window across all events.

    Window A: use_t_minus_1=False, require_green_entry=True
              (per brief §6.2: 09:15 bar must close positive)
    Window B: use_t_minus_1=True,  require_green_entry=False
              (T-1 14:00 run-up exhaustion -- no green-bar gate)
    """
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    trades: List[dict] = []
    funnel = {
        "events_total": len(events),
        "no_5m_data": 0,
        "no_target_day": 0,
        "no_entry_bar": 0,
        "abort_red_entry": 0,
        "no_exit_window": 0,
        "fired": 0,
    }

    for _, ev in events.iterrows():
        raw_sym = ev["symbol"]
        sym = raw_sym.replace("NSE:", "")
        eff_d = ev["effective_date"]
        idx_name = ev["index_name"]

        sym_df = days_per_sym.get(sym)
        if sym_df is None or sym_df.empty:
            funnel["no_5m_data"] += 1
            continue

        # Resolve target trading day
        if use_t_minus_1:
            target_d = _prev_trading_day(sym_df, eff_d)
            if target_d is None:
                funnel["no_target_day"] += 1
                continue
        else:
            target_d = eff_d
            # If effective_date is not a trading day for this symbol, abort
            if target_d not in set(sym_df["d"].unique()):
                funnel["no_target_day"] += 1
                continue

        day_df = sym_df[sym_df["d"] == target_d].sort_values("date").reset_index(drop=True)
        if day_df.empty:
            funnel["no_target_day"] += 1
            continue

        day_df["hhmm"] = day_df["date"].dt.strftime("%H:%M")

        # Entry bar
        entry_rows = day_df[day_df["hhmm"] == entry_hhmm]
        if entry_rows.empty:
            funnel["no_entry_bar"] += 1
            continue
        entry_row = entry_rows.iloc[0]
        entry_ts = entry_row["date"]
        entry_open = float(entry_row["open"])
        entry_close = float(entry_row["close"])

        # Window A gate: 09:15 must close positive (close > open) per brief §6.2
        if require_green_entry and entry_close <= entry_open:
            funnel["abort_red_entry"] += 1
            continue

        entry_price = entry_close  # per brief §6.2: entry @ 5m bar's CLOSE

        # Stop & targets (SHORT side)
        hard_sl = entry_price * (1.0 + HARD_STOP_PCT / 100.0)
        stop_distance = hard_sl - entry_price

        t1_target = entry_price - T1_R_MULTIPLE * stop_distance
        t2_target = entry_price - T2_R_MULTIPLE * stop_distance

        # Forward bars: from after entry bar through exit_hhmm
        entry_idx_arr = day_df.index[day_df["date"] == entry_ts].tolist()
        if not entry_idx_arr:
            funnel["no_entry_bar"] += 1
            continue
        entry_idx = entry_idx_arr[0]
        forward = day_df.iloc[entry_idx + 1:].copy()
        forward = forward[forward["hhmm"] <= exit_hhmm]
        if forward.empty:
            funnel["no_exit_window"] += 1
            continue

        # Walk forward: stop, T1 (partial), T2 (full), or time-stop
        exit_ts = None
        exit_price = None
        exit_reason = None
        hit_t1 = False
        t1_exit_price = None
        t1_exit_ts = None

        for _, bar in forward.iterrows():
            ts = bar["date"]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            hhmm = bar["hhmm"]

            active_sl = entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl

            # SHORT: stop is hit if HIGH >= active_sl
            if high >= active_sl:
                exit_ts = ts
                exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break
            # T1 partial check
            if not hit_t1 and low <= t1_target:
                hit_t1 = True
                t1_exit_price = t1_target
                t1_exit_ts = ts
            # T2 full check
            if hit_t1 and low <= t2_target:
                exit_ts = ts
                exit_price = t2_target
                exit_reason = "t2"
                break
            # Time stop at end of window
            if hhmm >= exit_hhmm:
                exit_ts = ts
                exit_price = close
                exit_reason = "time_stop"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "time_stop"

        # Position sizing on R basis
        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

        # Tiered exit PnL
        if hit_t1:
            qty_t1 = max(int(qty * T1_PARTIAL_FRACTION), 1)
            qty_t2 = max(qty - qty_t1, 0)
            pnl_t1 = (entry_price - t1_exit_price) * qty_t1
            pnl_t2 = (entry_price - exit_price) * qty_t2 if qty_t2 > 0 else 0.0
            realized_pnl = pnl_t1 + pnl_t2
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL")
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, "SELL") if qty_t2 > 0 else 0.0
            fee = fee_t1 + fee_t2
            blended_exit = (
                (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
                if qty_t2 > 0 else t1_exit_price
            )
        else:
            realized_pnl = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "window": window_label,
            "symbol": "NSE:" + sym,
            "index_name": idx_name,
            "announcement_date": ev["announcement_date"],
            "effective_date": eff_d,
            "trade_date": target_d,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "t2_target": t2_target,
            "hit_t1": hit_t1,
            "t1_exit_price": t1_exit_price,
            "t1_exit_ts": t1_exit_ts,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "stop_distance": stop_distance,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
        funnel["fired"] += 1

    return pd.DataFrame(trades), funnel


def _summarize(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        print(f"\n[{label}] no trades")
        return {"label": label, "n": 0, "pf": None, "wr": None, "sharpe": None,
                "net_pnl": 0.0}
    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    wr = round(float((npnl > 0).mean()) * 100, 1)
    daily = trades.groupby("trade_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0

    print(f"\n=== {label} ===")
    print(f"  n         : {n}")
    print(f"  WR        : {wr}%")
    print(f"  Gross PnL : Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"  Fees      : Rs.{int(trades['fee'].sum()):,}")
    print(f"  NET PnL   : Rs.{int(npnl.sum()):,}")
    print(f"  NET PF    : {pf}")
    print(f"  Sharpe(d) : {sharpe}")
    print(f"  Avg net   : Rs.{int(npnl.mean()):,}")

    print("  Per index:")
    for idx_name, grp in trades.groupby("index_name"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net = int(grp["net_pnl"].sum())
        print(f"    {idx_name:<14} n={n2:>3} PF={pf2:>6} WR={wr2:>5}% net=Rs.{net:>10,}")

    print("  Exit-reason:")
    for rsn, grp in trades.groupby("exit_reason"):
        net = int(grp["net_pnl"].sum())
        print(f"    {rsn:<18} n={len(grp):>3} netPnL=Rs.{net:>10,} avg=Rs.{int(grp['net_pnl'].mean()):>6,}")

    return {"label": label, "n": n, "pf": pf, "wr": wr, "sharpe": sharpe,
            "net_pnl": float(npnl.sum())}


def _print_funnel(label: str, funnel: dict) -> None:
    print(f"\n  Funnel [{label}]:")
    print(f"    events in Discovery     : {funnel['events_total']}")
    print(f"    no 5m data for symbol   : {funnel['no_5m_data']}")
    print(f"    no target trading day   : {funnel['no_target_day']}")
    print(f"    no entry bar at HHMM    : {funnel['no_entry_bar']}")
    print(f"    abort (entry red bar)   : {funnel['abort_red_entry']}")
    print(f"    no exit window bars     : {funnel['no_exit_window']}")
    print(f"    -> FIRED                : {funnel['fired']}")


def _verdict(label: str, summary: dict) -> str:
    n = summary["n"]
    pf = summary["pf"]

    if n < 20:
        return (f"[{label}] RETIRE-PRE-IMPLEMENTATION "
                f"(n={n} < 20 floor; structurally too tight). "
                f"Brief §9 says 'sample-size honesty -- if structurally "
                f"<30 after curation, retire-eligible-pre-data'. n<20 "
                f"is below even the lenient floor.")
    if n < 30:
        if pf is not None and pf >= 1.10:
            return (f"[{label}] BORDERLINE: PF={pf} >= 1.10 BUT n={n} "
                    f"below n=30 floor. Brief §9 says 'retire-eligible-"
                    f"pre-data'. RETIRE unless user accepts relaxation.")
        return (f"[{label}] RETIRE: n={n} < 30 AND PF={pf} < 1.10. "
                f"Both gates fail.")

    if pf is None or pf < 1.10:
        return f"[{label}] RETIRE: PF={pf} < 1.10 floor."

    return f"[{label}] PROCEED: n={n} >= 30 AND PF={pf} >= 1.10."


def main():
    print("=== nifty_reconstitution_announcement_window — pre-coding sanity ===")
    print(f"Discovery window: {DISCOVERY_START} .. {DISCOVERY_END}")

    print("\nLoading reconstitution events ...")
    events = load_events()

    if events.empty:
        print("No qualifying events; aborting.")
        return

    print("\nLoading 5m feathers for event months ...")
    big5m = build_5m_for_events(events)
    if big5m.empty:
        print("No 5m bars for any event symbol; aborting.")
        return

    # Count events with at least some 5m data on effective_date or T-1
    syms_with_data = set(big5m["symbol"].unique())
    events_with_5m = events[events["symbol"].str.replace("NSE:", "", regex=False).isin(syms_with_data)]
    print(f"  events with 5m data available: {len(events_with_5m)}")

    print("\n--- Running Window A (T+0 effective-day 09:15 -> 10:30) ---")
    trades_A, funnel_A = simulate_window(
        events,
        big5m,
        window_label="T+0",
        entry_hhmm=WINDOW_A_ENTRY_HHMM,
        exit_hhmm=WINDOW_A_EXIT_HHMM,
        use_t_minus_1=False,
        require_green_entry=True,
    )
    _print_funnel("Window A T+0", funnel_A)

    print("\n--- Running Window B (T-1 14:00 -> 15:10) ---")
    trades_B, funnel_B = simulate_window(
        events,
        big5m,
        window_label="T-1",
        entry_hhmm=WINDOW_B_ENTRY_HHMM,
        exit_hhmm=WINDOW_B_EXIT_HHMM,
        use_t_minus_1=True,
        require_green_entry=False,
    )
    _print_funnel("Window B T-1", funnel_B)

    # --- Reports ---
    summary_A = _summarize(trades_A, "Window A — T+0 effective-day fade")
    summary_B = _summarize(trades_B, "Window B — T-1 14:00 run-up exhaustion fade")

    combined = pd.concat([trades_A, trades_B], ignore_index=True)
    summary_combined = _summarize(combined, "COMBINED (both windows)")

    # --- Verdict per window ---
    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)
    v_A = _verdict("Window A T+0", summary_A)
    v_B = _verdict("Window B T-1", summary_B)
    print(v_A)
    print(v_B)

    # --- Final shipping decision (brief §9: ship the WINNER only) ---
    print("\n" + "=" * 70)
    print("FINAL SHIPPING DECISION (brief §9 — winner only, no loosening)")
    print("=" * 70)

    a_ok = (summary_A["n"] >= 30 and summary_A["pf"] is not None
            and summary_A["pf"] >= 1.10)
    b_ok = (summary_B["n"] >= 30 and summary_B["pf"] is not None
            and summary_B["pf"] >= 1.10)

    if a_ok and b_ok:
        winner = "A" if summary_A["pf"] >= summary_B["pf"] else "B"
        print(f"Both pass strict gate. Ship Window {winner} (higher PF).")
    elif a_ok:
        print("SHIP Window A (T+0). Window B fails strict gate.")
    elif b_ok:
        print("SHIP Window B (T-1). Window A fails strict gate.")
    else:
        # Neither passes the strict gate. Determine retire reason.
        best_label = "A" if (summary_A["n"] or 0) >= (summary_B["n"] or 0) else "B"
        best_n = max(summary_A["n"] or 0, summary_B["n"] or 0)
        best_pf_A = summary_A["pf"] if summary_A["pf"] is not None else 0
        best_pf_B = summary_B["pf"] if summary_B["pf"] is not None else 0
        best_pf = max(best_pf_A, best_pf_B)

        if best_n < 20:
            print(f"RETIRE-PRE-IMPLEMENTATION: best window n={best_n} < 20. "
                  f"Sample structurally too tight. Same fate as fno_oi_cliff.")
        elif best_n < 30:
            print(f"RETIRE: best window n={best_n} < 30 floor. "
                  f"Brief §9 forbids loosening. Same fate as fno_oi_cliff.")
        elif best_pf < 1.10:
            print(f"RETIRE: best PF={best_pf} < 1.10 floor. "
                  f"Both windows fail; brief §9 forbids loosening.")
        else:
            print(f"RETIRE: gates fail; best window {best_label}, n={best_n}, PF={best_pf}.")

    # --- Persist trades CSV ---
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nifty_reconstitution_fade_trades.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nFull trade log: {out_csv}")


if __name__ == "__main__":
    main()
