"""SL/T1/T2/T1_qty/time-stop sweep on the RSI short-overbought cluster.

Re-simulates each event in the small_cap × SHORT × RSI>=80 universe with a
grid of SL_BUFFER, T1_R, T2_R, T1_QTY, TIME_STOP. Reports top combos by
Discovery PF, then validates them on the OOS window.

Event detection is invariant to SL/target params, so we run detect_events
ONCE per period (Discovery & OOS) using the existing sanity-script math, then
re-walk 5m bars per combo with the new exit rules.

Grid:
  SL_BUFFER: [0.3%, 0.5%, 0.7%]            # 3
  T1_R:      [0.5, 1.0, 1.5]                # 3
  T2_R:      [1.0, 1.5, 2.0, 3.0]           # 4
  T1_QTY:    [0.0, 0.33, 0.50, 0.67]        # 4 — includes single-target (0%)
  TIME_STOP: ['13:00', '14:30', '15:10']    # 3
  Total: 432 combos

Filter (broader cluster): direction=SHORT × cap_segment=small_cap × rsi_trigger>=80

Usage:
    python tools/sub9_research/_rsi_short_overbought_sweep.py
"""
from __future__ import annotations

import sys
from datetime import date
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Line-buffered stdout for live progress
sys.stdout.reconfigure(line_buffering=True)

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sub9_research.sanity_rsi_extreme_reversal import (  # noqa: E402
    DISCOVERY_START,
    DISCOVERY_END,
    OOS_START,
    OOS_END,
    SL_LOOKBACK_15M,
    SL_BUF_SHORT,
    build_universe,
    load_5m_for_period,
    aggregate_to_15m_with_rsi,
    detect_events,
)
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402

_OUT_DIR = _REPO / "reports" / "sub9_sanity"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

RISK_PER_TRADE_RUPEES = 1000

# Sweep grid
SL_BUFFERS = [0.003, 0.005, 0.007]                  # 0.3 / 0.5 / 0.7%
T1_RS = [0.5, 1.0, 1.5]
T2_RS = [1.0, 1.5, 2.0, 3.0]
T1_QTYS = [0.0, 0.33, 0.50, 0.67]
TIME_STOPS = ["13:00", "14:30", "15:10"]


def _pf(s: pd.Series) -> float:
    g = float(s[s > 0].sum())
    l = float(-s[s < 0].sum())
    return g / l if l > 0 else float("inf")


def _sharpe_daily(pnls: pd.Series, dates: pd.Series) -> float:
    if len(pnls) == 0:
        return 0.0
    daily = pd.Series(pnls.values, index=pd.to_datetime(dates).dt.date).groupby(level=0).sum()
    if daily.size < 2 or daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std())


def _monthly(pnls: pd.Series, dates: pd.Series) -> Tuple[int, int, float, float]:
    if len(pnls) == 0:
        return (0, 0, 0.0, 0.0)
    mo = pd.to_datetime(dates).dt.strftime("%Y-%m")
    monthly = pd.Series(pnls.values, index=mo).groupby(level=0).sum()
    n_mo = int(monthly.size)
    win_mo = int((monthly > 0).sum())
    win_pct = 100.0 * win_mo / n_mo if n_mo > 0 else 0.0
    tot = float(monthly.sum())
    top_pct = (100.0 * float(monthly.abs().max()) / abs(tot)) if abs(tot) > 1e-6 else 0.0
    return (n_mo, win_mo, win_pct, top_pct)


def filter_broader_short_overbought(events: pd.DataFrame, cap_map: Dict[str, str]) -> pd.DataFrame:
    """SHORT × small_cap × rsi_trigger >= 80 — the broader cluster theme."""
    if events.empty:
        return events
    out = events.copy()
    out["cap_segment"] = out["symbol"].map(cap_map).fillna("unknown")
    keep = (
        (out["direction"] == "short")
        & (out["cap_segment"] == "small_cap")
        & (out["rsi_trigger"] >= 80.0)
    )
    return out[keep].copy()


def simulate_one_combo(events: pd.DataFrame,
                       df15_by_sym: Dict[str, pd.DataFrame],
                       sym_sess_5m: Dict[Tuple[str, date], pd.DataFrame],
                       cap_map: Dict[str, str],
                       sl_buffer: float, t1_r: float, t2_r: float,
                       t1_qty_pct: float, time_stop_hhmm: str) -> pd.DataFrame:
    """Re-simulate the given SHORT events under one combo.

    SL formula: hard_sl = max(recent 4 15m bars HIGH * (1 + sl_buffer),
                              entry * (1 + sl_buffer))     <-- min floor
    R = hard_sl - entry  (always > 0)
    T1 = entry - t1_r * R
    T2 = entry - t2_r * R
    T1 partial = t1_qty_pct
    Runner = 1 - t1_qty_pct  (BE trail after T1)
    Time stop = time_stop_hhmm (exit at bar close)

    If t1_qty_pct == 0.0, the whole position is treated as runner with no
    BE trail (hard SL stays active until exit).
    """
    trades: List[dict] = []

    for _, ev in events.iterrows():
        sym = ev["symbol"]
        sd = ev["session_date"]
        conf_date = pd.Timestamp(ev["next_date"])
        entry_price = float(ev["next_close"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        sym_15 = df15_by_sym.get(sym)
        if sym_15 is None:
            continue
        mask = sym_15["date"] <= conf_date
        lookback = sym_15.loc[mask].tail(SL_LOOKBACK_15M)
        if lookback.empty:
            continue

        # SHORT only — this sweep is locked to the small_cap × short × rsi>=80 cluster
        sl_base = float(lookback["high"].max())
        hard_sl = sl_base * (1.0 + sl_buffer)
        stop_distance = hard_sl - entry_price
        min_stop = entry_price * sl_buffer
        if stop_distance < min_stop:
            hard_sl = entry_price + min_stop
            stop_distance = min_stop
        if stop_distance <= 0:
            continue
        t1 = entry_price - t1_r * stop_distance
        t2 = entry_price - t2_r * stop_distance

        qty = max(int(RISK_PER_TRADE_RUPEES / stop_distance), 1)
        qty_t1 = int(qty * t1_qty_pct)
        qty_runner = qty - qty_t1

        bars = sym_sess_5m.get((sym, sd))
        if bars is None or bars.empty:
            continue
        walk_start = conf_date + pd.Timedelta(minutes=15)
        post = bars[bars["date"] >= walk_start]
        if post.empty:
            continue

        t1_hit = False
        t1_exit_price: Optional[float] = None
        t2_exit_price: Optional[float] = None
        sl_exit_price: Optional[float] = None
        time_exit_price: Optional[float] = None
        exit_reason: Optional[str] = None

        for _, bar in post.iterrows():
            ts = bar["date"]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            hhmm = ts.strftime("%H:%M") if hasattr(ts, "strftime") else pd.Timestamp(ts).strftime("%H:%M")

            # When t1_qty_pct == 0, the strategy is single-target: hard_sl stays
            # active forever (no BE trail).
            if t1_qty_pct > 0 and t1_hit:
                active_sl = entry_price
            else:
                active_sl = hard_sl

            # SHORT semantics
            if high >= active_sl:
                sl_exit_price = active_sl
                exit_reason = "be" if (t1_qty_pct > 0 and t1_hit) else "sl"
                break
            # T1 partial fill (only if t1_qty_pct > 0)
            if t1_qty_pct > 0 and not t1_hit and low <= t1:
                t1_hit = True
                t1_exit_price = t1
            # T2 / final target
            if low <= t2:
                t2_exit_price = t2
                exit_reason = "t2"
                break
            if hhmm >= time_stop_hhmm:
                time_exit_price = close
                exit_reason = "time"
                break

        if exit_reason is None:
            last = post.iloc[-1]
            time_exit_price = float(last["close"])
            exit_reason = "last_bar"

        # PnL composition (SHORT)
        pnl = 0.0
        if t1_qty_pct > 0 and t1_hit and t1_exit_price is not None:
            pnl += (entry_price - t1_exit_price) * qty_t1

        if t1_qty_pct > 0 and t1_hit:
            final_qty = qty_runner
        else:
            final_qty = qty

        if t2_exit_price is not None:
            final_exit = t2_exit_price
        elif sl_exit_price is not None:
            final_exit = sl_exit_price
        else:
            final_exit = time_exit_price if time_exit_price is not None else entry_price

        pnl += (entry_price - final_exit) * final_qty

        legs = []
        if t1_qty_pct > 0 and t1_hit and t1_exit_price is not None:
            legs.append((qty_t1, t1_exit_price))
        legs.append((final_qty, final_exit))
        total_q = sum(q for q, _ in legs)
        avg_exit = sum(q * p for q, p in legs) / total_q if total_q > 0 else entry_price

        fee = calc_fee(entry_price, avg_exit, qty, "SELL")
        net_pnl = pnl - fee

        trades.append({
            "session_date": sd,
            "symbol": sym,
            "net_pnl": net_pnl,
        })

    return pd.DataFrame(trades)


def run_period(label: str, start: date, end: date,
               universe: set, cap_map: Dict[str, str]):
    print(f"\n=== {label.upper()}  {start} .. {end} ===")
    big5m = load_5m_for_period(start, end, universe)
    if big5m.empty:
        return None
    df15 = aggregate_to_15m_with_rsi(big5m)
    events = detect_events(df15)
    if events.empty:
        return None

    events = filter_broader_short_overbought(events, cap_map)
    print(f"  cluster events (SHORT × small_cap × rsi>=80): {len(events):,}")
    if events.empty:
        return None

    # Build lookup structures once
    df15_by_sym = {sym: g.sort_values("date").reset_index(drop=True)
                   for sym, g in df15.groupby("symbol", sort=False)}
    sd_floor = big5m["date"].dt.floor("D")
    sym_sess_5m: Dict[Tuple[str, date], pd.DataFrame] = {}
    for (sym, sd_ts), g in big5m.groupby([big5m["symbol"], sd_floor], sort=False):
        sym_sess_5m[(sym, sd_ts.date())] = g.sort_values("date").reset_index(drop=True)

    return dict(events=events, df15_by_sym=df15_by_sym, sym_sess_5m=sym_sess_5m)


def sweep(label: str, ctx: dict, cap_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    combos = list(product(SL_BUFFERS, T1_RS, T2_RS, T1_QTYS, TIME_STOPS))
    print(f"\n  sweeping {len(combos)} combos on {label} ...")
    for idx, (sl, t1, t2, q1, ts) in enumerate(combos, 1):
        if t2 < t1:  # nonsensical
            continue
        tr = simulate_one_combo(
            ctx["events"], ctx["df15_by_sym"], ctx["sym_sess_5m"], cap_map,
            sl, t1, t2, q1, ts,
        )
        if tr.empty:
            continue
        pnl = tr["net_pnl"]
        pf = _pf(pnl)
        wr = 100.0 * float((pnl > 0).mean())
        sh = _sharpe_daily(pnl, tr["session_date"])
        n_mo, win_mo, win_pct, top_pct = _monthly(pnl, tr["session_date"])
        rows.append({
            "sl_buf": sl, "t1_r": t1, "t2_r": t2, "t1_qty": q1, "time_stop": ts,
            "n": int(len(tr)),
            "pf": pf, "wr": wr, "sh": sh,
            "n_mo": n_mo, "win_mo": win_mo, "win_mo_pct": win_pct,
            "top_mo_pct": top_pct, "net": float(pnl.sum()),
        })
        if idx % 40 == 0:
            print(f"    [{idx}/{len(combos)}] last: sl={sl:.3f} t1={t1} t2={t2} "
                  f"q1={q1} ts={ts} -> n={len(tr)} pf={pf:.3f}")
    out = pd.DataFrame(rows)
    return out


def main() -> None:
    print("=== RSI short-overbought SL/T1/T2 sweep ===")
    print(f"Grid: SL{SL_BUFFERS} x T1{T1_RS} x T2{T2_RS} x Q{T1_QTYS} x TS{TIME_STOPS}")
    universe, cap_map = build_universe()
    print(f"Universe: {len(universe):,} symbols")

    print("\n--- DISCOVERY ---")
    disc = run_period("discovery", DISCOVERY_START, DISCOVERY_END, universe, cap_map)
    if disc is None:
        print("No discovery events. Exiting.")
        return
    disc_results = sweep("discovery", disc, cap_map)
    disc_csv = _OUT_DIR / "rsi_short_overbought_sweep_discovery.csv"
    disc_results.to_csv(disc_csv, index=False)
    print(f"  wrote {disc_csv}")

    print("\n--- DISCOVERY TOP 10 BY PF (with n>=125) ---")
    disc_filt = disc_results[disc_results["n"] >= 125].sort_values("pf", ascending=False)
    print(disc_filt.head(10).to_string(index=False))

    print("\n--- OOS ---")
    oos = run_period("oos", OOS_START, OOS_END, universe, cap_map)
    if oos is None:
        print("No OOS events. Exiting.")
        return
    oos_results = sweep("oos", oos, cap_map)
    oos_csv = _OUT_DIR / "rsi_short_overbought_sweep_oos.csv"
    oos_results.to_csv(oos_csv, index=False)
    print(f"  wrote {oos_csv}")

    # Validate top-10 discovery combos on OOS
    print("\n--- DISCOVERY TOP 10 VALIDATED ON OOS ---")
    join_keys = ["sl_buf", "t1_r", "t2_r", "t1_qty", "time_stop"]
    top = disc_filt.head(20)[join_keys].copy()
    oos_top = oos_results.merge(top, on=join_keys, how="inner")
    oos_top = oos_top.sort_values("pf", ascending=False)
    print(oos_top.to_string(index=False))

    # Combined ranking: Discovery & OOS both must clear PF>=1.30, n>=50
    print("\n--- COMBOS PASSING BOTH (Disc PF>=1.30 n>=125 & OOS PF>=1.30 n>=50) ---")
    merged = disc_results.merge(oos_results, on=join_keys, suffixes=("_disc", "_oos"))
    survivors = merged[
        (merged["pf_disc"] >= 1.30) & (merged["n_disc"] >= 125)
        & (merged["pf_oos"] >= 1.30) & (merged["n_oos"] >= 50)
    ].sort_values("pf_oos", ascending=False)
    print(f"  surviving combos: {len(survivors)}")
    cols = ["sl_buf", "t1_r", "t2_r", "t1_qty", "time_stop",
            "n_disc", "pf_disc", "sh_disc", "win_mo_disc",
            "n_oos", "pf_oos", "sh_oos", "win_mo_oos"]
    print(survivors[cols].head(10).to_string(index=False))

    surv_csv = _OUT_DIR / "rsi_short_overbought_sweep_survivors.csv"
    survivors.to_csv(surv_csv, index=False)
    print(f"  wrote survivors -> {surv_csv}")


if __name__ == "__main__":
    main()
