"""Parameter sweep on C4a/C4b: vary SL buffer, T1/T2 R-multiples, partial qty,
time stop. Find combos that show ship-eligible edge under per-regime + per-month
stability checks.

Reuses the simulation loop from sanity_gap_down_intraday but varies:
  SL_BUFFER:    [0.05%, 0.10%, 0.15%, 0.20%]    (4)
  T2_R:         [1.0, 1.5, 2.0, 2.5]             (4)
  T1_R:         [0.5]  (fixed — standard project pattern)
  T1_QTY_PCT:   [0.0, 0.5]                        (2)  — all-in vs partial
  TIME_STOP:    ["11:30", "13:00", "14:30"]       (3)

Total: 4 × 4 × 2 × 3 = 96 combos × 2 setups (C4a/C4b) = 192 runs.

For each combo, output:
  - aggregate PF (pre_rule, post_rule)
  - n trades per regime
  - % months winning
  - top-month concentration

Then filter to combos where post_rule passes realistic ship criteria:
  PF >= 1.30, n >= 125, winning months >= 55%, top-month NET share < 40%.

Usage:
    python -m tools.research.post_sebi.sweep_gap_down_params
"""
from __future__ import annotations

import sys
from datetime import date
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402

_FEATHER_DIR = _REPO / "backtest-cache-download" / "monthly"
_NIFTY50_CSV = _REPO / "assets" / "ind_nifty50list.csv"
_OUT_DIR = _REPO / "reports" / "research" / "post_sebi" / "gap_down_intraday"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

GAP_PCT_MAX = -0.005       # gap_pct <= -0.5% (locked)
RISK_PER_TRADE_RUPEES = 1000
RULE_DATE = date(2025, 2, 1)

# Pre-registered ship criteria (post-realistic-bars adjustment from prior discussion)
SHIP_PF = 1.30
SHIP_N = 125
SHIP_WIN_PCT = 55.0
SHIP_TOPMONTH_PCT = 40.0


def load_nifty50() -> set:
    df = pd.read_csv(_NIFTY50_CSV)
    return set(df["Symbol"].dropna().astype(str).str.strip().unique())


def load_all_5m(symbols: set) -> pd.DataFrame:
    feathers = sorted(_FEATHER_DIR.glob("20*_5m_enriched.feather"))
    parts = []
    for fp in feathers:
        try:
            df = pd.read_feather(fp)
            df = df[df["symbol"].isin(symbols)]
            if not df.empty:
                parts.append(df[["symbol", "date", "open", "high", "low", "close", "volume"]].copy())
        except Exception:
            pass
    big = pd.concat(parts, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"])
    big = big.sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    big["hhmm"] = big["date"].dt.strftime("%H:%M")
    return big


def build_pdc_map(big: pd.DataFrame) -> Dict:
    daily = big.groupby(["symbol", "d"]).agg(last_close=("close", "last")).reset_index()
    daily["pdc"] = daily.groupby("symbol")["last_close"].shift(1)
    daily = daily.dropna(subset=["pdc"])
    return {(r["symbol"], r["d"]): float(r["pdc"]) for _, r in daily.iterrows()}


def precompute_events(big: pd.DataFrame, pdc_map: Dict) -> List[Dict]:
    """Pre-compute the event list (one row per qualifying gap-down session).
    This is invariant across parameter sweeps — sweep only varies SL/T1/T2/time-stop."""
    events = []
    sessions = big.groupby(["symbol", "d"])
    for (sym, sd), grp in sessions:
        pdc = pdc_map.get((sym, sd))
        if pdc is None or pdc <= 0:
            continue
        grp = grp.sort_values("date")
        bar_915 = grp[grp["hhmm"] == "09:15"]
        bar_920 = grp[grp["hhmm"] == "09:20"]
        if bar_915.empty or bar_920.empty:
            continue

        open_915 = float(bar_915.iloc[0]["open"])
        close_915 = float(bar_915.iloc[0]["close"])
        high_915 = float(bar_915.iloc[0]["high"])
        low_915 = float(bar_915.iloc[0]["low"])
        open_920 = float(bar_920.iloc[0]["open"])
        close_920 = float(bar_920.iloc[0]["close"])
        high_920 = float(bar_920.iloc[0]["high"])
        low_920 = float(bar_920.iloc[0]["low"])

        if open_915 <= 0:
            continue
        gap_pct = (open_915 - pdc) / pdc
        if gap_pct > GAP_PCT_MAX:
            continue

        bar1_up = close_920 > open_920
        bar1_dn = close_920 < open_920
        if not (bar1_up or bar1_dn):
            continue

        if bar1_up:
            setup = "c4a"
            direction = "long"
        else:
            setup = "c4b"
            direction = "short"

        post = grp[grp["date"] > bar_920.iloc[0]["date"]]
        if post.empty:
            continue

        events.append({
            "symbol": sym, "session_date": sd, "pdc": pdc,
            "open_915": open_915, "close_920": close_920,
            "low_915": low_915, "high_915": high_915,
            "low_920": low_920, "high_920": high_920,
            "gap_pct": gap_pct, "setup": setup, "direction": direction,
            "post_bars": post.to_dict("records"),
        })
    return events


def simulate_one_event(ev: Dict, sl_buffer: float, t1_r: float, t2_r: float,
                       t1_qty_pct: float, time_stop: str) -> Dict:
    """Simulate one event with a specific parameter combo."""
    entry_price = ev["close_920"]
    if ev["direction"] == "long":
        combined_low = min(ev["low_915"], ev["low_920"])
        hard_sl = combined_low * (1 - sl_buffer)
        stop_distance = entry_price - hard_sl
    else:
        combined_high = max(ev["high_915"], ev["high_920"])
        hard_sl = combined_high * (1 + sl_buffer)
        stop_distance = hard_sl - entry_price

    if stop_distance <= 0:
        return None
    R = stop_distance

    if ev["direction"] == "long":
        t1 = entry_price + t1_r * R
        t2 = entry_price + t2_r * R
    else:
        t1 = entry_price - t1_r * R
        t2 = entry_price - t2_r * R

    qty = max(int(RISK_PER_TRADE_RUPEES / R), 1)
    qty_at_t1 = int(qty * t1_qty_pct)
    qty_runner = qty - qty_at_t1

    t1_hit = False
    t1_exit_price = None
    t2_exit_price = None
    sl_exit_price = None
    time_exit_price = None

    for bar in ev["post_bars"]:
        ts_str = pd.Timestamp(bar["date"]).strftime("%H:%M")
        high = float(bar["high"])
        low = float(bar["low"])

        if ev["direction"] == "long":
            if low <= hard_sl:
                sl_exit_price = entry_price if t1_hit else hard_sl
                break
            if not t1_hit and t1_qty_pct > 0 and high >= t1:
                t1_hit = True
                t1_exit_price = t1
            if high >= t2:
                t2_exit_price = t2
                break
            if ts_str >= time_stop:
                time_exit_price = float(bar["close"])
                break
        else:
            if high >= hard_sl:
                sl_exit_price = entry_price if t1_hit else hard_sl
                break
            if not t1_hit and t1_qty_pct > 0 and low <= t1:
                t1_hit = True
                t1_exit_price = t1
            if low <= t2:
                t2_exit_price = t2
                break
            if ts_str >= time_stop:
                time_exit_price = float(bar["close"])
                break

    pnl = 0.0
    if t1_hit and t1_exit_price is not None:
        if ev["direction"] == "long":
            pnl += (t1_exit_price - entry_price) * qty_at_t1
        else:
            pnl += (entry_price - t1_exit_price) * qty_at_t1

    final_runner_qty = qty_runner if t1_hit else qty
    if t2_exit_price is not None:
        if ev["direction"] == "long":
            pnl += (t2_exit_price - entry_price) * final_runner_qty
        else:
            pnl += (entry_price - t2_exit_price) * final_runner_qty
        avg_exit = t2_exit_price
    elif sl_exit_price is not None:
        if ev["direction"] == "long":
            pnl += (sl_exit_price - entry_price) * final_runner_qty
        else:
            pnl += (entry_price - sl_exit_price) * final_runner_qty
        avg_exit = sl_exit_price
    elif time_exit_price is not None:
        if ev["direction"] == "long":
            pnl += (time_exit_price - entry_price) * final_runner_qty
        else:
            pnl += (entry_price - time_exit_price) * final_runner_qty
        avg_exit = time_exit_price
    else:
        last_close = float(ev["post_bars"][-1]["close"])
        if ev["direction"] == "long":
            pnl += (last_close - entry_price) * final_runner_qty
        else:
            pnl += (entry_price - last_close) * final_runner_qty
        avg_exit = last_close

    leg = "BUY" if ev["direction"] == "long" else "SELL"
    fee = calc_fee(entry_price, avg_exit, qty, leg)
    net_pnl = pnl - fee

    return {
        "setup": ev["setup"],
        "session_date": ev["session_date"],
        "net_pnl": net_pnl,
    }


def pf_of(s: pd.Series) -> float:
    g = s[s > 0].sum()
    l = -s[s <= 0].sum()
    return g / l if l > 0 else float("inf")


def evaluate_combo(events: List[Dict], setup: str,
                   sl: float, t1r: float, t2r: float, t1q: float, ts: str) -> Dict:
    """Run all events for one setup × parameter combo, return aggregate + per-regime metrics."""
    setup_events = [ev for ev in events if ev["setup"] == setup]
    trades = []
    for ev in setup_events:
        tr = simulate_one_event(ev, sl, t1r, t2r, t1q, ts)
        if tr is not None:
            trades.append(tr)
    if not trades:
        return None
    df = pd.DataFrame(trades)
    df["regime"] = df["session_date"].apply(lambda d: "pre" if d < RULE_DATE else "post")
    df["month"] = pd.to_datetime(df["session_date"]).dt.strftime("%Y-%m")

    def block(sub):
        if sub.empty:
            return dict(n=0, pf=0.0, wr=0.0, net=0.0, win_months=0,
                        total_months=0, win_month_pct=0.0, top_month_pct=0.0)
        net = sub["net_pnl"].sum()
        wins = sub[sub["net_pnl"] > 0]
        gw = wins["net_pnl"].sum()
        gl = -sub[sub["net_pnl"] <= 0]["net_pnl"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        wr = 100 * len(wins) / len(sub)
        # Monthly stats
        monthly_net = sub.groupby("month")["net_pnl"].sum()
        win_months = int((monthly_net > 0).sum())
        total_months = len(monthly_net)
        win_month_pct = 100.0 * win_months / total_months if total_months > 0 else 0.0
        # Top-month concentration (signed): top month's net / total net
        if abs(net) > 0:
            top_month_pct = 100.0 * float(monthly_net.abs().max()) / abs(net)
        else:
            top_month_pct = 0.0
        return dict(
            n=len(sub), pf=pf, wr=wr, net=float(net),
            win_months=win_months, total_months=total_months,
            win_month_pct=win_month_pct, top_month_pct=top_month_pct,
        )

    return {
        "setup": setup, "sl": sl, "t1r": t1r, "t2r": t2r, "t1q": t1q, "ts": ts,
        "pre": block(df[df["regime"] == "pre"]),
        "post": block(df[df["regime"] == "post"]),
        "all": block(df),
    }


def main():
    print("Loading NIFTY-50 + 5m bars + pre-computing events ...")
    symbols = load_nifty50()
    big = load_all_5m(symbols)
    pdc_map = build_pdc_map(big)
    events = precompute_events(big, pdc_map)
    print(f"  events: {len(events):,}")
    print(f"  C4a long:  {sum(1 for e in events if e['setup']=='c4a'):,}")
    print(f"  C4b short: {sum(1 for e in events if e['setup']=='c4b'):,}")

    sweep_grid = list(product(
        [0.0005, 0.001, 0.0015, 0.002],            # sl_buffer
        [0.5],                                       # t1_r (fixed)
        [1.0, 1.5, 2.0, 2.5],                       # t2_r
        [0.0, 0.5],                                  # t1_qty_pct (all-in vs partial)
        ["11:30", "13:00", "14:30"],                 # time_stop
    ))
    print(f"\nSweeping {len(sweep_grid)} combos × 2 setups = {len(sweep_grid)*2} runs ...")

    results = []
    for i, (sl, t1r, t2r, t1q, ts) in enumerate(sweep_grid):
        for setup in ("c4a", "c4b"):
            r = evaluate_combo(events, setup, sl, t1r, t2r, t1q, ts)
            if r is not None:
                results.append(r)
        if (i + 1) % 12 == 0:
            print(f"  ... {i+1}/{len(sweep_grid)} combos done")

    flat = []
    for r in results:
        flat.append({
            "setup": r["setup"],
            "sl_pct": round(r["sl"]*100, 2),
            "t1r": r["t1r"],
            "t2r": r["t2r"],
            "t1q_pct": int(r["t1q"]*100),
            "time_stop": r["ts"],
            "pre_n": r["pre"]["n"],
            "pre_pf": round(r["pre"]["pf"], 3) if r["pre"]["pf"] != float("inf") else 999.0,
            "pre_wr": round(r["pre"]["wr"], 1),
            "pre_win_mo_pct": round(r["pre"]["win_month_pct"], 1),
            "pre_top_mo_pct": round(r["pre"]["top_month_pct"], 1),
            "post_n": r["post"]["n"],
            "post_pf": round(r["post"]["pf"], 3) if r["post"]["pf"] != float("inf") else 999.0,
            "post_wr": round(r["post"]["wr"], 1),
            "post_net": round(r["post"]["net"], 0),
            "post_win_mo_pct": round(r["post"]["win_month_pct"], 1),
            "post_top_mo_pct": round(r["post"]["top_month_pct"], 1),
        })
    out_df = pd.DataFrame(flat)
    out_path = _OUT_DIR / "sweep_gap_down_params_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Filter to ship-eligible (post-rule)
    ship = out_df[
        (out_df["post_pf"] >= SHIP_PF)
        & (out_df["post_n"] >= SHIP_N)
        & (out_df["post_wr"] >= 50.0)
        & (out_df["post_win_mo_pct"] >= SHIP_WIN_PCT)
        & (out_df["post_top_mo_pct"] < SHIP_TOPMONTH_PCT)
    ].copy()

    print(f"\n{'='*82}")
    print(f"SHIP-ELIGIBLE COMBOS (post-rule):")
    print(f"  PF >= {SHIP_PF}, n >= {SHIP_N}, WR >= 50%, "
          f"win_months >= {SHIP_WIN_PCT}%, top_month < {SHIP_TOPMONTH_PCT}%")
    print('=' * 82)
    if ship.empty:
        print("  NONE")
    else:
        ship = ship.sort_values(["setup", "post_pf"], ascending=[True, False])
        print(ship.to_string(index=False))

    # Best per setup regardless of all gates
    print(f"\n{'='*82}")
    print("TOP 5 COMBOS PER SETUP (by post_rule PF, requires post_n >= 125):")
    print('=' * 82)
    for setup in ("c4a", "c4b"):
        sub = out_df[(out_df["setup"] == setup) & (out_df["post_n"] >= 125)]
        sub = sub.sort_values("post_pf", ascending=False).head(5)
        print(f"\n--- {setup} ---")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
