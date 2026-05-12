"""SL / time-stop / partial-exit sweep for circuit_t1_fade_short.

Re-simulates the existing trigger CSV under a grid of
(stop_buffer_pct x min_stop_pct x time_stop_hhmm x partial_mode) combos.

CRITICAL FINDING DURING DEV: the original sanity script (PF=1.404) uses
NO T1 partial — it rides the trade all the way to T2 (full gap fill) or
SL. But production config has t1_qty_pct=0.5 — meaning production runs a
DIFFERENT exit logic than what was validated. This sweep verifies what
the partial-exit choice actually costs vs the validated baseline.

Trigger source:
  reports/sub9_sanity/circuit_t1_fade_short_trades.csv  (Discovery 2024)
  reports/sub9_sanity/circuit_t1_fade_short_trades_holdout.csv  (Holdout)

Usage:
    python tools/sub9_research/_circuit_t1_sl_target_sweep.py --period discovery
    python tools/sub9_research/_circuit_t1_sl_target_sweep.py --period holdout
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402

ENTRY_HHMM = "10:30"
RISK_PER_TRADE_RUPEES = 1000


# ---- Sweep grid ----
STOP_BUFFER_PCTS = [0.10, 0.25, 0.50, 1.0, 2.0]   # production = 0.5
MIN_STOP_PCTS    = [0.50, 1.0, 1.5, 2.0]          # production = 1.0
TIME_STOPS       = ["12:30", "13:30", "14:30", "15:10"]  # production = 15:10
# partial_mode: ("all_in", "partial_50_be_trail", "partial_50_no_trail")
PARTIAL_MODES    = ["all_in", "partial_50_be_trail", "partial_50_no_trail"]

# Sanity baseline (the one that produced PF=1.404):
#   stop_buffer=0.5, min_stop=1.0, time_stop=15:10, partial=all_in
SANITY_BASELINE = (0.50, 1.0, "15:10", "all_in")
# Production current:
#   stop_buffer=0.5, min_stop=1.0, time_stop=15:10, partial=partial_50_be_trail (t1_qty_pct=0.5)
PROD_CURRENT = (0.50, 1.0, "15:10", "partial_50_be_trail")


def pf_of(s: pd.Series) -> float:
    g = s[s > 0].sum()
    l = -s[s < 0].sum()
    return float(g / l) if l > 0 else float("inf")


def load_triggers(period: str) -> pd.DataFrame:
    if period == "discovery":
        csv = _REPO / "reports" / "sub9_sanity" / "circuit_t1_fade_short_trades.csv"
    elif period == "holdout":
        csv = _REPO / "reports" / "sub9_sanity" / "circuit_t1_fade_short_trades_holdout.csv"
    else:
        raise ValueError(f"Unknown period: {period}")
    df = pd.read_csv(csv)
    df["T1_entry_date"] = pd.to_datetime(df["T1_entry_date"]).dt.date
    df["symbol_bare"] = df["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    print(f"  loaded {len(df)} triggers from {csv.name}")
    return df


def load_forward_bars(triggers: pd.DataFrame) -> Dict[Tuple[str, date], pd.DataFrame]:
    months_needed = set((d.year, d.month) for d in triggers["T1_entry_date"].unique())
    print(f"  loading 5m feathers for {len(months_needed)} months ...")
    syms_needed = set(triggers["symbol_bare"].unique())
    print(f"  unique symbols to load: {len(syms_needed)}")

    bars_per_key: Dict[Tuple[str, date], pd.DataFrame] = {}
    cols = ["date", "symbol", "high", "low", "close"]
    for yyyy, mm in sorted(months_needed):
        fp = _REPO / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
        if not fp.exists():
            print(f"    miss: {fp.name}")
            continue
        m = pd.read_feather(fp, columns=cols)
        m = m[m["symbol"].isin(syms_needed)]
        if m.empty:
            continue
        m["d"] = m["date"].dt.date
        m["hhmm"] = m["date"].dt.strftime("%H:%M")
        m = m[m["hhmm"] >= ENTRY_HHMM]
        for (sym, d), g in m.groupby(["symbol", "d"]):
            g = g.sort_values("date")
            bars_per_key[(sym, d)] = g[["hhmm", "high", "low", "close", "date"]].reset_index(drop=True)
    print(f"  loaded {len(bars_per_key)} (symbol, date) bar groups")
    return bars_per_key


def simulate_one(
    row: pd.Series,
    forward: Optional[pd.DataFrame],
    stop_buffer_pct: float,
    min_stop_pct: float,
    time_stop_hhmm: str,
    partial_mode: str,
) -> Optional[Dict]:
    """Re-simulate this one trade under the given sweep params.

    partial_mode:
      - "all_in": no partial exit; ride to SL/T2/time_stop
      - "partial_50_be_trail": 50% at T1 (=t1_open), breakeven trail on rest
      - "partial_50_no_trail": 50% at T1, remainder rides to SL/T2/time_stop
    """
    if forward is None or forward.empty:
        return None

    t0_close = float(row["t0_close"])
    t1_open = float(row["t1_open"])
    t1_high = float(row["t1_high"])
    entry_price = float(row["entry_price"])

    sl_from_high = t1_high * (1.0 + stop_buffer_pct / 100.0)
    sl_from_min = entry_price * (1.0 + min_stop_pct / 100.0)
    hard_sl = max(sl_from_high, sl_from_min)
    stop_distance = hard_sl - entry_price
    if stop_distance <= 0:
        return None

    t1_target = t1_open
    t2_target = t0_close

    qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

    fw_hhmm = forward["hhmm"].values
    fw_high = forward["high"].values.astype(float)
    fw_low = forward["low"].values.astype(float)
    fw_close = forward["close"].values.astype(float)
    fw_dates = forward["date"].values

    # Skip the entry bar itself
    start_i = 0
    for i, h in enumerate(fw_hhmm):
        if h > ENTRY_HHMM:
            start_i = i
            break
    else:
        return None

    use_partial = partial_mode != "all_in"
    use_be_trail = partial_mode == "partial_50_be_trail"

    hit_t1 = False
    t1_exit_price: Optional[float] = None
    exit_ts = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    for i in range(start_i, len(fw_hhmm)):
        h = fw_hhmm[i]
        high = fw_high[i]
        low = fw_low[i]
        close_b = fw_close[i]

        # Stop check (SL trails to BE only if use_be_trail AND hit_t1)
        active_sl = (entry_price if (hit_t1 and use_be_trail) else hard_sl)
        if high >= active_sl:
            exit_ts = fw_dates[i]
            exit_price = active_sl
            exit_reason = "breakeven_trail" if (hit_t1 and use_be_trail) else "stop"
            break
        # T1 partial (if enabled and not yet hit)
        if use_partial and (not hit_t1) and (low <= t1_target):
            hit_t1 = True
            t1_exit_price = t1_target
        # T2 hit (after T1 if partial, or any time if all_in)
        target_to_test = t2_target
        if low <= target_to_test:
            exit_ts = fw_dates[i]
            exit_price = target_to_test
            exit_reason = "t2"
            break
        # Time stop
        if h >= time_stop_hhmm:
            exit_ts = fw_dates[i]
            exit_price = close_b
            exit_reason = "time_stop"
            break

    if exit_price is None:
        exit_ts = fw_dates[-1]
        exit_price = float(fw_close[-1])
        exit_reason = "eod"

    if use_partial and hit_t1:
        qty_t1 = qty // 2
        qty_t2 = qty - qty_t1
        pnl_t1 = (entry_price - t1_exit_price) * qty_t1
        pnl_t2 = (entry_price - exit_price) * qty_t2
        realized_pnl = pnl_t1 + pnl_t2
        fee = (calc_fee(entry_price, t1_exit_price, qty_t1, "SELL")
               + calc_fee(entry_price, exit_price, qty_t2, "SELL"))
    else:
        realized_pnl = (entry_price - exit_price) * qty
        fee = calc_fee(entry_price, exit_price, qty, "SELL")
    net_pnl = realized_pnl - fee

    return {
        "T1_entry_date": row["T1_entry_date"],
        "symbol": row["symbol"],
        "cap_segment": row.get("cap_segment", ""),
        "exit_reason": exit_reason,
        "net_pnl": net_pnl,
    }


def sweep(triggers: pd.DataFrame, bars: Dict[Tuple[str, date], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    combos = list(product(STOP_BUFFER_PCTS, MIN_STOP_PCTS, TIME_STOPS, PARTIAL_MODES))
    print(f"  running {len(combos)} combos x {len(triggers)} trades = {len(combos)*len(triggers):,} simulations")

    for combo_i, (sb, ms, ts, pm) in enumerate(combos):
        trades = []
        for _, r in triggers.iterrows():
            key = (r["symbol_bare"], r["T1_entry_date"])
            forward = bars.get(key)
            t = simulate_one(r, forward, sb, ms, ts, pm)
            if t is not None:
                trades.append(t)
        if not trades:
            continue
        tdf = pd.DataFrame(trades)
        n = len(tdf)
        net_sum = tdf["net_pnl"].sum()
        pf = pf_of(tdf["net_pnl"])
        wr = (tdf["net_pnl"] > 0).mean() * 100
        daily = tdf.groupby("T1_entry_date")["net_pnl"].sum()
        sharpe = (daily.mean() / daily.std()) if daily.std() > 0 else 0.0
        ex = tdf["exit_reason"].value_counts(normalize=True) * 100
        rows.append(dict(
            stop_buffer_pct=sb,
            min_stop_pct=ms,
            time_stop=ts,
            partial_mode=pm,
            n=n,
            net=net_sum,
            pf=pf,
            wr=wr,
            sharpe=sharpe,
            pct_stop=ex.get("stop", 0) + ex.get("breakeven_trail", 0),
            pct_t2=ex.get("t2", 0),
            pct_time=ex.get("time_stop", 0) + ex.get("eod", 0),
            is_sanity_baseline=(sb, ms, ts, pm) == SANITY_BASELINE,
            is_prod_current=(sb, ms, ts, pm) == PROD_CURRENT,
        ))
        if (combo_i + 1) % 30 == 0:
            print(f"    [{combo_i+1}/{len(combos)}]  sb={sb} ms={ms} ts={ts} pm={pm}  PF={pf:.3f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["discovery", "holdout"], required=True)
    args = ap.parse_args()

    triggers = load_triggers(args.period)
    bars = load_forward_bars(triggers)
    results = sweep(triggers, bars)

    results = results.sort_values("pf", ascending=False).reset_index(drop=True)

    out_csv = _REPO / "reports" / "sub9_sanity" / f"_circuit_t1_sl_target_sweep_{args.period}.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nResults: {out_csv}")
    print()
    # Print baselines first, then top 20
    print("=== BASELINE REFERENCES ===")
    base = results[results["is_sanity_baseline"] | results["is_prod_current"]]
    print(f"{'sb':>4} {'ms':>4} {'ts':>6} {'partial_mode':>22} {'n':>4} {'PF':>6} {'WR':>5} {'Sharpe':>7} {'%stop':>6} {'%t2':>5} {'%time':>6} {'net':>12} tag")
    for _, r in base.iterrows():
        tag = " <-SANITY" if r["is_sanity_baseline"] else (" <-PROD" if r["is_prod_current"] else "")
        print(f"{r['stop_buffer_pct']:>4} {r['min_stop_pct']:>4} {r['time_stop']:>6} {r['partial_mode']:>22} {int(r['n']):>4} {r['pf']:>6.3f} {r['wr']:>4.1f}% {r['sharpe']:>7.3f} {r['pct_stop']:>5.1f}% {r['pct_t2']:>4.1f}% {r['pct_time']:>5.1f}% Rs.{r['net']:>9,.0f}{tag}")

    print("\n=== TOP 25 BY PF ===")
    for _, r in results.head(25).iterrows():
        tag = " <-SANITY" if r["is_sanity_baseline"] else (" <-PROD" if r["is_prod_current"] else "")
        print(f"{r['stop_buffer_pct']:>4} {r['min_stop_pct']:>4} {r['time_stop']:>6} {r['partial_mode']:>22} {int(r['n']):>4} {r['pf']:>6.3f} {r['wr']:>4.1f}% {r['sharpe']:>7.3f} {r['pct_stop']:>5.1f}% {r['pct_t2']:>4.1f}% {r['pct_time']:>5.1f}% Rs.{r['net']:>9,.0f}{tag}")


if __name__ == "__main__":
    main()
