"""Analytical target sweep on delivery_pct trades using MFE/MAE/close-at-HHMM.

For each candidate (T1_R, T2_R, time_stop_hhmm) combo, computes hypothetical
aggregate PnL using the empirical excursion data captured in the trades CSV.

This avoids re-running 24 monthly 5m feathers per parameter combo. Validate
the top 1-2 candidates with a real sanity re-run.

Logic per trade (assuming MFE-then-reverse path; conservative on stop ordering):
  if mfe_r >= T2_R: T2 hit → payoff = 0.5 * T1_R + 0.5 * T2_R (R units)
  elif mfe_r >= T1_R: T1 hit only, BE trail closes 0 on remainder → payoff = 0.5 * T1_R
  else (mfe_r < T1_R):
    if mae_r >= 1.0: stop hit → payoff = -1.0
    else: time-stop at close_at_HHMM → payoff = (entry - close_at_HHMM) / R for SHORT
                                              (close_at_HHMM - entry) / R for LONG

Fees approximated by current trade-level fees (proxies fee structure).

Usage:
    python tools/sub9_research/_target_sweep_delivery_pct.py
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402
df = pd.read_csv(REPO / "reports" / "sub9_sanity" / "nse_delivery_pct_anomaly_trades.csv")
print(f"Loaded {len(df):,} trades")

# 2D sweep grid: T1_R/T2_R x time_stop. Brief asserted TS=13:00 but never
# swept — we cross R-targets x time-stops to find the data-optimal corner.
# CSV needs close_at_HHMM columns for each time-stop (sanity script writes
# 1100/1200/1300/1400/1500/1525 after the 2026-05-13 extension).
TIME_STOPS = [1100, 1200, 1300, 1400, 1500, 1525]  # 1525 ≈ EOD (MIS auto-square)
R_GRID = [
    # (T1_R, T2_R, label_R)
    (1.0,  2.0,  "baseline 1.0R/2.0R"),
    (1.0,  1.5,  "tight T2 1.0R/1.5R"),
    (1.0,  1.25, "tight T2 1.0R/1.25R"),
    (0.75, 1.5,  "T1=p60 T2=p83 0.75R/1.5R"),
    (0.75, 1.25, "T1=p60 T2=p77 0.75R/1.25R"),
    (0.50, 1.0,  "half-targets 0.5R/1.0R"),
    (0.50, 0.75, "very tight 0.5R/0.75R"),
    (0.25, 0.75, "quarter-half 0.25R/0.75R"),
]
combos = [
    (t1, t2, ts, f"{label_R} @ TS={ts//100:02d}:{ts%100:02d}")
    for (t1, t2, label_R) in R_GRID
    for ts in TIME_STOPS
]


def compute_hypothetical_pnl(row, T1_R, T2_R, time_stop_hhmm):
    """Per-trade hypothetical net_pnl at given (T1, T2, TS) combo."""
    mfe = row["mfe_r"]
    mae = row["mae_r"]
    qty = row["qty"]
    entry = row["entry_price"]
    R_inr = row["stop_distance"]
    side = row["side"]

    # Pick close at chosen time-stop
    close_col = f"close_at_{int(time_stop_hhmm)}"
    close_ts = row.get(close_col, np.nan)

    # PARTIAL exit qty
    partial_q = max(int(qty * 0.5), 1)
    remain_q = qty - partial_q

    fee_side = "SELL" if side == "SHORT" else "BUY"

    if mfe >= T2_R:
        # T2 hit. Partial at T1, remainder at T2.
        gross = (0.5 * T1_R + 0.5 * T2_R) * qty * R_inr
        # Approximate fee = round-trip per leg, two legs
        # Use entry as proxy price; fee is small compared to gross
        if side == "SHORT":
            t1_exit = entry - T1_R * R_inr
            t2_exit = entry - T2_R * R_inr
        else:
            t1_exit = entry + T1_R * R_inr
            t2_exit = entry + T2_R * R_inr
        fee = (calc_fee(entry, t1_exit, partial_q, fee_side) +
               calc_fee(entry, t2_exit, remain_q, fee_side))
        return gross - fee, "t2"

    elif mfe >= T1_R:
        # T1 hit + BE trail. Partial at T1, remainder at BE (=entry).
        gross = 0.5 * T1_R * qty * R_inr
        if side == "SHORT":
            t1_exit = entry - T1_R * R_inr
        else:
            t1_exit = entry + T1_R * R_inr
        # Remainder closes at BE (entry)
        fee = (calc_fee(entry, t1_exit, partial_q, fee_side) +
               calc_fee(entry, entry, remain_q, fee_side))
        return gross - fee, "t1_be"

    else:
        # MFE < T1: stop or time-stop
        if mae >= 1.0:
            # Stop hit
            gross = -1.0 * qty * R_inr
            if side == "SHORT":
                exit_p = entry + R_inr
            else:
                exit_p = entry - R_inr
            fee = calc_fee(entry, exit_p, qty, fee_side)
            return gross - fee, "stop"
        else:
            # Time-stop at close_ts
            if pd.isna(close_ts):
                close_ts = row["exit_price"]
            if side == "SHORT":
                gross = (entry - close_ts) * qty
            else:
                gross = (close_ts - entry) * qty
            fee = calc_fee(entry, close_ts, qty, fee_side)
            return gross - fee, "time_stop"


def pf_of(s):
    g = s[s > 0].sum()
    l = -s[s < 0].sum()
    return float(g / l) if l > 0 else float("inf")


def sweep_one(combo):
    T1_R, T2_R, TS, label = combo
    pnls = []
    reasons = []
    for _, r in df.iterrows():
        pnl, reason = compute_hypothetical_pnl(r, T1_R, T2_R, TS)
        pnls.append(pnl)
        reasons.append(reason)
    arr = np.array(pnls)
    return {
        "T1_R": T1_R, "T2_R": T2_R, "TS": TS, "label": label,
        "n": len(arr),
        "PF": pf_of(pd.Series(arr)),
        "WR": float(100 * (arr > 0).mean()),
        "NET": float(arr.sum()),
        "avg": float(arr.mean()),
        "t2_pct": 100 * pd.Series(reasons).eq("t2").mean(),
        "t1_be_pct": 100 * pd.Series(reasons).eq("t1_be").mean(),
        "stop_pct": 100 * pd.Series(reasons).eq("stop").mean(),
        "time_stop_pct": 100 * pd.Series(reasons).eq("time_stop").mean(),
    }


print("\n=== ANALYTICAL TARGET SWEEP — ALL trades ===")
print(f"{'T1':>5} {'T2':>5} {'TS':>5}  {'n':>5}  {'PF':>5}  {'WR':>5}  {'NET':>11}  {'t2%':>5}  {'t1_be%':>6}  {'stop%':>5}  {'TS%':>5}  label")
print("-" * 140)
results = []
for c in combos:
    r = sweep_one(c)
    results.append(r)
    print(f"{r['T1_R']:>5.2f} {r['T2_R']:>5.2f} {r['TS']:>5}  {r['n']:>5,}  "
          f"{r['PF']:>5.3f}  {r['WR']:>4.1f}%  Rs.{r['NET']:>9,.0f}  "
          f"{r['t2_pct']:>4.1f}%  {r['t1_be_pct']:>5.1f}%  {r['stop_pct']:>4.1f}%  {r['time_stop_pct']:>4.1f}%  {r['label']}")

# Repeat for SHORT side only (the better-performing side)
print("\n=== ANALYTICAL TARGET SWEEP — SHORT ONLY (n=1,505) ===")
df_short = df[df["side"] == "SHORT"].reset_index(drop=True)
df_full = df.copy()
df = df_short
print(f"{'T1':>5} {'T2':>5} {'TS':>5}  {'n':>5}  {'PF':>5}  {'WR':>5}  {'NET':>11}  {'t2%':>5}  {'t1_be%':>6}  {'stop%':>5}  {'TS%':>5}  label")
print("-" * 140)
for c in combos:
    r = sweep_one(c)
    print(f"{r['T1_R']:>5.2f} {r['T2_R']:>5.2f} {r['TS']:>5}  {r['n']:>5,}  "
          f"{r['PF']:>5.3f}  {r['WR']:>4.1f}%  Rs.{r['NET']:>9,.0f}  "
          f"{r['t2_pct']:>4.1f}%  {r['t1_be_pct']:>5.1f}%  {r['stop_pct']:>4.1f}%  {r['time_stop_pct']:>4.1f}%  {r['label']}")

# Restore for cell scan
df = df_full
print("\n=== Best combo applied to cells ===")
best = max(results, key=lambda r: r["PF"])
print(f"Best ALL-trades combo: T1={best['T1_R']} T2={best['T2_R']} TS={best['TS']}  PF={best['PF']:.3f}")
