"""Day-walk simulation of real slot configs on a fixed pool. No closed form.

The fixed-capital table in rank_ablation_close_dn_overnight.py computes
position size as capital/(lock*k) and multiplies by the trade count. That
assumes every slot is occupied every day. It is not a simulation, and on a book
whose fire count varies 0-14 per day the assumption does the work: a wide config
looks bad partly because the formula charges it for slots it never fills.

This walks the actual fire sequence day by day, holds each position for the real
settlement lock, and only deploys capital it actually has. It answers the
question in the form it is actually asked: on Rs300k, is 6 trades/day at Rs25k
better or worse than 3/day at Rs50k?

Ledger is the 5-bar family (no 15:25 look-ahead); net_pnl_inr already carries
the corrected CNC rate card, so returns are scaled per rupee of notional.

Usage:
    python tools/sub9_research/slot_config_sim_close_dn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from tools.sub9_research.rank_ablation_close_dn_overnight import (  # noqa: E402
    load_cell, SPLITS, PNL_COL,
)

LEDGER_NOTIONAL = 100_000.0     # sanity script's fixed per-trade notional
CAPITAL = 300_000.0
LOCK_DAYS = 2                   # buy T, sell T+1 open, cash settles T+2


def simulate(df: pd.DataFrame, new_per_day: int, pos_size: float,
             capital: float = CAPITAL, lock: int = LOCK_DAYS) -> dict:
    """Walk days in order. Take up to `new_per_day` fires if cash allows."""
    df = df.sort_values(["signal_date", "signed_vol_ratio"])
    days = sorted(df["signal_date"].unique())
    open_pos: list[tuple] = []          # (release_index, cash)
    taken = missed_cash = missed_cap = 0
    pnl = 0.0
    deployed_samples = []
    for i, day in enumerate(days):
        open_pos = [p for p in open_pos if p[0] > i]        # settled -> cash back
        used = sum(p[1] for p in open_pos)
        todays = df[df["signal_date"] == day]
        n_new = 0
        for r in todays.itertuples():
            if n_new >= new_per_day:
                missed_cap += 1
                continue
            if used + pos_size > capital:
                missed_cash += 1
                continue
            ret = getattr(r, PNL_COL) / LEDGER_NOTIONAL     # fractional, size-free
            pnl += ret * pos_size
            open_pos.append((i + lock, pos_size))
            used += pos_size
            n_new += 1
            taken += 1
        deployed_samples.append(used)
    mean_dep = sum(deployed_samples) / len(deployed_samples) if deployed_samples else 0.0
    return dict(taken=taken, missed_cap=missed_cap, missed_cash=missed_cash,
                pnl=pnl, mean_deployed=mean_dep,
                utilisation=100 * mean_dep / capital)


def main() -> int:
    configs = [
        ("3/day x Rs50k  (production)", 3, 50_000.0),
        ("6/day x Rs25k  (proposal)", 6, 25_000.0),
        ("4/day x Rs37.5k", 4, 37_500.0),
        ("2/day x Rs75k", 2, 75_000.0),
        ("12/day x Rs12.5k", 12, 12_500.0),
        ("6/day x Rs50k  (needs Rs600k)", 6, 50_000.0),
    ]
    print("Day-walk simulation | pool Rs%s | %d-day settlement lock" % (
        format(CAPITAL, ",.0f"), LOCK_DAYS))
    print("ledger: 5-bar (no 15:25 look-ahead)\n")
    grand = {}
    for split in SPLITS:
        df = load_cell(split)
        if df.empty:
            continue
        days = df["signal_date"].nunique()
        print("=== %s | %d fires / %d days (%.1f per day) ===" % (
            split.upper(), len(df), days, len(df) / days))
        print("  %-30s %6s %8s %8s %12s %9s" % (
            "config", "taken", "no-slot", "no-cash", "net Rs", "util%"))
        for label, k, size in configs:
            r = simulate(df, k, size)
            grand.setdefault(label, []).append(r["pnl"])
            print("  %-30s %6d %8d %8d %12s %8.0f%%" % (
                label, r["taken"], r["missed_cap"], r["missed_cash"],
                format(r["pnl"], "+,.0f"), r["utilisation"]))
        print()
    print("=== total across splits ===")
    base = sum(grand.get("3/day x Rs50k  (production)", []))
    for label, vals in grand.items():
        tot = sum(vals)
        print("  %-30s %12s   vs production %+.1f%%" % (
            label, format(tot, "+,.0f"), 100 * (tot / base - 1) if base else 0))
    dynamic_table()
    return 0



def simulate_dynamic(df: pd.DataFrame, new_per_day: int, min_size: float,
                     capital: float = CAPITAL, lock: int = LOCK_DAYS,
                     target_slots: int | None = None) -> dict:
    """Deploy the FULL pool across whatever is actually held.

    The fixed-size configs leave cash idle whenever fewer names fire than the
    cap allows. Here each new position is sized from the cash actually free and
    the slots realistically still to be filled, so a wide cap buys breadth
    WITHOUT the utilisation penalty — which is the thing the fixed-size
    comparison conflates.

    `min_size` floors the position so it never drops into the fee-inefficient
    zone (Zerodha's Rs20/order cap stops helping below ~Rs67k notional; the
    live-measured CNC rate is flat at 0.22% so the floor is about order
    economics, not the rate card).
    """
    df = df.sort_values(["signal_date", "signed_vol_ratio"])
    days = sorted(df["signal_date"].unique())
    slots = target_slots or (new_per_day * lock)
    open_pos: list[tuple] = []
    taken = missed = 0
    pnl = 0.0
    dep = []
    for i, day in enumerate(days):
        open_pos = [p for p in open_pos if p[0] > i]
        used = sum(p[1] for p in open_pos)
        todays = df[df["signal_date"] == day]
        n_new = 0
        for r in todays.itertuples():
            if n_new >= new_per_day:
                missed += 1
                continue
            free_slots = max(1, slots - len(open_pos))
            size = max(min_size, (capital - used) / free_slots)
            if used + size > capital:
                missed += 1
                continue
            pnl += (getattr(r, PNL_COL) / LEDGER_NOTIONAL) * size
            open_pos.append((i + lock, size))
            used += size
            n_new += 1
            taken += 1
        dep.append(used)
    return dict(taken=taken, missed=missed, pnl=pnl,
                utilisation=100 * (sum(dep) / len(dep)) / capital if dep else 0)


def dynamic_table() -> None:
    print("\n=== DYNAMIC sizing: fill the pool across whatever is held ===")
    print("   (position = free cash / remaining slots, floored at min_size)\n")
    cfgs = [("3/day, floor Rs40k", 3, 40_000.0),
            ("4/day, floor Rs35k", 4, 35_000.0),
            ("6/day, floor Rs30k", 6, 30_000.0),
            ("6/day, floor Rs25k", 6, 25_000.0),
            ("8/day, floor Rs25k", 8, 25_000.0)]
    tot = {}
    for split in SPLITS:
        df = load_cell(split)
        if df.empty:
            continue
        print("  === %s ===" % split.upper())
        print("    %-22s %6s %8s %12s %8s" % ("config", "taken", "missed", "net Rs", "util%"))
        for lab, k, floor in cfgs:
            r = simulate_dynamic(df, k, floor)
            tot.setdefault(lab, []).append(r["pnl"])
            print("    %-22s %6d %8d %12s %7.0f%%" % (
                lab, r["taken"], r["missed"], format(r["pnl"], "+,.0f"), r["utilisation"]))
        print()
    print("  === total vs fixed 3/day x Rs50k (+359,542) ===")
    for lab, v in tot.items():
        s = sum(v)
        print("    %-22s %12s  %+.1f%%" % (lab, format(s, "+,.0f"), 100 * (s / 359542.0 - 1)))


if __name__ == "__main__":
    raise SystemExit(main())
