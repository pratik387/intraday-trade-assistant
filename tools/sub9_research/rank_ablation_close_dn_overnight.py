"""Which within-day ranking key should close_dn use to spend its limited slots?

close_dn fires more candidates per day than the capital pool can fund (~7 fires
vs 3 slots). Production rations by `signed_vol_ratio` ascending — "deepest
capitulation first" — a choice the config records as "verified vs 30-seed random
on Disc/OOS/HO 2026-07-02".

Live paper contradicts that. Over 42 sessions, holding execution constant by
scoring every fire at the PAPER ledger's idealized fill, the names production
took returned +0.127% gross while the ones it skipped returned +0.802%. The
within-day paired gap is -0.778pp (t=-3.10), and it is entirely a CNC
phenomenon: CNC -1.413pp (t=-3.80 within-day) versus MTF -0.025pp (t=-0.31),
which is a clean null control. Among CNC fires the discriminator was not svr
(-0.814 taken vs -0.724 skipped, no separation) but closing_30m_volume_z
(+9.64 taken vs +17.49 skipped) — the cell's own edge variable.

So the hypothesis is that svr sorts low-vol_z names to the top and production
systematically buys the weak half of its own signal.

This script tests that on the locked research ledgers rather than on 42 days of
paper. For each split it ranks each day's cell-5 fires by a candidate key, takes
the top K, and reports PF and net P&L. `random` (many seeds) is the chance
baseline the original claim was measured against; `all` is the no-cap ceiling.

Fees are already inside `exit_0915_open_net_pnl_inr`, computed by
sanity_close_dn_overnight_long.calc_fee_cnc, which carries the post-Jun-2026
rate card (Rs 0 delivery brokerage, STT both sides, txn 0.00307%). Note the
cell_lock.json `fee_model` block still records the PRE-audit numbers (Rs 20/side
brokerage, sell-only STT) — that block is stale metadata and is NOT what
produced these ledgers.

Usage:
    python tools/sub9_research/rank_ablation_close_dn_overnight.py [--k 3] [--seeds 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

LEDGER_DIR = _REPO_ROOT / "reports" / "sub9_sanity"
SPLITS = ("discovery", "oos", "holdout", "recent")
PNL_COL = "exit_0915_open_net_pnl_inr"      # production exit: sell at T+1 open

# Locked cell #5 (tools/sub9_research/close_dn_overnight_long_cell_lock.json)
CELL = {"closing_30m_volume_z_bin": "extreme", "prior_day_return_bin": "up_gt_3pct"}


def profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    loss = -pnl[pnl < 0].sum()
    if loss <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / loss)


def load_cell(split: str) -> pd.DataFrame:
    path = LEDGER_DIR / f"_close_dn_overnight_long_multi_exit_{split}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col, val in CELL.items():
        df = df[df[col] == val]
    df = df.dropna(subset=[PNL_COL, "signal_date"])
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
    return df.reset_index(drop=True)


def take_top_k(df: pd.DataFrame, key: str, ascending: bool, k: int) -> pd.DataFrame:
    """Rank each day's fires by `key` and keep the first k."""
    d = df.sort_values(["signal_date", key], ascending=[True, ascending])
    d = d.assign(_rank=d.groupby("signal_date").cumcount() + 1)
    return d[d["_rank"] <= k]


def random_baseline(df: pd.DataFrame, k: int, seeds: int) -> tuple[float, float, float]:
    """Mean PF / mean net / fraction of seeds beating zero, over `seeds` draws."""
    pfs, nets = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        d = df.assign(_r=rng.random(len(df)))
        d = d.sort_values(["signal_date", "_r"])
        d = d.assign(_rank=d.groupby("signal_date").cumcount() + 1)
        sel = d[d["_rank"] <= k][PNL_COL]
        pfs.append(profit_factor(sel))
        nets.append(sel.sum())
    pfs = [p for p in pfs if np.isfinite(p)]
    return float(np.mean(pfs)), float(np.mean(nets)), float(np.std(nets))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="slots per day (production = 3)")
    ap.add_argument("--seeds", type=int, default=200, help="random-baseline draws")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep slots/day to size the BREADTH effect vs the RANKING effect")
    args = ap.parse_args()

    if args.sweep:
        ks = [1, 2, 3, 4, 6, 8, 12]
        print("Breadth sweep: net Rs by slots/day, cell-5, exit_0915_open")
        print()
        for split in SPLITS:
            df = load_cell(split)
            if df.empty:
                continue
            days = df["signal_date"].nunique()
            ceiling = df[PNL_COL].sum()
            print("=== %s | %d fires / %d days (%.1f per day) | no-cap ceiling Rs%s ===" % (
                split.upper(), len(df), days, len(df) / days, format(ceiling, "+,.0f")))
            print("  %-4s %7s %14s %14s %14s %9s" % (
                "k", "n", "svr asc (prod)", "vol_z desc", "random mean", "%ceiling"))
            for k in ks:
                svr = take_top_k(df, "signed_vol_ratio", True, k)[PNL_COL]
                vz = take_top_k(df, "closing_30m_volume_z", False, k)[PNL_COL]
                _, rnet, _ = random_baseline(df, k, min(args.seeds, 40))
                print("  %-4d %7d %14s %14s %14s %8.0f%%" % (
                    k, len(svr), format(svr.sum(), "+,.0f"), format(vz.sum(), "+,.0f"),
                    format(rnet, "+,.0f"), 100 * svr.sum() / ceiling if ceiling else 0))
            print()
        return 0

    print(f"close_dn within-day ranking ablation | k={args.k} slots/day | "
          f"cell = volume_z extreme x prior_ret up_gt_3pct")
    print(f"pnl column: {PNL_COL} (fees already applied, post-Jun-2026 rate card)\n")

    variants = [
        ("svr asc (PRODUCTION)", "signed_vol_ratio", True),
        ("svr desc (inverted)", "signed_vol_ratio", False),
        ("vol_z desc (CANDIDATE)", "closing_30m_volume_z", False),
        ("vol_z asc (inverted)", "closing_30m_volume_z", True),
        ("prior_ret desc", "prior_day_return_pct", False),
    ]

    for split in SPLITS:
        df = load_cell(split)
        if df.empty:
            print(f"=== {split.upper()}: no ledger ===\n")
            continue
        days = df["signal_date"].nunique()
        print(f"=== {split.upper()} | {len(df)} fires over {days} days "
              f"({len(df)/days:.1f}/day) ===")

        allp = df[PNL_COL]
        print(f"  {'take ALL (no cap, ceiling)':<26} n={len(allp):5d} "
              f"PF={profit_factor(allp):6.3f}  net=Rs{allp.sum():+12,.0f}")

        mpf, mnet, snet = random_baseline(df, args.k, args.seeds)
        print(f"  {'random ' + str(args.seeds) + ' seeds':<26} "
              f"n={'':5} PF={mpf:6.3f}  net=Rs{mnet:+12,.0f}  (sd {snet:,.0f})")

        for label, key, asc in variants:
            if key not in df.columns:
                continue
            sel = take_top_k(df, key, asc, args.k)[PNL_COL]
            z = (sel.sum() - mnet) / snet if snet else float("nan")
            print(f"  {label:<26} n={len(sel):5d} PF={profit_factor(sel):6.3f}  "
                  f"net=Rs{sel.sum():+12,.0f}  ({z:+.2f} sd vs random)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
