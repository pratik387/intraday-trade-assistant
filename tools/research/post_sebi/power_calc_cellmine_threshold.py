"""Power calculation: derive a statistically defensible n threshold for
cell-mining in this project.

Setup:
  H0: cell's true PF = 1.10 (the project's "no meaningful edge" boundary)
  H1: cell's true PF = 1.30 (the project's "interesting edge" target)
  alpha = 0.05  (false positive — accepting a 1.10 cell as shippable)
  beta  = 0.20  (false negative — missing a 1.30 cell — power = 80%)

Method:
  Bootstrap from actual cell-level trade distributions in this codebase.
  For each candidate n, resample n trades with replacement N=10000 times
  from a reference cell whose true PF we treat as the H1 value (1.30).
  Compute the empirical 5th percentile of bootstrap PF distribution.
  The smallest n at which 5th pct >= 1.10 is the defensible threshold.

This avoids parametric assumptions (Gaussian per-trade PnL) and uses the
real distribution shape (skewed by stops/partials/runners).

Reference cells used (from our existing sanity outputs):
  - delivery_pct Cell B (mid_cap × 09:30-10:30): n=295 PF=1.245 in OOS Jan-Sep 2025
  - gap_fade locked combo: PF=1.71 OOS — too far above 1.30, less useful here
  - C4b 'bar1_strength=0.1-0.25': n=138 PF=1.33 post-rule

We'll bootstrap from delivery_pct OOS (closest to PF 1.30 with large n).

Usage:
    python -m tools.research.post_sebi.power_calc_cellmine_threshold
"""
from __future__ import annotations

import json, os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RNG = np.random.default_rng(20260514)
N_BOOTSTRAPS = 10000

# Targets:
PF_NULL = 1.10        # H0: not-meaningful
PF_TARGET = 1.30      # H1: meaningful edge
ALPHA = 0.05          # type-I floor
POWER_TARGET = 0.80   # 1 - beta

# Candidate n values to test
N_CANDIDATES = [50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 750, 1000]


def pf_of(arr: np.ndarray) -> float:
    g = arr[arr > 0].sum()
    l = -arr[arr <= 0].sum()
    return g / l if l > 0 else float("inf")


def build_synthetic_trades(pf_target: float, mean_R: float, std_R: float,
                            wr: float, n: int = 5000) -> np.ndarray:
    """Build a synthetic per-trade R-multiple distribution with:
       - given win-rate wr
       - given mean win/loss
       - target PF achieved through win/loss magnitude balance
    Returns array of n trade-PnL values in rupees-per-trade units.
    """
    n_wins = int(n * wr)
    n_losses = n - n_wins
    # Asymmetric draws: wins around +mean_win, losses around -mean_loss
    # PF = (wr × mean_win) / ((1-wr) × mean_loss) => mean_win/mean_loss = PF × (1-wr)/wr
    ratio = pf_target * (1 - wr) / wr
    mean_loss = 1.0
    mean_win = ratio * mean_loss
    wins = RNG.normal(mean_win * mean_R, std_R, n_wins)
    losses = RNG.normal(-mean_loss * mean_R, std_R, n_losses)
    return np.concatenate([wins, losses])


def load_real_cell_trades() -> np.ndarray:
    """Load a real cell's per-trade net_pnl distribution to use as reference.

    Uses C4b SHORT 'bar1_strength=0.1-0.25' cell (n=138, PF=1.33) since
    that's our actual observed cell at the PF threshold of interest.
    """
    trades_path = _REPO / "reports" / "research" / "post_sebi" / "gap_down_intraday" / "gap_down_intraday_trades.parquet"
    if not trades_path.exists():
        return np.array([])
    df = pd.read_parquet(trades_path)
    df = df[df["setup"] == "c4b_gap_down_continuation_short"]
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    df["regime"] = df["session_date"].apply(
        lambda d: "pre_rule" if d < pd.Timestamp("2025-02-01").date() else "post_rule"
    )
    df = df[df["regime"] == "post_rule"]
    # Apply same bar1_strength bucket
    df["bar1_strength_pct"] = (df["close_920"] - df["open_915"]).abs() / df["open_915"] * 100.0
    df = df[(df["bar1_strength_pct"] >= 0.1) & (df["bar1_strength_pct"] < 0.25)]
    return df["net_pnl"].values


def bootstrap_pf_distribution(source: np.ndarray, n_sample: int,
                              n_boots: int = N_BOOTSTRAPS) -> np.ndarray:
    """Sample n_sample trades with replacement from source, compute PF, repeat n_boots times."""
    pfs = np.zeros(n_boots)
    src = np.asarray(source, dtype=float)
    for i in range(n_boots):
        sample = RNG.choice(src, size=n_sample, replace=True)
        pfs[i] = pf_of(sample)
    return pfs


def main():
    print("=" * 78)
    print("POWER CALCULATION FOR CELL-MINING n THRESHOLD")
    print("=" * 78)
    print(f"H0: true PF = {PF_NULL}  (cell has no meaningful edge)")
    print(f"H1: true PF = {PF_TARGET}  (cell has interesting edge)")
    print(f"alpha = {ALPHA}, target power = {POWER_TARGET}")
    print(f"bootstrap N = {N_BOOTSTRAPS}")
    print()

    # ---- Approach 1: synthetic distribution with PF=1.30, WR=60% ----
    print("--- Approach 1: synthetic distribution (PF=1.30, WR=60%) ---")
    synth = build_synthetic_trades(pf_target=PF_TARGET, mean_R=1000, std_R=200,
                                    wr=0.60, n=10000)
    achieved_pf = pf_of(synth)
    print(f"  synthetic pool: n={len(synth)}, achieved PF={achieved_pf:.3f}")
    print()
    print(f"  {'n':>5}  {'p5':>7}  {'p50':>7}  {'p95':>7}  {'P(>1.10)':>9}  {'P(>1.30)':>9}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}")
    synth_results = []
    for n in N_CANDIDATES:
        pfs = bootstrap_pf_distribution(synth, n)
        p5 = float(np.percentile(pfs, 5))
        p50 = float(np.median(pfs))
        p95 = float(np.percentile(pfs, 95))
        p_above_null = float((pfs > PF_NULL).mean())
        p_above_target = float((pfs > PF_TARGET).mean())
        synth_results.append({
            "n": n, "p5": p5, "p50": p50, "p95": p95,
            "p_above_null": p_above_null, "p_above_target": p_above_target,
        })
        print(f"  {n:>5}  {p5:>7.3f}  {p50:>7.3f}  {p95:>7.3f}  "
              f"{p_above_null:>9.3f}  {p_above_target:>9.3f}")

    # Find minimum n where P(boot_PF > PF_NULL) >= POWER_TARGET
    synth_df = pd.DataFrame(synth_results)
    qualifying = synth_df[synth_df["p_above_null"] >= POWER_TARGET]
    if not qualifying.empty:
        n_synth = int(qualifying.iloc[0]["n"])
        print(f"\n  >>> Minimum n where 80% of bootstraps exceed PF=1.10: n >= {n_synth}")
    else:
        print(f"\n  No tested n achieved power {POWER_TARGET} on synthetic.")
        n_synth = None
    print()

    # ---- Approach 2: real cell distribution ----
    print("--- Approach 2: real cell (C4b bar1_strength=0.1-0.25 post_rule) ---")
    real = load_real_cell_trades()
    if len(real) < 20:
        print(f"  insufficient real data (n={len(real)})")
    else:
        real_pf = pf_of(real)
        print(f"  real pool: n={len(real)}, observed PF={real_pf:.3f}")
        print()
        print(f"  {'n':>5}  {'p5':>7}  {'p50':>7}  {'p95':>7}  {'P(>1.10)':>9}  {'P(>1.30)':>9}")
        print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}")
        real_results = []
        for n in N_CANDIDATES:
            pfs = bootstrap_pf_distribution(real, n)
            p5 = float(np.percentile(pfs, 5))
            p50 = float(np.median(pfs))
            p95 = float(np.percentile(pfs, 95))
            p_above_null = float((pfs > PF_NULL).mean())
            p_above_target = float((pfs > PF_TARGET).mean())
            real_results.append({
                "n": n, "p5": p5, "p50": p50, "p95": p95,
                "p_above_null": p_above_null, "p_above_target": p_above_target,
            })
            print(f"  {n:>5}  {p5:>7.3f}  {p50:>7.3f}  {p95:>7.3f}  "
                  f"{p_above_null:>9.3f}  {p_above_target:>9.3f}")
        real_df = pd.DataFrame(real_results)
        qualifying = real_df[real_df["p_above_null"] >= POWER_TARGET]
        if not qualifying.empty:
            n_real = int(qualifying.iloc[0]["n"])
            print(f"\n  >>> Minimum n where 80% of bootstraps exceed PF=1.10: n >= {n_real}")
        else:
            print(f"\n  No tested n achieved power {POWER_TARGET} on real data.")
            n_real = None
    print()

    # ---- Verdict ----
    print("=" * 78)
    print("THRESHOLD RECOMMENDATION")
    print("=" * 78)
    if n_synth is not None:
        print(f"Synthetic-based minimum n: {n_synth}  (assumes PF=1.30, WR=60%, symmetric)")
    if 'n_real' in dir() and n_real is not None:
        print(f"Real-cell-based minimum n: {n_real}  (uses actual C4b cell distribution)")
    print()
    print("The defensible n threshold for declaring a cell 'ship-eligible'")
    print(f"(P(true PF >= {PF_NULL}) >= {POWER_TARGET} given observed PF ~ {PF_TARGET}):")
    if n_synth is not None and 'n_real' in dir() and n_real is not None:
        recommended = max(n_synth, n_real)
        print(f"  RECOMMENDED: n >= {recommended}")
        print()
        print(f"Compare to existing _cell_mine_tier_a thresholds:")
        print(f"  N_MIN_SHIP (current) = 200")
        print(f"  Statistically defensible = {recommended}")
        if recommended < 200:
            print(f"  -> current threshold is CONSERVATIVE (safer than needed)")
        elif recommended > 200:
            print(f"  -> current threshold is too LENIENT for this confidence level")
        else:
            print(f"  -> current threshold is well-calibrated")


if __name__ == "__main__":
    main()
