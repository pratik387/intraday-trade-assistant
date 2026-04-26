"""Gauntlet-lite filter sweep for gap_fade_short setup (sub7-Phase2).

Loads gap_fade_short.parquet, adds derived features, sweeps single-dim and
2-dim filter combinations, and identifies the best combined filter.

Outputs:
  reports/sub7_validation/gauntlet/gap_fade_sweep.json
  reports/sub7_validation/gauntlet/gap_fade_sweep.md
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "reports/sub7_validation/gap_fade_short.parquet"
OUT_DIR = ROOT / "reports/sub7_validation/gauntlet"

# Phase-2 thresholds
MIN_PF = 1.25
MIN_SHARPE = 0.60
MIN_N_PHASE2 = 1000  # for combined-filter search
MIN_N_2DIM = 200     # for 2-dim cross filter floor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_trades": 0, "net_pnl": 0.0, "pf": 0.0, "sharpe": 0.0, "wr": 0.0}
    n = df["net_pnl"]
    wins = n[n > 0].sum()
    losses = n[n < 0].abs().sum()
    pf = float(wins / losses) if losses > 0 else 999.0
    daily = df.groupby("session_date")["net_pnl"].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    return {
        "n_trades": int(len(df)),
        "net_pnl": round(float(n.sum()), 2),
        "pf": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "wr": round(float((n > 0).mean()), 3),
    }


# ---------------------------------------------------------------------------
# Load + feature engineering
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date.astype(str)
    df["decision_ts"] = pd.to_datetime(df["decision_ts"])
    df["month"] = df["decision_ts"].dt.strftime("%Y-%m")
    df["dow"] = df["decision_ts"].dt.dayofweek   # Mon=0..Fri=4
    df["hour"] = df["decision_ts"].dt.hour
    df["minute"] = df["decision_ts"].dt.minute   # intraday bucket (9:20/25/30)
    notional = df["qty"] * df["entry_price"]
    df["pnl_pct"] = np.where(notional > 0, df["net_pnl"] / notional * 100, np.nan)
    print(f"Loaded {len(df)} trades | columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Single-dim sweep
# ---------------------------------------------------------------------------

def single_dim_sweep(df: pd.DataFrame, col: str) -> list[dict]:
    rows = []
    for k, grp in df.groupby(col):
        m = compute_metrics(grp)
        m["filter"] = f"{col}={k}"
        m["dim"] = col
        m["val"] = k
        rows.append(m)
    return rows


# ---------------------------------------------------------------------------
# 2-dim cross sweep
# ---------------------------------------------------------------------------

def two_dim_sweep(df: pd.DataFrame, col_a: str, col_b: str, min_n: int = MIN_N_2DIM) -> list[dict]:
    rows = []
    for (a, b), grp in df.groupby([col_a, col_b]):
        if len(grp) < min_n:
            continue
        m = compute_metrics(grp)
        m["filter"] = f"{col_a}={a} & {col_b}={b}"
        m["dim_a"] = col_a
        m["val_a"] = a
        m["dim_b"] = col_b
        m["val_b"] = b
        rows.append(m)
    return sorted(rows, key=lambda x: x["pf"], reverse=True)


# ---------------------------------------------------------------------------
# Best combined filter (maximise net_pnl, n >= 1000)
# ---------------------------------------------------------------------------

def find_best_combined(df: pd.DataFrame) -> dict:
    """
    Enumerate subsets formed by restricting each of the 4 filter dimensions
    (cap_segment, regime, dow, hour) individually and in pairs, keeping only
    those with n_trades >= MIN_N_PHASE2, and return the one with highest net_pnl.
    """
    dims = ["cap_segment", "regime", "dow", "minute"]
    candidates: list[dict] = []

    # 1-dim subsets
    for col in dims:
        for k, grp in df.groupby(col):
            if len(grp) < MIN_N_PHASE2:
                continue
            m = compute_metrics(grp)
            m["filter_spec"] = {col: k}
            candidates.append(m)

    # 2-dim subsets
    for col_a, col_b in combinations(dims, 2):
        for (a, b), grp in df.groupby([col_a, col_b]):
            if len(grp) < MIN_N_PHASE2:
                continue
            m = compute_metrics(grp)
            m["filter_spec"] = {col_a: a, col_b: b}
            candidates.append(m)

    # 3-dim subsets
    for trio in combinations(dims, 3):
        for keys, grp in df.groupby(list(trio)):
            if len(grp) < MIN_N_PHASE2:
                continue
            m = compute_metrics(grp)
            m["filter_spec"] = dict(zip(trio, keys))
            candidates.append(m)

    if not candidates:
        return {"message": "No subset with n_trades >= 1000 found"}

    best = max(candidates, key=lambda x: x["net_pnl"])
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # Single-dim sweeps
    all_single: list[dict] = []
    for col in ["cap_segment", "regime", "dow", "minute"]:
        all_single.extend(single_dim_sweep(df, col))
    all_single.sort(key=lambda x: x["pf"], reverse=True)
    top10_single = all_single[:10]

    # 2-dim crosses
    cross_pairs = [
        ("cap_segment", "regime"),
        ("cap_segment", "dow"),
        ("cap_segment", "minute"),
        ("regime", "minute"),
        ("regime", "dow"),
        ("dow", "minute"),
    ]
    all_2dim: list[dict] = []
    for a, b in cross_pairs:
        all_2dim.extend(two_dim_sweep(df, a, b))
    all_2dim.sort(key=lambda x: x["pf"], reverse=True)
    top10_2dim = all_2dim[:10]

    # Best combined filter
    best_combined = find_best_combined(df)

    # Phase-2 verdict
    def passes_phase2(m: dict) -> bool:
        return (m.get("pf", 0) >= MIN_PF
                and m.get("sharpe", 0) >= MIN_SHARPE
                and m.get("n_trades", 0) >= MIN_N_PHASE2)

    phase2_single = [m for m in all_single if passes_phase2(m)]
    phase2_2dim = [m for m in all_2dim if passes_phase2(m)]

    verdict = {
        "any_single_passes": bool(phase2_single),
        "any_2dim_passes": bool(phase2_2dim),
        "best_combined_passes": passes_phase2(best_combined),
        "passing_single": phase2_single,
        "passing_2dim": phase2_2dim,
    }

    # Full JSON output
    result = {
        "baseline": compute_metrics(df),
        "top10_single": top10_single,
        "top10_2dim": top10_2dim,
        "best_combined": best_combined,
        "phase2_verdict": verdict,
    }

    out_json = OUT_DIR / "gap_fade_sweep.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {out_json}")

    # Markdown report
    lines: list[str] = []
    lines.append("# gap_fade_short — Gauntlet-Lite Filter Sweep")
    lines.append("")
    b = result["baseline"]
    lines.append(f"**Baseline**: n={b['n_trades']:,} | PF={b['pf']:.3f} | Sharpe={b['sharpe']:.3f} | "
                 f"WR={b['wr']:.1%} | Net PnL=Rs{b['net_pnl']:,.0f}")
    lines.append("")
    lines.append("## Top 10 Single-Dim Slices (by PF)")
    lines.append("")
    lines.append("| Filter | n | PF | Sharpe | WR | Net PnL |")
    lines.append("|--------|---|----|--------|----|---------|")
    for m in top10_single:
        lines.append(f"| {m['filter']} | {m['n_trades']:,} | {m['pf']:.3f} | "
                     f"{m['sharpe']:.3f} | {m['wr']:.1%} | Rs{m['net_pnl']:,.0f} |")

    lines.append("")
    lines.append(f"## Top 10 2-Dim Crosses (n >= {MIN_N_2DIM}, by PF)")
    lines.append("")
    if top10_2dim:
        lines.append("| Filter | n | PF | Sharpe | WR | Net PnL |")
        lines.append("|--------|---|----|--------|----|---------|")
        for m in top10_2dim:
            lines.append(f"| {m['filter']} | {m['n_trades']:,} | {m['pf']:.3f} | "
                         f"{m['sharpe']:.3f} | {m['wr']:.1%} | Rs{m['net_pnl']:,.0f} |")
    else:
        lines.append("_No 2-dim cross met the n >= 200 floor._")

    lines.append("")
    lines.append(f"## Best Combined Filter (n >= {MIN_N_PHASE2})")
    lines.append("")
    if "filter_spec" in best_combined:
        lines.append(f"**Filter**: `{best_combined['filter_spec']}`")
        lines.append(f"**Metrics**: n={best_combined['n_trades']:,} | PF={best_combined['pf']:.3f} | "
                     f"Sharpe={best_combined['sharpe']:.3f} | WR={best_combined['wr']:.1%} | "
                     f"Net PnL=Rs{best_combined['net_pnl']:,.0f}")
    else:
        lines.append(best_combined.get("message", "N/A"))

    lines.append("")
    lines.append("## Phase-2 Verdict (PF >= 1.25 AND Sharpe >= 0.60 AND n >= 1000)")
    lines.append("")
    any_pass = verdict["any_single_passes"] or verdict["any_2dim_passes"] or verdict["best_combined_passes"]
    lines.append(f"**PASS** ✓" if any_pass else "**FAIL** — no filter meets Phase-2 thresholds")
    if verdict["passing_single"]:
        lines.append("")
        lines.append("### Passing single-dim filters")
        for m in verdict["passing_single"]:
            lines.append(f"- {m['filter']}: n={m['n_trades']:,}, PF={m['pf']:.3f}, Sharpe={m['sharpe']:.3f}")
    if verdict["passing_2dim"]:
        lines.append("")
        lines.append("### Passing 2-dim crosses")
        for m in verdict["passing_2dim"]:
            lines.append(f"- {m['filter']}: n={m['n_trades']:,}, PF={m['pf']:.3f}, Sharpe={m['sharpe']:.3f}")

    out_md = OUT_DIR / "gap_fade_sweep.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")

    # Console summary
    print("\n=== TOP 5 SINGLE-DIM SLICES ===")
    for m in top10_single[:5]:
        print(f"  {m['filter']:40s}  n={m['n_trades']:5d}  PF={m['pf']:.3f}  Sharpe={m['sharpe']:.3f}")

    print("\n=== TOP 5 2-DIM CROSSES ===")
    if top10_2dim:
        for m in top10_2dim[:5]:
            print(f"  {m['filter']:55s}  n={m['n_trades']:5d}  PF={m['pf']:.3f}  Sharpe={m['sharpe']:.3f}")
    else:
        print("  (none met n >= 200 floor)")

    print("\n=== BEST COMBINED FILTER ===")
    if "filter_spec" in best_combined:
        print(f"  Spec   : {best_combined['filter_spec']}")
        print(f"  n      : {best_combined['n_trades']:,}")
        print(f"  PF     : {best_combined['pf']:.3f}")
        print(f"  Sharpe : {best_combined['sharpe']:.3f}")
        print(f"  WR     : {best_combined['wr']:.1%}")
        print(f"  Net PnL: Rs{best_combined['net_pnl']:,.0f}")
    else:
        print(f"  {best_combined}")

    print(f"\n=== PHASE-2 VERDICT ===")
    print(f"  Any filter crosses PF=1.25 AND Sharpe=0.60 AND n>=1000: {'YES' if any_pass else 'NO'}")


if __name__ == "__main__":
    main()
