"""Per-setup metrics + breakdowns report (sub7-T10).

Loads <setup>.parquet from build_per_setup_pnl output, splits by Discovery
period (2023-01-01 to 2024-12-31 by default), computes:
  - Aggregate net metrics (PF, Sharpe, total PnL, n_trades, WR)
  - Per-month breakdown (decay check)
  - Per-cap-segment breakdown
  - Per-regime breakdown
  - Per-day-of-week breakdown
Writes JSON + markdown report.

CLI:
    python tools/sub7_validation/per_setup_report.py \\
        --setup-parquet reports/sub7_validation/mis_unwind_short.parquet \\
        --output-dir reports/sub7_validation/mis_unwind_short/ \\
        --period-start 2023-01-01 --period-end 2024-12-31
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_trades": 0, "n_sessions": 0, "net_pnl": 0.0,
                "net_pf": 0.0, "net_sharpe": 0.0, "wr": 0.0,
                "trades_per_day": 0.0, "losing_days_pct": 0.0,
                "max_dd": 0.0}
    n = df["net_pnl"]
    wins = n[n > 0].sum()
    losses = n[n < 0].abs().sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    daily = df.groupby("session_date")["net_pnl"].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    cumret = daily.cumsum()
    max_dd = float((cumret - cumret.cummax()).min())
    return {
        "n_trades": int(len(df)),
        "n_sessions": int(daily.size),
        "net_pnl": float(n.sum()),
        "net_pf": round(pf, 3) if pf != float("inf") else 999.0,
        "net_sharpe": round(sharpe, 3),
        "wr": round(float((n > 0).mean()), 3),
        "trades_per_day": round(len(df) / daily.size, 2) if daily.size else 0.0,
        "losing_days_pct": round(100 * (daily < 0).sum() / daily.size, 1) if daily.size else 0.0,
        "max_dd": round(max_dd, 0),
    }


def breakdown_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for k, grp in df.groupby(col):
        m = compute_metrics(grp)
        m[col] = k
        rows.append(m)
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--period-start", default="2023-01-01")
    p.add_argument("--period-end", default="2024-12-31")
    args = p.parse_args()

    df = pd.read_parquet(args.setup_parquet)
    setup_name = Path(args.setup_parquet).stem
    df = df[(df["session_date"] >= args.period_start) & (df["session_date"] <= args.period_end)]
    print(f"Loaded {len(df)} trades for {setup_name} in {args.period_start} to {args.period_end}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = compute_metrics(df)
    df["month"] = df["session_date"].astype(str).str[:7]
    by_month = breakdown_by(df, "month")
    by_cap = breakdown_by(df, "cap_segment") if "cap_segment" in df.columns else pd.DataFrame()
    by_regime = breakdown_by(df, "regime") if "regime" in df.columns else pd.DataFrame()

    # Pass/fail per Phase 1 bar
    bar = {"net_pf_min": 1.10, "n_trades_min": 500, "net_sharpe_min": 0.0}
    passes = (
        agg["net_pf"] >= bar["net_pf_min"]
        and agg["n_trades"] >= bar["n_trades_min"]
        and agg["net_sharpe"] >= bar["net_sharpe_min"]
    )

    result = {
        "setup": setup_name,
        "period": {"start": args.period_start, "end": args.period_end},
        "aggregate": agg,
        "phase1_pass_criteria": bar,
        "phase1_passes": bool(passes),
    }
    (out_dir / "01-metrics.json").write_text(json.dumps(result, indent=2))

    by_month.to_csv(out_dir / "02-by-month.csv", index=False)
    if not by_cap.empty:
        by_cap.to_csv(out_dir / "03-by-cap-segment.csv", index=False)
    if not by_regime.empty:
        by_regime.to_csv(out_dir / "04-by-regime.csv", index=False)

    md = [f"# {setup_name} — Per-setup Report",
          f"\n**Period:** {args.period_start} to {args.period_end}",
          f"\n**Phase 1 verdict:** {'PASS' if passes else 'FAIL'}",
          "\n## Aggregate Metrics", "```json", json.dumps(agg, indent=2), "```",
          "\n## Per-month breakdown", by_month.to_markdown(index=False)]
    if not by_cap.empty:
        md.extend(["\n## Per-cap-segment", by_cap.to_markdown(index=False)])
    if not by_regime.empty:
        md.extend(["\n## Per-regime", by_regime.to_markdown(index=False)])
    (out_dir / "05-report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Pass: {passes}  PF={agg['net_pf']}  n={agg['n_trades']}  Sharpe={agg['net_sharpe']}")


if __name__ == "__main__":
    main()
