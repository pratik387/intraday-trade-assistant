"""Stage 1: Universe pruning. Drops setups below minimum trade count or
below profit-factor floor. These are noise — not worth further evaluation.

Pass criteria (BOTH required):
  - N >= 500 trades in Discovery period
  - PF >= 0.8

Per spec Section 3.3.
"""
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from tools.edge_discovery.metrics import summary_stats
from tools.edge_discovery.report_writer import write_stage_report, write_json_artifact


MIN_N = 500
MIN_PF = 0.8


def run_stage1(
    trades: pd.DataFrame,
    report_path: Path,
    survivors_json: Path,
    min_n: int = MIN_N,
    min_pf: float = MIN_PF,
) -> List[Dict[str, Any]]:
    """Run Stage 1 universe pruning.

    Args:
        trades: DataFrame with columns setup_type, total_trade_pnl
        report_path: where to write the markdown report
        survivors_json: where to write the JSON list of surviving setups

    Returns:
        List of per-setup result dicts. Keys: setup, n, pf, avg_pnl, total_pnl, passed.
    """
    rows: List[Dict[str, Any]] = []
    for setup, grp in trades.groupby("setup_type"):
        stats = summary_stats(grp["total_trade_pnl"])
        passed = (stats["n"] >= min_n) and (stats["pf"] >= min_pf)
        rows.append({
            "setup": str(setup),
            "n": stats["n"],
            "pf": round(stats["pf"], 3) if stats["pf"] != float("inf") else 999.0,
            "avg_pnl": round(stats["avg_pnl"], 2),
            "total_pnl": round(stats["total_pnl"], 2),
            "passed": bool(passed),
        })
    rows.sort(key=lambda r: (-int(r["passed"]), -r["n"]))

    write_stage_report(
        path=report_path,
        stage_name="Stage 1 — Universe Pruning",
        criteria=f"N >= {min_n} AND PF >= {min_pf}",
        summary_rows=rows,
    )
    survivors = [r["setup"] for r in rows if r["passed"]]
    write_json_artifact(survivors_json, {"stage": "1", "survivors": survivors, "details": rows})
    return rows
