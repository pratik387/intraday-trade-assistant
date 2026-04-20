"""Stage 2: Univariate setup screening. Tighter version of Stage 1 with
sub-period consistency check.

Pass criteria (ALL required):
  - PF >= 1.2 on full Discovery
  - PF >= 1.0 in BOTH Discovery halves
  - Sharpe >= 0.7 on full Discovery
  - Max DD < 30% of total profit

Per spec Section 3.3.
"""
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from tools.edge_discovery.metrics import summary_stats, profit_factor
from tools.edge_discovery.periods import DiscoveryConfig, get_discovery_subperiods
from tools.edge_discovery.report_writer import write_stage_report, write_json_artifact


MIN_PF_FULL = 1.2
MIN_PF_SUBPERIOD = 1.0
MIN_SHARPE = 0.7
MAX_DD_PCT = 30.0


def run_stage2(
    trades: pd.DataFrame,
    cfg: DiscoveryConfig,
    survivors_input: List[str],
    report_path: Path,
    survivors_json: Path,
) -> List[Dict[str, Any]]:
    """Run Stage 2 on Stage 1 survivors only.

    Args:
        trades: DataFrame (must have session_date_dt col — actual date objects)
        cfg: DiscoveryConfig for sub-period split
        survivors_input: setup names from Stage 1
        report_path: markdown output path
        survivors_json: JSON output path

    Returns list of per-setup result dicts.

    Sub-period consistency: if a sub-period has zero trades, its PF is 0.0,
    which fails the >= 1.0 sub-period gate. This is the intended behavior —
    a setup with no evidence in one half of Discovery cannot be trusted. The
    h1_n / h2_n fields in the output make empty sub-periods explicitly visible
    in the audit trail.
    """
    (h1_start, h1_end), (h2_start, h2_end) = get_discovery_subperiods(cfg)

    def _in_range(s: pd.Series, start, end) -> pd.Series:
        return (s >= start) & (s <= end)

    rows: List[Dict[str, Any]] = []
    for setup in survivors_input:
        grp = trades[trades["setup_type"] == setup]
        if grp.empty:
            continue
        all_pnl = grp["total_trade_pnl"]
        h1_pnl = grp[_in_range(grp["session_date_dt"], h1_start, h1_end)]["total_trade_pnl"]
        h2_pnl = grp[_in_range(grp["session_date_dt"], h2_start, h2_end)]["total_trade_pnl"]

        full = summary_stats(all_pnl)
        pf_h1 = profit_factor(h1_pnl)
        pf_h2 = profit_factor(h2_pnl)

        passed = (
            full["pf"] >= MIN_PF_FULL
            and pf_h1 >= MIN_PF_SUBPERIOD
            and pf_h2 >= MIN_PF_SUBPERIOD
            and full["sharpe"] >= MIN_SHARPE
            and full["max_dd_pct"] < MAX_DD_PCT
        )
        rows.append({
            "setup": setup,
            "n": full["n"],
            "h1_n": len(h1_pnl),
            "h2_n": len(h2_pnl),
            "pf_full": round(full["pf"], 3) if full["pf"] != float("inf") else 999.0,
            "pf_h1": round(pf_h1, 3) if pf_h1 != float("inf") else 999.0,
            "pf_h2": round(pf_h2, 3) if pf_h2 != float("inf") else 999.0,
            "sharpe": round(full["sharpe"], 3) if full["sharpe"] != float("inf") else 999.0,
            "max_dd_pct": round(full["max_dd_pct"], 2) if full["max_dd_pct"] != float("inf") else 999.0,
            "wr_pct": round(full["wr_pct"], 2),
            "passed": bool(passed),
        })

    rows.sort(key=lambda r: (-int(r["passed"]), -r["pf_full"]))
    write_stage_report(
        path=report_path,
        stage_name="Stage 2 — Univariate Setup Screening",
        criteria=(
            f"PF_full >= {MIN_PF_FULL} AND PF_h1 >= {MIN_PF_SUBPERIOD} AND "
            f"PF_h2 >= {MIN_PF_SUBPERIOD} AND Sharpe >= {MIN_SHARPE} AND "
            f"max_DD < {MAX_DD_PCT}%"
        ),
        summary_rows=rows,
    )
    survivors = [r["setup"] for r in rows if r["passed"]]
    write_json_artifact(survivors_json, {"stage": "2", "survivors": survivors, "details": rows})
    return rows
