"""Validation gate — per-rule OOS check.

Applies 90 Discovery-approved rules against a Validation period wide-open trade
stream. Per design spec §3.4 (specs/2026-04-15-edge-discovery-design.md):

Pass criteria per rule (ALL required):
  - pf_val >= 1.0 on Validation period
  - n_val >= 50 in Validation period
  - |wr_val - wr_discovery| <= 10.0 percentage points

ONE-SHOT discipline — criteria locked, no tuning. Failed rules die; they do
not proceed to holdout gate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from tools.edge_discovery.metrics import profit_factor, win_rate_pct
from tools.edge_discovery.report_writer import write_json_artifact


# Pass-criteria thresholds per design spec §3.4
MIN_PF = 1.0
MIN_N = 50
MAX_WR_DELTA_PP = 10.0


def validate_rule(
    trades: pd.DataFrame,
    rule: Dict[str, Any],
    discovery_stats: Dict[str, float],
) -> Dict[str, Any]:
    """Apply one rule to `trades`, return per-rule validation result.

    Args:
        trades: Validation-period wide-open trades DataFrame.
        rule: {"setup": "...", "conditions": [(key, val), ...]}
        discovery_stats: {"wr_pct": float, "pf": float, "n": int}

    Returns:
        Dict with setup, rule_id, n_val, pf_val, wr_val, wr_delta_pp,
        passed (bool), fail_reasons (list).
    """
    setup = rule["setup"]
    conditions = rule["conditions"]
    rule_id = f"{setup}__" + "+".join(k for k, _ in conditions) + "=" + "+".join(v for _, v in conditions)

    # Filter trades by setup + conditions
    sub = trades[trades["setup_type"] == setup]
    for key, val in conditions:
        sub = sub[sub[key] == val]

    pnl = sub["total_trade_pnl"] if len(sub) else pd.Series([], dtype=float)
    n_val = int(len(sub))
    pf_val = profit_factor(pnl)
    wr_val = win_rate_pct(pnl)
    wr_discovery = float(discovery_stats.get("wr_pct", 0.0))
    wr_delta_pp = abs(wr_val - wr_discovery)

    fail_reasons: List[str] = []
    if n_val < MIN_N:
        fail_reasons.append(f"n_val={n_val}<{MIN_N}")
    if pf_val < MIN_PF:
        fail_reasons.append(f"pf_val={pf_val:.3f}<{MIN_PF}")
    if wr_delta_pp > MAX_WR_DELTA_PP:
        fail_reasons.append(f"wr_delta={wr_delta_pp:.1f}pp>{MAX_WR_DELTA_PP}")
    passed = len(fail_reasons) == 0

    return {
        "setup": setup,
        "rule_id": rule_id,
        "n_val": n_val,
        "pf_val": round(pf_val, 3) if pf_val != float("inf") else 999.0,
        "wr_val": round(wr_val, 2),
        "wr_discovery": round(wr_discovery, 2),
        "wr_delta_pp": round(wr_delta_pp, 2),
        "pf_discovery": round(float(discovery_stats.get("pf", 0.0)), 3),
        "n_discovery": int(discovery_stats.get("n", 0)),
        "passed": bool(passed),
        "fail_reasons": fail_reasons,
    }


def run_validation_gate(
    trades: pd.DataFrame,
    approved_rules: List[Dict[str, Any]],
    discovery_stats_by_rule: Dict[str, Dict[str, float]],
    report_path: Path,
    survivors_json: Path,
) -> Dict[str, Any]:
    """Run full validation gate, emit markdown report + JSON survivors list."""
    per_rule: List[Dict[str, Any]] = []
    for r in approved_rules:
        rule_id = r["setup"] + "__" + "+".join(k for k, _ in r["conditions"]) + "=" + "+".join(v for _, v in r["conditions"])
        disc = discovery_stats_by_rule.get(rule_id, {"wr_pct": 0.0, "pf": 0.0, "n": 0})
        per_rule.append(validate_rule(trades, r, disc))

    # Sort: passed first (desc by pf_val), then failed (desc by pf_val)
    per_rule.sort(key=lambda r: (-int(r["passed"]), -r["pf_val"]))

    n_passed = sum(1 for r in per_rule if r["passed"])
    n_failed = len(per_rule) - n_passed

    # Markdown report
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Validation Gate Results",
        "",
        f"**Criteria (per spec §3.4, ALL required):** pf_val >= {MIN_PF} AND "
        f"n_val >= {MIN_N} AND |wr_delta| <= {MAX_WR_DELTA_PP}pp",
        "",
        f"**Rules evaluated:** {len(per_rule)}",
        f"**Passed:** {n_passed}",
        f"**Failed:** {n_failed}",
        "",
        "## Per-rule results",
        "",
        "| rule_id | status | n_val | pf_val | wr_val | wr_disc | wr_delta | fail_reasons |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in per_rule:
        status = "PASS" if r["passed"] else "FAIL"
        reasons = "; ".join(r["fail_reasons"]) if r["fail_reasons"] else "—"
        lines.append(
            f"| {r['rule_id']} | {status} | {r['n_val']} | {r['pf_val']} | "
            f"{r['wr_val']}% | {r['wr_discovery']}% | {r['wr_delta_pp']}pp | {reasons} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # JSON artifact
    survivors = [r for r in per_rule if r["passed"]]
    write_json_artifact(survivors_json, {
        "stage": "validation_gate",
        "criteria": {"min_pf": MIN_PF, "min_n": MIN_N, "max_wr_delta_pp": MAX_WR_DELTA_PP},
        "n_passed": n_passed,
        "n_failed": n_failed,
        "survivors": [{"setup": r["setup"], "rule_id": r["rule_id"]} for r in survivors],
        "per_rule": per_rule,
    })
    return {"n_passed": n_passed, "n_failed": n_failed, "per_rule": per_rule}
