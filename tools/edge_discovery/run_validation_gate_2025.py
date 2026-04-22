"""Run validation_gate on 2025 wide-open backtest data.

Applies the 90 narrative-approved Discovery rules against Validation period
(2025-01-01 → 2025-09-30) of the 20260421-134338_full OCI backtest. Kills
rules failing per-rule criteria per design spec §3.4.

Output: docs/edge_discovery/2026-04-22-validation-gate/06-validation-results.md
        + stage6_validation_survivors.json
"""
import json
import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

from tools.edge_discovery.data_loader import load_run
from tools.edge_discovery.gates.validation_gate import run_validation_gate

ROOT = Path(__file__).parent.parent.parent
BACKTEST_DIR = ROOT / "20260421-134338_full"
DISCOVERY_SURVIVORS = ROOT / "analysis" / "edge_discovery_runs" / "2026-04-20" / "stage3_survivors.json"
FILL_SCRIPT = ROOT / "tools" / "edge_discovery" / "fill_narratives.py"
OUT_DIR = ROOT / "docs" / "edge_discovery" / "2026-04-22-validation-gate"

VALIDATION_START = date(2025, 1, 1)
VALIDATION_END = date(2025, 9, 30)


def load_approved_rules_and_discovery_stats():
    """Return (approved_rules_list, discovery_stats_by_rule_id_dict)."""
    spec = importlib.util.spec_from_file_location("fill_narratives", FILL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rejected = mod.REJECTED_RULES
    survivors = json.loads(DISCOVERY_SURVIVORS.read_text(encoding="utf-8"))

    rules = []
    stats_by_rule_id = {}
    for cell in survivors["details"]:
        if not cell.get("passed"):
            continue
        rule_id = f"{cell['setup']}__{cell['conditioner']}={cell['cell_value']}"
        if rule_id in rejected:
            continue
        rules.append({
            "setup": cell["setup"],
            "conditions": list(zip(cell["conditioner"].split("+"), cell["cell_value"].split("+"))),
        })
        stats_by_rule_id[rule_id] = {
            "wr_pct": float(cell["wr_pct"]),
            "pf": float(cell["pf"]) if cell["pf"] != 999.0 else float("inf"),
            "n": int(cell["n"]),
        }
    return rules, stats_by_rule_id


def main():
    print(f"Loading 2025 backtest: {BACKTEST_DIR}")
    data = load_run(BACKTEST_DIR)
    trades = data.trades
    print(f"Loaded {len(trades):,} trades across {data.sessions_loaded} sessions")

    # Restrict to Validation period
    val_trades = trades[
        (trades["session_date_dt"] >= VALIDATION_START)
        & (trades["session_date_dt"] <= VALIDATION_END)
    ].copy()
    print(f"Validation period ({VALIDATION_START} to {VALIDATION_END}): {len(val_trades):,} trades")

    rules, stats_by_rule = load_approved_rules_and_discovery_stats()
    print(f"Evaluating {len(rules)} Discovery-approved rules")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_validation_gate(
        trades=val_trades,
        approved_rules=rules,
        discovery_stats_by_rule=stats_by_rule,
        report_path=OUT_DIR / "06-validation-results.md",
        survivors_json=OUT_DIR / "stage6_validation_survivors.json",
    )
    print()
    print("=" * 60)
    print(f"Validation gate: {result['n_passed']} PASSED / {result['n_failed']} FAILED")
    print("=" * 60)
    # Per-setup breakdown
    from collections import Counter
    passed_by_setup = Counter(r["setup"] for r in result["per_rule"] if r["passed"])
    failed_by_setup = Counter(r["setup"] for r in result["per_rule"] if not r["passed"])
    print(f"\nPer-setup survivor counts:")
    all_setups = sorted(set(list(passed_by_setup.keys()) + list(failed_by_setup.keys())))
    for s in all_setups:
        p = passed_by_setup.get(s, 0)
        f = failed_by_setup.get(s, 0)
        print(f"  {s:30s} {p:3d} passed / {p + f:3d} total")


if __name__ == "__main__":
    main()
