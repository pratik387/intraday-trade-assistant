"""One-shot OOS validator for gauntlet_v2 (sub-project #5, Phases 3-4).

Applies a frozen config-override dict to a Validation or Holdout period
gate_input dir. Reports metrics + pass/fail. Refuses re-runs against the
same period/output-dir unless --force is passed (master-plan discipline).

Output filenames per period:
    validation -> 07-validation-result.json + 08-validation-report.md
    holdout    -> 09-holdout-result.json    + 10-holdout-report.md

Exit codes:
    0 = pass criteria met
    1 = refused re-run (result file already exists, no --force)
    2 = ran but failed pass criteria
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.gauntlet_v2.trial import run_trial


PERIOD_FILES = {
    "validation": ("07-validation-result.json", "08-validation-report.md"),
    "holdout":    ("09-holdout-result.json", "10-holdout-report.md"),
}

PASS_CRITERIA = {
    "validation": {"pf_min": 1.2, "sharpe_min": 0.7},
    "holdout":    {"pf_min": 1.0, "sharpe_min": 0.5, "losing_days_max_pct": 40.0},
}


def _evaluate_pass(period: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    c = PASS_CRITERIA[period]
    checks = []
    checks.append(("pf", metrics["pf"] >= c["pf_min"], c["pf_min"]))
    checks.append(("sharpe", metrics["sharpe"] >= c["sharpe_min"], c["sharpe_min"]))
    if "losing_days_max_pct" in c:
        checks.append(("losing_days_pct", metrics["losing_days_pct"] <= c["losing_days_max_pct"], c["losing_days_max_pct"]))
    overall = all(ok for _, ok, _ in checks)
    return {"overall": overall, "checks": checks}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-cfg", required=True)
    p.add_argument("--config", required=True, help="Frozen config-override JSON")
    p.add_argument("--gate-input-dir", required=True)
    p.add_argument("--pnl-index", required=True)
    p.add_argument("--period", required=True, choices=list(PERIOD_FILES))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing result (breaks one-shot discipline)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / PERIOD_FILES[args.period][0]
    rep_path = out_dir / PERIOD_FILES[args.period][1]

    if res_path.exists() and not args.force:
        print(
            f"[validate] {res_path} already exists. The master plan requires "
            f"one-shot OOS. Pass --force to override (strongly discouraged).",
            file=sys.stderr,
        )
        sys.exit(1)

    base = json.loads(Path(args.base_cfg).read_text(encoding="utf-8"))
    overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pnl_df = pd.read_parquet(args.pnl_index)

    metrics = run_trial(overrides, Path(args.gate_input_dir), pnl_df, base)
    verdict = _evaluate_pass(args.period, metrics)

    result = {
        "period": args.period,
        "frozen_config_overrides": overrides,
        "metrics": metrics,
        "verdict": verdict,
    }
    res_path.write_text(json.dumps(result, indent=2))

    lines = [
        f"# Gauntlet v2 -- {args.period.upper()} Report",
        "",
        f"- **overall:** {'PASS' if verdict['overall'] else 'FAIL'}",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
        "",
        "## Pass criteria",
        "",
        "| metric | threshold | actual | ok |",
        "|---|---|---|---|",
    ]
    for name, ok, threshold in verdict["checks"]:
        lines.append(f"| {name} | {threshold} | {metrics[name]} | {'PASS' if ok else 'FAIL'} |")
    rep_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(result, indent=2))
    sys.exit(0 if verdict["overall"] else 2)


if __name__ == "__main__":
    main()
