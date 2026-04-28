"""Per-cell Optuna search for borderline Stage-3 cells.

Patches the gauntlet v2 chain so it can operate on sub8 setups that aren't in
the validated survivors JSON or the conviction model's training distribution:

  1. Generate a synthetic survivors JSON containing ONLY the target cell rule
     so RuleFilterGate admits matching candidates.
  2. Force live_gate_chain.enabled = True so the 8 search dims actually fire
     (search.py forgot to flip this — it only forces wide_open_mode=False).
  3. Conviction model's predictions for sub8 are noise (model trained on
     pre-sub8 data). The optimizer's `min_predicted_r` sweep finds the
     threshold that lets sub8 through; if all predictions cluster, the gate
     becomes binary (pass-all vs reject-all) and Optuna picks the right side.

Run per cell, compare best Sharpe / PF vs the unfiltered baseline. If the
filtered cell clears Stage 3 thresholds (PF >= 1.30, h1+h2 PF >= 1.10), it
joins the survivor set; otherwise it stays killed.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from tools.gauntlet_v2.search import run_search


def make_cell_survivors_json(setup: str, conditioner_keys: List[str], conditioner_vals: List[str], out_path: Path) -> Path:
    """Write a single-rule survivors JSON for one cell."""
    rule_id = f"{setup}__{'+'.join(conditioner_keys)}={'+'.join(conditioner_vals)}"
    payload = {
        "stage": "borderline_passthrough",
        "criteria": {"min_pf": 0.0, "min_n": 0, "max_wr_delta_pp": 999.0},
        "n_passed": 1,
        "n_failed": 0,
        "survivors": [{"setup": setup, "rule_id": rule_id}],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def patch_base_cfg_for_cell(base_cfg: Dict[str, Any], survivors_path: Path) -> Dict[str, Any]:
    """Return a deep-copied cfg with the chain enabled + survivors path overridden."""
    cfg = copy.deepcopy(base_cfg)
    cfg["wide_open_mode"] = False
    cfg.setdefault("live_gate_chain", {})["enabled"] = True
    cfg.setdefault("rule_filter_gate", {})["survivors_path"] = str(
        survivors_path.resolve().relative_to(_REPO_ROOT)
        if survivors_path.is_absolute() and str(survivors_path).startswith(str(_REPO_ROOT))
        else survivors_path
    )
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", required=True, help="Setup name e.g. orb_15")
    p.add_argument("--conditioner-key", required=True, help="cap_segment | regime | hour_bucket | <k1+k2>")
    p.add_argument("--conditioner-val", required=True, help="mid_cap | chop | ... | <v1+v2>")
    p.add_argument("--gate-input-dir", required=True)
    p.add_argument("--pnl-index", required=True, help="Per-cell filtered pnl_index parquet")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-cfg", default="config/configuration.json")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--min-n-trades", type=int, default=300)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = args.conditioner_key.split("+")
    vals = args.conditioner_val.split("+")
    survivors_path = out_dir / f"survivors__{args.setup}.json"
    make_cell_survivors_json(args.setup, keys, vals, survivors_path)
    print(f"[search_borderline] wrote {survivors_path}")

    base_cfg = json.loads(Path(args.base_cfg).read_text(encoding="utf-8"))
    base_cfg = patch_base_cfg_for_cell(base_cfg, survivors_path)
    print(f"[search_borderline] patched cfg: wide_open_mode=False, live_gate_chain.enabled=True, "
          f"rule_filter_gate.survivors_path={base_cfg['rule_filter_gate']['survivors_path']}")

    pnl_df = pd.read_parquet(args.pnl_index)
    print(f"[search_borderline] pnl_index: {len(pnl_df):,} rows for cell {args.setup}+{args.conditioner_key}={args.conditioner_val}")

    run_search(
        base_cfg=base_cfg,
        gate_input_dir=Path(args.gate_input_dir),
        pnl_index=pnl_df,
        output_dir=out_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        min_n_trades=args.min_n_trades,
    )

    # Read back the report for a quick stdout summary
    report = (out_dir / "06-search-report.md").read_text(encoding="utf-8")
    print("\n" + "=" * 80)
    print(report.split("## Top 10 by sharpe")[0])
    print("=" * 80)


if __name__ == "__main__":
    main()
