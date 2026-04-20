#!/usr/bin/env python3
"""Edge Discovery Gauntlet - CLI entry point.

Orchestrates Stages 1-3 + Stage 5 template generation. Stage 4 (SHAP) is
optional and deferred until Stage 3 produces survivors.

Usage:
    python tools/edge_discovery/run_gauntlet.py \\
        --backtest-dir cloud_results/20260419_discovery \\
        --output-dir docs/edge_discovery/2026-04-20-run \\
        --discovery-start 2023-01-01 --discovery-end 2024-12-31 \\
        --validation-start 2025-01-01 --validation-end 2025-09-30 \\
        --holdout-start 2025-10-01 --holdout-end 2026-03-31

Per spec Section 3.6.
"""
import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict

from tools.edge_discovery.data_loader import load_run
from tools.edge_discovery.periods import DiscoveryConfig
from tools.edge_discovery.stages.stage1_universe_prune import run_stage1
from tools.edge_discovery.stages.stage2_univariate import run_stage2
from tools.edge_discovery.stages.stage3_conditional import run_stage3
from tools.edge_discovery.stages.stage5_narrative import generate_narrative_templates


def run_gauntlet_all(
    backtest_dir: Path,
    output_dir: Path,
    cfg_dates: Dict[str, date],
) -> Dict[str, Any]:
    """Run full gauntlet: load data, then stages 1, 2, 3, 5.

    Stage 4 (SHAP) is skipped in this orchestrator. It can be run separately
    if Stage 3 produces survivors.
    """
    backtest_dir = Path(backtest_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DiscoveryConfig(**cfg_dates)

    (output_dir / "00-run-config.json").write_text(
        json.dumps(
            {
                "backtest_dir": str(backtest_dir),
                "discovery_start": cfg.discovery_start.isoformat(),
                "discovery_end": cfg.discovery_end.isoformat(),
                "validation_start": cfg.validation_start.isoformat(),
                "validation_end": cfg.validation_end.isoformat(),
                "holdout_start": cfg.holdout_start.isoformat(),
                "holdout_end": cfg.holdout_end.isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[gauntlet] Loading {backtest_dir} ...")
    data = load_run(backtest_dir)
    trades = data.trades
    trades = trades[
        (trades["session_date_dt"] >= cfg.discovery_start)
        & (trades["session_date_dt"] <= cfg.discovery_end)
    ].copy()
    print(f"[gauntlet] Loaded {len(trades):,} Discovery trades across "
          f"{data.sessions_loaded} sessions")

    print("[gauntlet] Stage 1: Universe pruning ...")
    s1 = run_stage1(
        trades,
        report_path=output_dir / "01-universe-pruning.md",
        survivors_json=output_dir / "stage1_survivors.json",
    )
    s1_survivors = [r["setup"] for r in s1 if r["passed"]]
    print(f"[gauntlet]   Stage 1 survivors: {len(s1_survivors)}")

    print("[gauntlet] Stage 2: Univariate screening ...")
    s2 = run_stage2(
        trades,
        cfg=cfg,
        survivors_input=s1_survivors,
        report_path=output_dir / "02-univariate-screening.md",
        survivors_json=output_dir / "stage2_survivors.json",
    )
    s2_survivors = [r["setup"] for r in s2 if r["passed"]]
    print(f"[gauntlet]   Stage 2 survivors: {len(s2_survivors)}")

    print("[gauntlet] Stage 3: Conditional edge ...")
    s3 = run_stage3(
        trades,
        cfg=cfg,
        survivors_input=s2_survivors,
        report_path=output_dir / "03-conditional-edge.md",
        survivors_json=output_dir / "stage3_survivors.json",
    )
    s3_pass_cells = [c for c in s3 if c["passed"]]
    print(f"[gauntlet]   Stage 3 passing cells: {len(s3_pass_cells)}")

    print("[gauntlet] Stage 5: Narrative templates ...")
    s3_json = json.loads(
        (output_dir / "stage3_survivors.json").read_text(encoding="utf-8")
    )
    narrative_paths = generate_narrative_templates(
        stage3_survivors=s3_json["survivors"],
        stage3_details=s3_json["details"],
        out_dir=output_dir / "05-narrative-gate",
    )
    print(f"[gauntlet]   Templates generated: {len(narrative_paths)}")

    return {
        "stage1_count": len(s1_survivors),
        "stage2_count": len(s2_survivors),
        "stage3_count": len(s3_pass_cells),
        "narrative_templates_generated": len(narrative_paths),
    }


def main():
    p = argparse.ArgumentParser(description="Edge Discovery Gauntlet")
    p.add_argument("--backtest-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--discovery-start", required=True)
    p.add_argument("--discovery-end", required=True)
    p.add_argument("--validation-start", required=True)
    p.add_argument("--validation-end", required=True)
    p.add_argument("--holdout-start", required=True)
    p.add_argument("--holdout-end", required=True)
    args = p.parse_args()

    cfg_dates = {
        "discovery_start": date.fromisoformat(args.discovery_start),
        "discovery_end": date.fromisoformat(args.discovery_end),
        "validation_start": date.fromisoformat(args.validation_start),
        "validation_end": date.fromisoformat(args.validation_end),
        "holdout_start": date.fromisoformat(args.holdout_start),
        "holdout_end": date.fromisoformat(args.holdout_end),
    }
    result = run_gauntlet_all(
        backtest_dir=Path(args.backtest_dir),
        output_dir=Path(args.output_dir),
        cfg_dates=cfg_dates,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
