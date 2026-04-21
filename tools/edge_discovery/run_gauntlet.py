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

import pandas as pd

from tools.edge_discovery.data_loader import load_run
from tools.edge_discovery.periods import DiscoveryConfig
from tools.edge_discovery.stages.stage1_universe_prune import run_stage1
from tools.edge_discovery.stages.stage2_univariate import run_stage2
from tools.edge_discovery.stages.stage3_conditional import run_stage3
from tools.edge_discovery.stages.stage5_narrative import generate_narrative_templates
from tools.edge_discovery.stages.stage5b_ruleset_simulation import run_stage5b
from tools.edge_discovery.stages.stage5c_cross_sectional_simulation import run_stage5c

ROOT = Path(__file__).parent.parent.parent


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

    # Stage 5b: ruleset simulation — aggregate behavior of approved rules.
    # Approved = Stage 3 survivors minus narrative-gate REJECTED_RULES (if present).
    print("[gauntlet] Stage 5b: Ruleset simulation ...")
    approved_rules = _load_approved_rules(s3_json)
    if approved_rules:
        run_stage5b(
            trades=trades,
            approved_rules=approved_rules,
            report_path=output_dir / "06-ruleset-simulation.md",
            summary_json=output_dir / "stage5b_simulation.json",
        )
        print(f"[gauntlet]   Simulated {len(approved_rules)} approved rules as union filter")
    else:
        print("[gauntlet]   Skipped — no approved rules to simulate")

    # Stage 5c: cross-sectional filter simulation (F1+F2)
    stage5c_run = False
    print("[gauntlet] Stage 5c: Cross-sectional filter simulation ...")
    try:
        cfg_path = ROOT / "config" / "configuration.json"
        full_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cs_cfg = full_cfg.get("cross_sectional_gate")
        if cs_cfg and cs_cfg.get("enabled") and approved_rules:
            # Normalize column names to what Stage 5c expects:
            # - `setup` (stage5c) vs `setup_type` (data_loader)
            # - `symbol_raw` (stage5c) vs `symbol` with "NSE:" prefix (data_loader)
            trades_5c = trades.copy()
            if "setup" not in trades_5c.columns and "setup_type" in trades_5c.columns:
                trades_5c["setup"] = trades_5c["setup_type"]
            if "symbol_raw" not in trades_5c.columns:
                trades_5c["symbol_raw"] = trades_5c["symbol"].astype(str).str.replace(
                    "NSE:", "", regex=False
                )

            # Restrict to approved-filter trades (match Stage 5b universe)
            from tools.edge_discovery.stages.stage5b_ruleset_simulation import apply_filter
            filtered_trades = apply_filter(trades_5c, approved_rules)

            # Synthesize decision_ts if missing (data_loader emits session_date_dt +
            # minute_of_day; derive timestamp at bar-close)
            if "decision_ts" not in filtered_trades.columns:
                filtered_trades = filtered_trades.copy()
                filtered_trades["decision_ts"] = (
                    pd.to_datetime(filtered_trades["session_date_dt"])
                    + pd.to_timedelta(filtered_trades["minute_of_day"].astype(int), unit="m")
                ).dt.strftime("%Y-%m-%d %H:%M:%S")

            # Load OHLCV from monthly feathers — same source as probe
            monthly_dir = ROOT / "backtest-cache-download" / "monthly"
            trade_syms = set(filtered_trades["symbol_raw"].unique())
            ohlcv_parts = []
            for f in sorted(monthly_dir.glob("*_5m_enriched.feather")):
                df = pd.read_feather(f, columns=["date", "symbol", "volume"])
                df = df[df["symbol"].isin(trade_syms)]
                ohlcv_parts.append(df)
            if ohlcv_parts:
                ohlcv_big = pd.concat(ohlcv_parts, ignore_index=True)
                if ohlcv_big["date"].dt.tz is not None:
                    ohlcv_big["ts"] = ohlcv_big["date"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                else:
                    ohlcv_big["ts"] = ohlcv_big["date"]
                ohlcv_big["mod"] = (ohlcv_big["ts"].dt.hour * 60 + ohlcv_big["ts"].dt.minute).astype("int16")
                ohlcv_big["date_only"] = ohlcv_big["ts"].dt.date
                ohlcv_big = ohlcv_big[["symbol", "date_only", "mod", "volume"]]
            else:
                ohlcv_big = pd.DataFrame(columns=["symbol", "date_only", "mod", "volume"])

            print(f"[gauntlet]   Stage 5c inputs: trades={len(filtered_trades):,} "
                  f"ohlcv_rows={len(ohlcv_big):,} symbols={len(trade_syms):,}")
            run_stage5c(
                trades=filtered_trades,
                ohlcv=ohlcv_big,
                cfg=cs_cfg,
                report_path=output_dir / "07-cross-sectional-simulation.md",
                summary_json=output_dir / "stage5c_simulation.json",
            )
            stage5c_run = True
            print("[gauntlet]   Stage 5c complete")
        elif not cs_cfg or not cs_cfg.get("enabled"):
            print("[gauntlet]   Stage 5c skipped (cross_sectional_gate disabled in config)")
        else:
            print("[gauntlet]   Stage 5c skipped (no approved rules)")
    except Exception as e:
        print(f"[gauntlet]   Stage 5c ERROR: {e}")
        import traceback
        traceback.print_exc()

    return {
        "stage1_count": len(s1_survivors),
        "stage2_count": len(s2_survivors),
        "stage3_count": len(s3_pass_cells),
        "narrative_templates_generated": len(narrative_paths),
        "stage5b_rules_simulated": len(approved_rules),
        "stage5c_run": stage5c_run,
    }


def _load_approved_rules(s3_json: Dict[str, Any]) -> list:
    """Build the list of approved rules for Stage 5b.

    Approved = every Stage 3 passing cell, minus any rule_id listed in
    `tools.edge_discovery.fill_narratives.REJECTED_RULES` (the narrative
    gate's rejection registry). If fill_narratives is unavailable, all
    Stage 3 survivors are treated as approved (fail-open).
    """
    try:
        from tools.edge_discovery.fill_narratives import REJECTED_RULES
        rejected = REJECTED_RULES
    except Exception:
        rejected = set()
    rules = []
    for cell in s3_json.get("details", []):
        if not cell.get("passed"):
            continue
        rule_id = f"{cell['setup']}__{cell['conditioner']}={cell['cell_value']}"
        if rule_id in rejected:
            continue
        conditions = list(zip(cell["conditioner"].split("+"), cell["cell_value"].split("+")))
        rules.append({"rule_id": rule_id, "setup": cell["setup"], "conditions": conditions})
    return rules


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
