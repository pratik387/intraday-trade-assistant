"""
Compare OLD (ranker.py + planner_internal.py) vs NEW (pipelines) setup processing.

This script analyzes backtest jsonl files to identify:
1. Setups that pass gates but get rejected by quality
2. Screening/gate rejection patterns
3. Approved setups and their scores
4. Category-specific rejection analysis

Usage:
    python tools/compare_old_vs_new_pipeline.py <backtest_dir>
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import json


def categorize_setup(setup_type: str) -> str:
    """Categorize a setup type into BREAKOUT, LEVEL, REVERSION, or MOMENTUM."""
    setup_lower = setup_type.lower()

    # BREAKOUT category
    breakout_keywords = ["breakout", "breakdown", "break_of_structure", "bos", "choc",
                         "orb_breakout", "orb_breakdown", "gap_breakout"]
    if any(kw in setup_lower for kw in breakout_keywords):
        return "BREAKOUT"

    # REVERSION category
    reversion_keywords = ["fade", "failure", "reversal", "vwap_mean", "gap_fill",
                          "liquidity_sweep", "fair_value_gap", "fvg", "range_deviation",
                          "volume_spike_reversal"]
    if any(kw in setup_lower for kw in reversion_keywords):
        return "REVERSION"

    # MOMENTUM category
    momentum_keywords = ["trend_continuation", "trend_pullback", "momentum_trend",
                         "flag_continuation", "squeeze_release"]
    if any(kw in setup_lower for kw in momentum_keywords):
        return "MOMENTUM"

    # LEVEL category (default for support/resistance plays)
    level_keywords = ["support", "resistance", "bounce", "rejection", "vwap_reclaim",
                      "premium", "discount", "order_block", "pullback", "retest"]
    if any(kw in setup_lower for kw in level_keywords):
        return "LEVEL"

    return "UNKNOWN"


def parse_jsonl_logs(backtest_dir: Path) -> Dict[str, Any]:
    """Parse jsonl files to extract rejection and approval patterns."""

    results = {
        "planning_approved": [],
        "planning_rejected": defaultdict(list),
        "screening_rejected": defaultdict(list),
        "ranking_data": [],
        "category_stats": defaultdict(lambda: {"approved": 0, "rejected": 0, "by_reason": defaultdict(int)}),
        "setup_type_stats": defaultdict(lambda: {"approved": 0, "rejected": 0}),
    }

    day_dirs = sorted(backtest_dir.glob("*/"))

    for day_dir in day_dirs:
        # Parse planning.jsonl
        planning_file = day_dir / "planning.jsonl"
        if planning_file.exists():
            with open(planning_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        category = data.get("category", "UNKNOWN")
                        setup_type = data.get("strategy_type", "unknown")
                        action = data.get("action", "")
                        reason = data.get("reason", "")
                        symbol = data.get("symbol", "")

                        if action == "reject":
                            results["planning_rejected"][category].append({
                                "symbol": symbol,
                                "setup_type": setup_type,
                                "reason": reason,
                                "date": day_dir.name
                            })
                            results["category_stats"][category]["rejected"] += 1
                            results["category_stats"][category]["by_reason"][reason] += 1
                            results["setup_type_stats"][setup_type]["rejected"] += 1
                        elif action == "approve" or data.get("stage") == "approved":
                            results["planning_approved"].append({
                                "symbol": symbol,
                                "setup_type": setup_type,
                                "category": category,
                                "structural_rr": data.get("structural_rr"),
                                "score": data.get("score"),
                                "date": day_dir.name
                            })
                            results["category_stats"][category]["approved"] += 1
                            results["setup_type_stats"][setup_type]["approved"] += 1
                    except json.JSONDecodeError:
                        continue

        # Parse screening.jsonl
        screening_file = day_dir / "screening.jsonl"
        if screening_file.exists():
            with open(screening_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("action") == "reject":
                            category = data.get("category", "UNKNOWN")
                            results["screening_rejected"][category].append({
                                "symbol": data.get("symbol", ""),
                                "setup_type": data.get("strategy_type", ""),
                                "reason": data.get("reason", ""),
                                "date": day_dir.name
                            })
                    except json.JSONDecodeError:
                        continue

        # Parse ranking.jsonl
        ranking_file = day_dir / "ranking.jsonl"
        if ranking_file.exists():
            with open(ranking_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        results["ranking_data"].append({
                            **data,
                            "date": day_dir.name
                        })
                    except json.JSONDecodeError:
                        continue

    return results


def parse_agent_logs(backtest_dir: Path) -> Dict[str, Any]:
    """Parse agent.log files to extract GATES_COMPLETE stats."""

    results = {
        "gates_complete": [],
        "quality_logs": [],
        "approved": [],
    }

    # Regex patterns
    gates_pattern = re.compile(r"GATES_COMPLETE \| (\d+)->(\d+) symbols \((\d+\.\d+)%\)")
    quality_pattern = re.compile(r"QUALITY_(\w+): (NSE:\w+) structural_rr=(\d+\.\d+) status=(\w+)")
    approved_pattern = re.compile(r"\[(\w+)\] (NSE:\w+) (\w+) APPROVED: score=(\d+\.\d+)")

    day_dirs = sorted(backtest_dir.glob("*/"))

    for day_dir in day_dirs:
        log_file = day_dir / "agent.log"
        if not log_file.exists():
            continue

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Parse GATES_COMPLETE
                match = gates_pattern.search(line)
                if match:
                    before, after, pct = match.groups()
                    results["gates_complete"].append({
                        "before": int(before),
                        "after": int(after),
                        "pass_rate": float(pct)
                    })
                    continue

                # Parse QUALITY logs
                match = quality_pattern.search(line)
                if match:
                    category, symbol, rr, status = match.groups()
                    results["quality_logs"].append({
                        "category": category,
                        "symbol": symbol,
                        "structural_rr": float(rr),
                        "status": status,
                    })
                    continue

                # Parse APPROVED
                match = approved_pattern.search(line)
                if match:
                    category, symbol, setup_type, score = match.groups()
                    results["approved"].append({
                        "category": category,
                        "symbol": symbol,
                        "setup_type": setup_type,
                        "score": float(score),
                    })

    return results


def print_report(jsonl_results: Dict[str, Any], agent_results: Dict[str, Any]):
    """Print comprehensive analysis report."""

    print("\n" + "="*80)
    print("PIPELINE ANALYSIS REPORT")
    print("="*80)

    # Gate pass rate from agent logs
    if agent_results["gates_complete"]:
        print("\n" + "-"*40)
        print("GATE PASS RATE (from agent.log)")
        print("-"*40)
        total_before = sum(g["before"] for g in agent_results["gates_complete"])
        total_after = sum(g["after"] for g in agent_results["gates_complete"])
        rate = (total_after / total_before * 100) if total_before > 0 else 0
        print(f"Total candidates entering gates: {total_before:,}")
        print(f"Total passing gates: {total_after:,}")
        print(f"Overall gate pass rate: {rate:.1f}%")

    # Category stats from planning.jsonl
    print("\n" + "-"*40)
    print("CATEGORY BREAKDOWN (from planning.jsonl)")
    print("-"*40)

    total_approved = 0
    total_rejected = 0
    for category in ["BREAKOUT", "LEVEL", "REVERSION", "MOMENTUM", "UNKNOWN"]:
        stats = jsonl_results["category_stats"].get(category, {"approved": 0, "rejected": 0, "by_reason": {}})
        approved = stats["approved"]
        rejected = stats["rejected"]
        total = approved + rejected
        total_approved += approved
        total_rejected += rejected

        if total > 0:
            approval_rate = approved / total * 100
            print(f"\n{category}: {total:,} total setups")
            print(f"  - Approved: {approved:,} ({approval_rate:.1f}%)")
            print(f"  - Rejected: {rejected:,} ({100-approval_rate:.1f}%)")

            # Top rejection reasons for this category
            reasons = stats.get("by_reason", {})
            if reasons:
                print(f"  - Top rejection reasons:")
                sorted_reasons = sorted(reasons.items(), key=lambda x: -x[1])[:5]
                for reason, count in sorted_reasons:
                    pct = count / rejected * 100 if rejected > 0 else 0
                    print(f"      {reason}: {count:,} ({pct:.1f}%)")

    print(f"\n{'='*40}")
    print(f"TOTAL: {total_approved + total_rejected:,} setups evaluated")
    print(f"  Approved: {total_approved:,} ({total_approved/(total_approved+total_rejected)*100:.1f}%)")
    print(f"  Rejected: {total_rejected:,} ({total_rejected/(total_approved+total_rejected)*100:.1f}%)")

    # Setup type analysis
    print("\n" + "-"*40)
    print("SETUP TYPE PERFORMANCE")
    print("-"*40)

    setup_stats = jsonl_results["setup_type_stats"]
    sorted_setups = sorted(
        [(k, v) for k, v in setup_stats.items()],
        key=lambda x: -(x[1]["approved"] + x[1]["rejected"])
    )[:20]  # Top 20 setup types

    print(f"\n{'Setup Type':<35} {'Approved':>10} {'Rejected':>10} {'Rate':>8}")
    print("-" * 70)
    for setup_type, stats in sorted_setups:
        approved = stats["approved"]
        rejected = stats["rejected"]
        total = approved + rejected
        rate = approved / total * 100 if total > 0 else 0
        print(f"{setup_type:<35} {approved:>10,} {rejected:>10,} {rate:>7.1f}%")

    # Rejection reason analysis across all categories
    print("\n" + "-"*40)
    print("TOP REJECTION REASONS (ALL CATEGORIES)")
    print("-"*40)

    all_reasons = defaultdict(int)
    for category, rejected_list in jsonl_results["planning_rejected"].items():
        for item in rejected_list:
            all_reasons[item["reason"]] += 1

    total_rejections = sum(all_reasons.values())
    print(f"\nTotal rejections: {total_rejections:,}")

    sorted_reasons = sorted(all_reasons.items(), key=lambda x: -x[1])[:15]
    for reason, count in sorted_reasons:
        pct = count / total_rejections * 100 if total_rejections > 0 else 0
        print(f"  {reason}: {count:,} ({pct:.1f}%)")

    # Key insight: Compare REVERSION/MOMENTUM with BREAKOUT/LEVEL
    print("\n" + "="*80)
    print("KEY INSIGHT: Pipeline Category Imbalance")
    print("="*80)

    breakout_total = jsonl_results["category_stats"]["BREAKOUT"]["approved"] + jsonl_results["category_stats"]["BREAKOUT"]["rejected"]
    level_total = jsonl_results["category_stats"]["LEVEL"]["approved"] + jsonl_results["category_stats"]["LEVEL"]["rejected"]
    reversion_total = jsonl_results["category_stats"]["REVERSION"]["approved"] + jsonl_results["category_stats"]["REVERSION"]["rejected"]
    momentum_total = jsonl_results["category_stats"]["MOMENTUM"]["approved"] + jsonl_results["category_stats"]["MOMENTUM"]["rejected"]

    print(f"\nSetups by category:")
    print(f"  BREAKOUT:  {breakout_total:>6,} ({breakout_total/(breakout_total+level_total+reversion_total+momentum_total+1)*100:.1f}%)")
    print(f"  LEVEL:     {level_total:>6,} ({level_total/(breakout_total+level_total+reversion_total+momentum_total+1)*100:.1f}%)")
    print(f"  REVERSION: {reversion_total:>6,} ({reversion_total/(breakout_total+level_total+reversion_total+momentum_total+1)*100:.1f}%)")
    print(f"  MOMENTUM:  {momentum_total:>6,} ({momentum_total/(breakout_total+level_total+reversion_total+momentum_total+1)*100:.1f}%)")

    reversion_approved = jsonl_results["category_stats"]["REVERSION"]["approved"]
    momentum_approved = jsonl_results["category_stats"]["MOMENTUM"]["approved"]

    if reversion_approved == 0 and reversion_total > 0:
        print(f"\n  [WARNING] REVERSION has {reversion_total} setups but 0 approved!")
        print(f"            This category may have too strict screening/quality filters")

    if momentum_approved == 0 and momentum_total > 0:
        print(f"\n  [WARNING] MOMENTUM has {momentum_total} setups but 0 approved!")
        print(f"            This category may have too strict screening/quality filters")


def main():
    if len(sys.argv) < 2:
        # Try to find the latest backtest directory
        backtest_dirs = list(Path(".").glob("backtest_*_extracted"))
        if not backtest_dirs:
            print("Usage: python tools/compare_old_vs_new_pipeline.py <backtest_dir>")
            print("No backtest directories found in current directory")
            sys.exit(1)
        backtest_dir = max(backtest_dirs, key=lambda p: p.stat().st_mtime)
        print(f"Using latest backtest: {backtest_dir}")
    else:
        backtest_dir = Path(sys.argv[1])

    if not backtest_dir.exists():
        print(f"Error: Directory not found: {backtest_dir}")
        sys.exit(1)

    print(f"Parsing logs from {backtest_dir}...")

    jsonl_results = parse_jsonl_logs(backtest_dir)
    agent_results = parse_agent_logs(backtest_dir)

    print(f"Found {len(jsonl_results['planning_approved'])} approved setups in planning.jsonl")
    print(f"Found {sum(len(v) for v in jsonl_results['planning_rejected'].values())} rejected setups in planning.jsonl")

    print_report(jsonl_results, agent_results)

    # Save detailed analysis
    output_file = backtest_dir / "pipeline_analysis.json"

    # Convert defaultdicts to regular dicts for JSON serialization
    serializable = {
        "approved_count": len(jsonl_results["planning_approved"]),
        "rejected_count": sum(len(v) for v in jsonl_results["planning_rejected"].values()),
        "category_stats": {k: {
            "approved": v["approved"],
            "rejected": v["rejected"],
            "by_reason": dict(v["by_reason"])
        } for k, v in jsonl_results["category_stats"].items()},
        "setup_type_stats": {k: dict(v) for k, v in jsonl_results["setup_type_stats"].items()},
    }

    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
