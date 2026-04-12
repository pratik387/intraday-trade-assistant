"""Analyze timing.jsonl from an instrumented backtest run.

Produced by utils.perf_timer when TRADING_PERF_TIMER=1 is set in the environment.
Reports total/avg/p50/p95/max duration per (stage, substage), ranked by total
wall time, and highlights per-detector-class breakdown from main_detector events.

Usage:
    python tools/analyze_timing.py logs/backtest_20260411_120156/timing.jsonl
    python tools/analyze_timing.py --top 30 logs/backtest_latest/timing.jsonl
    python tools/analyze_timing.py --stage gate logs/backtest_latest/timing.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load(path: Path):
    rows = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                skipped += 1
    return rows, skipped


def _pctile(sorted_vals, p: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = int(round(p * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return sorted_vals[idx]


def _aggregate(rows, stage_filter=None):
    by_key = defaultdict(list)
    by_class_detector = defaultdict(float)
    n_marks = 0
    for r in rows:
        if r.get("event") == "mark":
            n_marks += 1
            continue
        stage = r.get("stage", "?")
        substage = r.get("substage", "")
        if stage_filter and stage != stage_filter:
            continue
        d = r.get("duration_ms")
        if d is None:
            continue
        by_key[(stage, substage)].append(float(d))

        # Special handling for main_detector per-class breakdown
        if stage == "detector" and substage == "aggregate":
            by_class = r.get("by_class_ms", {})
            if isinstance(by_class, dict):
                for cname, cms in by_class.items():
                    try:
                        by_class_detector[cname] += float(cms)
                    except Exception:
                        pass

    return by_key, by_class_detector, n_marks


def _print_top_stages(by_key, top_n: int):
    rows = []
    for (stage, substage), vals in by_key.items():
        vals.sort()
        n = len(vals)
        total = sum(vals)
        rows.append(
            {
                "stage": stage,
                "substage": substage,
                "n_calls": n,
                "total_s": total / 1000.0,
                "avg_ms": total / n if n else 0.0,
                "p50_ms": _pctile(vals, 0.50),
                "p95_ms": _pctile(vals, 0.95),
                "max_ms": vals[-1] if vals else 0.0,
            }
        )
    rows.sort(key=lambda r: -r["total_s"])
    grand_total = sum(r["total_s"] for r in rows)

    print()
    print("=" * 105)
    print(
        f"{'STAGE':<12} {'SUBSTAGE':<28} {'N':>8} "
        f"{'TOTAL_s':>10} {'AVG_ms':>10} {'P50_ms':>10} {'P95_ms':>10} {'MAX_ms':>10}  {'%':>6}"
    )
    print("-" * 105)
    for r in rows[:top_n]:
        pct = (r["total_s"] / grand_total * 100.0) if grand_total else 0.0
        print(
            f"{r['stage']:<12} {r['substage']:<28} {r['n_calls']:>8,} "
            f"{r['total_s']:>10.2f} {r['avg_ms']:>10.2f} {r['p50_ms']:>10.2f} "
            f"{r['p95_ms']:>10.2f} {r['max_ms']:>10.2f}  {pct:>5.1f}%"
        )
    print("-" * 105)
    print(f"Grand total instrumented time: {grand_total:.2f}s ({grand_total/60:.2f} min)")
    if len(rows) > top_n:
        print(f"(showing top {top_n} of {len(rows)} stages)")


def _print_detector_breakdown(by_class):
    if not by_class:
        return
    total = sum(by_class.values()) / 1000.0
    print()
    print("Detector breakdown by class (from structures/main_detector.py):")
    print("-" * 60)
    print(f"{'CLASS':<30} {'TOTAL_s':>10}  {'%':>6}")
    print("-" * 60)
    for cname, cms in sorted(by_class.items(), key=lambda x: -x[1]):
        sec = cms / 1000.0
        pct = (sec / total * 100.0) if total else 0.0
        print(f"{cname:<30} {sec:>10.2f}  {pct:>5.1f}%")
    print("-" * 60)
    print(f"Total detector wall time: {total:.2f}s ({total/60:.2f} min)")


def _summarize_pids(rows):
    pid_counts = defaultdict(int)
    for r in rows:
        pid = r.get("pid")
        if pid is not None:
            pid_counts[pid] += 1
    return pid_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="Path to timing.jsonl")
    ap.add_argument("--top", type=int, default=20, help="Top N stages to show (default 20)")
    ap.add_argument("--stage", type=str, default=None, help="Filter to a single stage (scan/gate/orch/detector)")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    rows, skipped = _load(args.path)
    print(f"Loaded {len(rows):,} timing events from {args.path}")
    if skipped:
        print(f"Skipped {skipped} malformed lines")
    print(f"File size: {args.path.stat().st_size / 1024:.1f} KB")

    pid_counts = _summarize_pids(rows)
    if pid_counts:
        print(f"Processes recorded: {len(pid_counts)} (pids: {sorted(pid_counts)[:5]}{'...' if len(pid_counts) > 5 else ''})")

    by_key, by_class, n_marks = _aggregate(rows, stage_filter=args.stage)
    if n_marks:
        print(f"Marker events (zero-duration): {n_marks}")

    if not by_key:
        print("\nNo duration events found. Was TRADING_PERF_TIMER=1 set during the backtest?")
        return

    _print_top_stages(by_key, args.top)
    if not args.stage:
        _print_detector_breakdown(by_class)


if __name__ == "__main__":
    main()
