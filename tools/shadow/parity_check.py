"""Parity check - sub-project #4.

Two modes:
  Parity: --live <events_decisions.jsonl> --sim <sim_admits.csv>
    Asserts admit-set equality. Exits 0 on match, 1 on divergence.
  A/B:    --baseline <csv> --variant <csv>
    Reports admit count delta, setup mix delta, hourly distribution delta.
    Always exits 0 (informational).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Set, Tuple


def _load_sim_admits(path: Path) -> Set[Tuple[str, str, str]]:
    """Returns set of (ts, symbol, setup_type) for stage=admitted rows."""
    out = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("stage") == "admitted":
                out.add((r["ts"], r["symbol"], r["setup_type"]))
    return out


def _load_live_admits(path: Path) -> Set[Tuple[str, str, str]]:
    """Returns set of (timestamp, symbol, strategy_type) from events_decisions.jsonl
    for rows where action != reject."""
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("action") == "reject":
                continue
            out.add((o.get("timestamp", ""), o.get("symbol", ""), o.get("strategy_type", "")))
    return out


def _run_parity(live_path: Path, sim_path: Path) -> int:
    live = _load_live_admits(live_path)
    sim = _load_sim_admits(sim_path)
    common = live & sim
    only_live = live - sim
    only_sim = sim - live

    if not only_live and not only_sim:
        print(f"MATCH {len(common)}/{len(live)} admits — bit-exact parity")
        return 0

    print(f"DIVERGENCE: {len(common)}/{max(len(live), len(sim))} match")
    print(f"  in LIVE but missing from SIM ({len(only_live)}):")
    for ts, sym, setup in sorted(only_live):
        print(f"    {ts}  {sym:<20s}  {setup}")
    print(f"  in SIM but missing from LIVE ({len(only_sim)}):")
    for ts, sym, setup in sorted(only_sim):
        print(f"    {ts}  {sym:<20s}  {setup}")
    return 1


def _hour_of(ts: str) -> str:
    # ts is ISO; "2025-01-02T09:20:00" -> "09"
    try:
        return ts.split("T", 1)[1][:2]
    except (IndexError, AttributeError):
        return "??"


def _run_ab(baseline_path: Path, variant_path: Path) -> int:
    base = _load_sim_admits(baseline_path)
    var = _load_sim_admits(variant_path)

    print(f"admit count: {len(base)} -> {len(var)} (delta {len(var) - len(base):+d})")

    base_setups = Counter(s for _, _, s in base)
    var_setups = Counter(s for _, _, s in var)
    all_setups = sorted(set(base_setups) | set(var_setups))
    print("\nsetup mix:")
    print(f"  {'setup':<30s}  baseline  variant  delta")
    for s in all_setups:
        b, v = base_setups.get(s, 0), var_setups.get(s, 0)
        print(f"  {s:<30s}  {b:>8d}  {v:>7d}  {v-b:+d}")

    base_hours = Counter(_hour_of(ts) for ts, _, _ in base)
    var_hours = Counter(_hour_of(ts) for ts, _, _ in var)
    all_hours = sorted(set(base_hours) | set(var_hours))
    print("\nhourly distribution:")
    print(f"  {'hour':<6s}  baseline  variant  delta")
    for h in all_hours:
        b, v = base_hours.get(h, 0), var_hours.get(h, 0)
        print(f"  {h:<6s}  {b:>8d}  {v:>7d}  {v-b:+d}")

    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", help="Path to events_decisions.jsonl (parity mode)")
    p.add_argument("--sim", help="Path to sim_admits.csv (parity mode)")
    p.add_argument("--baseline", help="Path to baseline sim CSV (A/B mode)")
    p.add_argument("--variant", help="Path to variant sim CSV (A/B mode)")
    args = p.parse_args()

    if args.live and args.sim:
        sys.exit(_run_parity(Path(args.live), Path(args.sim)))
    if args.baseline and args.variant:
        sys.exit(_run_ab(Path(args.baseline), Path(args.variant)))
    p.error("specify either (--live + --sim) or (--baseline + --variant)")


if __name__ == "__main__":
    main()
