"""Build per-setup decay-monitor input parquets from production-truth backtest events.

Each per-day backtest directory contains an `events.jsonl` with EXIT records
of the form:
    {"type": "EXIT", "ts": "<YYYY-MM-DD HH:MM:SS>", "exit": {"pnl": <float>,
     "diagnostics": {"setup_type": "<name>", ...}}}

This script walks one or more `<run>_full/` directories, aggregates EXIT
events by setup, and writes a per-setup parquet at
`reports/decay_inputs/<setup>.parquet` with the schema expected by
`decay_monitor_runner` (`signal_date` + `net_pnl_inr` columns).

Why not use `_walkfwd_combined_*_canonical.csv`? Those files contain broader-
detector fires (pre-cell-lock / pre-some-other-gate) and overstate the trade
count by 10-50× vs production. The PF measured from them is therefore not
production-truth, and the decay monitor built on them was misleading. Per-day
`events.jsonl` is the actual production-truth source.

Usage:
    python -m tools.edge_discovery.prepare_decay_inputs

Source dirs are set in BACKTEST_DIRS below — edit when you have a fresh run.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "reports" / "decay_inputs"

# Full-history backtest run directories. Order doesn't matter — the script
# concatenates and sorts by signal_date.
BACKTEST_DIRS: list[str] = [
    "20260601-124154_full",  # 2023-01 to 2024-12
    "20260601-133615_full",  # 2025-01 to 2026-04
]


def _iter_exit_events(events_path: Path):
    """Yield (setup_type, signal_date, pnl) tuples from one events.jsonl."""
    if not events_path.exists() or events_path.stat().st_size == 0:
        return
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "EXIT":
                continue
            exit_blk = rec.get("exit") or {}
            diag = exit_blk.get("diagnostics") or {}
            setup = diag.get("setup_type")
            pnl = exit_blk.get("pnl")
            ts = rec.get("ts") or ""
            if not setup or pnl is None or len(ts) < 10:
                continue
            yield setup, ts[:10], float(pnl)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    # setup -> list of (signal_date, net_pnl_inr) rows
    rows_by_setup: dict[str, list[tuple[str, float]]] = defaultdict(list)
    dir_count = 0
    for rel_root in BACKTEST_DIRS:
        root = _REPO / rel_root
        if not root.exists():
            print(f"WARN: backtest dir missing: {root}")
            continue
        for day_dir in sorted(root.iterdir()):
            if not day_dir.is_dir():
                continue
            dir_count += 1
            for setup, sig_date, pnl in _iter_exit_events(day_dir / "events.jsonl"):
                rows_by_setup[setup].append((sig_date, pnl))
    print(f"scanned {dir_count} day-dirs across {len(BACKTEST_DIRS)} backtest runs")

    for setup, rows in sorted(rows_by_setup.items()):
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["signal_date", "net_pnl_inr"])
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        df = df.sort_values("signal_date").reset_index(drop=True)
        out_path = _OUT / f"{setup}.parquet"
        df.to_parquet(out_path, index=False)
        print(
            f"{setup:<35s}  exits={len(df):>6}  "
            f"range=[{df.signal_date.min().date()} -> {df.signal_date.max().date()}]"
        )


if __name__ == "__main__":
    main()
