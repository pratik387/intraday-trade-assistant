#!/usr/bin/env python
"""Parallel-chunked overnight backtest driver.

Same OCI pattern as tools/engine.py (intraday): split the date range into N
chunks, run them in parallel as subprocesses, merge results.

Each chunk has ISOLATED state (per-chunk overnight_slots.json) so they don't
collide. Boundary loss: trades opened on the last 1-2 days of a chunk that
would settle in the next chunk get lost (small — ~2 trades per chunk × N-1
boundaries = a few trades out of hundreds, acceptable for research backtest).

Speedup: 4 workers × N=4 chunks => roughly 4x faster than sequential.
For HO (~210 days), this turns a 5-hour run into ~75 minutes.

USAGE
=====
    .venv/Scripts/python tools/engine_overnight_parallel.py \\
        --from 2025-07-01 --to 2026-04-30 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.util import is_trading_day


def _split_range(start: date, end: date, n_chunks: int) -> List[Tuple[date, date]]:
    """Split [start, end] into n_chunks roughly-equal contiguous chunks.

    Boundaries fall on trading days (so each chunk's last day is a real trading day).
    """
    days = []
    d = start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    if not days:
        return []
    n = len(days)
    chunk_size = (n + n_chunks - 1) // n_chunks
    chunks = []
    for i in range(0, n, chunk_size):
        sub = days[i:i + chunk_size]
        if sub:
            chunks.append((sub[0], sub[-1]))
    return chunks


def _run_chunk(start_iso: str, end_iso: str, chunk_id: int, parent_run_dir: str) -> dict:
    """Run a single chunk in a subprocess via engine_overnight_inprocess.

    Returns the chunk's summary dict (n_trades, sums, cohort split).
    Boundary trades (opened in last 1-2 days of chunk) are lost — handled by
    the next chunk picking up from a fresh state.
    """
    parent = Path(parent_run_dir)
    chunk_out = parent / f"chunk_{chunk_id:02d}"
    chunk_out.mkdir(parents=True, exist_ok=True)

    # Inproc driver creates its own inproc_<ts>/ dir under reports/engine_overnight.
    # We don't want that — we want it under chunk_out. Workaround: have inproc
    # write to a known temp location, then move. Simplest: just call inproc
    # with --from/--to and find its output dir by timestamp.
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "engine_overnight_inprocess.py"),
        "--from", start_iso,
        "--to", end_iso,
    ]
    log_path = chunk_out / "stdout.log"
    print(f"[chunk {chunk_id:02d}] {start_iso} -> {end_iso} (log: {log_path.relative_to(ROOT)})", flush=True)
    with open(log_path, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        return {"chunk_id": chunk_id, "rc": rc, "n_trades": 0, "error": f"non-zero rc {rc}"}

    # Find this chunk's output dir (most recent inproc_* not already claimed)
    # Trick: the inproc driver prints "[+] Run output dir: <path>" in its stdout.
    out_dir = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if "[+] Run output dir:" in line:
                out_dir = Path(line.split(":", 1)[1].strip())
                break
    if out_dir is None or not (out_dir / "summary.json").exists():
        return {"chunk_id": chunk_id, "rc": rc, "n_trades": 0,
                "error": "could not locate inproc output dir"}
    with open(out_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    summary["chunk_id"] = chunk_id
    summary["rc"] = rc
    summary["inproc_dir"] = str(out_dir)
    return summary


def _parse_args():
    ap = argparse.ArgumentParser(description="Parallel-chunked overnight backtest")
    ap.add_argument("--from", dest="from_date", required=True)
    ap.add_argument("--to", dest="to_date", required=True)
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()


def run() -> int:
    args = _parse_args()
    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / "engine_overnight" / f"parallel_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = _split_range(start, end, args.workers)
    print(f"[+] Range {start} .. {end} split into {len(chunks)} chunks for {args.workers} workers")
    for i, (a, b) in enumerate(chunks):
        print(f"    chunk {i:02d}: {a} -> {b}")
    print(f"[+] Output: {out_dir.relative_to(ROOT)}")

    summaries: List[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_run_chunk, a.isoformat(), b.isoformat(), i, str(out_dir)): (i, a, b)
            for i, (a, b) in enumerate(chunks)
        }
        for fut in as_completed(futs):
            i, a, b = futs[fut]
            try:
                s = fut.result()
                summaries.append(s)
                err = s.get("error", "")
                tag = f" ERR={err}" if err else ""
                print(f"  [chunk {i:02d} done] {a}..{b}: n_trades={s.get('n_trades', 0)}{tag}", flush=True)
            except Exception as e:
                print(f"  [chunk {i:02d} FAILED] {a}..{b}: {e!r}", flush=True)

    # Merge: read each chunk's trades.csv into one DF
    import pandas as pd
    all_trades = []
    for s in sorted(summaries, key=lambda x: x.get("chunk_id", 0)):
        inproc = s.get("inproc_dir")
        if not inproc:
            continue
        trades_csv = Path(inproc) / "trades.csv"
        if not trades_csv.exists() or trades_csv.stat().st_size < 100:
            continue
        df = pd.read_csv(trades_csv, comment="#")
        if not df.empty:
            df["chunk_id"] = s["chunk_id"]
            all_trades.append(df)

    if all_trades:
        merged = pd.concat(all_trades, ignore_index=True)
    else:
        merged = pd.DataFrame()
    merged_csv = out_dir / "trades_merged.csv"
    if not merged.empty:
        merged.to_csv(merged_csv, index=False)

    # Compute aggregate summary
    if not merged.empty:
        pnls = merged["realized_pnl_inr"].dropna()
        wins = pnls[pnls > 0].sum()
        losses = -pnls[pnls < 0].sum()
        pf = wins / losses if losses > 0 else float("inf")
        agg = {
            "n_trades": len(pnls),
            "sum_realized_pnl_inr": float(pnls.sum()),
            "wins": int((pnls > 0).sum()),
            "losses": int((pnls < 0).sum()),
            "gross_PF": round(pf, 4),
        }
        # Variant B cohort split
        if "paper_variant_b" in merged.columns:
            b_mask = merged["paper_variant_b"] == True
            b_pnls = merged.loc[b_mask, "realized_pnl_inr"].dropna()
            nb_pnls = merged.loc[~b_mask, "realized_pnl_inr"].dropna()
            def _pf(s):
                w = s[s > 0].sum(); l = -s[s < 0].sum()
                return round(w / l, 4) if l > 0 else None
            agg["cohort_split"] = {
                "variant_b_true_n": int(len(b_pnls)),
                "variant_b_true_PF": _pf(b_pnls),
                "variant_b_true_sum": float(b_pnls.sum()),
                "baseline_only_n": int(len(nb_pnls)),
                "baseline_only_PF": _pf(nb_pnls),
                "baseline_only_sum": float(nb_pnls.sum()),
            }
    else:
        agg = {"n_trades": 0, "error": "no trades emitted across chunks"}

    with open(out_dir / "summary_merged.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": summaries, "merged": agg}, f, indent=2)

    print("\n=== MERGED SUMMARY ===")
    print(json.dumps(agg, indent=2))
    print(f"merged trades: {merged_csv.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
