#!/usr/bin/env python
"""OCI-style backtest driver for OVERNIGHT setups (currently: close_dn_overnight_long).

Mirrors tools/engine.py's pattern but with the multi-day state-machine awareness
that overnight setups require. Each trading day in the range gets TWO subprocess
invocations:

  1. `--action=verify-exit` (at simulated 09:30 IST): settles AMO fills from
     positions opened the prior trading day; releases T+2 cash settles from
     two-trading-days-ago positions.
  2. `--action=entry` (at simulated 15:25 IST): runs detector across the
     close_dn_overnight_long universe, reserves slots, places MOC BUY + AMO SELL.

State persists across day boundaries via the slot pool (`state/overnight_slots.json`)
and decay tripwire (`state/decay_tripwire_close_dn_overnight_long.json`) files.

Unlike intraday engine.py, these days CANNOT be parallelized — the state files
are shared mutable inputs/outputs of each subprocess. The flow is strictly
sequential.

After the loop, a final verify-exit pass on the day AFTER END_DATE settles any
positions that opened on END_DATE itself.

OUTPUT
======
A per-run directory under reports/engine_overnight/ contains:
  - trades.csv      — one row per settled overnight position
  - summary.json    — aggregate PnL, PF, win rate, regime split
  - per_day.csv     — daily counts (fires, settles, releases, orphans)
  - daily_logs/     — captured stderr from each subprocess invocation

USAGE
=====
Edit the SETTINGS block below (START_DATE, END_DATE), then:

    .venv/Scripts/python tools/engine_overnight.py

Optional CLI overrides:

    --from <YYYY-MM-DD>   override START_DATE
    --to   <YYYY-MM-DD>   override END_DATE
    --no-clear-state      do NOT delete prior state files (continue from existing state)
    --max-workers N       (sequential by design; N is ignored, retained for engine.py parity)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# ----- repo root on sys.path -----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.util import is_trading_day  # noqa: E402


def _load_setups_config() -> dict:
    """Read config/configuration.json (the setups source-of-truth).

    NOT load_base_config() — base_config.json is a separate file and does not
    contain the `setups.*` blocks.
    """
    cfg_path = ROOT / "config" / "configuration.json"
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)

# ====== SETTINGS ======
START_DATE = "2026-04-20"   # YYYY-MM-DD (inclusive)
END_DATE   = "2026-04-30"   # YYYY-MM-DD (inclusive)
MAIN_PATH  = ROOT / "main.py"
# ======================


def _daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def _next_trading_day(d: date, max_lookahead: int = 7) -> date:
    """Next NSE trading day after d, scanning ahead up to max_lookahead days."""
    cur = d + timedelta(days=1)
    for _ in range(max_lookahead):
        if is_trading_day(cur):
            return cur
        cur += timedelta(days=1)
    raise RuntimeError(f"no trading day found within {max_lookahead} days after {d}")


def _build_cmd(py_exe: str, day: date, action: str) -> List[str]:
    """Build subprocess command for a single day's overnight action.

    main.py treats --mode=overnight + --action={entry,verify-exit} as a
    short-lived cron-style invocation (does NOT start the intraday daemon).
    """
    root_path = str(ROOT).replace("\\", "/")
    main_path = str(MAIN_PATH).replace("\\", "/")
    cmd = [
        py_exe, "-c",
        f"import sys; sys.path.insert(0, r'{root_path}'); "
        f"__file__ = r'{main_path}'; "
        f"exec(open(r'{main_path}').read())",
        "--dry-run",
        "--session-date", day.isoformat(),
        "--mode", "overnight",
        "--action", action,
    ]
    return cmd


def _run_action(day: date, action: str, log_dir: Path) -> Tuple[int, str, str]:
    """Run a single overnight action subprocess. Capture stderr to log_dir."""
    py = sys.executable
    cmd = _build_cmd(py, day, action)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{day.isoformat()}_{action}.log"
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        # Persist captured streams for post-mortem
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n")
        # Extract one-line summary from stderr (overnight handlers print to stderr)
        summary_line = ""
        for ln in (result.stderr or "").splitlines():
            if ln.startswith(f"[overnight {action.replace('-', ' ')}"):
                summary_line = ln.strip()
                break
        return (result.returncode, summary_line, str(log_path))
    except Exception as e:
        return (999, f"exception: {e!r}", str(log_path))


def _state_files_from_config(cfg: dict) -> List[Path]:
    """Resolve all overnight-setup state file paths from config."""
    out: List[Path] = []
    setups = cfg.get("setups", {})
    for name, raw in setups.items():
        if not isinstance(raw, dict) or raw.get("mode") != "overnight":
            continue
        ca = raw.get("capital_allocation", {})
        sf = ca.get("state_file")
        if sf:
            out.append(ROOT / sf)
        dt = raw.get("decay_tripwire", {})
        sf2 = dt.get("state_file")
        if sf2:
            out.append(ROOT / sf2)
    return out


def _clear_state_files(state_paths: List[Path]) -> None:
    """Remove prior state files so the backtest starts fresh."""
    for sp in state_paths:
        if sp.exists():
            print(f"[engine_overnight] removing prior state file: {sp}")
            sp.unlink()


def _aggregate_trades(state_paths: List[Path], out_dir: Path) -> dict:
    """Read the slot pool state file and emit trades.csv + summary.json.

    The slot pool's persisted JSON contains all slots that have transitioned
    through t0_open -> t1_settling/settled. Settled slots carry realized_pnl_inr.
    """
    import csv

    trades: List[dict] = []
    slot_state: Optional[Path] = next(
        (p for p in state_paths if p.name == "overnight_slots.json"), None
    )
    if slot_state is None or not slot_state.exists():
        print(f"[engine_overnight] no slot state file found; trades.csv will be empty")
    else:
        with open(slot_state, encoding="utf-8") as f:
            blob = json.load(f)
        slots = blob.get("slots", [])
        for s in slots:
            # Only emit slots that completed the full lifecycle (have a sell fill)
            if s.get("sell_fill_price") is None:
                continue
            trades.append({
                "slot_id": s.get("slot_id"),
                "symbol": s.get("symbol"),
                "product": s.get("product"),
                "reserved_today": s.get("reserved_today"),
                "expected_exit_date": s.get("expected_exit_date"),
                "buy_fill_price": s.get("buy_fill_price"),
                "sell_fill_price": s.get("sell_fill_price"),
                "buy_fill_ts": s.get("buy_fill_ts"),
                "sell_fill_ts": s.get("sell_fill_ts"),
                "qty": int(round((s.get("notional_inr") or 0.0) / (s.get("buy_fill_price") or 1.0))),
                "notional_inr": s.get("notional_inr"),
                "margin_inr": s.get("margin_inr"),
                "leverage": s.get("leverage"),
                "fees_inr": s.get("fees_inr"),
                "interest_inr": s.get("interest_inr"),
                "realized_pnl_inr": s.get("realized_pnl_inr"),
                "paper_variant_b": s.get("paper_variant_b"),
                "status": s.get("status"),
            })

    # Write CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trades.csv"
    if trades:
        cols = list(trades[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(trades)
    else:
        # Empty placeholder
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("# no trades emitted during run\n")

    # Summary stats
    summary = {
        "n_trades": len(trades),
        "sum_realized_pnl_inr": sum((t.get("realized_pnl_inr") or 0.0) for t in trades),
        "sum_fees_inr": sum((t.get("fees_inr") or 0.0) for t in trades),
        "sum_interest_inr": sum((t.get("interest_inr") or 0.0) for t in trades),
        "wins": sum(1 for t in trades if (t.get("realized_pnl_inr") or 0.0) > 0),
        "losses": sum(1 for t in trades if (t.get("realized_pnl_inr") or 0.0) < 0),
    }
    summary["gross_PF"] = _profit_factor(
        [t.get("realized_pnl_inr") for t in trades if t.get("realized_pnl_inr") is not None]
    )
    # Cohort split (baseline vs variant_b)
    variant_b_trades = [t for t in trades if t.get("paper_variant_b") is True]
    baseline_only_trades = [t for t in trades if t.get("paper_variant_b") is False]
    summary["cohort_split"] = {
        "variant_b_true_n": len(variant_b_trades),
        "variant_b_true_PF": _profit_factor(
            [t.get("realized_pnl_inr") for t in variant_b_trades if t.get("realized_pnl_inr") is not None]
        ),
        "variant_b_true_sum": sum((t.get("realized_pnl_inr") or 0.0) for t in variant_b_trades),
        "baseline_only_n": len(baseline_only_trades),
        "baseline_only_PF": _profit_factor(
            [t.get("realized_pnl_inr") for t in baseline_only_trades if t.get("realized_pnl_inr") is not None]
        ),
        "baseline_only_sum": sum((t.get("realized_pnl_inr") or 0.0) for t in baseline_only_trades),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _profit_factor(pnls: List[float]) -> Optional[float]:
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses <= 0:
        return float("inf") if wins > 0 else None
    return round(wins / losses, 4)


def _write_per_day_csv(per_day_rows: List[dict], out_dir: Path) -> Path:
    import csv
    p = out_dir / "per_day.csv"
    if not per_day_rows:
        with open(p, "w", encoding="utf-8") as f:
            f.write("# no per-day rows recorded\n")
        return p
    cols = list(per_day_rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(per_day_rows)
    return p


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OCI-style overnight backtest driver")
    ap.add_argument("--from", dest="from_date", default=None,
                    help=f"YYYY-MM-DD start (default {START_DATE})")
    ap.add_argument("--to", dest="to_date", default=None,
                    help=f"YYYY-MM-DD end inclusive (default {END_DATE})")
    ap.add_argument("--no-clear-state", action="store_true",
                    help="Do NOT delete prior state files (continue from existing state)")
    ap.add_argument("--max-workers", type=int, default=1,
                    help="Ignored (overnight is sequential by state-machine design). "
                         "Retained for engine.py parity.")
    return ap.parse_args()


def run() -> int:
    args = _parse_args()
    if not MAIN_PATH.exists():
        print(f"ERROR: main.py not found at {MAIN_PATH}", file=sys.stderr)
        return 2

    start_s = args.from_date or START_DATE
    end_s   = args.to_date   or END_DATE
    try:
        start = date.fromisoformat(start_s)
        end   = date.fromisoformat(end_s)
    except Exception:
        print("ERROR: START/END must be YYYY-MM-DD", file=sys.stderr)
        return 2
    if end < start:
        print("ERROR: --to must be >= --from", file=sys.stderr)
        return 2

    # Per-run output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / "engine_overnight" / f"run_{ts}"
    log_dir = out_dir / "daily_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Run output dir: {out_dir}")

    # State files — read configuration.json (setups source-of-truth), NOT base_config.json
    cfg = _load_setups_config()
    state_paths = _state_files_from_config(cfg)
    print(f"[+] State files governed by config: {[str(p.relative_to(ROOT)) for p in state_paths]}")
    if not args.no_clear_state:
        _clear_state_files(state_paths)
    else:
        print("[+] --no-clear-state: existing state files preserved")

    # Build trading-day list
    days_all = list(_daterange(start, end))
    days = [d for d in days_all if is_trading_day(d)]
    skipped = [d for d in days_all if d not in days]
    for d in skipped:
        print(f"[skip] {d} (non-trading day)")
    if not days:
        print("No trading days in the requested range.")
        return 0

    print(f"\n=== Sequential overnight DRY RUN ({len(days)} trading day(s)) ===")
    print(f"    Each day: verify-exit (09:30) -> entry (15:25)")
    print(f"    After loop: final verify-exit on day after {days[-1]} to settle last day's positions\n")

    per_day_rows: List[dict] = []

    # Loop: sequential — state file is shared mutable input across days
    for i, d in enumerate(days, start=1):
        # 1. verify-exit (settles AMO fills from prior trading day's entries,
        #    releases T+2 cash settles from two days ago)
        rc1, sum1, log1 = _run_action(d, "verify-exit", log_dir)
        # 2. entry (places new MOC BUY + AMO SELL for tomorrow)
        rc2, sum2, log2 = _run_action(d, "entry", log_dir)
        row = {
            "day": d.isoformat(),
            "verify_exit_rc": rc1, "verify_exit_summary": sum1,
            "entry_rc": rc2, "entry_summary": sum2,
        }
        per_day_rows.append(row)
        status = "OK" if rc1 == 0 and rc2 == 0 else "FAIL"
        print(f"[{i:3d}/{len(days)}] [{status}] {d} | verify-exit: {sum1 or 'no fires'} | entry: {sum2 or 'no fires'}")
        if rc1 != 0 or rc2 != 0:
            print(f"           failing rcs: verify-exit={rc1}, entry={rc2}; see {log_dir}")

    # Final settlement pass: positions opened on END_DATE need to be settled on
    # the NEXT trading day. Without this, the slot state file leaves them in
    # t0_open and they're invisible to the aggregator.
    final_d = _next_trading_day(days[-1])
    print(f"\n[+] Final settle pass: verify-exit on {final_d} (closes any t0_open from {days[-1]})")
    rc_final, sum_final, log_final = _run_action(final_d, "verify-exit", log_dir)
    per_day_rows.append({
        "day": final_d.isoformat(),
        "verify_exit_rc": rc_final, "verify_exit_summary": sum_final,
        "entry_rc": None, "entry_summary": "<skipped — final settle only>",
    })
    final_status = "OK" if rc_final == 0 else "FAIL"
    print(f"     [{final_status}] {final_d} | verify-exit: {sum_final or 'no fires'}")

    # Aggregate
    per_day_csv = _write_per_day_csv(per_day_rows, out_dir)
    summary = _aggregate_trades(state_paths, out_dir)

    print(f"\n=== Summary ===")
    print(f"  trades.csv:   {(out_dir / 'trades.csv').relative_to(ROOT)}")
    print(f"  summary.json: {(out_dir / 'summary.json').relative_to(ROOT)}")
    print(f"  per_day.csv:  {per_day_csv.relative_to(ROOT)}")
    print(f"")
    print(f"  N trades:        {summary['n_trades']}")
    print(f"  Sum realized:    Rs {summary['sum_realized_pnl_inr']:+,.2f}")
    print(f"  Wins / Losses:   {summary['wins']} / {summary['losses']}")
    pf = summary['gross_PF']
    print(f"  Gross PF:        {pf if pf is not None else 'n/a'}")
    cs = summary["cohort_split"]
    print(f"  Variant B cohort: n={cs['variant_b_true_n']} PF={cs['variant_b_true_PF']} sum={cs['variant_b_true_sum']:+,.2f}")
    print(f"  Baseline cohort:  n={cs['baseline_only_n']} PF={cs['baseline_only_PF']} sum={cs['baseline_only_sum']:+,.2f}")

    # Non-zero rc if any subprocess failed
    failed = any(r.get("verify_exit_rc") not in (None, 0) or r.get("entry_rc") not in (None, 0)
                 for r in per_day_rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
