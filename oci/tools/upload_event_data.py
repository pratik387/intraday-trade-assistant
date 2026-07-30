#!/usr/bin/env python3
"""Upload the earnings-calendar and ASM/GSM surveillance parquets to OCI.

Both are single-file parquets consumed by the SAME setup
(`earnings_downshock_continuation_short`), so they share one script rather
than following the strict one-script-per-dataset convention of
upload_delivery_pct.py / upload_cross_day_rvol.py. Same mechanics as those:
OCI CLI subprocess from a scratch cwd, --skip-existing, --dry-run.

    data/earnings_calendar/earnings_events.parquet  -> earnings_calendar/...
    data/asm_gsm_history/asm_gsm_events.parquet     -> asm_gsm_history/...

Consumed by oci/docker/entrypoint.py::download_earnings_calendar() and
::download_asm_gsm_history() at pod startup, then read by
services/earnings_reaction_enrichment.py and services/surveillance_lookup.py.

WHY THIS EXISTS: the OCI entrypoint's earnings download was DELETED when
`earnings_day_intraday_fade` was retired (2026-05-14), the same way
download_iv_rank() went with options_vol_iv_rank_revert. The ASM/GSM parquet
never had one — it was first materialised 2026-07-28. Without both, an OCI
run of earnings_downshock_continuation_short produces a silently EMPTY
universe and zero fires (the below_vwap trap, lesson #16).

Rebuild locally before uploading:
    python tools/earnings_calendar/fetch_earnings.py --start <d> --end <d>
    python tools/asm_gsm_history/fetch_asm_gsm.py --start <d> --end <d>

Usage:
    python oci/tools/upload_event_data.py                      # both
    python oci/tools/upload_event_data.py --dataset earnings
    python oci/tools/upload_event_data.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

_OCI_CLI = str(Path(sys.executable).parent / "oci")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUCKET = "backtest-cache"

# dataset -> (local path, bucket prefix, rebuild hint)
_DATASETS: Dict[str, Tuple[Path, str, str]] = {
    "earnings": (
        _REPO_ROOT / "data" / "earnings_calendar" / "earnings_events.parquet",
        "earnings_calendar",
        "python tools/earnings_calendar/fetch_earnings.py --start 2023-01-01 --end <today>",
    ),
    "asm_gsm": (
        _REPO_ROOT / "data" / "asm_gsm_history" / "asm_gsm_events.parquet",
        "asm_gsm_history",
        "python tools/asm_gsm_history/fetch_asm_gsm.py --start 2023-01-01 --end <today>",
    ),
}

# Run the OCI CLI from a scratch cwd so this project's local `oci/` package
# does not shadow the OCI SDK during the CLI's bootstrap import.
_SCRATCH_CWD = str(Path.home())


def _collect(which: List[str]) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    missing = False
    for name in which:
        local, prefix, hint = _DATASETS[name]
        if not local.exists():
            print(f"ERROR: {name} parquet not found: {local}", file=sys.stderr)
            print(f"       Rebuild with: {hint}", file=sys.stderr)
            missing = True
            continue
        out.append((local, f"{prefix}/{local.name}"))
    if missing:
        sys.exit(1)
    return out


def _list_remote(prefix: str) -> set:
    try:
        res = subprocess.run(
            [_OCI_CLI, "os", "object", "list", "--bucket-name", _BUCKET,
             "--prefix", f"{prefix}/", "--all",
             "--query", "data[*].name", "--raw-output"],
            capture_output=True, check=True, text=True, cwd=_SCRATCH_CWD,
        )
        try:
            return set(json.loads(res.stdout))
        except Exception:
            return {ln.strip().strip('",') for ln in res.stdout.splitlines() if ln.strip()}
    except subprocess.CalledProcessError:
        return set()


def _upload_one(local: Path, object_name: str) -> bool:
    size_mb = local.stat().st_size / (1024 * 1024)
    print(f"  Uploading {object_name} ({size_mb:.1f} MB)...", end=" ", flush=True)
    try:
        subprocess.run(
            [_OCI_CLI, "os", "object", "put", "--bucket-name", _BUCKET,
             "--name", object_name, "--file", str(local), "--force"],
            capture_output=True, check=True, cwd=_SCRATCH_CWD,
        )
        print("OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {e.stderr.decode() if e.stderr else e}")
        return False


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["earnings", "asm_gsm", "all"], default="all")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    which = list(_DATASETS) if args.dataset == "all" else [args.dataset]
    items = _collect(which)
    print(f"Found {len(items)} parquet(s) to upload")

    if args.skip_existing:
        keep = []
        for local, obj in items:
            if obj in _list_remote(obj.split("/")[0]):
                print(f"  SKIP (exists): {obj}")
            else:
                keep.append((local, obj))
        items = keep

    if args.dry_run:
        for _, obj in items:
            print(f"  WOULD UPLOAD: {obj}")
        return 0

    ok = fail = 0
    for local, obj in items:
        if _upload_one(local, obj):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} uploaded, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
