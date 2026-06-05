#!/usr/bin/env python3
"""Upload daily snapshot of close_dn_overnight_long state to OCI.

The overnight setup is cron-driven — state lives in JSON files on the VM
that get overwritten each day. To preserve a per-day historical record
(for the dashboard's future historical view and for post-hoc debugging),
this script uploads the day's snapshot to OCI.

Six files are uploaded (all optional — missing files are skipped, not
fatal):
    Source                                              -> Object name
    state/overnight_slots.json                          -> overnight_slots.json
    state/decay_tripwire_close_dn_overnight_long.json   -> decay_tripwire.json
    data/close_dn_baseline/baseline_<date>.json         -> baseline.json
    data/close_dn_baseline/candidates_<date>.json       -> candidates.json
    logs/overnight_verify_<date>.log                    -> verify_exit.log
    logs/overnight_entry_<date>.log                     -> entry.log

Object prefix:
    <bucket>/overnight/close_dn_overnight_long/<date>/<object_name>

Note: `state/overnight_slots.json` and `state/decay_tripwire_*.json` are
MUTABLE — they get rewritten each day. The snapshot captures their state
AT THE TIME THIS SCRIPT RUNS, which is meant to be after the entry cron
has finished (so end-of-day state). For each session_date, the object
represents EOD state on that day.

Usage:
    python oci/tools/upload_overnight_state.py                # today
    python oci/tools/upload_overnight_state.py --date 2026-06-05
    python oci/tools/upload_overnight_state.py --dry-run

Default bucket: `paper-trading-logs` (shared with intraday paper sessions).
Override with --bucket.

Pattern mirrors oci/tools/upload_cross_day_rvol.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date as _date
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


_OCI_CLI = str(Path(sys.executable).parent / "oci")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BUCKET = "paper-trading-logs"
_SETUP_NAME = "close_dn_overnight_long"

# Run OCI CLI from a scratch cwd so the project's local `oci/` package
# doesn't shadow the OCI SDK during CLI bootstrap.
_SCRATCH_CWD = str(Path.home())


def _collect_files(session_date: str) -> List[Tuple[Path, str]]:
    """Return (local_path, target_object_name) pairs for the session.

    Skips files that don't exist on disk — partial uploads are fine
    (e.g. logs might not be flushed yet, baseline rebuild might have
    failed). The dashboard's historical view will surface what's
    actually there.
    """
    sources: List[Tuple[Path, str]] = [
        (
            _REPO_ROOT / "state" / "overnight_slots.json",
            "overnight_slots.json",
        ),
        (
            _REPO_ROOT / "state" / f"decay_tripwire_{_SETUP_NAME}.json",
            "decay_tripwire.json",
        ),
        (
            _REPO_ROOT / "data" / "close_dn_baseline" / f"baseline_{session_date}.json",
            "baseline.json",
        ),
        (
            _REPO_ROOT / "data" / "close_dn_baseline" / f"candidates_{session_date}.json",
            "candidates.json",
        ),
        (
            _REPO_ROOT / "logs" / f"overnight_verify_{session_date}.log",
            "verify_exit.log",
        ),
        (
            _REPO_ROOT / "logs" / f"overnight_entry_{session_date}.log",
            "entry.log",
        ),
    ]
    return [(src, name) for src, name in sources if src.exists()]


def _upload_one(local: Path, object_name: str, bucket: str) -> bool:
    size_kb = local.stat().st_size / 1024
    print(f"  {object_name:<22} ({size_kb:>7.1f} KB) ", end="", flush=True)
    try:
        subprocess.run(
            [
                _OCI_CLI, "os", "object", "put",
                "--bucket-name", bucket,
                "--name", object_name,
                "--file", str(local),
                "--force",
            ],
            capture_output=True, check=True, cwd=_SCRATCH_CWD,
        )
        print("OK")
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        print(f"FAIL: {stderr[:200]}")
        return False


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Session date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=_DEFAULT_BUCKET,
        help=f"OCI bucket (default: {_DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be uploaded without doing it",
    )
    args = parser.parse_args(argv)

    session_date = args.date or _date.today().isoformat()
    try:
        _date.fromisoformat(session_date)
    except ValueError:
        print(f"ERROR: --date must be YYYY-MM-DD (got {session_date!r})", file=sys.stderr)
        return 2

    items = _collect_files(session_date)
    print(f"Overnight state archive for session_date={session_date}")
    print(f"  setup: {_SETUP_NAME}")
    print(f"  bucket: {args.bucket}")
    print(f"  prefix: overnight/{_SETUP_NAME}/{session_date}/")
    print(f"  files found: {len(items)}")
    if not items:
        print("  -> Nothing to upload. Check that the cron has run today.")
        return 0

    prefix = f"overnight/{_SETUP_NAME}/{session_date}"

    if args.dry_run:
        for local, name in items:
            size_kb = local.stat().st_size / 1024
            print(f"  WOULD UPLOAD: {prefix}/{name}  ({size_kb:.1f} KB from {local.relative_to(_REPO_ROOT)})")
        return 0

    ok = 0
    fail = 0
    for local, name in items:
        full_name = f"{prefix}/{name}"
        if _upload_one(local, full_name, args.bucket):
            ok += 1
        else:
            fail += 1
    print()
    print(f"Done: {ok} uploaded, {fail} failed -> {args.bucket}/{prefix}/")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
