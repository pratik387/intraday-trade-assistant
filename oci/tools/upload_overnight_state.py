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

Uses the OCI Python SDK directly (no `oci` CLI dependency, which isn't
installed in the engine's venv). Pattern mirrors the SDK usage in
oci/tools/upload_trading_session.py — import oci_sdk BEFORE adding the
project root to sys.path so the local `oci/` package doesn't shadow the
SDK.

Usage:
    python oci/tools/upload_overnight_state.py                # today
    python oci/tools/upload_overnight_state.py --date 2026-06-05
    python oci/tools/upload_overnight_state.py --dry-run

Default bucket: `paper-trading-logs` (shared with intraday paper sessions).
Override with --bucket.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Import OCI SDK BEFORE adding project root to path
# (to avoid local oci/ folder shadowing the package).
try:
    import oci as oci_sdk
    HAS_OCI = True
except ImportError:
    oci_sdk = None
    HAS_OCI = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_BUCKET = "paper-trading-logs"
_SETUP_NAME = "close_dn_overnight_long"


def _collect_files(session_date: str) -> List[Tuple[Path, str]]:
    """Return (local_path, target_object_name) pairs for the session.

    Skips files that don't exist on disk — partial uploads are fine
    (e.g. logs might not be flushed yet, baseline rebuild might have
    failed). The dashboard's historical view will surface what's
    actually there.

    Backfill semantics: state files (overnight_slots.json,
    decay_tripwire.json) are MUTABLE on disk — they reflect TODAY's
    state, not past EOD state. When uploading a past date:
      - overnight_slots.json is SKIPPED. Today's pool snapshot is not
        a faithful representation of EOD on a past date.
      - decay_tripwire.json IS uploaded. Each ledger entry has ts_iso,
        so the dashboard's historical reader filters by date and
        reconstructs the correct as-of-EOD cumulative view from any
        snapshot containing all relevant trades.
    """
    is_today = session_date == _date.today().isoformat()

    sources: List[Tuple[Path, str]] = []
    if is_today:
        # Today's slot pool is the actual EOD-now snapshot.
        sources.append((
            _REPO_ROOT / "state" / "overnight_slots.json",
            "overnight_slots.json",
        ))
    sources.extend([
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
    ])
    return [(src, name) for src, name in sources if src.exists()]


def _upload_one(client, namespace: str, bucket: str, local: Path,
                object_name: str) -> bool:
    size_kb = local.stat().st_size / 1024
    print(f"  {object_name:<22} ({size_kb:>7.1f} KB) ", end="", flush=True)
    try:
        with open(local, "rb") as f:
            client.put_object(
                namespace_name=namespace,
                bucket_name=bucket,
                object_name=object_name,
                put_object_body=f,
            )
        print("OK")
        return True
    except Exception as e:
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "..."
        print(f"FAIL: {msg}")
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

    if not HAS_OCI:
        print("ERROR: OCI SDK not installed. Run: pip install oci", file=sys.stderr)
        return 1

    try:
        config = oci_sdk.config.from_file()
        client = oci_sdk.object_storage.ObjectStorageClient(config)
        namespace = client.get_namespace().data
    except Exception as e:
        print(f"ERROR: failed to initialize OCI client: {e}", file=sys.stderr)
        return 1

    ok = 0
    fail = 0
    for local, name in items:
        full_name = f"{prefix}/{name}"
        if _upload_one(client, namespace, args.bucket, local, full_name):
            ok += 1
        else:
            fail += 1
    print()
    print(f"Done: {ok} uploaded, {fail} failed -> {args.bucket}/{prefix}/")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
