#!/usr/bin/env python3
"""Upload local NSE F&O OI parquet store to OCI Object Storage.

Mirrors the pattern of upload_monthly_cache.py / upload_regime_caches.py.
Reads `data/option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet` (the local
backfill produced by tools/option_chain/fetch_oi_snapshot.py) and uploads
each file to the OCI cache bucket under the same key:
    option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet

Consumed by oci/docker/entrypoint.py::download_option_chain() at OCI
pod startup. Required by structures/expiry_pin_strike_reversal_structure
(via services/option_chain_loader.find_max_oi_strike).

Usage:
    python oci/tools/upload_option_chain.py                      # Upload all
    python oci/tools/upload_option_chain.py --year 2024          # Year filter
    python oci/tools/upload_option_chain.py --year 2024 --month 06   # Month
    python oci/tools/upload_option_chain.py --skip-existing      # Skip already-uploaded
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


_OCI_CLI = str(Path(sys.executable).parent / "oci")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OI_DIR = _REPO_ROOT / "data" / "option_chain"
_BUCKET = "backtest-cache"
_BUCKET_PREFIX = "option_chain"


def _list_local_parquets(year: str | None, month: str | None) -> List[Tuple[Path, str]]:
    """Return list of (local_path, object_name) tuples to upload."""
    if not _OI_DIR.exists():
        print(f"ERROR: local OI store not found: {_OI_DIR}", file=sys.stderr)
        print(
            "       Run: python tools/option_chain/fetch_oi_snapshot.py "
            "--start 2023-01-02 --end <today>",
            file=sys.stderr,
        )
        sys.exit(1)

    out: List[Tuple[Path, str]] = []
    for parquet in sorted(_OI_DIR.rglob("*.parquet")):
        # Path is data/option_chain/<YYYY>/<MM>/<file>.parquet
        rel = parquet.relative_to(_OI_DIR)
        parts = rel.parts
        if len(parts) != 3:
            print(f"  (skipping unexpected path: {parquet})", file=sys.stderr)
            continue
        y, m, fname = parts
        if year is not None and y != year:
            continue
        if month is not None and m != month:
            continue
        object_name = f"{_BUCKET_PREFIX}/{y}/{m}/{fname}"
        out.append((parquet, object_name))
    return out


def _list_remote_object_names() -> set[str]:
    """List existing object names under the option_chain prefix in the bucket."""
    try:
        result = subprocess.run(
            [
                _OCI_CLI, "os", "object", "list",
                "--bucket-name", _BUCKET,
                "--prefix", f"{_BUCKET_PREFIX}/",
                "--all",
                "--query", "data[*].name",
                "--raw-output",
            ],
            capture_output=True, check=True, text=True,
        )
        # Output is a JSON array printed by --raw-output (one element per line in
        # the OCI CLI default). Try JSON load first, fall back to line-split.
        import json
        try:
            return set(json.loads(result.stdout))
        except Exception:
            return {ln.strip().strip('",') for ln in result.stdout.splitlines() if ln.strip()}
    except subprocess.CalledProcessError:
        # Bucket prefix might not exist yet — treat as empty
        return set()


def _upload_one(local: Path, object_name: str) -> bool:
    size_mb = local.stat().st_size / (1024 ** 2)
    print(f"  Uploading {object_name} ({size_mb:.2f} MB)...", end=" ", flush=True)
    try:
        subprocess.run(
            [
                _OCI_CLI, "os", "object", "put",
                "--bucket-name", _BUCKET,
                "--name", object_name,
                "--file", str(local),
                "--force",
            ],
            capture_output=True, check=True,
        )
        print("OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {e.stderr.decode() if e.stderr else e}")
        return False


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=str, default=None,
                        help="Filter to one YYYY (e.g., 2024)")
    parser.add_argument("--month", type=str, default=None,
                        help="Filter to one MM (zero-padded, e.g., 06). Requires --year.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files already present in the bucket")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be uploaded without doing it")
    args = parser.parse_args(argv)

    if args.month is not None and args.year is None:
        parser.error("--month requires --year")

    items = _list_local_parquets(args.year, args.month)
    print(f"Found {len(items)} local OI parquets in {_OI_DIR}")
    if not items:
        return 0

    if args.skip_existing:
        print("Listing existing bucket objects...")
        existing = _list_remote_object_names()
        items = [(l, o) for (l, o) in items if o not in existing]
        print(f"  -> {len(items)} after skip-existing filter")

    if args.dry_run:
        for _, obj in items[:25]:
            print(f"  WOULD UPLOAD: {obj}")
        if len(items) > 25:
            print(f"  ... and {len(items) - 25} more")
        return 0

    ok = 0
    fail = 0
    total_bytes = 0
    for local, obj in items:
        if _upload_one(local, obj):
            ok += 1
            total_bytes += local.stat().st_size
        else:
            fail += 1
    print()
    print(f"Done: {ok} uploaded, {fail} failed, "
          f"{total_bytes / (1024**2):.1f} MB total to bucket={_BUCKET}/{_BUCKET_PREFIX}/")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
