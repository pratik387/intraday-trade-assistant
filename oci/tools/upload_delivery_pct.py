#!/usr/bin/env python3
"""Upload local NSE delivery-percentage parquet to OCI Object Storage.

Mirrors the pattern of upload_iv_rank.py / upload_option_chain.py.
Uploads `data/delivery_pct/delivery_history.parquet` (built by
tools/delivery_pct/fetch_delivery.py) to the OCI cache bucket under:
    delivery_pct/delivery_history.parquet

Consumed by oci/docker/entrypoint.py::download_delivery_pct() at OCI pod
startup. Required by structures/delivery_pct_anomaly_short_structure.py
(via services/delivery_pct_enrichment.py which adds `delivery_pct` to
each symbol's daily_df during the screener daily-cache seed).

Single file (~39 MB), no per-month sharding — one upload covers the
whole 2023-2026 backfill plus daily top-ups via the same
fetch_delivery.py CLI (idempotent).

Usage:
    python oci/tools/upload_delivery_pct.py                  # Upload
    python oci/tools/upload_delivery_pct.py --skip-existing
    python oci/tools/upload_delivery_pct.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


_OCI_CLI = str(Path(sys.executable).parent / "oci")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DELIVERY_DIR = _REPO_ROOT / "data" / "delivery_pct"
_BUCKET = "backtest-cache"
_BUCKET_PREFIX = "delivery_pct"


def _list_local_parquets() -> List[Tuple[Path, str]]:
    if not _DELIVERY_DIR.exists():
        print(f"ERROR: local delivery_pct store not found: {_DELIVERY_DIR}", file=sys.stderr)
        print(
            "       Run: python -m tools.delivery_pct.fetch_delivery "
            "--start 2023-01-01 --end 2026-04-30 "
            "--out data/delivery_pct/delivery_history.parquet --workers 8",
            file=sys.stderr,
        )
        sys.exit(1)

    out: List[Tuple[Path, str]] = []
    for parquet in sorted(_DELIVERY_DIR.glob("*.parquet")):
        object_name = f"{_BUCKET_PREFIX}/{parquet.name}"
        out.append((parquet, object_name))
    return out


# Run OCI CLI subprocess from a scratch cwd so the project's local
# `oci/` directory (this very package) doesn't shadow the OCI SDK on
# `from oci import fips` inside the CLI binary's bootstrap.
_SCRATCH_CWD = str(Path.home())


def _list_remote_object_names() -> set:
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
            capture_output=True, check=True, text=True, cwd=_SCRATCH_CWD,
        )
        import json
        try:
            return set(json.loads(result.stdout))
        except Exception:
            return {ln.strip().strip('",') for ln in result.stdout.splitlines() if ln.strip()}
    except subprocess.CalledProcessError:
        return set()


def _upload_one(local: Path, object_name: str) -> bool:
    size_mb = local.stat().st_size / (1024 * 1024)
    print(f"  Uploading {object_name} ({size_mb:.1f} MB)...", end=" ", flush=True)
    try:
        subprocess.run(
            [
                _OCI_CLI, "os", "object", "put",
                "--bucket-name", _BUCKET,
                "--name", object_name,
                "--file", str(local),
                "--force",
            ],
            capture_output=True, check=True, cwd=_SCRATCH_CWD,
        )
        print("OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {e.stderr.decode() if e.stderr else e}")
        return False


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files already present in the bucket")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be uploaded without doing it")
    args = parser.parse_args(argv)

    items = _list_local_parquets()
    print(f"Found {len(items)} local delivery_pct parquets in {_DELIVERY_DIR}")
    if not items:
        return 0

    if args.skip_existing:
        print("Listing existing bucket objects...")
        existing = _list_remote_object_names()
        items = [(l, o) for (l, o) in items if o not in existing]
        print(f"  -> {len(items)} after skip-existing filter")

    if args.dry_run:
        for _, obj in items:
            print(f"  WOULD UPLOAD: {obj}")
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
          f"{total_bytes / (1024 * 1024):.1f} MB total to bucket={_BUCKET}/{_BUCKET_PREFIX}/")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
