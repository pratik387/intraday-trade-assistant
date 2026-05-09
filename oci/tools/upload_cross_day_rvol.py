#!/usr/bin/env python3
"""Upload precomputed cross-day RVOL baseline parquet to OCI Object Storage.

Mirrors the pattern of upload_delivery_pct.py / upload_iv_rank.py.
Uploads `data/cross_day_rvol/rvol_baseline.parquet` (built by
tools/cross_day_rvol/build_baseline.py) to:
    backtest-cache/cross_day_rvol/rvol_baseline.parquet

Consumed by oci/docker/entrypoint.py::download_cross_day_rvol() at OCI
pod startup. Required by structures/delivery_pct_anomaly_short_structure.py
(via services/cross_day_rvol_enrichment.py).

Why: the screener caps df_5m at `screener_store_5m_max=120` (~1.5 days),
so the detector cannot compute 20-prior-session same-tod RVOL from df_5m
alone. The static parquet ships precomputed (symbol, date, hhmm) →
rolling-20-day mean volume so the detector divides today's bar volume
by the cached baseline.

Single file (~77 MB), no per-month sharding — one upload covers
2023-01 through 2026-04 backfill plus daily top-ups via the same
build script (idempotent rebuild from monthly 5m feathers).

Usage:
    python oci/tools/upload_cross_day_rvol.py
    python oci/tools/upload_cross_day_rvol.py --skip-existing
    python oci/tools/upload_cross_day_rvol.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


_OCI_CLI = str(Path(sys.executable).parent / "oci")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_DIR = _REPO_ROOT / "data" / "cross_day_rvol"
_BUCKET = "backtest-cache"
_BUCKET_PREFIX = "cross_day_rvol"


def _list_local_parquets() -> List[Tuple[Path, str]]:
    if not _LOCAL_DIR.exists():
        print(f"ERROR: local cross_day_rvol store not found: {_LOCAL_DIR}", file=sys.stderr)
        print(
            "       Run: python tools/cross_day_rvol/build_baseline.py "
            "2023-01-01 2026-04-30",
            file=sys.stderr,
        )
        sys.exit(1)

    out: List[Tuple[Path, str]] = []
    for parquet in sorted(_LOCAL_DIR.glob("*.parquet")):
        object_name = f"{_BUCKET_PREFIX}/{parquet.name}"
        out.append((parquet, object_name))
    return out


# Run OCI CLI from a scratch cwd so the project's local `oci/` package
# doesn't shadow the OCI SDK during CLI bootstrap.
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
    print(f"Found {len(items)} local cross_day_rvol parquets in {_LOCAL_DIR}")
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
