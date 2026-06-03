#!/usr/bin/env python3
"""Upload the local MIS-short eligibility map to OCI Object Storage.

Mirrors the pattern of upload_delivery_pct.py / upload_cross_day_rvol.py.
Uploads `data/mis_short_eligibility/*.json` (built by
jobs/refresh_mis_short_eligibility.py — a daily Kite order_margins leverage
sweep) to the OCI cache bucket under:
    mis_short_eligibility/<name>.json

Consumed by oci/docker/entrypoint.py::download_mis_short_eligibility() at OCI
pod startup. Required by services/setup_universe.py::up_spike_fade_short_universe
(admits only symbols with broker intraday SELL/MIS leverage > 1). Without it the
universe fail-closes (require_short_eligibility_map=true) and up_spike_fade_short
produces ZERO fires.

Small file (~30 KB). Re-upload whenever the map is refreshed:
    KITE_API_KEY=... python jobs/refresh_mis_short_eligibility.py
    python oci/tools/upload_mis_short_eligibility.py

Usage:
    python oci/tools/upload_mis_short_eligibility.py            # Upload
    python oci/tools/upload_mis_short_eligibility.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


_OCI_CLI = str(Path(sys.executable).parent / "oci")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ELIG_DIR = _REPO_ROOT / "data" / "mis_short_eligibility"
_BUCKET = "backtest-cache"
_BUCKET_PREFIX = "mis_short_eligibility"

# Run OCI CLI from a scratch cwd so the project's local `oci/` package doesn't
# shadow the OCI SDK on `from oci import fips` in the CLI bootstrap.
_SCRATCH_CWD = str(Path.home())


def _list_local_maps() -> List[Tuple[Path, str]]:
    if not _ELIG_DIR.exists():
        print(f"ERROR: local MIS-short eligibility store not found: {_ELIG_DIR}", file=sys.stderr)
        print(
            "       Run: KITE_API_KEY=<key> python jobs/refresh_mis_short_eligibility.py",
            file=sys.stderr,
        )
        sys.exit(1)
    out: List[Tuple[Path, str]] = []
    for f in sorted(_ELIG_DIR.glob("*.json")):
        out.append((f, f"{_BUCKET_PREFIX}/{f.name}"))
    if not out:
        print(f"ERROR: no *.json maps in {_ELIG_DIR} — run the refresher first.", file=sys.stderr)
        sys.exit(1)
    return out


def _upload_one(local: Path, object_name: str) -> bool:
    size_kb = local.stat().st_size / 1024
    print(f"  Uploading {object_name} ({size_kb:.1f} KB)...", end=" ", flush=True)
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
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be uploaded without doing it")
    args = parser.parse_args(argv)

    items = _list_local_maps()
    print(f"Found {len(items)} local MIS-short eligibility map(s) in {_ELIG_DIR}")

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
    print(f"\nDone: {ok} uploaded, {fail} failed to bucket={_BUCKET}/{_BUCKET_PREFIX}/")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
