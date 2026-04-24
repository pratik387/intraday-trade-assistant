"""Prepare the config override + document the OCI submit command for a
wide-open gauntlet_v2 Discovery+Validation+Holdout capture.

This script does NOT submit the OCI job (task #82/#83 owns that). It writes
the two-flag config override file and prints the submit command the operator
runs manually. Sub-project #5 Step 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="Path to write the OCI config override JSON")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-03-31")
    args = p.parse_args()

    overrides = {
        "wide_open_mode": True,
        "gate_input_logging": {"enabled": True},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(overrides, indent=2))

    print(f"[oci_runner] wrote {out}")
    print(f"[oci_runner] OCI submit (run manually):")
    print(f"  ./oci_submit.sh --config-overrides {out} "
          f"--start-date {args.start_date} --end-date {args.end_date} "
          f"--output-dir cloud_results/gauntlet_v2_discovery/")


if __name__ == "__main__":
    main()
