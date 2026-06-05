#!/bin/bash
# Triggered at 16:00 IST every weekday by cron.
# Uploads today's overnight setup state (slot pool snapshot, decay
# tripwire ledger, baseline + candidates, verify-exit + entry logs)
# to OCI Object Storage for historical reconstruction.
#
# Runs AFTER the 15:27 entry cron has finished (which itself finishes
# in ~13s). 16:00 gives a safe buffer.
#
# Files are uploaded under:
#   <bucket>/overnight/close_dn_overnight_long/<YYYY-MM-DD>/<name>
#
# Default bucket is `paper-trading-logs` (set in upload_overnight_state.py).
# Override with BUCKET env var if needed.
#
# Failure of this cron does NOT affect trading — it's archival only.
# Missing files (e.g. if a cron didn't run) are skipped, not fatal.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Auto-detect venv layout (Linux=.venv/bin, Windows=.venv/Scripts).
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON_BIN=".venv/bin/python"
    else
        PYTHON_BIN=".venv/Scripts/python"
    fi
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_archive_$(date +%Y-%m-%d).log"

EXTRA_ARGS=()
if [[ -n "${BUCKET:-}" ]]; then
    EXTRA_ARGS+=(--bucket "$BUCKET")
fi

"$PYTHON_BIN" oci/tools/upload_overnight_state.py "${EXTRA_ARGS[@]}" >> "$LOG_FILE" 2>&1
