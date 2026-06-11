#!/bin/bash
# Triggered at 15:35 IST every weekday by cron.
# Uploads the latest paper-trading session folder (logs/paper_YYYYMMDD_*)
# to OCI Object Storage so the dashboard's Historical view + run-analyzer
# have today's data once main.py has shut down.
#
# Runs AFTER:
#   - 15:30 market close (paper main.py session has finalized its writes)
#   - 15:26 overnight entry cron (orders placed)
#
# 15:35 gives a 5-min buffer for main.py to flush + close out the session.
# The 16:00 overnight archive cron runs separately on the state/ files —
# it's not coupled to this session-log upload.
#
# The upload script auto-discovers the most recent paper_*/live_* folder
# under logs/, so no session id needs to be passed. The --prefix flag
# matches the dashboard's instance name (fixed -> oci://paper-trading-logs/fixed/<session>/).
#
# Failure of this cron does NOT affect trading — it's archival only.

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
LOG_FILE="$LOG_DIR/trading_session_upload_$(date +%Y-%m-%d).log"

# Default to "fixed" prefix (the dashboard's paper instance). Override
# with PREFIX env var if a different instance is added later.
PREFIX="${PREFIX:-fixed}"

"$PYTHON_BIN" oci/tools/upload_trading_session.py --prefix "$PREFIX" >> "$LOG_FILE" 2>&1
