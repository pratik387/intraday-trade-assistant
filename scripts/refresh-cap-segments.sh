#!/bin/bash
# Weekly cron — refresh NSE cap-segment classification from niftyindices.com.
# Schedule example (Sunday 09:00 IST, well ahead of Monday market open):
#   0 9 * * 0 cd /home/ubuntu/intraday_fixed/intraday-trade-assistant && ./scripts/refresh-cap-segments.sh
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
LOG_FILE="$LOG_DIR/refresh_cap_segments_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" scripts/refresh_cap_segments.py >> "$LOG_FILE" 2>&1
