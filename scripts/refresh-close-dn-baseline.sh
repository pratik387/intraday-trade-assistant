#!/bin/bash
# Daily cron — refresh close_dn_overnight_long closing-25m volume baseline.
# Schedule example (Mon-Fri 08:00 IST, before 09:15 market open):
#   0 8 * * 1-5 cd /home/ubuntu/intraday_fixed/intraday-trade-assistant && ./scripts/refresh-close-dn-baseline.sh
#
# Output: data/close_dn_baseline/baseline_<date>.json + baseline_latest.json
# Read by: structures/close_dn_overnight_long_structure._load_baseline_snapshot()
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
LOG_FILE="$LOG_DIR/refresh_close_dn_baseline_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" tools/build_close_dn_baseline.py --session-date "$(date +%F)" >> "$LOG_FILE" 2>&1
