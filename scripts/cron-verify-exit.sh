#!/bin/bash
# Triggered at 09:30 IST every weekday by cron.
# Verifies yesterday's overnight AMO fills, releases settled slots.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Auto-detect venv layout (Linux=.venv/bin, Windows=.venv/Scripts).
# Override via PYTHON_BIN env var if your venv lives elsewhere.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON_BIN=".venv/bin/python"
    else
        PYTHON_BIN=".venv/Scripts/python"
    fi
fi

# Mode: default = paper trading via Upstox data. Set LIVE=1 in the crontab
# line to trade live via Kite. (Empty MODE_FLAGS="" is NOT live — `${VAR:-default}`
# treats empty as unset and falls back to paper. See cron-entry.sh.)
if [[ "${LIVE:-0}" == "1" ]]; then
    MODE_FLAGS=""
else
    MODE_FLAGS="--paper-trading --data-source upstox --session-date $(date +%F)"
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_verify_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode overnight --action verify-exit $MODE_FLAGS >> "$LOG_FILE" 2>&1
