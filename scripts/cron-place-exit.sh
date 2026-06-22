#!/bin/bash
# Triggered at 16:05 IST every weekday by cron (after the 16:00 AMO window opens).
# Places the exit AMO SELL + GTT catastrophe stop for each overnight position
# opened by today's 15:26 entry run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then PYTHON_BIN=".venv/bin/python"; else PYTHON_BIN=".venv/Scripts/python"; fi
fi

# Default = paper (Upstox data + simulated orders). For LIVE, set MODE_FLAGS=""
# in the crontab line so main.py builds the Kite hybrid broker.
MODE_FLAGS="${MODE_FLAGS:---paper-trading --data-source upstox --session-date $(date +%F)}"

LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_place_exit_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode overnight --action place-exit $MODE_FLAGS >> "$LOG_FILE" 2>&1
