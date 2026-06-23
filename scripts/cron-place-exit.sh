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

# Mode: default = paper (Upstox data + simulated orders). Set LIVE=1 in the
# crontab line to trade live via Kite. (Empty MODE_FLAGS="" is NOT live —
# `${VAR:-default}` treats empty as unset and falls back to paper.)
if [[ "${LIVE:-0}" == "1" ]]; then
    MODE_FLAGS=""
else
    MODE_FLAGS="--paper-trading --data-source upstox --session-date $(date +%F)"
fi

LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_place_exit_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode overnight --action place-exit $MODE_FLAGS >> "$LOG_FILE" 2>&1
