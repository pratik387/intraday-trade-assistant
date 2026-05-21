#!/bin/bash
# Triggered at 09:30 IST every weekday by cron.
# Verifies yesterday's overnight AMO fills, releases settled slots.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/Scripts/python}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_verify_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode overnight --action verify-exit >> "$LOG_FILE" 2>&1
