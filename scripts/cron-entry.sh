#!/bin/bash
# Triggered at 15:25 IST every weekday by cron.
# Runs the overnight setup entry handler: computes signal, places BUY, places AMO SELL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Default to the local venv; override via PYTHON_BIN env var if needed.
PYTHON_BIN="${PYTHON_BIN:-.venv/Scripts/python}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_entry_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode overnight --action entry >> "$LOG_FILE" 2>&1
