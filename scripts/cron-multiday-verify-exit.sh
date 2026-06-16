#!/bin/bash
# Triggered at ~09:30 IST every weekday by cron.
# Confirms the multi-day CNC/MTF AMO BUYs placed by cron-multiday-entry.sh filled
# at today's open and records the entry price (run_verify_entries). Unfilled AMOs
# are dropped (paper) / failsafe-bought (live).
#
# Mirrors scripts/cron-verify-exit.sh (close_dn_overnight) — same paper wiring,
# only --mode differs (multi_day vs overnight).
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

# Default flags = paper trading via Upstox data. Override with MODE_FLAGS=""
# (live + Kite) or any other combination from the crontab line.
# --session-date is REQUIRED for MockBroker (paper mode); see cron-entry.sh.
MODE_FLAGS="${MODE_FLAGS:---paper-trading --data-source upstox --session-date $(date +%F)}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/multiday_verify_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode multi_day --action verify-exit $MODE_FLAGS >> "$LOG_FILE" 2>&1
