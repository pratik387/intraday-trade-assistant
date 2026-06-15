#!/bin/bash
# Triggered at ~15:28 IST every weekday by cron — the PRE-CLOSE EXIT PASS.
# Squares off multi-day CNC/MTF positions whose K-day hold is due today
# (run_eod phase=exits): sell at ~the 15:30 close, matching the backtest's
# "exit at the K-day close".
#
# MUST run BEFORE the 15:30 close — a close exit cannot be placed after the
# market shuts. This is intentionally a SEPARATE job from the entry pass
# (cron-multiday-entry.sh @ ~15:35), which needs the COMPLETE post-close day-T
# bar for its signal. One combined run cannot satisfy both (exit before close
# vs signal after close) — splitting them is what makes backtest = paper = live.
#
# Mirrors scripts/cron-entry.sh paper wiring; only --mode/--action differ.
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
LOG_FILE="$LOG_DIR/multiday_exit_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode multi_day --action exit $MODE_FLAGS >> "$LOG_FILE" 2>&1
