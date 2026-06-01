#!/bin/bash
# Triggered at 15:27 IST every weekday by cron.
# Runs the overnight setup entry handler: computes signal, places BUY, places AMO SELL.
#
# Why 15:27 and not 15:25? The setup's signal bar is 15:25 (covering
# 15:25-15:30). Upstox's V3 intraday 5m endpoint doesn't expose the
# 15:25-timestamped bar until ~2 min after the bar period starts —
# at 15:25:00 the latest available bar is still 15:20. Running at
# 15:27 gives Upstox time to surface the 15:25 bar (verified empirically
# 2026-06-01). Still leaves 3 min before market close at 15:30 for the
# paper BUY (live MOC SELL at close). See Lesson #23.
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
# --session-date is REQUIRED for MockBroker (paper mode) — without it,
# broker.get_daily() crashes inside the silent except-Exception in
# _gather_daily_dict and the universe quietly comes back empty.
MODE_FLAGS="${MODE_FLAGS:---paper-trading --data-source upstox --session-date $(date +%F)}"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_entry_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode overnight --action entry $MODE_FLAGS >> "$LOG_FILE" 2>&1
