#!/bin/bash
# Daily 07:30 IST weekdays: download NSE's price-band lists (sec_list_DDMMYYYY.csv)
# for the last few days BEFORE the daemons start. The file dated D holds the bands
# for the next trading day; up_spike_fade_short sizes by distance to the circuit
# and the daemon refuses to start without the session's file (validate_price_bands).
# The file is published the evening before, so a failure here shows up at startup.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then PYTHON_BIN=".venv/bin/python"; else PYTHON_BIN=".venv/Scripts/python"; fi
fi
LOG_FILE="logs/price_bands_refresh_$(date +%F).log"
mkdir -p logs
"$PYTHON_BIN" tools/fetch_nse_price_bands.py >> "$LOG_FILE" 2>&1 || echo "[price bands] FAILED (daemon startup will refuse if the session's file is missing)" >> "$LOG_FILE"
echo "[price bands] done -> $LOG_FILE"
