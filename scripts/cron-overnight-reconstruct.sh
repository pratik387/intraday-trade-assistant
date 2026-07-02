#!/bin/bash
# Triggered at 09:45 IST every weekday by cron (after the 09:30 verify-exit).
# Reconstructs the Rs1L idealized PAPER trades for the previous trading day's
# fired signals (traded live AND rejected alike) into the paper ledger, and
# writes the real-vs-idealized slippage report. READ-ONLY vs the broker.
# Idempotent: re-running a day replaces that day's reconstructed entries.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON_BIN=".venv/bin/python"
    else
        PYTHON_BIN=".venv/Scripts/python"
    fi
fi

LOG_FILE="logs/overnight_reconstruct_$(date +%F).log"
mkdir -p logs

"$PYTHON_BIN" tools/overnight_paper_slippage.py --auto >> "$LOG_FILE" 2>&1
echo "[overnight reconstruct] done -> $LOG_FILE"
