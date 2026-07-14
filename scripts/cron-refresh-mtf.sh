#!/bin/bash
# Daily 08:45 IST weekdays: refresh the Zerodha MTF approved list BEFORE the
# trading day (entries read approved_mtf_securities_latest.json). 2026-07-14:
# an 8-week-stale snapshot left 3 held names trading off-list — live that
# risks placement rejections and broker force-handling of MTF positions.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then PYTHON_BIN=".venv/bin/python"; else PYTHON_BIN=".venv/Scripts/python"; fi
fi
LOG_FILE="logs/mtf_refresh_$(date +%F).log"
mkdir -p logs
"$PYTHON_BIN" tools/scrape_zerodha_mtf.py >> "$LOG_FILE" 2>&1 || echo "[mtf refresh] FAILED (kept previous latest.json)" >> "$LOG_FILE"
echo "[mtf refresh] done -> $LOG_FILE"
