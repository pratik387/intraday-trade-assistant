#!/bin/bash
# Triggered at ~16:05 IST every weekday by cron — the POST-CLOSE ENTRY PASS.
# Runs the multi-day CNC/MTF capitulation entry leg only (run_eod phase=entries):
# ranks the MTF universe on day-T's COMPLETE daily bar and places AMO BUYs that
# the broker submits at the next session's pre-open (09:00) -> fill at T+1 open
# (mtf_capitulation / low52 / zscore / crash2d).
#
# Timing is constrained on BOTH sides:
#   - AFTER 15:30 close: the signal needs day-T's full close + full-day volume
#     (turnover-shock filter), only final once the session closes. The day-T bar
#     is synthesized from the complete 5m (get_daily drops today's partial bar) —
#     same 5m pipeline clean_daily_from5m.feather was built from, so the live
#     signal reproduces the backtest's.
#   - AFTER 16:00: Zerodha's NSE-equity AMO window is 16:00-08:58 IST. An AMO
#     placed in the 15:30-16:00 dead zone (or during market hours) is REJECTED
#     by the broker -> zero entries. So this MUST run >= 16:00.
#     (src: support.zerodha.com .../order/articles/auto-amo)
# The EXITS are a SEPARATE pre-close job (cron-multiday-exit.sh @ ~15:28) — a
# close-exit can't be placed after the market shuts.
#
# Mirrors scripts/cron-entry.sh paper wiring (MockBroker + UpstoxDataClient via
# --paper-trading --data-source upstox); only --mode/--action differ.
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
LOG_FILE="$LOG_DIR/multiday_entry_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode multi_day --action entry $MODE_FLAGS >> "$LOG_FILE" 2>&1
