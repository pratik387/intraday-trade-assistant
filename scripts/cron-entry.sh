#!/bin/bash
# Triggered at 15:26 IST every weekday by cron.
# Runs the overnight setup entry handler: computes signal and places the BUY order only.
# The exit AMO SELL + GTT catastrophe stop are placed separately by cron-place-exit.sh
# at 16:05 IST (after the Zerodha AMO window opens at 16:00).
#
# Why 15:26? The detector's trigger bar is 15:20 (covers 15:20-15:25
# wall-clock, finalized at 15:25:00). The 15:25 bar (which covers
# 15:25-15:30) is NOT used at all by the detector — signal computation
# reads bars 15:00-15:20 only, and the active-window sentinel was moved
# from 15:25 to 15:20 on 2026-06-09 (see _ACTIVE_HHMM comment in
# structures/close_dn_overnight_long_structure.py). Pre-2026-06-09 the
# cron ran at 15:27 to wait for Upstox to surface the 15:25 bar (verified
# 2026-06-01, commit 77a256b); now that the 15:25 bar is no longer
# required as a trigger, 15:26 is safe and buys ~1m30s extra buffer
# before the 15:30 market close — meaningful when fetch wall-clock grows
# on wider universes.
#
# Wall-clock budget (worst case at 15:26 start):
#   15:26:00  cron starts → 5m batch fetch begins (rps=20, conc=30, ~2000 syms)
#   15:27:45  fetch done (~105s)
#   15:27:55  detector loop done (~10s)
#   15:28:05  orders placed (~10s for ~5-10 fires)
#   15:30:00  market close (1m55s buffer)
# See Lesson #23 (cron-driven setups need API-availability buffer).
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

# Mode: default = paper trading via Upstox data. Set LIVE=1 in the crontab
# line to trade live via Kite. (Do NOT revive the old MODE_FLAGS="" convention:
# `${VAR:-default}` treats an empty string as unset, so MODE_FLAGS="" silently
# fell back to PAPER — that bug cost a live session on 2026-06-23.)
# --session-date is REQUIRED for MockBroker (paper mode) — without it,
# broker.get_daily() crashes inside the silent except-Exception in
# _gather_daily_dict and the universe quietly comes back empty.
if [[ "${LIVE:-0}" == "1" ]]; then
    MODE_FLAGS=""
else
    MODE_FLAGS="--paper-trading --data-source upstox --session-date $(date +%F)"
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_entry_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode overnight --action entry $MODE_FLAGS >> "$LOG_FILE" 2>&1
