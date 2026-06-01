#!/bin/bash
# Diagnose why setups are no-firing in a paper-trade session.
# Usage: bash tools/diagnose_silent_setups.sh logs/paper_<TS>
#
# Reads screening.jsonl (per-symbol per-bar decisions, contains
# per-detector rejection_reason) and aggregates by setup so you can
# see exactly which gate is filtering each one.
set -euo pipefail
DIR="${1:?usage: $0 logs/paper_<TS>}"
JL="$DIR/screening.jsonl"
[[ -f "$JL" ]] || { echo "missing $JL"; exit 1; }

echo "=== Per-setup rejection_reason counts (top 8 per setup) ==="
echo
for SETUP in long_panic_gap_down below_vwap_volume_revert_long or_window_failure_fade_short delivery_pct_anomaly_short; do
    echo "--- $SETUP ---"
    grep -oE "\"reasons\":\[[^]]*$SETUP[^]]*\]" "$JL" 2>/dev/null \
        | grep -oE "$SETUP:[^\"]*" \
        | sort | uniq -c | sort -rn | head -8
    echo
done

echo "=== Accepts by setup ==="
grep -oE '"setup_type":"[^"]*"' "$JL" 2>/dev/null \
    | grep -v unknown \
    | sort | uniq -c | sort -rn

echo
echo "=== Bar-level: scans with any non-gap_fade detector reason ==="
grep -cE '(long_panic|below_vwap|or_window_fail|delivery_pct)[^,]*:' "$JL" 2>/dev/null || true
