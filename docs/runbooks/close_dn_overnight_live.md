# close_dn_overnight_long — Go-Live Runbook

## Live crontab (IST; CRON_TZ=Asia/Kolkata). LIVE=1 forces live + Kite.
## NOTE: do NOT use MODE_FLAGS="" — `${VAR:-default}` treats empty as unset and
## falls back to paper (this bug ran a "live" session as paper on 2026-06-23).
26 15 * * 1-5  cd /opt/intraday-trade-assistant && LIVE=1 scripts/cron-entry.sh
05 16 * * 1-5  cd /opt/intraday-trade-assistant && LIVE=1 scripts/cron-place-exit.sh
30 09 * * 1-5  cd /opt/intraday-trade-assistant && LIVE=1 scripts/cron-verify-exit.sh

## Preconditions (do ALL before flipping enabled)
1. Confirm the VM forward-paper ledger exists and PF is acceptable (local state/ is backtest-seeded).
2. Confirm KITE_ACCESS_TOKEN daily refresh runs before 15:26 (reuse the intraday daemon's refresh).
3. Dry-run E2E on a known Cell-#5 fire date: entry -> place-exit -> verify-exit produces a clean trade.

## Pilot (1 slot) then scale
- Set capital_allocation active keys to the _live_pilot_* values; set enabled=true. Run >= several sessions.
- Watch: real BUY fill, 16:05 AMO + GTT placement, 09:30 AMO fill + GTT cancel. No dangling GTTs.
- Then swap active keys to the full _live_* values (25 slots / Rs2.5L).

## Rollback
- Set enabled=false. In-flight positions still settle via verify-exit (idempotent); GTTs auto-cancel on settle.
