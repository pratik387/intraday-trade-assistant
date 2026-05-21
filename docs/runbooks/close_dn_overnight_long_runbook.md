# `close_dn_overnight_long` — Operations Runbook

**Setup:** post-rally EOD-flush overnight reversion (LONG CNC/MTF)
**Status as of 2026-05-21:** paper-trade phase — `enabled: false`, `paper_enabled: true`
**SHIPPABLE record:** `specs/2026-05-21-close_dn_overnight_long-SHIPPABLE-cell-5.md`
**Cell-lock JSON (authoritative config):** `tools/sub9_research/close_dn_overnight_long_cell_lock.json`
**Implementation spec:** `specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md`

---

## TL;DR for daily operations

Two cron jobs run automatically:

```
15:25 IST every weekday  →  signal + place orders
09:30 IST every weekday  →  verify exits + settle slots
```

**Day-to-day check:** look at `state/overnight_slots.json` after each run. All slots should cycle through `free → t0_open → t1_settling → free` over 2 trading days. If any slot is stuck in `t0_open` for >24h, see [Troubleshooting: orphan slot](#troubleshooting-orphan-slot).

**Decay alert:** if `state/decay_tripwire_close_dn_overnight_long.json` shows `paused_since != null`, the setup has paused itself. See [Decay tripwire](#decay-tripwire).

---

## 1. What this setup does

At 15:25 IST every weekday, the entry job evaluates ~500-1500 NSE symbols. It fires for symbols meeting Cell #5:

1. **Closing 25-minute window** (bars 15:00, 15:05, 15:10, 15:15, 15:20) shows heavy sell-volume — `signed_vol_ratio ≤ -0.5`
2. **Volume z-score** vs prior-20-day baseline is `≥ 2.0` (extreme bucket)
3. **Prior session** closed up at least 3% vs the session before
4. Symbol is MTF-eligible (per Zerodha approved list, non-ETF) OR cap ∈ {large, mid, small} (CNC fallback)

When it fires:
- Places **MOC BUY** with `product=MTF` (preferred, ~2.79× avg leverage) or `product=CNC` (1× fallback)
- Places **AMO SELL** (variety=amo) for next-day pre-open auction

Position holds ~17-18 hours overnight. The 09:30 verify-exit job confirms the AMO filled, computes realized PnL (including MTF interest if applicable), and releases capital after T+2 settlement.

**Capital plan**: Rs 5L total = 4 active slots × Rs 1L margin each + Rs 1L cushion. Max 2 new positions per day.

---

## 2. Pre-flight checklist (one-time setup)

Before running the cron jobs for the first time:

### 2.1 Verify config is correct

```bash
.venv/Scripts/python -c "
import json
with open('config/configuration.json', encoding='utf-8') as f:
    c = json.load(f)
cd = c['setups']['close_dn_overnight_long']
print(f'enabled       : {cd[\"enabled\"]}      (False = paper-only; do NOT flip until paper validates)')
print(f'paper_enabled : {cd[\"paper_enabled\"]}')
print(f'mode          : {cd[\"mode\"]}')
print(f'active window : {cd[\"active_window_start\"]} - {cd[\"active_window_end\"]}')
print(f'state file    : {cd[\"capital_allocation\"][\"state_file\"]}')
print(f'MTF snapshot  : {cd[\"mtf\"][\"approved_list_snapshot_path\"]}')
"
```

Expected output:
```
enabled       : False
paper_enabled : True
mode          : overnight
active window : 15:25 - 15:25
state file    : state/overnight_slots.json
MTF snapshot  : data/mtf_universe/approved_mtf_securities_2026-05-21.json
```

### 2.2 Verify MTF snapshot exists and is fresh

```bash
.venv/Scripts/python -c "
from pathlib import Path
from services.mtf_universe import MtfUniverse
mtf = MtfUniverse(Path('data/mtf_universe/approved_mtf_securities_2026-05-21.json'))
print(f'Entries: {len(mtf)}')
print(f'Age: {mtf.snapshot_age_days()} days')
"
```

Expected: `Entries: 1489`, `Age: 0` (or recent). If age > 7 days, refresh:

```bash
.venv/Scripts/python tools/scrape_zerodha_mtf.py
```

Then update `setups.close_dn_overnight_long.mtf.approved_list_snapshot_path` in `config/configuration.json` to the new dated filename.

### 2.3 Verify tests pass

```bash
.venv/Scripts/python -m pytest tests/services/execution/test_overnight_handlers.py tests/services/risk/test_decay_tripwire.py tests/structures/test_close_dn_overnight_long_structure.py -v
```

Expected: 35 passed (no failures, no skips).

### 2.4 Smoke run

Run a dry-run for a historical date with known Cell #5 fires to confirm the pipeline works:

```bash
.venv/Scripts/python main.py --dry-run --mode overnight --action entry --session-date 2025-09-11
```

Expected: exits code 0, stderr summary line like `[overnight entry] summary: fired=X skipped=Y rejected=Z`. The `fired` count may be 0 if the synthetic universe builder doesn't fully match production (e.g., earlier OCI bucket states had different MTF coverage); the exit code 0 is what matters.

### 2.5 Initial state files

Both state files will be auto-created on first run if missing. To pre-seed:

```bash
cp state/overnight_slots.example.json state/overnight_slots.json
```

Leave `state/decay_tripwire_close_dn_overnight_long.json` absent — it's created on the first settled trade.

---

## 3. Cron setup

### 3.1 Crontab entries (Linux/Mac)

```cron
# IST timezone — set CRON_TZ if your system isn't already Asia/Kolkata
CRON_TZ=Asia/Kolkata

# close_dn_overnight_long
25 15 * * 1-5  /opt/intraday-trade-assistant/scripts/cron-entry.sh
30 09 * * 1-5  /opt/intraday-trade-assistant/scripts/cron-verify-exit.sh
```

Replace `/opt/intraday-trade-assistant` with the actual repo path on your VM.

### 3.2 Windows Task Scheduler equivalent

Two tasks, both "Trigger: Daily, Days: Mon-Fri":

| Task | Time | Action: Start a program |
|---|---|---|
| `close_dn_overnight_entry` | 15:25 IST | `E:\Codebase\intraday-trade-assistant\scripts\cron-entry.sh` (via bash) or invoke `python.exe main.py --mode overnight --action entry` directly |
| `close_dn_overnight_verify` | 09:30 IST | `scripts\cron-verify-exit.sh` |

The shell scripts handle log redirection and PYTHON_BIN selection. Direct invocation skips that — you'd need to redirect to a log file yourself.

### 3.3 Override Python binary (if needed)

```bash
PYTHON_BIN=/usr/local/bin/python3.10 scripts/cron-entry.sh
```

Default is `.venv/Scripts/python` (Windows venv layout). On Linux/Mac, set `PYTHON_BIN=.venv/bin/python` via your crontab or wrapper.

---

## 4. Daily operations — what each job does

### 4.1 15:25 IST entry job

```bash
scripts/cron-entry.sh
└─ python main.py --mode overnight --action entry
   └─ services.execution.overnight_handlers.run_entry()
      ├─ Load state/overnight_slots.json
      ├─ Load setups (filter mode="overnight" AND paper_enabled=true)
      ├─ Check decay tripwire — skip dispatch if is_paused()
      ├─ For each setup:
      │   ├─ Build universe (close_dn_overnight_long_universe)
      │   ├─ For each symbol:
      │   │   ├─ Build MarketContext from 5m bars
      │   │   ├─ detector.detect() — signed_vol + volume_z + cell filter + ETF guard
      │   │   ├─ On fire: reserve slot, place MOC BUY, place AMO SELL
      │   │   └─ On reject: log reason, continue
      │   └─ Persist updated slot state
      └─ Exit code 0
```

**Expected log lines** in `logs/overnight_entry_YYYY-MM-DD.log`:

```
INFO  run_entry: universe size for close_dn_overnight_long = 432
INFO  close_dn_overnight_long fired | symbol=NSE:RELIANCE svr=-0.612 vol_z=2.83 prior_ret=4.21% product=MTF lev=3.85
INFO  run_entry: complete | fired=2 skipped=0 rejected=430
[overnight entry] summary: fired=2 skipped=0 rejected=430
```

**Run wallclock**: 1-5 minutes depending on universe size.

### 4.2 09:30 IST verify-exit job

```bash
scripts/cron-verify-exit.sh
└─ python main.py --mode overnight --action verify-exit
   └─ services.execution.overnight_handlers.run_verify_exit()
      ├─ Load state/overnight_slots.json
      ├─ Phase 1: settle t0_open slots whose expected_exit_date <= today
      │   ├─ Look up next-day 09:15 open (paper) or check AMO status (live)
      │   ├─ Compute fees + interest via calc_fee_mtf / calc_fee_cnc
      │   ├─ pool.settle() — slot transitions t0_open → t1_settling
      │   └─ decay_tripwire.record_trade(realized_pnl)
      ├─ Phase 2: release t1_settling slots whose T+2 has passed
      │   └─ pool.release() — slot transitions t1_settling → free
      └─ Persist updated state
```

**Expected log lines** in `logs/overnight_verify_YYYY-MM-DD.log`:

```
INFO  run_verify_exit: settle slot 1 NSE:RELIANCE: buy=2500.00 sell=2515.00 pnl=+Rs1,650
INFO  DecayTripwire[close_dn_overnight_long]: trade #15 recorded
INFO  run_verify_exit: release slot 1 (T+2 settled)
INFO  run_verify_exit: complete | settled=2 released=2 orphan_t0=0
[overnight verify-exit] summary: settled=2 released=2 orphan_t0=0
```

**Run wallclock**: < 30 seconds (no symbol iteration).

---

## 5. State files

### 5.1 `state/overnight_slots.json` — capital pool state

Authoritative state for slot lifecycle. Inspect anytime:

```bash
.venv/Scripts/python -c "
import json
from pathlib import Path
s = json.loads(Path('state/overnight_slots.json').read_text())
print(f'Max slots: {s[\"max_slots\"]}')
print(f'Active slots:')
for slot in s['slots']:
    if slot['status'] != 'free':
        print(f'  slot {slot[\"slot_id\"]}: {slot[\"status\"]} | {slot[\"symbol\"]} | '
              f'product={slot[\"product\"]} lev={slot[\"leverage\"]} '
              f'buy={slot[\"buy_fill_price\"]} sell={slot[\"sell_fill_price\"]} '
              f'pnl={slot[\"realized_pnl_inr\"]}')
print(f'Free slots: {sum(1 for s2 in s[\"slots\"] if s2[\"status\"] == \"free\")}/{len(s[\"slots\"])}')
"
```

**Slot lifecycle (each slot cycles through these states):**

| Status | When | What it means |
|---|---|---|
| `free` | Initial / after release | Slot available for next signal |
| `t0_open` | After 15:25 entry | BUY filled, AMO SELL queued |
| `t1_settling` | After 09:15 next-day | SELL filled, cash pending T+2 |
| (back to `free`) | After T+2 morning | Cash settled, slot reusable |

### 5.2 `state/decay_tripwire_close_dn_overnight_long.json` — decay monitor

Created on the first settled trade. Inspect:

```bash
.venv/Scripts/python -c "
from pathlib import Path
from services.risk.decay_tripwire import DecayTripwire
cfg = {
    'state_file': 'state/decay_tripwire_close_dn_overnight_long.json',
    'window_trades': 30, 'pf_floor': 1.20, 'sustained_weeks': 6,
}
if Path(cfg['state_file']).exists():
    tw = DecayTripwire('close_dn_overnight_long', Path(cfg['state_file']),
                       cfg['window_trades'], cfg['pf_floor'], cfg['sustained_weeks'])
    print(tw.state_summary())
else:
    print('No tripwire state yet — no trades settled.')
"
```

Key fields:
- `trade_count` — total trades recorded
- `current_rolling_pf` — PF over last 30 trades (None if < 30 trades)
- `first_below_floor_ts` — when PF first dropped below 1.20 (None if not breaching)
- `paused_since` — timestamp when tripwire paused dispatch (None if active)

### 5.3 Trade records

The cron job logs each fire and settle to `logs/`. For per-trade analysis, parse the log files or query the slot state at each release event. (No separate analytics jsonl yet — that's a separate hookup if needed.)

---

## 6. Decay tripwire

The tripwire pauses dispatch if rolling 30-trade PF drops below 1.20 sustained for 6 weeks.

### 6.1 Why this exists

The Cell #5 forward expectation is PF 1.2-1.6 (per the SHIPPABLE record's decay warning). Yearly PF showed degradation: 2023=3.80 → 2024=4.34 → 2025=2.88 → 2026=1.93. The tripwire is the safety net against further decay during paper validation.

### 6.2 What happens when it trips

- `run_entry` logs `setup close_dn_overnight_long is PAUSED by decay tripwire (since YYYY-MM-DDTHH:MM:SS); skipping dispatch`
- No new BUYs are placed
- Existing positions (slots in `t0_open` or `t1_settling`) continue their lifecycle — `run_verify_exit` still settles + releases them

### 6.3 Manual reset (resume dispatch)

After investigating WHY the tripwire fired (likely you need to retire the setup or re-validate):

```bash
# Option A: programmatic reset (keeps trade history, clears pause + breach watch)
.venv/Scripts/python -c "
from pathlib import Path
from services.risk.decay_tripwire import DecayTripwire
tw = DecayTripwire(
    'close_dn_overnight_long', Path('state/decay_tripwire_close_dn_overnight_long.json'),
    30, 1.20, 6
)
tw.reset()
print('Tripwire reset — dispatch will resume on next entry job.')
"

# Option B: nuclear option (delete state file, start fresh)
rm state/decay_tripwire_close_dn_overnight_long.json
```

Option A is preferred — preserves trade history so the next pause check uses real data.

---

## 7. Troubleshooting

### 7.1 No fires for several days

Expected behavior. Cell #5 is selective — historical median was 3-4 trades/day across the full universe, but with the 2-per-day cap and 4-slot capacity, real captures average ~2/day. **A 0-fire day is normal**; a multi-day drought is the candidate degrading.

Verify the universe is being computed:

```bash
grep "universe size" logs/overnight_entry_$(date +%Y-%m-%d).log
```

Expected: `universe size for close_dn_overnight_long = 432` (or similar). If 0, see [Troubleshooting: zero universe](#72-zero-universe-on-entry).

### 7.2 Zero universe on entry

```bash
grep "universe size" logs/overnight_entry_*.log | tail -5
```

If consistently 0:

1. **MTF snapshot stale or missing**:
   ```bash
   ls -la data/mtf_universe/approved_mtf_securities_*.json
   ```
   If empty or > 14 days old, refresh:
   ```bash
   .venv/Scripts/python tools/scrape_zerodha_mtf.py
   # then update setups.close_dn_overnight_long.mtf.approved_list_snapshot_path
   ```

2. **`nse_all.json` missing or corrupt**:
   ```bash
   .venv/Scripts/python -c "import json; print(len(json.load(open('nse_all.json'))))"
   ```
   Expected: ~2300. If much lower, restore from git.

3. **Broker daily_dict not returning data**: the universe builder iterates `broker.get_daily()` for each symbol. If broker is misconfigured (e.g., wrong access token in live mode), it returns empty dicts. Check broker log lines:
   ```bash
   grep -i "daily" logs/overnight_entry_*.log | tail -10
   ```

### 7.3 Orphan slot

A slot stuck in `t0_open` past its `expected_exit_date` means the AMO either didn't fire or our verify-exit isn't seeing the fill.

```bash
.venv/Scripts/python -c "
import json
from datetime import date
s = json.loads(open('state/overnight_slots.json').read_text())
today = date.today().isoformat()
for slot in s['slots']:
    if slot['status'] == 't0_open' and slot['expected_exit_date'] and slot['expected_exit_date'] < today:
        print(f'ORPHAN: slot {slot[\"slot_id\"]} ({slot[\"symbol\"]}) — expected exit {slot[\"expected_exit_date\"]}')"
```

Causes + fixes:

| Cause | Fix |
|---|---|
| AMO was rejected by broker (margin shortage, circuit limit, price freeze) | Check broker order history; manually close position via Kite app; then run `verify-exit` again |
| Cron job didn't run that morning (VM down, network down) | Re-run manually: `scripts/cron-verify-exit.sh`. Idempotent. |
| Public holiday on `expected_exit_date` | AMO carries over to next session; just wait one more day and re-run verify-exit |
| Symbol was suspended overnight | Manual close required; remove the symbol from MTF list or check NSE bulletin for the suspension reason |

If the position needs to be manually closed and the slot needs to be force-released:

```bash
# Edit state file manually (BE CAREFUL):
.venv/Scripts/python -c "
import json
from pathlib import Path
s = json.loads(Path('state/overnight_slots.json').read_text())
for slot in s['slots']:
    if slot['slot_id'] == 1:  # change to the orphan slot_id
        slot['status'] = 'free'
        slot['symbol'] = slot['product'] = None
        slot['leverage'] = 1.0
        slot['margin_inr'] = slot['notional_inr'] = 0.0
        slot['buy_fill_price'] = slot['buy_fill_ts'] = slot['buy_order_id'] = None
        slot['amo_sell_order_id'] = slot['expected_exit_date'] = None
        slot['sell_fill_price'] = slot['sell_fill_ts'] = None
        slot['realized_pnl_inr'] = slot['fees_inr'] = slot['interest_inr'] = None
        slot['reserved_today'] = None
        break
Path('state/overnight_slots.json').write_text(json.dumps(s, indent=2))
print('Slot 1 force-released. Verify with the inspection script.')
"
```

This loses the realized PnL for that trade — record it manually elsewhere if you need it for analytics.

### 7.4 Cron didn't run

If you see no `logs/overnight_entry_YYYY-MM-DD.log` file for a weekday:

1. **Check cron daemon**: `systemctl status cron` (Linux) or Task Scheduler History (Windows)
2. **Check timezone**: did the cron fire at 15:25 IST or at 15:25 UTC? `CRON_TZ=Asia/Kolkata` must be set, OR all crontab times must be in UTC (which is 09:55 UTC for our 15:25 IST entry)
3. **Re-run manually** — the entry job is idempotent only if it didn't reach the order-placement step. If you ran it AFTER 15:30, the bars look the same but the broker may have moved past MOC window in live mode. In paper/backtest, time is simulated so re-running is safe.

### 7.5 Tests fail after a code change

```bash
.venv/Scripts/python -m pytest tests/services/execution/ tests/services/risk/ tests/structures/test_close_dn_overnight_long_structure.py -v
```

Expected: 35 passed. If failures, the change likely broke one of the load-bearing contracts:
- ExitLevels new fields removed → check `structures/data_models.py`
- OvernightSlotPool JSON schema changed → check `services/capital_manager.py`
- setup_registry mode filter changed → check `services/dispatch/setup_registry.py`

Revert the breaking change or update the tests with intent.

### 7.6 Decay tripwire pausing prematurely

If `paused_since` is set unexpectedly:

```bash
.venv/Scripts/python -c "
from pathlib import Path
from services.risk.decay_tripwire import DecayTripwire
tw = DecayTripwire(
    'close_dn_overnight_long', Path('state/decay_tripwire_close_dn_overnight_long.json'),
    30, 1.20, 6
)
import json
print(json.dumps(tw.state_summary(), indent=2))
"
```

Check whether:
- `current_rolling_pf` is truly below 1.20 — if yes, the setup IS underperforming
- `first_below_floor_ts` timestamp is plausibly 6+ weeks before today's date
- The trade count is at least 30 (smaller windows shouldn't trip the gate)

If everything looks right, the setup is decaying and the tripwire is doing its job. Investigate WHY (regime change? structural change in NSE microstructure?) before resetting.

If the trip looks spurious (e.g., your 6-week window contained an outlier loss cluster from a market crash day):

```bash
# Manual reset preserves trade history; the gate re-evaluates on the next settle
.venv/Scripts/python -c "
from pathlib import Path
from services.risk.decay_tripwire import DecayTripwire
DecayTripwire('close_dn_overnight_long', Path('state/decay_tripwire_close_dn_overnight_long.json'),
              30, 1.20, 6).reset()
"
```

---

## 8. Emergency procedures

### 8.1 Disable the setup completely

```bash
# Edit config/configuration.json:
#   setups.close_dn_overnight_long.paper_enabled = false
#   setups.close_dn_overnight_long.enabled = false
```

The next entry cron run will skip the setup entirely. **In-flight positions** (slots in t0_open / t1_settling) continue their lifecycle — `run_verify_exit` still settles them. To force-close, see [Manual exit](#83-manual-position-exit).

### 8.2 Pause new entries (keep existing positions running)

Trigger the decay tripwire manually:

```bash
.venv/Scripts/python -c "
import json
from datetime import datetime
from pathlib import Path
state_path = Path('state/decay_tripwire_close_dn_overnight_long.json')
if state_path.exists():
    s = json.loads(state_path.read_text())
else:
    s = {'setup_name': 'close_dn_overnight_long', 'window_trades': 30,
         'pf_floor': 1.20, 'sustained_weeks': 6, 'trades': []}
s['first_below_floor_ts'] = datetime.now().isoformat()
s['paused_since'] = datetime.now().isoformat()
state_path.write_text(json.dumps(s, indent=2))
print('Manual pause applied — next entry job will skip dispatch.')
"
```

To unpause: `DecayTripwire.reset()` (see [Decay tripwire reset](#63-manual-reset-resume-dispatch)).

### 8.3 Manual position exit

To exit an open position outside the normal flow (e.g., bad-news symbol, broker margin call):

1. **Cancel the AMO SELL via Kite app** (web/mobile) — order_id is in the slot record's `amo_sell_order_id`
2. **Place a manual SELL** via Kite app to close the position
3. **Force-release the slot** (see [Orphan slot](#73-orphan-slot) for the script)
4. Record the realized PnL for analytics: log it in `state/manual_exits.log` or your spreadsheet

### 8.4 VM down

If the VM was offline during entry time (15:25 IST):
- **Missed entry** is missed — no recovery; the 15:25-15:30 MOC window has passed
- Just verify the cron picks up correctly tomorrow

If the VM was offline during verify-exit time (09:30 IST):
- AMO **did execute** at the broker (broker holds it independently)
- Just run verify-exit when VM is back up — idempotent
  ```bash
  scripts/cron-verify-exit.sh
  ```

---

## 9. Activating live (post-paper-trade)

Before flipping `enabled: true`:

### 9.1 Paper validation gates (from SHIPPABLE record)

8-12 weeks paper trading required. Pass to live if ALL of:

1. Median trades/day in paper matches OOS+HO baseline (3-5) within ±2
2. p95 trades/day ≤ 12 (vs HO p95=9)
3. MOC entry slippage: median ≤ 10bps vs 15:30 official close
4. Pre-open exit fill rate ≥ 95% of orders cleared at 09:15 auction
5. Rolling 4-week PF ≥ 1.10 throughout
6. No 4-consecutive-week PF < 0.9

### 9.2 Fail-stop conditions during paper

- 2 consecutive weeks PF < 0.7
- Any week NET drawdown > Rs 20K (on Rs 10L overnight capital)
- Execution-layer bug (wrong qty, wrong side, missed MOC fill)

If any of these trigger, retire the candidate or re-validate the cell.

### 9.3 Activation steps

```bash
# 1. Final config flip
# Edit config/configuration.json:
#   setups.close_dn_overnight_long.enabled = true
#   (paper_enabled stays true — doesn't hurt)

# 2. Verify cron jobs are pointing at live broker (not dry-run)
# main.py routes to KiteBroker when --dry-run + --paper-trading are BOTH absent

# 3. Set Zerodha credentials
export KITE_API_KEY=<your_api_key>
export KITE_ACCESS_TOKEN=<your_access_token>

# 4. Manual smoke (live) — confirm broker connectivity
.venv/Scripts/python -c "
import os
from broker.kite.kite_broker import KiteBroker
b = KiteBroker(api_key=os.environ['KITE_API_KEY'], access_token=os.environ['KITE_ACCESS_TOKEN'])
print(f'LTP for RELIANCE: {b.get_ltp(\"NSE:RELIANCE\")}')
"

# 5. First-day operations: watch the logs closely
tail -F logs/overnight_entry_$(date +%Y-%m-%d).log
```

---

## 10. Reference appendix

### 10.1 File paths

| Type | Path |
|---|---|
| Detector | `structures/close_dn_overnight_long_structure.py` |
| Universe builder | `services/setup_universe.py:close_dn_overnight_long_universe` |
| Cron handlers | `services/execution/overnight_handlers.py` |
| Capital pool | `services/capital_manager.py:OvernightSlotPool` |
| MTF loader | `services/mtf_universe.py` |
| Decay tripwire | `services/risk/decay_tripwire.py` |
| Fee helpers | `tools/sub7_validation/build_per_setup_pnl.py:calc_fee_cnc/_mtf` |
| Cron wrappers | `scripts/cron-entry.sh`, `scripts/cron-verify-exit.sh` |
| State (runtime, gitignored) | `state/overnight_slots.json`, `state/decay_tripwire_close_dn_overnight_long.json` |
| State (example, committed) | `state/overnight_slots.example.json` |
| MTF snapshot | `data/mtf_universe/approved_mtf_securities_YYYY-MM-DD.json` |
| MTF scraper | `tools/scrape_zerodha_mtf.py` |
| Cell-lock JSON (authoritative config snapshot) | `tools/sub9_research/close_dn_overnight_long_cell_lock.json` |
| SHIPPABLE record | `specs/2026-05-21-close_dn_overnight_long-SHIPPABLE-cell-5.md` |
| Implementation spec | `specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md` |
| Logs | `logs/overnight_entry_YYYY-MM-DD.log`, `logs/overnight_verify_YYYY-MM-DD.log` |

### 10.2 Config keys (all under `setups.close_dn_overnight_long`)

| Key | Default | Notes |
|---|---|---|
| `enabled` | `false` | Live activation flag. KEEP FALSE during paper phase. |
| `paper_enabled` | `true` | Paper-trade flag. Set false to fully disable the setup. |
| `mode` | `"overnight"` | Drives setup_registry filtering. Do not change. |
| `active_window_start` | `"15:25"` | Single-bar window. |
| `active_window_end` | `"15:25"` | Single-bar window. |
| `signed_vol_ratio_max` | `-0.5` | Primary filter — closing 25m must be sell-dominated. |
| `closing_25m_volume_z_min` | `1.0` | Primary filter — closing volume must exceed prior-20d baseline. |
| `min_signal_bar_count` | `4` | At least 4 of 5 signal bars must be present. |
| `cell_volume_z_min` | `2.0` | Cell #5 filter — extreme volume bucket. |
| `cell_prior_day_return_pct_min` | `3.0` | Cell #5 filter — post-up-3% day. |
| `baseline_rolling_days` | `20` | Days for closing-25m baseline computation. |
| `min_daily_avg_volume` | `50000` | Universe filter. |
| `min_trading_days_required` | `30` | Universe filter — symbol must have 30+ days of daily history. |
| `universe_max_symbols` | `1500` | Operational cap to bound iteration cost. |
| `capital_allocation.active_margin_inr` | `400000` | Sum of margin across active slots. Equals slots × margin_per_slot. |
| `capital_allocation.cushion_inr` | `100000` | Permanent buffer outside active deployment. |
| `capital_allocation.max_concurrent_slots` | `4` | Slot pool size. |
| `capital_allocation.margin_per_slot_inr` | `100000` | Margin reserved per BUY. |
| `capital_allocation.max_new_positions_per_day` | `2` | Per-day rate limit. |
| `capital_allocation.state_file` | `"state/overnight_slots.json"` | Slot pool persistence. |
| `mtf.approved_list_snapshot_path` | (dated path) | Zerodha MTF list location. |
| `mtf.interest_pct_per_day` | `0.0004` | 0.04% per day on borrowed amount. |
| `mtf.exclude_etf` | `true` | Skip MTF-eligible ETFs (mechanism mismatch). |
| `mtf.fallback_to_cnc_if_not_mtf` | `true` | Non-MTF large/mid/small caps use CNC instead. |
| `mtf.stale_snapshot_warn_days` | `7` | Log warning if MTF snapshot is older than this. |
| `decay_tripwire.window_trades` | `30` | Rolling-PF window. |
| `decay_tripwire.pf_floor` | `1.20` | Pause threshold. |
| `decay_tripwire.sustained_weeks` | `6` | Time below floor before pause. |
| `decay_tripwire.state_file` | `"state/decay_tripwire_close_dn_overnight_long.json"` | Tripwire persistence. |

### 10.3 Implementation commit history

```
5e5b90e Task 7 — decay tripwire monitor
4d7f1e5 Task 6 — cron handlers (entry + verify-exit)
69a965f Task 5 — MTF + AMO order routing in KiteBroker
a62f2c3 Task 4 — detector with 5-bar signal + ETF guard + MTF routing
e502782 Task 3 — OvernightSlotPool with JSON-persisted state
524c5b4 Task 2 — MTF universe loader + close_dn_overnight_long_universe
8e8b453 Task 1 — CNC + MTF fee helpers
de2fd82 Task 0 — --mode and --action CLI + setup_registry filter
ab1288a spec rewrite under cron model
d93b14d initial impl spec
3fcf73c Phase 5 SHIPPABLE Cell #5 record
```

### 10.4 Test coverage

```bash
# Setup-specific tests (35 total, all should pass)
.venv/Scripts/python -m pytest \
  tests/test_main_modes.py \
  tests/dispatch/test_setup_registry_mode.py \
  tests/sub7_validation/test_fee_modes.py \
  tests/services/test_close_dn_overnight_universe.py \
  tests/services/test_capital_manager_overnight.py \
  tests/structures/test_close_dn_overnight_long_structure.py \
  tests/broker/test_kite_broker_mtf_amo.py \
  tests/services/execution/test_overnight_handlers.py \
  tests/services/risk/test_decay_tripwire.py \
  -v
```

Expected: **85 passed**.
