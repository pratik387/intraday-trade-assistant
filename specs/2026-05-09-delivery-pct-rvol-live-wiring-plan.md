# delivery_pct cross-day RVOL — live-mode wiring plan

**Status:** Deferred. Backtest path (OCI) is wired and verified in commit
`924c2a3`. Live wiring is gated on the setup passing Phase-1 validation
in OCI. Picked up only if `delivery_pct_anomaly_short` ships.

**Owner:** sub-9 round-7 follow-up.

**Why this doc exists:** the live wiring is small but non-obvious work that
spans four subsystems (data feed, scheduler, screener startup, monitoring).
Capturing it once so the next pass doesn't have to re-derive it.

---

## What's already built (commit 924c2a3)

| Component | File | Live-ready? |
|---|---|---|
| Build script | `tools/cross_day_rvol/build_baseline.py` | **No** — reads from `backtest-cache-download/monthly/*5m_enriched.feather` (backtest-only source) |
| Lookup service | `services/cross_day_rvol_enrichment.py` | **Yes** — pure file-loader, environment-agnostic |
| Detector use | `structures/delivery_pct_anomaly_short_structure.py` `_cross_day_rvol` | **Yes** — calls the service, no env-specific code |
| OCI cache wiring | `oci/tools/upload_cross_day_rvol.py`, `oci/docker/entrypoint.py::download_cross_day_rvol()` | **Backtest only** — runs at OCI pod startup |
| Daily refresh job | — | **Missing** |
| Live 5m source | — | **Missing** — build script can't read live's `_precomputed_5m` or broker-fetched bars |
| Staleness guard | — | **Missing** — service silently no-ops if parquet is missing/stale |

## Mechanic (why the deferral is safe)

The baseline at `(symbol, today, hhmm)` = rolling-20-day mean of `volume`
at that same `hhmm` across the prior 20 trading days. **All 20 inputs
are in the past** — fully computable BEFORE today's market opens, using
data through yesterday EOD.

That makes the refresh trivially schedulable: any time after EOD on
day D, append D's bars and recompute means ending at D. By 09:00 IST
on D+1, the parquet has every (symbol, D+1, hhmm) row the detector
will need during D+1's session.

## Live wiring tasks (when the setup ships)

### 1. Source switch in `build_baseline.py`

Add a `--source` flag:
- `--source backtest` (current default): monthly feathers (where they exist)
- `--source live`: read from the live screener's 5m archive

The live source needs identification. Most likely candidates (verify
when picked up):
- Wherever `screener_live._precomputed_5m` is persisted between sessions
  (today the in-memory dict probably isn't durable — may need a new
  daily flush)
- Broker API end-of-day fetch for the day's 5m bars (Kite, Upstox both
  expose this)

Pick whichever the rest of the live pipeline already produces; don't
build a new collector just for this.

### 2. Daily refresh scheduler

Trigger: weekday 15:35–16:00 IST (15:35 = 5 min after MIS auto-square-off,
gives the live 5m archive time to flush the day's last bar).

Implementation choices (in preference order):
1. **Add to live operator's existing post-market job set.** If there's
   already a "build daily reports" or "sync logs to S3" cron, append
   one more step that runs:
   ```
   python tools/cross_day_rvol/build_baseline.py \
       --source live \
       --start <today-30d> \
       --end <today>
   ```
   Then upload to OCI bucket (if OCI gauntlet still consumes it) via
   `python oci/tools/upload_cross_day_rvol.py`.
2. **Standalone systemd timer / Windows Task Scheduler entry** if no
   post-market job set exists yet. Same command.
3. **Hook into the live screener's own EOD shutdown.** Cleanest but
   couples the screener process to a non-trading concern; only do this
   if the screener is already responsible for end-of-day flushes.

The build is incremental-friendly already (rolling means computed across
the whole window, but the script appends new rows without recomputing
historical baselines). For a full year refresh, runtime is ~30s.

### 3. Staleness guard in the enrichment service

Add to `services/cross_day_rvol_enrichment.py::_load()`:

```python
# After loading the parquet:
max_date = max(d for (_, d, _) in _LOOKUP.keys())
today = now_ist_naive().date()
if (today - max_date).days > 3:   # tolerate weekends
    # Don't return None silently — log and disable the detector
    logger.error(
        "cross_day_rvol baseline stale (max date %s, today %s); "
        "delivery_pct_anomaly_short will reject every signal until refreshed.",
        max_date, today,
    )
    return None
```

Tolerance: 3 days handles long weekends + holidays (Sat/Sun + Mon
holiday = 3 days since last business day). Anything older is a
genuinely missed refresh.

In live this matters because a silent no-fire is much worse than a
loud rejection — operator needs to see the alert and run the refresh.

### 4. Monitoring

Add the staleness check as a startup probe in the screener's
session-start sequence (alongside the existing daily-cache seed at
`services/screener_live.py:1168`). If stale, refuse to start with
`delivery_pct_anomaly_short` enabled. Same pattern as the existing
fail-fast around `daily_seed`.

Optionally surface in the existing health-check / heartbeat that
ops watches.

---

## Test plan when picking this up

1. **Source-switch unit test:** build script with `--source live`
   produces a parquet with the same schema / date range / row counts
   as `--source backtest` for a recent week.
2. **Staleness guard unit test:** mock parquet with `max_date = today-5`,
   assert `_load()` returns None and emits ERROR log.
3. **End-to-end live smoke test:** run live screener (paper mode) on a
   day after refresh; verify `delivery_pct_anomaly_short` evaluates
   without RVOL-unavailable rejections.

## Things NOT to forget

- **The existing OCI/backtest path stays exactly as-is.** Live wiring
  adds a flag to the build script; backtest build still uses
  `--source backtest` (or no flag) and reads monthly feathers.
- **Incremental builds** are essential — re-running the full
  2023-01..today rebuild every night burns time. The script already
  outputs date-trimmed parquets; running with last-30-days of source
  data produces correct rolling means for the new dates.
- **OCI bucket sync** (if the gauntlet keeps running on OCI in
  parallel with live): the upload tool stays the same; just rerun
  it after each daily build to keep the bucket fresh.
- **Symbol prefix consistency:** the parquet uses bare tickers ("ACC"
  not "NSE:ACC"); the enrichment service strips "NSE:" before lookup.
  Don't accidentally double-strip or re-add the prefix in the live
  source switch.
