# F&O Ban-List Event History — Data Notes

Output:    `data/fno_ban_history/fno_ban_events.parquet`
Generated: 2026-05-14 19:14:14 IST
Window:    2025-10-01 -> 2026-04-30

## Coverage

- Total events:        21
- Distinct ban_dates:  20
- Distinct symbols:    5
- Date span (data):    2025-10-01 -> 2026-04-09
- By event_type:       {'eod': 21}

## Sources

### EOD (primary)
`https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_<DDMMYYYY>.csv`

The archive file's URL date == the effective ban date (verified from the
file's "Trade Date" header text). One file per trading day. File format:
header line `Securities in Ban For Trade Date <DD-MON-YYYY>:` (or
`...: NIL`) followed by lines `<sr_no>,<SYMBOL>`. The public archive does
not include the MWPL% column; `mwpl_pct_at_entry` is therefore `<NA>`
unless intraday snapshots are wired up later.

### Intraday (stub — TODO)
Per the SEBI Nov 3 2025 framework NSE may publish 1-4 intraday ban-list
snapshots when random MWPL checks trigger a ban entry. The endpoint has
not been found in our May-2026 probe (404 on all guessed paths under
`content/fo/fo_secban_intraday_*.csv`). The function
`_fetch_intraday_snapshots()` returns [] until reverse-engineered. When
populated, events will be merged with `event_type='intraday'` and
`entry_snapshot_index` in 1..4.

## Stats

```json
{
  "days_attempted": 136,
  "days_with_csv": 136,
  "days_404": 0,
  "days_failed": 0,
  "days_empty_list": 18,
  "total_banned_rows_seen": 180,
  "eod_entry_events": 21,
  "intraday_entry_events": 0
}
```

Intraday probe: disabled

## Schema

- `symbol` (string — NSE F&O underlying, bare ticker)
- `ban_date` (date — IST-naive, EFFECTIVE trade date the symbol is in ban)
- `ban_entry_time` (Timestamp — NaT for EOD; HH:MM for intraday entries)
- `ban_exit_time` (Timestamp — first trading day after the ban-run end on
   which the symbol is absent from the list; 09:15 IST market-open marker;
   NaT if still pending at end-of-window or not yet observed)
- `mwpl_pct_at_entry` (float — MWPL% utilisation at entry; `<NA>` if not in
   source file)
- `event_type` ('eod' | 'intraday')
- `entry_snapshot_index` (int 1..4 for intraday; `<NA>` for EOD)

## Limitations

1. **EOD MWPL is NA**: the public archive file omits the MWPL% column.
   To recover MWPL at entry, cross-reference with the intraday snapshot
   (once endpoint is found) or scrape the NSE end-of-day FAOII bhavcopy.

2. **Intraday endpoint TBC**: probed `fo_secban_intraday_*.csv` variants
   all return 404 as of May 2026. Path needs to be discovered by network-
   tracing NSE's F&O surveillance page during market hours on a known
   intraday-entry day (e.g. a documented entry day from broker chatter).

3. **Ban exit timing**: `ban_exit_time` is set to 09:15 of the first
   trading day the symbol is absent. The actual cash-segment behaviour
   (lift of ban) is at market open of that date — fine for daily / 5m
   backtests.

4. **Window-edge entries**: a symbol present in the FIRST fetched day of
   the window is treated as a "fresh entry" (no prior-day snapshot to
   compare against). For accurate entry timing at the window-start
   boundary, include 1-2 trading days of pre-window context.
