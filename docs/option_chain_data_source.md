# Option-Chain Data Source — Decision Doc

> Sub-project #8, plan #6 (`expiry_pin_strike_reversal`), Phase A0 + A1.
> Companion to `specs/2026-04-29-expiry_pin_strike_reversal-plan.md`.

The detector needs **end-of-day settlement Open-Interest** for every NSE F&O
contract from **2023-01-02 onwards**. We need to identify the strike with the
highest aggregate OI (CE + PE) on each session — the "pin strike" that
market-makers' delta hedging tends to magnetize spot toward on expiry days.

This document records the four candidate sources we surveyed, the chosen path,
the storage schema, and the directory layout the ingestion writes to.

## Candidate Sources

### 1. NSE bhavcopy daily archive (CHOSEN)

End-of-day settlement bhavcopy ZIP files published on `archives.nseindia.com`
and `nsearchives.nseindia.com`. Contains every F&O contract that traded on the
session, with settlement OI, settlement price, daily volume, and `CHG_IN_OI`.

**Strengths:**
- Free, no auth required, no rate-limiting on backfill (static historical files)
- Authoritative — these are NSE's own end-of-day snapshots, used by clearing
- Full window: 2023-01-02 onwards is fully reachable
- Parseable as CSV in a ZIP (no JSON/HTML scraping)

**Schema split — two URL templates:**
- 2023-01-02 → 2023-12-31 (legacy):
  `archives.nseindia.com/content/historical/DERIVATIVES/<YYYY>/<MMM>/fo<DDMMMYYYY>bhav.csv.zip`
  e.g. `fo28DEC2023bhav.csv.zip`
- 2024-01-01 onwards (new):
  `nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_<YYYYMMDD>.csv.zip`
  e.g. `BhavCopy_NSE_FO_20240101.csv.zip`

The legacy archive remained reachable post-2024 cutover, so we can use whichever
URL template the date falls under (dispatch via `_NEW_SCHEME_CUTOVER` constant).

**Weaknesses:**
- Two schema variants (legacy: INSTRUMENT/STRIKE_PR/OPTION_TYP — new:
  FinInstrmTp/StrkPric/OptnTp) — handled by detection-and-rename in
  `_nse_bhavcopy_client.parse_bhavcopy`
- IV not present (placeholder column kept as NULL — not needed by the detector)
- New scheme uses `Call`/`Put` instead of `CE`/`PE` — normalized at parse time

### 2. Upstox option-chain endpoint

Public REST endpoint returning live option chain snapshots for a given symbol
+ expiry. Useful for live OI feed.

**Strengths:**
- No subscription required for ingredient data
- Already integrated for live trading (`upstox_cache_downloader.py`)

**Weaknesses:**
- **No bulk historical** — only live + intraday. Cannot back-fill 2023-2025.
- Per-symbol endpoint — would need many round-trips to assemble a full chain
- Subject to live-API rate limiting
- Different schema from NSE bhavcopy

**Verdict:** Out of scope for Part A. May be revisited later as the live-feed
bridge for the detector running in production.

### 3. Kite Connect historical option chain

Zerodha's Kite Connect historical API supports option-chain queries with a paid
subscription.

**Strengths:**
- Authoritative real-time / pre-clearing OI
- Same broker ecosystem we already trade through

**Weaknesses:**
- Requires paid subscription on top of the existing Kite trading subscription
- Different rate limits for historical vs live
- Identical settlement values to bhavcopy at end-of-day, so paying for it
  buys nothing for back-test purposes

**Verdict:** Rejected — bhavcopy gives the same end-of-day OI for free.

### 4. NSE archive ZIP — legacy `fo<DDMMMYYYY>bhav.csv.zip` (deprecated)

This is what's now URL #1's "legacy" branch. NSE deprecated this path
in late 2023 for a unified `BhavCopy_NSE_FO_*.csv` format; the legacy archive
is still reachable for old dates. Treated as part of source #1, not separate.

## Decision

**Chosen path: NSE bhavcopy daily archive (source #1) — legacy + new hybrid.**

The ingestion picks the right URL template based on `session_date >=
date(2024, 1, 1)`. This single source covers the full 2023-01-02 →
2026-04-29 backfill window with no rate-limit risk.

**Live OI fallback (out of Part A scope):** When the detector runs in live
mode, it should use the Upstox option-chain endpoint (source #2) for the
*current* session's intraday OI; the bhavcopy is end-of-day settlement and
arrives only after market close.

## Storage Schema

Canonical parquet schema written by `tools/option_chain/fetch_oi_snapshot.py`
and consumed by `services/option_chain_loader.py`:

| column | dtype | source field (legacy) | source field (new) |
|---|---|---|---|
| `session_date` | `date` | (set by ingestor) | (set by ingestor) |
| `symbol` | `str` | `SYMBOL` | `TckrSymb` |
| `expiry_date` | `date` | `EXPIRY_DT` | `XpryDt` |
| `strike` | `float64` | `STRIKE_PR` | `StrkPric` |
| `option_type` | `str` (CE/PE) | `OPTION_TYP` | `OptnTp` (Call/Put → CE/PE) |
| `oi` | `int64` | `OPEN_INT` | `OpnIntrst` |
| `oi_change` | `int64` | `CHG_IN_OI` | `ChngInOpnIntrst` |
| `vol` | `int64` | `CONTRACTS` | `TtlTradgVol` |
| `ltp` | `float64` | `CLOSE` | `ClsPric` |
| `settlement_price` | `float64` | `SETTLE_PR` | `SttlmPric` |
| `iv` | `pd.NA` | (not present) | (not present) |

Filtered to options only (`INSTRUMENT IN ('OPTIDX', 'OPTSTK')` for legacy or
`FinInstrmTp IN ('STO', 'IDO')` for the new scheme). Futures rows are
discarded — the detector never reads them.

**Why parquet:**
- Column-pruned reads — the loader only ever pulls `(symbol, expiry_date,
  strike, oi)` from a session, ~30% of bytes off disk
- ~10× smaller on disk than the equivalent CSV (binary, snappy compression)
- Schema travels with the file — no need to re-detect legacy vs new at read time
- Compatible with both pyarrow and fastparquet engines

## Directory Tree

```
data/option_chain/
├── .gitkeep
├── 2023/
│   ├── 01/
│   │   ├── 2023-01-02.parquet
│   │   ├── 2023-01-03.parquet
│   │   └── ...
│   └── 12/
└── 2024/
    └── 06/
        ├── 2024-06-05.parquet
        └── 2024-06-06.parquet
```

`.parquet` files are gitignored (large binary tree); only `.gitkeep` and the
docs are committed.

## Validation

Each session parquet must satisfy (enforced by `fetch_oi_snapshot._validate`
before write):
- ≥ 100 contracts (sanity floor — a bhavcopy with fewer rows is malformed)
- No null OI, strike, or option_type
- Every `expiry_date >= session_date` (settled-future rule — past expiries
  shouldn't appear)

A failed validation aborts the write and surfaces a `ValueError` to the caller,
counted in `ingest_range`'s `sessions_failed` summary.

## Backfill — User Deferred Step

The actual ~800-session backfill (2023-01-02 → 2026-04-29) is the user's
deferred step. This module is the executable infrastructure to run it:

```
.venv/Scripts/python tools/option_chain/fetch_oi_snapshot.py \
    --start 2023-01-02 --end 2026-04-29 \
    --out-root data/option_chain
```

Once populated, the loader API (`services.option_chain_loader.load_oi_snapshot`,
`find_max_oi_strike`, `is_monthly_expiry`) is ready for the Phase B detector.
