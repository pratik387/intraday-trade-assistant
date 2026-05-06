# Sub-Project 9 — Stock-to-Sector Mapping (Round-4 Sector Rotation Sanity)

**Date:** 2026-05-06
**Owner:** sub-9 round-4 `sector_rotation_relative_strength` candidate
**Output:** `assets/stock_sector_map.json`
**Builder:** `tools/build_stock_sector_map.py`

## Goal

Provide a deterministic `{NSE:SYMBOL: NSE_NIFTY_<SECTOR>}` mapping covering the
F&O 200 universe (`assets/fno_liquid_200.csv`, 153 active symbols), keyed to the
sector-index directory naming used in `backtest-cache-download/index_ohlcv/...`,
so the round-4 sanity tool can join a stock's intraday returns against its
sector index without a second lookup table.

## Method

1. Read NSE Indices public constituent CSVs already cached in `assets/`:
   - `ind_niftybanklist.csv` -> `NSE_NIFTY_BANK`
   - `ind_niftyitlist.csv` -> `NSE_NIFTY_IT`
   - `ind_niftyautolist.csv` -> `NSE_NIFTY_AUTO`
   - `ind_niftyfmcglist.csv` -> `NSE_NIFTY_FMCG`
   - `ind_niftypharmalist.csv` -> `NSE_NIFTY_PHARMA`
   - `ind_niftymetallist.csv` -> `NSE_NIFTY_METAL`
   - `ind_niftyenergylist.csv` -> `NSE_NIFTY_ENERGY` (downloaded fresh via
     archives.nseindia.com — niftyindices.com host timed out)
   - `ind_niftyfinancelist.csv` -> `NSE_NIFTY_FIN_SERVICE`
     (file is the Nifty Financial Services index, despite the `finance` filename)
   - `ind_niftyrealtylist.csv` -> `NSE_NIFTY_REALTY`
   - `ind_niftypsubanklist.csv` -> `NSE_NIFTY_PSU_BANK`
   - `ind_nifty50list.csv` -> `NSE_NIFTY_50` (broad-market fallback)

2. For each F&O symbol, take the FIRST sector match in the priority order
   `BANK -> IT -> AUTO -> FMCG -> PHARMA -> METAL -> ENERGY -> FIN_SERVICE
   -> REALTY -> PSU_BANK -> NIFTY 50`. This is deterministic and matches the
   task spec exactly.

3. Symbols with no sectoral match fall back to `NSE_NIFTY_50`.

## Coverage

Total symbols mapped: **153 / 153** (entire F&O liquid universe).

| Sector | Count |
|---|---|
| `NSE_NIFTY_PHARMA` | 20 |
| `NSE_NIFTY_ENERGY` | 17 |
| `NSE_NIFTY_FMCG` | 16 |
| `NSE_NIFTY_METAL` | 15 |
| `NSE_NIFTY_AUTO` | 15 |
| `NSE_NIFTY_FIN_SERVICE` | 15 |
| `NSE_NIFTY_50` (fallback) | 14 |
| `NSE_NIFTY_BANK` | 14 |
| `NSE_NIFTY_REALTY` | 10 |
| `NSE_NIFTY_IT` | 10 |
| `NSE_NIFTY_PSU_BANK` | 7 |
| **Total** | **153** |

## Fallback to NIFTY 50 (no specific sector found)

13 symbols are constituents of NIFTY 50 only (cross-sector heavyweights —
infrastructure, hospitality, paints, telecom, etc.):

`ADANIPORTS, APOLLOHOSP, ASIANPAINT, BEL, BHARTIARTL, ETERNAL, GRASIM, INDIGO, LT, MAXHEALTH, TITAN, TRENT, ULTRACEMCO`

1 symbol is unmatched even in NIFTY 50 and was tagged `NSE_NIFTY_50` by spec rule:

`GUJGASLTD` — Gujarat Gas (Oil & Gas index member but not Nifty Energy or Nifty 50).

## Overlap caveats

42 symbols are constituents of multiple sector indices. The priority-order rule
took the FIRST match. Notable cases worth understanding when reading sanity
results:

- **Banks vs Financial Services overlap (5 names)** — `AXISBANK, HDFCBANK,
  ICICIBANK, KOTAKBANK, SBIN` are in both `NIFTY_BANK` and `NIFTY_FIN_SERVICE`.
  Priority assigns them to `NIFTY_BANK` (correct: bank-specific beta dominates).
- **PSU Banks** — `BANKBARODA, CANBK, PNB, UNIONBANK, SBIN` are in both
  `NIFTY_BANK` and `NIFTY_PSU_BANK`. Priority assigns to `NIFTY_BANK` (broader
  bank index has more signal coverage; PSU rotation is a stricter sub-thesis
  and not the round-4 hypothesis).
- **NIFTY 50 overlap (~36 names)** — most large caps (RELIANCE, TCS, INFY,
  MARUTI, etc.) are also in NIFTY 50. The sectoral assignment wins per priority
  rule, which is what the rotation-vs-sector hypothesis needs.
- **ADANIENT** — currently mapped to `NIFTY_METAL` (it is a Metals & Mining
  member of NIFTY 50). This is sector-correct per NSE classification.

## Download status

| File | Source | Status |
|---|---|---|
| 9 sector CSVs | already cached in `assets/` | OK |
| `ind_niftyenergylist.csv` | `archives.nseindia.com/content/indices/` | OK (niftyindices.com host timed out, archives.nseindia.com served) |
| `ind_niftyfinservicelist.csv` (per task spec) | n/a | 404 — file does not exist by that name; the cached `ind_niftyfinancelist.csv` IS the Financial Services index (verified by reading constituents) |

No manual paste required. All 11 required indices are covered.

## Files written

- `assets/ind_niftyenergylist.csv` — fresh download (40 constituents)
- `assets/stock_sector_map.json` — 153 entries, sorted, JSON, human-readable
- `tools/build_stock_sector_map.py` — re-runnable builder
- `specs/2026-05-06-sub-project-9-stock-sector-mapping.md` — this file

## Re-run

```
.venv/Scripts/python tools/build_stock_sector_map.py
```

Idempotent. Reading any local CSV refresh and rerunning regenerates the JSON.
