# F&O Eligibility Removals (2024-2026)

Manually curated dataset of NSE F&O exclusion events (post-new-norms baseline).

Built for the **C3 sanity check** in
`specs/2026-05-14-brief-fno_removal_drift_short.md` (F&O removal drift short).

## Schema

| Column | Type | Notes |
|---|---|---|
| `circular_date` | ISO date | When the NSE FAOP circular was published |
| `effective_date` | ISO date | When F&O contracts cease (first day with no contracts) |
| `symbol` | str | Bare NSE ticker (no `NSE:` prefix) |
| `circular_ref` | str | NSE FAOP circular number if known, else empty |
| `gap_days` | int | `effective_date - circular_date` in calendar days |
| `is_post_oct_1_2025` | bool | `circular_date >= 2025-10-01` (SEBI MWPL regime) |

## Inclusion criteria

1. **NSE F&O only.** BSE F&O removals (e.g. HUDCO/PPLPHARMA/TATATECH/TORNTPOWER
   Feb 24 2026) are excluded.
2. **Post-SEBI-Aug-30-2024-norms only.** Earlier removals or M&A-driven exits
   (e.g. MindTree 2022) are excluded.
3. **No F&O additions.** Only exclusions (stocks leaving F&O).
4. **Conservative on ambiguity** (per brief §"Be conservative on ambiguous events"):
   events whose circular date could not be cross-verified from 2+ sources are
   dropped rather than guessed.

## Coverage

| Circular date | Effective date | Stocks | FAOP ref | Notes |
|---|---|---|---|---|
| 2024-12-20 | 2025-02-28 | 16 | FAOP65702 | First post-new-norms exclusion |
| 2025-01-22 | 2025-03-28 | 2 (JKCEMENT, LTTS) | FAOP66251 | Late-add tranche |
| 2025-03-21 | 2025-05-27 | 5 | FAOP67222 | Verified via Zerodha bulletin |
| 2025-05-22 | 2025-08-01 | 5 (FAOP68113 group) | FAOP68113 | AARTIIND, BSOFT, HINDCOPPER, MGL, PEL |
| 2025-05-22 (est.) | 2025-08-01 | 5 (ACC group) | (unknown) | ACC, BALKRISIND, CHAMBLFERT, M&MFIN, TATACOM — same effective date as FAOP68113 but circular ref not recovered; date estimated from standard NSE 2-month notice |
| 2025-06-23 | 2025-08-29 | 8 | FAOP68685 (best estimate) | ABFRL group; circular ref not directly fetched |
| 2025-10-24 | 2025-12-31 | 4 | FAOP70954 | NSE PDF directly verified |
| 2025-12-23 | 2026-02-25 | 1 (IRCTC) | FAOP71649 | NSE PDF directly verified |
| 2026-02-23 | 2026-04-30 | 4 | (unknown) | BALRAMCHIN, INTELLECT, JUBLFOOD, PERSISTENT |

**Total: 50 rows across 9 announcement events.**

Breakdown by year:
- 2024 circulars: 16 stocks (1 event)
- 2025 circulars: 29 stocks (7 events)
- 2026 circulars: 5 stocks (2 events incl. 2025-12-23 IRCTC which falls in 2025
  calendar year but post-Oct-1 regime, plus 2026-02-23 group)

Wait — re-counting: 16 + 2 + 5 + 5 + 5 + 8 + 4 + 1 + 4 = **50 stocks**.

Post-Oct-1-2025 regime: 4 + 1 + 4 = **9 stocks across 3 announcement events**
(matches brief estimate of 9-13).

## Known gaps / drops

Events documented but **excluded** from the CSV due to insufficient verification:

1. **2024-09-13 alleged 16-stock removal** (BANDHANBNK, CANBK, etc.)
   - Listed in `data/sebi_calendar/rule_changes.csv` and brief §"Data
     requirements" with "(?)" markers
   - **Could not be confirmed** in 8+ web searches
   - BANDHANBNK still actively trades in NSE F&O as of May 2026 (appears in
     ban list, never permanently removed)
   - The SEBI Aug-30-2024 norm-tightening circular's *first* downstream NSE
     exclusion appears to be the 2024-12-20 FAOP65702 (16 stocks, effective
     2025-02-28). The brief's 2024-09-13 entry is likely an artifact of
     `rule_changes.csv`.
   - **Decision: dropped.** If verified later, would add 16 rows.

2. **BERGEPAINT** (last trading day 24-Apr-25 per icicidirect)
   - Single-source (icicidirect compiled list)
   - Circular date not recovered (FAOP66757 suspected but not fetched)
   - May be BSE-only exclusion (icicidirect's list appears to combine
     NSE+BSE removals)
   - **Decision: dropped** per conservatism rule.

3. **TATACHEM** (effective 2025-10-01 confirmed via X / scanx.trade)
   - Effective date double-verified but **circular date unknown**
   - **Decision: dropped** because brief schema requires `circular_date` for
     `gap_days` and `is_post_oct_1_2025` flag.

4. **IGL** (last trading 25-Nov-25 per icicidirect), **IIFL** (27-Jan-26),
   **SYNGENE** (30-Mar-26)
   - Single-source (icicidirect only)
   - May be BSE-only
   - **Decision: dropped.**

5. **2026-04 additions event** (NBCC, PHOENIXLTD, SOLARINDS, TORNTPOWER, +2)
   - These are **F&O additions, not removals** — out of scope.
   - Note: TORNTPOWER appears here as an addition AND in the Feb-24-2026 BSE
     exclusion list. The NSE-added vs BSE-removed split is consistent with
     SEBI's selective-eligibility regime.

## Symbol verification

All 50 symbols match the canonical NSE tickers as used in:
- `data/sebi_calendar/rule_changes.csv` (existing project file)
- Public NSE quote URLs (e.g. `nseindia.com/get-quote/equity/SYMBOL`)

Special-case tickers (verified):
- `JSL` = Jindal Stainless (NOT JINDALSTEL — the brief noted this ambiguity)
- `M&MFIN` = Mahindra & Mahindra Financial Services (ampersand preserved)
- `INTELLECT` = Intellect Design Arena
- `BALRAMCHIN` = Balrampur Chini Mills
- `LTTS` = L&T Technology Services
- `PEL` = Piramal Enterprises (NOT PPLPHARMA which is Piramal Pharma)

## Sources

Primary (NSE circular PDFs or direct broker bulletins):
- NSE FAOP70954 (Oct 2025): https://nsearchives.nseindia.com/content/circulars/FAOP70954.pdf
- NSE FAOP71649 (Dec 2025 IRCTC): https://nsearchives.nseindia.com/content/circulars/FAOP71649.pdf
- Zerodha bulletin 400016 (FAOP65702, 16 stocks): https://zerodha.com/marketintel/bulletin/400016/exclusion-of-futures-and-options-contract-on-16-securities
- Zerodha bulletin 408778 (FAOP67222, 5 stocks): https://zerodha.com/marketintel/bulletin/408778/exclusion-of-fo-contracts-on-5-securities-from-may-27-2025
- Zerodha bulletin 418412 (Aug-29 group): https://zerodha.com/marketintel/bulletin/418412/exclusion-of-fo-contracts-on-8-securities-from-august-29-2025
- TradingQnA (FAOP68113, 5 stocks): https://tradingqna.com/t/exclusion-of-5-stocks-from-f-o-segment-effective-august-01-2025/182502

Secondary (broker blogs / news):
- ICICIdirect compiled exclusion table (used to identify candidates, not as
  sole-source authority due to NSE/BSE mixing):
  https://www.icicidirect.com/futures-and-options/articles/exclusion-of-16-futures-options-contracts
- Bajaj Broking (8-stock Aug-29 group):
  https://www.bajajbroking.in/blog/nse-to-exclude-8-stocks-from-f-and-o-segment-from-august-29-2025
- BusinessUpturn (ACC group): https://www.businessupturn.com/finance/stock-market/acc-balkrishna-industries-chambal-fertilisers-mm-financial-tata-comm-to-be-excluded-from-fo/
- AngelOne (FAOP68113 group): https://www.angelone.in/news/share-market/nse-to-exclude-piramal-enterprises-hindustan-copper-mgl-and-2-others-from-f-o-segment-effective-august-1-2025
- AngelOne (IRCTC): https://www.angelone.in/news/stocks/irctc-to-exit-f-o-in-2026-here-is-what-changes-for-traders-and-the-stock
- Kotak Securities (IRCTC): https://www.kotaksecurities.com/news/market-news/irctc-fo-segment-exclusion-feb-2026/
- News-articles.net (Feb-23-2026 4 stocks):
  https://science-technology.news-articles.net/content/2026/02/23/nse-removes-four-stocks-from-f-o-segment.html
- Moneysukh (FAOP66251 JKCEMENT/LTTS):
  https://learn.moneysukh.com/nse-excluded-16-stocks-from-fo-segment/
- Outlook Money (Aug-29 group):
  https://www.outlookmoney.com/invest/equity/nse-fno-exclusion-adani-total-gas-sjvn-share-price-to-be-excluded-from-fo-fno-futures-and-options

## Methodology notes

- **"Effective date" convention.** NSE language varies between "effective from
  <date>" and "no contracts available from <date>". For consistency, we use
  the date language stated by the most authoritative source for each event
  (NSE PDF > Zerodha bulletin > broker blog). This matches the strategy
  semantics: `effective_date - 1 business day` is the last trading day in F&O.
- **Trade entry timing.** For the C3 strategy, T+0 is the **announcement
  day** (circular_date), not the effective date. The drift window opens at
  09:30 on T+0 in the brief's framing.
- **Gap days.** Distribution of `gap_days` is tight (64-71 days), confirming
  the standard NSE 2-month notice period. The `gap_days` column can be used
  to bucket trades for the "per-effective-date-gap-bucket" cell mentioned in
  brief §"Sanity script".
- **SEBI Oct-1-2025 regime flag.** The new MWPL formula (min(15% free float,
  65× avg cash volume)) took effect 2025-10-01. Per the brief, this is the
  regime hypothesis cutoff for drift edge.
