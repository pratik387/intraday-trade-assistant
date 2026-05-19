# Brief: `fno_removal_drift_short`

**Sub-project:** Research / post-SEBI-Oct-1-2025 edge candidates (Candidate 3)
**Status:** DRAFT — pre-registration (no data peeked yet)
**Branch:** `research/post-sebi-edge-setups`
**Date:** 2026-05-14
**Predecessor:** `specs/2026-05-14-research-post-sebi-edges.md` (Candidate 3 row)

## Mechanism (one sentence)

NSE publishes quarterly F&O eligibility-cycle circulars (`FAOP*`) listing single
stocks that have failed the SEBI Aug-30-2024 entry/exit criteria for 3
consecutive months; these names lose institutional hedge demand and absorb a
forced cash-segment unwind from arbitrage/calendar-spread desks, producing a
predictable negative drift in the **announcement-day-and-after** window
(arguably starting 1-3 days BEFORE the circular as the quarter-sigma data is
publicly computable on the 16th of each month).

## Direction

**SHORT-only.** Three independent reasons:

1. **Asymmetric flow mechanic.** Removal triggers one-way selling pressure from
   (a) arb desks unwinding cash-vs-futures basis, (b) market-makers closing
   options books, (c) F&O-only mandates rebalancing OUT of the name. No
   symmetric "buy on removal" institutional flow exists.
2. **Empirical peer-reviewed evidence (negative side dominates).** Kumar &
   Chakrabarti 2024 (Indian Economic Journal): excluded firms underperform the
   market by **-4.07%** in event window; -5/+5 CAR mean **-2.82%** / median
   **-2.34%**. Inclusions show positive but smaller/less significant reactions
   (asymmetry-of-magnitude consistent with our existing portfolio's
   short-only bias).
3. **Project-wide rule (sub-9 §3.2 #5).** All three surviving setups
   (`gap_fade_short`, `circuit_t1_fade_short`, `delivery_pct_anomaly_short`)
   are SHORT-only on the Indian-intraday losing flow side. The long side of
   F&O removals has no peer-reviewed support and would re-create the
   `gap_and_go_continuation` long-bias failure mode.

## Regulatory dependencies

- **regulatory_sensitivity:** `rule_dependent`
- **depends_on:** `["single_stock_FO", "F&O_speculation"]`
- **reasoning:**
  - The MECHANISM (forced cash-side unwind on F&O eligibility loss) exists
    under any F&O eligibility regime. It is NOT a `rule_creating` setup
    (which would only work post-Oct-1-2025).
  - However, the SIZE of the mechanism is tied to the SEBI Aug-30-2024
    threshold step-change (MWPL Rs.500cr → Rs.1500cr, MQSOS Rs.25L →
    Rs.75L, ADDV Rs.10cr → Rs.35cr). That step-change made many more stocks
    fall out per quarter, increasing both sample size AND the size of
    cash-side unwind per event.
  - The 2025-10-01 critical rule change (MWPL formula tightened further) is
    LISTED in our `depends_on` because it acts in the same direction (more
    removals, more unwind), but does NOT break the mechanism. So the
    rule_dependent tag is correct, not `rule_creating`.
  - **Regime-break gate**: validation windows must NOT straddle either
    2024-08-30 (announcement of new norms) or 2025-10-01 (MWPL tightening)
    when ingesting events. Use 2024-09-13 (first post-norm quarter removal)
    as the earliest **comparable** sample point.

## Differentiation from retired `nifty500_deletion_short`

Both setups are SHORT-only event-driven forced-flow trades. Differences:

| Aspect | nifty500_deletion_short (RETIRED 2026-05-05) | fno_removal_drift_short (this) |
|---|---|---|
| Flow source | Passive index ETF/MF rebalance (₹9.7L cr AUM) | F&O arb desks + market-makers + F&O-mandate funds unwinding cash hedges |
| Window | Pre-effective T-1 single intraday session | T-3 to T+10 multi-day, peak at T-1 to T+2 |
| Forced-seller concentration | At/near effective close (VWAP/TWAP) | At circular publication + over 2-3 month rolloff to last-expiry-month |
| Peer-reviewed CAR | Marisetty 2025: stronger initial neg, then +1.17% reversal | Kumar/Chakrabarti 2024: -4.07% over event window; -2.82% on -5/+5 |
| Retired reason | Index-arb / CNC infra needed; retail-MIS-single-day didn't fit | (TBD — this brief tests whether MIS-single-day captures enough) |
| Indian quant precedent | ZERO (the retire reason) | Modest (Kumar/Chakrabarti is published India-specific) |
| Event count post-SEBI window | ~80-120/yr theoretical, semi-annual concentration | ~16-24/yr actual (quarterly, 4-16 per circular) |

**Why this isn't a re-tread of the retired setup:**
- F&O removal forced flow comes from **arb desks** (not passive funds).
  Arb-desk flow concentrates at the moment basis goes uneconomical, which is
  RIGHT AT the circular announcement (not at effective date). That's an
  intraday-detectable, single-day event window — fits MIS infrastructure.
- Index deletions have a 28-day announcement-to-effective gap with VWAP/TWAP
  execution. F&O removals have a 2-3 month rolloff but **the announcement
  itself is the priced-in moment**; the multi-month contract rolloff is
  already known mechanic, not novel selling.
- The retire reason for `nifty500_deletion_short` (no precedent in Indian
  retail/pro quant resources) does NOT apply here: F&O exclusion event
  trading is documented in Kumar/Chakrabarti 2024 with explicit CAR
  measurement around announcement date.

## Cells (pre-registered — must NOT change after seeing data)

Pre-registration locked 2026-05-14. Any cell change after seeing post-SEBI
data invalidates this brief and forces re-promotion to a new pre-registration.

### Primary cell (single-entry MIS variant)

- **Trigger:** stock named in an NSE `FAOP*` circular adding it to the
  "stocks excluded from F&O segment" list. Stock is on T+0 the announcement
  day (publication date of the circular).
- **Universe:** any cap segment EXCLUDING `micro_cap` (short-side liquidity
  defense). Cash-market T-30 ADV >= Rs.5 cr.
- **Entry timing:** T+0 close of next 5m bar AFTER 09:25 IST (i.e., 09:30
  bar close). Why 09:30:
  - Circulars are typically published EOD T-1 or pre-open T+0
  - 09:15-09:25 is opening auction noise + DPR-flex re-anchor
  - 09:30 is the first stable price after retail-attention has surfaced the news
  - Aligns with Kumar/Chakrabarti's "announcement day" definition (event date
    is the trading day on which the circular's contents become public)
- **Direction:** SHORT
- **Confirmation gates at 09:30:**
  - 5m bar's close < bar's open (red bar on entry candle)
  - Volume on 09:15-09:30 bars > 1.5× average 30-day 5m volume (news priced)
- **Hard SL:** entry × (1 + 1.5%) OR T+0 day-high-so-far × 1.005, whichever
  is lower. Min stop distance 1.0%.
- **T1 (50% qty):** entry × (1 - 1.0%)
- **T2 (50% qty):** entry × (1 - 2.0%)
- **Time stop:** 15:10 IST
- **Latch:** one fire per (symbol, circular_date). No re-entry intraday.

### Secondary cell (T-1 pre-announcement variant — exploratory only)

Per the roadmap mechanism statement ("drift starts 2-3 days BEFORE the
official announcement"), an aggressive variant entering T-1 (the trading
session immediately before the circular date) when:
- Stock has been on the "watch list" (failed 1 or 2 of 3 consecutive months
  per the publicly-computable quarter-sigma)
- Same SL/T1/T2 structure

**This cell is exploratory only.** Pre-announcement leak detection is an
unverified hypothesis. Sanity must run BOTH cells and report independently.
The primary cell is the production candidate; the secondary is research-only.

## Falsification thresholds (pre-registered)

Setup is **retired and not eligible for revival under this brief** if ANY of
the following triggers on Discovery (Oct-1-2025 → 2026-05-14):

1. **Sample size below floor:** n < 20 events on primary cell over the entire
   post-SEBI window. (Acknowledged THIN — see sample-size section. The hard
   floor is dropped below the standard n=30 only because the event is
   inherently rare; with n<20 there is no statistical hope.)
2. **NET PF < 1.10** on primary cell (using full post-Apr-2026 STT fees +
   MIS-leveraged qty + min-stop guards).
3. **WR < 40%** — fade setups typically run WR 30-45% with asymmetric R; if
   WR drops below 40%, the asymmetric-R isn't compensating.
4. **NET Sharpe (per-trade) <= 0** on the post-Apr-2026 STT regime sub-sample.
5. **Direction asymmetry collapses:** if a hypothetical LONG-side cell on the
   same trigger has comparable PF, the mechanic isn't directional and the
   thesis is wrong.
6. **Cell-shift after seeing data:** if the only cell that passes is one that
   changes ANY parameter declared in §"Cells" above, the setup is retired
   per sub-9 cell-mining rule (Common Failure Mode #3).
7. **Single-day MIS doesn't capture enough:** if T+0 single-session intraday
   PF < 1.10 but a hypothetical multi-day swing version (T+0 to T+10) shows
   PF >= 1.30, the answer is **"real edge, wrong infra"** — retire from
   intraday MIS catalog and revisit when CNC/SLB infra is built. Do not
   loosen the brief to a non-MIS variant within this brief.
8. **Post-Apr-2026 STT decay observed:** if the strategy PF in Apr-May 2026
   (post-STT hike) is materially worse than Oct-2025-Mar-2026 (pre-STT
   hike), the post-STT regime is not friendly. Report both sub-window PFs
   and gate on the worse one.

## Data requirements

### F&O removal events 2023-2026 (best-effort list)

Verified events from web research (announcement_date → effective_date):

| Effective date | Announcement | # stocks | Notable names | Source |
|---|---|---|---|---|
| 2022-11-22 | ~2022-10 | 1 | MindTree (M&A driven, anomalous) | NSE archives |
| 2024-09-13 (?) | 2024-09-12 (?) | 16 | BANDHANBNK, CANBK, etc. (per `rule_changes.csv`) | NSE/FAOP/2024-09 |
| 2025-02-28 | 2024-12-23 | 16 | ABBOTINDIA, ATUL, BATAINDIA, CANFINHOME, COROMANDEL, CUB, GNFC, GUJGASLTD, INDIAMART, IPCALAB, LALPATHLAB, METROPOLIS, NAVINFLUOR, PVRINOX, SUNTV, UBL | NSE FAOP65702/66251/66757/67222/67710; bajajbroking.in 2024-12-23 |
| 2025-08-29 | ~2025-06-24 | 8 | ABFRL, ATGL, CESC, GRANULES, IRB, JINDALSTEL?, JSL, POONAWALLA, SJVN | bajajbroking.in 2025-06-24 |
| 2025-12 expiry → 2026-01 series gone | 2025-10-24 | 4 | CYIENT, HFCL, NCC, TITAGARH | niftytrader.in / NSE FAOP 2025-10-24 |
| 2026-02-25 | 2025-12-23 | 1 | IRCTC (Navratna PSU) | NSE 2025-12-23; reported by Angel One / Goodreturns |
| 2026-02-XX (other) | 2026-02-23 | 4 | (4 stocks per news-articles.net) | NSE 2026-02-23 |
| 2026-04 | 2026-03 | 0 removals; **6 additions** | NBCC, PHOENIXLTD, SOLARINDS, TORNTPOWER, +2 | NSE/FAOP/2026-03 |

**Total verifiable removal events in post-SEBI-Oct-1-2025 window:**
- 2025-10-24 announcement → 4 stocks
- 2025-12-23 announcement → 1+ stocks (IRCTC at least)
- 2026-02-23 announcement → 4 stocks
- (More may exist in non-headline-grabbing FAOP circulars)
- **Confirmed n ≈ 9-13 stocks across 3 announcement events**

**Total verifiable removal events 2024-09-13 onwards (post-new-norms baseline):**
- 2024-09-13 → 16
- 2024-12-23 → 16
- 2025-06-24 → 8
- 2025-10-24 → 4
- 2025-12-23 → 1
- 2026-02-23 → 4
- **Total ≈ 49 stocks across 6 announcement events Aug-2024 to May-2026**

### Data infrastructure needed

1. **Manually curated CSV** `data/fno_eligibility/removals_2024_2026.csv` with
   columns: `circular_date`, `effective_date`, `symbol`, `circular_ref`,
   `gap_days`, `is_post_oct_1_2025` (bool). To be hand-built from web
   research + NSE archive scrape.
2. **Existing 5m feathers** for symbols in (1) — already on disk.
3. **Existing daily bars** for PDH/PDC anchors.
4. **No new live-pipeline data** required for sanity. If the setup ships,
   live mode needs a daily NSE circular scraper polling FAOP feed
   (lightweight, EOD-only).

## Implementation sketch

### Sanity script (only piece built pre-approval)

`tools/sub9_research/sanity_fno_removal_drift_short.py`:
1. Hand-curate the removals CSV (above).
2. For each (symbol, circular_date), load T+0 09:15-15:30 5m bars.
3. Apply confirmation gates at 09:30 (red candle, vol spike).
4. Simulate SHORT entry at 09:30 close, exits per primary cell rules.
5. Compute NET PF using `tools/sub7_validation/build_per_setup_pnl.py:calc_fee`
   with `is_post_apr_2026=True` for post-Apr-2026 events.
6. Report:
   - Aggregate PF, n, WR, Sharpe
   - Pre-STT-hike vs post-STT-hike sub-windows separately
   - Primary cell vs secondary cell separately
   - Per-cap-segment breakdown
   - Per-effective-date-gap-bucket (some removals announced 2 months out;
     some 3 weeks; does the gap matter?)

**Sanity decision per §3.3:**
- PF >= 1.30 AND n >= 20: STRONG PROCEED to detector implementation
- PF 1.10-1.29 AND n >= 20: MARGINAL — extend to swing variant (multi-day
  exit) as side-research, do not ship intraday
- PF < 1.10 OR n < 20: RETIRE under this brief; document as negative knowledge

### Post-approval detector path (only if sanity passes)

1. `data/fno_eligibility/removals.parquet` — automated NSE FAOP scraper.
2. `services/fno_eligibility_loader.py` — load events keyed by date.
3. `structures/fno_removal_drift_short_structure.py` — detector. Cross-day
   state: at T+0 09:15, check if symbol in today's removal list.
4. Unit tests including regime-break-detector pre-flight.
5. Config block `setups.fno_removal_drift_short` with full pre-registered
   cell parameters.

## Risks / open questions

### 1. n=8-16 sample size in post-SEBI window — how to handle?

The roadmap notes this is THIN. Three options:

**Option A (preferred): Extend Discovery back to 2024-09-13.**
The SEBI Aug-30-2024 norm step-change is itself a regime break for this
setup (more stocks now fall out per quarter). Pre-Aug-2024 removals operate
under a different threshold regime. Post-Aug-30-2024 events are comparable
to each other. Using 2024-09-13 → 2026-05-14 gives ~49 events — workable.

Risk: 2025-10-01 MWPL further-tightening is INSIDE this window. Mitigation:
gate on the depends_on tag in `services.regime_break_detector` and report
pre-Oct-2025 PF / post-Oct-2025 PF / combined as 3 separate cells. If any
single sub-window has PF < 1.10, the regime-specific edge is wrong.

**Option B: Accept n=8-16, lower confidence.**
Run the sanity only on post-Oct-1-2025 events. Report n explicitly. If
sanity PF is high but n=8-16, do NOT ship without paper-trade Holdout
producing additional n.

**Option C: Pool with adjacent event types.**
If the mechanism is "forced cash-side unwind on F&O eligibility loss",
related events that share the mechanism include: (a) F&O ban-list entries
under post-Nov-2025 intraday monitoring (mechanism: hedge demand drops
intraday) — but this is Candidate 2's territory, separate setup. (b) Bulk
volatility spikes that trigger SEBI ASM/GSM transitions — different
mechanism, different data, do not pool.

**Decision:** Sanity script implements Option A AND Option B in parallel.
The two PF readings are compared. If Option A passes but Option B fails on
small n with high PF, ship under Option A's wider window and gate on
production with paper-trade Holdout to accumulate post-Oct-1 n.

### 2. Intraday vs swing trade

The roadmap explicitly asks. Honest answer:

- **Peer-reviewed CAR is multi-day.** Kumar/Chakrabarti's -2.82% is on -5/+5
  window. Single-intraday MIS captures at most 1-2 days of that drift.
- **If the academic effect is mostly the announcement-day-1 + day-2 move,
  intraday MIS captures most of it.** If the effect is uniformly distributed
  over 5 days, intraday MIS captures ~1/5 = -0.56% per session — barely
  above the post-Apr-2026 fee break-even.
- **The 09:30 confirmation gate (red candle + vol spike) is designed to fire
  only on days where the announcement has IN FACT been priced overnight.**
  This concentrates the trade into the highest-density-drift day. If the
  drift is uniformly distributed, the gate fails to fire and the setup
  produces few trades — falsifier #1 catches this.
- **Falsifier #7** explicitly handles the "real edge, wrong infra" outcome:
  if T+0 single-session PF is below break-even but multi-day PF would be
  attractive, we retire from this brief and revisit when CNC/SLB infra
  exists — not loosen this brief to chase the wrong infrastructure.

**Recommendation for sanity:**
- Primary mode = intraday MIS T+0 (as designed)
- Side-research mode = simulate hypothetical multi-day exit at T+5 close
  with same SL/entry, no fees beyond entry+exit (assume CNC)
- Report both PFs. If multi-day is materially better, write up as future
  CNC/SLB-infra setup proposal (don't ship as intraday).

### 3. Pre-announcement leak detection (secondary cell)

The roadmap says "drift starts 2-3 days BEFORE". Real test: secondary cell
attempts T-1 entry. Mechanism would require either:
- Insider/leak (illegal but possible — quarter-sigma data publicly computable
  on 16th of each month per NSE selection-criteria doc → professional algo
  desks compute the exit candidates BEFORE NSE publishes the circular)
- Public computability — anyone with NSE bhavcopy + cash-volume data can run
  the 3-consecutive-month-failure check themselves, with NO insider info,
  and forecast the next circular's list with high accuracy

If secondary cell shows meaningful edge, that's an indication of
public-computability arbitrage — interesting research finding regardless of
shipping decision. Sanity reports both cells independently.

### 4. Decay risk

Two sources of decay:
- **Mechanism decay:** more pros computing the same quarter-sigma forecast
  → faster pricing-in → less drift available to harvest. Per Kumar/
  Chakrabarti, the effect was observable 2001-2020; if it's surviving in
  post-Aug-2024 data, it's persistent.
- **Regime decay:** further SEBI tightening of MWPL formula creates more
  removals (helps sample) but also signals to pros that the list will keep
  growing (faster pricing-in, hurts edge). Net direction unclear.

Mitigation: explicit pre-STT-hike vs post-STT-hike PF reporting in sanity
will catch the worst-case decay scenario before shipping.

### 5. Special-case events that contaminate the universe

- **M&A-driven removals** (e.g., MindTree 2022) are NOT mechanism-driven
  exits; the stock is removed because the merger eliminates the security
  itself. These produce different price action (typically GAP, not drift).
  Exclusion: filter out symbols with concurrent M&A/merger announcements.
- **PSU mandate-driven holders** (e.g., IRCTC 2026) may see government-fund
  buying offsetting F&O-side selling. Flag in sanity, do not pre-filter
  unless data shows the offset is reliable.
- **Re-eligibility candidates** within 12 months of prior removal cannot
  re-enter (SEBI rule). Excludes some symbols from the candidate pool but
  shouldn't affect the short-side mechanic.

## References

### Academic (peer-reviewed)

1. **Kumar, R. & Chakrabarti, P. (2024). "Price Impact of Derivatives
   Listing and Delisting: Evidence from India." *The Indian Economic
   Journal.*** Event study on Indian derivatives delisting 2001-2020.
   Mean CAR -2.82%, median -2.34% on -5/+5 window. Excluded firms
   underperform by -4.07% in event window. Liquidity reduction and price
   efficiency decline identified as mechanism channels.
   URL: https://journals.sagepub.com/doi/abs/10.1177/00194662221137261

2. **NYU Stern working paper (Sundaram et al?) — "Do Derivatives Matter?
   Evidence From A Policy Experiment."** Used the SEBI 51-stock delisting
   order as a natural experiment. Found volatility does NOT change post-
   delisting (contrary to regulator expectation), but price effects are
   real.
   URL: https://www.stern.nyu.edu/sites/default/files/assets/documents/Do%20Derivatives%20Matter.pdf

3. **Sehgal et al. (referenced in `circuit_t1_fade_short`)** — Indian
   momentum/reversal evidence. Generally supports asymmetric direction
   findings in Indian equity microstructure.

### NSE / SEBI primary sources

4. **SEBI circular SEBI/HO/MRD/MRD-PoD-2/P/CIR/2024/116, dated 2024-08-30** —
   revised F&O entry/exit eligibility norms. MQSOS Rs.25L → Rs.75L; MWPL
   Rs.500cr → Rs.1500cr; ADDV Rs.10cr → Rs.35cr. Three-consecutive-month
   failure triggers exit, one-year ineligibility cooldown.
   URL: https://www.sebi.gov.in/sebi_data/meetingfiles/jul-2024/1720932411247_1.pdf

5. **NSE selection-criteria page.** Confirms quarter-sigma calculation on
   16th of each month, rolling 6-month windows.
   URL: https://www.nseindia.com/static/products-services/equity-derivatives-selection-criteria

6. **SEBI/HO/MRD/TPD-1/P/CIR/2025/79, dated 2025-05-29 (effective 2025-10-01)** —
   MWPL formula tightening (critical regime break for `delivery_pct_anomaly_short`,
   relevant boundary for this setup).
   URL: https://www.5paisa.com/news/sebi-implements-stricter-fo-rules-from-october-1-to-strengthen-market-stability

### Industry confirmation sources (event timelines)

7. **bajajbroking.in 2024-12-23** — 16-stock Feb-28-2025 exclusion list.
   URL: https://www.bajajbroking.in/blog/nse-to-exclude-16-securities-from-f-and-o-contracts-by-feb-28
8. **icicidirect.com — exclusion-of-16-futures-options-contracts** — full
   ticker list with FAOP circular refs 65702/66251/66757/67222/67710.
   URL: https://www.icicidirect.com/futures-and-options/articles/exclusion-of-16-futures-options-contracts
9. **bajajbroking.in 2025-06-24** — 8-stock Aug-29-2025 exclusion list.
   URL: https://www.bajajbroking.in/blog/nse-to-exclude-8-stocks-from-f-and-o-segment-from-august-29-2025
10. **niftytrader.in 2025-10-24** — 4-stock Jan-2026 exclusion.
    URL: https://www.niftytrader.in/content/nse-excludes-four-stocks-from-fo-segment/
11. **angelone.in / goodreturns 2025-12-23** — IRCTC Feb-25-2026 exclusion.
    URL: https://www.goodreturns.in/news/irctc-shares-rise-despite-nse-removing-stock-from-f-o-list-from-february-25-2026-what-traders-need-t-1477644.html
12. **paytmmoney.com 2026-03** — Apr-2026 ADDITIONS (control/baseline for
    asymmetry: additions vs removals).
    URL: https://www.paytmmoney.com/blog/fo-segment-expansion-nse-adds-six-stocks-derivatives-april-2026/

### Internal cross-references

- `specs/2026-05-14-research-post-sebi-edges.md` — parent roadmap; Candidate 3
- `data/sebi_calendar/rule_changes.csv` — 2024-09-13 + 2025-10-01 entries
  must be honored by the regime-break detector pre-flight
- `specs/2026-05-03-sub-project-9-brief-nifty500_deletion_short.md` —
  RETIRED comparable setup; failure-mode lessons
- `specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md` — APPROVED
  template structure
- `docs/retired_setups.md` — Common Failure Modes #1-#8; falsifier design
  honors all eight

---

## Decision required (for the human reviewer)

User to indicate:
- [ ] APPROVED — proceed to sanity script + manual event CSV curation
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong

**Honest recommendation:** APPROVE for sanity check with the following
escalations made explicit upfront:
1. Sanity MUST run Option A (2024-09-13 → 2026-05-14 wide window) AND
   Option B (Oct-1-2025 → 2026-05-14 strict post-SEBI) in parallel.
2. Pre-STT-hike vs post-STT-hike PF must be reported separately.
3. Multi-day swing PF must be reported as side-research even though we
   ship only intraday MIS.

If sanity returns intraday PF < 1.10 but multi-day PF > 1.30, the answer
is "real edge, wrong infra" — retire from this brief, write up as a
future CNC/SLB-infra candidate. Do NOT loosen to multi-day under this brief.
