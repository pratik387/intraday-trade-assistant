# §3.3 Brief: `pledge_margin_call_cascade_short`

**Sub-project:** #9 (microstructure-first redesign) — Round-5 candidate
**Status:** DRAFT — pending §3.3 sanity-check feasibility (data + n-floor)
**Date:** 2026-05-07

---

## Predecessor / context

- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (round-1 shortlist)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (round-2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (round-3)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED — same SEBI/NSE-mechanical lineage; used as template)
- specs/2026-05-06-sub-project-9-brief-earnings_day_intraday_fade.md (APPROVED — same data-backfill-pattern lineage)

This is a **round-5 candidate** in the SEBI/NSE-mechanical-event lineage that produced both surviving setups so far (`gap_fade_short`, `circuit_t1_fade_short`). It deliberately avoids the round-1/2 failure mode of generic asymmetries; the trigger is a **codified disclosure event** (SEBI LODR Reg. 31 + Reg. 30) followed by a **codified margin-call mechanic** at the lender level.

---

## Asymmetry

**Name:** Indian high-pledge-stock margin-call cascade — forced lender liquidation under SEBI Reg 31 disclosure regime.

**Indian-specific mechanical source:**

- **SEBI LODR Reg. 31** mandates promoters disclose creation, invocation, or release of pledge on listed-company shares within **7 working days** to both stock exchanges. Reg. 31(4) further requires disclosure when aggregate pledge crosses **20%** or any **5% incremental** change beyond that threshold. Source: https://www.sebi.gov.in/legal/regulations/sep-2015/securities-and-exchange-board-of-india-listing-obligations-and-disclosure-requirements-regulations-2015_30884.html
- **SEBI LODR Reg. 30** mandates disclosure of "material events" — including creditor margin calls and lender-invoked-pledge events — promptly (within 24 hours of the event becoming known to the company).
- **Lender margin-call mechanic:** banks/NBFCs lending against pledged shares run a daily mark-to-market on collateral. When stock price falls beyond a contractually defined haircut/LTV trigger (typically 30-40% drawdown from pledge-creation price for high-LTV NBFC loans), the lender issues a margin call. If promoter cannot top up cash/collateral within 24-72 hours, the lender **invokes the pledge** and dumps the pledged shares into the open market — usually as a single block sale, often via off-market block window then continued open-market unwind across multiple sessions.
- **Cascading liquidation in low-float stocks:** when promoter pledge >50%, the **free-float** is structurally small. A forced lender sale into a thin float collapses price further, triggering additional lender margin calls in a daisy chain. This is the **multi-day cascade** documented in Vakrangee 2018 (-67% in 4 days), Manpasand Beverages 2018 (-50% in 2 days), Coffee Day Enterprises 2019 (-65% on the founder's death + pledge-invocation news), Reliance Capital / Reliance Naval 2019-20, and Future Retail 2020-21.

The exploitable asymmetry is the **T+0 / T+1 short** of a confirmed pledge-invocation or margin-call trigger: lenders are **price-insensitive forced sellers**; the only natural buyer at the post-trigger price is a value bottom-fisher, who waits multiple sessions. The order-flow imbalance is mechanical and one-sided.

## Participants

- **Forced sellers (alpha source):** banks, NBFCs (notably IIFL Wealth, Edelweiss, Avendus historical), and broker-margin desks holding the pledged shares as collateral. They are contractually obligated to liquidate; price-insensitive.
- **Defending side (failing):** the promoter, attempting to top up cash collateral or reverse-pledge to a new lender. Public statements from CFO / IR ("no margin call", "liquidity event imminent") are common but rarely succeed in halting the cascade once the first lender invokes.
- **Tactical predators (our cohort):** prop desks and PMSes that have read the SEBI Reg. 31 / Reg. 30 disclosure and short the stock T+0/T+1.
- **Retail (losing flow):** average-down buyers on the gap-down, "value-buying" the falling knife — exactly the losing flow our SHORT side absorbs.

## Persistence

Three structural reasons this should not arbitrage away in our holding period:

1. **SEBI mandate is codified.** Reg. 31 disclosure timing (7 working days for pledge events; 24 hours for lender invocation under Reg. 30) cannot be negotiated by the issuer. The disclosure event is forced, dated, and publicly indexed at NSE/BSE.
2. **Lender margin-call thresholds are codified at the bank/NBFC level.** RBI mandates LTV caps for loan-against-shares (currently 50% for non-bank lenders); when triggered, internal credit-committee policy is mechanical. Lenders cannot "wait out" volatility — RBI mandates same-day mark-to-market.
3. **Low-float operator-prone names cannot reflate quickly.** A high-pledge name with a depleted promoter balance sheet has no organic buyer absorbing the forced sale. The cascade duration (1-5 sessions in the case studies) is a function of float, not market mood.

These are SEBI/RBI-codified mechanics, not market-cycle-dependent. The edge does not require a particular volatility regime.

## Evidence (peer-reviewed / Indian-market specific)

1. **SEBI LODR Reg. 31 (Disclosure of Encumbered Shares)** — primary regulatory source. https://www.sebi.gov.in/legal/regulations/sep-2015/securities-and-exchange-board-of-india-listing-obligations-and-disclosure-requirements-regulations-2015_30884.html
2. **NSE corporate-filings page for pledged shares** (the practical disclosure feed): https://www.nseindia.com/companies-listing/corporate-filings-disclosures-of-pledged-shares
3. **BSE corporate-filings (analogous)**: https://www.bseindia.com/corporates/sastpledge.html
4. **NIBM working paper — "Promoter Share Pledging and Firm Outcomes in India"** (Singh & Singhvi, NIBM Pune, 2019). Documents that high-pledge Indian stocks exhibit (a) elevated tail-risk in returns, (b) significant negative announcement-effect on disclosure of pledge increases, (c) cascade-style price collapse on lender invocation. URL: https://www.nibmindia.org/static/working_paper/NIBM_WP66_Singh_Singhvi.pdf (working paper repository)
5. **IIM Bangalore working paper — "Share Pledging by Promoters and Firm Performance"** (Pandey & Prabhala, IIMB WP 2017). Cross-sectional Indian-equity evidence that high-pledge names have systematically worse drawdown profiles and abnormal-return responses to negative news. URL: https://www.iimb.ac.in/sites/default/files/2018-08/WP_No._545.pdf
6. **Mint / ET Markets case-study coverage** of historical cascades — Vakrangee (Jan 2018, Mint reporting), Manpasand Beverages (May 2018, ET Markets), Coffee Day (Jul-Aug 2019, multiple), Reliance Capital (Sep 2019), Future Retail (Mar 2020). Used as event-confirmation for the in-house event study.

(Retail-algo platforms intentionally excluded per round-5 mandate — this candidate is an institutional / regulatory mechanic, not a retail-published playbook.)

## Direction

**SHORT-only.** No LONG variant.

The mechanic is unidirectional: lender-forced selling produces a one-sided order-flow imbalance with cascading momentum, not mean-reversion. There is no symmetric "forced buyer" event in the Indian pledge-disclosure regime. Round-1 lessons strongly oppose adding a LONG side absent a directly-symmetric mechanic; this candidate respects that.

## Mechanic

**Setup name:** `pledge_margin_call_cascade_short`
**Side:** SHORT only.

**Sequence:**

1. **T-0 EOD detection** (post-15:30 IST):
   - Pull today's NSE+BSE pledge-disclosure filings (Reg. 31) and material-event filings (Reg. 30, lender-invocation flag)
   - Cross-reference against the **rolling pledge-ratio table** (per-symbol, last-known pledge %): if disclosed pledge ratio ≥ **50%** OR delta-pledge ≥ **+5pp** in this filing
   - Mark all small_cap + mid_cap names matching above
2. **T+0 trigger conditions** (combined gating, all must hold):
   - Disclosed event today (BMO or AMC) per step 1
   - Stock has fallen **≥5% over the past 5 trading sessions** (recent stress, lender margin call plausible)
   - T+0 day's volume ≥ **2× 20-day median** (institutional / lender flow signature)
3. **T+1 entry** at **09:30 IST** (15-min open auction settles, opening pump fades):
   - Confirmation: T+1 09:15-09:30 prints a **lower-low vs T+0 close** AND no upper-circuit hit
   - Entry price: 09:30 5m bar close
   - Latch: one fire per (symbol, T+1) — no re-entry
4. **Stop-loss:** **recent swing high** = max(T-3, T-2, T-1, T+0) intraday highs × 1.005. Min stop distance 1.5% of entry (qty-inflation guard).
5. **Targets** (tiered R-multiples, anchor `arithmetic_R`):
   - **T1 = 1R** — exit 50% qty
   - **T2 = 2R** — exit remaining 50%
   - **Breakeven trail** after T1 hit
6. **Time stop:** **14:30 IST** — capture the intraday cascade leg only; do not hold into the 14:30-15:15 MIS-unwind / EOD-positioning noise where reflex-bounce risk dominates. (Multi-day cascade exists per case studies, but MIS-only constraint forces single-session exit; sub-project bias is intraday-only.)

## Universe

**Allowed cap segments:** `small_cap` + `mid_cap` only.

- Excluded: `large_cap` (Nifty50 / Nifty100 names rarely hit pledge >50%; promoter holdings are diluted; lenders are larger and absorb shocks; mechanic does not apply)
- Excluded: `micro_cap` (insufficient SHORT-side liquidity; stock-borrow scarce or absent)
- Pre-filter: stocks where last-known **pledge ratio ≥ 50%** at any point in the rolling lookback (operator-prone names — exactly the population that cascades)

**Approximate universe size after pledge filter:** ~30-60 stocks (high-pledge names are rare; many names fall in/out of the >50% bucket as filings update).

## Active window

- **Setup formation:** T+0 EOD (post-15:30 IST disclosure scan)
- **Entry:** T+1 09:30 IST (single 5m bar)
- **Exit:** T1/T2/BE-trail, time-stop **14:30 IST**
- **Hold horizon:** max ~5h MIS, typical hold 60-180 min when T1/T2 hit

This sits between `gap_fade_short` (09:15-09:30) and the late-session gap left empty by current setups, with no overlap on universe (`circuit_t1_fade_short` is broader-universe; this is high-pledge subset).

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails (project-locked):**
   - PF < **1.10**
   - n < **30** trades over discovery window (round-3 floor for thin event candidates)
   - |WR delta| > 10pp (vs naive baseline)
2. **Sample size structurally too thin:** historical cascade events are rare (~10-30/year per Indian-market event-study coverage). Even over 3yr Discovery → ~30-90 events maximum. **Tight n.** If 2-yr Discovery yields <30 events post-gating, candidate is **retire-eligible-pre-data** without further engineering.
3. **Concentration check:** if a single event drives >40% of PnL (e.g., one Vakrangee-class blowup dominates), the edge is fragile — retire.
4. **Disclosure-lag contamination:** if NSE/BSE filings are systematically dated to the wrong session (filing posted Friday 18:00 but timestamped Monday), the T+0 / T+1 windows are misaligned. Sanity-check spot-validation of ≥20 historical events required.
5. **Borrow / SLB unavailability:** high-pledge cascading names often hit "ban-period F&O" or have no SLB borrow during a cascade — making SHORT impossible to execute. If >40% of historical events would have been unborrowable in practice, the live setup is non-tradeable; retire.

## Pre-coding sanity-check plan (mandatory per §3.3, BEFORE detector)

- **Pre-requisite:** historical pledge-disclosure event data must be backfilled (see §13). Without it, no sanity-check is possible.
- Use existing 2024 5m feathers + backfilled pledge-events parquet
- Identify trigger events: pledge-disclosure date (Reg. 31) OR material-event date (Reg. 30 lender-invocation) where pledge ratio ≥ 50% AND price fell ≥5% over preceding 5 sessions AND T+0 volume ≥ 2× 20d-median
- Simulate T+1 09:30 SHORT with swing-high stop and 1R/2R targets, time-stop 14:30
- Compute NET PF using existing Indian fee model
- **Decision per §3.3:**
  - n ≥ 30 AND PF ≥ 1.10 → **APPROVE** → proceed to detector code
  - n ≥ 30 AND 1.0 ≤ PF < 1.10 → **REVISE** (probe gating tightness)
  - n < 30 over 2-3yr discovery → **RETIRE-eligible-pre-data** (sample structurally too thin)
  - PF < 1.0 → retire

## Data engineering plan

### 13.1 Historical pledge-disclosure backfill (PRE-REQUISITE — must complete BEFORE sanity-check)

**FLAG: pledge data is NOT currently on disk.** This is analogous to the earnings-calendar pre-requisite for the `earnings_day_intraday_fade` brief. Without this backfill, the candidate cannot be sanity-checked and is **APPROVE-blocked**.

**Sources:**

- **Primary — NSE corporate filings (pledged shares):** https://www.nseindia.com/companies-listing/corporate-filings-disclosures-of-pledged-shares — public, no-auth, paginated JSON behind the page.
- **Primary — BSE corporate filings:** https://www.bseindia.com/corporates/sastpledge.html — public, scriptable.
- **Secondary — SEBI Reg. 31 SAST/LODR disclosures** (cross-reference for completeness): https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=6&smid=0
- **Tertiary — Reg. 30 material-event filings** (lender-invocation events, scraped from NSE corporate announcements with subject filter "pledge invocation" / "loan default" / "creditor margin call").

**Pattern to adapt:** `tools/earnings_calendar/fetch_earnings.py` (already built; same NSE cookie-bootstrap + half-month chunking + 5s polite-sleep approach). Pledge-disclosure endpoint is structurally identical — paginated JSON behind a corporate-filings page.

**Scope:**
- Universe: F&O 200 + broader small/mid_cap that have ever appeared in pledge filings (~250-400 candidate symbols)
- Window: **24-36 months** Discovery (2022-01 → 2024-12) targeting ≥30 trade floor; expand to 36mo if 24mo insufficient
- Schema: `[symbol, filing_date, filing_time, pledge_pct, pledge_pct_delta, event_type ("creation"|"invocation"|"release"|"reg30_material"), filer ("promoter"|"creditor"), source ("NSE"|"BSE")]`
- Output: `data/pledge_filings/<YYYY>/pledge_events.parquet`
- Spot-validation: random 20 events vs NSE/BSE source pages; require ≥95% accuracy on (date, pledge_pct, event_type)

**Effort estimate:**
- Endpoint reverse-engineering (NSE + BSE pledge-disclosure pages): 3-4 hours
- Scraper adapted from `tools/earnings_calendar/fetch_earnings.py`: 4-6 hours
- Parser + classification (Reg 31 vs Reg 30 vs SAST): 3 hours
- Spot-validation tooling: 1 hour
- Backfill run (paginated, 36 months, both NSE and BSE): 8-10 hours wall-clock
- **Total: ~2 engineering days** for first usable parquet; sanity-check unblocked the same week.

**Tool:** `tools/sub9_research/backfill_pledge_filings.py` — one-time + incremental top-up. Production-mode reuses the same scraper at EOD daily.

### 13.2 Sanity-check tool (after 13.1)

`tools/sub9_research/sanity_pledge_margin_call_cascade.py` — parallels the earnings_day sanity-check. Reads pledge_events parquet + 5m feathers; no detector code yet. Will be retired post-decision.

### 13.3 (post-sanity-check, only if APPROVED)

- `services/pledge_filing_service.py` — daily EOD scraper + per-symbol rolling-pledge-ratio lookup
- `structures/pledge_margin_call_cascade_short_structure.py` — the detector
- `data/pledge_filings/` directory persisted for live + replay parity

## §3.3 acceptance criteria recap

Candidate is **APPROVE-eligible only if both** hold:

- [ ] **(a)** Pledge-data ingestion path is clear and engineering-feasible (Q: is the NSE/BSE pledge-disclosure endpoint scrapeable with the existing `tools/earnings_calendar/fetch_earnings.py` pattern?). Currently assessed: **YES** — endpoints are public, no-auth, structurally identical to earnings-calendar endpoint.
- [ ] **(b)** Sample size is feasibly above the **n ≥ 30** floor over a 2yr Discovery window. Currently assessed: **MARGINAL** — case studies suggest ~10-30 cascade events/year, so 2yr → 20-60 events; post-gating may drop below 30. **Pre-flight pledge-event count from §13.1 backfill is the gating evidence.**

Gates beyond (a) + (b):

- [ ] §13.1 pledge backfill complete (pre-requisite)
- [ ] Sanity-check NET PF ≥ 1.10
- [ ] n ≥ 30 over backfill window (2-3yr)
- [ ] |WR delta| ≤ 10pp vs baseline
- [ ] No single event > 40% of PnL (concentration)
- [ ] Random spot-validation of pledge events ≥95% accurate
- [ ] Borrow-availability check: ≥60% of triggered events historically borrowable (SLB / F&O eligible) — non-borrowable events excluded from live, but if backfill universe is <60% borrowable the live setup is non-tradeable.

---

## Decision required

User to indicate:

- [ ] **APPROVED** — proceed to §13.1 pledge-filing backfill, then sanity-check
- [ ] **RETIRE-pre-data** — sample size structurally too thin; do not invest engineering
- [ ] **REJECTED** — reason
- [ ] **REVISE** — specify what's missing / wrong

Per sub-9 §3.3, no detector code is written until APPROVED. §13.1 backfill IS engineering work but is PRE-REQUISITE data infrastructure (parallels the earnings-calendar backfill); backfill can begin on APPROVED status. **Strong recommendation: run a 1-day reconnaissance scrape of the NSE pledge-disclosure endpoint BEFORE full APPROVE — count raw events 2022-2024 to validate the n-floor empirically before committing 2 engineering days.**
