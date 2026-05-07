# §3.3 Brief: `nifty_reconstitution_announcement_window`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — DATA-BLOCKED + SAMPLE-BLOCKED. Awaiting user APPROVE / REJECT / RETIRE-PRE-DATA.**
**Date:** 2026-05-07
**Round:** 5

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (G — Index rebalancing CONDITIONAL)
- specs/2026-05-03-sub-project-9-brief-nifty500_deletion_short.md (**RETIRED 2026-05-05** — same family; fails CNC/SLB infra; this brief differentiates by mechanic + universe + side)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (3-gate filter; round-5 inherits)

**Differentiation from retired `nifty500_deletion_short` (mandatory):**
The earlier brief shorted **NIFTY 500 deletions** on T-1 of effective using single-leg MIS, and was retired because the forced flow does not concentrate in a single intraday window (passive funds VWAP/TWAP across the 28-day pre-effective window) — the real edge required CNC + SLB infra the project does not have. **This brief inverts three things at once:** (a) **side** — fade the run-up exhaustion at T-1 close OR the effective-day open (both opposite to the round-4 short-the-deletion idea), (b) **universe** — F&O liquid universe only (NIFTY 50 / Next 50 / Bank inclusions, where reconstitution events are concentrated and SLB is not required), and (c) **window** — the same-day intraday window that IS MIS-compatible (T+0 09:15-10:30 effective-day fade, OR T-1 14:00-15:15 run-up exhaustion). The `nifty500_deletion_short` retirement does NOT apply here because we trade the **liquid-inclusion side**, where buyer-flow IS concentrated at the effective-day open (passive funds must own the new constituent before the close on effective day, and are mandated against pre-effective full positioning by tracking-error rules — so the bulk lands at the open auction or in the first 15-30 minutes).

---

## 1. Asymmetry

**Name:** Inclusion-side fade of the passive-buying exhaustion at the NIFTY 50 / Next 50 / Bank reconstitution effective-day open.

**Mechanic:** NSE Indices reconstitutes the NIFTY 50, NIFTY Next 50, NIFTY Bank, and NIFTY 500 family **semi-annually** (effective last Friday of March and September, announced ~T-30 calendar days before, per NSE Indices methodology document). Announced inclusions trigger mandatory passive-fund buying — current Indian passive AUM tracking these indices is ~₹9.7 lakh crore (AMFI, Oct 2025). The pre-effective drift up (T-30 → T-1) is well-documented in Indian academic literature (Marisetty 2025; Chhatwani 2018). The narrow exploitable window is the **effective-day open (T+0 09:15)**, where (i) residual passive-buying that did NOT land via T-1-close VWAP execution lands at the open auction or in the first 5-15 minutes, (ii) front-running arb desks unwind their long-inclusion position into this same buying, and (iii) once the passive buyer is finished, there is no natural buyer at the elevated price → **mean-reversion fade for the rest of the morning**.

**Alternative window (sanity will pick winner):** T-1 14:00-15:15 IST run-up exhaustion fade — passive funds executing T-1 VWAP / TWAP into the close create a final-hour distribution that fades on closing-auction volume drop-off.

NSE Indices methodology: https://www1.nseindia.com/products/dynaContent/equities/indices/index_methodology.htm

## 2. Participants

- **Passive index funds and ETFs (~₹9.7 lakh cr AUM, AMFI Oct 2025).** Tracking-error rules force them to buy inclusions on/near the effective day. For NIFTY 50 inclusions specifically, ETF AUM concentration is high — a single inclusion can attract ₹2,000-5,000 crore of mandated buying (industry note range, scaled from index weight × passive AUM).
- **Front-running arb desks.** Buy on T-30 announcement, sell into the effective-day passive flow. They are the natural counter-party we ride with on the fade — they distribute into 09:15-10:30 effective-day demand.
- **Late-FOMO retail.** Indian retail blogs and Moneycontrol cover inclusion announcements; some retail buys at the run-up peak (T-2/T-1) and is the marginal late distribution at T+0 open. SEBI FY23 retail-loss study confirms long-FOMO on news-event days is the structurally losing side.
- **Index-rebalance arb (Tier-1 institutional).** Operates in the spread (basket vs index), not a single-leg directional. Their flow is washed in the same effective-day window but does not fight our fade.

We sit on the **post-passive-buying-exhaustion side** — opposite of the retail late-FOMO + same side as arb-desk distribution.

## 3. Persistence

1. **NSE Indices reconstitution is codified.** Methodology doc (linked in §1) defines: announcement T-30 (Feb / Aug press release), effective last Friday of March / September, formulaic eligibility (free-float mcap rank, F&O eligibility for NIFTY 50). Not discretionary; will not change without SEBI consultation.
2. **Indian passive AUM keeps growing.** ₹9.7 lakh cr passive index AUM (Oct 2025) up from ~₹0.5 lakh cr (2018). Sectoral fund AUM ₹3.4 lakh cr (Mar 2026). The mandated-buyer pool is structurally larger every year — this is the OPPOSITE of the Greenwood-Sammon "disappearing index effect" trajectory in US markets (where passive saturated post-2010). Indian markets are at the **rapid-passive-growth phase**; effect is currently strong and expected to persist 5-10 years before saturation.
3. **Retail-algo absence.** Stratzy, Wright Research, Religare, AlgoTest, uTrade — none publish reconstitution-window strategies. Cited as a feature, not a bug, per round-3 lesson: WINNERS in this project are tied to NSE-codified events absent from retail-algo catalogues; LOSERS are globally-published generic asymmetries. Reconstitution belongs in the WINNER bucket.

## 4. Evidence

- **NSE Indices Methodology Document** (regulatory) — https://www1.nseindia.com/products/dynaContent/equities/indices/index_methodology.htm — defines reconstitution schedule, eligibility, rebalance window.
- **Greenwood & Sammon, NBER w30748 (2022) — "The Disappearing Index Effect"** — global decline of inclusion effect, BUT documents the strong mid-cycle Indian-market analog. https://www.nber.org/system/files/working_papers/w30748/w30748.pdf
- **Marisetty 2025, SSRN 5642110 — 81 NIFTY events 2010-2024.** Pre-effective CAAR (additions): ~+1.10% on announcement, statistically insignificant; **−1.17% on effective day** (the fade we want to capture). Deletions: stronger negative drift then partial recovery. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5642110
- **Chhatwani (Indian Journals, Publishing India) — addition-day positive abnormal return + 60-day reversal.** Indian-specific NIFTY inclusion study. https://www.indianjournals.com/
- **Greenwood & Sammon (2010 working paper) — "Predictable Investor Trades and Stock Returns"** — foundational for the front-run-the-passive-flow framework. (User-cited reference.)
- **Indian-broker research notes** (institutional, not retail-algo) — Nirmal Bang and Kotak publish inclusion / exclusion research notes ahead of each March / September reconstitution. Confirms institutional desks systematically position; does not document the effective-day fade because their hold horizon is multi-day.
- **Springer chapter on Indian-equity passive flow** (Indian Capital Markets handbook) — confirms the structural growth of passive AUM as the persistence driver.

Indian-retail-algo Gate A: **NOT MET via positive publication** — counted as evidence for capacity-unsaturated edge per round-3 lesson.

## 5. Direction

**Effective-day fade (PRIMARY):** SHORT inclusions at T+0 09:15 open after the open-auction passive-buying lands; cover into 10:30 mean-reversion.
**T-1 run-up exhaustion fade (SECONDARY, sanity-comparison):** SHORT inclusions at T-1 14:00 5m close; cover at T-1 15:15.

Both are SHORT-only on the inclusion side. **Exclusion side is excluded** from this brief — the prior `nifty500_deletion_short` brief retired exclusion-shorting due to passive VWAP-distribution across 28 days (no single-day concentration). Inclusion-buying IS concentrated at effective day because of tracking-error rules; exclusion-selling is not concentrated by symmetry.

**Long-bias guardrail:** N/A. SHORT-only by construction.

## 6. Mechanic

**Setup name:** `nifty_reconstitution_inclusion_fade`

1. **T-30 prep:** parse NSE Indices announcement PDF (Feb / Aug). Output curated table: `(symbol, index, action=include, effective_date)`. Two events/year × 5-10 inclusions each across NIFTY 50 + Next 50 + Bank = **10-20 inclusion events / year**.
2. **T+0 09:15 entry:**
   - Filter: stock is in announced inclusion list for today's effective_date AND in F&O 200 (liquidity + SLB availability) AND large_cap or mid_cap (no small_cap inclusions exist in NIFTY 50 / Next 50 / Bank).
   - Entry trigger: 09:15 5m bar's CLOSE is positive (gap-up open confirms passive-buyer demand cleared the auction). If 09:15 prints negative, ABORT — passive flow may already have washed pre-open.
   - **Entry price:** 09:15 5m bar's close.
   - **Side:** SHORT.
3. **Stop-loss:** entry × 1.010 (1.0% above entry; defends against secondary-flow wave from late ETF rebalances).
4. **Targets:** T1 entry × 0.992 (0.8R if SL is 1.0%), T2 entry × 0.985 (1.5R). 50% qty per leg.
5. **Time stop:** 10:30 IST (effective-day fade has its strongest mean-reversion in the first 75 minutes; after 10:30 noise dominates).
6. **Latch:** one fire per (symbol, effective_date).

**target_anchor_type:** `r_multiple` (no clean structural anchor on inclusion days; gap edges are not the right reference because there is no T-1 gap-down).

**T-1 14:00 variant (for sanity comparison):** entry T-1 14:00 5m close, SL T-1 day high × 1.005, T1 / T2 = 1R / 2R, time stop 15:15.

## 7. Universe

**F&O 200 only.** Cap segments: large_cap and mid_cap. Indices: NIFTY 50, NIFTY Next 50, NIFTY Bank. NIFTY 500 inclusions excluded (sample-size advantage but illiquid; revisit only if NIFTY 50/Next 50/Bank sample alone fails the n-floor).

## 8. Active window

- **Effective-day primary:** T+0 09:15-10:30 IST (single fire at 09:15; cover by 10:30 time-stop).
- **T-1 secondary:** T-1 14:00-15:15 IST.
- Both are MIS-compatible (intraday only); no overnight.

## 9. Risks / falsification

Locked thresholds (round-5 standard):
- NET PF ≥ 1.10
- |WR delta| ≤ 10pp vs equal-weight baseline
- n ≥ 30 over 2-year discovery window (round-5 floor; relaxed from main n≥500 floor for event-driven semi-annual setups)

**Sample-size honesty (binding constraint):** NIFTY 50 + Next 50 + Bank inclusions = 5-10 events/announcement × 2 announcements/year = **10-20 events/year**. Over 2yr Discovery: **20-40 events**. Over 3yr Discovery: 30-60. The acceptance gate is **n ≥ 30 on F&O over 2yr**; if structurally <30 after curation (some announcements have zero inclusion changes; NIFTY Bank reconstitution is sparse), candidate is **retire-eligible-pre-data** without further work.

**Other falsifiers:**
- Effect decayed faster than projected (Greenwood-Sammon): if 2024 sample PF < 1.10 but 2018-2020 PF > 1.30, edge is decaying; retire.
- Effective-day fade window wrong: if 09:15 fade fails but T-1 14:00 variant passes ≥1.10, ship the T-1 variant only.
- Both windows fail: retire decisively (no loosening).

## 10. Pre-coding sanity-check plan

**Data prerequisite:** reconstitution-event history table — `(announcement_date, effective_date, symbol, action ∈ {include, exclude}, index ∈ {NIFTY 50, Next 50, Bank, 500})`. **Not on disk.** This is the binding data block (see §11).

**Sanity-check tool (only after §11 data ingest):** `tools/sub9_research/sanity_nifty_reconstitution_fade.py` — reads curated CSV + 5m feathers + consolidated_daily.feather. Manual curation of ~20-40 events × 2 windows (T+0 + T-1 variant) is feasible at sanity scope. NET PF computed via existing Indian fee model. Decision per §3.3: PF ≥ 1.10 AND n ≥ 30 → proceed; PF < 1.0 → retire; n < 30 structural → retire-pre-data.

## 11. Data engineering plan (PRE-REQUISITE — flagged)

**FLAG:** reconstitution-event history is **not on disk**. Unlike the retired `nifty500_deletion_short` brief (which proposed scraping NSE press releases), this brief restricts to **NIFTY 50 / Next 50 / Bank only** — a more tractable curation.

**Sources:**
- NSE Indices press releases (Feb / Aug PDFs, semi-annual). Public, downloadable.
- Nirmal Bang and Kotak research notes (institutional reports, ahead of each rebalance) — useful for cross-validation.
- Tickertape index-changes archive — third-party aggregator, useful as sanity backup.

**Curation effort:** ~6 announcements over 3 years × ~5-10 changes per announcement = 30-60 manual rows. **Curation by hand in ~2-3 hours** is feasible at sanity scope; no scraper required for sanity. Production live-mode would need a once-per-6-months scraper — trivial vs the round-4 NSE-corporate-filings backfill.

**Comparison vs retired `nifty500_deletion_short` data path:** that brief proposed scraping NSE 500 deletion lists (50-70 deletions per announcement, illiquid stocks, SLB-borrow checks needed). This brief restricts to **liquid-inclusion side of NIFTY 50 / Next 50 / Bank only** — narrower, manually curatable, no SLB dependency (F&O 200 inclusions all have futures-segment liquidity). Data path is **clear and significantly easier** than the retired brief.

---

## §3.3 acceptance criteria recap

- [ ] Manual curation of NIFTY 50 / Next 50 / Bank reconstitution events 2023-2024 complete (~20-40 rows)
- [ ] Sanity-check NET PF ≥ 1.10
- [ ] |WR delta| ≤ 10pp
- [ ] n ≥ 30 over 2yr discovery
- [ ] T+0 09:15 vs T-1 14:00 variants compared; ship the winner only
- [ ] If structurally n < 30 after curation: **retire-pre-data**

## Decision required

User to indicate:
- [ ] APPROVED — proceed to §11 manual curation, then sanity-check
- [ ] REJECTED — reason
- [ ] RETIRE-PRE-DATA — sample size too tight without curating

**My read:** APPROVE conditional on user accepting the n=30 floor (relaxed from main n=500). The Indian inclusion effect is the strongest peer-reviewed asymmetry in the round-1 candidate list AND data is hand-curatable. Worst-case sanity outcome is "real edge but n<30" → that becomes a useful negative finding for sub-9 closure and informs whether this re-emerges when reconstitution data infra is built.
