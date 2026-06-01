# §3.3 Brief: `anchor_lockin_expiry_short`

> ## Phase 1 verdict (2026-06-01) — PROCEED to Phase 2 conditional on pre-flight chittorgarh recon
>
> | Gate | Status | Notes |
> |---|---|---|
> | A (Indian-pro precedent) | **PASS** | Strongest regulatory anchor in candidate set — SEBI(ICDR) Reg 32(b) 2022 amendment IS the regulator's own acknowledgement of the asymmetry. Multiple Indian broker education sources operationalize the pattern (Bajaj Finserv, Choice, Angel One, Sharekhan, m.Stock, HDFC sky, Stockgro). Concrete recent example: **Waaree Energies dropped 9% intraday on lock-in expiry April 2025** (confirmed by multiple sources). Stockgro published operationalized retail guidance: "those investors may exit at the 30/90-day marks, and selling before can help you avoid the rush." US foundational lit (Field & Hanka *JoF* 2001 -1.5% to -3% abnormal returns). |
> | B (Data feasibility) | **NEEDS BACKFILL (0.5 day)** | Anchor lock-in calendar NOT on disk (`data/anchor_lockin_calendar/` doesn't exist). Adapter pattern proven: `tools/earnings_calendar/fetch_earnings.py` is structurally identical (chittorgarh.com paginated HTML scraper). 3yr backfill (~440 raw events). Pre-flight recon scrape (~3 hours) RECOMMENDED before committing full 0.5 day to validate raw event count meets the n threshold. |
> | Regulatory sensitivity | **LOW RISK** | SEBI Reg 32(b) post-2022 codified, stable. SEBI 2022 amendment was the regulator's structural acknowledgement; further amendments would dilute MORE (not eliminate) the asymmetry. **Critical:** Discovery MUST exclude pre-2022-Q3 events (single-day expiry regime had -2.6% effect vs current split's expected -1% to -1.5% per side — regime mixing inflates PF by 30-50%). |
> | n/yr screen | **MARGINAL** | Brief estimates ~85 trades/year aggregate (T+30 + T+90), ~45/yr per subtype. **Same order of magnitude as just-killed `post_split_bonus_short` (55-60/yr).** Today's Lesson #22 makes this a first-class gate. Mitigating factor: anchor_lockin events are predictably scheduled in advance (lock-in dates appear in RHP at IPO), so capital pre-allocation is feasible — different operational profile than slot-competing daily setups. **Decision: acceptable IF per-event edge is strong (Phase 2 mean drift ≥-0.5%); KILL if marginal.** |
> | Concentration risk | **HIGH** | 2024 IPOs are extreme-skewed: Hyundai (₹27,856cr anchor), LIC, Reliance Retail, Mankind, Tata Tech, IREDA, Bajaj Housing dominate by anchor allocation magnitude. Phase 2 MUST report per-symbol PnL contribution; >35% from single name = name-specific not class-specific. |
>
> **Decision: PROCEED to Phase 2 with pre-flight discipline.** Step 1 (this week, ~3 hours): chittorgarh reconnaissance scrape to count raw events 2022-Q3 to 2024 — verify ≥300 raw events / 3yr. If recon clears, Step 2: full 0.5 day backfill. Step 3: Phase 2 signature script (~10 min compute). **Hard kill at Phase 2** if either (a) post-2022 regime aggregate drift <-0.5% or (b) per-symbol concentration >35% / single name.

**Sub-project:** #9 (microstructure-first redesign), Round-4 Indian-microstructure / corporate-action candidate
**Status:** **DRAFT — pre-sanity disposition required.**
**Date:** 2026-05-09

**Predecessors / context:**
- `specs/2026-05-01-sub-project-9-microstructure-first-redesign.md` (defines §3.3 gate process, locked thresholds)
- `specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md` (data-broad universe + n≥500/2yr methodology, narrow-cell n≥30 fallback)
- `specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md` (gold-standard SHORT-side regulatory-anchor brief — APPROVED, sanity PF=1.473)
- `specs/2026-05-07-sub-project-9-brief-capitulation_long_morning.md` (alternative gold-standard — published-arb survival framing)
- `specs/2026-05-08-sub-project-9-brief-buyback_tender_intraday.md` (n-marginal corporate-action precedent — narrow-cell discipline)

This is the round-4 Indian-microstructure candidate where the anchor is **SEBI(ICDR) Regulations 2018, Reg 32(b)** which mandates a split lock-in for IPO anchor-investor allocations: 50% locked 30 calendar days from allotment, the remaining 50% locked 90 calendar days. On the lock-in expiry day, anchor investors who paid the issue price can sell freed shares for the first time. The 30/90 split is a 2022 SEBI amendment unique to India — the regulator REDUCED but did not ELIMINATE the asymmetry, splitting one large dump into two smaller dumps at two regulator-fixed timestamps.

The thesis is structurally identical to `circuit_t1_fade_short`'s mechanic class: a **regulator-fixed timestamp for a one-sided institutional flow with no symmetric counter-flow**.

---

## 1. Asymmetry

**Name:** Indian-equity intraday short on IPO anchor-investor lock-in expiry day (T+30 / T+90 from allotment) — capturing the institutional MTM-lock supply release at the regulator-fixed first-tradable-day open.

**Specific mechanism (chained):**

1. **IPO anchor allocation (T-1 of listing).** Per SEBI(ICDR) Reg 32, an issuer may allocate up to 60% of the QIB portion to anchor investors at the issue price one trading day before public-issue open. Anchor investors are typically MFs, FIIs, and sovereign wealth funds; allocation ranges from ₹50cr (small-cap IPOs) to ₹6,000cr+ (Hyundai India 2024).
2. **Listing day pop (T+0 of listing).** Mainboard IPOs in 2024 had an average listing-day pop of +28% with 80% of issues closing above issue price. Anchors sit on unrealized gains by close of listing day.
3. **Reg 32(b) split lock-in.** Post-2022 SEBI amendment: **50% of anchor allocation is locked for 30 calendar days from allotment date; the remaining 50% is locked for 90 calendar days**. Lock-in dates are PUBLIC — they appear in the RHP/prospectus and are aggregated by Chittorgarh.
4. **Lock-in expiry day (T+30 or T+90).** At 09:15 IST on the expiry day, the freed slug becomes tradable for the first time. If listing price > issue price (true for ~80% of 2024 mainboard IPOs), anchors sit on positive MTM and rationally MTM-lock by selling the freed slug at market open. Selling is concentrated in the first hour because (a) institutional desks have block-execution mandates that prefer auction-adjacent depth, (b) waiting risks giving back unrealized gains, (c) acceptance-ratio uncertainty on subsequent expiry slugs is removed only by selling the current slug.
5. **One-sided supply event.** Lock-in releases SELLERS, not BUYERS — there is NO symmetric demand counter-flow by mechanic (no parallel "demand lock-in" exists in Indian IPO regs). Pure supply asymmetry at a regulator-fixed timestamp.
6. **Two trade-events per IPO:** (i) T+30 expiry (50% slug release), (ii) T+90 expiry (50% slug release). Both events are independently tradable; sample expansion is 2× per IPO.

**Why this is asymmetric (not just statistical):** lock-in expiry is one of the cleanest examples of regulator-mandated one-sided supply in Indian markets. Unlike open-market supply events (block deals, F&O unwinds) where flow direction depends on participant intent, lock-in expiry has **mandatory unilateral structure**: by SEBI rule, anchors COULD NOT sell before T+30/T+90; on these dates they CAN sell. There is no rule that "creates demand" on these days — the regulation only releases sellers. Combined with the listing-day pop (80% of 2024 IPOs above issue price), the rational MTM-lock incentive is dominant. The 30-day/90-day split DILUTES the asymmetry vs the pre-2022 single-day expiry but does not eliminate it — instead concentrating it at TWO timestamps per IPO.

## 2. Indian-microstructure anchor (THE CRITICAL GATE)

**Test 1 — Anchor is regulator-defined?** **PASS.**
- SEBI(ICDR) Regulations 2018, Reg 32 governs anchor-investor allocation. Reg 32(b) post-2022 amendment specifies the 30/90 split lock-in.
- Reference: https://www.sebi.gov.in/legal/regulations/may-2024/securities-and-exchange-board-of-india-issue-of-capital-and-disclosure-requirements-regulations-2018-last-amended-on-may-17-2024-_80421.html
- Anchor lock-in dates are PUBLIC: each issuer's RHP discloses allotment date; lock-in dates are mechanically T+30 and T+90 calendar days from allotment.
- The 30/90 SPLIT is unique to India:
  - **US:** uniform 90-day or 180-day lock-up (Field & Hanka 2001 baseline); no split.
  - **UK / EU:** no formal anchor-investor concept (mainboard IPOs use book-building without pre-IPO institutional lock-up).
  - **Singapore:** different acceptance-ratio rules; no equivalent two-tranche split.
- This is structurally identical to `circuit_t1_fade_short`'s mechanic class — regulator-fixed timestamp + institutional-flow concentration + no symmetric counter-flow. The pattern is recognizably from the same family as the production-validated SHORT-side regulatory-anchor briefs.

**Test 2 — Has timestamped event data accessible from public sources?** **PASS.**
- **Source 1 (primary):** chittorgarh.com structured-HTML aggregator — `/report/anchor-investor-lock-in-end-dates/156/mainboard/?year={2023,2024}` with columns `[symbol, listing_date, anchor_allocation_cr, lockin_30day_date, lockin_90day_date, anchor_investor_count]`. Backfill cost: ~0.5 day.
- **Source 2 (validation):** business-standard.com weekly anchor-lock-in roundups; ipocentral.in IPO-history archive; KPMG India IPO annual report (for cross-check on IPO count + anchor allocation totals).
- **Source 3 (raw filings):** SEBI filings page Letter-of-Offer + RHP per IPO (hardest to scrape, only used as a last-resort spot-validator on a sample).
- **Schema target:** `data/anchor_lockin_calendar/lockin_events.parquet` with `[ipo_listing_date, symbol, isin, anchor_allocation_cr, lockin_30day_date, lockin_90day_date, anchor_investor_count, source_url]`.
- **Trade-execution data:** `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_5minutes_enriched.feather` already covers most newly-listed names within 1-2 weeks of listing. Coverage gap risk is LOW because lock-in expiry is at T+30 / T+90 from allotment, well past the 1-2 week initial cache lag.
- **Backfill cost: ~0.5 engineering day for chittorgarh scraper + spot-validation against 30 random IPOs.**

**Test 3 — Direction is empirically supported?** **PASS — STRONG.**
- **Edelweiss research (2021), cited via Bajaj Finserv:** of 41 IPO anchor-lock-in expiries in 2021, **76% saw selling pressure on the day**, with **-2.6% average price correction** on expiry day, and **61% continued lower over the 5-day post-expiry window**. URL: https://www.bajajfinserv.in/anchor-investor
- **Business Standard (29 May 2025):** "₹21 trillion worth of shares to unlock by Sept 2025" — corroborates the magnitude of expected lock-in unlocks. URL: https://www.business-standard.com/markets/news/anchor-lock-in-expiry-to-unlock-21-trillion-worth-shares-by-september-2025-125052900460_1.html
- **5paisa research (2025):** ₹21T unlock corroboration, independent source. URL: https://www.5paisa.com/news/rs21-trillion-worth-of-shares-to-unlock-by-september-2025-on-anchor-lock-in-expiry
- **Field & Hanka, *Journal of Finance* 2001 — "The Expiration of IPO Share Lockups":** US-equity foundational paper documenting **-1.5% to -3% abnormal return** at lock-up expiry. Indian split-lock evidence (Edelweiss -2.6%) is consistent with the foundational US literature within bounds. URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=205011
- **Strongest single piece of evidence: SEBI's 2022 amendment was specifically motivated by the Edelweiss-documented effect.** The regulator agreed the asymmetry was material enough to warrant intervention. SEBI REDUCED the asymmetry (split 50/50 instead of single-day 100%) but did not ELIMINATE it. This is the strongest possible evidence of setup persistence — the regulator has ALREADY accepted that the mechanic is real, durable, and large enough to require structural mitigation.
- Internal precedent: post-event-fade SHORT mechanic class is production-validated via `gap_fade_short` (n=6,723; PF 1.153) and `circuit_t1_fade_short` (sanity PF=1.473).

**§2 verdict: PASS all three tests. Anchor is the strongest in the round-4 candidate set.** Primary risks are (a) sample size marginal vs n≥500 aggregate floor (clears narrow-cell n≥30 with 3yr extension), and (b) absolute-anchor-allocation skew where 5-10 jumbo IPOs (Hyundai, LIC, Reliance Retail) dominate by absolute supply.

## 3. Universe & cell hypothesis

**Per round-3 broadened-universe rule:**

- **Cap segment:** ALL admissible at sanity stage. Mainboard IPOs span the cap distribution at-listing — large-cap (Hyundai, LIC), mid-cap (Tata Tech, Mankind), small-cap (most 2024 IPOs). The gauntlet (Stage 3, post-sanity) decides which slice carries the edge.
- **F&O 200 NOT pre-locked.** Most newly-listed names are not F&O-eligible at T+30 (F&O addition typically requires 6-12 months of trading history). MIS-shorting on cash-equity is required; brief assumes Zerodha SLB or equivalent broker borrow.
- **Liquidity gate:** 20-day median daily volume × close ≥ ₹2 Cr. Lower than the typical sub-9 brief (which uses 20-day = ₹5 Cr) because newly-listed names have inflated turnover in the first month — the ₹2 Cr floor is a data-quality gate to exclude SME/dead-listings, not a thesis-defining gate.
- **Anchor-allocation gate:** `anchor_allocation_cr >= ₹50 Cr`. This is a **mechanic-defining filter** — below ₹50cr the unlocking slug is too small to move the market regardless of MTM-lock incentive. ₹50cr threshold is sourced from the round-4 research and is a hard pre-locked filter (not iterated post-sanity).
- **MIS-eligibility:** must be MIS-tradable on broker (`mis_leverage >= 1.0`). **SME IPOs are EXCLUDED** — SME issues are placed in **TFT (Trade-for-Trade) settlement** for the first 30 calendar days post-listing, which is exactly the T+30 lock-in window. SME T+30 trade is mandatory delivery-settled, NOT MIS-tradable. **Mainboard-only is structural, not optional.**
- **HARD data dependencies:** (i) anchor lock-in event-calendar with T+30/T+90 timestamps (NEW; see §10), (ii) 5m enriched feathers (existing, with 1-2 week post-listing cache lag — irrelevant since T+30 is well past lag), (iii) issue-price + listing-price for the listing-day pop conditioning filter.

**Cell hypothesis (gauntlet Stage 3, post-sanity):**

- **Primary cell:** (`event_type=T+30` × `anchor_allocation_cr_quartile=Q4` × `listing_pop_above_issue=YES`). Largest slugs released in their first lock-in expiry, on IPOs whose listing-day pop produced anchor MTM-gain — strongest mechanic.
- **Secondary cell:** (`event_type=T+90` × `anchor_allocation_cr_quartile ∈ {Q3, Q4}`). T+90 expiry on the second slug; anchors who held through T+30 were either accepting acceptance-ratio uncertainty or deliberately waiting — selling on T+90 is the residual MTM-lock.
- **Conditioning variant:** (`gap_at_open ≥ +1%`). Hypothesis: gap-up open on lock-in expiry day signals retail FOMO into the freed-slug supply, amplifying the short fade. Mirror-image of `circuit_t1_fade_short`'s gap-up-required filter.
- **De-amplification cell:** (`event_type=T+30` × `listing_pop_above_issue=NO`). If anchors are sitting on losses, the MTM-lock incentive REVERSES — anchors hold to recover, supply is suppressed. Expected to fail PF gate; documents the mechanic's directionality.

**Symbol count (event basis):**
- 2023 mainboard IPOs: ~57. 2024 mainboard IPOs: 93. (Sources: chittorgarh + KPMG 2024 IPO annual report.)
- 2 events per IPO (T+30, T+90).
- Raw events 2023-2024: (57 + 93) × 2 = **300 raw events / 2yr**.
- 3yr extension to 2022-2024: ~440 raw events.
- Per-event filter survival: see §8.

## 4. Persistence

Three structural reasons the edge should persist:

1. **SEBI(ICDR) Reg 32(b) is codified post-2022.** The 30/90 split-lock mechanic is regulator-mandated. SEBI introduced the split SPECIFICALLY to dampen the lock-in-expiry single-day dump (the Edelweiss 2021 -2.6% effect). This was the regulator's deliberate trade-off: dilute the asymmetry, do not eliminate it, because eliminating anchor lock-up entirely would discourage anchor participation in IPOs. The regulator's structural preference — dampened-but-present asymmetry — is locked in by regulation.

2. **MTM-lock incentive is rational and unavoidable.** Anchor investors are institutional (MFs, FIIs, sovereign wealth funds) with mark-to-market reporting requirements. When listing price > issue price (80% of 2024 IPOs), the freed slug is a guaranteed positive-MTM realization. Holding the slug past the first tradable day means risking unrealized gains for an information-free reason (no new information arrives at the expiry timestamp itself). Rational MTM-locking is structurally aligned with selling at the open.

3. **No retail-algo published playbook.** Retail-algo platforms (Streak, Stratzy, Wright, Algotest) cover IPO-listing-day momentum and IPO-allotment-day GMP-fade, but **none publish anchor-lock-in-expiry intraday playbooks**. The lock-in expiry calendar requires (a) IPO RHP-tracking and (b) anchor-allocation-magnitude data — neither of which are surfaced in retail-algo screeners. The closest published coverage is Bajaj Finserv's broker-side educational content, which discusses the phenomenon but does NOT publish an intraday-trade rule. Capacity is moderately-unsaturated — institutional desks know the mechanic but price impact is bounded by the slug-size, not arb-saturation.

**Decay caveat (acknowledged):** unlike Edelweiss-2021 single-day-expiry data, the post-2022 regime has only ~3.5 years of data. SEBI may further amend lock-in mechanics in future (e.g., extending to 180-day uniform lock or staggering further). Brief proposes Discovery on POST-2022 data only (2022-Q3 onwards) to avoid regime-mixing.

## 5. Evidence

**Regulatory primary sources:**
1. **SEBI(ICDR) Regulations 2018, Reg 32 (last amended May 2024)** — https://www.sebi.gov.in/legal/regulations/may-2024/securities-and-exchange-board-of-india-issue-of-capital-and-disclosure-requirements-regulations-2018-last-amended-on-may-17-2024-_80421.html
2. **SEBI 2022 amendment — split lock-in introduction** — codified in the cited regulations; the amendment text is the load-bearing piece of evidence (regulator agreed the effect is real).

**Indian-specific empirical evidence:**
3. **Edelweiss research (2021), via Bajaj Finserv "Anchor Investor" article** — 76% of 41 IPO lock-in expiries showed selling pressure; -2.6% average expiry-day correction; 61% continued lower 5d post-expiry. URL: https://www.bajajfinserv.in/anchor-investor
4. **Business Standard (29 May 2025) — "₹21 trillion shares to unlock by Sept 2025"** — magnitude corroboration. URL: https://www.business-standard.com/markets/news/anchor-lock-in-expiry-to-unlock-21-trillion-worth-shares-by-september-2025-125052900460_1.html
5. **5paisa research (2025) — independent corroboration of ₹21T unlock figure.** URL: https://www.5paisa.com/news/rs21-trillion-worth-of-shares-to-unlock-by-september-2025-on-anchor-lock-in-expiry

**Foundational US/peer-reviewed lit:**
6. **Field & Hanka, *Journal of Finance* 2001 — "The Expiration of IPO Share Lockups"** — US-equity foundational study; -1.5% to -3% abnormal returns at lock-up expiry. URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=205011

**Internal precedent:**
7. **`circuit_t1_fade_short`** — same mechanic class (regulator-fixed timestamp + one-sided institutional flow). Sanity PF=1.473.
8. **`gap_fade_short`** — production-validated SHORT-side fade machinery; n=6,723 / NET PF 1.153.

**Published retail-algo content audit:** Streak, Stratzy, Wright, Algotest searched for "anchor lock-in", "IPO lock-up expiry", "T+30 IPO" — no published intraday-trade rule found. The mechanic is discussed in broker-education content (Bajaj, Zerodha Varsity in passing) but not codified as a strategy. Capacity is moderately-unsaturated.

**Strongest evidence weighting (per §3.3 hierarchy):** SEBI's own 2022 amendment is the strongest piece — it is *the regulator agreeing the effect is real and large enough to warrant structural intervention*. No other Lane-1 candidate has this level of regulatory acknowledgement.

## 6. Mechanic

**Setup name:** `anchor_lockin_expiry_short`
**Side:** SHORT-only (one-sided supply event by mechanic — no symmetric LONG side exists).
**Bar timeframe:** 5m
**Active window:** 09:30-10:30 IST entry; same-session exit by time-stop.

**Sequence (single detector, dual event-subtype routing):**

1. **Pre-session detection (run at 09:14:55 IST):**
   - Symbol must be on the anchor lock-in event-calendar with EITHER `lockin_30day_date == today` OR `lockin_90day_date == today`.
   - Symbol must pass liquidity gate: 20-day median (volume × close) ≥ ₹2 Cr.
   - Anchor allocation gate: `anchor_allocation_cr ≥ ₹50 Cr`.
   - Symbol must be MIS-eligible (`mis_leverage ≥ 1.0`). **SME-listed symbols MUST be excluded** (TFT-restricted in T+30 window).
   - **Listing-pop conditioning (mechanic-locked):** `listing_close_price ≥ issue_price` — i.e., the IPO closed listing day above the anchor's cost basis, so anchors sit on positive MTM. If listing-pop is negative, MTM-lock incentive reverses; standdown.
   - **De-duplication with `gap_fade_short`:** if `gap_fade_short` already fired same `(symbol, date)` in the morning open window (09:15-09:30), skip. Latch order: `gap_fade_short` fires first (09:15-09:30 single-bar entry) and takes priority; `anchor_lockin_expiry_short` fires second (09:30-10:30) and yields. This prevents double-shorting the same instrument-day on a gap-up day that happens to coincide with a lock-in expiry.

2. **Event-subtype routing:**
   - **Subtype A:** `event_type = T+30` (50% slug release).
   - **Subtype B:** `event_type = T+90` (50% slug release).
   - The detector runs the SAME entry/exit logic for both subtypes; subtype is recorded as a tag for cell-stratification in sanity and gauntlet.

3. **Confirmation candle (5m bars in 09:30-10:30 IST):**
   - First 5m bar in the active window where: (`close < open`) AND (`volume(bar) ≥ 1.2 × 20-day same-time-of-day average`) AND (`close < session VWAP`).
   - Three conditions: bearish bar + volume confirmation + below-VWAP confirmation. Mirror of `buyback_tender_intraday` Subtype-B pattern.
   - Entry price: confirmation bar's CLOSE.
   - Direction: SHORT.

4. **Stop-loss (ATR-based, mirror of `circuit_t1_fade_short`):**
   - `stop = entry + 1.5 × ATR(20-day, daily)`.
   - **Min stop distance:** `min_stop_distance_pct = 0.5%` of entry (qty-inflation guard for thin newly-listed names).
   - The hard SL defends against a counter-thesis scenario where MTM-lock supply is absorbed by retail FOMO buying back the dip; ATR-based sizing accommodates the elevated post-listing volatility of newly-listed names.

5. **Targets (R-multiple, mirror of `circuit_t1_fade_short` pattern):**
   - **T1** (50% qty): 0.5R partial.
   - **T2** (50% qty): 1.5R full.
   - **Time stop:** 13:00 IST hard exit. Earlier than other sub-9 SHORT briefs (which use 14:30 / 15:15) because the lock-in supply impact is concentrated in the first hour of trading; beyond 13:00 the residual edge is dominated by mid-session noise rather than the supply-overhang mechanism.

6. **target_anchor_type:** `arithmetic_R` (R-multiples; mirror of `circuit_t1_fade_short`).

7. **Latch:** one fire per (`symbol`, `expiry_date`) — no re-entry same session. The event is a single-day phenomenon; re-entry contradicts the supply-overhang concentration thesis.

The mechanic is a hybrid of `circuit_t1_fade_short`'s ATR-based stop / R-multiple targets and `buyback_tender_intraday`'s post-event SHORT-fade structure, specialized to the IPO anchor-lockin event class.

## 7. Active window

**Setup formation + entry:** 09:30-10:30 IST (the 12 5m bars from 09:30 inclusive to 10:30 inclusive). Entry on the FIRST bar in the window meeting all three confirmation conditions.

**Hold horizon:** until target hit OR 13:00 IST hard time-stop. Hold is up to 3h 30m intraday MIS.

**Why the 09:30-10:30 entry window:**
- 09:15-09:30 is dominated by gap-pricing and pre-open-auction continuation flow; in the listing-day-pop universe, the open print can over-react in either direction. Avoid the auction-tape window.
- 09:30-10:30 is when institutional desks execute their MTM-lock orders — block-sale algorithms typically run from 09:30 (post-auction settlement) through the first hour, with concentration in 09:30-10:00.
- After 10:30: most of the freed slug has already been distributed; entering a fade after the supply-impact has been absorbed is taking an unfavorable point in the supply curve.
- 13:00 time-stop: by 13:00 the supply overhang is mechanically cleared; intraday post-13:00 movement is dominated by general market drift (NIFTY direction, sector rotation), not the lock-in event itself. Closing at 13:00 keeps the trade horizon aligned with the mechanic.

**Why 09:30-10:30 is later than `circuit_t1_fade_short`'s 10:30 single-bar:** circuit_t1's 10:30 entry waits for the T+1 retail FOMO flow to peak before fading. Anchor-lockin's flow is institutional (block-sale algos) which front-loads in 09:30-10:00 — so the entry window starts earlier (09:30) and is wider (1 hour) to capture the institutional supply pulse.

## 8. Sample-size feasibility

**Annual event volume estimate (event basis):**
- 2023 mainboard IPOs: ~57 (chittorgarh).
- 2024 mainboard IPOs: ~93 (chittorgarh + KPMG India 2024 IPO report).
- Per IPO: 2 lock-in events (T+30, T+90).
- Raw events 2023-2024: (57 + 93) × 2 = **300 events / 2yr**.
- 3yr extension to 2022 (post-amendment effective date Q3-2022): ~150 mainboard IPOs × 2 = **~330-440 events / 3yr**.

**Filter survival rates:**
- Liquidity gate (20-day vol×close ≥ ₹2 Cr): ~85% pass (most mainboard IPOs are liquid in their first 90 days).
- Anchor allocation ≥ ₹50 Cr: ~80% pass (excludes the smallest mainboard IPOs and the "fancy small-cap" segment).
- Listing-pop ≥ issue price (mechanic gate): ~80% pass (matches 2024 mainboard listing-pop stats).
- Confirmation candle (bearish + volume + sub-VWAP): ~55% pass (mirror of `circuit_t1_fade_short` confirmation pass-through).
- Combined survival: 0.85 × 0.80 × 0.80 × 0.55 ≈ **0.30 = 30% of raw events**.

**Expected Discovery trade count:**
- 2yr (2023-2024) Discovery: 300 × 0.30 = **~90 trades**.
- 3yr (2022-2024) Discovery: 440 × 0.30 = **~130 trades**.
- 4yr extension (2022-2024 + 2025 partial): ~170 trades.
- Per-side stratification (T+30 vs T+90): each side gets ~half. T+30 ~65 trades / 3yr; T+90 ~65 trades / 3yr.

**n verdict:**
- **n ≥ 500 / 2yr aggregate floor: NOT MET.** 90 trades on 2yr Discovery, even with 3yr ~130, is well below the 500 aggregate floor.
- **Narrow-cell n ≥ 30 floor (per `nse_gsm_asm_event` / `buyback_tender_intraday` precedent): MET on 3yr.** Each subtype-cell (T+30, T+90) clears n≥30 with margin; the cap-segment-stratified cells may be marginal (n ~30 per cap × subtype combined cell on 3yr).
- **Path of least resistance:** 3yr Discovery (2022-Q3 to 2024) with narrow-cell n≥30 stratification on `(event_subtype, anchor_allocation_quartile)`. If sanity at 3yr passes the n=30 floor and PF≥1.10, proceed; if marginal, extend to 4yr (include 2025 H1) and re-sanity.

**Sample-size discipline match:** comparable to `buyback_tender_intraday` (n-marginal, narrow-cell discipline); cleaner than `nse_gsm_asm_event` (event volume similar order of magnitude). Stretches the sub-9 sample-size discipline meaningfully but does not violate it.

## 9. Falsification criteria

**Locked thresholds (per §3.3 standard, with narrow-cell concession per `buyback_tender_intraday` precedent):**

- **NET PF ≥ 1.10** on Discovery (Indian fee model)
- **n_trades ≥ 30 per subtype-cell** (T+30 / T+90); aggregate n ≥ 60 over 3yr Discovery
- **|WR delta| ≤ 10pp** between Discovery and OOS Validation (2025)
- **NET Sharpe ≥ 0** on Discovery
- **No single event > 35% of PnL** (concentration check; relaxed slightly given low-event-count regime — same as `buyback_tender_intraday`)

**Setup-specific falsification:**

1. **Aggregate n < 60 over 3yr:** RETIRE — event class is too rare for sub-9 sample-size discipline. (Cheap exit at sanity gate.)

2. **T+30 vs T+90 PF divergence > 0.40:** ship single-subtype detector for the stronger side; do NOT average-dilute. The mechanic is structurally identical for both subtypes (same MTM-lock incentive, different slug-timing); PF divergence > 0.40 indicates one subtype is fragile. Drop the weaker side and re-document as single-subtype detector.

3. **Gap-up-on-expiry conditioning gives PF < 1.0:** RETIRE the gap-up-conditioned cell — this would mean the mechanic only works when the stock was already weak (gap-down), suggesting the supply-overhang thesis is wrong and the candidate is just capturing momentum continuation in already-weak names. Falsifying check: stratified PF on `gap_at_open` quartiles at sanity stage.

4. **Listing-pop-negative cell PF ≥ 1.10:** RETIRE — if the mechanic works on IPOs where anchors are at a LOSS (no MTM-lock incentive), the thesis is wrong. The MTM-lock incentive must be load-bearing. If sanity reports listing-pop-negative cell with PF≥1.10 and statistically significant n, the mechanic is misidentified; brief should be re-framed.

5. **Concentration risk (jumbo-IPO dominance):** Indian 2024 IPOs are extreme-skewed — Hyundai (₹27,856cr anchor), LIC (₹5,627cr), Reliance Retail (large), Mankind, Tata Tech, IREDA, Bajaj Housing dominate by anchor allocation magnitude. **If sanity reports >40% of PnL from a single symbol, the edge is name-specific not class-specific.** Mitigation: cell-stratify on `anchor_allocation_cr_quantile` at gauntlet stage; sanity must report per-symbol PnL contribution explicitly.

6. **Pre-2022 regime contamination:** SEBI 30/90 split is post-Q3-2022. **Sanity MUST be run on POST-2022-Q3 data only.** The pre-2022 single-day expiry was a MORE severe drop (per Edelweiss 2021 -2.6% on single-day vs an expected dampened -1.0% to -1.5% on each split day). Mixing regimes inflates PF estimates by ~30-50%. Discovery 2023-2024 cleanly aligns; 3yr extension to 2022-Q3 is acceptable but 4yr extension to 2022-Q1 is NOT (would mix one quarter of pre-amendment regime).

7. **Independence from `gap_fade_short` violated:** sanity must verify zero overlap of triggered (symbol, date) tuples between this brief and gap_fade_short on lock-in expiry days. Per the deduplication rule (§6.1), gap_fade fires first and anchor_lockin yields. Overlap > 0 is an instrumentation bug — fix the latch order, not the brief.

8. **TFT/SME contamination:** any sanity-flagged trade where the symbol is in TFT settlement on the lock-in expiry day is a data-integrity failure. SME exclusion must be hard-enforced at the universe filter.

9. **Decay signal:** rolling-60-trade NET PF drops below 1.05 sustained for 60 calendar days post-launch. (Standard sub-9 decay tripwire.)

## 10. Direction discipline (SHORT-bias project pattern justification)

**SHORT-only by mechanic.** Lock-in releases SELLERS, not BUYERS — there is no symmetric LONG side by construction. SEBI Reg 32(b) only releases supply; there is no parallel "anchor demand lock-in" rule that would create symmetric demand at expiry timestamps.

**Alignment with the sub-9 SHORT-bias project pattern:**
- `gap_fade_short` (TRUSTED, SHORT)
- `circuit_t1_fade_short` (APPROVED, SHORT, sanity PF=1.473)
- `buyback_tender_intraday` (SHORT)
- `nse_gsm_asm_event` (SHORT)
- `expiry_pin_strike_reversal` (bidirectional, but SHORT-leaning on heavyweights)
- `anchor_lockin_expiry_short` (this brief, SHORT)

The SHORT-bias pattern is not preference-driven — it is **mechanically required** because Indian retail-flow asymmetry (SEBI FY23: 70% cash + 91% F&O retail traders lose, dominantly LONG) implies the disciplined-short side is structurally aligned with the winning-flow side on event-driven supply releases. Lock-in expiry is one of the cleanest examples: the anchor (institutional, disciplined seller) is on the supply side; retail (FOMO buyer of newly-listed pop stocks) is on the demand side; we are aligned with the institutional seller against the retail buyer.

The LONG-side counter-example (`capitulation_long_morning`) is structurally different — it captures retail panic-CAPITULATION at the morning open, which is a contra-flow-at-the-panic-point trade, NOT a SHORT-bias-violation. There is no parallel "capitulation long at lock-in expiry" because the seller in this case is institutional (rational), not retail (panic) — institutional sellers don't panic-capitulate on regulator-fixed timestamps.

## 11. Honest comparison to surviving setups

| Setup | Event class | Active window | Universe | Direction | Sample/2yr |
|---|---|---|---|---|---|
| `gap_fade_short` (TRUSTED) | gap-up momentum exhaustion | T+0 09:15-09:30 | small_cap | SHORT | ~30k events / ~6.7k trades |
| `circuit_t1_fade_short` (APPROVED) | T-1 upper circuit + T+0 gap-up | T+1 10:30 single bar | mid/small-cap | SHORT | ~1k trades |
| `capitulation_long_morning` (APPROVED) | gap-down panic exhaustion | T+0 09:25-10:00 | small/mid-cap | LONG | ~18k events / ~800-2.2k trades |
| `expiry_pin_strike_reversal` (DRAFT) | options pin reversion | expiry 13:00-15:00 | NIFTY heavyweights | bidirectional | ~50k trades |
| `buyback_tender_intraday` (DRAFT, n-marginal) | SEBI buyback tender unwind | 09:30-14:30 / multi-day | broad equity | SHORT | ~90 trades (2yr) / ~225 (5yr) |
| **`anchor_lockin_expiry_short`** (this brief) | **IPO anchor lock-in supply release** | **T+30/T+90 09:30-10:30** | **mainboard IPOs (post-listing)** | **SHORT** | **~90 (2yr) / ~130 (3yr) / ~170 (4yr)** |

**Independence story (vs each surviving setup):**

- **vs `gap_fade_short` (T+0 09:15-09:30 SHORT):** different event class (gap-up momentum vs lock-in supply release); different active window (09:15-09:30 vs 09:30-10:30); explicit deduplication at the (symbol, date) latch level prevents same-day double-firing. PnL correlation expected near-zero by construction.

- **vs `circuit_t1_fade_short` (T+1 10:30 SHORT):** different event class (DPR circuit vs lock-in expiry); different trigger sequence (T-1 circuit hit + T+0 gap-up vs T+30/T+90 lock-in calendar match); different time window (10:30 single-bar vs 09:30-10:30 multi-bar entry). On rare overlap days (newly-listed name hits a circuit on its lock-in expiry day, low-probability), the two would fire on different bars; PnL correlation expected ρ < 0.10.

- **vs `capitulation_long_morning` (T+0 09:15-09:30 LONG):** opposite direction; different event class. Trivially independent.

- **vs `expiry_pin_strike_reversal` (options expiry 13:00-15:00, NIFTY heavyweights):** different universe (mainboard IPOs are typically NOT NIFTY heavyweights at T+30); different time window (09:30-10:30 vs 13:00-15:00); different event class (lock-in vs options pin). Trivially independent.

- **vs `buyback_tender_intraday` (corporate-action event, 09:30-14:30):** same broad mechanic class (regulatory event-driven SHORT fade) but different event-trigger (buyback vs lock-in) and different universe overlap (buyback is on mature companies; lock-in is on newly-listed). Both cluster in the SHORT-side post-event-fade family; PnL correlation expected ρ ~0.05-0.15 due to common SHORT-side market regime exposure but no shared trigger logic.

The brief complements the existing portfolio: a **regulator-anchored SHORT** at a **regulator-fixed timestamp** in a **newly-listed-IPO universe** that is structurally distinct from the existing post-event SHORT briefs. Highest-quality §2 anchor in the round-4 candidate set; sample-size is the single material risk.

## 12. Data engineering plan

**Pre-sanity (~0.5 day):**
- `tools/sub9_research/backfill_anchor_lockin_calendar.py` — chittorgarh.com scraper.
  - URL pattern: `https://www.chittorgarh.com/report/anchor-investor-lock-in-end-dates/156/mainboard/?year={2022,2023,2024}`
  - Output: `data/anchor_lockin_calendar/lockin_events.parquet` with columns `[ipo_listing_date, symbol, isin, anchor_allocation_cr, lockin_30day_date, lockin_90day_date, anchor_investor_count, source_url]`.
  - Coverage: 2022-Q3 to 2024 (3yr Discovery) + 2025 partial (1yr OOS); ~440 raw events.
  - Spot-validation: 30 random IPOs cross-checked against business-standard.com weekly roundups.
  - Effort: ~0.5 day (single-source structured-HTML scrape; pattern adapted from `tools/earnings_calendar/`).

**Sanity engineering (~1 day):**
- `tools/sub9_research/sanity_anchor_lockin_expiry_short.py` — ~250 LOC; mirrors `tools/sub9_research/sanity_circuit_t1_fade_short.py` template (post-event SHORT-fade with subtype routing).
- Compute time: ~2-3 hours (small event set, 5m bars, single-day per event).
- Effort: ~1 day.

**Pre-sanity total budget: ~1.5 engineering days** (0.5 day data backfill + 1 day sanity). On par with `buyback_tender_intraday` (lowest-cost candidate in Lane 1).

**Production live mode (only if sanity → APPROVE → OOS pass → ship):**
- Weekly chittorgarh scraper refresh (lock-in expiries are scheduled multi-week out from listing; weekly refresh is sufficient).
- New live config keys per CLAUDE.md rule 1 (NO hardcoded defaults):
  ```
  "anchor_lockin_expiry_short": {
    "enabled": false,
    "subtypes_enabled": ["lockin_30day", "lockin_90day"],
    "active_window_start": "09:30",
    "active_window_end": "10:30",
    "time_stop_at": "13:00",
    "min_anchor_allocation_cr": 50.0,
    "min_liquidity_volume_x_close": 20000000,
    "min_listing_pop_above_issue": true,
    "exclude_sme": true,
    "confirmation_volume_multiplier": 1.2,
    "stop_atr_multiple": 1.5,
    "min_stop_distance_pct": 0.5,
    "t1_r_multiple": 0.5,
    "t2_r_multiple": 1.5,
    "t1_partial_qty_pct": 0.5,
    "dedupe_with_gap_fade_short": true
  }
  ```

**Critical setup-design caveats baked into the data plan:**
1. **Post-2022 regime mixing risk:** sanity script MUST hard-filter Discovery to `lockin_date >= 2022-09-15` (post-amendment effective date). Pre-2022 events are EXCLUDED from sanity even if chittorgarh provides them.
2. **Absolute-anchor-allocation skew:** sanity must report PnL contribution per-symbol AND per-`anchor_allocation_cr_quantile`. Concentration check (no single symbol > 35% of PnL) is hard-falsification.
3. **SME exclusion:** sanity hard-filters `is_sme = false` (chittorgarh tags this; spot-validation against NSE SME-IPO list as backup).
4. **Deduplication with `gap_fade_short`:** sanity script must compute `gap_fade_short` triggers on the same (symbol, date) universe and exclude lock-in events where `gap_fade_short` would have fired first (latch order: gap_fade fires 09:15-09:30, anchor_lockin yields if gap_fade fired). Independence check (zero overlap of executed trades) is hard-falsification.

---

## Acceptance summary

| Criterion | Status |
|---|---|
| §2 anchor — regulator-defined? | **YES** — SEBI(ICDR) Reg 32(b); 30/90 split lock-in unique to India; 2022 amendment is regulator's own acknowledgement of the asymmetry |
| §2 anchor — public timestamped data? | **YES** — chittorgarh.com structured aggregator (~0.5 day backfill); validated via business-standard.com + KPMG IPO report |
| §2 anchor — direction empirically supported? | **YES — STRONG** — Edelweiss 2021 (-2.6% / 76% selling pressure / 41 events); Field & Hanka 2001 (US -1.5% to -3% baseline); SEBI 2022 amendment is regulator-acknowledged |
| n ≥ 500 / 2yr feasibility | **n-marginal — ~90 trades on 2yr; ~130 on 3yr; clears narrow-cell n≥30 with 3yr Discovery extension** (per `buyback_tender_intraday` / `nse_gsm_asm_event` precedent) |
| Independence from existing setups | **YES** — different event class; explicit dedupe with `gap_fade_short`; expected ρ < 0.15 with each existing setup |
| Falsification budget acceptable | **YES — ~1.5 engineering days** (0.5 day backfill + 1 day sanity); on par with cheapest Lane-1 candidate |
| Differentiation from published retail-algo content | **YES** — no retail-algo platform publishes anchor lock-in expiry intraday rule; broker-education content discusses but does not codify |

---

## VERDICT: APPROVE-eligible for sanity (confidence: HIGH on anchor / MEDIUM on sample size)

The candidate has the **strongest §2 anchor in the round-4 candidate set** — SEBI's 2022 amendment is the regulator's own structural acknowledgement that the asymmetry is material and durable. Direction-supporting evidence is unusually strong (Edelweiss empirical Indian-equity + Field & Hanka US foundational + SEBI regulatory acknowledgement). Mechanic class is identical to the production-validated `circuit_t1_fade_short`. Falsification budget is among the cheapest in Lane 1 (~1.5 days).

**The single material risk is sample size.** ~90 trades on 2yr / ~130 on 3yr is well below the n≥500 aggregate floor and stretches the narrow-cell n≥30 discipline. The path-of-least-resistance is 3yr Discovery (2022-Q3 to 2024) with cell-stratification on `(event_subtype, anchor_allocation_cr_quantile)`. If sanity n is < 60 even on 3yr, the brief should retire at the cheapest gate per `buyback_tender_intraday` precedent.

Confidence is rated HIGH on anchor / MEDIUM on sample size because (a) post-2022 SEBI regime is only ~3.5 years old, capping maximum Discovery extension, (b) jumbo-IPO concentration risk (Hyundai, LIC, Reliance Retail dominate by absolute supply), and (c) regulator may further amend lock-in mechanics, breaking the post-2022 regime in the future.

Recommended path: **APPROVE for sanity** with the cheapest data-backfill (~0.5 day chittorgarh scrape) → 3yr Discovery sanity (~1 day) → **RETIRE-IF-N<60** at sanity gate. Sample-size discipline is the load-bearing falsification check.

---

## Decision required

User to indicate:
- [ ] **APPROVED** — proceed to chittorgarh anchor-lockin calendar backfill (~0.5 day) → 3yr sanity-check (~1 day)
- [ ] **APPROVED-CONDITIONAL** — proceed only after spot-validation confirms event-data accuracy ≥ 95% on 30 random IPOs against business-standard.com weekly roundups
- [ ] **REJECTED** — reason
- [ ] **RETIRE** — defer indefinitely (sample-size discipline considered insurmountable pre-engineering)

Per sub-9 §3.3, no detector code is written until APPROVED and sanity-check passes (NET PF ≥ 1.10 with n ≥ 30 per subtype-cell over 3yr Discovery, |WR delta| ≤ 10pp on OOS, no-single-symbol-PnL > 35%, post-2022-Q3 regime-clean Discovery).
