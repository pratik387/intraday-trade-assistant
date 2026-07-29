# §3.3 Brief: `asm_gsm_stage_transition`

**Sub-project:** #9 (microstructure-first redesign), Round-8 / Lane 1 Indian-microstructure candidate #1
**Status:** **KILLED at Phase 2, 2026-07-29** (was DRAFT / pre-sanity disposition required since
2026-05-08; the blocking ASM/GSM stage calendar finally materialised 2026-07-28, 1.07M rows).
**Date:** 2026-05-08 (brief) / 2026-07-29 (kill)

**KILL RECORD (2026-07-29).** Gate-B PASSED — 486 tradeable strict stage-move events
(era_A 314 / era_B 151), so this was NOT a frequency block. Killed on evidence:
- **A5-b decisive:** the drift is real in RELATIVE terms (+0.14pp vs same-symbol non-event
  baseline, both eras, t≈2.3 on Discovery) but **net-negative in BOTH eras absolutely** —
  10:30 stop, un-gated: net PF 0.903 (era_A) / 0.807 (era_B). Raw drift ≈ +0.17-0.24% against
  a 0.3065% round trip: **the edge is about half the cost.**
- The brief's OWN pre-registered gap+confirmation gates make it worse (PF 0.620 / 0.300) and
  collapse n to 42/14 — below the brief's own n≥30/cell floor. The event pool passes; the
  brief's construction does not.
- **Salience runs BACKWARDS** (Stage II > Stage I > Stage III; Stage III era_B PF 0.217) — a
  forced-unwind mechanism must strengthen with stage severity. Strongest single piece of
  evidence against the stated mechanism.
- **Short leg partly does not exist:** 26% of strict events are Stage III trade-for-trade (no
  intraday square-off), 0% are F&O-present, and real margin is 1× not 5× (the §3
  `mis_leverage >= 1.0` filter is vacuous — nse_all.json is a current snapshot, and ASM names
  carry 100% margin by rule). Shortable-subset restriction does not rescue: PF 0.905/0.640.
- Monotone decay 2023→2026 (0.990 / 0.847 / 0.688 / 0.497); every variant negative post-Oct-2025.
- **GSM arm unobservable** — NSE GSM circulars are image PDFs; only 19 tradeable GSM events
  survive via BSE ISIN-mapping, so the §3 GSM III→IV hypothesis cannot be tested at all.
- Context change since drafting: `circuit_t1_fade_short` was RETIRED 2026-05-23, removing both
  this brief's directional precedent (§5/§2 Test-3) and its §8 Path-C merger escape.
**Residual lead (NOT a rescue — needs its own brief):** `stage_move × ASM_LONGTERM ×
hold-to-EOD` is era-consistent on the absolute statistic (net PF 1.247 / 1.138 / 1.098
post-Oct-25) — but EOD hold is a swept dimension, not the brief's 10:30 stop, it is barely
above the 1.10 floor, and it surfaced from a 6,240-cell sweep.
Evidence: `reports/sub9_sanity/_asm_gsm_stage_transition_phase2.csv` (+ `_trades.csv`),
`tools/sub9_research/phase2_asm_gsm_stage_transition_signature.py`, ledger line 2026-07-29.

**Predecessors / context:**
- `specs/2026-05-01-sub-project-9-microstructure-first-redesign.md` (defines §3.3 gate process, locked thresholds)
- `specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md` (data-broad universe + n≥500/2yr methodology)
- `specs/2026-05-07-sub-project-9-brief-capitulation_long_morning.md` (gold-standard §3.3; Discovery PF 1.24 cell, Indian-anchor)
- `specs/2026-05-07-sub-project-9-brief-nse_price_band_approach_short.md` (DPR-anchored gold-standard for SEBI-rule briefs)
- `specs/2026-05-07-sub-project-9-brief-nse_gsm_asm_event.md` — **PREDECESSOR (Round-5)** focused on additions/removals, status: drafted but not yet sanity-passed. **This brief refines the predecessor by narrowing to *stage transitions* specifically (Stage I→II, II→III, III→IV, and reverse demotions) rather than the broader ADD/REMOVE event set.** Stage transitions are higher-conviction events because they involve continuous-coverage stocks already in the surveillance regime; the price reaction to a transition is unambiguously a regulator-driven stage signal, not a noisy first-tag event.
- `specs/2026-05-08-sub-project-9-brief-squeeze_release.md` (RETIRE-PRE-SANITY example for contrast)

This is the Lane-1 Indian-microstructure candidate where the anchor is **SEBI/NSE-codified surveillance program (GSM stages 1-4 + ASM stages 1-4)**, with an event window narrowed to **stage transitions** (between-stages moves on already-listed-surveillance stocks). The predecessor `nse_gsm_asm_event` brief (Round-5) addressed the broader event set; that brief stalled on data-engineering scope (~2.5 days for full GSM/ASM HTML/PDF backfill). This refined brief preserves the regulatory anchor while narrowing the event population to the highest-conviction subset.

---

## 1. Asymmetry

**Name:** Indian-equity intraday reaction to NSE/BSE surveillance-stage transitions (GSM Stage I→II→III→IV and ASM equivalents).

**Specific mechanism (chained):**

1. **Stage-transition trigger.** A stock currently on GSM/ASM moves up or down a surveillance stage in the post-market 17:30-18:30 IST circular. Stage promotion (e.g., II→III, III→IV) tightens trading restrictions: 100% margin, T+0 settlement, 5%/2%/0% price band, F&O exclusion, no MIS leverage. Stage demotion (e.g., II→I, III→II) loosens them.
2. **Forced unwinding (promotion side, T+1 09:15).** Operators and retail holders face mechanical liquidation pressure overnight: 100-200% margin requirement at T+1 makes intraday rolling impossible, F&O contracts (if previously eligible) halt, MIS leverage drops to 1× from prior 5×. The marginal seller at T+1 09:15 is a forced-unwinder, not a discretionary participant.
3. **Relief-rally retail FOMO (demotion side, T+1 09:15).** Stage demotion is broadcast on Moneycontrol, ET, Kite app stage-tags as a "regulator gives clearance" signal. Retail piles in at T+1 open replicating the post-circuit-fade FOMO pattern (`circuit_t1_fade_short`'s validated machinery). Disciplined trade is to FADE the FOMO bounce after 09:30 confirmation candle.
4. **Stage-transition asymmetry vs first-tag asymmetry.** First tag (un-flagged → Stage I) is noisy: many first-tags are data-cleansing/threshold-hit events that do not prefigure a real regime change. Stage *transitions* on already-listed surveillance stocks are higher-conviction because the regulator is explicitly escalating/de-escalating an ongoing concern.

**Why this is asymmetric (not just statistical):** the transition event is a regulator-driven binary state change with mechanical clearing-layer consequences (margin %, settlement type, band %, F&O availability). Participants cannot adapt away the mechanic — the exchange enforces it at order-entry validation. The information content is "regulator's ongoing surveillance has reached a new threshold of concern", which is qualitatively different from generic price-volume anomaly signals that retail screeners surface.

## 2. Indian-microstructure anchor (THE CRITICAL GATE)

Per round-7 / round-8 mandate, three tests:

**Test 1 — Anchor is regulator-defined or NSE-published?** PASS.
- GSM (Graded Surveillance Measure): joint SEBI + NSE/BSE framework, March 2017 introduction. Reference: NSE GSM page https://www.nseindia.com/companies-listing/securities-information-gsm and SEBI master circular on surveillance.
- ASM (Additional Surveillance Measure): exchange-driven framework, March 2018. Reference: NSE ASM https://www.nseindia.com/reports?archives=%5B%7B%22name%22%3A%22ASM%22%7D%5D
- Stage definitions, trigger criteria, and consequences are all SEBI-codified. No global equivalent exists — US Reg-SHO threshold lists, EU short-sale-reporting, and Tokyo/HKEX price-monitoring lists are structurally different (they don't impose graded margin/settlement/F&O changes).

**Test 2 — Has timestamped event data accessible from public sources?** PASS-with-engineering-cost.
- NSE publishes daily ASM/GSM circulars at 17:30-18:30 IST (post-market) via Exchange Communique archive https://www.nseindia.com/regulations/exchange-communique-circulars
- ASM daily CSV downloads available (per predecessor brief's analysis): https://www.nseindia.com/reports?archives=%5B%7B%22name%22%3A%22ASM%22%7D%5D
- GSM is HTML-rendered with stage-snapshot history; requires daily snapshot scrape for transition reconstruction.
- Backfill cost: estimated 2-3 engineering days (per predecessor brief §13.1). Sanity uses partial (ASM-only) backfill as cheaper proxy.

**Test 3 — Direction is empirically supported?** PASS-conditional.
- Stage-promotion (forced unwinding) → SHORT on T+1 09:30 confirmation. Direction supported by the same machinery validated for `circuit_t1_fade_short` (forced-flow asymmetry: operators with concentrated positions face mechanical liquidation pressure).
- Stage-demotion (relief-rally) → SHORT (FADE the FOMO bounce) on T+1 09:30 confirmation. Direction supported by `circuit_t1_fade_short`'s direct precedent (T+1 retail-FOMO fade is the validated pattern).
- Both directions converge on SHORT-only at T+1 09:30 — consistent with the surviving sub-9 setup library (`gap_fade_short`, `circuit_t1_fade_short`, both SHORT-only) and the sub7/8 11-failure long-bias caution.

**§2 verdict: PASS all three tests. Brief is APPROVE-eligible for sanity.**

## 3. Universe & cell hypothesis

**Per round-3 broadened-universe rule:**

- Cap segment: ALL of `large_cap`, `mid_cap`, `small_cap`, `micro_cap` admissible at sanity. GSM/ASM is concentrated in mid-/small-/micro-cap names (large-caps rarely enter surveillance), but no pre-lock at sanity.
- F&O 200 NOT pre-locked. ASM Stage III/IV typically EXCLUDES the stock from F&O, so F&O-membership is a *consequence* of the event, not a pre-filter.
- Liquidity gate: 20-day median daily volume ≥ 50K shares AND `mis_leverage ≥ 1.0` from `nse_all.json` (must be MIS-eligible to short intraday). ASM Stage IV stocks at 0% band are excluded — no intraday move possible.
- HARD data dependencies: (i) GSM/ASM event calendar with stage-transition timestamps (NEW; see §11), (ii) 5m enriched feathers (existing).

**Cell hypothesis (gauntlet Stage 3, post-sanity):** strongest cells expected in (cap_segment=small_cap × stage_change=III→IV × event_type=promotion) and (cap_segment=mid_cap × stage_change=III→II × event_type=demotion). Promotion-on-small-cap is the operator-forced-unwind cell; demotion-on-mid-cap is the retail-FOMO-fade cell. Cell selection at gauntlet, not at sanity.

**Symbol count after liquidity gate (event-day basis):** estimated 60-150 transition events / yr across ~1500 NSE listed stocks.

## 4. Persistence

Three structural reasons:

1. **GSM/ASM frameworks are SEBI-codified circulars.** Removal would require formal SEBI policy reversal — historically improbable. Stage transitions follow rule-based trigger criteria (rolling-90/180-day price-volume metrics, fundamental thresholds), not discretionary regulator action. Reference: SEBI master circular on Surveillance https://www.sebi.gov.in/legal/master-circulars
2. **Margin/band/settlement consequences are mechanical.** Once a stock transitions stages, leverage caps + settlement type + band width are enforced at the exchange clearing layer; no participant adaptation can bypass.
3. **Capacity is unsaturated.** None of the round-3-audited Indian retail-algo platforms (Streak, Stoxra, Stratzy, Wright, Tickertape, Algotest) publish GSM/ASM event playbooks. Operators actively avoid the topic because it implicates their own concentrated holdings. Greenwood/Sammon decay pressure is therefore minimal — the absent-from-public-publishing argument is the strongest in the sub-9 candidate library.

## 5. Evidence

**Regulatory primary sources:**
1. **NSE GSM information page** — https://www.nseindia.com/companies-listing/securities-information-gsm
2. **NSE ASM reports** — https://www.nseindia.com/reports?archives=%5B%7B%22name%22%3A%22ASM%22%7D%5D
3. **SEBI master circular on Surveillance** — https://www.sebi.gov.in/legal/master-circulars (search "surveillance" / "ASM" / "GSM")
4. **NSE Exchange Communique / Circulars** — daily archive https://www.nseindia.com/regulations/exchange-communique-circulars

**Peer-reviewed Indian-equity evidence:**
5. **Sehgal, Subramaniam et al., *Pacific-Basin Finance Journal* 2024** — Indian-equity surveillance-list inclusions show asymmetric volume + volatility responses; same paper used for `circuit_t1_fade_short`. https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640
6. **Internal precedent — `circuit_t1_fade_short`** — the T+1-FOMO-fade machinery is production-validated (NET PF 1.473 on n=654). This brief tests the same machinery on a different regulatory event class (surveillance stage transition vs price-band hit).

**Published retail-algo content audit:** none of Streak, Stratzy, Wright, Tickertape, Algotest publish GSM/ASM stage-transition playbooks (verified via round-3 audit + round-5 re-audit). Capacity-unsaturated.

## 6. Mechanic

**Setup name:** `asm_gsm_stage_transition`
**Side:** SHORT-only (both promotion and demotion converge on T+1 fade short).
**Bar timeframe:** 5m
**Active window:** T+1 09:30 (single-bar entry)

**Sequence:**

1. **T-evening event detection (post-17:30 IST):**
   - Parse NSE daily ASM/GSM circular (auto-pull from Exchange Communique archive)
   - Compare against prior session's GSM/ASM stage snapshot per symbol
   - Flag transitions: (symbol, prev_stage, new_stage, list_type, direction) where direction ∈ {PROMOTION, DEMOTION}
   - Stage filter: ASM Stages I-III + GSM Stages I-III. Stage IV excluded (0% band, no intraday move).
2. **T+1 09:15 open classification:**
   - Compute gap = (open_09_15 − pdc) / pdc × 100.
   - PROMOTION direction: if gap ≤ −5%, stand down (most of the move is already priced in). Trigger if −5% < gap < +1%.
   - DEMOTION direction: if gap ≥ +5%, stand down (FOMO bounce already exhausted). Trigger if −1% < gap < +5%.
3. **T+1 09:30 confirmation candle (5m bar 09:25-09:30):**
   - Bearish: close < open AND close < 09:15 bar's low.
   - Entry = 09:30 bar's CLOSE (SHORT).
   - Latch: one fire per (symbol, T+1).
4. **Stop-loss:** T-1 close × 1.005 (PROMOTION side — ceiling above prior-day close, surveillance stigma should not allow re-rating up); 09:15 bar's high × 1.003 (DEMOTION side — FOMO ceiling).
5. **Targets (R-multiple, anchor type = arithmetic_R):** T1 = 1R (50% qty, BE-trail after); T2 = 2R (50% qty).
6. **Time stop:** 10:30 IST (matches `circuit_t1_fade_short`'s window).

**target_anchor_type:** `arithmetic_R`. Stage transitions do not have a clean structural level (no DPR band like `circuit_t1`); R-multiple is the disciplined anchor.

## 7. Independence from existing production setups

- **vs `gap_fade_short` (TRUSTED, T+0 09:15-09:30):** different event (T+1 surveillance circular vs T+0 gap), different time window (T+1 09:30 vs T+0 09:15-09:30), no overlap of trigger populations by event-day construction.
- **vs `circuit_t1_fade_short` (APPROVED, T+1 10:30):** different event class (surveillance stage vs DPR hit), different time-of-day (09:30 vs 10:30). Possible co-occurrence: stock that hit DPR T+0 AND received GSM/ASM stage promotion at T+0 17:30 → both detectors fire. Sanity-check must enforce mutual exclusion at trigger-time: if symbol is in `circuit_t1_fade_short`'s prior-day-hit list, exclude from this brief's universe (prefer the validated detector).
- **vs `capitulation_long_morning` (APPROVED, LONG, T+0 09:15-09:30):** opposite direction + different event. Independence is mechanical.
- **Aggregate PnL correlation expectation:** ρ < 0.15 vs each existing setup.

## 8. Sample-size feasibility

**Annual event volume estimate:**
- ASM additions: ~50-100 / yr (NSE ASM CSV daily counts of new entries, summed)
- GSM additions: ~20-30 / yr
- ASM stage transitions (within-list moves): ~30-50 / yr
- GSM stage transitions: ~10-20 / yr
- ASM removals: ~50-100 / yr
- GSM removals: ~20-30 / yr
- **Total transitions (this brief's scope, including ADD/REMOVE as Stage 0↔I transitions):** ~150-300 events / yr.

**Filter survival rates:**
- T+1 09:15 gap-window filter (avoid pre-priced moves): ~70% pass.
- T+1 09:30 bearish-confirmation candle: ~50-55% pass.
- Liquidity + MIS-eligibility filter: ~80% pass.
- Cross-detector exclusion (vs `circuit_t1_fade_short`): ~95% pass.
- Combined survival: 0.70 × 0.52 × 0.80 × 0.95 ≈ **0.28 = 28% of raw events** survive.

**Expected Discovery trade count:** 200 events/yr × 2yr × 0.28 = **~110 trades / 2yr**. **n ≥ 500 IS NOT MET.**

**n-marginal verdict.** The brief fails the round-3 hard floor of n=500. Two paths to clear:

**Path A — Narrow-cell n≥30 floor (per predecessor brief):** the predecessor `nse_gsm_asm_event` brief explicitly invoked the round-5 narrow-cell n≥30 floor (line 134: "n < 30 trades per cell — narrow-cell floor, NOT the 500 floor used elsewhere because GSM/ASM events are intentionally rare"). The narrow-cell floor is the per-cell threshold; the aggregate n=110 over 2yr corresponds to 55/yr, which clears n≥30/cell easily for 2 cells (promotion / demotion).

**Path B — Discovery period extension to 3yr:** extending Discovery to 2022-2024 gives ~330 trades, still short of n=500 but ~3× the narrow-cell floor.

**Path C — Combined with existing production setups (`circuit_t1_fade_short`) under a unified "T+1-regulatory-fade" superdetector:** if both detectors have similar PF and similar mechanic (T+1 09:30 fade of regulator-event-overnight retail/operator imbalance), they could be merged with cell-stratification rather than running as separate detectors. This is a **post-sanity architectural decision**, not a sanity-time concern.

**Recommended sanity gate:** run sanity with narrow-cell n≥30 floor explicit in §9 (per predecessor convention), extend Discovery to 3yr if n<60 on 2yr, and re-evaluate merger with `circuit_t1_fade_short` only if PF profile matches.

## 9. Falsification criteria

**Locked thresholds (per round-3 + predecessor brief):**
- **NET PF ≥ 1.10** on Discovery (fees + taxes + slippage; Indian fee model)
- **n_trades ≥ 30 per cell** (narrow-cell floor; aggregate n≥60 over 2yr; ≥110 over 3yr)
- **|WR delta| ≤ 10pp** Discovery vs OOS Validation (2025)
- **NET Sharpe ≥ 0** on Discovery
- **No single event > 30% of PnL** (concentration check)

**Setup-specific falsification:**
1. **Demotion side inverts thesis (LONG bounce edge instead of fade-able FOMO).** If demotion cell shows persistent LONG-bounce PF ≥ 1.10, the sub7/8 long-bias caution applies: drop demotion side, ship promotion-only.
2. **Cross-detector overlap >10% with `circuit_t1_fade_short`.** Empirical check: intersect this brief's trigger set with `circuit_t1`'s on overlapping (symbol, T+1) keys. If overlap >10%, G2 (cross-detector exclusion) gate is buggy — fix before sanity-pass.
3. **NSE circular publish-feed unreliable.** If 17:30 daily circular has >5% missing/late publish dates over 2yr backfill, live trigger reliability is at risk. Spot-validation against 30 random circulars must show ≥ 95% accuracy.
4. **Capacity-saturation signal.** If rolling-12-month PF in 2024 ≤ 60% of 2022-2023 PF, retail-algo platforms may have begun publishing surveillance event playbooks — re-audit and re-evaluate.
5. **Decay signal:** rolling-60-trade NET PF drops below 1.05 sustained for 60 calendar days post-launch.

## 10. Falsification budget

**Data engineering:**
- GSM/ASM event calendar backfill (per predecessor brief §13.1): ~2-2.5 engineering days. Includes ASM CSV scraper (4-5h), GSM HTML/PDF parser (4-5h), spot-validation tool (2h), backfill run (4-6h wall-clock).
- Cheaper proxy for sanity-only: ASM-CSV-only backfill (~1 day; covers ~70% of events; defer GSM HTML parsing until post-approval).

**Sanity engineering:**
- `tools/sub9_research/sanity_asm_gsm_stage_transition.py` — ~250 LOC; mirrors predecessor's sanity script template (`sanity_circuit_t1_fade_short.py`).
- Compute time: ~2-3 hours (small event set, single-bar entry).
- Effort estimate: ~1 engineering day for sanity script + run.

**Total falsification budget: ~2.5-3.5 engineering days** (1-1.5 day data backfill + 1 day sanity + 0.5 day OOS + cross-detector audit). This is comparable to `earnings_day_intraday_fade` (~1.5 days) and lower than `nse_price_band_approach_short` (~3.5 days). Acceptable allocation.

## 11. Data engineering plan

**Pre-sanity (cheaper proxy, ASM-only):**
- `tools/sub9_research/backfill_asm_calendar.py` — daily ASM CSV scraper; reuses NSE auth pattern from `tools/option_chain/_nse_bhavcopy_client.py`.
- Output: `data/surveillance_calendar/asm_events.parquet` with columns `[event_date, symbol, list_type, prev_stage, new_stage, event_type, source_circular_url, trade_date]` where `trade_date = next_trading_day(event_date)`.
- Coverage: 2023-2024 daily ASM snapshots; ~150-300 transition events.
- Effort: ~1 day.

**Post-sanity (GSM extension, only if sanity PASSES):**
- `tools/sub9_research/backfill_gsm_calendar.py` — daily GSM HTML/PDF parser. Cross-validate via Exchange Communique archive PDFs.
- Effort: ~1.5 additional days.

**Production live mode:** daily 18:30 IST scraper against same endpoints; incremental 1-day window. Same pattern as `earnings_day_intraday_fade`'s live calendar.

**Config keys (after sanity passes):**
```
"asm_gsm_stage_transition": {
  "enabled": false,
  "active_window_entry": "09:30",
  "time_stop_at": "10:30",
  "min_gap_pct_promotion": -5.0,
  "max_gap_pct_promotion": 1.0,
  "min_gap_pct_demotion": -1.0,
  "max_gap_pct_demotion": 5.0,
  "stop_buffer_promotion_pct": 0.005,
  "stop_buffer_demotion_pct": 0.003,
  "t1_r_multiple": 1.0,
  "t2_r_multiple": 2.0,
  "t1_partial_qty_pct": 0.5,
  "stage_filter": {"asm_min": 1, "asm_max": 3, "gsm_min": 1, "gsm_max": 3},
  "exclude_circuit_t1_overlap": true,
  "min_liquidity_volume": 50000
}
```

---

## Acceptance summary

| Criterion | Status |
|---|---|
| §2 anchor — regulator-defined? | **YES** — SEBI/NSE GSM+ASM circulars; no global equivalent |
| §2 anchor — public timestamped data? | **YES** — NSE daily circulars + Exchange Communique archive (~2.5 days backfill) |
| §2 anchor — direction empirically supported? | **YES** — both promotion (forced-unwind) and demotion (FOMO-fade) converge on T+1 09:30 SHORT, validated machinery from `circuit_t1_fade_short` |
| n ≥ 500 / 2yr feasibility | **n-marginal — ~110 trades on 2yr; clears narrow-cell n≥30 per cell convention; extension to 3yr → ~330** |
| Independence from existing setups | **YES** — different event class + cross-detector exclusion gate vs `circuit_t1_fade_short` |
| Falsification budget acceptable | **YES** — ~2.5-3.5 engineering days |

---

## VERDICT: APPROVE-eligible for sanity (confidence: HIGH)

The candidate has the strongest §2 anchor in the Lane-1 cohort: SEBI/NSE-codified surveillance program with (a) clear regulatory authority, (b) mechanical clearing-layer enforcement, (c) capacity-unsaturated (no published retail-algo playbook), (d) DIRECT VALIDATED PRECEDENT in `circuit_t1_fade_short`'s T+1-fade machinery. The primary risk is **n-marginal sample size**, mitigated by the narrow-cell n≥30 convention from the predecessor brief and the optional 3yr Discovery extension. Data engineering cost is moderate (~2.5 days for full GSM+ASM, ~1 day for ASM-only proxy).

---

## Decision required

User to indicate:
- [ ] **APPROVED** — proceed to ASM-only data backfill (~1 day) → sanity-check (~1 day)
- [ ] **APPROVED-CONDITIONAL** — proceed only after full GSM+ASM backfill (~2.5 days)
- [ ] **REJECTED** — reason
- [ ] **RETIRE** — defer indefinitely

Per sub-9 §3.3, no detector code is written until APPROVED and sanity-check passes (NET PF ≥ 1.10 with n ≥ 30 per cell over 2yr or 3yr Discovery, |WR delta| ≤ 10pp on OOS).
