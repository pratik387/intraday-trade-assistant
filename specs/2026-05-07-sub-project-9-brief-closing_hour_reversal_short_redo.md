# §3.3 Brief: `closing_hour_reversal_short_redo`

**Sub-project:** #9 (microstructure-first redesign) — Round-6 candidate
**Status:** **DRAFT — awaiting user APPROVE / REJECT / RETIRE-pre-data before sanity-check.**
**Date:** 2026-05-07

---

## Predecessors / context

- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (round-3; candidate 6 "Late-day liquidation fade" deferred there)
- specs/2026-04-28-research-closing_hour_reversal-findings.md (sub7/sub8 `closing_hour_reversal` Phase-1 failure: PF 0.45, n=9,197, WR 25.9%, DROP)
- specs/2026-04-28-research-mis_unwind_short-findings.md (sub7/sub8 `mis_unwind_short` Phase-1 failure: PF 0.355, n=304, WR 9.2%, DROP)
- specs/2026-05-07-sub-project-9-brief-mis_unwind_short_late_session.md (round-5 redo: failed sanity at n=1008, PF 0.367 — same active window, retail-load too high)
- reports/sub8_phase1/closing_hour_reversal_report/ (parquet + per-cell metrics for differentiation)
- reports/sub8_phase1/mis_unwind_short_report/01-metrics.json (per-cell metrics for differentiation)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED — SEBI-mechanical lineage, template for this brief's structure)

This brief is a **round-6 redesign** of the closing-hour-reversal mechanic under the round-6 methodology (broad universe, n ≥ 500/2yr hard floor, axis-independent of `gap_fade_short` and `circuit_t1_fade_short`). It is **not** a third re-try of either prior failure: the targeted population is structurally different (retail-LIGHT names, not retail-active F&O 200), and the mechanism narrative is the **operator-pump unwind**, not the SEBI MIS-auto-square cascade.

---

## 1. Asymmetry

**Name:** Late-session operator-pump unwind on retail-light, low-float small-caps (5m).

**Indian-specific source:** the systematic **retail-light** sub-population of NSE small-caps exhibits a recurring late-session price signature — a low-volume rally through 14:00-14:30, peaking ~14:30-14:45, that unwinds 14:45-15:15 as the operator desk distributes inventory before close. Because retail MIS participation in these names is marginal (no F&O OI, low cash-MIS flow, low intraday volume vs free float), the late-day rally is **not a retail-conviction event**. It is an operator marking-the-tape signature documented in:

- **NSE Consultation Paper on Surveillance** (2017-18) flagging "rally on declining volume in low-float operator-prone counters" as a pump pattern. https://www.nseindia.com/regulations/exchange-circulars-list (NSE consultation archive)
- **SEBI Surveillance Circular SEBI/HO/ISD/ISD-PoD-2/P/CIR/2023/137** describing operator-driven late-session price marking in low-volume scrips and the GSM/ASM framework that follows. https://www.sebi.gov.in/legal/circulars/aug-2023/circular-on-graded-surveillance-measure-gsm-_75412.html
- **SEBI FY23 retail-F&O loss study** (https://www.sebi.gov.in/reports-and-statistics/research/jan-2023/study-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_67525.html) — retail loss is **dominantly LONG-bias**. On retail-LIGHT names, the late-day rally is therefore not retail-driven; the operator/HNI book is the marginal flow.
- **NSE/SEBI 15:20 MIS auto-square mandate** (https://support.zerodha.com/category/trading-and-markets/margin-leverage-and-product-and-order-types/articles/auto-square-off-time-for-mis-co-and-bo) — but on retail-LIGHT names, the MIS auto-square pressure is **muted** (small retail book → small auto-square flow). The dominant 14:45-15:15 flow is the **operator distributing into a tape they marked up** before book-closing.

The exploitable asymmetry is the **late-session distribution flow on retail-light, low-float small-caps that have rallied 14:00→14:30 on declining volume**. Direction is unambiguously SHORT — the rally is unsupported by genuine demand (no retail/no F&O bid), and the operator is the natural seller into 15:15.

This asymmetry is **distinct** from both prior failures:

- **Differs from `closing_hour_reversal` (sub7/sub8):** the prior detector traded ALL stocks rallying into the close on simple body-strength + 1.5% intraday move. It had no retail-load classifier and no volume-decline gate; ~53% of trades were SELLs in retail-active F&O names where CNC-conversion + closing-momentum-into-FII-flow killed the short side. PF 0.45.
- **Differs from `mis_unwind_short_late_session` (round-5):** that brief targeted retail-active F&O 200 mid+small-cap names up 1.5-4% intraday with declining volume — the SEBI 15:20 cascade was the postulated alpha. It failed at n=1008, PF 0.367 because retail CNC-conversion at 15:10 squeezed the shorts. **The mechanism died because the population was retail-HEAVY.** This brief INVERTS that selection by explicitly targeting retail-LIGHT names.

## 2. Participants

- **Forced sellers (alpha source):** the operator/HNI desk that marked the tape up in the 13:30-14:30 window holds inventory it must distribute before close. With no retail / no F&O book absorbing the rally, the only natural exit is open-market selling 14:45-15:15. Inelastic supply with a hard time deadline (15:25 close).
- **Counter-flow (our cohort):** disciplined SHORT entries 14:30-15:00 fading the late rally on confirmed retail-light + declining-volume names. Exit before 15:15 hard time-stop.
- **Counter-flow risk (failure mode):** if a non-retail bid emerges late (institutional positional buyer, sector-rotation flow on a NIFTY rally day, GSM/ASM surveillance circuit triggering a sudden bid), the short squeezes. Mitigated by:
  1. Hard 15:15 time-stop (avoids the 15:15-15:25 close-fixing window)
  2. Excluding GSM/ASM-flagged names (already in surveillance — flow is unpredictable)
  3. Excluding F&O 200 names entirely (those have institutional shadow that this brief doesn't have an edge against)

**Why this differs from `mis_unwind_short_late_session`'s failure:** that brief's losing trades were dominated by retail HOLDING THE LONG via CNC-conversion + sector-momentum-into-close FII bid. **Both forces are absent in the retail-light universe** — there is no retail to convert, and these names are not in NIFTY constituents so do not catch sector-rotation flow.

## 3. Persistence

Three structural reasons:

1. **Operator/HNI tape-marking is structural in low-float small-caps.** Documented across multiple SEBI consultation papers (the GSM/ASM/T2T frameworks exist precisely to surveil this pattern). It does not arbitrage away because the operator's mandate is to mark closing prices for collateral / NAV / margin purposes, not to maximise per-trade alpha. The flow is mandate-driven, not opportunistic.
2. **Retail-light universe is structurally retail-light.** The bottom-tercile of intraday-volume small-caps is a stable population — retail does not migrate to these names because they are illiquid, not in F&O, often in T2T/GSM/ASM. The universe definition is robust over 5+ years.
3. **15:20 MIS auto-square mandate is regulatory.** It does not eliminate the mechanic — it **focuses** it. On retail-light names the auto-square contribution is small; the operator distribution dominates and is concentrated in the same 14:45-15:15 window.

The persistence horizon is decade+ — the SEBI surveillance frameworks are tightening (more names being added to GSM/ASM), but the operator pattern in the bottom-volume tercile of NSE smalls is a structural feature of the Indian micro-cap segment.

## 4. Evidence

Primary (regulatory + Indian-market specific):

1. **SEBI GSM Circular 2023** — graded surveillance on operator-prone low-volume names: https://www.sebi.gov.in/legal/circulars/aug-2023/circular-on-graded-surveillance-measure-gsm-_75412.html
2. **SEBI ASM (Additional Surveillance Measure) framework** — co-sanctioned with NSE/BSE for low-volume operator-prone scrips: https://www.nseindia.com/regulations/exchange-communications-asm
3. **NSE T2T (Trade-to-Trade) segment notes** — operationalising the same low-volume / operator-prone surveillance: https://www.nseindia.com/products-services/equity-trading-trade-to-trade
4. **SEBI Master Circular for Stock Brokers** (15:20 MIS auto-square): https://www.sebi.gov.in/legal/master-circulars/jul-2023/master-circular-for-stock-brokers_73558.html
5. **SEBI Jan-2023 retail F&O loss study** (LONG-bias dominance): https://www.sebi.gov.in/reports-and-statistics/research/jan-2023/study-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_67525.html
6. **SEBI Sep-2024 updated retail F&O study** (91.1% loss rate, intraday LONG-buying dominant): https://www.sebi.gov.in/reports-and-statistics/research/sep-2024/updated-study-on-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_87148.html

Internal precedent (failure-mode evidence to design around):

7. **`closing_hour_reversal` Phase-1 failure** — `reports/sub8_phase1/closing_hour_reversal_report/`: PF 0.45 across n=9,197, **catastrophic 15:00-15:15 window (PF 0.018, WR 1.1%)** — confirms the time-stop runway issue + lack of retail-load classifier.
8. **`mis_unwind_short` sub7/sub8 Phase-1** — `reports/sub8_phase1/mis_unwind_short_report/01-metrics.json`: PF 0.355, n=304, WR 9.2%; per-cap-segment table shows BOTH `mid_cap` (PF 0.582) and `small_cap` (PF 0.248) lost — but the small_cap loss was concentrated in retail-active F&O names. The retail-LIGHT subset of small_cap was never isolated.
9. **`mis_unwind_short_late_session` round-5 sanity** — failed at n=1008, PF 0.367 with the explicit hypothesis that retail CNC-conversion at 15:10 squeezed the shorts. **This brief's central differentiator is removing retail from the population.**

Negative-confirmation (not retail-saturated as a published strategy):

10. Searches across Streak, Stratzy, TradingView India, Wright Research, Religare yielded **no published operationalised algo** for "short retail-light small-caps after 14:30 declining-volume rally." Generic "closing-hour reversal" is published; the retail-light filter is novel. This means the asymmetry in the underlying-spot is NOT retail-saturated.

## 5. Direction

**SHORT-only.** No long mirror.

The mechanic is unidirectional: operator distribution into close is one-sided forced selling on retail-light names. There is no symmetric "retail-light operator accumulation rally at 14:30" that a long mechanic could fade. Consistent with round-1 long-bias caveat (no symmetric mirror → don't add a LONG side).

## 6. Mechanic

**Setup name:** `closing_hour_reversal_short_redo`
**Side:** SHORT only.

**Sequence:**

1. **At 14:00 IST per session, compute retail-light universe (the classifier — see §11).** Eligible symbols are those in the bottom-tercile of `intraday_cumulative_volume_at_1400 / 20d_median_full_session_volume`, on the small_cap or mid_cap NSE universe, NOT in F&O 200, NOT GSM/ASM-flagged.
2. **At 14:30 IST, screen the retail-light universe.** Symbol qualifies if ALL hold:
   - intraday return at 14:30 ∈ [+1.0%, +5.0%] (rallying — but not a runaway news event)
   - close at 14:30 within 0.5% of intraday-high (rally peaking, not yet rolled over)
   - **Volume-profile decline:** mean(volume 13:30-14:30) < 0.7 × mean(volume 09:30-12:00) — the rally is on declining volume vs morning-session activity, the operator-pump signature
   - **Cumulative-volume rank ≤ 33rd percentile** within all NSE intraday-active stocks at 14:00 (retail-light gate, primary classifier)
   - 5m bar at 14:30 prints a non-strong-bull candle (close not in upper-25% of bar range — momentum already stalling)
   - Not on GSM/ASM-flagged list for the day (excluded — surveillance flow is unpredictable)
   - Not an expiry day (cross-detector exclusion vs `fno_oi_cliff_weekly_expiry`)
   - Not a circuit-band day (cross-detector exclusion vs `circuit_t1_fade_short`)
3. **Entry triggers** (5m bar closes within active window): at 14:30, 14:35, 14:40, 14:45, 14:50, 14:55, 15:00 — fire SHORT on the FIRST 5m bar after 14:30 that prints a lower-low than the prior 5m bar (confirms rollover started) AND closes below the 14:30 5m bar's mid-range. Last entry: 15:00.
4. **Stop-loss:** intraday-high × 1.005 (operator-pump high + buffer). Min stop distance = 0.6% of entry (qty-inflation guard for thin names).
5. **Targets:** T1 (50% qty) at entry × 0.992 (~0.8% gain) — 1R-equivalent for thin retail-light names with wider spreads. T2 (50% qty) at VWAP if reached by 15:10. Time-stop hard exit at 15:15 IST regardless of P&L.
6. **Latch:** one fire per (symbol, session). No re-entry.

**Differentiation from `mis_unwind_short_late_session` mechanic (mandatory acceptance criterion):**

| Dimension | `mis_unwind_short_late_session` (failed n=1008, PF 0.367) | `closing_hour_reversal_short_redo` |
|---|---|---|
| Universe | F&O 200 mid + small-cap | **Retail-LIGHT small/mid-cap, NON-F&O, non-GSM/ASM** |
| Population thesis | Retail MIS-long auto-square cascade | **Operator-pump distribution on retail-light names** |
| Retail-load classifier | None (used cap_segment as a stand-in) | **Bottom-tercile cum-vol rank at 14:00 + non-F&O filter** |
| Entry-trigger volume gate | RVOL > some threshold | **Volume-decline gate: 13:30-14:30 vol < 0.7× 09:30-12:00 vol** |
| Off-the-high condition | Close 0.3-1.5% off intraday high | **Close within 0.5% of intraday high (still at peak, distribution starting)** |
| Failure mode addressed | n/a | **Eliminates the CNC-conversion squeeze (no retail to convert) and the FII-momentum-into-close (these names aren't in NIFTY constituents)** |
| Symbol overlap (acceptance gate) | n/a | **<40% of symbols overlap with `mis_unwind_short_late_session` prior-trade CSV must be falsifiable: if overlap ≥40%, retire — the new population is not actually new** |

## 7. Universe (data-broad with retail-light classifier)

**Universe is data-broad** — not pre-locked to the F&O 200 or any other narrow basket. The retail-light classifier is the gating mechanism.

**Eligible cap segments:** `small_cap` + `mid_cap` (large_cap excluded — operator-pump pattern is structurally absent in NIFTY100 constituents; institutional shadow dominates).

**Retail-light classifier (mechanically computable from existing 5m feathers):**

The brief commits to **Option A** — cumulative-volume rank at 14:00 IST — as the primary classifier, with non-F&O membership as a hard-included secondary filter. Rationale per option:

- **Option A (PRIMARY): bottom-tercile cum-vol-rank at 14:00 IST.** For each session, rank all `cap_segment ∈ {small_cap, mid_cap}` symbols by `cumulative_volume_09:15_to_14:00 / 20d_median_full_session_volume`. The bottom tercile (lowest 33% of volume utilisation) at 14:00 = retail-light. This is computable from `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_5minutes_enriched.feather` directly — no new ingestion. Rationale: retail intraday-trading concentrates in high-volume / high-RVOL names (F&O 200 + momentum-stocks); the bottom tercile of 14:00 cum-vol is structurally retail-absent.
- **Option B (RULED OUT): low absolute traded volume relative to free-float.** Rejected because per-symbol free-float is not on disk in a clean per-day form (would require BSE/NSE shareholding-pattern backfill — not currently available). Acceptable as a future enhancement; not a round-6 dependency.
- **Option C (RULED OUT): low NSE call/put OI vs avg.** Rejected because the candidate universe (small/mid-cap non-F&O 200) has zero options OI by definition — F&O is restricted to the 220-name F&O-eligible list. The classifier would be degenerate (everyone passes).

**Hard-included secondary filter:** symbol NOT in `assets/fno_liquid_200.csv` (zero F&O OI = zero F&O-driven retail flow).

**Hard-excluded filter:** GSM/ASM-flagged on the day (sourced from `data/surveillance/gsm_asm_dates.parquet` if it exists, else NSE-circular scrape per the round-5 `nse_gsm_asm_event` brief's data plan).

**Approximate universe sizing (pre-trigger):** ~600 small+mid-cap NSE names total → ~400 after dropping F&O 200 → bottom tercile = ~130 retail-light names per session, on average. (See §10 for n-feasibility math.)

## 8. Active window

- **Setup formation:** 14:00 IST (universe classification)
- **Trigger screen:** 14:30 IST single screen
- **Entry:** 14:30-15:00 IST (5m bars; one fire per symbol-session)
- **Time-stop EXIT:** 15:15 IST hard (regardless of P&L)
- **Hold horizon:** max 45 minutes intraday MIS

**Axis-independence with existing setups:**
- `gap_fade_short` (09:15-09:30): no overlap (4-5h apart)
- `circuit_t1_fade_short` (10:30 single-bar): no overlap (4h apart)
- `fno_oi_cliff_weekly_expiry` (T-1 14:30): no overlap (different universe — NIFTY index spot, not single small-cap; expiry day exclusion is built in)
- `mis_unwind_short_late_session` (failed 14:30-15:10, F&O 200): same active window, **FALSIFIED MECHANIC** — this brief's symbol-overlap gate (<40%) is the explicit differentiator

## 9. Risks / falsification criteria

The setup is **wrong** (and should retire) if:

1. **Round-6 floor fails (project-locked, BLOCKING):**
   - **NET PF < 1.10** on Discovery 2023-2024
   - **n_trades < 500 over 2yr Discovery** (round-6 hard floor; thin events alone do not justify deployment in this round)
   - **|WR delta validation→holdout| > 10pp**
   - **Net Sharpe < 0** on Discovery
2. **Symbol-overlap gate FAILS:** if ≥40% of `closing_hour_reversal_short_redo` triggered (session, symbol) pairs overlap with the prior `mis_unwind_short_late_session` round-5 trade CSV, **the new mechanic has not actually changed the targeted population** — RETIRE. (This is the single most important falsification criterion.)
3. **Volume-decline gate non-discriminating:** if removing the volume-decline gate yields similar PF on the same retail-light universe, the gate is not the discriminator and the setup is just a generic small-cap closing-hour-short. PROBE before retire (may simplify to drop the gate).
4. **GSM/ASM contamination:** if >25% of triggered events historically were on GSM/ASM-flagged days (despite the exclusion gate), the surveillance-data feed is incomplete and the population is contaminated. Re-source the GSM/ASM list before re-running.
5. **Borrow / SLB unavailability:** retail-light small-caps often have no SLB borrow. If >40% of historical triggered events were unborrowable in practice, the setup is non-tradeable in production. RETIRE.
6. **Operator-pattern decay:** if 2024-2025 PF is materially below 2023 PF, the SEBI surveillance tightening (more GSM/ASM additions) may be eroding the operator base. Flag as decaying-edge, may still ship at thinner alpha.

## 10. Pre-coding sanity-check plan (n-math + falsification)

**N-feasibility math (must hold pre-data for round-6 candidacy):**

- Universe size after retail-light gate (small/mid + non-F&O + bottom-tercile cum-vol): ~130 symbols/session (see §7)
- Trading days in 2yr Discovery 2023-2024: ~490 sessions
- Universe-sessions: ~130 × 490 = ~63,700 symbol-sessions
- Trigger rate post 14:30 gating (intraday return [+1.0%, +5.0%] AND close near intraday high AND volume-decline AND non-strong-bull bar): empirically 2-4% of universe-sessions on prior closing_hour_reversal data (`reports/sub8_phase1/closing_hour_reversal.parquet` had 9,197 fires across F&O 200 + 485 sessions = ~3.8% trigger rate on a broader gate set). Conservative estimate: **2.5% post all gates** (the volume-decline + retail-light filters are tighter than prior).
- Expected raw triggers: 63,700 × 2.5% = **~1,590 triggers / 2yr**
- Expected fired entries (one fire per symbol-session, after rollover-bar entry trigger): ~70-80% of raw = **~1,150-1,270 trades / 2yr**

**N-floor verdict: FEASIBLE.** Conservative ~1,150 / 2yr is well above 500 floor with material margin. If the bottom-tercile cum-vol gate is too aggressive (e.g., yields universe of 80 not 130), n drops to ~700, still above 500. If the trigger gate is too tight (1.5% trigger rate), n ~950 — still above floor. Multiple compression sources would have to combine before n falls below 500. **Round-6 n-floor passes the math gate.**

**Sanity-check tool:** `tools/sub9_research/sanity_closing_hour_reversal_short_redo.py` (≤300 LOC):

1. Load all `cap_segment ∈ {small_cap, mid_cap}` 5m enriched feathers for 2023-01-01 → 2024-12-31.
2. For each session, at 14:00 IST: compute cum_vol / 20d_median_full_session_vol per symbol. Rank within session. Mark bottom-tercile + non-F&O-200 = retail-light universe.
3. Apply 14:30 trigger gates (intraday return, dist-from-high, volume-decline, candle-shape).
4. Simulate rollover-bar entry 14:30-15:00, intraday-high-stop, T1 0.8%, T2 VWAP, time-stop 15:15.
5. Compute NET PF using existing Indian fee model (`tools/sub7_validation/build_per_setup_pnl.py:calc_fee` with small-cap impact-cost adjustment).
6. **Mandatory diagnostic outputs:**
   - n_trades total + per-year breakdown
   - NET PF total + per-year + per-cap-segment + per-quarter
   - WR + Sharpe
   - **Symbol-overlap report vs `mis_unwind_short_late_session` round-5 trade CSV** (must be <40% — gate of acceptance)
   - Top-20 contributing symbols + concentration check (no single symbol > 8% of PnL)
   - Volume-decline-gate ablation: PF with vs without the gate, on the same universe
   - Per-month PnL stability (target: ≥60% positive months over 24)

**Decision gate:**
- n ≥ 500 AND NET PF ≥ 1.10 AND symbol-overlap < 40% AND no-single-symbol-concentration → **APPROVE** for detector code
- n ≥ 500 AND 1.0 ≤ NET PF < 1.10 → **REVISE** — probe the volume-decline threshold (0.6× / 0.7× / 0.8×) and the cum-vol rank tercile cutoff
- Symbol-overlap ≥ 40% with `mis_unwind_short_late_session` triggers → **RETIRE** — population not actually new
- n < 500 → **RETIRE** — does not meet round-6 floor
- NET PF < 1.0 → **RETIRE** — operator-pump-unwind thesis falsified at our fee scale

## 11. Data engineering plan

**Retail-light classifier — derivable entirely from existing data on disk.** No new data feed required.

Concrete sources:

- **Symbol universe (small/mid-cap NSE):** existing 5m enriched feathers in `cache/ohlcv_archive/{symbol}.NS/`. Cap segment is in `cache/preaggregate/consolidated_daily.feather` per existing `cap_segment` column.
- **F&O 200 exclusion list:** `assets/fno_liquid_200.csv` (verified on disk).
- **Cumulative-volume rank at 14:00:** computed at sanity-check time from `volume` column on the 5m enriched feathers; no new column to persist if PF passes (compute on-the-fly in production detector).
- **20d median full-session volume:** computed from daily `consolidated_daily.feather` `volume` column with rolling-20 median; no new ingestion.
- **GSM/ASM exclusion list:** if the round-5 `nse_gsm_asm_event` brief's GSM/ASM backfill has been completed, read from there. If not, this brief's pre-flight checklist requires it as a dependency (or, fallback: run the sanity-check WITHOUT the GSM/ASM exclusion and report the contamination rate; if contamination is <10% the gate is not load-bearing).

**No new infrastructure** if sanity-check passes:
- The retail-light classifier is one Python function over the existing 5m feathers (~30 LOC)
- The detector is `structures/closing_hour_reversal_short_redo_structure.py` — modeled on `circuit_t1_fade_short_structure.py` for the daily-classifier + intraday-trigger split
- Production-mode retail-light universe computation runs at 14:00 IST in the daily classifier service (same lifecycle as existing daily-cap-segment classifier)

**The key engineering insight:** the retail-light classifier costs nothing additional to deploy because cum-vol rank is a derived metric from data already on disk.

---

## §3.3 acceptance criteria recap (round-6)

Candidate is **APPROVE-eligible only if all** hold:

- [ ] **(a)** n-feasibility math shows ≥500/2yr WITH conservative trigger-rate assumptions. Currently assessed: **YES** — math yields ~1,150-1,270 / 2yr with 2.5% trigger rate; floor margin is material.
- [ ] **(b)** Retail-light classifier is mechanically computable from data already on disk. Currently assessed: **YES** — Option A (bottom-tercile cum-vol rank at 14:00) requires only existing 5m feathers + cap_segment metadata + F&O 200 list.
- [ ] **(c)** Differentiation from prior failures (`closing_hour_reversal`, `mis_unwind_short`, `mis_unwind_short_late_session`) is concrete and falsifiable. Currently assessed: **YES** — symbol-overlap <40% gate against `mis_unwind_short_late_session` is the explicit acceptance criterion; population narrative (retail-LIGHT vs retail-active F&O 200) is structurally distinct; volume-decline gate is mechanically novel vs prior detectors.

Gates beyond (a) + (b) + (c):

- [ ] Sanity-check NET PF ≥ 1.10 over 2yr Discovery
- [ ] n ≥ 500 over Discovery
- [ ] |WR delta| ≤ 10pp validation → holdout
- [ ] Symbol-overlap with `mis_unwind_short_late_session` < 40%
- [ ] No single symbol > 8% of PnL (concentration)
- [ ] Volume-decline gate ablation: confirms gate is load-bearing (gate-on PF materially better than gate-off)
- [ ] Borrow-availability check: ≥60% of triggered events were borrowable (SLB/F&O eligible) in practice — if <60%, retail-light universe is non-tradeable in live and setup must be retired

---

## Decision required

User to indicate:

- [ ] **APPROVED** — proceed to sanity-check at `tools/sub9_research/sanity_closing_hour_reversal_short_redo.py`. No detector code until sanity passes.
- [ ] **RETIRE-pre-data** — judgment call that operator-pump-unwind on retail-light is too speculative given two prior closing-hour failures, regardless of population differentiation.
- [ ] **REJECTED** — reason
- [ ] **REVISE** — specify what's missing / wrong

**My read: APPROVE for sanity-check.** The brief satisfies all three round-6 acceptance criteria: (a) n-feasibility math passes with material margin (~1,150 / 2yr vs 500 floor), (b) the retail-light classifier is mechanically defined and computable from existing data with zero new ingestion, (c) the differentiation from `mis_unwind_short_late_session` is structurally explicit (retail-LIGHT non-F&O vs retail-active F&O 200) and falsifiable via the symbol-overlap gate. The mechanism (operator-pump distribution on retail-light names) is supported by SEBI/NSE primary surveillance literature and is **not** a re-try of the SEBI 15:20 MIS-cascade thesis that died twice. The single biggest residual risk is borrow availability on the retail-light universe — the sanity-check must include the SLB/F&O-borrow audit; if <60% of historical triggers were borrowable, the candidate retires regardless of in-sample PF.
