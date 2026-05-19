# Brief: `fno_ban_entry_t1_fade`

**Sub-project:** Post-SEBI Edge Setups — Candidate 2 (F&O Ban-Entry Reaction Pattern)
**Status:** **DRAFT — pre-registered, pre-data. Awaiting APPROVE / REJECT / RETIRE before sanity-check.**
**Date:** 2026-05-14
**Branch:** `research/post-sebi-edge-setups`

---

## Predecessors / context

- `specs/2026-05-14-research-post-sebi-edges.md` — research roadmap (Candidate 2 definition)
- `data/sebi_calendar/README.md` — dependency-tag vocabulary; `intraday_ban` tag introduced for this candidate's class
- `data/sebi_calendar/rule_changes.csv` — rows `2025-10-01` (MWPL formula tightened) and `2025-11-03` (intraday MWPL/FutEq OI monitoring activated)
- `services/regime_break_detector.py` — mandatory pre-flight gate; will refuse to run on a window containing rows above
- `docs/retired_setups.md` — negative-knowledge archive (the prior "ban list fade" attempt is **not** explicitly documented as its own section — see §"How this differs from the retired ban-list setup" below for the reconstruction and the lessons that bind this re-research)
- `specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md` — APPROVED §3.3 template + structural-target style precedent (T+1 same-session SHORT fade); this brief follows that template
- `specs/2026-05-08-sub-project-9-brief-nse_delivery_pct_anomaly.md` — daily-EOD-signal-then-T+1-intraday-fade structural precedent (closest in shape to this brief: EOD ban-entry → T+1 SHORT fade)
- `tools/asm_gsm_history/fetch_asm_gsm.py` — proven daily-snapshot scraper pattern; the F&O ban list scraper (`tools/fno_ban_history/` — NEW) will follow this politeness/backoff template

---

## Mechanism (one sentence)

Under the Nov 3, 2025 intraday FutEq OI monitoring regime, a stock entering the F&O ban list (intraday or EOD) creates a one-way exit flow for the next session — only existing-position closures are allowed, fresh long buying via futures is mechanically blocked, and existing-long positions facing margin pressure unwind into a buyer-thin tape, producing a predictable T+1 negative drift.

## Direction (LONG / SHORT / both, why)

**SHORT-only on T+1.**

The directional bias is determined **structurally**, not statistically, by the rule mechanics:

1. **Fresh longs are blocked.** Under the new FutEq OI monitoring (Nov 3 2025), any new position that would *increase* net delta-adjusted exposure is disallowed during the ban. This eliminates marginal-buyer-of-the-day flow on the futures side.
2. **Square-offs are allowed.** Existing-long futures positions can be closed (and many are forced to close — see §4 Participants). Existing-short futures positions can also close (covering), but the prior literature on Indian F&O retail flow (SEBI FY23: 93% of F&O retail traders lose; flow is overwhelmingly net-long) implies the *existing position pool is asymmetrically long*. Therefore the squaring-off flow is asymmetrically a sell-flow on the futures, which transmits to the cash via cash-vs-futures arbitrage.
3. **Cash leg still trades.** A ban on F&O does NOT halt cash-segment trading. Cash retail can still buy and sell. But the absence of new-long-futures flow removes a structural source of cash demand (futures-long-cash-short basis arb desks would normally hedge against new futures longs; with no new longs, no hedge-driven cash buying).

The asymmetry — block fresh longs while letting longs close — is **directionally one-sided by regulatory design**. LONG-side is not testable: there is no symmetric mechanic blocking fresh shorts while allowing short-covers. SHORT-only is the only direction consistent with the rule's structural impact.

Single-side commitment is also aligned with sub-9's binding rule §3.2 #5 (asymmetric direction; two-sided setups inherit long-side losses per SEBI FY23 retail data).

---

## Regulatory dependencies

- **regulatory_sensitivity:** `rule_creating` — the setup's mechanism is *created* by the Oct 1 2025 + Nov 3 2025 SEBI rule pair. Before Oct/Nov 2025 the mechanism does not exist in its current form (the prior end-of-day-only ban regime had qualitatively different participant flow — see §"How this differs from the retired ban-list setup").

- **depends_on:** `["MWPL", "intraday_ban", "single_stock_FO", "F&O_speculation"]`

- **reasoning:**
  - `MWPL` — entry threshold is FutEq OI ≥ 95% of MWPL; the new MWPL formula (min(15% free float, 65× avg cash volume)) is load-bearing. Any change to MWPL math materially shifts the universe.
  - `intraday_ban` — the *intraday* monitoring (4 random snapshots) is the rule-change that creates this mechanism. The prior daily-snapshot regime did not produce the same flow pattern (see retired-attempt reconstruction below).
  - `single_stock_FO` — the setup only applies to F&O-eligible stocks. Quarterly F&O eligibility revisions (NSE/FAOP cycles, e.g., 2024-09-13 removed 16 stocks, 2026-04-01 added 6) change the universe but not the mechanic.
  - `F&O_speculation` — the participant flow being faded is F&O retail speculator unwind under ban-imposed constraints.

- **Regime-break pre-flight requirement:** any gauntlet on this setup MUST call `services.regime_break_detector.check_window(...)` for each window. The Discovery/OOS/Holdout windows specified in §"Falsification thresholds" are post-Nov-3-2025 by construction, so the pre-flight should pass — but if a future SEBI circular alters MWPL math, FutEq methodology, or ban thresholds, the new circular's `effective_date` will be added to `rule_changes.csv` and any straddling window will fail the gate. Apr 1 2026 STT hike is `critical` for `F&O_speculation` — pre-/post-Apr-1 splits are mandatory in the war-window report.

---

## How this differs from the retired ban-list setup

**Negative-knowledge inheritance from the prior attempt.**

The roadmap (Candidate 2 note) and the `intraday_ban` tag rollout state that "Sub-9 round 1 retired an earlier 'ban list fade' attempt under PRE-rule mechanics." That attempt is **not** itself documented as a standalone section in `docs/retired_setups.md` — it is referenced collectively with the sub-7/8 generic-pattern batch failures (see `docs/retired_setups.md` §"Sub-projects #5, #7, #8 — the 14-setup cargo-cult batch"). Per `docs/retired_setups.md` Common Failure Mode discipline, the lack of a dedicated retire section is itself a yellow flag — the original attempt's failure mode must be reconstructed from first principles before re-research can proceed.

**Best-evidence reconstruction of the pre-rule attempt's failure mode:**

| Dimension | PRE-rule attempt (retired) | POST-rule (this brief) |
|---|---|---|
| **Ban entry trigger** | End-of-day MWPL check; stock enters ban based on EOD OI snapshot only | 4 random intraday snapshots + EOD; stock can enter ban at any of ~5 points per session |
| **Information arrival** | Ban known only after 17:30 IST publication on T-1, fully priced into T+1 09:15 open via gap | Ban can hit intraday (e.g., 11:30 IST snapshot lights the ban); flow happens within the same session AND the next |
| **Participant flow direction** | Mostly two-way (longs and shorts both held positions to expiry; ban applied to "no new positions in either direction" — symmetric) | Asymmetrically biased: longs blocked from adding, structural long-flow exists, asymmetric squeeze on existing longs to reduce |
| **Discovery & gap-pricing** | T+1 09:15 gap-down "priced in" the ban perfectly: open absorbs ~80% of the EOD-anchored unwind move. Fade was statistically dead by 09:30. | Intraday entry creates real-time surprise (no perfect overnight pricing window). Same-session post-snapshot AND T+1 windows are both potentially tradeable. |
| **MWPL math** | Old methodology (notional OI as % of MWPL); MWPL = min(20% free float, 30× ADV) | New FutEq OI methodology (delta-adjusted); MWPL = min(15% free float, 65× ADV) — most mid-caps see MWPL cut 50-80% per the Oct 1 2025 row in rule_changes.csv |
| **Position-limit context** | Individual + prop limits were larger; ban entry was rare, mostly stuck names | Individual 10%, prop 20%, FPI/broker 30% — tighter position limits → more frequent ban entries, more "real" force-flatten flow |
| **Setup result (estimated)** | T+1 PF < 1.10 on the symmetric two-way thesis; retired with the sub-9 round 1 batch | Re-research from scratch; do NOT assume PF transfers. |

**Why the new mechanism might work where the old didn't:**

1. **Asymmetry from rule design.** The new rule explicitly allows position-*reductions* and blocks position-*increases*. Net delta has to drop. This is a structurally one-sided constraint. The old rule's "no new positions" symmetric language meant ban entries didn't produce asymmetric flow.
2. **Tighter MWPL → smaller-cap names hit ban more often.** Operator-pump territory (small-cap, low-float) now hits ban regularly. Operator-pump territory is exactly where surviving setups (`gap_fade_short`, `circuit_t1_fade_short`) extract their edge. Same participant cohort, new regulatory trigger.
3. **Intraday surprise creates same-session AND T+1 tradeability.** Two trade windows per event vs one. T+1 fade is the simpler, lower-execution-risk version and what this brief targets first.
4. **Forced-flatten under tighter individual/prop limits.** With individual limit cut to 10% and prop to 20%, a single FPI or prop desk hitting its sub-limit during the ban must reduce — and can ONLY reduce (per the SEBI rule, if you cross your individual limit you must bring it back below by EOD). This produces a measurable forced-seller flow on T+1 with timing concentrated near the 14:00-15:00 unwind window.

**Common Failure Modes to actively guard against** (per `docs/retired_setups.md` §"Common failure modes"):

- **FM #1 Look-ahead bias on intraday aggregates** — the sanity script must NOT use full-session aggregates (day_high/day_low) at signal time. SL must use only bars up to entry.
- **FM #2 wide_open OCI capture passed off as validation** — sanity must apply the FULL filter stack (universe + ADV + cap + gap-band).
- **FM #3 Cell-mining without OOS lock** — cells are pre-registered in §"Cells" below BEFORE any data is touched.
- **FM #4 Regime classifier non-reproducibility** — this setup does not depend on regime classification (the trigger is binary: ban or no ban). Good.
- **FM #5 Holdout n below floor** — sample-size estimate in §"Data requirements" below shows n=200-700 events feasible; floor enforced at n ≥ 100 for Holdout.
- **FM #6 WR delta vs OOS** — pre-registered: |WR_Holdout - WR_OOS| ≤ 10pp.
- **FM #7 MIS-leveraged fee math** — sanity script uses Indian fee stack with `mis_leverage` applied per-trade.
- **FM #8 Target-anchor stripping** — this setup uses `target_anchor_type = "structural"` (T1/T2 anchored to ban-day's high and PDC). No r_multiple recalc bug risk.

---

## Cells (pre-registered, must NOT change after seeing data)

The following cells are locked *before* touching ban-list data. If the discovery phase reveals a better cell, that cell must be re-tested as a separate OOS pass — it does NOT inherit Discovery validation.

**Primary cell (Cell A):**
- `event_type ∈ {intraday_entry, eod_entry}` — both flavors aggregated for primary
- `cap_segment ∈ {mid_cap, small_cap}` — operator-pump + new-MWPL-cut hot zone; excludes large_cap (genuine institutional flow dominates, mechanic muted)
- `mis_leverage ≥ 1.0` (must be MIS-eligible-short on cash leg)
- `prior_day_return ≥ +1.5%` (entry-day was an up-move into the ban; rules out pure churning that hit MWPL on volume without price action) — anchors the fade-from-elevated thesis
- `T+1 09:15 gap ∈ [-2%, +2%]` — exclude already-priced-in gap-downs (no fade left) and continuation-news gap-ups (different mechanic)

**Secondary cell (Cell B), intraday-entry only:**
- `event_type = intraday_entry` (ban entered between 09:15-15:00 on T+0)
- `cap_segment ∈ {mid_cap, small_cap}`
- `entry: same-session at the bar AFTER the snapshot that triggered the ban` (4 known random snapshot windows; trigger is a published ban-list update)
- `mis_leverage ≥ 1.0`

**Tertiary cell (Cell C), large-cap variant — investigative only, NOT a primary candidate:**
- `cap_segment = large_cap`
- Same gating as Cell A otherwise
- Pre-registered ONLY for completeness; expected n thin; PF likely below 1.10 given large-caps don't have the operator-pump unwind asymmetry. If PF ≥ 1.20 on n ≥ 50 in Discovery, flag for investigation as a separate brief.

**Universe by cell, pre-data:**
- F&O 200 names — approximate 200 stocks
- After MIS-eligible filter — ~190
- After cap_segment filter for Cell A — ~140 (small + mid F&O eligible)

---

## Falsification thresholds (pre-registered)

These thresholds are locked *before* the sanity script runs. Each threshold below MUST be met to advance to the next phase. Any threshold miss in any cell triggers RETIRE for that cell.

### Pre-coding sanity check (must pass to write detector code)

Window: Nov 3, 2025 → Dec 31, 2025 (~40 trading days; pre-war post-rule)

Threshold:
- **Cell A**: NET PF ≥ 1.20 AND n ≥ 50 AND NET Sharpe > 0 AND fee-aware (MIS leverage applied) AND WR ≥ 30%
- **Cell B**: NET PF ≥ 1.15 AND n ≥ 25 (lower threshold, exploratory)
- **Cell C**: report-only; no decision threshold

If primary Cell A misses, the brief retires immediately (no cell-mining rescue).

### Discovery (Oct 1 2025 - Dec 31, 2025)

Wait — Oct 1 2025 is AFTER the new MWPL formula effective date but BEFORE intraday monitoring. Pure post-Oct-1 captures the new MWPL effect; pure post-Nov-3 captures both. The two regimes are mechanism-distinct.

**Resolution:** Discovery split into:
- **Discovery-A** (Oct 1 - Nov 2, 2025): new MWPL, OLD daily-snapshot regime. Tests: does the new MWPL alone create the fade? If yes, the intraday-monitoring claim weakens.
- **Discovery-B** (Nov 3 - Dec 31, 2025): new MWPL, new intraday monitoring. This is the canonical post-rule window.

Thresholds:
- **Cell A on Discovery-B**: PF ≥ 1.20, n ≥ 50
- **Cell A on Discovery-A vs Discovery-B PF delta ≤ 0.15**: if Discovery-A PF is materially higher than B, the *intraday-monitoring* claim is wrong (the edge is just new-MWPL-driven, not intraday-mechanic-driven). Re-frame the setup as "MWPL-tightening fade" and re-investigate.

### OOS (Jan 1 2026 - Apr 30 2026, war-aware split)

War period: Feb 28 - Apr 8, 2026 (per roadmap). Volatility tailwind; cross-check.

War-aware split:
- **Pre-war OOS**: Jan 1 - Feb 27, 2026 (~40 trading days)
- **War OOS**: Feb 28 - Apr 8, 2026 (~28 trading days)
- **Post-war OOS**: Apr 9 - Apr 30, 2026 (~16 trading days)

Thresholds:
- **Pre-war OOS Cell A**: PF ≥ 1.15, n ≥ 30
- **War OOS Cell A**: PF ≥ 1.00 (war is tailwind for shorts — DO NOT count war as edge); n ≥ 20
- **Post-war OOS Cell A**: PF ≥ 1.15, n ≥ 10 (thin window; advisory only)
- **WR delta (OOS pre-war vs Discovery-B)**: |Δ| ≤ 10pp
- **Combined OOS Cell A**: PF ≥ 1.15, n ≥ 60

### Holdout (paper-trade, May 14 2026 onward)

Per the post-SEBI methodology: paper-trade only, 60+ days minimum, live SEBI rules + post-war volatility.

Thresholds:
- **Holdout Cell A**: PF ≥ 1.15, n ≥ 30 (60-day minimum window expected to produce ~50-100 events)
- **WR delta vs OOS**: |Δ| ≤ 10pp
- **Monthly PF**: no individual rolling-30d window below 0.85 (rolling-monthly kill criterion)

### Apr 1 2026 STT hike post-flight

Apr 1 2026 STT hike (futures 0.02% → 0.05%, options on premium 0.1% → 0.15%) is `critical` severity. The pre-Apr-1 vs post-Apr-1 OOS sub-window split is mandatory:
- pre-Apr-1 PF ≥ 1.15
- post-Apr-1 PF ≥ 1.05 (lower bar; fee-stack drag is structural)

If post-Apr-1 PF < 1.00, the setup is fee-killed and should be retired or sized down materially regardless of overall OOS PF.

---

## Data requirements

### New data infrastructure required (sanity prerequisite)

1. **F&O ban-list daily history scraper** — `tools/fno_ban_history/fetch_fno_ban_list.py` (NEW).
   - Source 1 (primary, EOD): `https://nsearchives.nseindia.com/content/fo/fo_secban.csv` — daily-published CSV with the next-day ban list, lists all stocks with their MWPL utilization. Format per anecdotal evidence: header + rows of `Sr. No., Symbol, MWPL %`. Published ~17:30 IST.
   - Source 2 (intraday, NEW post-Nov-3): NSE publishes intraday ban-list snapshots when the 4 random checks trigger a ban entry. Specific endpoint TBC — likely `nsearchives.nseindia.com/content/fo/fo_secban_intraday_*.csv` (pattern reverse-engineered from `fno_combined_oi*` similar files) — to be verified at scrape design time.
   - Politeness/backoff: follow `tools/asm_gsm_history/fetch_asm_gsm.py` template (≥2s between requests, exponential backoff on 429/5xx, curl_cffi chrome-impersonation for NSE Akamai).
   - Output: `data/fno_ban_history/fno_ban_events.parquet` with schema: `[symbol, ban_date, ban_entry_time, ban_exit_time, mwpl_pct_at_entry, event_type (intraday|eod), entry_snapshot_index (1-4 or NaN if EOD)]`.
   - Backfill window: Oct 1, 2025 → Apr 30, 2026 (Discovery + OOS); ~140 trading days × 5-15 entries/day = ~1500 raw events expected.
   - Engineering estimate: 1-2 days.

2. **Existing infrastructure (already available)**:
   - 5m enriched feathers (production data layer) — `cache/preaggregate/consolidated_daily.feather` + per-symbol 5m parquets.
   - `nse_all.json` for `mis_leverage` and `cap_segment` metadata.
   - Indian fee model (`services/fee_model.py` — applies STT, brokerage, GST, SEBI fee, stamp duty; MIS-leverage-aware).
   - `services/regime_break_detector.py` for pre-flight checks.

3. **Cross-reference data (already partially available)**:
   - SEBI calendar (`data/sebi_calendar/rule_changes.csv`) — already lists 2025-10-01, 2025-11-03, 2026-04-01 rows.
   - F&O eligibility snapshots (NSE quarterly revisions) — needed to filter out symbols added/removed mid-window; can be reconstructed from NSE FAOP circulars (low priority for v1 sanity).

### Sample-size estimate (pre-data, methodology only)

Approach:
- Industry chatter + retail-broker ban-list pages suggest typical daily ban list has **~3-10 stocks** under the new regime, with high variance day to day (some days 0, some days 15+ during high-volatility windows).
- Per the roadmap's Candidate 2 note: "intraday ban entries are common — ~5-15 per week under new regime."
- **Conservative pre-data estimate** for Oct 1 2025 - Apr 30 2026 (~140 trading days, 7 months):
  - ~5 NEW ban-entries per week × 4 weeks/month × 7 months = ~140 unique-stock-entry events
  - After cap_segment filter (mid + small only, excludes ~30% large-caps) = ~100 events
  - After T+1 gap-band filter ([-2%, +2%]) = ~70 events (rough — gap distribution post-ban entry skews wider)
  - After data-quality filters (missing feathers, holidays, low-volume days) = ~50-60 events
- **Holdout estimate** for May 14 - Jul 14 2026 (~40 paper-trade days):
  - ~25-40 ban entries pre-filter
  - ~15-20 events post-filter
  - n ≥ 30 Holdout floor may be missed → may need 90-day paper-trade window

**Risk:** sample size for Holdout is the binding constraint. If n < 30 over 60 paper-trade days, this candidate becomes a longer paper-trade hold (120+ days) before any real-capital consideration. Pre-registered: extend paper-trade rather than relax n-floor.

### War-period contamination check

Per the roadmap: "Did ban entries spike during war? If so, war-period data is contaminated."

**Pre-data hypothesis:** ban-entry frequency likely **DID spike during the war window** (Feb 28 - Apr 8 2026) because:
1. Realized vol up → larger position sizes from speculation → more frequent MWPL-95% triggers.
2. Position-cutting pressure during war → asymmetric forced-unwind in some stocks (defensive-rotation flow).
3. Apr 1 2026 STT hike sits inside the war window, compounding the regime shift.

**Mitigation in the gauntlet:**
- War-aware OOS split (above) already separates pre-war/war/post-war.
- If war-period ban-entry frequency is >2x non-war baseline, war-period events are flagged in the analysis output as "regime-contaminated" — they count toward PF math but with a separate reporting line and a stricter PF floor (1.00, not 1.15).
- Cross-check: if the setup's edge is *all* in the war window, retire it (per "war volatility is a tailwind, not a stress test" — `specs/2026-05-14-research-post-sebi-edges.md` Hard Learning #3).

---

## Implementation sketch

### Pre-coding sanity (must run BEFORE any structure-class code)

**Tool path:** `tools/sub9_research/sanity_fno_ban_entry_t1_fade.py` (NEW; retired post-use per §3.3 convention).

**Logic flow:**
1. Load ban-list parquet (`data/fno_ban_history/fno_ban_events.parquet` from §"Data requirements" above).
2. For each ban event row, compute T+1 date (next trading day).
3. Load T+1 5m feathers for the banned symbol.
4. Apply cell filter (Cell A primary).
5. **Entry**: T+1 10:00 IST 5m bar close (later than circuit_t1's 10:30 because the ban-driven flow is faster — institutional unwinds start at 09:30, not 10:00 FOMO peak). Confirm 09:15 gap is in band; confirm latest 5m bar is red.
6. **SL**: T+1 day's 09:30-10:00 high + 0.5% buffer (window-anchored; NOT full-day). Min 1.0% of entry.
7. **T1** (50% qty): T+1 PDC (prior day close = ban-day close).
8. **T2** (50% qty): T+1 PDC × (1 - 0.5 × prior_day_return%) — half-fade of the prior-day move into-ban.
9. **Time stop**: 14:30 IST (90 min before MIS auto-square; ban-driven flow concentrates in 09:30-14:00).
10. Compute MIS-leveraged Indian fee-stack PnL per trade.
11. Emit per-cell PF / WR / Sharpe / n; emit per-event log to `reports/sub9_sanity/fno_ban_entry_t1_fade_trades.csv`.

**Sanity output file template:** `reports/sub9_sanity/fno_ban_entry_t1_fade_<timestamp>/summary.json` + `trades.csv`.

### If sanity passes (per thresholds in §"Falsification thresholds")

1. **`structures/fno_ban_entry_t1_fade_structure.py`** — production detector. Subclasses `BaseStructure`. Triggered T+1 09:15 by loading prior-day's ban list; armed on entry confirmation at 10:00 5m bar.
2. **`config/configuration.json` `setups.fno_ban_entry_t1_fade.*`** — cell params, entry time, SL/T1/T2 specs, MIS-leverage aware fee config.
3. **Registry**: `structures/main_detector.py` + `services/plan_orchestrator.py`.
4. **Universe**: `services/setup_universe.py` — F&O 200 + cap_segment filter.
5. **Tests**: `tests/structures/test_fno_ban_entry_t1_fade.py` — unit tests including a `regime_break_detector.check_window` smoke that asserts the setup refuses to run on a pre-Nov-3-2025 window.
6. **Data ingestion in OCI**: `oci/docker/entrypoint.py` — call the ban-list scraper for the relevant date range as part of pre-flight.

### Live execution path

Per `screener_live → plan_orchestrator → executor` pattern:
- Plan dict carries `target_anchor_type = "structural"` (no r_multiple recalc bug surface — see Common Failure Mode #8).
- Exit handling: T1 (PDC), T2 (half-fade target), time-stop 14:30, hard SL on prior-window high.

---

## Risks / open questions

### Data-source risks

1. **Intraday ban-list endpoint may not be publicly archived.** NSE may publish intraday ban list as a live-only feed (no daily archive). If so, the only practical capture is forward-only via paper-trading + live scrape. Discovery window then needs to be reconstructed from Twitter/broker-blog references (low quality). **Mitigation:** Phase 0 (1 day) must verify intraday endpoint availability before committing engineering effort. If unavailable, retire the intraday cell (Cell B) and reduce scope to T+1-only fade (Cell A).

2. **MWPL % at entry not always published.** The ban-list CSV publishes that the symbol IS in the ban — the exact MWPL % at entry timestamp may not be in the daily file. This blocks the "OI / MWPL ratio" feature engineering for future cells. Acceptable for v1 (the ban-list-membership IS the trigger; OI level is a secondary feature).

### Sample-size risks

3. **Holdout n < 30 risk (HIGH).** Per `docs/retired_setups.md` Common Failure Mode #5 (Holdout n below floor; `options_vol_iv_rank_revert` killed by this), thin Holdout is the canonical death mode. Pre-registered mitigation: extend paper-trade to 120 days before any real-capital consideration if 60-day paper produces n < 30.

4. **Cell A may collapse to small_cap only.** Mid-cap F&O names rarely hit ban (large free float). If post-data the cell splits into ~85% small_cap + 15% mid_cap and the mid_cap leg is statistical noise, the small_cap-only sub-cell is the real setup. **Pre-registered:** small_cap-only is acceptable as the production cell IF it independently passes all OOS/Holdout thresholds.

### Mechanism risks

5. **Ban-entry pre-pricing (the prior-attempt killer).** The most plausible reason the post-rule mechanism still fails: the *ban list publication* at 17:30 IST T-1 is fully visible to all market participants before T+1 open. The T+1 09:15 open may absorb 60-80% of the unwind move, leaving only the residual 20-40% intraday tail. This was the pre-rule killer. The post-rule defenses against this are: (a) intraday ban entries are real-time surprises with no perfect priced-in window; (b) the actual squaring-off-only flow happens *during* T+1, not at the open. **Pre-registered test:** if T+1 09:15-09:30 already captures ≥75% of the day's downside, the same-day fade is too thin → retire.

6. **Liquidity collapse during ban.** With F&O contracts paused for fresh longs, cash-segment depth may also thin (less arbitrage flow). Short-side execution on small-caps already has slippage risk; ban regime may amplify. **Pre-registered:** apply 25 bps slippage penalty in the sanity fee stack (vs the standard 10 bps) for ban-period trades.

7. **Tax bracket / STT hike interaction.** Apr 1 2026 STT hike is `critical` severity. Pre/post-Apr-1 split is mandatory (see §"Falsification thresholds"). If the setup's PF is fully concentrated in pre-Apr-1, it's structurally fee-killed under current rates.

### Operational risks

8. **F&O eligibility revisions mid-window.** A stock removed from F&O segment loses ban-list eligibility entirely. Pre-registered: only trade T+1 if the symbol is still F&O-eligible AS OF T+1 09:15 (not just at T-1 ban-entry time).

9. **War-window contamination.** See §"Data requirements" — pre-registered war-aware split addresses this.

### Open questions (resolve during sanity or Phase 1)

- **Q1:** Does NSE publish per-symbol intraday ban-entry timestamps in a machine-readable archive? (If no → Cell B retired pre-sanity.)
- **Q2:** Is the FutEq OI at entry (not just "≥95%" but the exact %) published in the daily file? (Affects future feature engineering, not v1.)
- **Q3:** Does the ban list distinguish "MWPL-tripped" vs "individual/prop-limit-tripped" entries? Different mechanisms may have different T+1 behavior. (Sanity should split by visible fields if possible.)
- **Q4:** Empirically, what fraction of ban entries are *intraday* vs *EOD* under the new regime? (Determines Cell B feasibility — if intraday is < 20% of entries, Cell B is too thin to be its own cell.)

---

## References

### Primary regulatory + NSE

1. **SEBI Circular SEBI/HO/MRD/TPD-1/P/CIR/2025/79** (May 29, 2025) — "Measures for Enhancing Trading Convenience and Strengthening Risk Monitoring in Equity Derivatives". Introduces FutEq OI methodology, new MWPL formula, single-stock position limits, and intraday MWPL monitoring. Effective dates: MWPL formula Oct 1 2025; intraday monitoring Nov 3 2025. https://www.sebi.gov.in/legal/circulars
2. **NSE F&O ban list source** — `https://nsearchives.nseindia.com/content/fo/fo_secban.csv` (daily-updated CSV; "Security in ban period for F&O segment").
3. **NSE Position Limits page** — https://www.nseindia.com/products-services/equity-derivatives-risk-management-sec-ban — describes the post-Nov-3-2025 intraday monitoring regime.
4. **NSE All Reports — Derivatives** — https://www.nseindia.com/all-reports-derivatives

### Plain-English summaries (cross-checked for mechanic accuracy)

5. **Kotak Neo, "SEBI's New F&O Rules Explained: Delta-Based OI, New MWPL & What It Means for Your Trading"** — confirms 4 intraday checks at random intervals; FutEq OI replacing notional OI; ban triggers at 95% MWPL, exit at 80%. https://www.kotakneo.com/bulletins/sebi-new-f-o-rules-explained-delta-based-oi-new-mwpl-what-it-means-for-your-trading/
6. **BusinessToday, "Big F&O shake up: SEBI introduces FutEq OI, tweaks ban rules, position limits"** — confirms tightened individual (10%), prop (20%), FPI/broker (30%) limits; MWPL = min(15% free float, 65× ADV). https://www.businesstoday.in/markets/story/sebi-equity-fo-rules-futeq-method-mwpl-risk-management-478321-2025-05-29
7. **The Statesman / IBTimes (multiple Sept-Oct 2025 wire summaries)** — confirm Nov 3 2025 as the intraday-monitoring start date.
8. **Zerodha Support — "Why do F&O contracts enter ban period?"** — confirms FutEq OI methodology, mechanics of ban entry/exit, allowed/blocked actions during ban. https://support.zerodha.com/category/trading-and-markets/trading-faqs/f-otrading/articles/why-do-futures-and-option-scrips-enter-ban-period-what-does-it-mean
9. **5paisa news, "SEBI Implements Stricter F&O Rules from Oct 1"** — already cited in `data/sebi_calendar/rule_changes.csv` row 2025-10-01. https://www.5paisa.com/news/sebi-implements-stricter-fo-rules-from-october-1-to-strengthen-market-stability

### Indian-market academic + structural references

10. **SEBI FY23 Study on F&O Retail Loss** — 93% of F&O retail traders lose; flow is overwhelmingly net-long. Load-bearing for the asymmetric-flow argument in §"Direction".
11. **Sehgal et al., *Pacific-Basin Finance Journal* 2024** — Indian-equity momentum/reversal evidence; documents next-day-fade-after-pump SHORT mechanic in operator-pump territory (the cap-segment cell A covers). https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640
12. **Llorente, Michaely, Saar, Wang, *Review of Financial Studies* 2002 — "Dynamic Volume-Return Relation of Individual Stocks"** — foundational return-volume relationship as noise-trader signature; load-bearing for the SHORT-fade thesis when volume signal (here: MWPL trigger = volume + OI saturation) is high.

### Internal precedents (this codebase)

13. `specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md` — APPROVED, validated production setup. Same structural-target style, same SHORT-only, same T+1 entry window. Best precedent for this brief's mechanic.
14. `specs/2026-05-08-sub-project-9-brief-nse_delivery_pct_anomaly.md` — APPROVED, daily-EOD-signal-then-T+1-fade structural precedent. Sanity-script template directly reusable.
15. `tools/asm_gsm_history/fetch_asm_gsm.py` — proven daily-snapshot scraper pattern; NSE Akamai-aware via curl_cffi. Reuse politeness/backoff for the new F&O ban scraper.
16. `services/regime_break_detector.py` — mandatory pre-flight; this setup must declare its depends_on tags so any future rule change auto-flags the gauntlet.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to ban-list scraper (1-2 days engineering) then pre-coding sanity
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong

Per sub-9 §3.3 (binding for new setups), **no code is written until APPROVED.** Per the post-SEBI methodology (`specs/2026-05-14-research-post-sebi-edges.md` Phase 0), research execution is **deferred until after the 2026-06-14 paper-window review** — this brief is parked pending that review.
