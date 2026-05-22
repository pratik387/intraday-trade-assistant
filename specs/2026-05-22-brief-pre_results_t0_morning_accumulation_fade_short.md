# `pre_results_t0_morning_accumulation_fade_short` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1 Indian-market research)
**Predecessor:** Brainstorming session 2026-05-22 (3-candidate batch)
**Direction:** SHORT
**Window:** Intraday MIS (square 15:25)
**Portfolio rationale:** Refines an under-explored window (T+0 results-day morning) using already-available earnings-calendar data. Distinct from retired `pre_results_t1_fade` (T-1, FII-positioning-dependent, RETIRED 2026-05-19).

## 1. Mechanism statement (ONE sentence)

On AMC (After-Market-Close) results-day, F&O underlyings with announced AMC results that print a 10:00-11:00 retail-FOMO accumulation move (price >= morning_high * 1.005 with vol_ratio >= 1.3x cumulative-prior-mean) get SHORT-faded in 11:30-13:00 because institutional desks holding pre-disclosure analyst-research insight (research desks, sell-side analyst calls, pre-release leaks) know most positive surprises were already priced in via accumulated positioning before the retail-FOMO bid arrived.

## 2. Falsifiers (3 conditions that would invalidate the thesis)

1. **Mechanism falsifier (institutional-knowledge cohort split):** If institutional knowledge is the driver, the fade should be measurably stronger on names with active analyst coverage (≥3 sell-side analysts publishing) than on names without. Test: cohort split. If PF gap between coverage-heavy and coverage-light cohorts is < 0.15 (i.e., <15% PF delta), the mechanism is wrong (no institutional-knowledge edge) → KILL.

2. **Regime falsifier (war-volatility + low-retail-flow regimes):** Mechanism depends on retail FOMO concentration during earnings week. During war-volatility regimes (R7, Jan-Apr 2026) retail risk-on flow drops and the morning accumulation magnitude shrinks. Per-regime breakdown should show PF CI lower bound > 1.0 in R1+R2+R5 (normal retail-flow regimes) and may underperform in R4+R7. If PF CI lower bound < 1.0 across all post-2023 regimes including R5, mechanism is wrong → KILL.

3. **Infra falsifier (SEBI LODR tightening) — DOWNGRADED RISK after Phase 1:** Phase 1 research confirmed SEBI's **Dec-2024 LODR Third Amendment** actually **WIDENED** (not narrowed) the asymmetry window — the amendment extends the disclosure window to 3 hours if a board meeting closes more than 3 hours before the next trading session. The mechanism is structurally REINFORCED, not threatened, by the most recent SEBI rule change in the 2023-2026 validation window. Continue to monitor future LODR amendments for any rule that requires intraday selective-disclosure pre-results.

## 3. Adjacent setups + correlation/effective-M assessment

| Setup | Status | Direction | Universe | Mechanism overlap | Correlation est. | M penalty |
|---|---|---|---|---|---|---|
| `pre_results_t1_fade` | RETIRED | SHORT | mostly large/mid F&O | Same data source (earnings_calendar) but T-1 timing | Different mechanism (T-1 institutional de-risk vs T+0 retail-accumulation fade) | 0.5-1.0 (similar data, different mechanism) |
| `gap_fade_short` | active | SHORT | small-cap | Same direction, gap-only trigger | Low (earnings day filter changes universe) | 0.3 (overlap only on earnings-day gap-ups) |
| `or_window_failure_fade_short` | active | SHORT | mid/small | Same direction, no event filter | Low (no earnings-day specific overlap) | 0.3 |
| `circuit_t1_fade_short` | active | SHORT | small | Same direction, circuit-band trigger | Low | 0 |
| `delivery_pct_anomaly_short` | DISABLED | SHORT | small | Same direction, delivery-day trigger | Low | 0 |

**Effective M estimate (Harvey-Liu input):** 1.0-1.5 (mostly vs retired pre_results_t1_fade due to shared data dependency; partial vs existing SHORTs for overlap on earnings days).

**Portfolio impact if shipped:** adds a 4th SHORT but in an under-utilized window (11:30-13:00 lunch lull) with event-conditioned firing (only ~30-50 fires/month max), so absolute portfolio-overlap exposure is small.

**Retired-setup risk (Lesson #19):** Inherits earnings_calendar dependency from retired `pre_results_t1_fade`. The Layer-3 data-classification issue (announcements_fr source died after Mar 2025) is now known and the v2 `{AMC, scheduled}` filter is documented. Use that filter from day 1 to avoid the same misclassification trap.

## 4. Phase 1 research outline (Gate A + Gate B)

### Gate A — Precedent (PASSED 2026-05-22)

Sources confirmed via Phase 1 research agent:
1. **SEBI LODR Reg. 30 + Dec-2024 Third Amendment** — 30-min post-board-meeting disclosure rule operative. Dec-2024 amendment **WIDENED** the asymmetry window (3 hours if meeting closes >3hrs before next trading session). The mechanism is structurally REINFORCED in 2023-2026.
2. **SCIRP — Post-Earnings Announcement Drift Anomaly in India (NSE 2002-2017)** — statistically significant Indian-equity PEAD; pre-announcement abnormal returns confirm an institutional-leak / accumulation channel exists in NSE earnings events.
3. **Tandfonline — Predictability of Earnings & Impact on Stock Returns (India)** — 67 large-cap Indian stocks, 33 quarters: all stocks show pre-announcement return premiums; market anticipates better earnings before disclosure (operationalizable for cohort split via analyst coverage).
4. **JM Financial — Indian intraday timing** — Indian broker note confirming 10:15-14:30 is the institutional-fade window after morning retail volatility settles; "fade trade" defined exactly as short-at-top-of-range used in this brief.

**Acceptance threshold met:** SEBI LODR + 2 Indian academic studies + 1 Indian broker operationalization. All Indian-specific.

### Gate B — Data feasibility (PASSED 2026-05-22 with prerequisite task)

| Required data | On disk? | Source | Notes |
|---|---|---|---|
| 5m bars per symbol | ✅ | `backtest-cache-download/monthly/*_5m_enriched.feather` | Standard |
| `data/earnings_calendar/earnings_events.parquet` | ✅ | 36,530 rows confirmed | **Schema fix: brief originally referred to `announce_class`/`date`; ACTUAL columns are `announce_time_class`/`announce_date`.** Phase 2 code must use actuals. |
| `announce_time_class` field with AMC | ✅ | In earnings_events.parquet | v2 `{AMC, scheduled}` filter MANDATORY from Day 1 (Lesson #11) |
| F&O universe list | ⚠️ PROXY VIA ADV | `data/fno_universe.json` does NOT exist; `data/fno_eligibility/` has only `removals_2024_2026.csv` (delistings only) | **PHASE 2 PREREQUISITE:** build F&O universe from ADV proxy or scrape NSE F&O eligibility CSV |
| ProductionUniverseGate (Lesson #19) | ✅ | `tools/sub9_research/production_universe.py` | Required for sanity universe filter |
| Analyst-coverage data (for falsifier #1) | ⚠️ | NOT on disk | Phase 5 covariate; not Phase 2 blocker. Scrape feasibility to be evaluated in Phase 3. |

**Critical findings from Phase 1 source-priority audit (more severe than retired `pre_results_t1_fade` Layer-3 investigation):**

**TWO sources rotated in 2025**, not one:

| Year | announcements_fr | announcements_bmo | board_meetings |
|---|---|---|---|
| 2024 | 8159 | 36 | (steady) |
| 2025 | 2136 | **5796 (161× surge)** | (carries `scheduled` class) |
| 2026 | 0 (extinct) | 2475 | (continues) |

The retired setup's v2 investigation flagged ONLY the `announcements_fr` death. The `announcements_bmo` 161× surge is a **NEW finding** that wasn't in Layer 3.

v2 `{AMC, scheduled}` filter recovers events per year (v1 → v2 deltas): 2022 +3.7%, 2023 +9.0%, 2024 +7.9%, **2025 +25.0%**, 2026 +14.3%. The +25% 2025 jump is the data-trap recovery — all 1,414 recovered events come from `board_meetings` source.

**Phase 2 audit MUST check both source migrations** via:
```python
df.groupby([df['announce_date'].dt.to_period('M'), 'source', 'announce_time_class']).size()
```

## 5. Phase 2 empirical signature plan (preview only)

Once Phase 1 confirms precedent (DONE 2026-05-22):

- **Universe:** F&O underlyings (top-200 by F&O turnover); cap_segment ∈ {large_cap, mid_cap, unknown} (to include F&O liquid names like M&M/BAJAJ-AUTO that tag unknown).
- **Signal definition:** for each (sym, T+0) where T+0 has AMC results in `earnings_events.parquet` (filter: `announce_time_class.isin(['AMC', 'scheduled'])`), look for 10:00-11:00 5m bar where price >= morning_high * 1.005 AND vol_ratio >= 1.3x cumulative-prior-mean. Mark as accumulation event.
- **Baseline:** all F&O underlyings on AMC-results-T+0 without accumulation event.
- **Drift measure:** signed mean return signal_event → 13:00 (mid-day end), vs baseline.
- **Acceptance threshold:** ≥ +0.15% drift delta (slightly tighter than 0.1% floor because the SHORT direction needs >= ~0.4% gross signal to survive Indian fees + STT).
- **Cohort split (falsifier #1 setup):** stratify by `analyst_coverage_count` if data is acquirable in Phase 1; if not, defer to Phase 5.
- **Required schema actuals:** use `announce_date` (NOT `date`) and `announce_time_class` (NOT `announce_class`).
- **Required source-priority audit:** print `df.groupby([month, source, announce_time_class]).size()` BEFORE Phase 5 cell-lock to verify both `announcements_fr` extinction and `announcements_bmo` surge are captured.

## 6. Status checklist for advance to Phase 2

- [x] Gate A — Indian sources cited (SEBI LODR + SCIRP PEAD + Tandfonline India earnings + JM Financial; PASS 2026-05-22)
- [x] Gate B — data feasibility verified (PASS with prerequisite task: F&O universe build)
- [x] {AMC, scheduled} filter pre-locked from Day 1 (Lesson #11)
- [x] Effective M estimate against retired pre_results_t1_fade documented (1.0-1.5)
- [x] No SEBI LODR rule change scheduled within the 2023-2026 validation window that closes the asymmetry window (Dec-2024 amendment actually WIDENS it)
- [ ] **Prerequisite: build F&O universe list** (proxy from ADV or scrape NSE F&O eligibility CSV)
- [ ] **Schema fixes baked into Phase 2 code: `announce_date` + `announce_time_class` (not the brief's original `date` + `announce_class`)**
- [ ] **Phase 2 source-priority audit must check BOTH `announcements_fr` extinction AND `announcements_bmo` 161× surge** (retired pre_results_t1_fade only flagged the first)

## 7. Next action

Phase 1 research (Gate A + Gate B verification) — runs as a parallel agent task per session plan 2026-05-22.
