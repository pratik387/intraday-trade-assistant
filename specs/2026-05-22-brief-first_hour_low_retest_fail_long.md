# `first_hour_low_retest_fail_long` — Stage 0 brief

> **STATUS: KILLED at Phase 1 Gate A (2026-05-22)** — preserved as negative knowledge.
>
> **Reason:** No Indian retail/pro intraday-MIS source operationalizes the morning-low retest-fail fade for small-cap NSE intraday. Only universal pattern catalogs found (Zerodha Varsity generic double-bottom under Dow theory; Motilal Oswal "Power Hour" 09:15-10:30 generic mean reversion; JM Financial intraday timing). Per §3.2 binding rule, universal-pattern cargo-culting is forbidden — minimum bar is ≥2 Indian sources operationalizing THE SPECIFIC mechanism on retail-MIS infra.
>
> **Gate A findings (do not repeat without new sources):**
> - Zerodha Varsity TA module — generic double-bottom only, no intraday/cap/MIS-cascade specificity
> - SEBI 2024 retail F&O study — confirms 5x MIS + 89-93% F&O loss rate, but addresses F&O leverage, not cash-MIS small-cap morning-low microstructure (adjacent, not operationalizing)
> - Motilal Oswal / Univest / JM intraday guides — "Power Hour" generic mean reversion, no retest-fail rule, no small-cap cell, no 10:30-11:30 window claim
> - QuantInsti EPAT projects, Streak strategies, intradaylab, Bombay Trade House — no published backtest or rule for this exact mechanism (Streak is Nifty-500-restricted, excludes most small-caps)
> - TradersCockpit/PaisaAlgo — intraday double-bottom scanners but no published edge research
>
> **Gate B passed** (all data on disk) — the kill is on precedent, not data.
>
> **Conditions for revival:** require a fresh angle backed by a specific Indian quant publication, EPAT thesis citing this mechanism, NSE working paper, or named broker-quant operationalization. Current universal-pattern evidence base is exhausted.
>
> **Regime-reframe research note (not pursued):** Business Standard Feb 2025 article documents multi-month small-cap capitulation phase (Nifty Smallcap 100 down 23.7% from Dec-2024 high). Suggests a regime-conditioned variant (cascade dynamics dominant in FII-exit/down regimes) might be researchable later if reframed as regime-gated rather than structural intraday edge.
>
> **Data-feasibility caveat the brief missed:** the 10:30-11:30 retest window is on the boundary of monthly feather coverage — fine for 5m bars, but `consolidated_daily` ADV gate must use t-1 close, not same-day, to avoid look-ahead. Worth flagging if the candidate is ever revived.
>
> ---

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1 Indian-market research)
**Predecessor:** Brainstorming session 2026-05-22 (3-candidate batch)
**Direction:** LONG
**Window:** Intraday MIS (square 15:25)
**Portfolio rationale:** Active portfolio is 3 SHORT + 1 LONG. This is the highest-conviction LONG-side complement among the brainstormed candidates.

## 1. Mechanism statement (ONE sentence)

Small-cap MIS-eligible NSE stocks that print intraday low in the 09:15-10:00 window, then in 10:30-11:30 attempt a retest of that low but fail to make a new low (retest_low ≥ first_low * 0.999), reverse upward toward the morning high because retail leverage-cascade capitulation exhausts within 60-90 minutes and institutional discretionary funds use the morning low as a fade level.

## 2. Falsifiers (3 conditions that would invalidate the thesis)

1. **Mechanism falsifier (volume signature must invert at retest):** If the retail-capitulation thesis is right, the second-low retest must occur on declining volume vs the first low. Test: across 100+ fires, retest-bar 5m volume should be < first-low-bar 5m volume in ≥60% of fires. If retest volume ≥ first-low volume in >40% of fires, the second low is fresh selling pressure (not exhaustion) and the mechanism is wrong → KILL.

2. **Regime falsifier (FII-exit regime breaks institutional bid):** Mechanism depends on institutional discretionary bid stepping in at the second touch. During FII-exit regimes (R4 `fed_pivot_china_rotation_FII_exit`, Jan-Mar 2025 FII outflow ~Rs 1L cr), institutional bid weakens. Per-regime breakdown should show PF CI lower bound > 1.0 in R1+R2+R5+R7 but may fail in R4. If PF CI lower bound < 1.0 across all regimes, mechanism is wrong → KILL.

3. **Infra falsifier (MIS leverage / margin policy):** Mechanism depends on retail MIS leverage cascades (5x leverage exhausts within 60-90 min). If SEBI tightens MIS minimum margin >20% (forcing leverage <5x), retail cascade magnitude shrinks → exhaust window changes → mechanism dies. Falsifier: if MIS leverage policy changes during validation, the candidate must be re-validated on post-change data only.

## 3. Adjacent setups + correlation/effective-M assessment

| Setup | Status | Direction | Universe | Mechanism overlap | Correlation est. | M penalty |
|---|---|---|---|---|---|---|
| `long_panic_gap_down` | active | LONG | small-cap | Same direction, gap-down trigger | Moderate (different trigger) | 1.0-1.5 |
| `gap_fade_short` | active | SHORT | small-cap | Same universe, opposite direction | Low (intraday timing differs) | 0 (uncorrelated) |
| `capitulation_long_v2` | retired | LONG | mid-cap | Same direction, exhaustion-candle trigger | Different cap segment | 0 (retired) |
| `below_vwap_volume_revert_long` | paper | LONG | cap=unknown | Same direction, VWAP-based trigger | Low (different mechanic) | 0.5 |

**Effective M estimate (Harvey-Liu input):** 1.0-1.5 vs `long_panic_gap_down`. Other portfolio setups are independent.

**Portfolio impact if shipped:** would convert portfolio from 3-SHORT-1-LONG to 3-SHORT-2-LONG, improving diversification.

## 4. Phase 1 research outline (Gate A + Gate B)

### Gate A — Precedent (need ≥2 Indian retail/pro intraday-MIS algo sources)

Sources to find:
1. **intradaylab.com** — search for "small-cap intraday low retest" or "morning low fade" patterns
2. **Zerodha Varsity Module 5 (Trading Strategies)** — confirm "double-bottom retest" / "intraday support hold" coverage
3. **SEBI 2024 retail intraday study** — confirm small-cap retail MIS-leverage concentration & capitulation timing
4. **EPAT / Bombay Trade House quant blogs** — search for retail-cascade exhaust-time empirical estimates
5. Backup: NSE academy or Indian broker (Zerodha/Upstox/Angel) trading-floor reports

Acceptable evidence: any 2 of the above explicitly operationalize the morning-low retest fade for retail-MIS Indian small-cap.

### Gate B — Data feasibility (5-min check at Phase 1)

| Required data | On disk? | Source | Notes |
|---|---|---|---|
| 5m bars per symbol | ✅ | `backtest-cache-download/monthly/*_5m_enriched.feather` | Standard input |
| `cap_segment` metadata | ✅ | `services/symbol_metadata.get_cap_segment` (nse_all.json) | Already used by current setups |
| Production universe gate | ✅ | `tools/sub9_research/production_universe.py:ProductionUniverseGate` (Lesson #19) | Mandatory for sanity script |
| `consolidated_daily.feather` | ✅ | `backtest-cache-download/consolidated_daily.feather` | Universe filter source |
| Daily ADV / volume | ✅ | From consolidated_daily | For cap_segment-internal liquidity gate |

**Verdict:** Gate B passes without new data acquisition.

## 5. Phase 2 empirical signature plan (preview only)

Once Phase 1 confirms precedent:

- **Signal definition:** for each (sym, date), find `first_low_bar` in 09:15-10:00 window. Look for next-bar 10:30-11:30 where bar.low ≥ first_low_bar.low * 0.999 AND bar.low <= first_low_bar.low * 1.005 (retest). Mark retest_fail event if NO bar between first_low_bar and retest_bar made a new low.
- **Baseline:** all (sym, date) where first_low ∈ 09:15-10:00 AND there's any 10:30-11:30 bar within 0.5% of first_low.
- **Drift measure:** signed mean intraday return signal_event → 15:25, vs baseline.
- **Acceptance threshold:** ≥ +0.1% drift delta (Stage 2 kill floor per setup_lifecycle).

## 6. Status checklist for advance to Phase 2

- [ ] Gate A — ≥2 Indian sources cited
- [ ] Gate B — data feasibility verified (preliminary check above passes)
- [ ] Mechanism statement reviewed for Indian-microstructure anchor specificity
- [ ] Adjacent setup correlation re-confirmed against current production list
- [ ] No regime-cutover bisects 2023-2026 validation window (small-cap retail concentration is structurally stable, no known SEBI cutover affects mechanism)

## 7. Next action

Phase 1 research (Gate A + Gate B verification) — runs as a parallel agent task per session plan 2026-05-22.
