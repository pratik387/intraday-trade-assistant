# Stage 5 Narrative Gate — Summary

**Date:** 2026-04-21
**Signed:** Claude (as Pratik — narratives grounded in detector audit + backtest data)
**Input:** 104 surviving cells from Stage 3 conditional analysis
**Output:** 90 APPROVED + 14 REJECTED

Narratives filled in `docs/edge_discovery/2026-04-20-run/05-narrative-gate/` (local only, 104 markdown files). Canonical source: `tools/edge_discovery/fill_narratives.py` — all setup-level narratives + cell-specific addendums + rejection reasons are embedded in that script.

## Approach

Per spec §3.3 Stage 5, each rule needs PARTICIPANT + BEHAVIOR + STRUCTURAL REASON IT PERSISTS, defendable to another trader. LLM-plausible is explicitly insufficient.

Grounded each narrative in:
1. Detector code in `structures/ict_structure.py`, `structures/range_structure.py`, `structures/support_resistance_structure.py`, `structures/vwap_structure.py`
2. Existing audit docs in `docs/edge_discovery/audit/`
3. Empirical data checks against the 484-session backtest (regime distribution, hour × outcome matrices, winner/loser asymmetry, cap-segment symbol inspection)

Data checks corrected three initial LLM-plausible assumptions:
- "Indian intraday is range-bound on ~70% of sessions" — FALSE. Classifier shows 32% chop + 1% squeeze; trending regimes dominate.
- "'unknown' cap_segment is mysterious" — it's T-group / Z-group / penny-tier names like RPOWER, IBULLSLTD, KERNEX, FOODSIN.
- "Opening and afternoon use the same mechanism" — they don't. Opening is gap-fade (low WR, big winners); afternoon is established-range reversion (high WR, moderate winners).

## Two edge-source families discovered

From the data cross-check, the 5 surviving setups split into two mechanism families that sub-project #2 (Conviction Architecture) should treat differently:

**Family 1 — Retail-reversion shorts** (premium_zone_short, range_bounce_short, order_block_short, resistance_bounce_short):
- Work best in small / mid / unknown cap
- Edge concentrated in afternoon / late (MIS-unwind amplified)
- Mechanism: retail momentum exhaustion + MIS 15:15 forced unwind + ICT/SMC adoption gap in Indian retail

**Family 2 — Institutional-reference shorts** (vwap_lose_short):
- Works best in large-cap only
- Edge from institutional execution benchmarks (VWAP) breaking
- Mechanism: PMS / AIF execution algos cascading on VWAP loss

## Rule counts by setup

| Setup | Total Cells | APPROVED | REJECTED |
|---|---|---|---|
| premium_zone_short | 19 | 17 | 2 |
| range_bounce_short | 43 | 36 | 7 |
| order_block_short | 26 | 23 | 3 |
| vwap_lose_short | 8 | 7 | 1 |
| resistance_bounce_short | 8 | 7 | 1 |
| **TOTAL** | **104** | **90** | **14** |

Spec prediction was "~50% narrative rejection rate." Actual 13.5% rejection is lower — but this reflects that my CONSOLIDATE cells (1-way subsumed by 2-ways) were marked APPROVED with notes rather than REJECTED, since they have mechanism even if redundant. Sub-project #2 will deduplicate during conviction scoring.

## Strongest approved rules (top 10 by PF × N robustness)

| Setup | Rule | N | PF | WR% |
|---|---|---|---|---|
| resistance_bounce_short | cap_segment+hour_bucket=unknown+afternoon | 157 | 2.11 | 63.7 |
| premium_zone_short | cap_segment+hour_bucket=unknown+afternoon | 431 | 1.94 | 57.8 |
| range_bounce_short | regime+hour_bucket=trend_up+late | 234 | 1.94 | 55.6 |
| range_bounce_short | cap_segment+hour_bucket=unknown+afternoon | 622 | 1.93 | 59.2 |
| range_bounce_short | regime+cap_segment=squeeze+micro_cap | 204 | 1.88 | 54.4 |
| range_bounce_short | regime+cap_segment=chop+micro_cap | 199 | 1.86 | 55.8 |
| range_bounce_short | cap_segment+hour_bucket=small_cap+late | 240 | 1.85 | 57.1 |
| range_bounce_short | regime+cap_segment=trend_up+unknown | 1585 | 1.84 | 56.4 |
| order_block_short | regime+hour_bucket=squeeze+morning | 453 | 1.81 | 67.1 |
| order_block_short | regime+cap_segment=squeeze+small_cap | 453 | 1.80 | 64.2 |

## Rejected rules (14) and reasons

| Rule | Reason |
|---|---|
| premium_zone_short__hour_bucket=lunch | Low-liquidity artifact, no distinct mechanism |
| premium_zone_short__regime+hour_bucket=trend_down+lunch | Small N (181), lunch-hour artifact |
| range_bounce_short__hour_bucket=opening | N=138 too small, WR 48.5% breakeven |
| range_bounce_short__regime+hour_bucket=trend_up+lunch | Counter-trend in wrong hour, sample variance |
| range_bounce_short__regime+hour_bucket=trend_down+late | Small N, subsumed by other cells |
| range_bounce_short__regime+hour_bucket=trend_down+lunch | Small N, lunch artifact |
| range_bounce_short__regime+hour_bucket=squeeze+lunch | Rare regime × low-liquidity hour |
| range_bounce_short__cap_segment+hour_bucket=unknown+lunch | Small N (106), lunch artifact |
| range_bounce_short__regime+hour_bucket=squeeze+morning | Rare regime × weak hour |
| order_block_short__hour_bucket=opening | OB retest needs prior structure; opening has none |
| order_block_short__regime+hour_bucket=trend_down+morning | Marginal PF 1.32, weak OB memory at morning |
| order_block_short__regime+cap_segment=chop+mid_cap | Marginal PF 1.30, no distinct mechanism |
| vwap_lose_short__regime+cap_segment=squeeze+large_cap | Small N (110), squeeze-VWAP false breaks |
| resistance_bounce_short__hour_bucket=late | WR 53.1% borderline, outlier-driven PF |

## Key empirical findings for sub-project #2

1. **MIS 15:15 forced-close is the single biggest amplifier** across 4 of 5 setups. Afternoon / late hour buckets dominate survivor list. Sub-project #2's conviction scoring should model MIS-unwind pressure explicitly (minutes to 15:15 as a feature).
2. **Cap_segment = "unknown" is structural, not mysterious.** These are T-group / Z-group / penny-tier names. Shared property: zero institutional flow, pure retail MIS. Should be explicit feature in conviction scoring.
3. **Morning hour is systematically weakest** across all setups (WR 40-60%, often below breakeven). Sub-project #2 should NOT trade these setups before 10:30 IST without strong additional confirmation.
4. **Large-cap works only for vwap_lose_short.** Other setups either explicitly block large-cap (range_bounce_short code) or see their large-cap cells fail Stage 1/2/3 gates. Family-specific cap-segment rules needed.

## Handoff for Task 12 (Validation gate)

90 APPROVED rules are ready for OOS verification on FY25 Validation period (Jan-Sep 2025). Blocked on:
1. OCI Docker image rebuild to ship the entrypoint.py `populate_analytics_from_events` fix (committed in `a66472b`)
2. OCI backtest run on 2025-01-01 → 2025-09-30 date range
3. Then run `tools/edge_discovery/gates/validation_gate.py` (not yet implemented — spec Task 12)

Validation pass criteria per spec §3.4:
- PF ≥ 1.0 on FY25 (looser than Discovery's 1.2-1.3 because OOS)
- WR within ±10 pp of Discovery WR
- N ≥ 50 in FY25

## Handoff for sub-project #2 (Conviction Architecture)

Eventually (post-Validation-and-Holdout), the 58 distinct APPROVED rules (subsumed 1-ways consolidated into 2-ways) become the "trusted setup universe" input for sub-project #2. Tag each rule with:
- Family (1 = retail-reversion, 2 = institutional-reference)
- Mechanism type (gap-fade, established-range-reversion, OB-retest, VWAP-break, S/R-rejection)
- Hour sensitivity (opening-only, afternoon-only, late-amplified, etc.)
- Cap sensitivity (small/unknown only, large-only, mid+small)

This metadata drives conviction scoring in sub-project #2.
