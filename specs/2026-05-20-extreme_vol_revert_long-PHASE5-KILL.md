# `extreme_vol_revert_long` — KILLED at Phase 5 (cell sweep)

**Date killed:** 2026-05-20
**Stage:** Phase 5 cell-sweep on Discovery (1.5yr 2023-01-02 to 2024-06-30)
**Branch:** `research/europe-open-13ist`
**Predecessors:**
  - Phase 1 brief: `specs/2026-05-20-brief-extreme_vol_revert_long.md`
  - Phase 2 v2 result: `specs/2026-05-20-attention-crowdedness-PHASE2-KILL.md`
  - Phase 4 sanity: `tools/sub9_research/sanity_extreme_vol_revert_long.py`

## Recap

Phase 2 v2 found a real but tiny signal: extreme-vol DOWN bars revert
~+0.05% above baseline at 30m horizon, monotonic across thresholds.
Below Indian retail fee floor (0.06% breakeven at 5x leverage). The
user pushed: "Why not write a sanity and try cell-sweep — maybe a
narrower cell beats the floor?" Reasonable per Phase 5 discipline.
This document records the cell-sweep outcome.

## Phase 4 sanity output (Discovery)

  Window: 2023-01-02 to 2024-06-30 (18 months)
  Bars loaded: 34,420,838
  Universe (>=200 trading days, >=50K daily volume): 1,099 symbols
  Signal candidates (vol_ratio>=5, DN_Q1..Q3, 09:30-14:55): 172,391
  Trades generated: 164,497

  Cohort breakdown:
    cap_segment: small=51K, mid=46K, large=38K, unknown=29K
    vol_ratio_bin: 5-7=65K, 7-10=41K, 10-15=26K, gte_15=33K
    hhmm_bucket: midday=66K, afternoon=63K, morning=36K
    bar_return_bin: DN_Q3=64K, DN_Q2=52K, DN_Q1=49K, DN_Q4=11

  Exit reasons:
    time_stop: 90,882 (55%)
    sl: 38,785 (24%)
    t1: 31,796 (19%)
    t2: 2,927 (2%)
    same_bar_sl: 107 (negligible)

  Aggregate Discovery economics (no leverage, no cell lock):
    Gross:  Rs -3,912,224
    Fees:   Rs +13,548,662
    Net:    Rs -17,460,885
    Per-trade avg net: Rs -106.15

## Phase 5 cell sweep outcome

Sweep space:
  - Grid entries: 24 (T1 in {0.5R, 1R, 1.5R} x T2 in {1R, 1.5R, 2R}
    with t2>t1 constraint, x TS in {14:30, 15:00} x partial_mode in
    {partial_50_no_trail, partial_50_be_trail})
  - dim_pool (pre-registered in brief): cap_segment, vol_ratio_bin,
    hhmm_bucket, bar_return_bin
  - k_max=2 (1D + 2D filter cells)
  - Floor: n>=200, PF_net>=1.10

Result: ZERO cells passed floor across all (grid x dim) combinations.

## Why

1. **Fee dominance**: fees are 3.4x gross loss (Rs 13.5M fees on Rs 3.9M
   gross loss). Strategy trades too frequently with too little gross edge
   per trade.

2. **Target hit rate too low**: T2 hit only 2% of the time, T1 only 19%.
   The expected bounce simply doesn't reach +1R often enough to amortize
   the round-trip Indian retail fee stack (~0.30% on capital basis).

3. **No fat cell**: cell sweep across 4 dims x 24 R-tuples = ~96 base
   sweeps each yielding ~100-1000 cells. Zero of them beat PF>=1.10.
   Generic-pattern signals don't isolate into a tradeable subset.

## Methodology check

This is a clean execution of lessons.md #3 Phase 1-5 discipline + the
docs/setup_lifecycle.md Stage 5 cell-lock gate:

  Phase 1 (brief): mechanism + falsifiers + precedent + data feasibility
  Phase 2 (signature): aggregate delta measured = +0.05% (below floor)
  Phase 3 (mechanism brief): pre-registered dims, gate criteria
  Phase 4 (sanity): emit per-trade canonical schema with anti-bias guards
  Phase 5 (cell sweep): no cell beats floor -> KILL

Total time on this candidate: ~70 min (15 brief + 30 sanity script with
1 bug-fix iteration + 25 sanity run + 1 cell sweep).
Saved: 1-2 days of structure-code + OCI integration if this had been
shipped to production based on Phase 2 v2's mild positive delta alone.

## Conditions for revival

1. **Lower fees**: if zero-brokerage Indian intraday becomes available
   (Zerodha promotional rates, etc.), fee floor drops from 0.30% to
   0.10%. The strongest Phase 2 v2 cell (vr>=10 + DN_Q1, +0.054% mean
   delta) might then clear breakeven. Cell sweep should be re-run.

2. **Higher leverage product**: NSE futures with no STT on sells (or
   GIFT-listed futures equivalents) cut fees ~40%. Still likely not
   enough alone.

3. **Mechanism-specific filter found**: if a specific Indian-retail
   behavioral pattern (e.g., promoter pledge disclosures, ASM/GSM
   transitions) co-occurs with extreme-vol DN bars and produces a
   materially stronger bounce, that's a different candidate that should
   be brief'd separately.

4. **Sub-bar tick data**: 5m bar aggregation may wash out true bounce
   that happens in first 30-60 seconds. Tick-level reconstruction
   might surface a tradeable edge that 5m can't capture.

## Bug also identified (not blocking the kill)

Holdout window run yielded 0 universe symbols / 0 trades despite
loading 25M bars. Likely the cap_segment filter or universe-day-count
filter behaves differently on the holdout date range. Did not debug
because Discovery already killed the candidate; if revived under
conditions above, this bug must be fixed first.

## Files

- Phase 1 brief: `specs/2026-05-20-brief-extreme_vol_revert_long.md`
- Phase 4 sanity: `tools/sub9_research/sanity_extreme_vol_revert_long.py`
- Sanity outputs: `reports/sub9_sanity/_extreme_vol_revert_long_trades_{discovery,oos,holdout}.csv`
- Phase 5 cell-sweep: `tools/sub9_research/run_cell_sweep_extreme_vol_revert.py`
- OOS/HO evaluator (unused): `tools/sub9_research/apply_locked_cell_extreme_vol_revert.py`
- This kill record: `specs/2026-05-20-extreme_vol_revert_long-PHASE5-KILL.md`

## Cumulative session count

5 cheap kills across ~3 hours:

| Candidate | Stage | Time | Reason |
|---|---|---|---|
| mpc_day_intraday_reversal | Phase 2 | 30 min | Cohort overshoot absent |
| europe_open narrow cohort | Phase 2 | 30 min | Less pattern than baseline |
| europe_open broad + sectors | Phase 2 Step 1 | 15 min | Broad-market null |
| attention_crowdedness | Phase 2 v2 | 45 min | Signal real, below fee floor |
| extreme_vol_revert_long | Phase 5 | 70 min | No cell beats fee floor |

Total ~190 min. Methodology saved an estimated 5-10 days of structure-code
+ OCI integration for candidates that didn't survive data tests.
