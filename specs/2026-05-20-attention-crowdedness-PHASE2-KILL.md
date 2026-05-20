# Attention-crowdedness (vol_ratio extreme) — KILLED at Phase 2 v2

**Date killed:** 2026-05-20
**Stage:** Phase 2 v2 (baseline-controlled signature test)
**Branch:** `research/europe-open-13ist` (continued research)

## Hypothesis (user-initiated)

User asked: "do we get active stocks list live?" The thread evolved into:
when a stock has a sudden intraday volume burst (appears on NSE's published
"most active" list), retail attention concentrates, crowd-in overshoots, then
exhausts within 30-60 min. Fade the exhaustion (mean-revert).

## Phase 2 v1: raw forward returns (no baseline)

Tested vol_ratio >= 3.0 events on full Discovery window (2023-2024).
Result: 430K events, mild directional pattern:
  - UP events: fwd_30m mean = -0.015% (mild fade)
  - DOWN events: fwd_30m mean = +0.043% (mild bounce)
  - Spread: +0.057%

Without baseline, unclear if this is attention-specific OR just the generic
intraday mean-reversion that exists in any large move regardless of volume.

## Phase 2 v2: baseline-controlled

Same forward-return measurement on:
  - Event cohort: vol_ratio >= [3.0, 5.0, 7.0, 10.0]
  - Baseline cohort: vol_ratio in [0.8, 1.5] (NORMAL volume)
Stratified by bar_return quintile (UP_Q1..Q5 and DN_Q1..Q5) within each
direction. The delta (event - baseline) at the same bar-return quintile
controls for "is this just generic mean reversion at large moves?"

## Result: signal real but below fee floor

Top 10 cells by |delta_30m|:

| Threshold | Bucket | n | delta_30m | delta_60m |
|---|---|---:|---:|---:|
| vr>=10.0 | DN_Q1 | 154,022 | +0.0539% | +0.0664% |
| vr>=7.0 | DN_Q1 | 238,191 | +0.0509% | +0.0615% |
| vr>=5.0 | DN_Q1 | 351,271 | +0.0470% | +0.0577% |
| vr>=3.0 | DN_Q1 | 600,032 | +0.0390% | +0.0488% |
| vr>=10.0 | ZERO | 47,005 | +0.0302% | +0.0547% |

Pattern is **monotonic** across thresholds: stricter vol_ratio threshold
produces stronger delta. EVERY cell across 4 thresholds x 11 buckets shows
POSITIVE delta (events bounce MORE than baseline). Statistical signal is
unambiguous.

## Fee-math kill

Indian retail intraday round-trip fee on capital basis: ~0.30%
(per Lesson #14, calibrated to real per-trade actuals).

At 5x MIS leverage:
  - Need: 5 x delta_per_bar - 0.30% > 0
  - => delta_per_bar > 0.06% to break even
  - Strongest cell at 30m: 0.0539% — BELOW breakeven
  - Strongest cell at 60m: 0.0664% — at breakeven, before tax (25-31.2%)

After 25% tax on net annual: 0.0664 * 5 - 0.30 = +0.032% per trade gross,
- 25% tax = +0.024% net. Marginal. Plus slippage.

The signal is REAL but BELOW the noise floor introduced by Indian retail
fee stack. Cannot trade profitably as a standalone setup.

## Useful by-product (NOT a setup but a meta-filter)

The consistent direction across ALL cells (positive delta = mean-revert)
suggests vol_ratio could be a META-FILTER on existing setups:

  - For a SHORT setup firing on a DN bar with vol_ratio >= 5,
    expect mild upward pull post-entry (-Rs 5-7 per trade after fees)
  - For a LONG setup firing on an UP bar with vol_ratio >= 5,
    expect mild downward pull post-entry

This could potentially:
  - Reduce false-positive entries on existing setups (block entries when
    vol_ratio extreme on the opposite-of-thesis direction)
  - Improve T1/SL placement (account for ~0.05% mean-revert pull)

This is a Phase 1 candidate observation for a "vol_ratio filter on existing
setups" research thread — distinct from a new standalone setup.

## Methodology check

This was the cheap-kill path working as designed:
  1. Phase 1: "Does the signal exist?" - user pushed back to verify FIRST
  2. Phase 2 v1: signal exists at 0.04-0.07% magnitude
  3. Phase 2 v2: vs baseline, delta is 0.04-0.05% (smaller after control)
  4. Fee-floor check: 0.05% < 0.06% breakeven => KILL

Total time: ~45 minutes (write v1, run, write v2, run, document).
Saved: 1-2 days of brief + sanity-script work for an idea that has signal
but fails the fee gate.

## Conditions for revival

This candidate could be revived if:

1. **Fees drop materially** — e.g., zero-brokerage broker for Indian
   intraday (Zerodha streak/etc. promotional rates). 0.10% fees instead
   of 0.30% would make the strongest cell tradeable.

2. **Higher-leverage product** — F&O futures (no STT on sells) or
   intraday CFDs. But Indian retail can't legally use foreign CFDs.

3. **Tighter cell with materially stronger signal** — would need
   meaningful (>3x) improvement on top cells. Could try:
   - Cross with cap_segment (small_cap might show stronger signal)
   - Cross with time-of-day (mid-day 12:00-13:00 might be cleaner)
   - Cross with sector membership (FII-heavy cohort)
   But this is cell-mining without a documented mechanism — high risk
   per Lesson #2.

4. **Convert to META-FILTER on existing setups** (the by-product above) —
   different framing, lower bar (just need to improve existing PF, not
   stand alone).

## Files

- Phase 2 v1 script: `tools/sub9_research/phase2_attention_crowdedness.py`
- Phase 2 v2 script: `tools/sub9_research/phase2_attention_v2_baseline.py`
- Raw events: `reports/sub9_sanity/_phase2_attention_crowdedness.csv` (430K)
- Baseline-controlled results: `reports/sub9_sanity/_phase2_attention_v2_baseline.csv`

## Cumulative kill count this session

| Candidate | Time | Verdict |
|---|---|---|
| mpc_day_intraday_reversal | 30 min | Phase 2 kill (cohort overshoot signature absent) |
| europe_open narrow Europe-revenue cohort | 30 min | Phase 2 kill (cohort showed less pattern than baseline) |
| europe_open broad market + 9 sectors | 15 min | Phase 2 kill (broad market null) |
| attention_crowdedness (vol_ratio events) | 45 min | Phase 2 kill (real signal below fee floor) |
| **TOTAL** | **2hr** | **4 cheap kills** |

Methodology saved estimated 4-7 days of sanity + structure-code work for
candidates that data falsifies in <1 hr each.
