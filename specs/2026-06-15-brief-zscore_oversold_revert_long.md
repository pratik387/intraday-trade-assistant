# Brief: `zscore_oversold_revert_long` (Stage 0-1)

**Status:** Stage 0 (idea). Candidate #4 for the 2-3 day CNC/MTF reversion batch (after A2 trailing-loser, C1 low52; C2/C3 killed).
**Date:** 2026-06-15
**Lifecycle:** `docs/setup_lifecycle.md` Stages 0→6, daily-cross-sectional CNC/MTF variant. Reuses the C1-C4 machine via a new `selection_mode`.
**Horizon:** 2-3 day hold (user constraint).

## Mechanism (one sentence)
An MTF-eligible tier-1 illiquid name whose close sits **≥ Z σ below its trailing-20d mean** (a volatility-normalized over-extension) **on a turnover shock** is statistically stretched below its own band; in a thin book with no institutional bid, the deviation mean-reverts toward the band over 2-3 sessions.

## Why distinct (not A2/C1/C2/C3 re-skinned)
The four tested triggers key on: raw 5d %-return (A2), price level vs 252d low (C1), consecutive-down-day count (C2), single-day return decile (C3). **None normalize by realized volatility.** Z-score = (close − mean20) / std20 selects on *statistical* extremity, so it picks up a *calm* name whose modest drop is large relative to its own noise (which A2's raw-%-cut would miss) and skips a *volatile* name whose large drop is within its normal range. Different cohort → genuine diversification of "how the bounce is detected."

## Phase-1: Indian-market basis (≥2 sources)
1. System's LIVE reversion edges (`gap_fade_short`, `panic_crash_revert_long`, `up_spike_fade_short`) + A2/C1 — same illiquid over-extension→reversion family, production/confidence-validated.
2. Bollinger-band / z-score mean-reversion is the canonical statistical-arb reversion form (Krauss 2017 pairs review; classic Bollinger). Strongest in illiquid names where the thin book lets deviations over-shoot.

## Phase-1: Data feasibility (Gate B)
On disk: `cache/preaggregate/clean_daily_from5m.feather` (CA-adjusted) → close, rolling 20d mean/std computable. MTF snapshot for eligibility/leverage. **Feasible**, no new data.

## Phase-1: Regulatory
CNC/MTF delivery (not MIS/F&O) → SEBI F&O Oct-2025 / Apr-2026 STT hike do not apply. Survivorship caveat (anachronistic MTF list, Lesson #27) → **paper is the production-faithful gate**.

## Falsifiers (pre-registered)
1. **Mechanism:** if the deviation is an informational regime shift (not noise over-shoot), z-oversold names keep deviating (no reversion).
2. **Regime:** needs volatility/dispersion; dead in low-vol consolidation (same FII-exit weakness C1 showed).
3. **Infra:** shock filter removes the edge (quiet drift-downs continue) → the over-extension must be volume-confirmed.

## Decision (Stage 0 gate)
- [x] Proceed to Stage 2 (Phase-2 signature, Discovery-only).
