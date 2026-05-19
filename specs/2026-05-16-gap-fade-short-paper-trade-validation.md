# gap_fade_short — Paper-Trade Validation Checkpoint

**Date:** 2026-05-16
**Branch:** `research/post-sebi-edge-setups`
**Status:** PRODUCTION COMBO RETAINED — re-evaluate after 2 months of paper-trade data
**Owner:** pratik387

---

## TL;DR

The current production-locked `gap_fade_short` combo **FAILS** the standalone ship-gate
threshold (Holdout PF ≥ 1.15) on the extended Holdout window (2025-10-01 → 2026-04-30, which
now includes the Feb–Apr 2026 war-month regime). A more conservative candidate combo with an
earlier time-stop and no partial-trail produces a positive Holdout PF (1.07 vs 0.93), but **0**
combos in the full sweep grid pass the formal ship gate. Decision: keep the current
PROD combo, run paper trading for 2 months (≈ end of 2026-07), and decide the switch
based on side-by-side paper-trade evidence.

---

## What was re-swept

- **Script:** `tools/sub9_research/_gap_fade_short_sl_target_sweep.py`
- **Aggregator:** `tools/sub9_research/_gap_fade_sweep_aggregate.py`
- **Grid (180 combos):**
  - `stop_buffer_pct`: 0.10, 0.20
  - `atr_stop_mult`: 1.0, 1.5, 2.0
  - `time_stop`: 10:15, 10:45, 11:30, 13:00, 15:10
  - `partial_mode`: all_in, partial_50_no_trail, partial_50_be_trail, partial_30_be_trail
- **Periods:**
  - Discovery: 2023-01-01 → 2024-12-31  (n = 10,505)
  - OOS:       2025-01-01 → 2025-09-30  (n = 4,411)
  - Holdout:   **2025-10-01 → 2026-04-30**  (n = 3,446)  ← extended from previous 2026-03-31
- **STT stress:** also run at 2.0× the equity-sell-side STT rate (`STT_RATE = 0.00025 → 0.00050`)

CSV inputs:
- `reports/sub9_sanity/_gap_fade_short_sl_target_sweep_{discovery,oos,holdout}.csv` (1× STT)
- `reports/sub9_sanity/_gap_fade_short_sl_target_sweep_{discovery,oos,holdout}_stt2x.csv` (2× STT)

Aggregated outputs:
- `reports/sub9_sanity/_gap_fade_sweep_aggregate_1x.csv`
- `reports/sub9_sanity/_gap_fade_sweep_aggregate_stt2x.csv`

---

## Headline numbers

### Production-locked combo (CURRENT, retained)

`stop_buffer_pct=0.10, atr_stop_mult=2.0, time_stop=13:00, partial_mode=partial_50_be_trail`

| Window  | STT | n      | PF    | NET (Rs)     | WR   | %stop / %t2 / %time |
|---------|-----|--------|-------|--------------|------|---------------------|
| Disc    | 1×  | 10,505 | 2.43  | +2,550,121   | 66.1 | 31.5 / 30.7 / 37.8  |
| OOS     | 1×  |  4,411 | 1.21  |   +204,992   | 58.6 | 33.6 / 23.9 / 42.6  |
| Holdout | 1×  |  3,446 | **0.93** | **−56,412** | 56.8 | 33.7 / 21.9 / 44.4  |
| Disc    | 2×  | 10,505 | 2.23  | +2,336,568   | 64.6 | —                   |
| OOS     | 2×  |  4,411 | 1.14  |   +142,502   | 57.3 | —                   |
| Holdout | 2×  |  3,446 | **0.89** | **−92,584** | 55.1 | —                   |

- **Standalone ship gate** (`Disc≥1.30 ∧ OOS≥1.20 ∧ Hold≥1.15`): FAIL on Holdout at both STT levels.
- The 2026-05-12 lock was made before Apr-2026 data was available; war-month inclusion broke the buffer.

### Candidate combo (RECOMMENDED — but parked pending paper-trade evidence)

`stop_buffer_pct=0.10, atr_stop_mult=2.0, time_stop=10:45, partial_mode=all_in`

| Window  | STT | n      | PF       | NET (Rs)    | WR   | %stop / %t2 / %time |
|---------|-----|--------|----------|-------------|------|---------------------|
| Disc    | 1×  | 10,505 | 2.08     | +2,064,280  | —    | **8.7** / 29.6 / **61.7** |
| OOS     | 1×  |  4,411 | **1.24** |   +227,765  | —    | 10.6 / 22.7 / 66.7  |
| Holdout | 1×  |  3,446 | **1.07** |    +52,805  | —    |  9.8 / 20.5 / 69.8  |
| Disc    | 2×  | 10,505 | 1.90     | +1,809,830  | —    | —                   |
| OOS     | 2×  |  4,411 | 1.17     |   +165,275  | —    | —                   |
| Holdout | 2×  |  3,446 | **1.02** |    +16,633  | —    | —                   |

- Loses ~Rs. 500K Discovery NET (≈ 20% less than PROD) but stays Holdout-positive even under 2× STT.
- Mechanism: earlier exit (10:45) avoids late-day war-regime reversals; no BE-trail whipsaw on T1.
- Sample size is identical (same signals; only target/SL/exit logic differs), so the PF gap is signal-quality-preserved, not selection-driven.

### Sweep-grid: combos passing the full ship gate

**0 combos** across all 180 (× 2 STT levels = 360 evaluations) pass `Disc≥1.30 ∧ OOS≥1.20 ∧ Hold≥1.15`.
Holdout cap is **1.07** (CANDIDATE) — well below 1.15 floor.

---

## STT 2× stress impact

|                          | top-10 avg PF (1× → 2×) | top-10 avg NET (1× → 2×) |
|--------------------------|------------------------:|--------------------------:|
| OOS (across grid)        |        1.27 → 1.20 (−0.06) | +Rs.292K → +Rs.242K (−Rs.50K) |
| Holdout (across grid)    |        1.02 → 0.97 (−0.05) | +Rs.13K → −Rs.36K (−Rs.49K)   |

Doubling equity STT shaves ~5 PF points and ~Rs. 50K NET per period — meaningful but not catastrophic.
The actual SEBI post-Oct-2025 hike was on the F&O side (which we do not trade for gap_fade_short),
so 1× STT remains the realistic fee model. The 2× run is a defensive sanity check.

---

## Decision: paper-trade for 2 months, then compare

**Plan (effective immediately):**

1. **Production stays on the current PROD-locked combo** — no config change. Live + paper trading both run this combo.
2. **For every paper-trade gap_fade_short trade**, also simulate the **candidate combo** on the same signal (same entry, same symbol, same day) using the existing sanity-script path-aware simulator. Persist both PnL streams.
3. **After 2 months** (target end-of-day **2026-07-16**), compute:
   - Realised live (PROD) NET / PF / WR / max-DD over the period.
   - Simulated candidate NET / PF / WR / max-DD over the same trades.
   - Trade-level paired difference (Wilcoxon on per-trade R).
4. **Switch rule** (binding, decided now to avoid mid-experiment reasoning):
   - Switch to CANDIDATE if **all three** are true on the live period:
     - Candidate NET ≥ PROD NET + **Rs.50,000** (per-period delta, not per-trade)
     - Candidate PF ≥ PROD PF + **0.10**
     - Wilcoxon paired p-value ≤ **0.05** (candidate − prod R > 0 distribution)
   - Otherwise, **retain PROD** and re-evaluate in another 2 months as live Holdout grows.
5. **Kill rule** (separate from switch — also binding):
   - If live PROD Holdout PF stays below **1.00** for the full 2-month window AND NET is negative,
     **disable `gap_fade_short` from production** (set `setups.gap_fade_short.enabled = false`)
     pending a full re-discovery, regardless of the candidate's number.

**How the simulated candidate is computed in paper-trade flow:**

The simplest path is to run the sanity-script trade simulator (`tools/sub9_research/sanity_gap_fade_short.py`)
in batch mode against the paper-trade `analytics.jsonl` once per week, with the candidate combo
parameters injected. No production-code branching is required — this is a side-channel evaluation,
not a runtime feature flag.

---

## Why the war months break the previous lock

The 2026-05-12 lock was tuned on Holdout that ended 2026-03-31. Apr-2026 carried over the
Feb-Mar war-regime tail: high midday reversals after morning fades complete, which is exactly
the regime that punishes BE-trail logic. The current PROD's `partial_50_be_trail` exits half at T1,
then sets stop to entry, which converts winners into break-even time-stops when the price re-reverses.
The candidate's `all_in + ts=10:45` avoids both by closing the entire position before the
midday reversal window opens.

This is **not** a "fee model" failure — STT doubling alone is a ~5 PF-point cost. The PF collapse
from 1.07 (candidate, all-in 10:45) to 0.93 (prod, partial-trail 13:00) is structural, not fee-driven.

---

## Files of record

- `_gap_fade_sweep_aggregate_1x.csv` — full grid at 1× STT, ranked by PF_min
- `_gap_fade_sweep_aggregate_stt2x.csv` — full grid at 2× STT
- `_gap_fade_short_sl_target_sweep_{discovery,oos,holdout}{,_stt2x}.csv` — raw sweep outputs
- `tools/sub9_research/sanity_gap_fade_short.py` — trade-by-trade simulator used for paper-trade comparison

---

## Reminder for future-self

When you re-open this in July 2026: the question is **NOT** "does the candidate combo beat
PROD in backtest?" — that's already answered (yes, on extended Holdout). The question is:
**did the candidate combo beat PROD on live paper-trade data?** The switch is binding on the
three-condition rule above. Don't relitigate the rule mid-stream.
