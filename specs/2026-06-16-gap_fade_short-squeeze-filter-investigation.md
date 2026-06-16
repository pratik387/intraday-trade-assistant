# gap_fade_short — daily-squeeze filter investigation (NEGATIVE result)

**Date:** 2026-06-16
**Origin:** 11-day intraday paper audit (2026-06-01→06-16). gap_fade_short lost
−₹7,804 (PF 0.61), with −₹6,161 of it on days my daily-regime classifier flagged
as squeeze. Hypothesis: gap_fade underperforms on daily-squeeze days; adding a
squeeze exclusion would improve its edge.

**Verdict: DO NOT add a squeeze filter. The effect is non-stationary** — it helps
only the recent holdout and hurts the multi-year samples. Shipping it overfits to
Oct'25–Mar'26.

## Method

Re-labelled every backtest trade by the **daily** regime the live gate uses
(`DailyRegimeDetector`, 210-bar NIFTU-50 window, classified off the prior session
— the same path enabled by the 2026-06-16 regime-starvation fix). This is the
correct label: the holdout parquet's own `regime` column is the rare *intraday*-5m
squeeze (11% of trades), which is a different, mostly-harmless regime — excluding
it changes nothing (PF 1.13→1.14). Only the **daily** squeeze matters for the gate.

## Result — squeeze cohort flips sign across periods

| period | span | n | ALL PF | ex-daily-squeeze PF | daily-squeeze cohort |
|---|---|--:|--:|--:|--:|
| Discovery (sub7) | 2023–2024 | 6,723 | 1.15 | 1.13 | **PF 1.24 / +₹77.6k** |
| Full cut (decay_inputs) | 2023-01→2026-04 | 4,773 | 1.85 | 1.82 | **PF 1.93 / +₹228.9k** |
| Holdout (sub8) | Oct'25–Mar'26 | 1,385 | 1.13 | **1.30** | **PF 0.76 / −₹17.8k** |

Daily-squeeze is gap_fade's **good** regime in both multi-year datasets and its
**worst** only in the recent 6-month holdout. Excluding it lifts holdout PF
1.13→1.30 but *lowers* Discovery (1.15→1.13) and the full cut (1.85→1.82).

## Conclusion

1. **No stable squeeze edge.** A filter tuned to the holdout would be curve-fit to
   the 2025-26 low-vol regime. This matches gap_fade's history: prior regime
   restrictions (e.g. trend_up-only) also failed holdout. gap_fade's regime-relative
   performance is non-stationary; gating on it doesn't generalize.
2. **The recent weakness is period/regime-specific, not squeeze-structural.** The
   June paper bleed and the holdout squeeze-weakness reflect the current market
   regime (low-vol grind, post-macro), not an intrinsic gap_fade × squeeze flaw.
3. **The real lever remains exit/payoff (audit Issue 1).** gap_fade is a high-WR,
   win-small/lose-big setup (paper avg win ₹365 vs avg loss −₹1,010; needs ~73% WR
   to break even, delivers 63%). That asymmetry — not regime — is what tips it
   negative when any regime softens.

## Input to the 2026-07-16 re-eval

- Do **not** add a regime/squeeze filter to gap_fade.
- The open question is whether gap_fade's overall edge is **decaying** in 2025-26
  (holdout PF 1.13 vs the higher-PF production cut) versus enduring a bad regime
  stretch. Decide that from forward paper + the decay tripwire, not from a regime cut.
- If exit re-tuning is pursued (Issue 1), test a tighter loss cap / earlier
  time-stop rather than a regime gate.

## Caveats

- A full lifecycle re-validation (Discovery + a clean intermediate OOS +
  confidence card with a Harvey-Liu/deflated-Sharpe haircut for this added trial)
  was **not** run — the cross-period sign flip already disqualifies the filter, so
  the full pipeline was unnecessary.
- Daily-squeeze label depends on `DailyRegimeDetector` (BB-width bottom-20th-pctile
  over 100 daily bars). The classification used the production detector unchanged.
