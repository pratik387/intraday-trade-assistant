# European-open ~12:30-13:30 IST signature — KILLED at Phase 2

**Date killed:** 2026-05-20
**Stage:** Phase 2 (empirical signature on Discovery 2023-2024)
**Branch:** `research/europe-open-13ist`
**Methodology:** `docs/setup_lifecycle.md` Stage 2 + `tasks/lessons.md` #3 Phase 2 kill criterion

## Hypothesis (Phase 1)

India is the only major equity market still trading when European markets open
(LSE 13:30 IST winter / 12:30 IST summer). Cross-border flow at European-open
time should leave a signature in Europe-exposed Indian large/mid-cap:
  - IT exporters (Europe = ~25-30% of revenue): TCS, INFY, WIPRO, HCLTECH, etc.
  - Banks with ADR/GDR listings: HDFCBANK, ICICIBANK, AXISBANK
  - Pharma exporters (EMA/MHRA regulated): SUNPHARMA, DRREDDY, etc.
  - Auto with European operations: TATAMOTORS (JLR)
  - Metals pegged to LME: TATASTEEL, JSWSTEEL, HINDALCO, VEDL

## Phase 1 verdict (borderline precedent)

Mechanism plausible (3 candidate flow channels: ADR/GDR convergence, FII
rebalancing, cross-border news). Data on disk. Regulatory clean.

**HONEST LIMITATION:** no Indian retail/pro source explicitly documents this
specific intraday 12:30-13:30 IST signature. Pieces of the mechanism are
documented (IT revenue concentration in Europe; J/U-shape intraday volume
profile per NSE/Monash research) but no source operationalizes a 13:00 IST
pattern. Phase 2 was used to settle the borderline-precedent gate.

## Phase 2 — Measured signature

Compared 22-symbol Europe cohort vs 23-symbol baseline (domestic/PSU) at
12:30, 13:00, 13:30 IST 5m bar anchors. 29,043 cohort observations vs 31,809
baseline observations over 24 months.

| Anchor | Europe cohort | Baseline | Delta |
|---|---:|---:|---:|
| 12:30 mean(vol_ratio) | 1.07 | 1.07 | 0.00 |
| 12:30 mean abs(bar_return) | 0.091% | 0.094% | -0.003pp |
| 12:30 corr(bar, fwd30) | +0.066 | +0.152 | -0.086 |
| 13:00 mean(vol_ratio) | 1.06 | 1.07 | -0.01 |
| 13:00 mean abs(bar_return) | 0.085% | 0.089% | -0.004pp |
| 13:00 corr(bar, fwd30) | -0.023 | -0.048 | +0.025 |
| 13:30 mean(vol_ratio) | 1.06 | 1.06 | 0.00 |
| 13:30 mean abs(bar_return) | 0.086% | 0.090% | -0.004pp |
| 13:30 corr(bar, fwd30) | -0.040 | -0.007 | -0.033 |

## Why killed

**Three signatures all ABSENT:**

1. **No volume burst** — Europe cohort and baseline have identical vol_ratio
   (1.06-1.07x prior-30d avg) at all three anchors. The "European open
   liquidity arrival" mechanism doesn't show up in Indian-cash volume.

2. **No directional impulse divergence** — Europe cohort actually moves
   SLIGHTLY LESS than baseline (0.085-0.091% vs 0.089-0.094%) at these
   anchors. Whatever generic intraday pattern exists at 12:30-13:30 IST,
   it's not amplified in Europe-exposed names.

3. **No distinctive forward-return signature** — at 12:30 IST, BASELINE shows
   higher continuation correlation (+0.152, high-vol: +0.282) than Europe
   cohort (+0.066, high-vol: +0.129). This is a generic post-noon momentum
   effect (likely related to the NSE J/U-shape volume pickup after the
   11:00-13:00 quiet zone), NOT a European-flow signature. If anything,
   Europe-exposed names show LESS of this generic pattern.

## Possible reasons the mechanism doesn't surface

These are SPECULATION — Phase 2 data falsifies the candidate regardless:

1. Cross-border flow at European-open is too thin to move Indian cash relative
   to domestic Indian flow at the same time.
2. Most "European exposure" trading happens overnight in ADR markets (NYSE/
   NASDAQ-listed Indian ADRs trade 6pm-1am IST) — by the time Indian market
   opens at 09:15 IST, European-exposed information is already priced via the
   open-gap.
3. European desks rebalance Indian exposure in larger weekly/monthly batches,
   not intraday.
4. The "Europe-exposed" categorization (revenue exposure) is too crude — actual
   intraday cross-border flow may concentrate in specific tickers via ADR-cash
   arb desks rather than the broad cohort.

## Time invested

~30 minutes total: 10 min Phase 1 web/precedent + 20 min Phase 2 script
+ run. Saved the 1-2 days of sanity-script work that a deeper-brief approach
would have committed to.

## Conditions for revival

This candidate could be revived if:

1. **A specific Indian-pro source surfaces** documenting an explicit intraday
   13:00 IST signature on a NARROWER cohort (e.g., LSE-listed-GDR-only:
   HDB, IBN, WIT, INFY) — narrowing the cohort might surface a signal that
   washes out at the broader-22-name aggregate level.

2. **Tick-level ADR-cash arb data** becomes available. The current Phase 2
   uses 5m bar aggregates; cross-border arb activity may concentrate in
   sub-bar windows.

3. **A regime change** (e.g., new SEBI rules permitting more cross-border
   broker flow, or a MiFID-style change in European desk behavior) creates
   new measurable intraday signature in this window.

## Files of record

- Phase 2 script: `tools/sub9_research/phase1_5_europe_open_signature.py`
- Raw measurements: `reports/sub9_sanity/_phase1_5_europe_open_signature.csv`
  (60,852 observations)

## Lesson surfaced (no new general lesson)

This is a clean execution of the lessons-#3 Phase 1-5 chain — including the
honest borderline-precedent acknowledgment at Phase 1 and the willingness
to let Phase 2 data falsify. The methodology is functioning as designed.
No new lesson required beyond reinforcing #3 and #15.
