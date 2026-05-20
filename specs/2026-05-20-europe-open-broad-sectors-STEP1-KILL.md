# European-open broad-market + sector signature — KILLED at Step 1

**Date killed:** 2026-05-20
**Stage:** Phase 2 / Step 1 of 5 (gated kill-chain)
**Branch:** `research/europe-open-13ist`
**Methodology:** `docs/setup_lifecycle.md` Stage 2 + Lesson #2 anti-salvage

## Hypothesis

After the first narrow-cohort kill (specs/2026-05-20-europe-open-13ist-PHASE2-KILL.md),
user pointed out the cohort selection was a guess. Phase 1 redone identified 5
candidate channels (FPI flow, GIFT Nifty lead, EUR/INR FX, Brent crude, EUREX
index futures). FPI-flow channel suggests the signature should hit the WHOLE
FPI book (broad large-cap), not a revenue-exposed sub-cohort.

Step 1: test the broad market (NIFTYBEES) + 9 sector cohorts (built from
constituents already on disk). Kill at Step 1 if even broad market shows no
distinguishable signature at European-open anchors.

## Test design

- **Anchors tested:** 12:30, 13:00, 13:30 IST (European-open zone) +
  11:30, 14:30 IST (control anchors, no European event)
- **Cohorts:** BROAD_MARKET_NIFTYBEES + BANK + IT + PHARMA + AUTO + METAL +
  FMCG_DOMESTIC + OIL_ENERGY + REALTY + PSU_BANK_DOMESTIC
- **Window:** 2023-01-02 to 2024-12-31 (Discovery)
- **Per-anchor metrics:** vol_ratio_30d, mean abs(bar_return), mean signed
  return, corr(bar_return, fwd_return_30m), high-vol-conditioned versions

## Broad market signature

NIFTYBEES (Nifty 50 ETF, n~460 obs per anchor):

| Anchor | vol_ratio | abs(ret)% | corr_30m | hv_corr30m |
|---|---:|---:|---:|---:|
| 11:30 (control) | 1.05 | 0.036 | +0.107 | +0.169 |
| 12:30 (Europe summer open) | 1.06 | 0.037 | +0.081 | +0.206 |
| 13:00 (Europe peak) | 1.07 | 0.038 | -0.169 | -0.277 |
| 13:30 (Europe winter open) | 1.09 | 0.043 | +0.056 | +0.312 |
| 14:30 (control) | 1.07 | 0.044 | +0.041 | -0.194 |

**Europe-vs-control deltas (all below kill thresholds):**

| Metric | Europe avg | Control avg | Delta | Kill threshold |
|---|---:|---:|---:|---:|
| vol_ratio | 1.078 | 1.058 | +0.020 | 0.10 |
| abs(ret)% | 0.0393 | 0.0401 | -0.0008pp | 0.02pp |
| corr_30m | -0.011 | +0.074 | -0.085 | 0.10 |

All three signature axes show NO material difference at European-open
anchors vs control anchors on the broad market.

## Why this is conclusive

The 13:00 corr_30m of -0.169 looks interesting in isolation — but the
13:30 corr_30m is +0.056 (opposite sign) and 14:30 is +0.041. If a stable
European-open signature existed, all three Europe anchors should point in
the same direction. The wild swing across 5 anchors (high-vol-corr ranges
-0.277 to +0.312) is consistent with random sampling variation, not a
mechanism-driven pattern.

## Anti-salvage discipline (Lesson #2 anti-p-hack)

Looking at the per-sector table, there are some non-trivial individual cells:

- PHARMA hv_corr30 at 12:30 = +0.285
- BANK hv_corr30 at 13:00 = -0.261

These look like candidates to mine. **Per Lesson #2, this is the salvage
trap that retires setups:** when 9 sectors x 5 anchors x 2 metrics = 90 cells
are scanned and broad market shows no signal, 2-3 cells crossing |corr|>0.25
is the EXPECTED noise rate at n_high_vol ~400 per cell (standard error of
correlation ~= 1/sqrt(400) = 0.05; ~2-sigma boundary is at 0.10; 3-sigma at 0.15).

Without a pre-registered theoretical reason why BANK should reverse at 13:00
specifically OR PHARMA should continue at 12:30 specifically, finding these
cells after the fact is p-hacking. The methodology kills the chain at
broad-market negative.

## Total time invested

Step 1 of 5: ~15 minutes (10 min script + 5 min run). Saved Steps 2-5
(~1.5 hours of data fetching + Phase 2 expansion). Plus saved the 1-2
days of brief + sanity + structure code that would have followed if Step 1
had been misinterpreted as "borderline" instead of clean kill.

## Conditions for revival

This candidate could be revived if:

1. **Stronger Phase 1 precedent emerges** — a published Indian-pro or NSE
   research paper explicitly documenting a 13:00 IST intraday signature
   would justify a more targeted test (specific cohort + specific
   conditioning).

2. **GIFT Nifty intraday data becomes available** — testing whether days
   with materially-moving GIFT Nifty pre-12:30 show a different cash-market
   reaction at 13:00 might surface a conditional edge that broad-market
   averaging washes out.

3. **A specific event subset shows signal** — e.g., test only days with
   ECB / BoE rate decisions (these print at 12:45 IST). Sample size becomes
   tight (~8/yr) but mechanism is cleaner. Not the broad-market hypothesis.

4. **Cross-listed names (ADR + Indian cash) at sub-bar resolution** — true
   arb activity happens in seconds, not 5-minute bars. Defer until tick
   data is available.

## Files

- Test script: `tools/sub9_research/phase2_europe_open_broad_sectors.py`
- Raw measurements: `reports/sub9_sanity/_phase2_europe_open_broad_sectors_raw.csv`
- Summary table: `reports/sub9_sanity/_phase2_europe_open_broad_sectors_summary.csv`

## Lesson reinforced

This is the third Phase 2 kill in two days (MPC, narrow-cohort Europe-open,
broad+sector Europe-open). The methodology is doing its job — three
candidates would have consumed ~5-7 days of sanity-script work each without
the cheap-kill gate. Total time on all three: ~75 minutes.

The user's correction to redo the cohort was important — it caught a
methodological error (pre-restricting the test cohort based on a guess
rather than letting the data settle it). The broader test still killed
the candidate, but for the right reason this time.
