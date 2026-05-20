# `mpc_day_intraday_reversal` — KILLED at Phase 2

**Date killed:** 2026-05-20
**Stage:** Phase 2 (empirical signature on Discovery)
**Methodology:** `docs/setup_lifecycle.md` Stage 2 kill criterion
**Predecessor brief:** `specs/2026-05-09-sub-project-9-brief-mpc_day_intraday_reversal.md`

## Why killed

The brief proposed a 10:00-10:30 overshoot + 10:30-12:00 partial-revert mechanism in the rate-sensitive cohort (banks/NBFCs/auto/real-estate) on RBI MPC announcement days. Phase 2 measured this signature directly on 5m feathers for the 12 MPC events in 2023-2024 vs a 50-day random-day baseline.

**Effect sizes (n=420 MPC observations, n=1645 baseline):**

| Metric | MPC days | Random days | Ratio |
|---|---:|---:|---:|
| mean \|move_10_30_pct\| | 0.394% | 0.390% | 1.01x |
| corr(10_30 vs 30_120) | +0.054 | +0.050 | delta +0.004 |
| mean signed move_10_30 | +0.165% | -0.027% | — |
| mean signed move_30_120 | -0.084% | +0.016% | — |

The proposed overshoot signature (MPC days should show LARGER 10:00-10:30 moves than random days) and the reversion signature (MPC days should show more-negative correlation between the overshoot and the next 90 minutes) are both **absent at cohort scale**.

## Why the brief's mechanism didn't appear

The most likely explanations, none of which can be tested without spending additional degrees of freedom:

1. **The overshoot happens inside the 10:00-10:05 bar.** The brief's 10:00-10:30 window assumes the move takes 30 minutes; the data is consistent with most action being a single-bar reaction that's already retraced by 10:05.
2. **Cohort dilution.** The rate-sensitive cohort is 37 symbols; perhaps only banks show the pattern and the other 24 symbols dilute the signal to zero. Cell-mining sub-cohorts would be post-hoc and burns the falsification.
3. **Outcome-conditional signal.** Only dovish-surprise or hawkish-surprise events may produce overshoot; "hold-as-expected" events (~half of MPC meetings) may produce no move. Conditioning by outcome is the brief's planned z-threshold filter — but at 12 events / 2yr there isn't sample size to test outcome-conditioned cells after also slicing by cohort.

## Methodology check — this is a CHEAP KILL, not a process failure

Per `docs/setup_lifecycle.md` Stage 2 kill criterion (Lesson #3 Phase 2 rule):
> If signature doesn't exist or is weak (<0.1% net drift), abandon — no methodology will rescue a non-existent edge. This is the cheapest kill in the pipeline.

Sample-size-feasibility math in the brief §11 already flagged this as MARGINAL (180-200 trades at z=0.6, just below the 200 floor). Phase 2 surfaced the deeper issue: even ignoring sample size, the cohort doesn't show the mechanism.

Total time invested: ~30 minutes (write `phase2_mpc_day_signature.py` + run + interpret). Saved: the 1-2 days of sanity-script work the brief proposed in §10.

## Conditions for revival

This candidate could be revived if any of the following becomes available:

1. **Tick-level data within 10:00-10:05.** The cohort-level overshoot may be sub-bar; needs tick prints to detect.
2. **MPC-outcome calendar with cut/hold/hike labels per event.** Then test the overshoot only on directional-decision events (~6/2yr) at sub-cohort level. Sample size becomes very tight (~50-100 trades) so the confidence framework's wide CIs would be the gate.
3. **A different cohort hypothesis (e.g., bank-only).** Would require new Stage 0/1 work and clear pre-registration to avoid the "I'll try one more cohort" trap (Lesson #2).

## Files of record

- Phase 2 script: `tools/sub9_research/phase2_mpc_day_signature.py`
- Phase 2 measurements CSV: `reports/sub9_sanity/_phase2_mpc_day_signature.csv` (n=2,065 across MPC + baseline)
- Predecessor brief: `specs/2026-05-09-sub-project-9-brief-mpc_day_intraday_reversal.md`

## Lesson surfaced

The brief was thorough on §1-9 (mechanism narrative, statutory anchor, cohort lock, mechanic spec, kill-switch overlap) but did not include a Phase 1.5 quick-signature check before the §10 data-engineering plan. Adding a `Phase 1.5: 30-minute signature feasibility on existing 5m feathers` to the brief template would have caught this without writing a 400-line brief in the first place. Worth a template update.
