# long_panic_gap_down — Variant Bp Paper-Validation Spec

**Created:** 2026-05-23
**Setup:** `long_panic_gap_down` (active, intraday LONG, small/mid cap)
**Variant rule:** `dow IN {Tuesday, Wednesday, Friday}` — equivalent: `dow NOT IN {Monday, Thursday}`
**Status:** SHIPPED as classification tag (not hard entry gate). Paper-validation A/B for 60-90 days.

## Background

War-regime-decomposed Holdout analysis (2026-05-23) on OCI v2 canonical
(`reports/oci_canonical_v2/long_panic_gap_down_oci_canonical.csv`) found that
the baseline setup currently has marginal pre-war Holdout edge (PF 1.06 on
n=123), while a dow-locked variant (excluding Monday and Thursday entries) lifts
pre-war HO PF to 1.602 on n=71 — a +51% uplift that passes Bonferroni M=3
multi-hypothesis correction (t=2.68 ≥ critical 2.32).

This spec follows the same pattern shipped earlier (2026-05-22) for
`close_dn_overnight_long` Variant B (see
`specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md`
§Task 8): the variant is a **classification tag** emitted on every detector
fire, not a hard entry gate. Paper-trade execution + analytics use the tag to
A/B-test baseline vs variant cohort PnL during a 60-90 day validation window.

## Mechanism hypothesis

The Mon+Thu exclusion is not a generic risk gate — there is a specific
mechanism story:

- **Monday gap-down panic** tends to extend through the week. Weekend news
  flow (geopolitical, regulatory, earnings pre-prints) creates risk-off open
  positioning that is NOT a single-day overreaction — institutional desks
  reduce risk through Tue/Wed rather than bounce-buy the open.
- **Thursday gap-down panic** tends to extend through F&O monthly expiry
  (Thursday or Tuesday post-2025-09-01) into next-session settlement. Hedge
  desks rolling positions are NET sellers on Thursday panic days, not
  bounce-buyers.
- **Tue/Wed/Fri panic-down days** are more often single-session overreactions
  that mean-revert intraday: institutional risk desks aren't constrained by
  weekend rolls or expiry cycles on those days, so dip-buying flow appears
  on the same session.

This is a mechanism *hypothesis*; the analysis above is the *evidence*. The
paper-validation A/B will discriminate whether the hypothesis holds going
forward.

## Validation evidence summary

**Sources (all on `reports/oci_canonical_v2/long_panic_gap_down_oci_canonical.csv`,
1,688 trades total, 2023-01-02 to 2026-04-30):**

| Window | n | Baseline net PF | Variant Bp net PF | Uplift |
|---|---|---|---|---|
| Discovery (2023-01..2024-12) | 1,194 | 1.362 | 1.370 | +0.008 |
| OOS (2025-01..2025-09) | 311 | 1.166 | 1.395 | +0.229 |
| **HO pre-war (2025-10-01 to 2026-02-27)** | 123 | **1.060** | **1.602** | **+0.541** |
| HO war (2026-02-28 to 2026-04-30) | 60 | 0.853 | 0.258 | -0.595 |
| HO full | 183 | 0.984 | 1.353 | +0.370 |

**Statistical significance (pre-war HO only, where variant is meaningfully
different from baseline):**

- n (variant subset) = 71
- mean pnl_pct (variant) = +0.546% per trade
- std pnl_pct (variant) = 1.719%
- t-stat vs zero = +2.68
- Bonferroni M=3 critical = 2.32 → **PASSES**
- Harvey-Liu Sharpe haircut (M=3, 18% deflation) = +0.261 (per-trade SR, post-haircut)
- naive 95% CI on mean pnl_pct: [+0.137%, +0.954%]

**Cross-validation:** Three independent re-implementations of the dow filter
(`dayofweek.isin([1,2,4])`, `day_name in {"Tuesday","Wednesday","Friday"}`,
`NOT day_name in {"Monday","Thursday"}`) all produce identical results
(n=71, PF=1.602) — filter implementation is bug-free.

**Spot-check:** 10 random pre-war HO trades verified field-by-field against
raw OCI `analytics.jsonl` in `20260522-105604_full/<date>/`. 0/10 mismatches.
Canonical aggregation pipeline is trustworthy.

## Caveats (must NOT be ignored)

**1. True M is likely 15-30, not 3.** The calendar-variant analysis explored
~15-30 (dimension × bucket) combinations across the full 4-setup batch before
shortlisting Variant Bp as one of 3 candidates. Bonferroni at M=20 has
critical ≈ 3.5 — at that haircut, Variant Bp at t=2.68 would NOT survive.
The "PASSES M=3" claim is conservative only within the post-shortlist scope.

**This is why Variant Bp ships as a classification tag, not a hard gate.**
Paper-validation on FRESH forward data is the only honest discriminator
because the same OCI v2 canonical that the variant was discovered on cannot
be used as the validation set.

**2. Pre-war HO n=71 is small.** Per Lesson #15 (confidence framework), n=71
is below the typical sample-size floor for cell-locked setups (n≥100 in a
6-month rolling window). The CI on the per-trade Sharpe is wide enough that
forward performance could plausibly land anywhere from 1.2× to 2.0× lift.

**3. War regime entirely inverts the variant.** During war HO (2026-02-28+),
the variant is much WORSE than baseline (0.26 vs 0.85, n=11 vs 60). The
dow-mechanism (Mon-extends-into-week, Thu-extends-into-expiry) may be
regime-dependent: war regime supplies the kind of cross-asset risk flow that
doesn't respect normal weekday institutional cycles. If we re-enter a war-like
regime during paper validation, expect Variant Bp to underperform baseline.

**4. Variant Bp at OOS+HO_pre_war differs from Discovery.** OOS Variant Bp lift
was +0.229 PF; pre-war HO was +0.541. Strong Discovery-OOS+HO trend exists
(positive in all 3) so it's not pure regime-locked, but the magnitude varies.

## Implementation

### Code changes (already landed in this commit)

**`services/calendar_utils.py`:**

```python
def passes_long_panic_variant_bp(d: date) -> bool:
    """Variant Bp paper-validation gate for long_panic_gap_down.

    Rule: dow IN {Tuesday, Wednesday, Friday}  (NOT IN {Monday, Thursday})
    """
    return d.weekday() in (1, 2, 4)
```

**`structures/long_panic_gap_down_structure.py`:**

- Imports `passes_long_panic_variant_bp`
- Reads `paper_calendar_variant_bp.enabled` from config (default False)
- At detect-time, computes `variant_bp_flag = bool(passes_long_panic_variant_bp(_sd))`
  with try/except fallback to False (calendar-data hiccup must NEVER block fires)
- Emits `paper_variant_classification = {"baseline": True, "variant_bp": variant_bp_flag}`
  in event.context
- Flows through TradePlan.notes via existing `notes=evt.context` plumbing

**`config/configuration.json:setups.long_panic_gap_down.paper_calendar_variant_bp`:**

```json
{
  "enabled": true,
  "rule": "dow IN {Tuesday, Wednesday, Friday}",
  "expected_uplift_pre_war_HO_pf": 1.602,
  "baseline_pre_war_HO_pf": 1.06,
  "t_stat": 2.68,
  "bonferroni_critical_M3": 2.32,
  "decision_window_paper_days": 60,
  "evidence_ledger": "reports/oci_canonical_v2/long_panic_gap_down_oci_canonical.csv",
  "war_cutoff_date": "2026-02-28"
}
```

### Tests (already passing)

**`tests/services/test_calendar_utils.py`** — 7 new tests for `passes_long_panic_variant_bp`:
- Tue / Wed / Fri pass
- Mon / Thu / weekend excluded
- Independence from expiry/holiday calendars

**`tests/structures/test_long_panic_gap_down_structure.py`** — 5 new tests:
- Classification disabled by default → variant_bp:False
- Wednesday with enabled → variant_bp:True
- Monday with enabled → variant_bp:False
- Thursday with enabled → variant_bp:False
- Classification flows into TradePlan.notes

All 18 structure tests + 32 calendar_utils tests pass.

## Paper-validation protocol

**Window:** 60 paper-trading days from first live deployment of the tag.
Earliest viable: 2026-05-26 (next Tuesday). Expected completion: ~2026-08-20.

**Decision criteria (at end of window):**

1. **Sufficient n in both cohorts.** Need n ≥ 30 in BOTH baseline-but-not-variant
   (Mon+Thu fires) and variant cohort (Tue/Wed/Fri fires). At ~5-8 fires/week
   historical rate, 60 days produces ~30-50 fires per cohort.

2. **Variant cohort PF >= 1.30 AND baseline-only cohort PF <= 1.10.** If
   variant lifts meaningfully AND baseline-only is materially worse, the
   evidence reinforces the analysis.

3. **Both cohorts above 1.0 by similar margin.** If both lift to ~1.4 and the
   gap is small, the dow mechanism didn't manifest in paper — leave as
   classification tag, do not hard-gate.

4. **Variant cohort PF below baseline-only cohort.** If the relationship
   inverts on paper data, the OCI v2 analysis was overfit — disable the tag
   and add a Lesson entry.

**Decision actions:**

- **Strong validation (criterion 2):** Convert Variant Bp into a hard entry
  gate in the detector. Update config block to `_status_2026_XX_HARD_GATED`.
- **Weak validation (criterion 3):** Leave as tag. Re-evaluate in 60 days.
- **Inversion (criterion 4):** Disable tag (set `enabled: false`), document
  failure in retired_setups.md as a partial-retirement entry, add Lesson.

## Files touched (this commit)

```
services/calendar_utils.py                              (+34 LOC — new helper)
tests/services/test_calendar_utils.py                   (+62 LOC — 7 tests)
structures/long_panic_gap_down_structure.py             (+25 LOC — wire-up)
tests/structures/test_long_panic_gap_down_structure.py  (+108 LOC — 5 tests)
config/configuration.json                               (+13 LOC — config block)
specs/2026-05-23-long_panic_gap_down-variant-bp-paper-trade-spec.md  (this file)
```

## Related work

- `specs/2026-05-21-close_dn_overnight_long-paper-trade-implementation-spec.md`
  §Task 8 — Variant B for `close_dn_overnight_long` (same pattern, different rule)
- `tasks/lessons.md` #21 — HO regime decomposition mandatory before retire/hard-gate
- `tasks/lessons.md` #23 — Calendar-variant Bonferroni at true M, not shortlisted M
- `docs/setup_lifecycle.md` Stage 4 (config-driven) + Stage 6 (production trial)
