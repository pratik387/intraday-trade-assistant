# Stage 5 Narrative Gate — Sub-project #8 Survivor Inventory

All 14 templates filled and APPROVED. 7 canonical narratives + 8 duplicate-redirects.

## Canonical narratives (read these for the actual mechanism)

| # | Template | Setup × Cell | N | PF (full) | h1 | h2 | Source | Caveat |
|---|---|---|---|---|---|---|---|---|
| 1 | `gap_fade_short__cap_segment-small_cap.md` | gap_fade_short × cap=small_cap | 3,797 | 1.496 | 1.557 | 1.438 | Stage 3 | — |
| 2 | `gap_fade_short__regime_and_cap_segment-squeeze_and_small_cap.md` | gap_fade_short × squeeze + small_cap | 499 | 1.556 | 2.034 | 1.378 | Stage 3 | small N |
| 3 | `gap_fade_short__regime_and_cap_segment-trend_up_and_small_cap.md` | gap_fade_short × trend_up + small_cap | 1,804 | 1.584 | 1.628 | 1.541 | Stage 3 | — |
| 4 | `gap_fade_short__regime_and_cap_segment-trend_down_and_small_cap.md` | gap_fade_short × trend_down + small_cap | 1,051 | 1.427 | 1.502 | 1.351 | Stage 3 | — |
| 5 | `pdh_pdl_reject__cap_segment-mid_cap.md` | pdh_pdl_reject × cap=mid_cap | 174 | 2.546 | 1.973 | 3.484 | Stage 3 | small N |
| 6 | `closing_hour_reversal__cap_segment-mid_cap.md` | closing_hour_reversal × cap=mid_cap | 340 | 1.361 | 1.619 | 1.272 | Stage 3 | small N |
| 7 | `cpr_mean_revert__regime-trend_up__optuna_filtered.md` | cpr_mean_revert × trend_up + Optuna#28 filter | 1,309 | 1.332 | 1.494 | 1.170 | Optuna | **decay flag** |

## Duplicate-redirects (point to canonical)

| Template | Canonical |
|---|---|
| `gap_fade_short__regime-squeeze.md` | #2 |
| `gap_fade_short__regime-trend_up.md` | #3 |
| `gap_fade_short__regime-trend_down.md` | #4 |
| `gap_fade_short__hour_bucket-opening.md` | #1 |
| `gap_fade_short__cap_segment_and_hour_bucket-small_cap_and_opening.md` | #1 |
| `gap_fade_short__regime_and_hour_bucket-squeeze_and_opening.md` | #2 |
| `gap_fade_short__regime_and_hour_bucket-trend_up_and_opening.md` | #3 |
| `gap_fade_short__regime_and_hour_bucket-trend_down_and_opening.md` | #4 |

These are duplicates because gap_fade_short fires only in opening hour by construction (active_window=09:15-09:30) and only on small/mid/micro_cap (per detector config), so the 1-way and 2-way cells with hour_bucket=opening collapse into the corresponding cap/regime cells.

## What survived → what dies

**Survivors (7 cells, 4 distinct setups):**
- gap_fade_short (4 cell variants, all small_cap-anchored) — anchor of the portfolio
- pdh_pdl_reject × mid_cap — small N, deploy reduced size
- closing_hour_reversal × mid_cap — moderate N, deploy standard
- cpr_mean_revert × trend_up + Optuna filter — KILL if 2025 H1 PF<1.10

**Killed (2 cells, 2 setups):**
- orb_15 × cap=mid_cap — Optuna 50 trials confirmed no edge
- narrow_cpr_breakout × regime=chop — Optuna 50 trials confirmed no edge

**Killed at Stage 3 (3 setups, no recoverable cell):**
- vwap_first_pullback (PF 0.48 unfiltered, no salvage path)
- mis_unwind_short (n=304, below stat floor)
- (orb_15, narrow_cpr_breakout — see above)

## Next: Phase 6 OOS Validation

These 7 frozen rules + filter configs apply UNCHANGED to FY2025 H1 (Jan-Sep 2025).

Pass criteria per design Section 3.4:
- PF >= 1.0 on FY24-25
- WR within ±10pp of Discovery WR
- N >= 50 in FY24-25
- ONE SHOT — no tuning to make pass

Blocker: 2025 OCI capture run not yet performed. Need to run sub8_oci_overrides.json
across 2025-01-01 to 2025-09-30 before validation can proceed.
