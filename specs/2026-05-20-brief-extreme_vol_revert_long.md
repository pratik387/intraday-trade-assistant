# `extreme_vol_revert_long` — Phase 3 brief

**Date:** 2026-05-20
**Branch:** `research/europe-open-13ist` (continued from Phase 2 attention work)
**Predecessor:** `specs/2026-05-20-attention-crowdedness-PHASE2-KILL.md`
  (Phase 2 v2 baseline-controlled signature; aggregate delta ~0.05% below
  fee floor — testing here whether cell-locked sub-cell clears the floor)

## 1. Mechanism statement (ONE sentence)

When a stock prints a mild downward 5m bar (DN_Q1 of bar-return distribution)
with extreme intraday volume (vol_ratio >= 5 vs prior-20d same-bar baseline),
the bar marks transient retail panic; the next 30-60 min bounces back as
institutional bids absorb the supply, producing a measurable mean-reversion
above the generic-large-bar baseline.

## 2. Indian-microstructure anchor

The Phase 2 v2 sweep over 6.7M events showed a universal +0.04-0.07%
delta-vs-baseline pattern, monotonic by vol_ratio threshold. The mechanism
is consistent with documented Indian retail behavior:

1. **Retail panic-selling on news/rumor** triggers volume spikes on mild
   down bars — retail sell-into-volume, not into-trend
2. **HNI/institutional supply absorption** within 30-60 min as bid
   liquidity restores
3. **SEBI 2024 retail study**: 76% of intraday retail traders lose money;
   the LOSING trade pattern is sell-on-spike-then-bounce-without-them
4. **NSE intraday J/U-shape volume profile** (Monash NSE liquidity paper):
   the mid-day quiet zone 11:00-13:00 has thin liquidity, so volume spikes
   in this window are disproportionately retail-driven

## 3. Falsifiers (3 conditions that would invalidate)

1. **Mechanism falsifier:** if the bounce is just generic mean-reversion
   (vol-conditioned == not-vol-conditioned), the per-cell delta in Phase 5
   would not exceed +0.10% on any cell with n>=200 — KILL.

2. **Regime falsifier:** if the bounce depends on a specific market regime
   (e.g., positive FII flow weeks), the per-regime PF would diverge by
   >0.30 across Disc/OOS/HO — flag as regime-conditional.

3. **Infra falsifier:** if the bounce is captured by sub-bar tick movement
   that 5m bars can't trade (i.e., bounce happens in first 30 sec of next
   bar), real-execution slippage will erase the edge — KILL on OOS PF<1.0.

## 4. Pro/retail precedent (>=2 Indian sources)

1. **Zerodha Varsity Module 5** discusses "intraday mean reversion" as a
   broad strategy class — though without volume-conditioning, this is
   tangential rather than direct precedent.
2. **intradaylab.com** documents "panic-bottom catching" patterns on
   spike-volume bars — direct mechanism precedent.
3. **NSE Investor Awareness materials**: warn retail against "selling into
   volume spikes" as a losing pattern — confirms the supply-side of our
   thesis.

This is BORDERLINE precedent (consistent with documented patterns but not
operationalized as an intraday LONG setup specifically). Phase 5 will settle
whether the documented mechanism + statistical signal translates to a
tradeable cell.

## 5. Pre-registered cell-sweep dimensions

Per Lesson #2, dimensions must be locked BEFORE running cell sweep.

**Filter dim_pool (max 2D combinations):**

| Dimension | Bins | Source |
|---|---|---|
| `cap_segment` | large_cap, mid_cap, small_cap, micro_cap | per_row (universe builder) |
| `vol_ratio_bin` | [5, 7), [7, 10), [10, 15), [15, inf) | computed at signal time |
| `hhmm_bucket` | morning_0930_1100, midday_1100_1300, afternoon_1300_1500 | per_row at signal time |
| `bar_return_bin` | DN_Q1, DN_Q2, DN_Q3 (only DN side; setup is LONG) | computed at signal time using PER-DAY quintile edges from baseline pool |
| `prior_5d_ret_bucket` | down_gt_5pct, down_1to5pct, flat, up_1to5pct, up_gt_5pct | precomputed from daily |

**Forbidden dimensions:** any using `day_high`/`day_low`/`day_close` at
signal time (Lesson #5 failure mode #1). EOD aggregates allowed only as
metadata, never as filter.

**R-grid:** T1 in {0.5R, 1.0R, 1.5R}, T2 in {1.0R, 1.5R, 2.0R}, hard_sl
at 1.0R below signal_bar.low buffer (mild buffer 0.2%). Partial mode swept
across {all_in, partial_50_no_trail, partial_50_be_trail}.

**Time stops:** 14:30 IST, 15:10 IST (sweep over).

## 6. Mechanic (single sentence)

LONG entry at next 5m bar's OPEN (Mode B) when:
  - vol_ratio >= 5.0
  - bar_return is in DN_Q1..Q3 (mild down bars, not extreme down)
  - stock is MIS-eligible + in declared universe (cap_segment filter)
Hard SL: min(signal_bar_low * 0.998, entry * 0.99) — whichever is deeper.
Targets: per R-grid sweep with structural alternatives.

## 7. Active window

09:30-15:00 IST (allow LONG to time-stop by 15:10 or 14:30). Excludes
the 09:15-09:30 opening-range zone (gap-fade territory) and the 15:10-15:25
EOD-squareoff window.

## 8. Independence-from-existing-edges story

- **vs `gap_fade_short` (active, SHORT 09:15-09:30):** different time window
  (this fires 09:30-15:00), different direction (LONG vs SHORT).
- **vs `long_panic_gap_down` (active, LONG 09:15-09:20):** different window
  (this is intraday after 09:30, not opening), different trigger
  (intraday-volume burst vs opening-gap-down).
- **vs `or_window_failure_fade_short` (active, SHORT 09:30-10:30):** same
  time-window overlap, opposite direction. Universe overlap likely small
  since or_window fires on a failed-breakout pattern (specific OR-range
  exit), this fires on volume-spike on mild down bar.
- **vs `circuit_t1_fade_short` (active, SHORT 10:30):** different direction
  and different fire condition.
- **vs `delivery_pct_anomaly_short` (active, SHORT 09:30-10:30):** different
  direction.

No active LONG setup operates in the 09:30-15:00 intraday window. Clear
non-overlap.

## 9. Sample-size feasibility

Phase 2 v2 yielded:
  - vr>=5 + DN buckets: ~700K observations across 24 months
  - vr>=10 + DN_Q1 (strongest cell): 154K observations
  - Cell-locked to (cap_segment, vol_ratio_bin, hhmm_bucket): expected
    ~5,000-20,000 per cell on Discovery

PF >= 1.10 with n >= 200 floor easily achievable on cell mining.

## 10. Acceptance criteria (Phase 5 ship gate)

1. **Discovery:** Cell with n >= 500, PF_net >= 1.20 on Disc+OOS combined
2. **OOS one-shot:** PF_net >= 1.10 with WR within 10pp of Discovery
3. **Holdout one-shot:** PF_net >= 1.10
4. **Stationarity:** max-min PF_net across (Disc, OOS, HO) <= 0.30
5. **Confidence framework verdict:** PF CI lower bound > 1.0 on combined
   D+OOS+HO; adj Sharpe > 0 after Harvey-Liu haircut for M=6 effective
   setups (5 current + this candidate)

If any acceptance gate fails -> KILL.

## 11. Files of record

- Phase 2 v1 raw: `tools/sub9_research/phase2_attention_crowdedness.py`
- Phase 2 v2 baseline: `tools/sub9_research/phase2_attention_v2_baseline.py`
- Phase 2 results: `reports/sub9_sanity/_phase2_attention_v2_baseline.csv`
- This brief: `specs/2026-05-20-brief-extreme_vol_revert_long.md`
- Sanity script (to write): `tools/sub9_research/sanity_extreme_vol_revert_long.py`
- Cell-lock target: `tools/sub9_research/extreme_vol_revert_long_cell_lock.json`
