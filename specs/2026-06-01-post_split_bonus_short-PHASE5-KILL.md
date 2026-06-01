# post_split_bonus_short — PHASE 5 KILL

**Date:** 2026-06-01
**Status:** KILLED before Phase 5 completion
**Reason:** trade count per year too low for portfolio inclusion

## Background

- §3.3 brief: `specs/2026-05-07-sub-project-9-brief-stock_split_bonus_exdate.md`
- Sanity tool: `tools/sub9_research/sanity_post_split_bonus_short.py`
- Phase 2 signature (locked 2026-05-24, n=262 across 4 years):
  - +4.30% mean drift over 5 trading days post ex-date
  - 73.7% SHORT hit rate
  - All 4 years positive (2023 +2.52%, 2024 +4.13%, 2025 +5.54%, 2026 partial +10.78%)
- Two variants:
  - **A**: 1-day MIS SHORT (entry D+1 09:15 open, cover D+1 15:25 close)
  - **B**: 5-day overnight CNC SHORT (entry D+1 09:15 open, cover D+5 15:25 close)

## Phase 5 evidence summary

| Window | Variant A | Variant B |
|---|---|---|
| Discovery (2023-2024, 2 years) | n=112 | n=115 |
| OOS (2025) | n=55 | n=54 |
| Holdout (2025-Q4 + 2026 through Apr) | n=35 | n=34 |
| Holdout pre-war (..2026-02-27) | n=27 | n=26 |
| Holdout war (2026-02-28..) | n=8 | n=8 |

**Annualized fire rate: ~55-60 trades/year.**

## Why kill before completing Phase 5 cell-mining

The Phase 5 step (cell-locking) is meant to narrow the universe to the
sub-population where edge is strongest. By construction it REDUCES n.
With Phase 4 n already at 55-60/yr, Phase 5 would drop the shippable
variant to 30-50/yr.

### Comparison to currently shipped/active setups (production-truth, 3.5yr backtest)

| Setup | Trades/year (est) |
|---|---|
| `gap_fade_short` | ~1,600 |
| `below_vwap_volume_revert_long` | ~1,200 (annualized from 5mo of data) |
| `long_panic_gap_down` | ~680 |
| `or_window_failure_fade_short` | ~530 |
| `close_dn_overnight_long` | ~400 |
| **`post_split_bonus_short` (this candidate, Phase 4)** | **~55-60** |

`post_split_bonus_short` would fire ~6-10× less often than the lowest-volume
shipped setup. Per quarter: ~14 trades. Per month: ~5 trades. **No statistical
confidence achievable on monthly PF** (one bad trade would dominate a month).

## Edge per trade was strong — why doesn't that compensate?

Phase 2 mean +4.30% / 5d × 60 trades/yr = ~258% notional / yr. Looks attractive
in isolation. But:

1. **Slot opportunity cost is real even at 60 fires/yr.** Each fire consumes a
   CNC short slot for 5 days (Variant B) — a 25%-30% portion of trading days
   with a slot occupied for a sparse-but-real edge competing with daily setups.
2. **Monthly PnL variance is uncontrollable.** With ~5 trades/month, a single
   trade going wrong takes the whole month negative. Decay monitoring would
   never converge — every month would look like noise or anomaly.
3. **Live execution risk on event-driven setups is non-trivial.** Bonus/split
   ex-dates aren't predictable far enough ahead for confident automation;
   adjusted-vs-raw price data discrepancies are common (already required a
   synthetic OHLC quality filter in the Phase 2 spec).
4. **The brief's "Variant B 5-day overnight CNC SHORT" requires SLB (stock
   lending & borrowing).** Live SLB infra not built; would need new pipeline.
   Not justified for ~30-50 trades/yr post Phase 5.

## Decision

KILL. Do not proceed to Phase 5 cell-mining. Do not write a paper-trade spec.

The Phase 2 signature itself (post-bonus/split drift exists at +4%/5d, 73% short
hit) is a real market behavior worth remembering as background context. If we
ever build a DISCRETIONARY signal dashboard (separate from automated trading),
this could be a manual-trade flag. **But it doesn't pass the auto-execution
threshold for inclusion in the live system.**

## Files preserved for reference

- `tools/sub9_research/sanity_post_split_bonus_short.py` — Phase 4 sanity tool
- `_tmp_split_bonus_events.parquet` — events dataset (bonus/split ex-dates)
- `reports/sub9_sanity/_post_split_bonus_short_trades_*.csv` — Phase 4 trade outputs

These can be deleted at next repo cleanup. Not needed for the active system.

## Lessons

This kill reinforces an existing principle (not a new lesson): **trade
frequency is a first-class gate alongside edge magnitude.** A candidate with
spectacular per-trade edge but <100 fires/yr is rarely worth shipping into an
automated system that needs operational diagnostics, decay monitoring, and
slot allocation.

When evaluating future event-driven candidates (corporate-action-triggered,
calendar-based, news-driven), screen for expected n/yr BEFORE investing in
Phase 4 sanity tooling. Add this as a Gate B check in the §3.3 brief template.
