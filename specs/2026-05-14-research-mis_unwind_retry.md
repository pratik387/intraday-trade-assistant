# Research note: `mis_unwind_short` v2 retry (tape-confirmed unwind)

**Date:** 2026-05-14
**Branch:** `research/post-sebi-edge-setups`
**Status:** Discovery sanity complete. **Verdict: RETIRE.**

## 1. Asymmetry being retried

Per `specs/2026-05-01-sub-project-9-microstructure-first-redesign.md` §4, candidate "C":

> "MIS auto-square 3:00-3:20 PM forced selling. SEBI rule + retail-flow asymmetry. Short. mis_unwind_short failed; thesis right, mechanic wrong; worth retry with different mechanic."

The SEBI 15:20 MIS auto-square cutoff is regulatory and stable. Retail intraday flow is structurally
net-long (SEBI FY23/FY24 studies: 91-93% of F&O retail traders lose, dominantly on directional longs).
Mid/small-cap names attract heaviest retail MIS leverage (5x available; large-cap throttled to 2-3x).
Therefore the asymmetric net-sell pressure in 14:30-15:15 should be most detectable in mid/small-cap.

## 2. Prior attempts and their failure modes

### Attempt 1: `mis_unwind_short` (sub-7) — RETIRED 2026-04
- Net PF 0.355, n=304, WR 9.2%
- Mechanic: SHORT entries 14:55-15:15 inside the squeeze window on fresh-intraday-high stocks
- Failure: caught CNC-conversion squeeze. High-conviction names were converted to delivery instead
  of being auto-squared, and the late-day bid stayed strong. Selecting "fresh-high + slight
  momentum decay" actively biased toward the conversion population.

### Attempt 2: `mis_unwind_short_late_session` (sub-9 round-5) — also FAILED
- Net PF 0.367, n=1,008, WR 40.7%, NET Rs.-114,834 (see `reports/sub9_sanity/mis_unwind_short_late_session_trades.csv`)
- Mechanic: SHORT entries 14:30-15:00 on F&O 200 names that were up 1.5-4% but 0.3-1.5% off the
  intraday-high with declining volume in 14:15-14:30 vs 13:00-14:00.
- Failure modes:
  1. Universe collapse — F&O 200 is 143 large_cap / 4 mid_cap / 6 unknown. The brief specified
     mid+small-cap but the script's `ALLOWED_CAPS` still included large_cap, so 988/1008 trades
     fired on large_cap (where the asymmetry is weakest). The thesis was never actually tested
     on its intended population.
  2. Anticipation, not confirmation. The "off-the-high + volume declining" pattern is still a
     prediction that the unwind will happen, not evidence that it HAS happened. If retail holds
     conviction (CNC-converts), the predicted unwind never materializes and short gets squeezed
     in the close.

Both failed for essentially the same reason: shorting in anticipation of an unwind that didn't
actually appear on tape.

## 3. Alternative mechanics considered

| # | Mechanic | Pros | Cons |
|---|---|---|---|
| A | Wait for tape-confirmation: short ONLY after a 5m bar in 14:30-15:00 that loses VWAP, prints red, and volume-spikes — i.e., the unwind has physically begun. | Avoids anticipation. Directly detects flow rollover. Self-filters: if CNC-conversion dominates, no confirmation prints and no short fires. | Late entry — gives back the first leg of the move. Risk-reward thinner. |
| B | Anchor to VWAP. Short when price rejects VWAP from above and rolls over in the 14:30-15:00 window. | Clean structural reference. | Same flaw as prior attempts — VWAP rejection alone is anticipation, not confirmation. |
| C | Target mid+small-cap explicitly with broad NSE universe (not F&O 200 which is large-cap dominant). | Honors the original thesis (MIS leverage concentration). | Combinable with (A); not standalone. |
| D | Require relative-volume spike in 14:30-15:00 confirming flow. | Confirmation rather than prediction. | Same as (A) — A subsumes it. |

## 4. Chosen mechanic — v2

**Combine A + C + D. The short fires only AFTER the unwind has physically materialized on tape, on
the population where the asymmetry should be strongest (mid+small-cap).**

### Universe
- Broad NSE equities, `cap_segment in {mid_cap, small_cap}` (NOT F&O 200)
- 20-day ADV * close >= Rs. 5 Cr (data-quality + tradability floor)

### Day-level qualification (computed at 14:00 IST)
- Intraday return at 14:00 >= +1.0% (the "long bag" exists to be unwound)
- Intraday return at 14:00 <= +6.0% (exclude runaways which typically CNC-convert)
- Not a gap day at open (|gap_pct| <= 2.0%) — cross-detector exclusion vs gap_fade_short

### Tape-confirmation entry trigger (5m bar in [14:30, 15:00] inclusive)
ALL conditions must hold at the bar:
1. `close < vwap` on the bar (VWAP loss — flow has rolled over)
2. `close < open` on the bar (red bar — selling pressure confirmed)
3. `close < min(prev 3 bars' lows)` (structural break)
4. `volume / mean(volume, last 12 prior bars same session) >= 1.5` (volume spike — institutional/algo
   participation, not noise)
5. Latch: first qualifying bar per (symbol, session)

### Entry / risk
- SHORT at qualifying bar's close
- Hard SL = `max(intraday_high_so_far, entry * 1.012)`
- T1 = `entry * 0.995` (~0.5% — half a percent), 50% partial, breakeven trail after
- T2/exit: HARD time-stop 15:10 bar close (5 min before SEBI 15:15 auto-square)

### Differentiation from prior failures (mandatory checks)
- vs Attempt 1: entries are 5-25 min EARLIER (14:30-15:00 vs 14:55-15:15) and use tape-confirmation
  rather than fresh-high selection.
- vs Attempt 2: universe is broad mid+small-cap (not F&O 200 large-cap-dominant); trigger is
  confirmation-based (VWAP loss + volume spike + structural break), not anticipation-based
  (off-the-high + declining volume).

## 5. Pre-registered thresholds (ship gates)

Per gauntlet-v2 spec (`specs/2026-04-25-sub-project-5-gauntlet-v2-postmortem.md`):

| Metric | Gate |
|---|---|
| n (aggregate, Discovery) | >= 500 |
| NET PF | >= 1.10 (marginal), >= 1.30 (strong proceed) |
| Daily Sharpe | > 0 |
| Per-month winning months | >= 55% |
| Top-month NET concentration | <= 40% of aggregate NET |

**Verdict mapping:**
- PF >= 1.30 AND n >= 100 AND per-month stability passes -> STRONG PROCEED -> detailed brief + OOS
- PF in [1.10, 1.30) -> MARGINAL — log for later
- PF < 1.10 OR per-month/concentration fail -> RETIRE with full evidence trail

## 6. Discovery window + regime considerations

- **Window:** 2024-09-01 to 2025-09-30 (13 months).
- **Rationale:** Post-SEBI Sep-2024 STT hike (effective 2024-10-01 hike F&O STT 60% — but does
  NOT affect cash-equity MIS retail flow which is what this setup harvests). Pre-2025-10-01 MWPL
  reform which is a critical fno_structure regime break. The 14-month window cleanly straddles the
  Indian retail behavioral regime after lot-size hikes (NIFTY 25->75, BANKNIFTY 15->30) which
  drove more retail toward single-stock cash MIS — exactly the population this setup targets.
- **Pre-flight regime_break_detector:** depends_on = `["MIS_leverage", "F&O_speculation"]`,
  min_severity=`high`. The window contains 2024-10-01 (STT hike, high — affects F&O speculation
  but the cash-equity MIS unwind is one-sided in the sense it harvests retail MIS-long unwind
  pressure regardless of options STT) and 2025-02-01 (full option-premium upfront, high — affects
  MIS_leverage but for options not cash equity).

  Decision: regime_break_detector will FLAG these. They are documented and accepted: the setup is
  cash-equity MIS unwind, not F&O speculation, and neither STT-on-options nor option-premium-margin
  affect retail's cash-equity MIS long inventory dynamics. The Discovery window is held as-is.

## 7. Out-of-sample plan (only if Discovery passes)

- OOS-1: 2025-10-01 to 2025-12-31 (post-MWPL reform; tests whether tighter F&O limits altered the
  retail cash flow regime).
- OOS-2: 2026-01-01 to 2026-04-30 (post-Budget 2026 STT hike on F&O; tests further regime stability).

Both are deferred until Discovery clears thresholds.

## 8. Outputs

- Sanity script: `tools/sub9_research/sanity_mis_unwind_short_v2.py`
- Trade log: `reports/sub9_sanity/mis_unwind_short_v2_trades.csv` (5,614 rows)
- Sanity console log: `reports/sub9_sanity/_mis_unwind_short_v2.log`
- This research note.

## 9. Discovery results

Run completed 2026-05-14. Pre-flight regime_break_detector flagged 5 high-severity changes in
window (STT Oct-2024, weekly-expiry consolidation Nov-2024, BANKNIFTY+NIFTY lot-size hikes Dec-2024,
option-premium-upfront Feb-2025). All accepted as documented in §6 — they affect F&O speculation
mechanics, not retail cash-equity MIS long inventory dynamics that this setup harvests.

### Aggregate

| Metric | Value | Gate | Pass? |
|---|---|---|---|
| n | 5,614 | >= 500 | YES |
| NET PF | **0.538** | >= 1.10 | **NO** |
| WR | 45.2% | informational | — |
| Daily Sharpe | **-0.495** | > 0 | **NO** |
| Gross PnL | Rs.-31,648 | — | — |
| Fees | Rs.279,602 | — | — |
| NET | **Rs.-311,250** | — | — |

### Per cap_segment

| cap_segment | n | NET PF | WR | NET |
|---|---|---|---|---|
| mid_cap | 2,925 | 0.533 | 44.2% | Rs.-172,418 |
| small_cap | 2,689 | 0.543 | 46.3% | Rs.-138,832 |

Both cap segments fail symmetrically — this is not a small-cap-only or mid-cap-only artifact.

### Per-month stability

13/13 Discovery months, **1 winning month** (2024-11 at PF 1.074 / +Rs.2,861). All other 12 months
are net-negative. Winning-months % = 7.7% (gate >= 55%). Top-month |NET| = 14.3% of aggregate
(passes the concentration gate, but only because losses are spread evenly across all losing months
— a worse failure pattern than concentration).

### Exit-reason breakdown

| reason | n | % of trades | avg_net |
|---|---|---|---|
| time_stop_1510 | 4,898 | 87.2% | Rs.-52 |
| breakeven_trail | 629 | 11.2% | Rs.+55 |
| stop | 87 | 1.6% | Rs.-1,047 |

**Diagnostic:** 87% of trades time-out at 15:10 with avg -Rs.52 — the tape-confirmation entries
fire, the price does NOT continue down enough to hit T1 (0.5%), and 15 minutes later we close
flat-to-slightly-down after fees. Only 11.2% of trades hit T1 partial. The unwind asymmetry, if
it exists at all on 5m mid/small-cap tape, does not propagate enough magnitude post-confirmation
to overcome the Indian fee stack on Rs.1000-risk positions.

### Falsification triggered

1. PF 0.538 < 1.10 floor — RETIRE.
2. Sharpe -0.495 <= 0 — RETIRE.
3. Per-month winning 7.7% < 55% — RETIRE (catastrophic stability failure; 12 of 13 months losing).

## 10. Conclusion

**The MIS auto-square 3:00-3:20 PM forced-unwind asymmetry has now failed under THREE different
mechanics:**

| Attempt | Mechanic | n | PF | WR |
|---|---|---|---|---|
| Original `mis_unwind_short` | Short 14:55-15:15 on fresh-intraday-high (anticipation) | 304 | 0.355 | 9.2% |
| `mis_unwind_short_late_session` | Short 14:30-15:00 off-the-high + declining volume (anticipation, but earlier) | 1,008 | 0.367 | 40.7% |
| **`mis_unwind_short_v2`** (this) | Short 14:30-15:00 ONLY after tape confirmation (VWAP loss + red bar + struct break + RVOL spike) on mid/small-cap | 5,614 | **0.538** | 45.2% |

The tape-confirmation mechanic produced 18x the sample size, doubled WR vs the original, and
materially improved PF (0.36 -> 0.54) — but is still 1/2 the breakeven gate. The marginal
improvement from confirmation is real (the signal-to-noise of tape confirmation IS better than
anticipation) but the asymmetry magnitude on 5m bars in 14:30-15:00 mid/small-cap is too small
to clear the Indian intraday fee stack at our position-sizing convention.

**Recommendation:** the MIS-unwind asymmetry should be **considered exhausted on 5m equity tape
at retail-fee scale**. It MAY still exist for:
- Institutional/HFT participants with much lower fee scales (sub-bps)
- A different data layer (1m or tick-level may show the unwind microstructure better)
- A different harvest mechanism (e.g., options-side: shorting weekly call premium on the same
  population, where the fee-to-asymmetry ratio inverts)

But on the data we own (5m equity bars) at retail fees (Rs.1000-risk MIS-leveraged), this
asymmetry is RETIRED. Future revival should require either (a) tick/1m data + lower-fee
infrastructure, or (b) an options-side execution path.

**Add to `docs/retired_setups.md`** as the third (and final) entry under the `mis_unwind_short`
lineage, with this evidence trail as the closing argument.
