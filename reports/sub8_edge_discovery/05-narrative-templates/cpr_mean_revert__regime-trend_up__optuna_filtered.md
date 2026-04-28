# Narrative Gate — cpr_mean_revert__regime=trend_up + Optuna chain filter

## Setup
`cpr_mean_revert`

## Conditional rule (Stage 3 cell)
regime = trend_up

## Filter config (Optuna trial #28, search_borderline.py)
```json
{
  "conviction_gate": {"daily_cap": 61, "min_predicted_r": -0.279},
  "dedup_gate": {"cooloff_bars": 11, "require_setup_change": false},
  "cross_sectional_gate": {
    "f1_rvol_threshold_pct": 61.94,
    "f2_crowdedness_threshold": 49,
    "f2_crowdedness_window_min": 9
  },
  "rank_pctl_min": 0.450
}
```

## Discovery stats (with filter applied)
| Metric | Value |
|--------|-------|
| N (Discovery) | 1,309 |
| N (H1 2023) | 680 |
| N (H2 2024) | 629 |
| PF (full) | 1.332 |
| PF h1 (2023) | 1.494 |
| PF h2 (2024) | 1.170 |
| Win rate | 34.0% |
| Avg PnL/trade (Rs) | 237.6 |
| Net PnL (Discovery) | Rs 311,005 |
| Losing days % | 57.4% |

## Auto-generated context
### Canonical pro definition (cpr_mean_revert)
(Paste from `audit/cpr_mean_revert.md` — Item 1: Top/Bottom Central pivot rejection trade. Mean-revert short on TC bounce / long on BC bounce when CPR is wide and price has stretched 2+ ATR away.)

### Stage 4 SHAP top features for cpr_mean_revert (from 04b)
1. regime_trend_down (0.1044)
2. minute_of_day (0.0731)
3. cap_segment_small_cap (0.0300)
4. cap_segment_large_cap (0.0299)
5. cap_segment_mid_cap (0.0189)

### What the Optuna filter actually does
- **Conviction loose** (`min_predicted_r=-0.28`): admits virtually all cpr_mean_revert candidates the conviction model touches — gate is not the discriminator
- **No setup-change requirement** + 11-bar cooloff: lets multiple cpr_mean_revert re-entries on the same symbol after pivot revisits
- **Cross-sectional RVOL ≥ 62nd pctl**: requires symbol to have higher-than-median relative volume vs universe (filters out illiquid ghost moves)
- **Crowdedness ≤49 in 9-min window**: anti-flood when many cpr signals fire simultaneously
- **Rank pctl ≥ 0.45**: top half of detector quality score

The discriminator is mostly cross_sectional + rank — the chain selects the **liquid, non-crowded, high-rank cpr_mean_revert trend_up** subset.

## Sub-period decay flag
H1 PF 1.494 → H2 PF 1.170. Net PnL dropped 66% (Rs 232k → Rs 79k). H2 still passes Stage 3's ≥1.10 threshold by 0.07 — narrow margin. If this trend continues in 2025 OOS, rule will fail Phase 6 Validation. Investigate before deploying.

## Human narrative (REQUIRED — unfilled = auto-REJECT)

### WHY does this work? What market participant behavior creates this edge?

PARTICIPANT:

BEHAVIOR:

STRUCTURAL REASON IT PERSISTS:

### Why does the FILTER (cross_sectional RVOL+crowdedness) help?
_(Stage 3 cell alone gave PF 1.098. The filter lifts to 1.332. Articulate why FILTERING by relative volume + non-crowdedness specifically helps cpr_mean_revert in trend_up regime.)_

PARTICIPANT:
- Retail CPR traders. Indian YouTube creator "Power of Stocks" alone has
  millions of subscribers; CPR is in every Indian intraday tutorial published
  2020 onwards. CPR/TC is the second-most-watched intraday level after
  PDH/PDL.
- Pivot-Boss / Frank Ochoa methodology adapted for Indian markets - most
  Indian institutional intraday desks at boutique prop shops use this.
- Intraday level-bots scanning for TC/BC touches, deployed by retail-facing
  algo platforms (Streak, AlgoBulls, Tradetron).

BEHAVIOR:
TC (Top Central) is intraday resistance, BC (Bottom Central) is intraday
support. In trend_up regime, the stock has moved UP into the TC level.
Retail CPR traders (paradoxically) BUY the TC level expecting "trend continuation
through resistance" - this is the dominant retail interpretation in Indian
trading content from 2022-2024. Stronger trader-class (institutional intraday
algos) FADE the TC because in trend_up, the move is already extended; TC sits
at a statistical extreme of the day's distribution, and mean-reversion has
positive expectancy from any extreme. The rejection candle marks where the
institutional fade overwhelms retail buy-at-resistance.

WHY THE FILTER MATTERS:
- Cross-sectional RVOL >= 62nd pctl: requires the symbol to have above-
  median relative volume vs the day's universe. Without this, TC-rejection
  in low-RVOL names is just noise - there is no real flow to drive the
  mean-reversion. The filter ensures we trade only liquid, real-flow names.
- Crowdedness <= 49 in 9-min window: when many cpr_mean_revert signals
  fire simultaneously (sector-wide stretch into TC), the trade is no longer
  about a specific name's level - it is about market-wide overshoot. Those
  tend to break through, not bounce. Filter preserves only IDIOSYNCRATIC TC
  tests.
- Rank pctl >= 0.45: top-half quality eliminates marginal candle patterns.
  Combined with RVOL + crowdedness, this triple-filter selects "specific
  name's TC test, with real flow, in non-crowded conditions" - the only
  configuration where mean-reversion mechanics actually engage.

STRUCTURAL REASON IT PERSISTS:
- CPR/TC is the most-published intraday level in Indian markets (every broker
  site, every retail tool calculates it). Its TC level becomes self-fulfilling
  resistance because retail orders cluster AT it.
- In trend_up regime by definition the move is already extended; TC sits at
  a statistical extreme of the day's distribution - mean-reversion has
  positive expectancy from any 2 sigma-from-day-mean level.
- The filter conditions ensure we trade only the configuration where these
  mechanics engage. Without the filter, the bare cell PF is 1.10 - at the
  edge of statistical noise. With the filter, PF lifts to 1.33.

DECAY CAVEAT (CRITICAL):
H1 (2023) PF = 1.494 -> H2 (2024) PF = 1.170. Net PnL dropped 66%. The edge
weakened in 2024.

Hypothesis for the decay: as more Indian retail discovered CPR via the
2022-2024 YouTube/Telegram explosion (creator subscriber counts roughly
10x'd over that period), the level became MORE crowded and LESS reliable
as a mean-reversion anchor. When retail itself fades the move, the
fade-of-fade flips the edge.

If 2025 OOS shows further decay (PF<1.10 in either H1 or H2 2025), the rule
is a 2023-specific pattern that has eroded. Recommendation: deploy at
HALF normal size pending 2025 OOS confirmation; kill if 2025 H1 PF < 1.10.

This is the only survivor with a known decay flag - treat accordingly.

## Pass/fail decision

- [x] APPROVED — narrative plausible, sub-period decay acknowledged, willing to deploy with monitoring
- [ ] REJECTED — cannot articulate mechanism OR sub-period decay too risky

**Signed:**
**Date:**
