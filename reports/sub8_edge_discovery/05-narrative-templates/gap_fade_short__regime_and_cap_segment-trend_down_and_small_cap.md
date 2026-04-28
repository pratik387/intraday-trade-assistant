# Narrative Gate — gap_fade_short__regime+cap_segment=trend_down+small_cap

## Setup
`gap_fade_short`

## Conditional rule
regime+cap_segment = trend_down+small_cap

## Discovery stats
| Metric | Value |
|--------|-------|
| N | 1051 |
| PF (full) | 1.427 |
| PF (h1) | 1.502 |
| PF (h2) | 1.351 |
| Win rate | 69.08% |
| Avg PnL (raw, Rs) | 63.47 |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/gap_fade_short.md` — Item 1)

### Stage 4 top features
(If Stage 4 was run, paste SHAP top features here; else leave blank.)

### Suggested microstructure rationale
This setup passes statistical gates in the cell `regime+cap_segment = trend_down+small_cap`. Candidate
mechanisms to consider:
- Retail long-bias: do losers in the opposite direction cluster under this condition?
- Institutional flow: does this regime correlate with measurable FII/DII activity?
- Microstructure: is price action at this hour bucket dominated by MIS unwinding,
  opening auction noise, or expiry gamma flow?

## Human narrative (REQUIRED — unfilled = auto-REJECT)

### WHY does this work? What market participant behavior creates this edge?
_(Human-written. Reference a specific participant, a specific behavior, and a
specific structural reason the edge persists. LLM-plausible is insufficient —
write only what you would defend to another trader.)_

PARTICIPANT:
- Bottom-pickers + dead-cat-bounce buyers (retail "value at any green candle"
  cohort)
- Short-cover flow from intraday-shorts who were profitable during the
  prior downtrend's last few days
- Trapped longs from earlier-trend higher prices who use any strength to exit
  (especially MIS retail that converted to delivery and is now stuck)

BEHAVIOR:
A gap-up against an established downtrend is news-driven (results, upgrade,
sector rotation) - but news in small-caps does not change the institutional
ownership; it just creates a temporary liquidity pocket for trapped supply
to exit. Retail bottom-pickers buy the gap, short-covers add buying - but
TRAPPED LONGS at higher prices use the strength to flatten. Forced supply
> opportunistic demand. The exhaustion candle marks when short-covers complete
and trapped supply dominates - usually faster than trend_up gaps because the
overhead supply has been stacking for days.

STRUCTURAL REASON IT PERSISTS:
- Small-cap downtrend creates an "overhead supply ladder" at every prior
  higher price. ANY relief rally hits dense supply at 5/10/20-day moving
  averages where prior longs entered.
- MIS retail cannot carry overnight; the losses on prior days are "delivery
  conversions" - those positions are now sitting in delivery accounts with
  unrealized losses. The first green candle is psychologically the trigger
  for "exit at break-even" which is just slightly higher.
- This is cleaner than trend_up because there is no confusion about direction.
  Distribution is happening at every tick once the gap-up shows pulse-strength.

## Pass/fail decision

- [x] APPROVED — narrative plausible and grounded in market reality
- [ ] REJECTED — cannot articulate why this would persist

**Signed:**
**Date:**
