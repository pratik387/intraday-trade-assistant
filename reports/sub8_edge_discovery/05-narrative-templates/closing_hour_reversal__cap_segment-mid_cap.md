# Narrative Gate — closing_hour_reversal__cap_segment=mid_cap

## Setup
`closing_hour_reversal`

## Conditional rule
cap_segment = mid_cap

## Discovery stats
| Metric | Value |
|--------|-------|
| N | 340 |
| PF (full) | 1.361 |
| PF (h1) | 1.619 |
| PF (h2) | 1.272 |
| Win rate | 49.41% |
| Avg PnL (raw, Rs) | 24.43 |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/closing_hour_reversal.md` — Item 1)

### Stage 4 top features
(If Stage 4 was run, paste SHAP top features here; else leave blank.)

### Suggested microstructure rationale
This setup passes statistical gates in the cell `cap_segment = mid_cap`. Candidate
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
- Retail intraday MIS traders (longs in up-move, shorts in down-move) who
  approach FORCED cover at 15:25 (Zerodha/Upstox auto-square-off; some brokers
  do it 5-10 min earlier).
- Late-FOMO retail who entered after 14:00 chasing the obvious intraday trend.
- Prop desks (Indian arb shops + a few foreign HFT) who systematically fade
  EOD extremes; this is a known book at multiple Indian quant funds.

BEHAVIOR:
14:30-15:15 is the MIS-cover hot zone. A mid-cap that is stretched >=1.5%
intraday in one direction has accumulated MIS positions on the trend side.
As 15:15 approaches, those traders MUST flatten - no MIS overnight under SEBI
rules. The cover flow goes AGAINST the established direction (longs SELL to
cover, shorts BUY to cover). Late FOMO buyers from 14:00-14:30 get caught:
they have no time to recover from any retracement before 15:25 force-close.
The exhaustion candle (large body, vol > recent) marks the inflection: the
LAST aggressive trend-direction entries before MIS-unwind dominates.

STRUCTURAL REASON IT PERSISTS:
- SEBI rule: MIS positions auto-square-off at 15:25 - HARD coded in broker
  risk systems, not optional. This is regulation, not preference.
- Mid-cap liquidity is high enough that MIS-cover flow actually moves price.
  Small-caps are already at intraday extremes by 14:30 (illiquid). Large-caps
  are too liquid for retail MIS cover to matter (institutional flow drowns it).
- The 14:30-15:15 window catches the inflection AS cover flow takes over from
  trend-direction flow. By 15:15 the cover is mostly done - the move is
  exhausted by then. Entering during the rollover is the structural sweet spot.
- This persists because SEBI's intraday-only-leverage policy is structural law
  enforced in broker code; it is not going to change.

CAVEAT: N=340 is moderate. WR 49% with PF 1.36 means winners are larger
than losers - typical for fade-the-extreme setups. Deploy at standard size
with 2025 OOS confirmation.

## Pass/fail decision

- [x] APPROVED — narrative plausible and grounded in market reality
- [ ] REJECTED — cannot articulate why this would persist

**Signed:**
**Date:**
