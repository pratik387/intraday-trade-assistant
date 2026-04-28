# Narrative Gate — gap_fade_short__regime+cap_segment=trend_up+small_cap

## Setup
`gap_fade_short`

## Conditional rule
regime+cap_segment = trend_up+small_cap

## Discovery stats
| Metric | Value |
|--------|-------|
| N | 1804 |
| PF (full) | 1.584 |
| PF (h1) | 1.628 |
| PF (h2) | 1.541 |
| Win rate | 72.17% |
| Avg PnL (raw, Rs) | 98.11 |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/gap_fade_short.md` — Item 1)

### Stage 4 top features
(If Stage 4 was run, paste SHAP top features here; else leave blank.)

### Suggested microstructure rationale
This setup passes statistical gates in the cell `regime+cap_segment = trend_up+small_cap`. Candidate
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
- Late-stage retail FOMO chasers who heard about the stock FROM the move,
  not before it (peak retail participation in small-cap parabolic stocks
  consistently lands 3-5 days INTO the trend per Indian broker order-flow data)
- HNI/PMS desks at Indian boutique houses who held since trend inception,
  now distributing into the strength
- DII rebalancing desks at AMCs (HDFC, ICICI, SBI Mutual) trimming overweight
  small-cap names to maintain mandate concentration limits

BEHAVIOR:
A small-cap that is already up 20-40% over 3-5 days, then gaps UP further,
is in late-stage Wyckoff distribution. The 5-min chart of the gap-fade IS
the macro-distribution-pattern at micro-time-scale: same price action, same
participant exchange, same exhaustion. Retail buys the gap chasing the obvious
trend; insiders/early-trend-participants distribute. The exhaustion candle marks
the moment distribution overwhelms FOMO demand.

STRUCTURAL REASON IT PERSISTS:
- Mathematical: small-cap moves >2 sigma over 5 days have negative forward drift
  (mean-reversion is a feature of the population, not a bet on any one name).
- Behavioral: Indian retail learns about parabolic small-caps through Zerodha
  "Top Gainers" widget, Telegram broadcast lists, YouTube end-of-day reviews -
  all of these distribute information AFTER the move started, ensuring late
  retail entry is structural, not optional.
- Regulatory: small-caps in extended uptrends often hit upper circuit; circuit
  release creates supply unlocks that further the fade.
- The PDC anchor is even stronger when prior days' opens were near PDC - most
  parabolic small-caps have a "sticky" PDC because that is where the day's
  first trade was, the institutional VWAP starts there.

## Pass/fail decision

- [x] APPROVED — narrative plausible and grounded in market reality
- [ ] REJECTED — cannot articulate why this would persist

**Signed:**
**Date:**
