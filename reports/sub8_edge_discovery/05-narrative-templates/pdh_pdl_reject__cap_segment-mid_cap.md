# Narrative Gate — pdh_pdl_reject__cap_segment=mid_cap

## Setup
`pdh_pdl_reject`

## Conditional rule
cap_segment = mid_cap

## Discovery stats
| Metric | Value |
|--------|-------|
| N | 174 |
| PF (full) | 2.546 |
| PF (h1) | 1.973 |
| PF (h2) | 3.484 |
| Win rate | 58.62% |
| Avg PnL (raw, Rs) | 642.65 |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/pdh_pdl_reject.md` — Item 1)

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
- Retail "level traders" - PDH/PDL is THE most-shared level in Indian retail
  Telegram/Discord groups; Zerodha Streak, Sensibull, TradingView all bake
  it into default templates.
- F&O option-writers (HDFC AMC, Kotak MF, ICICI Prudential intraday desks)
  who positioned around round-number strikes that often coincide with PDH/PDL.
- Algo level-test bots scanning for PDH/PDL touches across the F&O 200 universe.

BEHAVIOR:
Retail tests PDH (or PDL) hoping for a breakout. The "absence of volume" condition
is the key: a real breakout has institutional buying with volume confirmation
(volume > 1.5x recent). NO volume surge at PDH = retail-only probe, no real
institutional money pushing through. F&O option-writers (short calls at PDH-
adjacent strike, short puts at PDL-adjacent strike) have OI-sized bets that the
level holds - their economic incentive is to defend it through expiry. The
rejection candle (long upper wick at PDH, long lower wick at PDL) is the
moment writer-defense overwhelms the no-volume retail probe. We fade the
rejection direction.

STRUCTURAL REASON IT PERSISTS:
- Indian F&O OI is concentrated at round-number strikes (Rs 1000, Rs 1500,
  Rs 2000). NSE publishes max-pain calculations daily showing this clustering.
  PDH/PDL often round to within Rs 5 of these strikes, creating a coincidence
  between retail's most-watched level and writers' largest defensive positions.
- Mid-cap selection is critical: large-caps have too much float for option-
  writer defense to actually move price; small-caps do not have F&O at all.
  Mid-caps are the sweet spot where (a) F&O exists, (b) writer defense can
  actually drive price, (c) PDH/PDL is meaningful relative to typical
  intraday range.
- The "absence of volume" filter is microstructurally meaningful: with volume,
  the test BECOMES a breakout (real money pushing through writer defense);
  without volume, it is a probe that gets defended. This is a literal flow
  asymmetry, not chart pattern superstition.
- This persists because option-writers are MULTI-DAY positioned (their writes
  expire weekly/monthly). Their defense is structural, not opportunistic.

CAVEAT: N=174 is small. The mechanism is correct but sample size needs OOS
confirmation. Bootstrap CIs would be wide. Deploy with reduced size pending
2025 OOS validation.

## Pass/fail decision

- [x] APPROVED — narrative plausible and grounded in market reality
- [ ] REJECTED — cannot articulate why this would persist

**Signed:**
**Date:**
