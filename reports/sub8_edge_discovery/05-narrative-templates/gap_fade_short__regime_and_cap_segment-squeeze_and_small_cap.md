# Narrative Gate — gap_fade_short__regime+cap_segment=squeeze+small_cap

## Setup
`gap_fade_short`

## Conditional rule
regime+cap_segment = squeeze+small_cap

## Discovery stats
| Metric | Value |
|--------|-------|
| N | 499 |
| PF (full) | 1.556 |
| PF (h1) | 2.034 |
| PF (h2) | 1.378 |
| Win rate | 69.54% |
| Avg PnL (raw, Rs) | 81.93 |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/gap_fade_short.md` — Item 1)

### Stage 4 top features
(If Stage 4 was run, paste SHAP top features here; else leave blank.)

### Suggested microstructure rationale
This setup passes statistical gates in the cell `regime+cap_segment = squeeze+small_cap`. Candidate
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
Same as the cap=small_cap anchor (retail momo vs prop short inventory) PLUS:
positional traders who accumulated tight-stop positions during the squeeze
period - both longs and shorts with stops clustered at narrow range edges.

BEHAVIOR:
A small-cap that is compressed for days (BB width tight) has positional traders
on BOTH sides with tight stops because low vol allowed tight risk. The overnight
gap-up triggers a stop cascade: shorts stopped -> buy-to-cover adds to the early
retail momo flow. By 9:20-9:25 the cascade is complete; AFTER that, there is
nobody left to buy because the squeeze build had no fundamental flow underneath
it. The fade is faster and cleaner than non-squeeze gaps because there is literally
zero conviction underneath the move.

STRUCTURAL REASON IT PERSISTS:
- Volatility-compression-then-expansion is mathematically a coiled spring; the
  release direction is information, but the magnitude reverts. In small-caps
  with no fundamental anchor, the post-release reversion is mechanical.
- Squeeze stops cluster predictably at 1.5-2 sigma from the compression range.
  Every Indian retail platform's "auto-stop" defaults trigger here. The
  cascade is engineered, not random.
- This persists because the BB squeeze is a derivative of price; you cannot
  arbitrage away the math itself.

### Why the squeeze REGIME amplifies the cap=small_cap edge
- Higher PF (1.55 vs 1.50 anchor) at smaller N (499 vs 3,797) - the squeeze cell
  is a higher-quality subset of the anchor.
- The "squeeze gap" is a textbook Wyckoff "spring then up-thrust" - every textbook
  fades it; the Indian retail population that knows this is small enough that
  the edge survives crowd erosion.

## Pass/fail decision

- [x] APPROVED — narrative plausible and grounded in market reality
- [ ] REJECTED — cannot articulate why this would persist

**Signed:**
**Date:**
