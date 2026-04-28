# Narrative Gate — gap_fade_short__cap_segment=small_cap

## Setup
`gap_fade_short`

## Conditional rule
cap_segment = small_cap

## Discovery stats
| Metric | Value |
|--------|-------|
| N | 3797 |
| PF (full) | 1.496 |
| PF (h1) | 1.557 |
| PF (h2) | 1.438 |
| Win rate | 70.27% |
| Avg PnL (raw, Rs) | 80.0 |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/gap_fade_short.md` — Item 1)

### Stage 4 top features
(If Stage 4 was run, paste SHAP top features here; else leave blank.)

### Suggested microstructure rationale
This setup passes statistical gates in the cell `cap_segment = small_cap`. Candidate
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
Two opposing flows clash in the first 15 minutes of NSE trading on small-caps:
(a) Retail intraday MIS chasers via Zerodha/Upstox/Groww - buying at market in the
    first 5-10 mins after a tip-sheet/Telegram pump or "Top Gainers" widget catch.
(b) Prop HFT desks (AlphaGrep, Quadeye, Tower) and Indian arb shops who built short
    inventory in the pre-open call auction (9:00-9:08) at the offer side near gap_high.
The retail demand eats the offer book; once retail demand exhausts, prop desks
need price to drop to mark their book to profit, so they pull bids and refill offers.

BEHAVIOR:
Retail BUY-at-market is fixed flow - it is ALWAYS gone within 5-10 mins because
small-cap retail momo has no real institutional cohort behind it. The exhaustion
candle (long upper wick + small body + declining vol) is the literal microstructure
moment when retail demand dries up - price tested up, could not hold, came back to
body on shrinking volume. PDC is the overnight equilibrium price; once the
intraday squeeze pressure clears, price drifts back there because there is no
real overnight-to-intraday valuation gap that justifies the gap.

STRUCTURAL REASON IT PERSISTS:
1. SEBI MIS leverage cap (5x intraday, no overnight) post-2020 - small-cap retail
   has zero staying power. First SL hit = forced flat. The cover flow IS the fade.
2. NSE small-cap float is 70-90% retail/HNI. No DII benchmark coverage, no FII
   index inclusion. When retail flow ends, there is literally nobody else to bid.
3. Pre-open call auction (9:00-9:08) does not reach true equilibrium for thinly-
   traded small-caps. The 9:15 open is the FIRST real price discovery; the
   reactive 5-10 min after is structural unwinding.
4. Indian gap-fade studies (TradingQnA, Smallcase) consistently show ~50-60%
   gap-close rate on small-caps - the rate is high because the structural
   conditions above are SEBI/NSE rules that do not change with sentiment cycles.

This is not a behavior pattern that fades when "everyone knows it" - the
edge is in the FORCED flows (MIS cover, prop inventory unwind, no overnight
support). Those mechanics are codified in regulation and exchange architecture.

### Why does the FILTER help?
N/A - this is the unfiltered Stage 3 cell.

## Pass/fail decision

- [x] APPROVED — narrative plausible and grounded in market reality
- [ ] REJECTED — cannot articulate why this would persist

**Signed:**
**Date:**
