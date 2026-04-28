# Narrative Gate — gap_fade_short__regime=trend_up

## Setup
`gap_fade_short`

## Conditional rule
regime = trend_up

## Discovery stats
| Metric | Value |
|--------|-------|
| N | 3297 |
| PF (full) | 1.441 |
| PF (h1) | 1.457 |
| PF (h2) | 1.425 |
| Win rate | 70.15% |
| Avg PnL (raw, Rs) | 76.14 |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/gap_fade_short.md` — Item 1)

### Stage 4 top features
(If Stage 4 was run, paste SHAP top features here; else leave blank.)

### Suggested microstructure rationale
This setup passes statistical gates in the cell `regime = trend_up`. Candidate
mechanisms to consider:
- Retail long-bias: do losers in the opposite direction cluster under this condition?
- Institutional flow: does this regime correlate with measurable FII/DII activity?
- Microstructure: is price action at this hour bucket dominated by MIS unwinding,
  opening auction noise, or expiry gamma flow?

## Human narrative (REQUIRED — unfilled = auto-REJECT)

### WHY does this work? What market participant behavior creates this edge?

**SEE CANONICAL: `gap_fade_short__regime_and_cap_segment-trend_up_and_small_cap.md`**

Identical trade set - see canonical.

The narrative for this cell IS the narrative for the canonical cell.
Approving this duplicate is equivalent to approving the canonical.

PARTICIPANT: (see canonical)

BEHAVIOR: (see canonical)

STRUCTURAL REASON IT PERSISTS: (see canonical)

## Pass/fail decision

- [x] APPROVED — narrative plausible and grounded in market reality
- [ ] REJECTED — cannot articulate why this would persist

**Signed:**
**Date:**
