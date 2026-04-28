# Narrative Gate — gap_fade_short__hour_bucket=opening

## Setup
`gap_fade_short`

## Conditional rule
hour_bucket = opening

## Discovery stats
| Metric | Value |
|--------|-------|
| N | 6796 |
| PF (full) | 1.365 |
| PF (h1) | 1.363 |
| PF (h2) | 1.368 |
| Win rate | 68.7% |
| Avg PnL (raw, Rs) | 60.99 |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/gap_fade_short.md` — Item 1)

### Stage 4 top features
(If Stage 4 was run, paste SHAP top features here; else leave blank.)

### Suggested microstructure rationale
This setup passes statistical gates in the cell `hour_bucket = opening`. Candidate
mechanisms to consider:
- Retail long-bias: do losers in the opposite direction cluster under this condition?
- Institutional flow: does this regime correlate with measurable FII/DII activity?
- Microstructure: is price action at this hour bucket dominated by MIS unwinding,
  opening auction noise, or expiry gamma flow?

## Human narrative (REQUIRED — unfilled = auto-REJECT)

### WHY does this work? What market participant behavior creates this edge?

**SEE CANONICAL: `gap_fade_short__cap_segment-small_cap.md`**

gap_fade_short fires only in opening hour by construction; this cell == the entire setup population. The cap=small_cap cell is the higher-PF refinement.

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
