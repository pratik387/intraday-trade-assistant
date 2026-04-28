# Narrative Gate — gap_fade_short__cap_segment+hour_bucket=small_cap+opening

## Setup
`gap_fade_short`

## Conditional rule
cap_segment+hour_bucket = small_cap+opening

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
This setup passes statistical gates in the cell `cap_segment+hour_bucket = small_cap+opening`. Candidate
mechanisms to consider:
- Retail long-bias: do losers in the opposite direction cluster under this condition?
- Institutional flow: does this regime correlate with measurable FII/DII activity?
- Microstructure: is price action at this hour bucket dominated by MIS unwinding,
  opening auction noise, or expiry gamma flow?

## Human narrative (REQUIRED — unfilled = auto-REJECT)

### WHY does this work? What market participant behavior creates this edge?

**SEE CANONICAL: `gap_fade_short__cap_segment-small_cap.md`**

Identical trade set (opening hour is the full active window for this setup).

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
