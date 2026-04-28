# Narrative Gate — gap_fade_short__regime=squeeze

## Setup
`gap_fade_short`

## Conditional rule
regime = squeeze

## Discovery stats
| Metric | Value |
|--------|-------|
| N | 894 |
| PF (full) | 1.496 |
| PF (h1) | 1.549 |
| PF (h2) | 1.469 |
| Win rate | 69.69% |
| Avg PnL (raw, Rs) | 72.49 |

## Auto-generated context
### Canonical pro definition
(Paste from `docs/edge_discovery/audit/gap_fade_short.md` — Item 1)

### Stage 4 top features
(If Stage 4 was run, paste SHAP top features here; else leave blank.)

### Suggested microstructure rationale
This setup passes statistical gates in the cell `regime = squeeze`. Candidate
mechanisms to consider:
- Retail long-bias: do losers in the opposite direction cluster under this condition?
- Institutional flow: does this regime correlate with measurable FII/DII activity?
- Microstructure: is price action at this hour bucket dominated by MIS unwinding,
  opening auction noise, or expiry gamma flow?

## Human narrative (REQUIRED — unfilled = auto-REJECT)

### WHY does this work? What market participant behavior creates this edge?

**SEE CANONICAL: `gap_fade_short__regime_and_cap_segment-squeeze_and_small_cap.md`**

Identical trade set (gap_fade_short fires only in opening hour, only on small/mid/micro caps; the squeeze regime cell IS the squeeze+small_cap subset).

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
