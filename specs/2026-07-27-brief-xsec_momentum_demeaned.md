# Brief: `xsec_momentum_demeaned` — cross-sectional momentum with the confound controls actually run

**Date:** 2026-07-27
**Stage:** 0 (idea) + Stage 1 (Indian-market research) combined
**Family:** MOMENTUM-CONTINUATION, cross-sectional, 2-8 week CNC/MTF horizon
**Status:** **FREEZE DECLINED 2026-07-29 (user decision) — recorded as an era_A artifact for
long-only cash implementation.** Phase-5 was a technical PASS (35/72 eligible cells; locked
`all/f60/decile/hold20/asm_exit`, pooled PF_net 1.42) but the era split is disqualifying in
substance: **the locked cell is NET-NEGATIVE in era_B (PF_net 0.898, −0.57%/position, WR
45.2%)**, and its demeaned alpha stayed positive only because the universe mean fell harder.
f120 alpha goes outright negative in era_B. Since the fresh pool continues era_B structure, a
one-shot would almost certainly fail — the freeze commit and the ledger line were declined
rather than spent. **Overriding a pre-registered PASS in the CONSERVATIVE direction is
permitted; the reverse never is.**
**Live residue (a DIFFERENT candidate, needs its own brief):** momentum still beats the
universe in era_B (+0.2 to +1.0pp demeaned at f60) — that is an **index-hedged long/short**
thesis, a different product with different costs, borrow, and execution. Do not resurrect this
brief for it.
**Kept findings:** ASM falsifier #3 NOT triggered — surveillance migration truncates 5-20% of
positions and consumes ~10-25% of the long-hold winner tail (real regulatory haircut, not
confiscation); asm_exit helps marginally at h20, hurts at h40/h60; stops/targets clip the
tail-carried edge everywhere (the PEAD geometry finding replicates). Evidence:
`reports/sub9_sanity/_xsec_momentum_phase5_cells.csv`,
`tools/sub9_research/xsec_momentum_demeaned_cell_lock.json`, ledger line 2026-07-28.
**Lifecycle:** governed by `docs/setup_lifecycle.md` incl. 2026-07-27 amendments A1-A4.

---

## 1. Why this candidate exists (provenance — read first)

This is the **disciplined redo of the A4 weak kill** flagged by `tasks/lessons.md` #28
(2026-06-15). The original verdict — "mostly survivorship beta" — was ASSERTED, never
measured: the cross-section was never demeaned, so the claimed confound was named in the
conclusion without running the control that isolates it (lesson #28 rule 2, verbatim).

What was actually tested (driver `_tmp_a4_momentum_killtest.py`, preserved at repo root):

- LONG-only top decile of trailing L-day return, skip-2
- Formations {20, 60}d × holds {10, 20}d — i.e. 1-3 month formation only
- Weekly rebalance, ADV floor Rs.20L, CNC cost model + 20bp slip
- Raw (un-demeaned) returns, pooled 2023 → 2026-04, no size control, no short leg

Untested: **cross-sectional demeaning** (the whole point), size-neutralization,
6-12 month formations (where the Indian literature actually locates momentum), loser-leg
behavior, per-tier alpha attribution. **Fresh-pool status:** input data ended 2026-04-30 —
the May-2026+ fresh pool is UNTOUCHED by A4.

## 2. Mechanism statement (one sentence)

Indian equities exhibit cross-sectional momentum — past 3-12-month relative winners
continue to outperform relative losers — driven by gradual information diffusion and
retail disposition-effect selling (winners sold too early, losers held too long), and the
open question this candidate answers empirically is whether that alpha survives
demeaning, size control, and real CNC/MTF costs in OUR tradable universe.

## 3. Indian prior evidence (Stage-1 requirement: ≥2 sources)

1. **Sehgal & Balakrishnan, "Contrarian and Momentum Strategies in the Indian Capital
   Market", Vikalpa (2002)** — foundational Indian momentum evidence; momentum profits
   significant after market/size/value adjustment. Follow-up: Sehgal & Jain (2015),
   "Dissecting sources of price momentum: Evidence from India", Int. J. of Emerging
   Markets.
2. **Garg & Varshney, "Momentum Effect in Indian Stock Market: A Sectoral Study",
   Global Business Review (2015)**
   (https://journals.sagepub.com/doi/abs/10.1177/0972150915569940) — sector-level Indian
   momentum evidence.
3. **"Momentum, reversals and liquidity: Indian evidence", Pacific-Basin Finance Journal
   (2023)** (https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640) —
   recent, directly on the liquidity interaction that matters for our illiquid-tilted
   universe.
4. Practitioner corroboration: NSE's own **Nifty Momentum 30 / Alpha 50** factor indices
   have documented long-run alpha over Nifty 500 — momentum is a real, harvested factor in
   India at institutional scale (which cuts both ways: crowding risk, noted in §6).

## 4. Participants

- **Slow side (we join):** gradually-diffusing fundamental information; institutional
  flow that scales into 6-12-month winners over weeks (index/factor funds rebalance
  quarterly — a structural chaser).
- **Behavioral supply (why it persists):** retail disposition effect — documented heavy
  retail presence in Indian mid/small caps selling winners into strength provides the
  liquidity momentum buyers need; losers are held (no forced selling) so loser-drift is
  slower, which is why the LONG leg is expected to dominate.
- **Relationship to the book:** this is the empirically-orthogonal direction — the July
  2026 factor study showed our entire surviving book is capitulation-REVERSION; momentum
  is its natural diversifier (and the two are classically negatively correlated in
  crash months).

## 5. Pre-registered variant grid (the FULL battery)

| Dimension | Values |
|---|---|
| Return construction | raw; **cross-sectionally demeaned** (minus universe mean); market-adjusted (minus Nifty) |
| Size control | none; within-cap-segment ranking (per A3 cap segments) |
| Formation | 20, 60, **120, 250** trading days, skip-2 (literature: 6-12mo is where momentum lives; 1-3mo contaminated by reversal) |
| Hold | 10, 20, **40, 60** trading days |
| Legs | winner-decile LONG (primary); loser-decile as measurement (shortability quantified before any short claim) |
| Threshold | decile; quintile; top-5% |
| Universe | ADV tiers 1-5; MTF-leverageable subset; ASM/GSM-clean subset |
| Rebalance | weekly; monthly |

**The decisive Phase-2 statistic is DEMEANED alpha per cell** — raw return is reported but
carries no evidential weight (that was A4's mistake).

## 6. Falsifiers (3, pre-registered)

1. **Confound falsifier (the #28 control):** if cross-sectionally demeaned winner-decile
   alpha ≤ 0 across the formation grid, the raw effect was beta/survivorship — kill. This
   single number settles what A4 asserted.
2. **Capturability falsifier:** if demeaned alpha survives only in the non-MTF,
   cap=unknown thin tail (per A3 hardened universe + tradability intersection), the edge
   is not implementable at our capital — kill or park explicitly as capacity-blocked.
3. **Indian-specific structural falsifier — ASM/GSM migration:** small-cap winners
   systematically migrate INTO surveillance lists (100% margin, price bands), truncating
   momentum exactly when it's strongest. If alpha net of ASM-entry events collapses, the
   theoretical edge is regulatorily confiscated — kill. (Requires the ASM/GSM backfill —
   fetcher exists, `tools/asm_gsm_history/`, parquet never materialized; ~2.5-day chore
   that also unblocks the stalled `asm_gsm_stage_transition` candidate.)

## 7. Data feasibility + prerequisites

| Input | On disk | Coverage | Action needed |
|---|---|---|---|
| `cache/preaggregate/clean_daily_from5m.feather` (CA-adjusted — mandatory, lesson: the CNC reversal "edge" was 100% bad prints) | yes | 2023-01-02 → 2026-04-30 | **rebuild through 2026-07** |
| Cap segments + MTF universe | yes | data/cap_segments/, data/mtf_universe/ | intersect (A3) |
| ASM/GSM history | fetcher only | — | **run backfill** (falsifier #3 + sibling candidate) |
| ProductionUniverseGate | yes | — | mandatory from Phase 2 |

Note: 250-day formation on a 2023-start CA-adjusted daily set means first tradeable
cross-section ~2024-01 — Discovery window effectively 2024-2025 for long formations;
acceptable, documented here so it isn't "discovered" mid-study.

## 8. Regulatory sensitivity

- ASM/GSM surveillance framework is THE Indian-specific risk (falsifier #3) — no US-style
  momentum study accounts for it; this is where our result will differ from literature.
- MTF: long-side leverage available on the MTF list only; Rs-capital sizing must use
  MTF-eligible subset for the implementable claim.
- Delivery STT 0.20% + impact in illiquid names at 2-8-week holds: cost model from the
  A4 script carries over (it was correct); monthly-rebalance cells exist specifically to
  amortize it.
- No dependence on F&O rules (Oct-2025 cutover irrelevant to cash long leg).

## 9. Adjacent setups

- `illiquid_momentum_long` (PARKED, LPGD-passed, zero code): same family. Explicit
  relationship: THIS study's demeaning + ASM controls are exactly the confound checks that
  parked candidate never got. If `xsec_momentum_demeaned` confirms demeaned alpha,
  illiquid_momentum_long un-parks with its validation strengthened; if demeaned alpha is
  zero, it stays parked permanently. Either way one study settles two candidates — they
  share ONE factor budget (amendment A4 logic) and count as near-duplicates for ledger M.
- `pead_reaction_drift` (sibling brief): partial overlap (earnings moves enter formation
  returns); measure cross-correlation if both survive.
- Retired relatives: sub-7/8 momentum patterns (gap_and_go, orb_15, ema5 etc.) died as
  UNIVERSAL patterns without participant asymmetry — this candidate's asymmetry is the
  disposition-effect supply + ASM truncation structure, and it operates at weeks, not
  intraday, where those retail patterns bled.

## 9b. Phase-5 sweep pre-registration (written 2026-07-28, BEFORE the ASM parquet
materialized and BEFORE any 2025+ signal statistic was computed for this candidate)

Stage-4 passed (alpha preserved within 2-5%; MTF subset retains ~88%). The Stage-4
distributional read (sub-50% hit rate, tail-carried alpha, mean MAE ~−15%) couples exit
geometry to the ASM-truncation falsifier, so the exit grid includes an ASM-aware mode.
Grid locked now:

- **Development pool + era split (amendments A1+A5):** signals 2023-01 → 2026-04-30, split
  era_A = 2023-01→2024-12 (illiquid-bull) / era_B = 2025-01→2026-04 (post-clampdown).
  **Lockable cells require demeaned alpha > 0 in BOTH eras** (A5 sign rule) with n ≥ 100
  positions per era, plus pooled PF_net ≥ 1.20 (lifecycle ship floor). The demeaned-alpha
  statistic remains decisive; raw returns diagnostic only.
- **Cell dimensions:** formation {60, 120} (Stage-4 locked variants) × hold {20, 40, 60}
  sessions × cohort {winner decile, top-5%} × universe {all, MTF-eligible, ASM-clean-at-entry}.
  Rebalance weekly (Stage-4 locked). No dimensions added after results are seen.
- **Exit modes (selectable):** hold-to-H close; **exit-on-ASM-entry** (position's symbol
  enters any ASM/GSM stage → sell next session open, else hold to H). **Diagnostic-only
  (never selectable):** 3-sigma vol-scaled target and any stop variant — the PEAD Phase-5
  finding (targets/stops clip tail-carried drift edges) transfers.
- **Costs:** CNC round-trip (STT 0.20% + brokerage 0.06% + charges 0.047%) + 20bp slippage,
  per-position equal notional (A2-family cost model).
- **Ledger:** the sweep is ONE evaluation touching discovery+demoted windows → one ledger
  line regardless of cell count; the later fresh-pool one-shot is a second line.
- **After lock:** freeze commit (spec + cell JSON + geometry), then ONE shot on the fresh
  pool (2026-05-01 → present) + paper as the decisive gates. ASM data source:
  `data/asm_gsm_history/asm_gsm_events.parquet` (backfill in progress at time of writing;
  this section was committed before the file existed — verifiable by commit order).

## 10. A1/A2 compliance plan

Identical to the sibling brief: development on Discovery + demoted windows (2023 →
2026-04); freeze commit after cell-lock; decisive gate = one-shot fresh pool (2026-05-01+)
+ paper; every window evaluation logs one `docs/experiment_ledger.jsonl` line in-session.
