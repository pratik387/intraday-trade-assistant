# Brief: universe re-cut — ₹-turnover cap + continuous-liquidity-history gate

**Date:** 2026-07-29
**Type:** UNIVERSE re-specification (not a new setup) — applies to mechanisms already owned
**Status:** **REJECTED at the fresh-pool one-shot, 2026-07-29.** (Pre-registration committed
`550071f` BEFORE any fresh-pool statistic existed; freeze verifiable by commit order.)

**RESULT (§7 applied mechanically).** Power gate PASSED (pooled A 2,224 fires / B 556, gate
≥50). Outcomes, absolute statistic per A5-b:

| decision unit | n_A | exp_bps A | PF A | n_B | exp_bps B | PF B | B better |
|---|---|---|---|---|---|---|---|
| below_vwap_long | 621 | +6.20 | 1.018 | 170 | +15.91 | 1.084 | **yes** |
| capitulation_multiday | 466 | −38.79 | 0.786 | 326 | −45.35 | 0.748 | no |
| close_dn_overnight | 243 | **+19.34** | **1.601** | 59 | −16.93 | 0.682 | no |
| short_open_fade | 894 | −6.17 | 0.866 | 1 | −94.31 | — | ineligible (n_B<15) |
| **POOLED** | **2,224** | **−6.76** | **0.837** | **556** | **−23.69** | **0.759** | — |

**REJECT:** pooled B expectancy worse than A by ~250% of |A| (far outside the ±10% NO-CHANGE
band); majority test fails independently (B better in 1 of 3 eligible units). §8.2
decomposition flag NOT raised (B did not win). **Do not change production universes.**
Mechanistically the gates gut rather than refine: B retains only 24-27% of close_dn and
below_vwap fires and **0.1%** of or_window (894→1), and damages the very mechanism they were
meant to protect (capitulation −38.8 → −45.3 bps).

**PROCESS DISCLOSURE (recorded deliberately).** The evaluation script was run TWICE. Run 1
scored `below_vwap_volume_revert_long` on its raw sanity population (18,518 fires, ~100× its
production rate) instead of its shipped 3D cell lock + locked exit grid — an unfaithful
replication. It was corrected (via `cell_sweep._row_simulate`, the same resim
`apply_lock_below_vwap_oos_ho.py` uses) and re-run; the table above is run 2. **Both runs
returned REJECT** (run 1: pooled A −10.16 vs B −15.59 bps, 0/3 units B-better). The correction
could not have reached ADOPT (that needs 2 of 3 units; below_vwap is one). Run 1's ledger line
was deleted so exactly one line exists — the record is therefore "one evaluation, run twice,
first run defective", disclosed here rather than hidden.

**Caveats on the read:** Universe A is itself net-negative on this window (−6.76 bps, PF 0.84)
— this compares a losing quarter to a worse one. `or_window` fires ~12× production rate (its
sanity harness has no portfolio slot/capital gating; raw sanity is its documented standard).
`below_vwap` — the only unit where B won — is also the only setup whose data coverage begins
in 2026, so the 18-month history gate is doing something structurally different there.
Three setups could NOT be faithfully replicated on the fresh pool and were EXCLUDED, not
approximated: `panic_crash_revert_long` and `up_spike_fade_short` (no offline harness; only
source is the OCI pipeline, which has not run past 2026-04-30) and `long_panic_gap_down` (its
sanity sources PDH/PDL/PDC from `consolidated_daily.feather`, which ends 2026-04-30 and
silently returns 0 trades past it; repointing would change the signal-defining inputs).
Evidence: `reports/sub9_sanity/_universe_recut_freshpool.csv`,
`tools/sub9_research/oneshot_universe_recut_freshpool.py`, ledger line 2026-07-29.
**Lifecycle:** `docs/setup_lifecycle.md` A1 (fresh-pool decisive), A5-b (absolute statistic —
these are single-leg cash mechanisms), A2 (ledger line mandatory).

---

## 1. Origin + why this is a candidate, not an adoption

Derived from the 2026-07-29 diagnostic `tools/sub9_research/diag_universe_expectancy_by_era.py`
(16 ledgers / 31,460 dev-window trades; memory `project_universe_recalibration_2026_07`):

- Ex-capitulation pooled net PF by causal ADV20 turnover, era_A → era_B: <₹1cr **2.19→1.78**,
  ₹1-5cr 1.41→1.07, **₹5-25cr 1.14→0.92, ₹25-100cr 1.09→0.87, ≥₹100cr 1.21→0.97.** Every band
  above ₹5cr/day flipped below 1.0. The book lost its middle, not its tail.
- The retirement wave and this segment shift are the same event: retired setups drew 58-100%
  of era_A net from now-dead tiers; still-in-play setups drew ~0%.
- Four independent factor clusters are net-positive in sub-₹1cr era_B (capitulation counted
  once): capitulation_long, close_dn_overnight, short_open_fade, below_vwap_long.
- The founding "<500K shares" framing is the wrong axis — a fixed SHARE cap silently
  reclassifies names as prices rise; ₹-turnover is stable.
- Two findings forbidding naive thinning: the thin-band edge lives in the 1,230 symbols
  tracked since 2023 (PF 1.36) not the 1,353 added later (1.08); and the surviving band
  decays *within* era_B (ex-cap 2.67 → 1.25).

**The diagnostic ran entirely on burned windows and is NOT validation.** It generated exactly
one hypothesis, pre-registered below and tested once on data it never saw.

## 2. Hypothesis (one sentence)

Replacing each setup's inherited share-volume universe filter with **(i) an ADV20 turnover
ceiling of ₹5cr/day** and **(ii) a minimum 18-month continuous clean-daily-history
requirement** improves absolute net expectancy of the mechanisms we already run, because the
post-2024Q4 structure pays only in the thin band and the recently-archived cohort carries a
materially weaker version of that edge.

## 3. Definitions (locked)

- **Universe A (control):** each setup's CURRENT production universe, unchanged.
- **Universe B (treatment):** A ∩ {ADV20 turnover ≤ ₹5,00,00,000/day} ∩ {symbol has ≥ 18
  months of continuous daily bars in `clean_daily_from5m.feather` ending at the signal date}.
  ADV20 causal (`rolling(20).mean().shift(1)`), turnover = close × volume.
- Both universes are evaluated on the SAME mechanisms, SAME signals, SAME exits, SAME cost
  models. The only difference is which symbols are eligible.

## 4. Population under test

Live + paper mechanisms only (a universe change must be judged on what we actually run):
`close_dn_overnight_long` (live, cell #5), `panic_crash_revert_long`, `long_panic_gap_down`,
`up_spike_fade_short`, `or_window_failure_fade_short`, `below_vwap_volume_revert_long`, and
the multiday capitulation cluster (`mtf_capitulation` / `low52` / `zscore_oversold` /
`crash2d`) which counts as **ONE** factor for pooling and for any independence claim
(2026-07-27 factor study: r 0.55-0.84, PC1 38%).

## 5. Window

**Fresh pool ONLY: signals 2026-05-01 → present.** Never touched by the diagnostic, by the
era study, or by any candidate this cycle. Development windows are NOT re-run — the
hypothesis is already fitted to them and a development comparison would be circular.

## 6. Pre-registered power gate (evaluated FIRST, before any outcome statistic)

Count fires per universe per setup on the fresh pool.
- Pooled fires under Universe B **< 50** → **POWER-BLOCKED**: report counts only, compute NO
  outcome statistics, schedule the one-shot for the month projected to reach 50 (the C-09
  pattern), and do NOT burn the ledger line as a verdict.
- ≥ 50 → proceed to §7. Report per-setup counts regardless; setups with < 15 fires are
  reported but excluded from the pooled decision statistic.

## 7. Decision rule (pre-registered, absolute statistic per A5-b)

Primary statistic: **pooled net expectancy in bps of position notional**, plus pooled net PF.
Secondary: per-setup net expectancy, A vs B.
- **ADOPT** — B's pooled net expectancy > A's AND B's pooled net PF ≥ 1.10 AND B is better
  than A in a majority of setups with n ≥ 15 (clusters counted once). Adoption means:
  implement the filter in `services/setup_universe.py` behind config, paper-confirm, then live.
- **NO CHANGE** — B within ±10% of A on pooled expectancy, or the majority test fails. The
  diagnostic's signal is recorded as era-specific and not forward-actionable.
- **REJECT** — B's pooled expectancy < A's. Record that the re-cut failed forward validation;
  do not retest without a new mechanism-level reason.

## 8. Falsifiers / honest risks

1. **Three months is thin.** The fresh pool starts 2026-05-01; a NO-CHANGE verdict on small n
   is weak evidence, not proof the hypothesis is wrong — hence the explicit re-test schedule
   rather than a permanent reject on marginal power.
2. **Two components, one test.** The turnover ceiling and the history gate are bundled. If B
   wins, a follow-up must decompose which component carries it before implementation — a
   bundled win is not a validated pair.
3. **Selection inflation.** This hypothesis was chosen from a segment sweep across 16 ledgers;
   effective M is large. A marginal win is not a strong result — the ADOPT bar is deliberately
   set at "better AND PF ≥ 1.10 AND majority", not "better".
4. **The history gate may proxy for something else** (survivorship of the archive, or simply
   age/size). If B wins, check whether the gate's effect survives controlling for turnover
   alone.

## 9. A1/A2 compliance

Fresh-pool one-shot, ONE run, no iteration, no threshold changes after seeing results. One
ledger line: stage `universe_recut_oneshot`, windows `["post_freeze:2026-05-01"]`, verdict per
§7 (or `power_blocked` per §6). This brief is committed before the evaluation script runs.
