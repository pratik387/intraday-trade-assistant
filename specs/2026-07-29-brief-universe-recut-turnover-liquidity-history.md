# Brief: universe re-cut — ₹-turnover cap + continuous-liquidity-history gate

**Date:** 2026-07-29
**Type:** UNIVERSE re-specification (not a new setup) — applies to mechanisms already owned
**Status:** PRE-REGISTERED. Committed BEFORE any fresh-pool statistic is computed; the freeze
is verifiable by commit order.
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
