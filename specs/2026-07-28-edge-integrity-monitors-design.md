# Design: Edge-Integrity Monitors (mechanism preconditions + factor tripwire)

**Date:** 2026-07-28
**Status:** DRAFT — awaiting user review before commit/implementation.
**Motivation (data-backed, 2026-07-28 analyses):**
- The four SEBI-Oct-2025-rule-dependent shorts bled **≈ −₹206k combined in the 7 months
  between the public circular (2025-10-01) and the May-2026 retirement wave** (OCI-canonical:
  circuit_release −64k, mis_unwind −132k, delivery_pct −6k, round_number −4k). The cause was
  observable on day one; detection ran through PnL instead.
- The July-2026 factor study: A2/C1/C4/C6 + panic_crash are one factor (r 0.55-0.84, PC1
  38% of book variance; ≥70% of setups lose together on 20.3% of days vs 12.8% independent).
  No per-setup tripwire can see a factor-wide bad month.
- `pead_reaction_drift` kill: the suspected killer was a *data-source death*
  (announcements_fr, Mar-2025) — a third monitorable precondition class.

## 1. Scope

One nightly job: `jobs/check_edge_integrity.py` (cron, after EOD data refresh; same host
as existing `jobs/check_circuit_breakers.py`), three monitor classes, all config-driven
(NO hardcoded thresholds — every parameter from `config/configuration.json` under a new
`edge_integrity` block).

### Monitor 1 — Rule-change watch (mechanism preconditions, regulatory class)
- Source: `data/sebi_calendar/rule_changes.csv` (existing) + a small manual-entry workflow
  (a new rule row is added when a circular lands — human-in-the-loop by design; scraping
  SEBI circulars automatically is out of scope v1).
- Config: each setup block gains `preconditions: [{type: "rule", watch: "<rule_key>"}]`.
  Setups already carrying `regulatory_sensitivity: rule_dependent` MUST declare at least one.
- Behavior: when a watched rule_key gains a row with effective_date >= today−N (config),
  the job flips the setup to `cb_state: "paused_precondition"` (new state, same machinery
  as circuit breaker) and logs loudly. Un-pause is manual after researcher review.

### Monitor 2 — Data-source health (data class)
- Config: `preconditions: [{type: "data_source", path: "...", max_staleness_days: X,
  min_rows_per_week: Y}]` per setup that depends on a scraped feed (earnings calendar,
  delivery %, bulk deals, ASM/GSM).
- Behavior: staleness or volume collapse (announcements_fr pattern — the feed died but the
  file kept existing) → same pause + alert. This is cheap: a stat per parquet per night.

### Monitor 3 — Factor-cluster tripwire (portfolio class)
- Config: `edge_integrity.factor_clusters: {capitulation_reversion: {members: [...5 setups],
  lookback_days, drawdown_threshold_sigma}}` — thresholds derived from the OCI canonical
  combined-daily-PnL distribution at implementation time (documented in config comments,
  not hardcoded in code).
- Behavior: combined daily PnL of the cluster (paper + live legs both) breaching the
  drawdown threshold → pause NEW entries for ALL cluster members (existing positions run
  their exits), alert. Complements — never replaces — per-setup tripwires.

## 2. Non-goals (v1)
- No automatic un-pause. No SEBI scraping. No intraday monitors (nightly only). No changes
  to exit handling of open positions. No ML/regime prediction (monster-conditioning lesson).

## 3. Integration points
- `services/state/` for pause state (same pattern as decay_tripwire_*.json).
- The live intraday daemon + overnight/multiday crons all already consult `enabled`/cb
  state via config/state readers — the new pause state must be honored by BOTH the intraday
  path and the multiday/overnight crons (verify each entry point; list them in the plan).
- Alerting: same channel as existing failsafe alerts (overnight verify-exit precedent).

## 4. Rollout
1. Implement + unit tests (pause-state honored by each entry point; staleness math; cluster
   PnL aggregation matches the factor-study methodology).
2. Dry-run mode for 1 week (log-only, no pausing) to calibrate false-positive rate.
3. Enable pausing for paper setups first, then live after one clean week.

## 5. Config keys added (all new, no defaults in code)
`edge_integrity.enabled`, `edge_integrity.rule_watch_lookback_days`,
`edge_integrity.factor_clusters.*`, per-setup `preconditions[]`. Missing key on a
rule_dependent setup = startup validation error (fail fast, per CLAUDE.md).
