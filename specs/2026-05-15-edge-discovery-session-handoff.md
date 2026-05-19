# Edge-First Discovery Framework — Session Handoff (2026-05-15)

## Status: T14 Checkpoint paused — Phase 1 + most of Phase 2 complete

**Plan:** `specs/2026-05-15-edge-first-discovery-framework-plan.md` (22 tasks)
**Spec:** `specs/2026-05-15-edge-first-discovery-framework-design.md`
**Branch:** `research/post-sebi-edge-setups`

## What's done (17 commits, T0-T13)

Phase 1 framework core — all modules built, 37 tests passing:
- `tools/edge_discovery/types.py` (Event, ConditionalOutcomeTable with slice_by/joint_slice/top_edge_regions placeholder)
- `tools/edge_discovery/outcomes/{base,returns,costs}.py`
- `tools/edge_discovery/features/{base,symbol_features}.py` — Tier-A: 15 features
- `tools/edge_discovery/explorer.py`
- `tools/edge_discovery/universe.py`, `data_loader.py`
- `tools/edge_discovery/{rule_orthogonality,ship_gate,decay_monitor}.py`
- `tools/edge_discovery/validation/{walk_forward,parity_gate}.py`
- `tools/edge_discovery/targets/target_parity_{gap_fade,circuit_t1,delivery_pct}.py`
- Config block in `config/pipelines/base_config.json` (NOT `configuration.json` — plan was wrong about this; fixed)

Phase 2 parity gate finding (T12-T13):
- **gap_fade_short**: framework's parity script runs. small_cap filter applied. Framework reads parquet exactly: N=797, PF=1.13, WR=64%. Live `_live_status` text says PF=1.36 / WR=70% — does NOT match any stored parquet.
- **circuit_t1_fade_short**: no parquet in `reports/sub8_oos_holdout*/`, no `_live_status` in config — can't parity-validate.
- **delivery_pct_anomaly_short**: same as circuit_t1.

## Key finding: baseline-data drift

The documented `_live_status` for gap_fade_short (PF=1.36, WR=70%, N=797 in 117-session Holdout) does NOT match the most-recent stored parquet (`sub8_oos_holdout_clean/gap_fade_short.parquet` filtered to small_cap: PF=1.13, WR=64%, N=797). The Holdout N matches exactly so the cell-restriction is correct; the PF/WR gap is data-state.

Hypothesis: The `_live_status` was authored after a 2026-05-12 sweep (per config comment: "Disc PF 1.47→2.36 / OOS 1.06→1.71 / Hold 1.41→1.69") that lifted PF. The post-sweep run was not persisted as an updated parquet, so the parquet still reflects pre-sweep results.

## T14 user decision (in progress when paused)

User chose path 1+2 (fix gap_fade filter + generate missing parquets). Path 1 is DONE (commit `1661f97`). Path 2 + baseline reconciliation pending.

## Remaining tasks (T14 onward)

**T14 — CHECKPOINT (user input needed first):** Decide path to align gap_fade baseline:
- (A) Re-run gap_fade_short live config end-to-end on Holdout to produce a fresh parquet that reproduces `_live_status` PF=1.36 (significant work — needs main.py invocation against historical 5m feathers)
- (B) Update `_live_status` baseline in config to reflect actual stored parquet (PF=1.13, WR=64%) and accept that as ground truth
- (C) Soften framework parity tolerance to ±20% PF / ±10pp WR (acknowledge baseline drift as systemic)

**T15-T17 (Phase 3 Tier-B features):** Market features (FII/DII/USD-INR/crude/VIX/AD), event calendar (expiry/RBI/rebalance), edge region detection (1D+2D+3D scan)

**T18-T19 (research targets):** LONG-side panic-gap-down catch, ensemble feature mining on live setups (OPUS-class tasks)

**T20-T22 (decisions):** Report generator, decay_monitor_runner, findings doc + tag

## Fresh-session priming

To resume:
1. Read this handoff
2. Read `specs/2026-05-15-edge-first-discovery-framework-plan.md` (Tasks T14-T22)
3. Get T14 decision from user (3 options above)
4. Continue via subagent-driven-development from T15

## Generating circuit_t1 + delivery_pct parquets (for path 2)

Will likely require a small driver script that:
- Loads the setup config from `config/configuration.json`
- Runs through the existing engine (`tools/engine.py` or equivalent) on Holdout 5m feathers (Oct 2025 – Apr 2026)
- Captures trades in the same schema as `reports/sub8_oos_holdout/gap_fade_short.parquet` (cols: session_date, setup_type, realized_pnl, fee, net_pnl, qty, entry_price, e1_price, side, decision_ts, symbol, regime, cap_segment, rank_score)
- Writes to `reports/sub8_oos_holdout/{circuit_t1_fade_short,delivery_pct_anomaly_short}.parquet`

This is essentially "re-run Phase 7 OOS for the 2 missing setups". May be its own sub-project.

## Working tree state (clean)

- All edits committed
- Probe files in `tools/edge_discovery/` (e.g., `probe_*.py`, `volume_probe*.py`) remain UNTRACKED (intentionally — leftover from prior research, unrelated)
- Old Gauntlet v2 code moved to `tools/edge_discovery_legacy_gauntlet/` for reference
