# Walk-Forward Methodology — End-to-End Guide

This document is the runbook for validating a new setup or re-validating an existing one under the walk-forward methodology established 2026-05-19.

**Status:** Active. Replaces 3-period chronological (Discovery / OOS / Holdout) validation as of lesson #12.

## When to use

- **New setup candidate** reaches Phase 4 (cell-lock complete) → run walk-forward as Phase 5
- **Suspected dead setup** in production drawdown → run walk-forward retroactively to decide retire vs continue
- **Quarterly consistency check** of all active setups → walk-forward + circuit breaker threshold refresh

## Inputs required

1. A **trades CSV** covering full 2023-01 through 2026-03 (or longer) with columns: `signal_date`, `symbol`, `pnl_pct`. Combine Disc + OOS + HO trade ledgers via a small helper script (see `tools/methodology/_combine_trades.py` once added during Phase 2).
2. **Pre-registered mechanism docs** in `config/configuration.json` setup block:
   - `mechanism_tags`: list of tags from `assets/mechanism_tags_registry.yaml`
   - `mechanism_notes`: free-text explanation
   - Must be committed to git ≥ 1 commit BEFORE running walk-forward (engine refuses otherwise)
3. Setup name (key under `setups.*` in config).

## Running walk-forward

```bash
.venv/Scripts/python tools/methodology/run_walk_forward.py \
    --setup <setup_name> \
    --trades-csv reports/sub9_sanity/_walkfwd_combined_<setup>.csv
```

This will:

- Verify mechanism pre-registration (skip with `--skip-pre-registration-check` for dev only)
- Run 13 × 3-month windows from 2023-01-01 to 2026-03-31
- Compute per-window: `n`, `pf_real`, `pf_net` (with MIS leverage + Indian fees), bootstrap CI lower bound, pass/fail
- Classify tier (GREEN / AMBER / RED)
- Compute `cb_drawdown_threshold` (mean − 2σ of per-window NET PnL)
- Atomically write results back to `config/configuration.json` setup block

## Tier outcomes

| Tier | Pass rate | Live action |
|---|---|---|
| **GREEN** | ≥ 9 of 13 (~69%+) | Ship at full size with circuit breaker. `cb_state=enabled`, `position_size_multiplier=1.0` |
| **AMBER** | 6-8 of 13 (~46-62%) | 90-day forward-validation at 25% size. `cb_state=forward_validation`, `position_size_multiplier=0.25`. After 90 days, if live PF_net ≥ 1.0, manually promote to GREEN; else retire |
| **RED** | ≤ 5 of 13 (≤ 38%) | Retire. `cb_state=disabled`, `enabled=false`. Document in `docs/retired_setups.md` with walk-forward table |

## Per-window pass criteria (defense against noise)

A window passes if **BOTH**:

1. Point-estimate `pf_net` ≥ 1.10
2. 95% bootstrap confidence interval lower bound (2.5th percentile) > 1.0

The CI defense prevents small-n windows (e.g., n=35 in a slow month) from passing on a lucky run of wins. Real edge survives this filter; noise doesn't.

## Mechanism pre-registration

The `mechanism_tags` + `mechanism_notes` fields in the setup config block MUST be committed to git ≥ 1 commit before walk-forward runs. This is enforced by `tools/methodology/pre_registration.py` which checks `git log -S "mechanism_tags"` against HEAD.

**Why this matters:** Without pre-registration, AMBER tier becomes "I looked at which windows failed, then declared the cause." With pre-registration, you have to declare the mechanism dependency BEFORE knowing whether the corresponding regime windows actually failed. Falsifiable, not post-hoc.

Workflow:
1. Phase 4 (cell-lock) completes
2. Researcher writes `mechanism_tags` + `mechanism_notes` in config, commits as `chore(config): pre-register mechanism for <setup>`
3. Make at least one unrelated commit (or wait for one to land)
4. Run walk-forward — engine verifies git timestamps, then runs

## Active setup consistency check (Option C)

Active setups that come in as AMBER or RED are **NOT proactively retired**. Per Option C:

- They keep running at current size
- Circuit breaker monitoring is enabled (`cb_drawdown_threshold` set from walk-forward)
- They go on the watch list at `docs/active_setups_review.md`
- Monthly review checks for live degradation

Live drawdown is the signal — if degradation is real, the circuit breaker pulls them off the field within 60 days.

## Circuit breaker

`jobs/check_circuit_breakers.py` runs daily at EOD (after 16:00 IST). For each enabled setup with `cb_drawdown_threshold` set:

1. Reads trailing 60-day NET PnL from `trade_report.csv`
2. If trade count < `cb_min_trades_for_signal` (default 30): no action (insufficient data)
3. If PnL < `cb_drawdown_threshold`: atomic config update sets `cb_state=disabled`, `cb_disabled_at=today`, `cb_disabled_reason=60d_pnl_below_threshold`
4. Manual re-enable required (researcher inspects, decides retire vs un-disable)

Atomic write: temp file + `os.replace()` so the live screener never reads a partial config. Live screener picks up config changes on every tick.

## Forward-validation for AMBER tier

For setups un-retired or shipped as AMBER:

- `cb_state=forward_validation`
- `position_size_multiplier=0.25` (25% of full size)
- 90-day live evaluation period
- Daily check evaluates against tighter threshold (e.g., -1σ instead of -2σ) for early intervention

After 90 days:
- If forward PF_net ≥ 1.0: manually promote to `cb_state=enabled`, `position_size_multiplier=1.0`
- If forward PF_net < 1.0: manually demote to `cb_state=disabled`, retire

## What's NOT in scope

- Hidden Markov Model regime classifier (drawdown is the signal)
- Combinatorial Purged Cross-Validation (overkill for rule-based setups)
- Per-trade slippage modeling beyond current ~0.5% round-trip assumption
- Monte Carlo permutation test (optional belt-and-suspenders for borderline AMBER only)
- Full position-size threading from `position_size_multiplier` to live order sizing (Phase 1 of toolkit added the FIRE gate only; size threading is a follow-up task)

## References

- Spec: `docs/superpowers/specs/2026-05-19-walk-forward-methodology-design.md`
- Plan: `docs/superpowers/plans/2026-05-19-walk-forward-methodology.md`
- Research report: `reports/sub9_sanity/_backtest_methodology_research.md`
- Lesson: `tasks/lessons.md` #12
- Mechanism tag registry: `assets/mechanism_tags_registry.yaml`

## Toolkit components

| File | Purpose |
|---|---|
| `tools/methodology/bootstrap_ci.py` | Per-window bootstrap CI for Profit Factor |
| `tools/methodology/setup_metadata.py` | Atomic read/write of setup blocks in config |
| `tools/methodology/walk_forward.py` | Windowing engine + tier classifier |
| `tools/methodology/pre_registration.py` | Git-timestamp mechanism pre-registration check |
| `tools/methodology/run_walk_forward.py` | CLI driver |
| `jobs/check_circuit_breakers.py` | Daily EOD circuit breaker check |
| `services/plan_orchestrator.py` | Live `_setup_should_fire` gate honoring `cb_state` |
