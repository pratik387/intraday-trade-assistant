# Gauntlet v2 — Operational Runbook

**Sub-project #5.** Code is complete as of this file's commit. Runs against OCI
data produced by tasks #82 (Docker rebuild) + #83 (OCI wide-open submit).

## Prerequisites

- Task #82 complete: OCI Docker image contains Sub-project #4 + #5 changes
- Task #83 (or equivalent) has submitted a wide-open 2023-01-01 → 2026-03-31 job
- Results pulled locally to `cloud_results/gauntlet_v2_discovery/` with per-session subfolders

## Step 1 — Prepare the OCI override (before OCI submit)

```bash
.venv/Scripts/python tools/gauntlet_v2/oci_runner.py \
    --output cloud_results/gauntlet_v2_discovery/oci_overrides.json
```

Hand `oci_overrides.json` to the OCI submit infra.

## Step 2 — Build PnL index (after OCI completes)

```bash
.venv/Scripts/python tools/gauntlet_v2/build_pnl_index.py \
    --oci-dir cloud_results/gauntlet_v2_discovery/ \
    --output cloud_results/gauntlet_v2_discovery/pnl_by_admit.parquet
```

Expect ~100 MB parquet for 3.25 years × ~500 trades/day.

## Step 3 — Phase 1 sanity (production config baseline)

```bash
.venv/Scripts/python tools/gauntlet_v2/trial.py \
    --base-cfg config/configuration.json \
    --cfg-overrides '{}' \
    --gate-input-dir cloud_results/gauntlet_v2_discovery/ \
    --pnl-index cloud_results/gauntlet_v2_discovery/pnl_by_admit.parquet \
    > docs/gauntlet_v2/phase1_sanity_metrics.json
```

Compare to v1 gauntlet report. Expect aggregate metrics within ~5%.

## Step 4 — Phase 2 Bayesian search

```bash
.venv/Scripts/python tools/gauntlet_v2/search.py \
    --base-cfg config/configuration.json \
    --gate-input-dir cloud_results/gauntlet_v2_discovery/ \
    --pnl-index cloud_results/gauntlet_v2_discovery/pnl_by_admit.parquet \
    --output-dir docs/gauntlet_v2/$(date +%Y-%m-%d)-discovery/ \
    --n-trials 500 --n-jobs 20 --min-n-trades 500
```

Produces `best_config.json`, `study.db`, `trials.csv`, `06-search-report.md`.

## Step 5 — Phase 3 Validation (one shot)

Copy `best_config.json` to a frozen location. Then:

```bash
.venv/Scripts/python tools/gauntlet_v2/validate.py \
    --base-cfg config/configuration.json \
    --config docs/gauntlet_v2/frozen_config.json \
    --gate-input-dir cloud_results/gauntlet_v2_discovery/ \
    --pnl-index cloud_results/gauntlet_v2_discovery/pnl_by_admit.parquet \
    --period validation \
    --output-dir docs/gauntlet_v2/$(date +%Y-%m-%d)-discovery/
```

Exit 0 = pass. Exit 2 = fail. Pass criteria: PF ≥ 1.2, Sharpe ≥ 0.7.

## Step 6 — Phase 4 Holdout (one shot, binary)

Same as Step 5 but `--period holdout`. Pass criteria: PF ≥ 1.0, Sharpe ≥ 0.5, losing_days ≤ 40%.

- Holdout pass → handoff to Sub-project #6 (Deployment).
- Holdout fail → revert to v1 survivors or escalate to Option C (exit simulator rebuild).

## One-shot discipline

`validate.py` refuses to re-run against an already-tested period unless `--force` is passed. This is intentional: the master plan requires Validation and Holdout each to be a single-shot test. Any temptation to "tune until pass" defeats the OOS discipline.

## Output artifacts (typical layout)

```
docs/gauntlet_v2/
├── README.md                                  ← this file
├── phase1_sanity_metrics.json                 ← Step 3
└── 2026-04-XX-discovery/                      ← Step 4-6 outputs
    ├── study.db
    ├── best_config.json
    ├── trials.csv
    ├── 06-search-report.md
    ├── 07-validation-result.json
    ├── 08-validation-report.md
    ├── 09-holdout-result.json
    └── 10-holdout-report.md
```
