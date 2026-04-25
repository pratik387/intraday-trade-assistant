# Sub-project #5 (Gauntlet v2) — Post-Mortem

**Status:** Closed as FAILED per master plan discipline (Phase 3 + Phase 4 OOS both failed).
**Date:** 2026-04-25
**Authors:** post-mortem from gauntlet v2 execution + Indian market microstructure research.

## Executive summary

Gauntlet v2 found that an Optuna-tuned gate config produced Discovery (2023-2024) Sharpe 0.99 / PF 1.72 but collapsed OOS to Sharpe 0.31 / PF 1.37 (gross). Spec discipline says fail-and-revert. The post-mortem investigation revealed nine distinct failure modes, of which the most severe (Failure G) is **structural**: the setup library is cargo-culted from US/forex SMC literature and doesn't align with Indian intraday market structure. No amount of gate-config tuning can fix a mis-targeted strategy library.

The infrastructure built (parity simulator, wide_open OCI capture, gauntlet v2 pipeline) is sound and reusable. The setup library is the layer that requires redesign.

## What was attempted

1. Captured wide_open OCI runs for 2023-01-01 to 2026-03-31 (3.25 years, 745 sessions).
2. Built `tools/gauntlet_v2/` infrastructure: ETL, single-config evaluator, Optuna search, OOS validator.
3. Ran 500-trial Optuna search on Discovery 2023-2024, optimizing Sharpe with guards (PF ≥ 1.2, losing_days ≤ 40%, n_filled ≥ 500).
4. Tested winning config against Validation (Jan-Sep 2025) and Holdout (Oct 2025-Mar 2026) per Phase 3/4 spec.

## Outcome (per spec criteria)

```
                       Discovery     Validation    Holdout
Optuna best_config:
  Sharpe (gross)       0.990         0.514         0.315
  PF (gross)           1.722         1.429         1.373

Phase 3 thresholds:    Sharpe ≥ 0.7, PF ≥ 1.2     →  FAIL (Sharpe)
Phase 4 thresholds:    Sharpe ≥ 0.5, PF ≥ 1.0     →  FAIL (Sharpe)
                       losing_days ≤ 40%
```

Master plan prescribes: "Holdout fail → revert or escalate to Option C."

## Failure modes identified

| ID | Failure | Evidence | Severity |
|---|---|---|---|
| A | Sharpe ≥ 0.7 threshold set without reality-checking baseline OOS | Baseline (no Optuna) OOS Sharpe = 0.28 (val) / 0.56 (hol). Spec target unreachable for this strategy mix. | High |
| B | Optuna over-fit Discovery noise | Sharpe 0.99 → 0.31 across periods; PF degraded monotonically. Standard overfit signature despite parameter stability across top trials. | High |
| C | (Initially suspected) Sub-project #1 rule survivors include negative-edge setups | NOT a real failure on inspection: rule survivors are 100% short setups. Production gate correctly excludes longs. | Discarded |
| D | All analysis conducted on GROSS PnL | trade_report.csv has no fee column. Indian intraday round-trip ~0.05-0.06% turnover. | High |
| E | Gross PF decay year-over-year (1.34 → 1.22 → 1.08 shorts-only wide_open) | Suggests setup edges are eroding, not just regime noise. | Medium |
| F | No phase of gauntlet considered fees in objective or thresholds | Sub-project #1 used gross. Optuna optimized gross. Phase 3/4 thresholds defined gross. | High |
| G | **Setup library is cargo-culted from US/forex SMC literature** | Detectors include ICT order blocks, fair value gaps, premium/discount zones — all originating in 24/5 forex with no MIS, circuits, or asymmetric short rules. Long variants of these patterns systematically lose money in Indian intraday by structural design. | **Critical (root cause)** |
| H | No fee awareness in rule discovery (sub-project #1) | Rule survivors include shorts with NET PF as low as 0.75 (vwap_lose_short). | Medium |
| I | No exploitation of Indian-specific edges | System does not weight time-of-day (3:00-3:20 PM MIS unwind window), opening 9:15-9:30 gap fade, CPR/pivot mean reversion, FII/DII flow signals. | Medium-High |

## Indian market structure context (research synthesis)

Sourced from Zerodha Varsity, SEBI publications, QuantInsti EPAT, Stratzy, ScienceDirect academic papers, and broker documentation:

1. **MIS auto-square at 3:20 PM** creates structural net-sell pressure in the last 60-90 minutes. SEBI rules + retail behavior make intraday flow overwhelmingly net-long, so the unwind is asymmetrically a net sell. Pros exploit this with late-day shorts.
2. **SEBI study (FY23):** 70% of cash intraday traders lose money; 93% of F&O traders. The losing flow is overwhelmingly long.
3. **ICT/SMC patterns originate in forex** (24/5 continuous, no auction open, no circuits, no MIS unwind). Indian equity has session opens, T+1 settlement, gap risk, circuit halts, and asymmetric short rules — the structural assumptions break.
4. **Realistic Sharpe benchmarks:** retail >1, hedge funds >2, top quant shops >3 net. EPAT Indian equity gap-trading projects: ~1.5-2 net. Our 0.34 net is below the floor for "edge worth deploying."

## Production gate net economics (Discovery only — no OOS leakage)

```
Production gate (current configuration.json with gates active):
  Discovery 2023-2024:
    Trades: 11,814 over 487 sessions = 24.26/day
    Gross: PF 1.54, Sharpe 0.62, daily ₹6,451
    Fees:  ₹1.41M (45% of gross)
    Net:   PF 1.27, Sharpe 0.34, daily ₹3,557

  Per-setup NET (Discovery only):
    order_block_short          n=71    net_pf=1.77   net=+₹23K
    premium_zone_short         n=6741  net_pf=1.27   net=+₹915K   ← biggest contributor
    range_bounce_short         n=2908  net_pf=1.28   net=+₹491K
    resistance_bounce_short    n=1987  net_pf=1.27   net=+₹319K
    vwap_lose_short            n=107   net_pf=0.75   net=-₹16K    ← drop candidate
```

Production gate has positive net edge on Discovery but at a Sharpe (0.34) materially below professional benchmarks (1.0-2.0).

## What we did NOT do (and won't, per discipline)

- Did not re-test alternative configs on Validation/Holdout. Both OOS sets are spent.
- Did not deploy the Optuna winner. Master plan prescribes "fail → revert."
- Did not change Phase 3/4 thresholds post-hoc to convert FAIL into PASS.
- Did not improvise a "deploy shorts-only" change without a clean methodology basis.

## Lessons captured for future work

1. **Net PnL must be the unit of optimization from day one.** Build fee calculation into trade_report.csv writers and into all gauntlet metrics. Gross-only optimization wastes Discovery cycles on fee-eaten edges.
2. **Threshold setting requires baseline reality check.** Before declaring Sharpe ≥ 0.7 a target, measure baseline OOS Sharpe for the existing system. Targets disconnected from baselines invite false discipline.
3. **Setup library design is upstream of gate optimization.** Tuning gates over a structurally-mismatched setup library will overfit noise. Setup design must precede / accompany gate tuning.
4. **Cargo-culted patterns need market-specific validation.** ICT/SMC concepts may or may not hold in Indian equity; assuming they do is a hidden assumption worth surfacing.
5. **Parameter stability across top Optuna trials is necessary but not sufficient evidence against overfit.** Our top 10 had tight clustering; OOS still collapsed. Stability is a weak prior; OOS is the only test.
6. **Master plan's "Option C" (exit simulator + rule re-discovery) does not address the root cause** if the cause is setup mismatch. Polishing the catalog ≠ fixing the catalog.

## Recommended next action

Open a new sub-project scoped to **redesign the setup library for Indian intraday market structure** rather than continue gauntlet-style optimization over the existing library. See companion document: `specs/2026-04-25-sub-project-7-indian-native-setups-scope.md`.

## What stays from sub-project #5

The infrastructure is sound and reusable for any future setup library:
- `tools/shadow/parity_simulator.py` (bit-exact gate replay)
- `tools/gauntlet_v2/` (build_pnl_index, trial, search, validate)
- `wide_open_mode` cascade (level_pipeline, trade_decision_gate, LiveGateChain bypasses)
- Wide_open OCI capture format (gate_input.jsonl + trade_report.csv per session)
- LiveGateChain composed gate framework (RuleFilter → CrossSectional → Conviction → Dedup)

What does not get reused:
- The 74 rule survivors (derived from SMC-pattern Discovery, will be re-derived)
- best_config.json from Phase 2 (failed OOS, do not deploy)
- Phase 3/4 results on validation/holdout (informational only; OOS data sets are spent)
