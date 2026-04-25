# Sub-project #7 — Indian-Native Setup Library (DESIGN)

**Status:** Phase 0 brainstorming complete. Approved for implementation planning.
**Date:** 2026-04-25
**Predecessor:** Sub-project #5 (Gauntlet v2) — failed Phase 3/4 OOS. Post-mortem identified the existing setup library as cargo-culted from US/forex SMC literature and structurally mismatched to Indian intraday equity.
**Scope precursor:** `specs/2026-04-25-sub-project-7-indian-native-setups-scope.md` (proposal that triggered this design).

## 1. Goal

Build a setup library specifically designed around Indian intraday equity microstructure, validated per-setup against fee-aware bars BEFORE composition, then composed mechanically (no joint Optuna optimization). Target net Sharpe ≥ 1.0, deployable floor net Sharpe ≥ 0.5.

## 2. Decisions made in Phase 0 (the brainstorming output)

| # | Question | Decision |
|---|---|---|
| Q1 | Scope: stay intraday vs pivot? | Stay intraday equity. F&O dropped (no skill). Swing dropped (user preference). |
| Q2 | Single-thesis vs portfolio? | Portfolio of independently validated setups. Avoid sub-project #5's mistake of joint optimization across an unvalidated library. |
| Q3 | Success criteria? | Aspirational targets (Sharpe ≥ 1.0, PF ≥ 1.5, 30%/yr, DD ≤ 15%). Minimum acceptable (Sharpe ≥ 0.5, PF ≥ 1.15, +5% over Nifty, DD ≤ 25%). Kill switch (Sharpe < 0.3 OOS, PF < 1.0, can't beat baseline). |
| Q4 | Initial detector set? | Three detectors covering three time-of-day windows: MIS_unwind_short (15:00-15:20), Gap_fade_short (09:15-09:30), CPR_mean_revert (11:30-13:30). |
| Q5 | Validation methodology? | Bundled OCI run with new detectors only + wide_open_mode + cycle_limit raised. Local main.py --dry-run for development iteration. Bundled OCI for milestone validation. |
| Q6 | OOS data reservation? | Sub-project #5's val/holdout data is NOT burned for sub-project #7 — different setups, fresh data semantically. Same period split: Discovery 2023-2024 / Validation 2025-Sep / Holdout Oct 2025-Mar 2026. |
| Q7 | Kill conditions? | Hard kill: Phase 1 (0 of 3 detectors pass) OR Phase 2 (portfolio fails minimum after one iteration) OR Phase 4 (val or holdout fails minimum). Soft warning: only 1 of 3 passes Phase 1 → add Phase 2 candidates first. |

## 3. The three initial detectors

### 3.1 MIS_unwind_short (15:00-15:20 IST)

**Thesis:** SEBI requires MIS positions square off by 3:20 PM. Retail intraday flow is structurally net-long. The forced unwind in the last 60-90 minutes creates asymmetric net-sell pressure. Pros short into this.

**Detector logic (rough):**
- Active window: 14:55-15:15 IST only
- Scan: stocks with fresh intraday highs in last 30 min, above VWAP, with negative momentum_3bar (signs of weakening)
- Cap segment: prefer small/mid cap (more retail-MIS-driven)
- Signal strength: function of (distance_above_VWAP, momentum_decay_rate, rvol)

**Entry/exit:** Short at current bar close. Stop = recent intraday high or 0.8 ATR above. Target = VWAP or PDC. Time-stop: square off by 15:18.

### 3.2 Gap_fade_short (09:15-09:30 IST)

**Thesis:** FII gap-up + retail chase → exhaustion at first 5m bar. Mid/small caps over-extend, then mean-revert. Documented in Zerodha Varsity and academic Indian intraday literature.

**Detector logic (rough):**
- Active window: 09:15-09:30 (first three 5m bars)
- Scan: stocks with opening 5m gap > X% above PDC (X TBD, probably 1.5-3%)
- Cap segment: small/mid cap only (retail-dominated)
- Exhaustion pattern: long upper wick, body_size_pct < 30%, declining volume after first bar

**Entry/exit:** Short at exhaustion bar close. Stop = gap high. Target = PDC or opening price. Time-stop: square off by 10:00.

### 3.3 CPR_mean_revert (11:30-13:30 IST)

**Thesis:** Lunch lull (11:30-13:30) has low volume → range-trading dominates → stocks mean-revert to CPR (Central Pivot Range). Varsity-documented.

**Detector logic (rough):**
- Active window: 11:30-13:30 IST
- Scan: stocks at distance > 1 ATR from CPR midpoint
- Volume filter: current bar volume below 30-day intraday average for that bar
- Reversion candle: hammer/doji at extreme

**Entry/exit:** Both long and short directions. Trade direction = toward CPR. Stop = bar high/low. Target = CPR midpoint. Time-stop: square off by 13:45.

## 4. Architecture

### 4.1 Reused from sub-projects #4 + #5 (no rebuild needed)

```
structures/main_detector.py        ← orchestrates detectors per bar (extend with feature flag)
pipelines/level_pipeline.py        ← derives entry zone/stop/targets from detector signal
services/gates/trade_decision_gate ← gate filtering (rule_filter, etc.)
services/gate_chain/                ← cross_sectional, conviction, dedup gates
services/screener_live.py           ← ranking, slot competition
services/execution/                 ← entry zone watch, exit_executor
tools/shadow/parity_simulator.py    ← bit-exact gate replay
tools/gauntlet_v2/                  ← build_pnl_index, trial.py, validate.py
oci/tools/submit_oci_backtest.py   ← OCI runner
```

### 4.2 New code (sub-project #7 deliverables)

```
structures/mis_unwind_short_structure.py       ← detector 1 (~300 LOC)
structures/gap_fade_short_structure.py         ← detector 2 (~300 LOC)
structures/cpr_mean_revert_structure.py        ← detector 3 (~400 LOC)
config/configuration.json                       ← add `enabled_detectors` config key
structures/main_detector.py                     ← respect enabled_detectors filter (~20 LOC)
tools/sub7_validation/                          ← per-setup validation harness (~200 LOC)
  ├─ build_per_setup_pnl.py                    ← splits trade_report by setup_type, applies fees
  ├─ per_setup_report.py                       ← generates per-setup metrics + breakdowns
  └─ portfolio_composer.py                     ← Phase 2 mechanical composition
config/sub7_phase1_oci_overrides.json          ← OCI run config for Phase 1 capture
```

### 4.3 Engine extensions for non-pattern setups

MIS_unwind_short and Gap_fade_short fit existing pattern paradigm (have entry zone + stop + target). CPR_mean_revert fits cleanly too (CPR levels are "pattern" structure).

**No engine extension needed for the initial 3.** All fit existing detector → planner → executor pipeline.

If we add Phase 2 candidates later (FII/DII flow signal as direction prior, India VIX as position-sizing overlay), THAT requires engine extension — but those are deferred.

## 5. Phase structure

```
PHASE 0 — Brainstorming (DONE).
PHASE 1 — Per-setup independent validation (3 detectors).
PHASE 2 — Portfolio composition test (Discovery only).
PHASE 3 — Phase 2 cleanup + soft target hit.
PHASE 4 — One-shot Validation OOS test (Jan-Sep 2025).
PHASE 5 — One-shot Holdout OOS test (Oct 2025-Mar 2026).
```

### Phase 1 — Per-setup validation

**Steps:**
1. Build all 3 detectors locally with feature flag.
2. Iterate locally on subset (5-10 random Discovery sessions) using main.py --dry-run.
3. When all 3 detectors look reasonable, run ONE bundled OCI run:
   ```json
   {
     "enabled_detectors": ["mis_unwind_short", "gap_fade_short", "cpr_mean_revert"],
     "wide_open_mode": true,
     "gate_input_logging": {"enabled": true},
     "max_trades_per_cycle": 10000
   }
   ```
   Period: 2023-01-01 → 2026-03-31 (full 3.25 years; we capture validation+holdout in same run for code-version consistency).
4. Local analysis: split trade_report by setup_type, apply fees, compute per-setup metrics.
5. Pass criterion per detector:
   ```
   NET PF        ≥ 1.10  on Discovery (2023-2024 portion only)
   n_trades      ≥ 500
   Net Sharpe    > 0
   ```
6. Outcome:
   - 2-3 pass → proceed to Phase 2.
   - 1 pass → soft warning, add Phase 2 candidates (FII/DII, VIX) before composition.
   - 0 pass → KILL sub-project #7.

### Phase 2 — Portfolio composition

**Steps:**
1. Take Phase 1 winners.
2. Use existing LiveGateChain + screener_live for composition (slot competition, ranking) — NO joint Optuna.
3. Run mechanical compositions: equal-weight, then risk-parity (size by inverse-of-vol per setup).
4. Test composed portfolio on Discovery 2023-2024 only.
5. Pass criterion (composed portfolio):
   ```
   NET PF        ≥ 1.25
   Net Sharpe    ≥ 0.6
   Max DD        ≤ 20%
   Beats Nifty50 in 2024 AND 2023 (each year individually, not just average)
   ```
6. Outcome:
   - Pass → proceed to Phase 4.
   - Fail → ONE composition iteration variant (try different sizing). If still fail → KILL.

### Phase 4 — Validation OOS (one-shot)

**Steps:**
1. Apply final composed portfolio to Validation period (2025-01-01 → 2025-09-30) gate_input + trade_report from Phase 1 OCI capture.
2. Use existing tools/gauntlet_v2/validate.py with --period validation.
3. Pass criterion:
   ```
   NET PF        ≥ 1.15
   Net Sharpe    ≥ 0.5
   Max DD        ≤ 25%
   Annual return ≥ Nifty + 5%
   ```
4. Outcome:
   - Pass → Phase 5.
   - Fail → KILL sub-project #7. Document why. Consider Option D (alternative timeframe).

### Phase 5 — Holdout OOS (one-shot final)

Same as Phase 4 but on Oct 2025-Mar 2026 period. Same pass criterion.

If Phase 5 passes → handoff to deployment planning (sub-project #8 if pursued; user has indicated preference NOT to deploy without further skin-in-the-game evaluation).

If Phase 5 fails → close sub-project #7 as failed. Investigate whether the Discovery edge was overfit despite per-setup validation discipline.

## 6. Cost & timeline estimate

```
Phase 0 (brainstorming):           DONE
Phase 1 (build + per-setup OCI):   3-4 weeks active dev + 1 OCI run ($50-200) + 1 day analysis
Phase 2 (composition):             1 week local analysis + possibly 1 OCI variant ($50-200)
Phase 3 (cleanup, if needed):      0-1 week
Phase 4 (validation OOS):          1 day (uses Phase 1 OCI capture)
Phase 5 (holdout OOS):             1 day (uses Phase 1 OCI capture)

Total active engineering: ~5-7 weeks
Total OCI cost:           $50-400 (1-2 bundled runs)
Total elapsed time:       ~6-8 weeks
```

## 7. Risks

| Risk | Mitigation |
|---|---|
| All 3 detectors fail Phase 1 → sub-project killed | Acceptable failure mode. Saves us from polishing turds. Failure data informs next attempt. |
| Detector logic bugs hidden in OCI run | Local subset iteration first (5-10 sessions). Catch obvious bugs cheaply. |
| Composition adds unforeseen interactions | Phase 2 explicitly tests this; uses existing engine that we trust. |
| OOS still fails despite per-setup validation | Possible — Indian markets may have structural features we still don't understand. Failure documented; sub-project closes. |
| Fee model inaccurate | Use exact Indian intraday schedule from `services/logging/trading_logger.py`. No approximations. |
| Sub-project #5's val/holdout not actually fresh | Acknowledged in Q6. We're betting that conceptually-different setups make the data semantically fresh. If user disagrees, defer Phase 4-5 until live data fresh in 3-4 months. |

## 8. Out of scope

- F&O / options strategies
- Multi-asset signals beyond intraday equity
- Deployment / paper trading (separate decision after Phase 5 passes)
- Real-time data feed integration (FII/DII, India VIX) — deferred to Phase 2 candidates
- Custom slippage modelling (using simple fee model)
- ICT/SMC pattern preservation (sub-project #5 SMC detectors are explicitly NOT used here)

## 9. Success definition (final)

Sub-project #7 SUCCEEDS if:
- ≥ 2 of 3 detectors pass Phase 1 NET PF ≥ 1.10
- Composed portfolio passes Phase 2 minimum (Sharpe ≥ 0.6, PF ≥ 1.25)
- Validation passes minimum (Sharpe ≥ 0.5, PF ≥ 1.15)
- Holdout passes minimum (Sharpe ≥ 0.5, PF ≥ 1.15)

Sub-project #7 FAILS but provides learning if:
- Any kill-switch triggers (Phase 1, 2, or OOS minimum failure)
- Failure documented; informs next iteration

Sub-project #7 SUCCEEDS WITH STRETCH if:
- Aspirational targets met (Sharpe ≥ 1.0, PF ≥ 1.5, ≥ 30% annual, DD ≤ 15%)
- Strong candidate for productization (per longer-term goal)
