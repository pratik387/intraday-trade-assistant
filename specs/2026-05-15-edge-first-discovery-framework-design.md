# Edge-First Discovery Framework — Design Spec

**Date:** 2026-05-15
**Branch:** `research/post-sebi-edge-setups`
**Status:** Design — pending user review
**Supersedes:** prior sanity-script + gauntlet-v2 kill-test methodology used for setups in `tools/sub9_research/sanity_*.py` during the 2026-05-10..2026-05-15 research wave

---

## 1. Motivation

Twelve consecutive Indian intraday setup hypotheses were retired in a single research wave (inside-bar, RSI extreme, BB touch, pivot break, ICT/SMC, 3-bar HH both directions, 14:30 vertical drop on large-cap, Open=High Sustained Weakness, weekly expiry pin, etc.). Every retirement was driven by a sanity script + gauntlet-v2 cell-mine producing `0 ship-eligible cells` vs Bonferroni-expected false-positive baselines.

This rate of retirement is itself a red flag. Three concrete methodology failures explain it:

1. **Pattern interpretation too literal.** Triggers were translated as 5-line mechanical rules. Real Indian intraday edge lives in context (regime, flow, intermarket, time-of-day, event proximity), not in chart geometry alone. Encoding the trigger without the context kills the edge before it can be observed.

2. **Sanity simulator missing real-trader behavior.** Hard time-stops with uncapped downside, fixed 1R/1.5R targets, mechanical next-bar-open entries, no slippage/spread modeling, no portfolio constraints. The simulator's uniform bias produced uniformly-negative outcomes on most patterns — not evidence of no edge, but evidence of consistent simulator bias.

3. **Cell-mining methodology fished in a biased simulator.** Scanning 6,000-14,000 cells per pattern with weak gauntlet gates, accepting "0 ship-eligible / worse than chance" as proof of no edge — except the floor itself was bent by the simulator's bias. Cell-mining is a confirmatory tool, not a discovery tool.

The meta-bias: every test was structured as a *kill-test* ("does pattern X pass gauntlet?"). None was structured as a *discovery-test* ("where does conditional drift live in this market structure?"). This bias systematically favored maintaining the 3 shipped setups and rejecting candidates.

**The framework had never validated it could reproduce known-good edge before being used to retire candidates.** That is the foundational miss this spec corrects.

## 2. Goals

**Primary goal.** Build a discovery-test methodology that can:

1. Surface where conditional drift lives in any defined event population, sliced by a rich feature library — *before* a trigger or exit logic is committed.
2. Reproduce live `gap_fade_short` Holdout PF (≈1.36) to within 10% as a mandatory parity gate before any retire/ship decision.
3. Apply walk-forward validation across Indian regulatory shocks (STT Oct-2024, SEBI Feb-2025, SEBI Oct-2025, STT Apr-2026) to detect regime erosion early.
4. Classify edges by rule-orthogonality at hypothesis time, never pursuing rule-dependent candidates without an explicit hedging story.
5. Distinguish standalone-shippable edges from ensemble-feature-only edges via a two-tier ship gate.
6. Define decay-and-replace governance for already-shipped setups.

**Secondary goal.** Apply the framework to three initial research targets that test its assumptions and surface portfolio-complementary candidates:

- Parity test on `gap_fade_short` (mandatory; framework cannot be used until this passes)
- LONG-side panic-gap-down catch in small/mid-cap (portfolio is 100% SHORT today; market is structurally long-biased)
- Ensemble feature mining on the 3 live setups (find context-conditional sub-regions where production PF lifts)

**Non-goals.**

- Re-testing the 12 retired patterns immediately. The methodology fix is not a retirement-rescue mission. After the parity gate passes and the framework demonstrates value on the 3 targets, *some* retired candidates may be re-tested if the user requests it.
- Building an ensemble combiner / portfolio sizer in this iteration. Tracking, but out of scope.
- Adding pre-open auction features, multi-time-frame consistency checks, or sentiment-from-bar-shape features. Tier-3 backlog.

## 3. Architecture

The framework is one Python module + one configuration block + a fixed validation pipeline.

```
tools/edge_discovery/
├── explorer.py              # main entry: event population → conditional outcome distributions
├── features/
│   ├── symbol_features.py   # per-event: chart, distance, regime, prior-day, ADV, cap_segment
│   ├── market_features.py   # per-event: NIFTY/BankNIFTY direction, VIX, AD, FII/DII, basis, USD-INR, crude
│   └── event_features.py    # per-event: earnings_proximity, expiry_flag, rebalance_flag, RBI_flag
├── outcomes/
│   ├── returns.py           # +5m / +15m / +30m / +60m / +120m / EOD returns, MFE/MAE
│   └── costs.py             # per-cap-segment × ADV-bucket bid-ask spreads + market impact
├── validation/
│   ├── parity_gate.py       # reproduces live setups; framework gates itself on PF match
│   ├── walk_forward.py      # train-on-window / test-on-next-month, walking
│   └── rule_orthogonality.py # classification + check against known SEBI/STT change dates
├── ship_gate.py             # two-tier: standalone (n≥300/yr, PF≥1.30, walk-forward stable) vs ensemble-feature
├── decay_monitor.py         # for shipped setups: rolling 6mo PF, retire trigger if <1.0 / pause if <1.2
└── targets/
    ├── target_parity_gap_fade.py        # MANDATORY first target
    ├── target_long_panic_gap_down.py    # LONG-side small/mid catch
    └── target_ensemble_live_setups.py   # context-conditional sub-region mining on live 3 setups
```

The explorer's surface area is small: it takes (event_population_callable, context_feature_modules, outcome_modules) and returns a `ConditionalOutcomeTable` — a typed object with methods for slicing, ranking edge regions, and exporting reports.

Configuration in `config/configuration.json` adds an `edge_discovery` block with: feature library version, outcome window grid, ship-gate thresholds, decay thresholds, walk-forward window sizes, parity tolerance. No hardcoded defaults inside code.

## 4. Event Population Definition

An event population is a callable returning a DataFrame with at minimum: `symbol`, `event_time` (IST-naive), and any pattern-specific labels. The callable receives the framework's universe loader and 5m feather scaffolding — the same scaffolding the retired sanity scripts used — so existing data infrastructure is reused.

Event populations are intentionally BROAD (not pre-filtered to be rare). Examples:

- "all small-cap stocks with gap-up ≥1% on first 5m bar" (n ≈ 10,000+ over 24mo Discovery)
- "all small/mid-cap stocks with gap-down ≥1% on first 5m bar" (n ≈ 8,000+)
- "all 5m bars in 09:15-10:45 that close in top-30% of their range with vol > 1.3× 5-bar mean" (broad continuation signature, n ≈ 100,000+)

Pre-narrowing populations is the failure mode that produced the retired patterns. The explorer's job is to NARROW the population via conditional outcome slicing — not to receive it pre-narrowed.

Each event in the population is passed through all three feature modules (symbol, market, event-calendar) AND the outcome module. The result is a row in the conditional outcome table.

## 5. Context Feature Library

Three categories. Every event gets all three.

### 5.1 Symbol-level features

- `cap_segment` (small/mid/large/micro per `nse_all.json`)
- `adv_bucket` (avg daily vol over prior 20 sessions: tiers)
- `mis_leverage` (from `nse_all.json` or `trade_report.csv`)
- `dist_from_pdh_pct`, `dist_from_pdl_pct`
- `dist_from_20ema_pct`, `dist_from_50ema_pct`
- `prior_session_pct_change`, `prior_5sess_pct_change`
- `gap_pct` (today's open vs yesterday's close)
- `delivery_pct` (T-1, per NSE EOD bhavcopy — already in pipeline)
- `delivery_pct_5d_trend` (rising / falling / flat)
- `fno_eligible` (boolean, derived from F&O 200 list)
- `fno_oi_change_pct_t1` (for F&O names: prior-day OI change as % of OI)
- `fno_position_signature` (long-build / short-build / long-unwind / short-cover, derived from price + OI signs)
- `bar_range_pct`, `bar_body_pct`, `bar_upper_wick_ratio` (when event is bar-defined)
- `vwap_distance_pct` (intraday VWAP at event_time)

### 5.2 Market-level features

- `nifty_intraday_pct` (NIFTY direction at event_time)
- `banknifty_intraday_pct`
- `banknifty_vs_nifty_relative_strength`
- `india_vix` (level)
- `india_vix_5d_change`
- `advance_decline_ratio` (NSE-wide breadth at event_time)
- `fii_net_flow_t1_inr_cr` (T-1 FII net flow in cash, derived from NSE daily reports)
- `dii_net_flow_t1_inr_cr` (T-1 DII net flow)
- `nifty_futures_basis_pct` (NIFTY futures premium/discount to spot)
- `usd_inr_intraday_pct` (USD-INR direction at event_time; affects IT/Pharma/Exporters)
- `crude_intraday_pct` (Brent or WTI direction; affects O&G stocks oppositely)
- `sector_momentum` (event's sector intraday move vs NIFTY at event_time)
- `top3_sector_breadth` (count of top-3 sectors today that are up vs flat)

### 5.3 Event-calendar features

- `days_to_next_earnings` (for stock-specific dates; data needed from `data/earnings_calendar/`)
- `is_expiry_week` (Thursday is in the same week)
- `is_expiry_day` (specifically Thursday)
- `is_monthly_expiry_day` (last Thursday of month)
- `is_index_rebalance_day` (NIFTY/MSCI quarterly rebalance dates)
- `is_rbi_policy_day` (RBI MPC announcement days)
- `is_budget_day` (Union Budget date)
- `days_to_next_sebi_rule_change` (known forward dates: STT changes, F&O rule changes)
- `days_since_recent_sebi_rule_change` (post-change adjustment period)

Some features require data pipelines that don't fully exist today. Implementation plan must distinguish:
- **Tier-A features** (ready now): cap_segment, adv_bucket, dist_from_pdh/pdl, gap_pct, delivery_pct, mis_leverage, NIFTY intraday, INDIA VIX, bar features — all in current pipeline.
- **Tier-B features** (need backfill but data exists): FII/DII flow, USD-INR, crude, fno_oi, expiry/rebalance/RBI calendars — sourceable from NSE bhavcopy + RBI + Yahoo for index data.
- **Tier-C features** (need new pipeline): earnings calendar (partial — `data/earnings_calendar/` exists, completeness TBD), advance-decline at 5m granularity (may need rebuild from NSE-wide 5m feathers).

The framework launches with Tier-A only. Tier-B is phase-2. Tier-C is phase-3.

## 6. Outcome Computation + Execution Cost Modeling

For each event, the outcome module computes:

- **Forward returns** at +5m / +15m / +30m / +60m / +120m / EOD horizons, before costs
- **MFE** (maximum favorable excursion) and **MAE** (maximum adverse excursion) at each horizon
- **Time-to-MFE** and **time-to-MAE** (helps identify natural exit windows)

Then applies execution costs:

- **Bid-ask spread** per (cap_segment, adv_bucket) — calibrated from a one-time empirical pass over a sample of NSE 5m bars (high-low at minimum-volume intervals as a proxy for spread). Approximate baseline: small-cap with ADV<100K → 0.15% per side; small-cap ADV 100K-500K → 0.08%; mid-cap → 0.04%; large-cap → 0.02%.
- **SL slippage** during fast moves — model as 0.5× bar_range when SL gaps through, otherwise 0.1% additional.
- **Market impact** for size: linear in (order_size / ADV). At 1% of ADV, add 0.05% impact. Realistic upper limit: 5% of ADV in a single bar.

Pre-cost and post-cost outcomes are both reported. The post-cost figure is the one ship gates use.

## 7. Edge Region Detection

For a `ConditionalOutcomeTable`, edge region detection works as follows:

1. For each context feature, compute mean / median / Sharpe of each outcome horizon, bucketed by feature value (continuous → 5 quantile buckets; categorical → as-is).
2. For each pair of features (2D combinations), compute the same statistics on the joint buckets.
3. For each triple (3D combinations), same — but only for feature triples that include at least one bucket with n ≥ 50.
4. Rank candidate edge regions by (post-cost-mean-return-magnitude × √n) — a t-statistic-like signal accounting for both effect size and sample size.
5. Output top-20 candidate edge regions, with: feature cut, n, mean post-cost return at +120m, Sharpe, win-rate, walk-forward stability score (computed by validation/walk_forward.py), monthly distribution.

Critically, this is REPORTING, not VERDICTING. The explorer surfaces candidates; the ship gate decides. No automatic RETIRE label.

## 8. Validation Components

### 8.1 Parity Gate (mandatory before any other use)

`validation/parity_gate.py` re-implements the live `gap_fade_short` event population and outcome simulation INSIDE the framework, then compares the framework's Discovery + OOS + Holdout PF/WR/N to the live setup's recorded production values:

- Live Holdout target: PF ≈ 1.36, WR ≈ 70%, N ≈ 797 (from `config/configuration.json` _live_status field)
- Framework Holdout output: PF within ±10%, WR within ±5pp, N within ±10%

If the parity gate fails:
- The framework is BROKEN. No new research, no retirements.
- Failure-mode triage: check (a) event-population match, (b) feature pipeline alignment, (c) outcome-cost model alignment with production fees.
- Fix the framework. Re-run parity. Only proceed when ±10% threshold is met.

The parity gate is also run for `circuit_t1_fade_short` and `delivery_pct_anomaly_short` — all three live setups must reproduce within tolerance.

### 8.2 Walk-Forward Validation

`validation/walk_forward.py` operationalizes walk-forward simulation:

- Training window: 6 months
- Validation window: 1 month
- Step size: 1 month
- For each step, re-fit any tunable parameters on training, evaluate on validation, record validation PF
- Walk-forward stability score: 1 − (std of validation PFs / mean of validation PFs)
- A walk-forward stability score < 0.5 means the parameter optimization is fragile

This replaces the prior "rolling 6mo PF check" which was the weak version of this concept. Walk-forward catches regime drift in the optimization itself, not just in a fixed config.

### 8.3 Rule-Orthogonality Classification

At hypothesis time, every candidate gets classified:

- **rule_orthogonal**: edge source is structural microstructure (retail can't short cash; structural long-bias; institutional rebalance flows; auction effects). Examples: gap_fade_short, circuit_t1_fade, delivery_pct_anomaly, LONG-side panic-gap-down catch.
- **rule_dependent**: edge source is a specific F&O rule, STT level, MIS leverage cap, or other tunable parameter. Examples: STT-arbitrage trades, MIS-leverage-specific sizing strategies, F&O-specific spread trades.

The framework REJECTS rule_dependent candidates at hypothesis time unless an explicit hedging-against-rule-change story is included. After Oct-2025 SEBI changes plus Apr-2026 STT changes, the cost of pursuing rule_dependent candidates is high.

The classification cross-references known rule-change dates against the candidate's walk-forward PF series — if PF drops > 50% at or near any policy date, rule-orthogonality is auto-questioned.

## 9. Ship Gates

Two-tier:

### 9.1 Standalone ship gate

- Discovery PF ≥ 1.30 (post-cost)
- OOS PF ≥ 1.20 (post-cost)
- Holdout PF ≥ 1.15 (post-cost)
- N ≥ 300 trades/year on average
- Walk-forward stability score ≥ 0.5
- Rule-orthogonal OR has explicit hedging story
- Monthly winning months ≥ 55%
- Top-month NET concentration < 40%
- Parity gate has passed framework-wide

If all criteria met → ship as standalone setup. Adds to production portfolio.

### 9.2 Ensemble feature gate

- The candidate's identified edge region has n < 300/year (too small for standalone)
- Edge magnitude > 0.4× population standard deviation (effect-size sanity)
- Walk-forward stability score ≥ 0.5
- When applied as a context filter to one of the 3 live setups, conditional PF lift > 0.15

If criteria met → register as a candidate ensemble feature. Does NOT auto-deploy. Goes into a feature catalog (`tools/edge_discovery/feature_catalog.md`) for future ensemble work (out of scope this iteration).

## 10. Decay-and-Replace Governance

For every shipped setup (currently `gap_fade_short`, `circuit_t1_fade_short`, `delivery_pct_anomaly_short`):

- `decay_monitor.py` runs against live + paper-trading results
- Rolling 6-month post-cost PF tracked monthly
- Thresholds:
  - PF ≥ 1.20 → ACTIVE
  - 1.00 ≤ PF < 1.20 → CAUTION (sized down by 50%)
  - PF < 1.00 → PAUSED (zero allocation, pending re-evaluation)
  - PF < 0.80 for 2 consecutive months → RETIRED (removed from portfolio, post-mortem written)
- A retired setup's slot in capital_budget_pct is freed for ensemble feature deployment or new shipped setups

This component is mandatory because the prior research wave had no formal way to retire live setups should their edge erode post-policy-change.

## 11. First Three Research Targets

### Target 1 — Parity test on `gap_fade_short` (MANDATORY)

Reproduce live `gap_fade_short` PF≈1.36, WR≈70%, N≈797 on its Holdout window using the framework's event population, feature pipeline, and outcome model.

If parity fails: framework is broken, fix it.
If parity passes: framework is validated for use.

Estimated effort: 3 days. Mostly aligning the framework's universe + feature + cost models to production's internals.

### Target 2 — LONG-side panic-gap-down catch (small/mid-cap)

Event population: all small/mid-cap MIS-eligible names with gap-down ≥ 1% on first 5m bar (broader than the candidate trigger; let the explorer find the edge region).

Hypothesis: a sub-region of this population (e.g., gap-down ≥3%, no F&O eligibility, prior-5sess down, high delivery%) has positive forward drift because retail cannot short cash small-caps without F&O — only institutional value-bidders are on the BUY side, with no organic short-cover demand. Edge source is structural.

Rule-orthogonality: rule_orthogonal (depends on absence-of-cash-short-selling, which no current SEBI proposal changes).

Expected n: 1,500-4,000 events over 24mo Discovery. Sufficient for standalone ship if edge is broad; sufficient for ensemble feature otherwise.

Estimated effort: 4 days.

### Target 3 — Ensemble feature mining on the 3 live setups

For each of `gap_fade_short`, `circuit_t1_fade_short`, `delivery_pct_anomaly_short`:

- Load the live setup's event population from existing parquets (`reports/sub7_validation/`, `reports/sub8_oos_validation_clean/`, `reports/sub8_oos_holdout_clean/`).
- Apply the framework's full context feature library to those events.
- Run edge-region detection — find context-conditional sub-regions where post-cost PF lifts above each setup's current baseline.
- For any region meeting the ensemble feature gate: register the feature in the catalog.

Note that this is NOT modifying the live setups. It is identifying CANDIDATE filters for a future ensemble layer.

Estimated effort: 3 days.

**Total**: 10 days of focused implementation + research time.

## 12. Engineering Scope and Phasing

### Phase 1 — Framework core + Tier-A features (week 1)

- `explorer.py`, `symbol_features.py` (Tier-A subset), `outcomes/returns.py`, `outcomes/costs.py`
- `ship_gate.py`, `decay_monitor.py`, `rule_orthogonality.py`
- Config block in `configuration.json`
- Unit tests on all of the above (TDD)

### Phase 2 — Validation pipeline (week 2)

- `validation/parity_gate.py`
- `validation/walk_forward.py`
- Run parity gate on all 3 live setups. Iterate until passes.

### Phase 3 — Tier-B features + first research targets (week 3)

- `market_features.py` (FII/DII, basis, USD-INR, crude, VIX)
- `event_features.py` (expiry, rebalance, RBI)
- Run Target 1 (parity, again, with full Tier-A+B features)
- Run Target 2 (LONG panic-gap-down)
- Run Target 3 (ensemble features)
- Generate result reports

### Phase 4 — Decision + ship (week 4)

- Per-target verdict: ship / ensemble-feature / no-edge / framework-issue
- Update `decay_monitor.py` integration with live + paper-trading pipeline
- Commit results, propose portfolio changes

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Parity gate fails repeatedly | Triage by component (event pop / feature pipeline / cost model). Time-box to 5 days. If still failing, escalate to user — framework rebuild may need cuts. |
| Tier-B data pipelines (FII/DII, USD-INR) more work than estimated | Phase 1 uses Tier-A only; Tier-B can shift to phase 3 if needed. Targets 1 and 2 are runnable with Tier-A. |
| LONG-side panic-gap-down has insufficient n in tight edge region | Two-tier gate handles this — region with n<300 becomes ensemble feature, not standalone. |
| Walk-forward shows live setups themselves have decaying edge post-Oct-2025 | `decay_monitor.py` is the right tool to surface this. Decisions on what to do (pause / re-tune / accept) are governance not framework. |
| User loses patience with 4-week implementation while research is stalled | Phase 1 + parity test = 2 weeks for the foundational validation. New research can resume only after that. There is no shortcut that preserves the methodology fix. |

## 14. Out of Scope (Tracking for Future)

- Pre-open auction features (Tier-3 backlog)
- Multi-time-frame consistency (Tier-3 backlog)
- Sentiment-from-bar-shape (Tier-3 backlog)
- Strategy ensemble combiner (next iteration after this framework is operational)
- Kelly-derived position sizing (next iteration)
- Order-execution algorithms (TWAP/VWAP/POV) for live deployment — separate sub-project

## 15. Open Questions

None that block the spec. The implementation plan (next step) will surface specific data-pipeline decisions (e.g., FII/DII flow data source: NSE daily reports vs Moneycontrol API vs custom scrape) that can be answered then.

---

## Acceptance Criteria for the Framework

The framework is operational when:

1. Parity gate passes for all 3 live setups (PF within ±10%, WR within ±5pp, N within ±10%).
2. Walk-forward validation correctly identifies STT Oct-2024 and SEBI Oct-2025 as regime breaks in at least one live setup's PF series.
3. Edge-region detection on the LONG-side panic-gap-down target produces a ranked list of candidate cuts with full post-cost statistics.
4. Decay monitor integrates with live + paper-trading and outputs monthly health reports for all 3 live setups.
5. Ensemble feature catalog has at least 5 entries from the live-setup ensemble mining target.

If any of these fail at the end of the 4-week phase, the framework is incomplete and decisions about new setups are blocked.

---

End of design spec.
