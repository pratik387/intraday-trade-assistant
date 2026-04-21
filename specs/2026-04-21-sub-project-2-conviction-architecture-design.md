# Sub-Project #2: Conviction Architecture — Design Spec

**Date:** 2026-04-21
**Sub-project:** #2 of 6 in the Trading System Rebuild
**Status:** Design approved, awaiting writing-plans
**Predecessor:** Sub-project #1 (Edge Discovery — 90 approved rules from gauntlet) + sub-project #3 (Cross-Sectional Features — F1+F2 filter, ~227/day output)
**Successor:** Sub-project #4 (Shadow/Parity Loop) — validates live ML predictions match backtest

---

## 1. Context: why this exists

The Stage-5c filter output is ~227 candidate trades/day on 2025 OOS data, with aggregate PF 1.28 after cross-sectional filtering. Production target is 15-20 trades/day. The remaining 11x cut is a **ranking problem**, not a filtering problem — filters are monotonic and further tightening them would overfit; a ranker picks the top N by conviction score.

Sub-project #2 builds that ranker. It sits between sub-project #3 (filter output) and execution, taking ~227 daily candidates and selecting top 50 (v1; tightening to 20 later) by predicted risk-adjusted outcome.

**Design constraint (from user memo, lessons.md 2026-04-21):** This is personal capital, no shipping deadline. Priority is correctness, not velocity. No "v1 baseline, v2 real" phase shipping. Build the correct design on first pass; defer advanced features (Kelly sizing, setup-specific model production deployment) to later sub-projects only when shadow-loop evidence supports them.

### What this sub-project is NOT

- Filtering (sub-project #3 already does this)
- Live execution infrastructure (sub-project #6)
- Shadow/paper vs backtest parity checking (sub-project #4)
- Model retraining cadence / rule retirement (sub-project #5)
- Options-chain features (PCR, IV) — our universe is non-F&O

---

## 2. Empirical basis

From sub-project #1 (gauntlet) and sub-project #3 (cross-sectional):

- **178K Discovery trades** (Jan 2023 – Dec 2024), all with clean decision-time features + realized outcomes (PnL, R-multiple, MAE/MFE)
- **97K 2025 OOS trades** (Jan – Dec 2025) from the `20260421-134338_full` OCI backtest, clean analytics via entrypoint.py postprocess fix
- **Stage 5c validated on 2025 OOS** — filter F1+F2 produces +3.1% PF lift, identical trade-count cut (−44%), session Sharpe marginally positive. Evidence that cross-sectional filters generalize.
- **5 surviving setups after narrative gate**: premium_zone_short (Family 1 retail-reversion, dominant), range_bounce_short, resistance_bounce_short, order_block_short, vwap_lose_short (Family 2 institutional-reference)
- **Baseline PF degrades OOS from 1.36 → 1.24** — the 90-rule set alone has real but reduced OOS edge. Conviction ranking must pick the SUBSET of 227 candidates that generalize best.

### Broken predecessor: `rank_score`

Per design spec §1 of sub-project #1: the existing `rank_score` has −0.019 correlation with realized PnL. It is noise. Sub-project #2 replaces it with a properly-trained ML model.

### Why heuristic weighting was rejected

Initial brainstorm proposed a 7-feature heuristic with eyeballed weights. Rejected because:
1. Weights are indefensible without training data fitting → unjustifiable by design
2. The existing broken `rank_score` was itself a heuristic — we learned heuristic-scoring-on-this-problem fails
3. XGBoost on 178K labeled trades is not exotic; it's commodity quant practice
4. User's correctness-first memo: no placeholder baselines just to ship a v1

### Why this is not a novel research project

Per research (Indian pro algo: aaryansinha16/AI-trader reference, QuantInsti EPAT curricula, academic literature): XGBoost-based intraday trade scoring with SHAP interpretation is commodity technique. What's novel in our specific case:

1. **Our universe** — retail-dominated illiquid NSE cash, not F&O
2. **Our feature set** — ICT/SMC detector context (`pdz_confluence_count`, `ob_has_mss_confirmation`, etc.) that pros don't use
3. **Pre-filtered training data** — training on ~178K candidates that already passed Stage 1-5c rather than raw signal
4. **Small capacity** — we can trade anywhere pros can't due to capital-deploy limits

The ML architecture is standard. The edge is in #1 (finding durable patterns in under-studied universe).

---

## 3. Design decisions

### 3.1 Scoring approach

**Supervised ML (XGBoost regression).** Not heuristic weighting.

### 3.2 Target variable

**R-multiple (pnl / risk per trade)** — risk-normalized, cross-stock comparable, already computed per-trade in `trade_report.csv.r_multiple`.

Why R-multiple over raw PnL: a ₹100 penny stock and ₹1000 mid-cap have different natural PnL scales. R-multiple normalizes them for fair comparison.

Why regression over classification: binary (win/loss) loses magnitude — a 70% WR 0.5R setup is worse than a 50% WR 2R setup. R-multiple regression captures both win probability and payoff size.

### 3.3 Feature set (~35-40 features)

Five buckets, explicit whitelist approach (not blocklist — safer against leakage):

**1. Momentum / trend (standard pro, ~6 features):**
- `rsi_zone` (binned 0-30 / 30-70 / 70-100)
- `macd_signal_dist` (normalized by ATR)
- `vwap_distance_pct` (from `vwap5` in trade_report.csv)
- `ema_9_slope` (derived from bar5 close series)
- `momentum_3bar_pct` (already in trade_report.csv)
- `momentum_1bar_pct` (already in trade_report.csv)

**2. Volatility (standard pro, ~3 features):**
- `atr_pct_of_price` (derived from bb_width_proxy + price)
- `bb_width_proxy` (already in trade_report.csv)
- `range_size_pct_of_atr` (derived)

**3. Volume (standard pro + sub-project #3, ~6 features):**
- `volume5` (already present)
- `vol_z` (already present)
- `vol_ratio` (already present)
- `rvol_pct_tier` (from sub-project #3 UniverseRVOLState)
- `body_size_pct` (already present)
- `wick_ratio` (already present)

**4. ICT-specific (our edge, ~12 features):**
- `pdz_confluence_count` (count 0-3 of MSS+FVG+OB at zone)
- `pdz_range_position` (0.0-1.0, where in structural range)
- `pdz_has_mss_confluence` (bool)
- `pdz_has_fvg_confluence` (bool)
- `pdz_has_ob_confluence` (bool)
- `pdz_range_size_atr` (range width normalized)
- `pdz_htf_bearish` (bool, HTF bias alignment)
- `ob_confluence_count` (for order_block_short)
- `ob_has_liquidity_sweep` (bool)
- `resistance_touches` (for resistance_bounce_short)
- `resistance_strength` (scalar)
- `pattern_age_mins` (how old is the setup signal)

**5. Cross-sectional + session + regime (from #3 + standard, ~24 one-hot dims ≈ 6 semantic features):**
- `crowdedness_count` (from sub-project #3 CrowdednessCounter)
- `hour_bucket_onehot` (opening / morning / lunch / afternoon / late → 5 dims)
- `regime_onehot` (chop / trend_up / trend_down / squeeze → 4 dims)
- `cap_segment_onehot` (large / mid / small / micro / unknown → 5 dims)
- `day_of_week_onehot` (Mon–Fri → 5 dims)
- `setup_type_onehot` (5 surviving setups → 5 dims)

**Total: ~30 semantic features, ~35-40 columns after one-hot encoding.**

### 3.4 Leakage audit (explicit whitelist)

Columns that MUST be excluded from training (they are outcomes or derived from outcomes):
```
realized_pnl, total_trade_pnl, net_pnl
label_hit_t1, label_hit_t2
gross_exit_qty, position_closed
e1_*, e2_*, e3_* (all exit event fields)
last_exit_ts, last_exit_reason
mae, mfe, mae_pct, mfe_pct, r_multiple
bars_held, time_in_trade_minutes
exit_price, fees, slippage_bps
```

Implementation: `ALLOWED_FEATURES = [...]` whitelist in the training script. Any column not in the whitelist is dropped. Leakage becomes a whitelist-miss bug rather than a blocklist-miss catastrophe.

### 3.5 Feature coverage audit

For every whitelisted feature, measure:
- % null / zero in Discovery
- % null / zero in 2025 OOS
- Distribution similarity between Discovery and 2025 (Kolmogorov-Smirnov test p-value)

**Drop features with:**
- > 40% missing in either dataset
- Significant distribution shift (KS p < 0.01 with substantial effect size)

Document drops in the training log. Re-run with remaining features.

### 3.6 Selection mechanism

**Online realistic top-50 with calibration-derived minimum threshold.** Not batch top-50, not hour-bucketed forcing, not per-setup forcing.

At each candidate's decision time:
1. Compute features
2. Score: `predicted_R = model.predict(features)`
3. If `predicted_R >= threshold AND daily_count_so_far < 50`: admit
4. Otherwise: reject (with logged reason)

**Threshold derivation:**
- Build calibration plot on Discovery held-out fold (not 2025)
- X-axis: predicted_R buckets (deciles)
- Y-axis: realized_R median per bucket
- Pick threshold where realized_R begins exceeding some floor (e.g., 0.3R after expected slippage)
- Lock threshold
- Apply unchanged to 2025 OOS

### 3.7 Sizing

**Flat risk-based sizing — existing infrastructure, no change.**

Each admitted trade risks the same rupee amount (current `capital_manager.py` implementation using `risk_rupees`). Conviction is applied via selection (top-50), not via position size.

**Not using in v1:**
- Conviction-tiered size multipliers
- Kelly-derived sizing
- Variance-adjusted sizing

All of these are legitimate advanced techniques. They belong in sub-project #5 (Strategy Lifecycle) only after sub-project #4 (Shadow Loop) confirms the model's calibration is stable in live paper trading.

### 3.8 Model architecture — universal + per-setup sanity check

**Primary deployment: universal model.** One XGBoost regressor trained on all 178K Discovery trades with `setup_type` as a categorical feature.

**Architectural empirical test:** Also train separate models for premium_zone_short (104K trades) and range_bounce_short (58K trades) — the two setups with enough data. Compare OOS RMSE + PF on 2025 data between (universal-model-on-setup-X) vs (setup-X-specific-model).

**Decision rule:**
- If universal model is within 5% of per-setup model on OOS → ship universal
- If per-setup model is meaningfully better (>5% lift) → ship universal for small-N setups + per-setup for the 2 big setups

**vwap_lose_short, order_block_short, resistance_bounce_short** (1K, 7K, 8K trades respectively): universal model only — insufficient data for reliable per-setup training.

### 3.9 Training protocol (3-way OOS discipline)

Same protocol as sub-project #1 gauntlet:

- **Train:** Discovery (Jan 2023 – Dec 2024), 178K trades, split into 80% train / 20% validation fold for XGBoost early-stopping
- **Validate:** 2025 Q1-Q3 (Jan – Sep 2025), ~70K trades, used to measure OOS generalization + pick final hyperparameters (NOT to tune feature set or target — those stay locked)
- **Holdout:** 2025 Q4 – Mar 2026 (Oct 2025 – Mar 2026), ~27K trades, one-shot final check, no peeking during training/validation

XGBoost hyperparameter tuning via time-series cross-validation within Discovery (walk-forward, never peeking forward), NOT via validation fold peeking.

### 3.10 Daily trade cap

**v1: 50/day. Not 20/day.**

Rationale (per user direction):
- Less aggressive cutoff = more room for ML to not be perfect
- More trades = more statistical data in shadow loop
- Matches typical broker MIS slot capacity (~50 concurrent)
- Tighter cap to 20 earned later via shadow-loop evidence, not from design-time speculation

---

## 4. Architecture

### 4.1 Module structure

```
services/conviction/                       ← NEW package
├── __init__.py
├── feature_extractor.py                   ← Computes ~35 features from bar5 + trade context
├── scorer.py                              ← XGBoost wrapper (load model, predict)
├── gate.py                                ← ConvictionGate — selection + threshold
└── calibration.py                          ← Threshold derivation from held-out fold

models/
└── conviction/
    ├── 2026-04-21-universal-xgboost.pkl   ← Trained model artifact
    ├── 2026-04-21-feature-spec.json       ← Exact feature list + encoding
    ├── 2026-04-21-calibration.json        ← Threshold + calibration curve
    └── 2026-04-21-shap-analysis.md        ← Top features overall + per-setup

tools/edge_discovery/stages/
└── stage5d_conviction_simulation.py       ← Backtest replay (mirrors stage5c)

tools/conviction/                          ← Training scripts
├── train_universal.py
├── train_per_setup.py
├── build_calibration.py
└── compare_architectures.py               ← universal vs per-setup OOS comparison

tests/conviction/
├── __init__.py
├── test_feature_extractor.py
├── test_scorer.py
├── test_gate.py
├── test_calibration.py
└── test_stage5d.py
```

### 4.2 Data flow

```
Pattern detector fires setup signal
   ↓
Candidate proposed (symbol, setup_type, decision_ts, trade context)
   ↓
services.cross_sectional.gate.CrossSectionalGate (sub-project #3) — pass/reject
   ↓ (if pass)
services.conviction.feature_extractor.extract(candidate) → feature dict
   ↓
services.conviction.scorer.predict(features) → predicted_R
   ↓
services.conviction.gate.ConvictionGate.evaluate(candidate, predicted_R):
  - Check: predicted_R >= threshold ?
  - Check: daily_count < 50 ?
  - Return: ALLOW or REJECT (reason)
   ↓ (if ALLOW)
Existing sizing + execution pipeline (capital_manager.py)
```

### 4.3 Training flow

```
trade_report.csv (Discovery) + analytics.jsonl (PnL, R-multiple labels)
   ↓
feature_extractor.build_training_frame() → X_train, y_train (R-multiple)
   ↓
leakage_audit(X_train) → assert no forbidden columns
   ↓
coverage_audit(X_train, X_validation_2025) → drop low-coverage features
   ↓
xgboost.train(X_train, y_train, validation=holdout_fold) → universal_model.pkl
   ↓
build_calibration(universal_model, holdout_fold) → threshold, calibration_curve.json
   ↓
evaluate_oos(universal_model, X_2025, y_2025) → OOS RMSE, PF@top-50, Sharpe
   ↓
compare_architectures(universal, per_setup_models) → ship decision
```

### 4.4 Integration with existing codebase

- `ConvictionGate` slots into `trade_decision_gate` / `screener_live.py` the same way `CrossSectionalGate` does in sub-project #3
- Runs AFTER CrossSectionalGate (cross-sectional filter → conviction ranker)
- Model artifact + feature spec + calibration loaded once per worker process at startup
- Stateless at evaluation time (predict is pure function of features)
- Config block in `configuration.json`:

```json
"conviction_gate": {
  "enabled": true,
  "model_artifact": "models/conviction/2026-04-21-universal-xgboost.pkl",
  "feature_spec": "models/conviction/2026-04-21-feature-spec.json",
  "calibration": "models/conviction/2026-04-21-calibration.json",
  "daily_cap": 50,
  "min_predicted_r": 0.3
}
```

All thresholds config-driven per project standard.

---

## 5. Testing

### 5.1 Unit tests

- `test_feature_extractor.py`: given a candidate context dict, produces exact expected feature dict; handles missing/None values deterministically; leakage audit passes
- `test_scorer.py`: loads model artifact, predicts, returns float; handles unknown categorical values gracefully
- `test_gate.py`: daily cap enforced, threshold enforced, reason strings correct
- `test_calibration.py`: threshold derivation is deterministic given a sample calibration plot
- `test_stage5d.py`: backtest replay produces DataFrame with admitted/rejected columns

### 5.2 Integration test

- End-to-end synthetic fixture: 100 trades with known predicted_R distribution, verify top-N selection + threshold filter behave as expected

### 5.3 Model validation tests (one-shot on OOS 2025)

Four tests the trained model must pass:

1. **OOS PF lift over baseline.** Top-50 by predicted_R → PF_ml. Random 50 per session → PF_random. Required: `PF_ml > PF_random` consistently on a monthly basis. Fail means the model is not discriminating.

2. **Calibration monotonicity.** Bucket 2025 predictions into deciles. Realized_R should increase monotonically across deciles. Non-monotonic = model is not calibrated.

3. **SHAP stability.** Top-10 features by SHAP magnitude on Discovery vs on 2025 validation fold. Must overlap on ≥7/10. Large divergence = overfit to Discovery.

4. **Per-session Spearman rank correlation.** Per session, Spearman rho between predicted_R and realized_R across candidates. Median across sessions should be significantly > 0 (p < 0.01 via sign test).

If any of the 4 fails, model does not ship — diagnose and retrain.

---

## 6. Success criteria

This sub-project ships when ALL of:

1. **Feature extractor implemented + leakage audit passes (whitelist coverage)** + coverage audit produces clean feature list
2. **Universal XGBoost model trained** on Discovery + validated on 2025 + holdout set aside for one-shot final check
3. **All 4 model validation tests pass** (PF lift, calibration, SHAP stability, rank correlation)
4. **Universal vs per-setup architecture compared** — documented decision for shippable architecture
5. **ConvictionGate integrated** into trade_decision_gate / screener_live.py with config-driven toggle
6. **Stage 5d backtest replay produces clean 2025 OOS result** with top-50 + threshold filter
7. **Aggregate 2025 OOS PF ≥ 1.5 after full pipeline** (Stage 5b rule filter → Stage 5c cross-sectional → Stage 5d conviction). This is the bar that says the full pipeline is worth shipping to shadow loop.
8. **Documented handoff to sub-project #4** — predicted_R vs realized_R stream schema, expected drift thresholds, retraining cadence recommendation

**Note on criterion #7:** if aggregate OOS PF after full pipeline doesn't hit 1.5, ship anyway if it beats baseline top-50-random by 10%+. The bar is relative improvement over baseline, not an absolute target — absolute targets are how we ended up with spec'd success criteria that miss reality (as in sub-project #3's spec vs 2025 OOS results).

---

## 7. Out of scope (explicit)

- **Conviction-weighted or Kelly-derived sizing:** deferred until sub-project #4 confirms calibration stability
- **Setup-specific production models for small-N setups** (order_block_short, vwap_lose_short, resistance_bounce_short): universal model only in v1
- **Online model updating / continual learning:** sub-project #5 (Lifecycle)
- **Live fill-quality vs prediction parity:** sub-project #4 (Shadow Loop)
- **Model retirement criteria / automated retraining cadence:** sub-project #5
- **Capital scaling rules** (start small → scale up): sub-project #6 (Deployment)
- **Slot contention** (if all 50 MIS slots open when candidate fires): sub-project #6

---

## 8. Known risks & mitigations

| Risk | Mitigation |
|---|---|
| Model overfits Discovery and fails 2025 OOS | 3-way OOS discipline + SHAP stability test + calibration monotonicity. If model fails validation, do not ship — diagnose feature set, data leakage, or mis-specified target. |
| Training data distribution (2023-2024) differs materially from 2025 | Coverage audit measures KS distribution shift per feature. Drop features with large shift. Accept residual risk: regime-level changes are addressed in sub-project #5 (retraining cadence), not here. |
| Feature leakage (post-decision data sneaks in) | Explicit whitelist — any column not on allowlist is dropped. Reviewed in code review before training. |
| Single-setup sample size too small for per-setup training | Accept universal-only model for 3 small-N setups. Document this as known limitation; revisit when more data accumulates. |
| Threshold derivation data-snoops | Threshold computed on Discovery held-out fold only (NEVER on 2025). Locked before OOS measurement. |
| ML predictions diverge between backtest and live (fill timing, slippage) | Explicitly not handled here — sub-project #4 (Shadow/Parity Loop) handles live-vs-backtest parity. |
| Model artifact rot (saved model loads differently across environments) | Feature spec + version hash embedded in artifact metadata. Check spec match at load time. |
| `predicted_R` scores drift over time as market regime changes | Sub-project #5 addresses retraining cadence. In v1, model weights are locked after training — drift is detected by shadow loop, not auto-corrected. |

---

## 9. Sources

**Reference architectures (Indian NSE pro algo):**
- [aaryansinha16/AI-trader — NIFTY F&O XGBoost intraday (reference implementation)](https://github.com/aaryansinha16/AI-trader)
- [QuantInsti EPAT curriculum](https://www.quantinsti.com/epat)
- [Estee Advisors — Indian prop-desk quant methodology](https://esteeadvisors.com/proprietary-trading.php)

**Academic:**
- [Intraday volume forecasting in equity markets with ML (2025, arxiv)](https://arxiv.org/html/2505.08180v1)
- [Identifying Trades Using Technical Analysis and ML/DL models (2024)](https://arxiv.org/pdf/2304.09936)
- [Cross-sectional reversal of intraday returns in emerging markets (ScienceDirect 2023)](https://www.sciencedirect.com/science/article/pii/S2214845023000029)

**Empirical evidence from this project:**
- Sub-project #1: `analysis/edge_discovery_runs/2026-04-20/` — 90 rules, aggregate PF 1.36 on Discovery
- Sub-project #3: `analysis/edge_discovery_runs/2026-04-21-validation/` — F1+F2 generalizes OOS (+3.1% PF)
- 2025 OOS dataset: `20260421-134338_full/` — 97K trades, Stage 5b baseline PF 1.24

---

**End of design spec.**
