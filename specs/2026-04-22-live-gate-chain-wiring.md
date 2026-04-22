# Live Gate Chain Wiring — Design + Implementation Plan

**Date:** 2026-04-22
**Status:** Approved (verbal), ready to execute
**Branch:** `feat/premium-zone-ict-fix`
**Predecessors:** sub-projects #1 (Discovery+Validation), #2 (Conviction), #3 (Cross-sectional) — all shipped, all GAUNTLET-only today

---

## 1. Goal

Make the live decision pipeline (`python main.py [--dry-run|--paper-trading]`) actually filter candidates through the validated gate stack BEFORE they become orders. Today the gates only exist as offline gauntlet replay; production trade flow is unfiltered by ML.

This unblocks: (a) realistic backtests producing 20-50 trades/day, (b) holdout-period validation, (c) paper trading, (d) live deployment.

## 2. Why now

- Sub-project #4 (shadow loop) and any meaningful holdout backtest are blocked without this — they'd just paper-trade the unfiltered ~370 trades/day rather than the ML-gated ~20-50/day
- Gauntlet has served its purpose: validated the model's signal exists. It does NOT predict production behavior because production has different upstream filtering
- Several existing gates were found dead (`market_sentiment` is a no-op stub, `structure_manager` is never instantiated) — see audit findings in conversation 2026-04-22 + LW-1 task

## 3. Architecture

### 3.1 Composed gate chain

A new module `services/gate_chain/live_gate_chain.py` exposes a single class that owns lifecycle of three sub-gates. Single insertion point in the orchestrator. Lifecycle, config, error handling, and disable-flags all in one place.

```
Detector fires candidate (1500 stocks, ~2-min existing budget)
   │
   ▼
quality_filters at gate level (KEEP — load-bearing input distribution
                               for trained model)
   │
   ▼
TradeDecisionGate.evaluate()  (existing — structure_detector,
                                regime_gate disabled, event/news disabled)
   │
   ▼
Orchestrator.process_setup_candidates() ranks plans
   │
   ▼
LiveGateChain.evaluate(ranked_candidates)        ← NEW INSERTION POINT
   ├─ [A] RuleFilterGate
   │      Allow only candidates whose
   │      (setup_type, regime, cap_segment, hour_bucket) tuple matches
   │      one of the 74 validation-gate-surviving rules.
   │      Cost: O(1) hash lookup per candidate.
   │
   ├─ [B] CrossSectionalGate (existing services/cross_sectional/gate.py)
   │      F1 RVOL low-filter (cap-conditional)
   │      F2 Crowdedness low-filter (window-decayed counter)
   │      Cost: O(1) state lookups per candidate.
   │
   └─ [C] ConvictionGate (existing services/conviction/gate.py)
          XGBoost score (BATCHED across all surviving candidates)
          min_predicted_r threshold + 50/day session cap
          Cost: ~30ms one batch predict per bar; 0.05ms gate eval per cand.
   │
   ▼
order_queue.enqueue(admitted plans)
```

### 3.2 Insertion point details

**File:** `pipelines/orchestrator.py` — function `process_setup_candidates`
**Location:** AFTER existing ranking + risk/sizing checks, BEFORE eligible_plans is appended for enqueue
**Method signature:**

```python
admitted_plans = live_gate_chain.evaluate(ranked_eligible_plans)
```

Where `ranked_eligible_plans` is the list of `(candidate, plan)` tuples that survived all existing checks.

Returns: subset of input list with `plan.gate_decision` annotated (`admitted` | `rejected_reason`) for downstream logging.

### 3.3 State management

`LiveGateChain` owns ONE singleton instance per process, instantiated at `ScreenerLive.__init__`:

| Component | State | Reset trigger |
|---|---|---|
| `RuleFilterGate` | Stateless (just the 74-rule lookup table) | Never (loaded once at startup from `analysis/edge_discovery_runs/2026-04-22-validation-gate/stage6_validation_survivors.json`) |
| `UniverseRVOLState` | 30-session rolling window of per-symbol per-bar volumes | Continuous (warmed at backtest startup, persists in live) |
| `CrowdednessCounter` | per-setup time-decay counter | Per session (auto-reset on first candidate after session boundary) |
| `XGBoostScorer` | Loaded model + feature_spec | Never (loaded once at startup) |
| `ConvictionGate` | per-session admitted_today counter | Per session (auto-reset via `cand['session_date']` change) |

### 3.4 Backtest warmup

`UniverseRVOLState` requires ~30 sessions of prior bar volumes before its rank percentiles are meaningful. In live this happens organically (system runs continuously). In backtest mode, before the first trading bar of session N fires, RVOL state must be warmed from sessions [N-30, N-1].

**Hook:** `ScreenerLive` already loads historical OHLCV at backtest startup for indicator warmup. We add a warmup-feed call into `UniverseRVOLState.on_bar_close()` for each historical bar in the warmup window. Same I/O, no new data loads.

### 3.5 Batched scoring

`XGBoostScorer.predict()` currently takes one feature dict at a time. For live performance with 50-200 candidates per bar, a batched method is added:

```python
def predict_batch(self, feat_list: List[Dict[str, float]]) -> np.ndarray:
    """Vectorize predict over multiple candidates. Returns 1D array of predictions."""
    # Assemble (n_candidates, n_features) matrix in training order
    # Call self.model.predict(matrix) once
```

XGBoost natively predicts on a 2D matrix at sub-linear cost vs N individual calls. ~10x latency win for typical batch sizes.

### 3.6 Feature extraction at decision time

Detectors emit candidate context dicts containing the same features the gauntlet pulls from `trade_report.csv` (e.g., `pdz_confluence_count`, `pdz_range_position`, `bb_width_proxy`). The `extract_features()` helper from `services/conviction/feature_spec.py` is reused unchanged — guarantees live==backtest==gauntlet feature parity, zero drift risk.

Verification step in LW-7: log a sample of (candidate context dict → extracted features) at decision time and assert parity with what gauntlet's Stage 5d sees on the same trade.

### 3.7 Failure modes

| Mode | Behavior | Rationale |
|---|---|---|
| `conviction_gate.enabled=true` but model artifact missing | Raise at startup | Project rule: fail fast, no silent fallback |
| `cross_sectional_gate.enabled=true` but config keys missing | Raise at startup | Same |
| Survivor rules JSON missing | Raise at startup | Same |
| Per-candidate scorer crashes | Log + drop the single candidate (don't kill the bar) | Single bad input shouldn't take down trading; counter wraps the predict in try/except |
| Backtest warmup fails (insufficient history) | Run with cold RVOL state, log warning, mark first N sessions as "warmup" in trade_report | Don't hard-fail backtest; degraded operation is OK as long as it's flagged |

### 3.8 Latency budget

For worst case 1500 candidates per bar (every stock fires; never happens in reality):

| Step | Cost | % of 2-min existing budget |
|---|---|---|
| RVOL state update on bar close | ~10ms | ~0.008% |
| RuleFilterGate evaluate (1500 × O(1)) | ~30ms | ~0.025% |
| CrossSectional evaluate (1500 × O(1)) | ~150ms | ~0.13% |
| Feature extraction (per surviving candidate) | ~750ms | ~0.6% |
| Batched XGBoost predict | ~80ms | ~0.07% |
| ConvictionGate evaluate (1500 × O(1)) | ~75ms | ~0.06% |
| **Total worst case** | **~1.1s** | **~0.9%** |

Realistic case (50-200 candidates per bar that survive existing upstream filters): ~150ms.

**Doesn't move the needle vs the existing 2-min budget.** The 2-min cost is upstream per-stock detector work; gates are downstream on aggregated candidates and don't multiply with stock count.

## 4. Files affected

| Path | Action | Purpose |
|---|---|---|
| `services/gate_chain/__init__.py` | CREATE | Package marker |
| `services/gate_chain/live_gate_chain.py` | CREATE | Composed module |
| `services/gate_chain/rule_filter.py` | CREATE | Stage 5b live equivalent |
| `tests/gate_chain/__init__.py` | CREATE | Package marker |
| `tests/gate_chain/test_rule_filter.py` | CREATE | RuleFilterGate tests |
| `tests/gate_chain/test_live_gate_chain.py` | CREATE | Composed integration tests |
| `services/conviction/scorer.py` | MODIFY | Add `predict_batch()` |
| `tests/conviction/test_scorer.py` | MODIFY | Add `test_predict_batch_*` |
| `services/cross_sectional/universe_rvol.py` | MODIFY (maybe) | Add `is_warm()` introspection if missing |
| `services/screener_live.py` | MODIFY | Instantiate LiveGateChain at init; backtest warmup hook |
| `pipelines/orchestrator.py` | MODIFY | Insert LiveGateChain.evaluate() in process_setup_candidates |
| `config/configuration.json` | MODIFY | Add `live_gate_chain` enable block; disable `market_sentiment`; add `rule_filter_gate.survivors_path` |

## 5. Task plan

### LW-3: Build RuleFilterGate

**File:** `services/gate_chain/rule_filter.py`

```python
"""Rule filter — Stage 5b live equivalent.

Loads the validation-gate-surviving rules from JSON at startup. Each rule
specifies (setup_type, regime, cap_segment, hour_bucket) constraints. A
candidate is admitted iff its tuple matches at least one rule.

The 74 surviving rules are the trained-distribution input to the conviction
model. Trades not matching any rule are setups the conviction model has
never seen scored data for — drop them.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class RuleFilterGate:
    """Stateless filter on (setup_type, conditioner_keys, conditioner_values)."""

    def __init__(self, survivors_json_path: Path):
        self._survivor_set: Set[Tuple] = self._load(survivors_json_path)

    @staticmethod
    def _load(path: Path) -> Set[Tuple]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rules: Set[Tuple] = set()
        for s in data["survivors"]:
            rule_id = s["rule_id"]
            setup, cond_part = rule_id.split("__", 1)
            cond_key_part, cond_val_part = cond_part.split("=", 1)
            keys = tuple(cond_key_part.split("+"))
            vals = tuple(cond_val_part.split("+"))
            rules.add((setup, keys, vals))
        return rules

    def evaluate(self, cand: Dict[str, Any]) -> Tuple[bool, str]:
        """Return (allow, reason).

        cand must contain `setup_type` and the conditioner keys referenced
        by the survivor rules (cap_segment, regime, hour_bucket).
        """
        setup = cand.get("setup_type")
        for r_setup, r_keys, r_vals in self._survivor_set:
            if setup != r_setup:
                continue
            if all(cand.get(k) == v for k, v in zip(r_keys, r_vals)):
                return True, "matched_rule"
        return False, "no_matching_survivor_rule"
```

**Tests** (`tests/gate_chain/test_rule_filter.py`):
- `test_loads_74_rules_from_validation_survivors_json`
- `test_admits_candidate_matching_a_rule`
- `test_rejects_candidate_with_unknown_setup`
- `test_rejects_candidate_whose_conditioner_value_doesnt_match`
- `test_handles_missing_conditioner_key_gracefully` (returns False, not crash)

**Commit:** `feat(gate_chain): RuleFilterGate (Stage 5b live equivalent) (LW-3)`

---

### LW-4: XGBoostScorer.predict_batch()

**File:** `services/conviction/scorer.py` — MODIFY (add method)

```python
def predict_batch(self, feat_list: List[Dict[str, float]]) -> np.ndarray:
    """Vectorize predict over many feature dicts. Returns 1D array of predictions.

    Assembles a (len(feat_list), n_features) matrix in training feature order,
    calls XGBoost predict once, returns the resulting array. Empty input → empty array.
    """
    if not feat_list:
        return np.array([], dtype=np.float32)
    matrix = np.array(
        [[float(feat.get(f, 0.0)) for f in self.features] for feat in feat_list],
        dtype=np.float32,
    )
    return self.model.predict(matrix)
```

**Tests** (extend `tests/conviction/test_scorer.py`):
- `test_predict_batch_matches_per_call_predict_within_tolerance` (correctness vs single-call)
- `test_predict_batch_empty_input_returns_empty`
- `test_predict_batch_preserves_feature_order` (same dict insertion order test pattern as existing scorer test)

**Commit:** `feat(conviction): XGBoostScorer.predict_batch (LW-4)`

---

### LW-5: LiveGateChain composed module

**File:** `services/gate_chain/live_gate_chain.py`

```python
"""Live composed gate chain — RuleFilter -> CrossSectional -> Conviction.

Owns the lifecycle of all three sub-gates and their state. Singleton per
ScreenerLive process. Inserted in pipelines/orchestrator.py after existing
ranking, before order enqueue.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from services.cross_sectional.crowdedness_counter import CrowdednessCounter
from services.cross_sectional.universe_rvol import UniverseRVOLState
from services.cross_sectional.gate import CrossSectionalGate, Candidate as CSCand
from services.conviction.feature_spec import extract_features
from services.conviction.gate import ConvictionGate
from services.conviction.scorer import XGBoostScorer
from services.gate_chain.rule_filter import RuleFilterGate

log = logging.getLogger(__name__)


class LiveGateChain:
    """Composed three-stage gate. Stateful (RVOL + crowdedness counters)."""

    def __init__(self, full_cfg: Dict[str, Any], project_root: Path):
        self.cfg = full_cfg.get("live_gate_chain", {})
        self.enabled = bool(self.cfg.get("enabled", False))
        if not self.enabled:
            log.info("LiveGateChain disabled in config")
            return

        # [A] Rule filter
        rf_cfg = full_cfg.get("rule_filter_gate", {})
        survivors_path = project_root / rf_cfg.get(
            "survivors_path",
            "analysis/edge_discovery_runs/2026-04-22-validation-gate/stage6_validation_survivors.json",
        )
        self.rule_filter = RuleFilterGate(survivors_path)

        # [B] Cross-sectional
        cs_cfg = full_cfg.get("cross_sectional_gate", {})
        self.rvol = UniverseRVOLState(
            rolling_sessions=int(cs_cfg["f1_rolling_window_sessions"]),
            min_sessions=int(cs_cfg["f1_min_history_sessions"]),
        )
        self.crowdedness = CrowdednessCounter(
            window_min=int(cs_cfg["f2_crowdedness_window_min"])
        )
        self.cross_sectional = CrossSectionalGate(
            cs_cfg, rvol=self.rvol, crowdedness=self.crowdedness
        )

        # [C] Conviction
        cv_cfg = full_cfg.get("conviction_gate", {})
        self.scorer = XGBoostScorer(
            project_root / cv_cfg["model_artifact"],
            project_root / cv_cfg["feature_spec_path"],
        )
        self.conviction = ConvictionGate({
            "enabled": True,
            "daily_cap": int(cv_cfg["daily_cap"]),
            "min_predicted_r": float(cv_cfg["min_predicted_r"]),
        })

        # Stats
        self._stats = {"in": 0, "rule_drop": 0, "cs_drop": 0, "conv_drop": 0, "admitted": 0}

    def on_bar_close(self, bar_ts, bar_volumes: Dict[str, int], symbol_caps: Dict[str, str]) -> None:
        """Forward to UniverseRVOLState; called by ScreenerLive on each 5-min close."""
        if not self.enabled:
            return
        self.rvol.on_bar_close(ts=bar_ts, bar_volumes=bar_volumes, symbol_caps=symbol_caps)

    def evaluate(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates through 3-stage chain. Returns admitted subset.

        Each candidate dict must contain at minimum: symbol, setup_type, regime,
        cap_segment, hour_bucket, decision_ts, session_date_dt, plus the feature
        keys consumed by feature_spec.extract_features.

        Stages annotate each cand with `gate_reject_reason` if dropped.
        """
        if not self.enabled or not candidates:
            return candidates

        self._stats["in"] += len(candidates)

        # [A] Rule filter
        passed_a = []
        for c in candidates:
            ok, reason = self.rule_filter.evaluate(c)
            if ok:
                passed_a.append(c)
            else:
                c["gate_reject_reason"] = f"rule_filter:{reason}"
                self._stats["rule_drop"] += 1

        # [B] Cross-sectional
        passed_b = []
        for c in passed_a:
            cs_cand = CSCand(
                symbol=c["symbol"].replace("NSE:", ""),
                setup_type=c["setup_type"],
                cap_segment=c["cap_segment"],
                hour_bucket=c["hour_bucket"],
                decision_ts=c["decision_ts"],
            )
            ok, reason = self.cross_sectional.evaluate(cs_cand)
            if ok:
                passed_b.append(c)
            else:
                c["gate_reject_reason"] = f"cross_sectional:{reason}"
                self._stats["cs_drop"] += 1

        # [C] Conviction — BATCHED scoring then per-cand gate eval
        if passed_b:
            feat_list = [extract_features(c) for c in passed_b]
            preds = self.scorer.predict_batch(feat_list)
            admitted = []
            for c, pred in zip(passed_b, preds):
                cand_for_gate = {
                    "symbol": c["symbol"],
                    "decision_ts": c["decision_ts"],
                    "session_date": c["session_date_dt"],
                }
                ok, reason = self.conviction.evaluate(cand_for_gate, float(pred))
                if ok:
                    c["predicted_r"] = float(pred)
                    admitted.append(c)
                    self._stats["admitted"] += 1
                else:
                    c["gate_reject_reason"] = f"conviction:{reason}"
                    c["predicted_r"] = float(pred)
                    self._stats["conv_drop"] += 1
            return admitted

        return []

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)
```

**Tests** (`tests/gate_chain/test_live_gate_chain.py`):
- `test_disabled_chain_passes_all_through`
- `test_rule_filter_drops_unmatched_setup`
- `test_cross_sectional_drops_high_rvol_cap_segment` (with synthetic warmed RVOL)
- `test_conviction_caps_admitted_at_daily_cap` (with mock scorer)
- `test_chain_session_boundary_resets_caps`
- `test_chain_stats_count_correctly_at_each_stage`
- Integration test: 100 synthetic candidates → assert admitted count + stats consistent

**Commit:** `feat(gate_chain): LiveGateChain composed 3-stage filter (LW-5)`

---

### LW-6: RVOL warmup hook in ScreenerLive

**File:** `services/screener_live.py` — MODIFY

In the existing OHLCV indicator-warmup path (find via grep for "warmup"), add a callback that feeds each warmup bar's per-symbol volumes into `live_gate_chain.on_bar_close()`. This is read-only — no new I/O.

In backtest mode, the warmup window is `cs_cfg.f1_rolling_window_sessions` (currently 30) sessions. In live mode, the system warms organically over its first 30 trading sessions; first-30-day candidates flow through with cold RVOL state but still get conviction filtering.

**Tests:** integration-only — covered by LW-8 single-session backtest verification.

**Commit:** `feat(screener_live): warm UniverseRVOLState during backtest startup (LW-6)`

---

### LW-7: Wire LiveGateChain into orchestrator

**File:** `pipelines/orchestrator.py` — MODIFY

In `process_setup_candidates`, after the existing ranked-eligible-plans list is built but before items are appended for enqueue:

```python
if hasattr(self, "live_gate_chain") and self.live_gate_chain is not None and self.live_gate_chain.enabled:
    eligible_plans = self.live_gate_chain.evaluate(eligible_plans)
```

Where `self.live_gate_chain` is injected at orchestrator construction by `ScreenerLive`.

Also: in the per-bar `_on_5m_close` hook in screener_live.py, after assembling per-symbol bar volumes (which already happens for indicator updates), pass them to `live_gate_chain.on_bar_close(bar_ts, bar_vols, sym_caps)`.

**Tests:** integration via LW-8.

**Commit:** `feat(orchestrator): wire LiveGateChain into process_setup_candidates (LW-7)`

---

### LW-8: Single-session backtest verification

**Command:**
```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python main.py --dry-run --session-date 2025-01-20
```

**Expected:**
- Exits 0
- Trade count for the session is small (single-digit to low-double-digit; cap=50 means at most 50)
- Logs show `gate_reject_reason` distribution: most rejections at rule_filter or cross_sectional, fewer at conviction (cap likely doesn't bind on a single day)
- Per-candidate `predicted_r` is logged for surviving candidates

**Validation:**
- Compare admitted candidates' (setup_type, cap_segment, hour_bucket) to the 74-survivor list — every admitted candidate must match a rule
- Spot-check: load the same trade in the gauntlet's stage5d_simulation.json — does the live `predicted_r` match (within 1e-4) the gauntlet's predicted_r for the same trade_id? If yes, feature parity is confirmed.

**Commit:** `feat(verify): single-session backtest with LiveGateChain — N trades (LW-8)`

---

### LW-9: Multi-session backtest run

**Command:**
```bash
# Pick a representative span — Jan-Mar 2025 (~60 sessions) for first run
for d in 2025-01-{02..31} 2025-02-{03..28} 2025-03-{03..31}; do
    PYTHONIOENCODING=utf-8 .venv/Scripts/python main.py --dry-run --session-date "$d" 2>&1 | tail -3
done
```

(Or use the existing multi-session backtest harness if one exists — grep for batch/loop runner.)

**Expected output to capture:**
- Trade-count distribution across sessions
- Daily PnL series, total PnL, Sharpe, Win rate, PF — broken down per-session
- Average gate_reject_reason distribution
- Comparison table: live-backtest vs gauntlet projection (Stage 5b → 5c → 5d numbers from chained gauntlet)

If live numbers diverge materially from chained-gauntlet projection: investigate (likely candidates: feature drift, missing features at decision time, off-by-one in session boundary, RVOL cold-start issue).

**Commit (if successful):** `feat(verify): multi-session backtest with LiveGateChain — Jan-Mar 2025 (LW-9)`

---

## 6. Acceptance criteria

The work is DONE when:

- [ ] All unit tests pass (`pytest tests/gate_chain tests/conviction tests/cross_sectional`)
- [ ] `python main.py --dry-run --session-date 2025-01-20` exits 0 and produces ≤50 trades for that day
- [ ] Per-candidate `predicted_r` in live backtest matches gauntlet's predicted_r for the same `trade_id` within 1e-4 (feature parity)
- [ ] Multi-session backtest (Jan-Mar 2025) completes without errors
- [ ] Trade count and PF on the multi-session run are within ±25% of chained gauntlet projection (≈42 trades/day, PF≈1.38)
- [ ] `gate_reject_reason` distribution is logged and non-uniform (different stages dropping different fractions)
- [ ] No regression in existing tests (`pytest tests/` returns same prior pass count + new tests)

## 7. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Feature parity fails (live extracted features != gauntlet's) | LW-8 verification step compares predicted_r per trade_id; bug means a feature is missing at decision time vs `trade_report.csv` post-hoc |
| RVOL cold start drops too many candidates in first 30 days | Already handled: RVOL state has `min_sessions` floor; gate degrades to "admit" before warm-up complete |
| ConvictionGate session boundary edge case (date changes mid-session due to TZ) | All timestamps already IST-naive per project rule; session_date_dt is a date object, not datetime |
| Latency >120ms breaks real-time live execution | Worst-case 1.07s is still <1% of 2-min existing budget; profile if observed |
| Quality_filters split-brain bug surfaces inconsistency between live and backtest | Out of scope for this work; documented separately. As long as the ML-wired path consumes the SAME quality_filters config in live and backtest, behavior is consistent |
| Live trade flow diverges from gauntlet projection by >25% | LW-9 catches this. Investigate before any paper trading. |

## 8. Out of scope (deferred to future work)

- Fixing the `quality_filters` split-brain bug (`pipelines/base_config.json:96` vs `screener_live.py:568`) — separate cleanup
- Re-training the model on a TRULY wide-open backtest (with quality_filters disabled at gate level too) — experiment, not critical path
- Per-setup model deployment (T9 decided to ship universal only)
- Conviction-tiered or Kelly sizing — sub-project #5
- Live shadow-loop / parity monitoring — sub-project #4
- Holdout-period gauntlet (Oct 2025-Mar 2026) — needs OCI rebuild + run first
- Profiling the 2-min detector loop — separate work

## 9. Sequencing

```
LW-3 (RuleFilterGate)  ─┐
LW-4 (predict_batch)    ─┼─→ LW-5 (LiveGateChain composed) ─┐
                         │                                   ├─→ LW-7 (orchestrator wiring) ─→ LW-8 (single-session) ─→ LW-9 (multi-session)
                         └─→ LW-6 (RVOL warmup hook) ───────┘
```

LW-3 + LW-4 + LW-6 are independent and can be done in parallel. LW-5 depends on all three. LW-7 depends on LW-5. LW-8 and LW-9 are the verification steps.

Estimated active dev: ~4-5 hours. Backtest runtime: separate, attended.

## 10. Self-review

**Spec coverage:**
- Architecture defined ✓ — single composed module, insertion point, state ownership
- Files affected enumerated ✓ — 12 files, mostly creates
- Each task has full code or near-full code ✓
- Acceptance criteria are objective ✓
- Risks enumerated with mitigations ✓
- Latency budget proven ✓ (~1% of 2-min existing budget)
- Out-of-scope explicit ✓

**Type consistency:**
- `XGBoostScorer.predict_batch(feat_list: List[Dict[str, float]]) -> np.ndarray` — consistent with existing `predict(feat: Dict[str, float]) -> float`
- `LiveGateChain.evaluate(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]` — annotates input dicts with `gate_reject_reason` + `predicted_r`
- `RuleFilterGate.evaluate(cand: Dict) -> Tuple[bool, str]` — same shape as existing CrossSectionalGate.evaluate / ConvictionGate.evaluate

**Placeholder scan:** None.
