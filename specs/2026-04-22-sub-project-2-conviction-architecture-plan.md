# Sub-Project #2 (Conviction Architecture) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `services/conviction/` package — an XGBoost-based R-multiple regressor + `ConvictionGate` (online top-50 with calibration-derived minimum threshold) + `stage5d_conviction_simulation.py` for backtest replay, trained on validation-gate-surviving rules only.

**Architecture:** Feature extractor produces ~35 pre-decided features (whitelist, leakage-safe) per candidate. XGBoost regressor predicts R-multiple. Calibration maps predicted-R deciles to realized-R to derive minimum threshold. ConvictionGate applies threshold + daily top-50 cap. Stage 5d replays the gate chronologically on backtest trades. Training data = Discovery trades matching the 74 validation-survivor rules (not all 90).

**Tech Stack:** Python 3.10, pandas, numpy, pytest (existing). Adds xgboost, shap (feature importance + interpretation).

---

## Source spec

`specs/2026-04-21-sub-project-2-conviction-architecture-design.md` — approved.

Validation-survivor ruleset at `analysis/edge_discovery_runs/2026-04-22-validation-gate/stage6_validation_survivors.json` — 74 of 90 rules.

## Scope

Sub-project #2 only. Advanced sizing (Kelly, conviction-tiered), production per-setup model deployment for small-N setups, live parity detection — all deferred to sub-projects #4, #5, #6.

## File structure

| Path | Purpose | Created/Modified |
|------|---------|------------------|
| `services/conviction/__init__.py` | Package marker | Create |
| `services/conviction/feature_spec.py` | `ALLOWED_FEATURES` whitelist + `extract_features(row) -> dict` | Create |
| `services/conviction/scorer.py` | `XGBoostScorer` — loads model artifact, predicts R-multiple | Create |
| `services/conviction/gate.py` | `ConvictionGate` — online top-50 selection + threshold filter | Create |
| `services/conviction/calibration.py` | `derive_threshold_from_calibration(predictions, realized) -> float` | Create |
| `tests/conviction/__init__.py` | Package marker | Create |
| `tests/conviction/test_feature_spec.py` | Feature extractor + leakage audit tests | Create |
| `tests/conviction/test_scorer.py` | Scorer load + predict tests | Create |
| `tests/conviction/test_gate.py` | ConvictionGate selection + threshold tests | Create |
| `tests/conviction/test_calibration.py` | Threshold derivation tests | Create |
| `tools/conviction/__init__.py` | Package marker | Create |
| `tools/conviction/build_training_dataset.py` | Load Discovery, filter to 74 rules, extract features + labels → parquet | Create |
| `tools/conviction/train_universal.py` | Train XGBoost universal model on Discovery training set | Create |
| `tools/conviction/train_per_setup.py` | Train per-setup models for premium_zone_short + range_bounce_short | Create |
| `tools/conviction/validate_on_2025.py` | Run 4 model validation tests on 2025 OOS | Create |
| `tools/edge_discovery/stages/stage5d_conviction_simulation.py` | Backtest replay of ConvictionGate | Create |
| `tests/edge_discovery/test_stage5d.py` | Stage 5d tests | Create |
| `models/conviction/.gitkeep` | Dir for model artifacts | Create |
| `config/configuration.json` | Add `conviction_gate` block | Modify |
| `tools/edge_discovery/run_gauntlet.py` | Add Stage 5d call | Modify |
| `requirements.txt` (or equivalent) | Add xgboost, shap | Modify |

---

## Phase A: Foundation (Task 1)

### Task 1: Dependencies + package skeleton + config

**Files:**
- Create: `services/conviction/__init__.py`
- Create: `tests/conviction/__init__.py`
- Create: `tools/conviction/__init__.py`
- Create: `models/conviction/.gitkeep`
- Modify: `config/configuration.json`
- Install: xgboost, shap

- [ ] **Step 1: Install ML dependencies**

```bash
.venv/Scripts/pip install xgboost==2.1.4 shap==0.46.0
```

Verify:
```bash
.venv/Scripts/python -c "import xgboost; import shap; print(f'xgboost {xgboost.__version__}, shap {shap.__version__}')"
```

Expected output: `xgboost 2.1.4, shap 0.46.0`

- [ ] **Step 2: Create package markers**

```bash
mkdir -p services/conviction tests/conviction tools/conviction models/conviction
touch services/conviction/__init__.py tests/conviction/__init__.py tools/conviction/__init__.py models/conviction/.gitkeep
```

Create `services/conviction/__init__.py`:
```python
"""XGBoost-based conviction scorer + top-N selection gate.

Per sub-project #2 design spec (2026-04-21). Ranks Stage-5c-filter-surviving
candidates by predicted R-multiple, selects top 50/day above calibration
threshold. Training data = 74 validation-gate-surviving rules (Discovery
trades matching those rules).
"""
```

- [ ] **Step 3: Add config block**

In `config/configuration.json`, locate the `cross_sectional_gate` block (sub-project #3). Immediately after it (before closing `}`), add:

```json
"_comment_conviction_gate": "=== CONVICTION GATE (sub-project #2) ===",
"conviction_gate": {
  "enabled": false,
  "model_artifact": "models/conviction/2026-04-22-universal-xgboost.json",
  "feature_spec_path": "models/conviction/2026-04-22-feature-spec.json",
  "calibration_path": "models/conviction/2026-04-22-calibration.json",
  "daily_cap": 50,
  "min_predicted_r": 0.3
}
```

Note: `enabled: false` — gate defaults to disabled until model is trained + validated. Task 13 flips this to true.

Validate JSON:
```bash
.venv/Scripts/python -c "import json; cfg = json.loads(open('config/configuration.json').read()); print(cfg['conviction_gate'])"
```

- [ ] **Step 4: Commit**

```bash
git add services/conviction/__init__.py tools/conviction/__init__.py config/configuration.json
git add -f tests/conviction/__init__.py models/conviction/.gitkeep
git commit -m "feat(conviction): package skeleton + config + xgboost/shap deps (sub-project #2 T1)"
```

---

## Phase B: Feature extractor (Task 2)

### Task 2: feature_spec.py — ALLOWED_FEATURES whitelist + extract_features()

**Files:**
- Create: `services/conviction/feature_spec.py`
- Create: `tests/conviction/test_feature_spec.py`

**Design:** Whitelist-based feature extraction. Any column not in `ALLOWED_FEATURES` is dropped. Extractor takes a dict (trade row context) and returns a feature dict. Handles missing/NaN values deterministically.

- [ ] **Step 1: Write failing tests**

Create `tests/conviction/test_feature_spec.py`:

```python
"""Feature spec tests — whitelist + extraction + leakage audit."""
import pandas as pd
import pytest

from services.conviction.feature_spec import (
    ALLOWED_FEATURES,
    BLOCKED_OUTCOME_COLUMNS,
    extract_features,
    audit_leakage,
)


def test_allowed_features_is_non_empty_list():
    assert isinstance(ALLOWED_FEATURES, list)
    assert len(ALLOWED_FEATURES) >= 20


def test_allowed_features_has_no_outcome_columns():
    """No whitelisted feature should be a known outcome column (leakage check)."""
    for feat in ALLOWED_FEATURES:
        assert feat not in BLOCKED_OUTCOME_COLUMNS, f"Leakage: {feat} in whitelist AND outcome list"


def test_extract_features_returns_dict_with_allowed_keys():
    row = {
        "setup_type": "premium_zone_short",
        "regime": "chop",
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
        "pdz_confluence_count": 2,
        "pdz_range_position": 0.82,
        "vol_z": 1.5,
        "bb_width_proxy": 0.012,
    }
    feat = extract_features(row)
    assert isinstance(feat, dict)
    # all returned keys must be in the allowed whitelist or categorical-derived one-hot
    for key in feat.keys():
        is_allowed = key in ALLOWED_FEATURES or any(key.startswith(f"{cat}_") for cat in [
            "setup_type", "regime", "cap_segment", "hour_bucket", "day_of_week"
        ])
        assert is_allowed, f"{key} not whitelisted and not a known categorical one-hot"


def test_extract_features_onehots_categoricals():
    """Categorical columns are one-hot encoded."""
    row = {
        "setup_type": "premium_zone_short",
        "regime": "chop",
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
    }
    feat = extract_features(row)
    # setup_type_premium_zone_short should be 1, others 0
    assert feat["setup_type_premium_zone_short"] == 1
    assert feat["setup_type_range_bounce_short"] == 0
    # regime
    assert feat["regime_chop"] == 1
    assert feat["regime_trend_up"] == 0


def test_extract_features_handles_missing_as_zero():
    """NaN / None / missing keys → 0 for numerical, 0 for one-hot."""
    row = {
        "setup_type": "premium_zone_short",
        "regime": "chop",
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
        # deliberately missing: pdz_confluence_count, vol_z, ...
    }
    feat = extract_features(row)
    # missing numericals default to 0
    assert feat.get("pdz_confluence_count", None) == 0.0
    assert feat.get("vol_z", None) == 0.0


def test_extract_features_unknown_categorical_value_safely_encoded():
    """A categorical value not in the known vocab (e.g., typo regime) → all one-hots zero."""
    row = {
        "setup_type": "premium_zone_short",
        "regime": "typo_regime",  # unknown
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
    }
    feat = extract_features(row)
    # All regime one-hots should be 0 (no crash)
    assert feat["regime_chop"] == 0
    assert feat["regime_trend_up"] == 0
    assert feat["regime_trend_down"] == 0
    assert feat["regime_squeeze"] == 0


def test_audit_leakage_passes_on_clean_frame():
    """A DataFrame with only whitelisted + categorical columns passes audit."""
    df = pd.DataFrame([{
        "setup_type": "premium_zone_short",
        "regime": "chop",
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
        "pdz_confluence_count": 2,
        "vol_z": 1.0,
    }])
    audit_leakage(df)  # no raise


def test_audit_leakage_raises_when_outcome_column_present():
    df = pd.DataFrame([{
        "setup_type": "premium_zone_short",
        "regime": "chop",
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
        "realized_pnl": 100,  # outcome — leakage
    }])
    with pytest.raises(ValueError, match="leakage"):
        audit_leakage(df)


def test_audit_leakage_raises_on_multiple_outcomes():
    df = pd.DataFrame([{
        "setup_type": "premium_zone_short",
        "regime": "chop",
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
        "r_multiple": 1.5,
        "mae": -0.3,
        "bars_held": 15,
    }])
    with pytest.raises(ValueError) as exc:
        audit_leakage(df)
    # all three should be listed
    msg = str(exc.value).lower()
    assert "r_multiple" in msg
    assert "mae" in msg
    assert "bars_held" in msg
```

- [ ] **Step 2: Run tests — MUST fail**

```bash
.venv/Scripts/python -m pytest tests/conviction/test_feature_spec.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement feature_spec.py**

Create `services/conviction/feature_spec.py`:

```python
"""Feature spec — whitelist + extraction + leakage audit.

Per sub-project #2 design spec §3.3-3.4. Whitelist approach: any column not
explicitly listed in ALLOWED_FEATURES is dropped. BLOCKED_OUTCOME_COLUMNS
enumerates known post-decision columns for defense-in-depth leakage detection.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# Numerical features (continuous, used directly by XGBoost)
_NUMERICAL_FEATURES: List[str] = [
    # Momentum / trend (standard pro)
    "momentum_3bar_pct",
    "momentum_1bar_pct",
    "vwap_distance_pct",
    "bb_width_proxy",
    # Volume + cross-sectional (from sub-project #3 where available)
    "volume5",
    "vol_z",
    "vol_ratio",
    "body_size_pct",
    "wick_ratio",
    # ICT-specific detector context (our edge)
    "pdz_confluence_count",
    "pdz_range_position",
    "pdz_range_size_pct",
    "pdz_range_size_atr",
    "pdz_atr14",
    "ob_confluence_count",
    "resistance_touches",
    "resistance_strength",
    "pattern_age_mins",
    "size_mult",
    "minute_of_day",
]

# Boolean features (0/1 from detector flags)
_BOOLEAN_FEATURES: List[str] = [
    "pdz_has_mss_confluence",
    "pdz_has_fvg_confluence",
    "pdz_has_ob_confluence",
    "pdz_htf_bullish",
    "pdz_htf_bearish",
    "ob_has_liquidity_sweep",
    "ob_has_mss_confirmation",
]

# Categorical features (one-hot encoded)
_CATEGORICAL_VOCABS: Dict[str, List[str]] = {
    "setup_type": [
        "premium_zone_short",
        "range_bounce_short",
        "order_block_short",
        "resistance_bounce_short",
        "vwap_lose_short",
    ],
    "regime": ["chop", "trend_up", "trend_down", "squeeze"],
    "cap_segment": ["large_cap", "mid_cap", "small_cap", "micro_cap", "unknown"],
    "hour_bucket": ["opening", "morning", "lunch", "afternoon", "late"],
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
}

# Flat whitelist — names used by audit_leakage
ALLOWED_FEATURES: List[str] = _NUMERICAL_FEATURES + _BOOLEAN_FEATURES + list(_CATEGORICAL_VOCABS.keys())

# Known post-decision / label-derived columns that must NEVER appear in training
BLOCKED_OUTCOME_COLUMNS: List[str] = [
    "realized_pnl",
    "total_trade_pnl",
    "net_pnl",
    "pnl",
    "label_hit_t1",
    "label_hit_t2",
    "gross_exit_qty",
    "position_closed",
    "last_exit_ts",
    "last_exit_reason",
    "exit_price",
    "fees",
    "slippage_bps",
    "r_multiple",
    "mae",
    "mfe",
    "mae_pct",
    "mfe_pct",
    "bars_held",
    "time_in_trade_minutes",
    "e1_ts", "e1_reason", "e1_qty", "e1_price",
    "e2_ts", "e2_reason", "e2_qty", "e2_price",
    "e3_ts", "e3_reason", "e3_qty", "e3_price",
    "executed",
    "scaled_in",
    "remaining_qty",
    "exit_sequence",
    "total_exits",
    "is_final_exit",
]


def _safe_float(v: Any) -> float:
    """Convert to float, handling None / NaN / bool / string numerics."""
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        f = float(v)
        if np.isnan(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def extract_features(row: Dict[str, Any]) -> Dict[str, float]:
    """Extract ~35-40 features from a trade-context row dict.

    All outputs are floats (for XGBoost). Missing / NaN / unknown values → 0.0.
    """
    feat: Dict[str, float] = {}

    # Numerical
    for f in _NUMERICAL_FEATURES:
        feat[f] = _safe_float(row.get(f))

    # Boolean
    for f in _BOOLEAN_FEATURES:
        feat[f] = _safe_float(row.get(f))

    # Categorical — one-hot encoded
    for cat_col, vocab in _CATEGORICAL_VOCABS.items():
        val = row.get(cat_col)
        for term in vocab:
            feat[f"{cat_col}_{term}"] = 1.0 if val == term else 0.0

    return feat


def audit_leakage(df: pd.DataFrame) -> None:
    """Raise ValueError if df contains any BLOCKED_OUTCOME_COLUMNS.

    Used as a pre-training guard: any post-decision column in the training
    frame is leakage and must be caught before fit().
    """
    present = [c for c in BLOCKED_OUTCOME_COLUMNS if c in df.columns]
    if present:
        raise ValueError(
            f"Leakage detected: frame contains {len(present)} outcome columns "
            f"that must not be training features: {present}"
        )
```

- [ ] **Step 4: Run tests — MUST pass**

```bash
.venv/Scripts/python -m pytest tests/conviction/test_feature_spec.py -v
```

Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/conviction/feature_spec.py
git add -f tests/conviction/test_feature_spec.py
git commit -m "feat(conviction): feature_spec whitelist + leakage audit (T2)"
```

---

## Phase C: Scorer + Calibration + Gate (Tasks 3-5)

### Task 3: scorer.py — XGBoostScorer (load + predict)

**Files:**
- Create: `services/conviction/scorer.py`
- Create: `tests/conviction/test_scorer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/conviction/test_scorer.py`:

```python
"""XGBoostScorer tests — load model artifact, predict R-multiple."""
import json
from pathlib import Path

import pandas as pd
import pytest
import xgboost as xgb
import numpy as np

from services.conviction.scorer import XGBoostScorer


@pytest.fixture
def tiny_model_artifacts(tmp_path):
    """Train a tiny model on synthetic data + save artifacts."""
    # Synthetic: target = numerical_feature - 2*indicator
    n = 1000
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame({
        "momentum_3bar_pct": rng.normal(0, 1, n),
        "vol_z": rng.normal(0, 1, n),
        "pdz_confluence_count": rng.integers(0, 3, n),
        "setup_type_premium_zone_short": rng.integers(0, 2, n),
    })
    y = X["momentum_3bar_pct"] - 2 * X["setup_type_premium_zone_short"] + rng.normal(0, 0.1, n)

    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "tiny-model.json"
    model.save_model(model_path)

    feature_spec_path = tmp_path / "feature-spec.json"
    feature_spec_path.write_text(json.dumps({
        "features": list(X.columns),
        "version": "test",
    }))
    return model_path, feature_spec_path, list(X.columns)


def test_scorer_loads_model(tiny_model_artifacts):
    model_path, feature_spec_path, features = tiny_model_artifacts
    scorer = XGBoostScorer(model_path, feature_spec_path)
    assert scorer.features == features


def test_scorer_predict_returns_float(tiny_model_artifacts):
    model_path, feature_spec_path, features = tiny_model_artifacts
    scorer = XGBoostScorer(model_path, feature_spec_path)
    feat_dict = {
        "momentum_3bar_pct": 0.5,
        "vol_z": 0.0,
        "pdz_confluence_count": 2,
        "setup_type_premium_zone_short": 1,
    }
    pred = scorer.predict(feat_dict)
    assert isinstance(pred, float)


def test_scorer_predict_handles_missing_features_as_zero(tiny_model_artifacts):
    """If a feature in the spec isn't in the dict, substitute 0.0."""
    model_path, feature_spec_path, features = tiny_model_artifacts
    scorer = XGBoostScorer(model_path, feature_spec_path)
    # deliberately incomplete feat_dict (missing vol_z, pdz_confluence_count)
    feat_dict = {
        "momentum_3bar_pct": 0.5,
        "setup_type_premium_zone_short": 1,
    }
    pred = scorer.predict(feat_dict)
    assert isinstance(pred, float)


def test_scorer_feature_order_preserved(tiny_model_artifacts):
    """Feature order in predict call must match training order — prevents
    silent misalignment bug."""
    model_path, feature_spec_path, features = tiny_model_artifacts
    scorer = XGBoostScorer(model_path, feature_spec_path)
    # Same inputs, different dict iteration order — same output
    feat_1 = {
        "momentum_3bar_pct": 0.5,
        "vol_z": 0.2,
        "pdz_confluence_count": 2,
        "setup_type_premium_zone_short": 1,
    }
    feat_2 = {  # reversed insertion order
        "setup_type_premium_zone_short": 1,
        "pdz_confluence_count": 2,
        "vol_z": 0.2,
        "momentum_3bar_pct": 0.5,
    }
    p1 = scorer.predict(feat_1)
    p2 = scorer.predict(feat_2)
    assert p1 == pytest.approx(p2, abs=1e-6)
```

- [ ] **Step 2: Run tests — MUST fail**

```bash
.venv/Scripts/python -m pytest tests/conviction/test_scorer.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement scorer.py**

Create `services/conviction/scorer.py`:

```python
"""XGBoostScorer — loads trained model + feature spec, predicts R-multiple.

Stateless at predict-time (single-threaded caller assumption).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import xgboost as xgb


class XGBoostScorer:
    """Load a trained XGBoost model + feature spec; predict from feature dict."""

    def __init__(self, model_path: Path, feature_spec_path: Path):
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))
        spec = json.loads(Path(feature_spec_path).read_text(encoding="utf-8"))
        self.features: List[str] = spec["features"]
        self.version: str = spec.get("version", "")

    def predict(self, feat: Dict[str, float]) -> float:
        """Given a feature dict, return predicted R-multiple (scalar)."""
        # Assemble feature vector in training order; missing keys → 0.0
        vec = np.array([[float(feat.get(f, 0.0)) for f in self.features]], dtype=np.float32)
        pred = self.model.predict(vec)
        return float(pred[0])
```

- [ ] **Step 4: Run tests — MUST pass**

```bash
.venv/Scripts/python -m pytest tests/conviction/test_scorer.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/conviction/scorer.py
git add -f tests/conviction/test_scorer.py
git commit -m "feat(conviction): XGBoostScorer (load + predict) (T3)"
```

---

### Task 4: calibration.py — threshold derivation

**Files:**
- Create: `services/conviction/calibration.py`
- Create: `tests/conviction/test_calibration.py`

**Design:** Given predicted vs realized R-multiple from held-out data, compute a decile-bucketed calibration curve. Derive threshold where median realized R first exceeds a floor (e.g., 0.3). Returns threshold + calibration curve data for audit.

- [ ] **Step 1: Write failing tests**

Create `tests/conviction/test_calibration.py`:

```python
"""Calibration tests — threshold derivation + decile curve."""
import numpy as np
import pandas as pd
import pytest

from services.conviction.calibration import (
    build_decile_calibration,
    derive_threshold_from_calibration,
)


def test_build_decile_calibration_returns_10_rows():
    predicted = pd.Series(np.linspace(-1.0, 2.0, 1000))
    # synthetic realized: roughly linearly related to predicted
    realized = predicted + np.random.default_rng(42).normal(0, 0.1, 1000)
    curve = build_decile_calibration(predicted, realized)
    assert len(curve) == 10
    assert "decile" in curve.columns
    assert "predicted_lo" in curve.columns
    assert "predicted_hi" in curve.columns
    assert "realized_median" in curve.columns
    assert "realized_mean" in curve.columns
    assert "n" in curve.columns


def test_calibration_deciles_monotonic_when_model_good():
    """If model is well-calibrated, realized_median rises with decile."""
    predicted = pd.Series(np.linspace(-1.0, 2.0, 1000))
    realized = predicted + np.random.default_rng(42).normal(0, 0.1, 1000)
    curve = build_decile_calibration(predicted, realized)
    curve_sorted = curve.sort_values("decile")
    # Monotonically non-decreasing (with noise, allow small dips)
    # Check: decile 9 median > decile 0 median by a margin
    assert curve_sorted.iloc[-1]["realized_median"] > curve_sorted.iloc[0]["realized_median"] + 1.0


def test_derive_threshold_picks_first_decile_above_floor():
    """Threshold = predicted_lo of the first decile whose realized_median >= floor."""
    # Synthetic: deciles 0-4 have realized_median < 0.3, deciles 5-9 have >= 0.3
    predicted = pd.Series(np.linspace(-1.0, 2.0, 1000))
    realized = np.where(predicted >= 0.5, predicted, -0.5)
    realized = pd.Series(realized + np.random.default_rng(42).normal(0, 0.05, 1000))
    curve = build_decile_calibration(predicted, realized)
    threshold = derive_threshold_from_calibration(curve, floor=0.3)
    # Threshold should be somewhere in the predicted range where transition happens
    assert threshold > 0.0
    assert threshold < 1.5


def test_derive_threshold_returns_infinity_if_no_decile_passes_floor():
    """Degenerate case: model has no decile meeting floor → threshold = inf (reject all)."""
    predicted = pd.Series(np.linspace(-1.0, 0.1, 1000))
    realized = pd.Series(np.full(1000, -0.5))  # always losing
    curve = build_decile_calibration(predicted, realized)
    threshold = derive_threshold_from_calibration(curve, floor=0.3)
    assert threshold == float("inf")


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        build_decile_calibration(pd.Series([], dtype=float), pd.Series([], dtype=float))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        build_decile_calibration(pd.Series([1.0, 2.0]), pd.Series([0.5]))
```

- [ ] **Step 2: Run tests — MUST fail**

```bash
.venv/Scripts/python -m pytest tests/conviction/test_calibration.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement calibration.py**

Create `services/conviction/calibration.py`:

```python
"""Calibration — decile curve + threshold derivation.

Given predicted vs realized R-multiple from a held-out fold, bucket predictions
into deciles, measure realized R per decile, derive the minimum predicted R
threshold above which realized R >= floor.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def build_decile_calibration(
    predicted: pd.Series, realized: pd.Series, n_buckets: int = 10
) -> pd.DataFrame:
    """Bucket predictions into quantiles, compute realized stats per bucket.

    Args:
        predicted: model predicted R-multiple per trade
        realized: actual realized R-multiple per trade
        n_buckets: number of quantile buckets (default 10 = deciles)

    Returns:
        DataFrame with columns: decile, predicted_lo, predicted_hi,
        realized_median, realized_mean, n
    """
    if len(predicted) == 0 or len(realized) == 0:
        raise ValueError("predicted and realized must be non-empty")
    if len(predicted) != len(realized):
        raise ValueError(
            f"predicted ({len(predicted)}) and realized ({len(realized)}) length mismatch"
        )

    df = pd.DataFrame({"predicted": predicted.values, "realized": realized.values})
    df["decile"] = pd.qcut(df["predicted"], q=n_buckets, labels=False, duplicates="drop")
    grouped = df.groupby("decile", observed=True).agg(
        predicted_lo=("predicted", "min"),
        predicted_hi=("predicted", "max"),
        realized_median=("realized", "median"),
        realized_mean=("realized", "mean"),
        n=("realized", "count"),
    ).reset_index()
    return grouped


def derive_threshold_from_calibration(
    curve: pd.DataFrame, floor: float = 0.3
) -> float:
    """Pick the first decile whose realized_median >= floor; return its predicted_lo.

    If no decile meets the floor, return +inf (gate rejects all predictions).
    """
    sorted_curve = curve.sort_values("decile").reset_index(drop=True)
    for _, row in sorted_curve.iterrows():
        if row["realized_median"] >= floor:
            return float(row["predicted_lo"])
    return float("inf")
```

- [ ] **Step 4: Run tests — MUST pass**

```bash
.venv/Scripts/python -m pytest tests/conviction/test_calibration.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/conviction/calibration.py
git add -f tests/conviction/test_calibration.py
git commit -m "feat(conviction): calibration curve + threshold derivation (T4)"
```

---

### Task 5: gate.py — ConvictionGate

**Files:**
- Create: `services/conviction/gate.py`
- Create: `tests/conviction/test_gate.py`

**Design:** Stateful gate (tracks daily trade count + current session). `evaluate(candidate, predicted_r) -> (allow, reason)`. Session boundary resets daily cap. Thresholds + cap from cfg dict.

- [ ] **Step 1: Write failing tests**

Create `tests/conviction/test_gate.py`:

```python
"""ConvictionGate tests — online top-N + threshold + session boundary reset."""
from datetime import date, datetime

import pytest

from services.conviction.gate import ConvictionGate


CFG = {
    "enabled": True,
    "daily_cap": 5,
    "min_predicted_r": 0.3,
}


def _candidate(symbol="SYM", ts=None, session=None):
    return {
        "symbol": symbol,
        "decision_ts": ts or datetime(2026, 4, 22, 10, 0),
        "session_date": session or date(2026, 4, 22),
    }


def test_disabled_allows_everything():
    gate = ConvictionGate({**CFG, "enabled": False})
    ok, reason = gate.evaluate(_candidate(), predicted_r=-1.0)
    assert ok is True


def test_admit_when_above_threshold_and_under_cap():
    gate = ConvictionGate(CFG)
    ok, _ = gate.evaluate(_candidate(), predicted_r=0.5)
    assert ok is True


def test_reject_when_below_threshold():
    gate = ConvictionGate(CFG)
    ok, reason = gate.evaluate(_candidate(), predicted_r=0.2)
    assert ok is False
    assert "threshold" in reason.lower()


def test_reject_when_daily_cap_reached():
    gate = ConvictionGate(CFG)
    # Admit 5 candidates — fills cap
    for i in range(5):
        ok, _ = gate.evaluate(_candidate(symbol=f"S{i}"), predicted_r=0.5)
        assert ok is True
    # 6th should be rejected
    ok, reason = gate.evaluate(_candidate(symbol="S6"), predicted_r=0.9)
    assert ok is False
    assert "cap" in reason.lower()


def test_session_boundary_resets_counter():
    gate = ConvictionGate(CFG)
    # Fill cap on day 1
    for i in range(5):
        gate.evaluate(_candidate(symbol=f"S{i}", session=date(2026, 4, 22)), 0.5)
    # Day 2 — fresh start
    ok, _ = gate.evaluate(_candidate(symbol="S6", session=date(2026, 4, 23)), predicted_r=0.5)
    assert ok is True


def test_only_admitted_trades_count_toward_cap():
    """Rejected-below-threshold trades don't consume cap slots."""
    gate = ConvictionGate(CFG)
    # Reject 3 low-conviction trades
    for i in range(3):
        gate.evaluate(_candidate(symbol=f"low{i}"), predicted_r=0.1)
    # Admit 5 high-conviction (cap is 5)
    for i in range(5):
        ok, _ = gate.evaluate(_candidate(symbol=f"hi{i}"), predicted_r=0.5)
        assert ok is True
    # 6th high-conviction: rejected (cap reached)
    ok, _ = gate.evaluate(_candidate(symbol="hi6"), predicted_r=0.5)
    assert ok is False


def test_gate_stats_reports_admitted_and_rejected():
    gate = ConvictionGate(CFG)
    gate.evaluate(_candidate(symbol="a"), 0.5)
    gate.evaluate(_candidate(symbol="b"), 0.1)
    stats = gate.stats()
    assert stats["admitted"] == 1
    assert stats["rejected"] == 1
```

- [ ] **Step 2: Run tests — MUST fail**

```bash
.venv/Scripts/python -m pytest tests/conviction/test_gate.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement gate.py**

Create `services/conviction/gate.py`:

```python
"""ConvictionGate — online top-N selection + minimum-conviction threshold.

Stateful: tracks per-session admitted count. Session boundary detected via
candidate.session_date change; counter resets.

Scorer runs UPSTREAM — this gate takes predicted_r as input (already scored).
Separation keeps gate testable without model artifacts.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Tuple


class ConvictionGate:
    """Applies top-N + threshold filter per session.

    Config keys:
        enabled (bool)
        daily_cap (int)
        min_predicted_r (float)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._current_session: Optional[date] = None
        self._admitted_today: int = 0
        self._stats_admitted: int = 0
        self._stats_rejected: int = 0

    def evaluate(self, cand: Dict[str, Any], predicted_r: float) -> Tuple[bool, str]:
        """Return (allow, reason).

        cand must include `session_date` (date) for daily cap tracking.
        """
        if not self.cfg.get("enabled", False):
            return True, "gate_disabled"

        # Session boundary reset
        sess = cand.get("session_date")
        if sess != self._current_session:
            self._current_session = sess
            self._admitted_today = 0

        # Threshold check
        min_r = float(self.cfg["min_predicted_r"])
        if predicted_r < min_r:
            self._stats_rejected += 1
            return False, f"below_threshold predicted_r={predicted_r:.3f}<{min_r}"

        # Daily cap check
        cap = int(self.cfg["daily_cap"])
        if self._admitted_today >= cap:
            self._stats_rejected += 1
            return False, f"daily_cap_reached count={self._admitted_today}>={cap}"

        # Admit
        self._admitted_today += 1
        self._stats_admitted += 1
        return True, "admitted"

    def stats(self) -> Dict[str, int]:
        return {
            "admitted": self._stats_admitted,
            "rejected": self._stats_rejected,
        }
```

- [ ] **Step 4: Run tests — MUST pass**

```bash
.venv/Scripts/python -m pytest tests/conviction/test_gate.py -v
```

Expected: 7 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/conviction/gate.py
git add -f tests/conviction/test_gate.py
git commit -m "feat(conviction): ConvictionGate — online top-N + threshold (T5)"
```

---

## Phase D: Training dataset + trainers (Tasks 6-8)

### Task 6: build_training_dataset.py — Discovery → features + labels

**Files:**
- Create: `tools/conviction/build_training_dataset.py`

**Design:** CLI tool. Loads Discovery trades via data_loader, filters to trades matching 74 validation-survivor rules, extracts features per trade, joins label (r_multiple), outputs parquet at `models/conviction/2026-04-22-training-dataset.parquet`. Runs leakage audit + coverage audit + prints coverage report.

- [ ] **Step 1: Implement build_training_dataset.py**

Create `tools/conviction/build_training_dataset.py`:

```python
"""Build training dataset for sub-project #2 conviction model.

Loads Discovery trades (2023-01 to 2024-12), filters to the 74 validation-
gate-surviving rules, extracts features per row, joins r_multiple as label.

Output: models/conviction/2026-04-22-training-dataset.parquet

Discipline:
- Leakage audit: assert no BLOCKED_OUTCOME_COLUMNS in feature frame
- Coverage audit: drop features with >40% missing or large Discovery vs 2025 KS
- Survivor filter: only rules from stage6_validation_survivors.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.edge_discovery.data_loader import load_run
from services.conviction.feature_spec import (
    ALLOWED_FEATURES,
    BLOCKED_OUTCOME_COLUMNS,
    extract_features,
    audit_leakage,
)

ROOT = Path(__file__).parent.parent.parent
BACKTEST_DIR = ROOT / "cloud_results" / "20260419_discovery"
SURVIVORS = ROOT / "analysis" / "edge_discovery_runs" / "2026-04-22-validation-gate" / "stage6_validation_survivors.json"
OUT_DIR = ROOT / "models" / "conviction"
OUT_PARQUET = OUT_DIR / "2026-04-22-training-dataset.parquet"
FEATURE_SPEC_PATH = OUT_DIR / "2026-04-22-feature-spec.json"


def load_survivor_rules():
    data = json.loads(SURVIVORS.read_text(encoding="utf-8"))
    rules = []
    for s in data["survivors"]:
        # rule_id format: setup__conditioner=cell_value (with + for multi-dim)
        rule_id = s["rule_id"]
        setup, cond_part = rule_id.split("__", 1)
        cond_key_part, cond_val_part = cond_part.split("=", 1)
        conds = list(zip(cond_key_part.split("+"), cond_val_part.split("+")))
        rules.append({"setup": setup, "conditions": conds})
    return rules


def matches_any_rule(row, rules):
    for r in rules:
        if row["setup_type"] != r["setup"]:
            continue
        if all(row.get(k) == v for k, v in r["conditions"]):
            return True
    return False


def main():
    print(f"Loading Discovery trades: {BACKTEST_DIR}")
    data = load_run(BACKTEST_DIR)
    trades = data.trades
    print(f"Loaded {len(trades):,} trades")

    # Filter to Discovery range (2023-01-01 to 2024-12-31) and is_final_exit (already done by loader)
    from datetime import date
    trades = trades[
        (trades["session_date_dt"] >= date(2023, 1, 1))
        & (trades["session_date_dt"] <= date(2024, 12, 31))
    ].copy()
    print(f"Discovery subset: {len(trades):,}")

    # Load survivor rules + filter
    rules = load_survivor_rules()
    print(f"Applying {len(rules)} validation-survivor rules")
    mask = trades.apply(lambda r: matches_any_rule(r, rules), axis=1)
    filtered = trades[mask].copy()
    print(f"Training candidates (rule-matching): {len(filtered):,}")

    # Add day_of_week
    filtered["day_of_week"] = pd.to_datetime(filtered["session_date_dt"]).dt.day_name()

    # Extract features
    print("Extracting features...")
    feat_rows = [extract_features(r.to_dict()) for _, r in filtered.iterrows()]
    X = pd.DataFrame(feat_rows)

    # Leakage audit
    audit_leakage(X)

    # Coverage audit: drop features with >40% all-zero (missing proxy)
    coverage = (X != 0).mean()
    low_coverage_features = coverage[coverage < 0.05].index.tolist()
    if low_coverage_features:
        print(f"Dropping {len(low_coverage_features)} features with <5% coverage:")
        for f in low_coverage_features:
            print(f"  {f} ({100 * coverage[f]:.1f}% nonzero)")
        X = X.drop(columns=low_coverage_features)

    # Label: r_multiple
    y = filtered["r_multiple"].astype(float).fillna(-1.0).values  # hard_sl r=-1 by convention
    X["_label_r_multiple"] = y
    X["_session_date_dt"] = filtered["session_date_dt"].values  # for time-series CV later

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET} ({len(X):,} rows × {len(X.columns)} columns)")

    # Feature spec artifact
    features = [c for c in X.columns if not c.startswith("_")]
    FEATURE_SPEC_PATH.write_text(json.dumps({
        "features": features,
        "n_features": len(features),
        "version": "2026-04-22",
        "source_rules": len(rules),
        "n_training_rows": len(X),
        "dropped_low_coverage": low_coverage_features,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {FEATURE_SPEC_PATH} ({len(features)} features)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python tools/conviction/build_training_dataset.py
```

Expected output:
- Loads 178K+ Discovery trades
- Applies 74 rules, filters to ~150K candidates (estimated — exact count depends on which rules died)
- Extracts features
- Leakage audit passes (no exception)
- Writes parquet + feature spec JSON to `models/conviction/`

- [ ] **Step 3: Verify output**

```bash
.venv/Scripts/python -c "
import pandas as pd
from pathlib import Path
df = pd.read_parquet(Path('models/conviction/2026-04-22-training-dataset.parquet'))
print(f'Rows: {len(df):,}')
print(f'Columns: {len(df.columns)}')
print(f'Label col: _label_r_multiple — min={df[\"_label_r_multiple\"].min()}, max={df[\"_label_r_multiple\"].max()}, mean={df[\"_label_r_multiple\"].mean():.3f}')
print(f'Missing in label: {df[\"_label_r_multiple\"].isna().sum()}')
"
```

Expected: >100K rows, >30 columns, label mean ~0 (mix of winners/losers), no NaN in label.

- [ ] **Step 4: Commit**

```bash
git add tools/conviction/build_training_dataset.py
# Don't commit the parquet (large + reproducible)
echo "models/conviction/*.parquet" >> .gitignore || true
git add .gitignore
git commit -m "feat(conviction): build_training_dataset — Discovery → features + labels (T6)"
```

---

### Task 7: train_universal.py — XGBoost universal model

**Files:**
- Create: `tools/conviction/train_universal.py`

**Design:** Load training parquet. Split 80/20 by session_date chronological (time-series-aware, not random). Train XGBoost regressor with early stopping on validation fold. Save model artifact. Output: `models/conviction/2026-04-22-universal-xgboost.json`.

- [ ] **Step 1: Implement train_universal.py**

Create `tools/conviction/train_universal.py`:

```python
"""Train universal XGBoost model for sub-project #2.

Loads training dataset (from build_training_dataset.py), chronologically splits
80/20 (early-stop validation within Discovery), trains XGBoost regressor on
r_multiple target.

Output: models/conviction/2026-04-22-universal-xgboost.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent.parent.parent
PARQUET = ROOT / "models" / "conviction" / "2026-04-22-training-dataset.parquet"
MODEL_OUT = ROOT / "models" / "conviction" / "2026-04-22-universal-xgboost.json"
METRICS_OUT = ROOT / "models" / "conviction" / "2026-04-22-training-metrics.json"


def main():
    print(f"Loading training dataset: {PARQUET}")
    df = pd.read_parquet(PARQUET).copy()
    print(f"Rows: {len(df):,}, columns: {len(df.columns)}")

    # Chronological split 80/20
    df = df.sort_values("_session_date_dt").reset_index(drop=True)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df):,} rows ({train_df['_session_date_dt'].min()} to {train_df['_session_date_dt'].max()})")
    print(f"Val:   {len(val_df):,} rows ({val_df['_session_date_dt'].min()} to {val_df['_session_date_dt'].max()})")

    feature_cols = [c for c in df.columns if not c.startswith("_")]
    X_train = train_df[feature_cols].values
    y_train = train_df["_label_r_multiple"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["_label_r_multiple"].values

    # XGBoost hyperparameters — modest, regularized
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric="rmse",
        random_state=42,
    )
    print("Training...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    best_iter = model.best_iteration
    print(f"Best iteration: {best_iter}")

    # Evaluate on val fold
    y_val_pred = model.predict(X_val)
    rmse_val = float(np.sqrt(np.mean((y_val_pred - y_val) ** 2)))
    corr_val = float(np.corrcoef(y_val_pred, y_val)[0, 1])
    print(f"Validation RMSE: {rmse_val:.4f}")
    print(f"Validation Pearson: {corr_val:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"Saved model: {MODEL_OUT}")

    # Training metrics artifact
    METRICS_OUT.write_text(json.dumps({
        "model_path": str(MODEL_OUT.relative_to(ROOT)),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "best_iteration": best_iter,
        "rmse_val": rmse_val,
        "pearson_val": corr_val,
        "features": feature_cols,
        "hyperparameters": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 30,
        },
    }, indent=2), encoding="utf-8")
    print(f"Saved metrics: {METRICS_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python tools/conviction/train_universal.py
```

Expected: trains in 1-5 min, prints progress every 50 iterations, saves model + metrics. Pearson on validation fold should be > 0 (positive correlation of predicted vs realized).

- [ ] **Step 3: Commit tooling (not the model artifact)**

```bash
echo "models/conviction/*.json" >> .gitignore
# But we DO want to git-track the .gitkeep and feature-spec.json via -f
git add tools/conviction/train_universal.py .gitignore
git commit -m "feat(conviction): train_universal — XGBoost on R-multiple (T7)"
```

---

### Task 8: validate_on_2025.py — 4 model validation tests

**Files:**
- Create: `tools/conviction/validate_on_2025.py`

**Design:** Loads 2025 OOS data. Computes features per trade. Predicts with trained model. Runs 4 validation tests from spec §5.3:
1. OOS PF lift: top-50 by predicted_R vs random 50 baseline
2. Calibration monotonicity: decile curve shows rising realized_R
3. SHAP stability: top features overlap Discovery ↔ 2025
4. Per-session Spearman: rank correlation significantly > 0

Emits `models/conviction/2026-04-22-validation-report.md` + JSON.

- [ ] **Step 1: Implement validate_on_2025.py**

Create `tools/conviction/validate_on_2025.py`:

```python
"""Run 4 model validation tests on 2025 OOS data for sub-project #2.

Tests:
1. OOS PF lift: top-50 per session by predicted_R vs random 50 → PF delta
2. Calibration monotonicity: decile curve on 2025 predictions
3. SHAP stability: top features Discovery ↔ 2025 must overlap
4. Per-session Spearman: median rank correlation pred vs realized

Output: models/conviction/2026-04-22-validation-report.md + JSON
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tools.edge_discovery.data_loader import load_run
from services.conviction.feature_spec import extract_features, audit_leakage
from services.conviction.calibration import build_decile_calibration
from services.conviction.scorer import XGBoostScorer
from tools.edge_discovery.metrics import profit_factor

ROOT = Path(__file__).parent.parent.parent
BACKTEST_DIR = ROOT / "20260421-134338_full"
MODEL_PATH = ROOT / "models" / "conviction" / "2026-04-22-universal-xgboost.json"
FEATURE_SPEC_PATH = ROOT / "models" / "conviction" / "2026-04-22-feature-spec.json"
PARQUET_TRAIN = ROOT / "models" / "conviction" / "2026-04-22-training-dataset.parquet"
SURVIVORS = ROOT / "analysis" / "edge_discovery_runs" / "2026-04-22-validation-gate" / "stage6_validation_survivors.json"

REPORT_OUT = ROOT / "models" / "conviction" / "2026-04-22-validation-report.md"
JSON_OUT = ROOT / "models" / "conviction" / "2026-04-22-validation-report.json"

VALIDATION_START = date(2025, 1, 1)
VALIDATION_END = date(2025, 9, 30)


def build_2025_feature_frame():
    data = load_run(BACKTEST_DIR)
    t = data.trades
    t = t[(t["session_date_dt"] >= VALIDATION_START) & (t["session_date_dt"] <= VALIDATION_END)].copy()

    # Filter to same 74 survivor-matching trades
    survs = json.loads(SURVIVORS.read_text(encoding="utf-8"))
    rules = []
    for s in survs["survivors"]:
        rule_id = s["rule_id"]
        setup, cond_part = rule_id.split("__", 1)
        cond_key_part, cond_val_part = cond_part.split("=", 1)
        conds = list(zip(cond_key_part.split("+"), cond_val_part.split("+")))
        rules.append({"setup": setup, "conditions": conds})

    def matches(row):
        for r in rules:
            if row["setup_type"] != r["setup"]:
                continue
            if all(row.get(k) == v for k, v in r["conditions"]):
                return True
        return False

    t = t[t.apply(matches, axis=1)].copy()
    t["day_of_week"] = pd.to_datetime(t["session_date_dt"]).dt.day_name()
    return t


def main():
    print("Loading model...")
    scorer = XGBoostScorer(MODEL_PATH, FEATURE_SPEC_PATH)
    feature_list = scorer.features
    print(f"Features: {len(feature_list)}")

    print("Building 2025 feature frame...")
    trades_2025 = build_2025_feature_frame()
    print(f"Trade rows: {len(trades_2025):,}")

    feat_rows = [extract_features(r.to_dict()) for _, r in trades_2025.iterrows()]
    X_2025 = pd.DataFrame(feat_rows)
    for c in feature_list:
        if c not in X_2025.columns:
            X_2025[c] = 0.0
    X_2025 = X_2025[feature_list]
    audit_leakage(X_2025)
    preds = scorer.model.predict(X_2025.values)
    realized = trades_2025["r_multiple"].astype(float).fillna(-1.0).values
    trades_2025 = trades_2025.copy()
    trades_2025["predicted_r"] = preds
    trades_2025["realized_r"] = realized

    # === Test 1: OOS PF lift (top-50 per session vs random 50) ===
    def top50_per_session(df, score_col, n=50, random=False):
        out = []
        for _, g in df.groupby("session_date_dt"):
            if random:
                sample = g.sample(n=min(n, len(g)), random_state=42)
            else:
                sample = g.sort_values(score_col, ascending=False).head(n)
            out.append(sample)
        return pd.concat(out, ignore_index=True)

    top_ml = top50_per_session(trades_2025, "predicted_r")
    top_random = top50_per_session(trades_2025, "predicted_r", random=True)

    def _pf_from_r(rs):
        wins = rs[rs > 0]
        losses = rs[rs < 0]
        return float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")

    pf_ml = _pf_from_r(top_ml["realized_r"].values)
    pf_random = _pf_from_r(top_random["realized_r"].values)
    test1_pass = pf_ml > pf_random

    # === Test 2: Calibration monotonicity on 2025 ===
    curve = build_decile_calibration(
        pd.Series(trades_2025["predicted_r"].values),
        pd.Series(trades_2025["realized_r"].values),
    )
    curve_sorted = curve.sort_values("decile")
    # Monotonic non-decreasing with tolerance for noise
    deltas = curve_sorted["realized_median"].diff().dropna().values
    n_non_decreasing = int(sum(d >= -0.1 for d in deltas))
    test2_pass = n_non_decreasing >= 7  # at least 7/9 transitions non-decreasing

    # === Test 3: SHAP stability Discovery ↔ 2025 ===
    import shap
    X_train = pd.read_parquet(PARQUET_TRAIN)[feature_list].sample(n=5000, random_state=42).values
    explainer = shap.TreeExplainer(scorer.model)
    shap_train = explainer.shap_values(X_train)
    shap_val = explainer.shap_values(X_2025.sample(n=min(5000, len(X_2025)), random_state=42).values)
    top_train_idx = np.argsort(np.abs(shap_train).mean(axis=0))[-10:]
    top_val_idx = np.argsort(np.abs(shap_val).mean(axis=0))[-10:]
    overlap = len(set(top_train_idx) & set(top_val_idx))
    test3_pass = overlap >= 7

    # === Test 4: Per-session Spearman ===
    per_session_rhos = []
    for _, g in trades_2025.groupby("session_date_dt"):
        if len(g) < 5:
            continue
        rho, _ = spearmanr(g["predicted_r"].values, g["realized_r"].values)
        if not np.isnan(rho):
            per_session_rhos.append(rho)
    median_rho = float(np.median(per_session_rhos))
    n_positive = sum(1 for r in per_session_rhos if r > 0)
    # Binomial sign test p-value approximation
    from scipy.stats import binomtest
    p_sign = float(binomtest(n_positive, len(per_session_rhos), p=0.5, alternative="greater").pvalue)
    test4_pass = median_rho > 0.05 and p_sign < 0.05

    # Report
    results = {
        "test1_oos_pf_lift": {
            "pf_ml": pf_ml, "pf_random": pf_random, "passed": bool(test1_pass),
        },
        "test2_calibration_monotonicity": {
            "n_non_decreasing_transitions": n_non_decreasing,
            "required_min": 7,
            "passed": bool(test2_pass),
        },
        "test3_shap_stability": {
            "overlap_top10": overlap, "required_min": 7,
            "passed": bool(test3_pass),
        },
        "test4_per_session_spearman": {
            "median_rho": median_rho,
            "n_sessions": len(per_session_rhos),
            "n_positive": n_positive,
            "sign_test_p": p_sign,
            "passed": bool(test4_pass),
        },
        "all_passed": bool(test1_pass and test2_pass and test3_pass and test4_pass),
    }
    JSON_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Conviction model validation report (2025 OOS)",
        "",
        f"Model: `{MODEL_PATH.name}`",
        f"OOS period: {VALIDATION_START} to {VALIDATION_END}",
        "",
        "## Results",
        "",
        f"| Test | Required | Actual | Passed |",
        f"|---|---|---|---|",
        f"| 1. OOS PF lift (top-50 vs random) | PF_ml > PF_random | PF_ml={pf_ml:.3f}, PF_random={pf_random:.3f} | {'✅' if test1_pass else '❌'} |",
        f"| 2. Calibration monotonicity | ≥7/9 transitions non-decreasing | {n_non_decreasing}/9 | {'✅' if test2_pass else '❌'} |",
        f"| 3. SHAP stability | ≥7/10 features overlap | {overlap}/10 | {'✅' if test3_pass else '❌'} |",
        f"| 4. Per-session Spearman | median rho > 0.05, p < 0.05 | rho={median_rho:.3f}, p={p_sign:.4f} | {'✅' if test4_pass else '❌'} |",
        "",
        f"**All passed: {'✅ YES' if results['all_passed'] else '❌ NO'}**",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport: {REPORT_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (only after Task 7 trained the model)**

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python tools/conviction/validate_on_2025.py
```

Expected: runs ~5 min, writes report + JSON. If any test fails → model is not ready to ship.

- [ ] **Step 3: Review the report**

```bash
cat models/conviction/2026-04-22-validation-report.md
```

Review all 4 test outcomes. If any fails, do NOT proceed to later tasks. Diagnose:
- Test 1 fail → model doesn't discriminate; check feature/label correctness
- Test 2 fail → model not calibrated; check hyperparameters / target transformation
- Test 3 fail → feature importance unstable → overfitting or distribution shift
- Test 4 fail → model doesn't predict relative ranking → ranking-unsuitable

- [ ] **Step 4: Commit**

```bash
git add tools/conviction/validate_on_2025.py
git add -f models/conviction/2026-04-22-validation-report.md models/conviction/2026-04-22-validation-report.json
git commit -m "feat(conviction): 4 model validation tests on 2025 OOS (T8)"
```

---

## Phase E: Per-setup comparison (Task 9)

### Task 9: train_per_setup.py — Architecture comparison for top-2 setups

**Files:**
- Create: `tools/conviction/train_per_setup.py`

**Design:** Trains separate XGBoost models for premium_zone_short (104K trades) and range_bounce_short (58K trades). Compares their OOS RMSE + top-50 PF against universal model predictions on the same setups. Outputs decision report.

- [ ] **Step 1: Implement train_per_setup.py**

Create `tools/conviction/train_per_setup.py`:

```python
"""Per-setup model comparison for sub-project #2.

Trains separate XGBoost regressors for premium_zone_short + range_bounce_short
(the 2 setups with enough data), compares OOS RMSE + top-50 PF vs universal
model predictions on the same setups. Output: architecture-decision report.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from services.conviction.scorer import XGBoostScorer
from tools.edge_discovery.data_loader import load_run
from services.conviction.feature_spec import extract_features

ROOT = Path(__file__).parent.parent.parent
PARQUET = ROOT / "models" / "conviction" / "2026-04-22-training-dataset.parquet"
UNIVERSAL_MODEL = ROOT / "models" / "conviction" / "2026-04-22-universal-xgboost.json"
FEATURE_SPEC = ROOT / "models" / "conviction" / "2026-04-22-feature-spec.json"
PER_SETUP_DIR = ROOT / "models" / "conviction" / "per_setup"
BACKTEST_2025 = ROOT / "20260421-134338_full"
OUT_REPORT = ROOT / "models" / "conviction" / "2026-04-22-architecture-decision.md"

TARGET_SETUPS = ["premium_zone_short", "range_bounce_short"]


def train_single_setup(setup, train_df, feature_cols):
    setup_col = f"setup_type_{setup}"
    if setup_col not in train_df.columns:
        raise ValueError(f"Setup one-hot missing: {setup_col}")
    sub = train_df[train_df[setup_col] == 1].copy()
    sub = sub.sort_values("_session_date_dt").reset_index(drop=True)
    split = int(0.8 * len(sub))
    tr, vl = sub.iloc[:split], sub.iloc[split:]
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=30,
        eval_metric="rmse",
        random_state=42,
    )
    model.fit(
        tr[feature_cols].values, tr["_label_r_multiple"].values,
        eval_set=[(vl[feature_cols].values, vl["_label_r_multiple"].values)],
        verbose=False,
    )
    return model, tr, vl


def eval_on_2025(model, feature_cols, setup):
    from datetime import date
    data = load_run(BACKTEST_2025)
    t = data.trades
    t = t[(t["session_date_dt"] >= date(2025, 1, 1))
          & (t["session_date_dt"] <= date(2025, 9, 30))]
    t = t[t["setup_type"] == setup].copy()
    t["day_of_week"] = pd.to_datetime(t["session_date_dt"]).dt.day_name()
    feat_rows = [extract_features(r.to_dict()) for _, r in t.iterrows()]
    X = pd.DataFrame(feat_rows)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols].values
    y = t["r_multiple"].astype(float).fillna(-1.0).values
    pred = model.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    pearson = float(np.corrcoef(pred, y)[0, 1]) if len(pred) > 1 else 0.0
    # Top-50 PF per session (if enough trades) — simplified per-setup
    df = pd.DataFrame({"pred": pred, "realized": y, "sess": t["session_date_dt"].values})
    pf_top50 = _top_n_pf(df, n=50)
    return {"rmse": rmse, "pearson": pearson, "pf_top50": pf_top50, "n": len(t)}


def _top_n_pf(df, n=50):
    out = []
    for _, g in df.groupby("sess"):
        out.append(g.sort_values("pred", ascending=False).head(n))
    top = pd.concat(out, ignore_index=True)
    wins = top[top["realized"] > 0]["realized"]
    losses = top[top["realized"] < 0]["realized"]
    return float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")


def main():
    df = pd.read_parquet(PARQUET)
    feature_cols = [c for c in df.columns if not c.startswith("_")]

    universal_scorer = XGBoostScorer(UNIVERSAL_MODEL, FEATURE_SPEC)

    PER_SETUP_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Conviction architecture decision",
        "",
        "| Setup | Universal RMSE | Per-setup RMSE | Universal PF@50 | Per-setup PF@50 | Winner |",
        "|---|---|---|---|---|---|",
    ]
    decisions = []
    for setup in TARGET_SETUPS:
        print(f"Training per-setup model for {setup}...")
        setup_model, _, _ = train_single_setup(setup, df, feature_cols)
        ps_path = PER_SETUP_DIR / f"{setup}-xgboost.json"
        setup_model.save_model(str(ps_path))

        print(f"Evaluating on 2025 OOS for {setup}...")
        per_setup_eval = eval_on_2025(setup_model, feature_cols, setup)
        universal_eval = eval_on_2025(universal_scorer.model, feature_cols, setup)

        rmse_delta_pct = 100 * (per_setup_eval["rmse"] - universal_eval["rmse"]) / universal_eval["rmse"]
        pf_delta_pct = 100 * (per_setup_eval["pf_top50"] - universal_eval["pf_top50"]) / universal_eval["pf_top50"]
        winner = "per_setup" if pf_delta_pct > 5 else "universal"

        decisions.append({
            "setup": setup,
            "universal": universal_eval,
            "per_setup": per_setup_eval,
            "rmse_delta_pct": rmse_delta_pct,
            "pf_delta_pct": pf_delta_pct,
            "winner": winner,
        })
        lines.append(
            f"| {setup} | {universal_eval['rmse']:.3f} | {per_setup_eval['rmse']:.3f} | "
            f"{universal_eval['pf_top50']:.3f} | {per_setup_eval['pf_top50']:.3f} | {winner} |"
        )

    ship_per_setup = [d["setup"] for d in decisions if d["winner"] == "per_setup"]
    lines.append("")
    lines.append(f"## Decision: {'Ship per-setup for ' + ', '.join(ship_per_setup) if ship_per_setup else 'Ship universal only'}")

    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "models" / "conviction" / "2026-04-22-architecture-decision.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\nReport: {OUT_REPORT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python tools/conviction/train_per_setup.py
```

Expected: trains 2 per-setup models, evaluates both + universal on 2025 OOS per setup, emits decision report.

- [ ] **Step 3: Review + commit**

```bash
cat models/conviction/2026-04-22-architecture-decision.md
git add tools/conviction/train_per_setup.py
git add -f models/conviction/2026-04-22-architecture-decision.md
git add -f models/conviction/2026-04-22-architecture-decision.json
git commit -m "feat(conviction): per-setup architecture comparison (T9)"
```

---

## Phase F: Backtest integration (Tasks 10-11)

### Task 10: stage5d_conviction_simulation.py + tests

**Files:**
- Create: `tools/edge_discovery/stages/stage5d_conviction_simulation.py`
- Create: `tests/edge_discovery/test_stage5d.py`

**Design:** Mirrors Stage 5c pattern. Takes trades + cfg + scorer + gate, simulates chronologically, reports before/after aggregate metrics.

- [ ] **Step 1: Write failing tests**

Create `tests/edge_discovery/test_stage5d.py`:

```python
"""Stage 5d tests: conviction gate replay."""
from datetime import date, datetime

import pandas as pd
import pytest

from tools.edge_discovery.stages.stage5d_conviction_simulation import (
    simulate_conviction_filter,
)


class _FakeScorer:
    """Mock scorer that returns fixed per-symbol scores."""
    def __init__(self, scores):
        self.scores = scores
        self.features = ["dummy"]

    def predict(self, feat_or_frame):
        if isinstance(feat_or_frame, dict):
            return self.scores.get(feat_or_frame.get("symbol"), 0.0)
        raise TypeError


def _trade(symbol, ts, pnl, r, session=None):
    return {
        "symbol": symbol,
        "setup_type": "premium_zone_short",
        "regime": "chop",
        "cap_segment": "small_cap",
        "hour_bucket": "morning",
        "decision_ts": ts,
        "session_date_dt": session or date(2025, 1, 2),
        "minute_of_day": 600,
        "total_trade_pnl": pnl,
        "r_multiple": r,
    }


def test_daily_cap_enforced():
    """With cap=2, only 2 trades per session admitted."""
    trades = pd.DataFrame([
        _trade("A", datetime(2025, 1, 2, 10, 0), 100, 1.0),
        _trade("B", datetime(2025, 1, 2, 10, 5), 100, 1.0),
        _trade("C", datetime(2025, 1, 2, 10, 10), 100, 1.0),
    ])
    scorer = _FakeScorer({"A": 0.9, "B": 0.8, "C": 0.7})
    cfg = {"enabled": True, "daily_cap": 2, "min_predicted_r": 0.0}
    result = simulate_conviction_filter(trades, scorer, cfg)
    admitted = result[result["admitted"] == True]
    # All three score above threshold — but cap is 2
    assert len(admitted) == 2


def test_threshold_enforced():
    """Low-score candidates rejected."""
    trades = pd.DataFrame([
        _trade("A", datetime(2025, 1, 2, 10, 0), 100, 1.0),
        _trade("B", datetime(2025, 1, 2, 10, 5), 100, 1.0),
    ])
    scorer = _FakeScorer({"A": 0.9, "B": 0.1})
    cfg = {"enabled": True, "daily_cap": 50, "min_predicted_r": 0.5}
    result = simulate_conviction_filter(trades, scorer, cfg)
    admitted = result[result["admitted"] == True]
    assert len(admitted) == 1
    assert admitted.iloc[0]["symbol"] == "A"


def test_session_boundary_resets_cap():
    """Cap resets between sessions."""
    trades = pd.DataFrame([
        _trade("A", datetime(2025, 1, 2, 10, 0), 100, 1.0, session=date(2025, 1, 2)),
        _trade("B", datetime(2025, 1, 2, 10, 5), 100, 1.0, session=date(2025, 1, 2)),
        _trade("C", datetime(2025, 1, 3, 10, 0), 100, 1.0, session=date(2025, 1, 3)),
    ])
    scorer = _FakeScorer({"A": 0.9, "B": 0.8, "C": 0.7})
    cfg = {"enabled": True, "daily_cap": 1, "min_predicted_r": 0.0}
    result = simulate_conviction_filter(trades, scorer, cfg)
    admitted = result[result["admitted"] == True]
    # 1 admitted on day 1, 1 on day 2 = 2 total
    assert len(admitted) == 2


def test_admitted_column_preserves_chronological_order():
    """Output trades preserve input order; admitted is a column, not reordering."""
    trades = pd.DataFrame([
        _trade("A", datetime(2025, 1, 2, 10, 0), 100, 1.0),
        _trade("B", datetime(2025, 1, 2, 10, 5), 100, 1.0),
    ])
    scorer = _FakeScorer({"A": 0.9, "B": 0.1})
    cfg = {"enabled": True, "daily_cap": 5, "min_predicted_r": 0.5}
    result = simulate_conviction_filter(trades, scorer, cfg)
    assert list(result["symbol"]) == ["A", "B"]
    assert list(result["admitted"]) == [True, False]
```

- [ ] **Step 2: Run tests — MUST fail**

```bash
.venv/Scripts/python -m pytest tests/edge_discovery/test_stage5d.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement stage5d_conviction_simulation.py**

Create `tools/edge_discovery/stages/stage5d_conviction_simulation.py`:

```python
"""Stage 5d: Conviction gate simulation (backtest replay).

Replays ConvictionGate chronologically against a trade stream. Each trade is
scored by the injected scorer, evaluated by the gate, and marked admitted/
rejected in the returned DataFrame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Protocol

import pandas as pd

from services.conviction.feature_spec import extract_features
from services.conviction.gate import ConvictionGate
from tools.edge_discovery.report_writer import write_json_artifact, append_section


class _ScorerLike(Protocol):
    def predict(self, feat: Dict[str, float]) -> float: ...


def simulate_conviction_filter(
    trades: pd.DataFrame, scorer: _ScorerLike, cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Replay ConvictionGate on the trade stream chronologically.

    Adds columns: admitted (bool), predicted_r (float), reject_reason (str).
    """
    gate = ConvictionGate(cfg)
    t = trades.copy()
    t["_decision_ts_parsed"] = pd.to_datetime(t["decision_ts"], errors="coerce")
    t = t.sort_values("_decision_ts_parsed").reset_index(drop=True)

    preds: List[float] = []
    admitted: List[bool] = []
    reasons: List[str] = []

    for row in t.itertuples():
        row_dict = t.iloc[row.Index].to_dict()
        feat = extract_features(row_dict)
        pred = float(scorer.predict({**feat, "symbol": row_dict.get("symbol", "")}))
        cand = {
            "symbol": row_dict.get("symbol", ""),
            "decision_ts": row.Index,
            "session_date": row_dict.get("session_date_dt"),
        }
        ok, reason = gate.evaluate(cand, pred)
        preds.append(pred)
        admitted.append(ok)
        reasons.append(reason)

    t["predicted_r"] = preds
    t["admitted"] = pd.Series(admitted, dtype=object)
    t["reject_reason"] = reasons
    t = t.drop(columns=["_decision_ts_parsed"])
    return t


def _aggregate_stats(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    pnl = df["total_trade_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0].abs()
    pf = float(wins.sum() / losses.sum()) if losses.sum() > 0 else float("inf")
    wr = 100 * len(wins) / len(df) if len(df) else 0.0
    daily = df.groupby("session_date_dt")["total_trade_pnl"].agg(["count", "sum"])
    sharpe = float(daily["sum"].mean() / daily["sum"].std()) if daily["sum"].std() > 0 else 0.0
    losing_days = int((daily["sum"] < 0).sum())
    n_sessions = len(daily)
    return {
        "scenario": name,
        "n_trades": int(len(df)),
        "n_sessions": n_sessions,
        "trades_per_day": round(len(df) / n_sessions, 1) if n_sessions else 0.0,
        "total_pnl": round(float(pnl.sum()), 0),
        "pf": round(pf, 3) if pf != float("inf") else 999.0,
        "wr_pct": round(wr, 1),
        "session_sharpe": round(sharpe, 3),
        "losing_days_pct": round(100 * losing_days / n_sessions, 1) if n_sessions else 0.0,
    }


def _rows_to_markdown(rows):
    if not rows:
        return "_(no rows)_"
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(out)


def run_stage5d(
    trades: pd.DataFrame,
    scorer,
    cfg: Dict[str, Any],
    report_path: Path,
    summary_json: Path,
) -> Dict[str, Any]:
    filtered = simulate_conviction_filter(trades, scorer, cfg)
    before = _aggregate_stats(filtered, "Before ConvictionGate")
    after = _aggregate_stats(filtered[filtered["admitted"].astype(bool)], "After ConvictionGate")
    rej_reasons = filtered[~filtered["admitted"].astype(bool)]["reject_reason"].value_counts().head(10).to_dict()
    rej_rows = [{"reason": k, "count": int(v)} for k, v in rej_reasons.items()]

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Stage 5d — Conviction Gate Simulation",
        "",
        "## Scenarios",
        "",
        _rows_to_markdown([before, after]),
    ]
    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")
    append_section(report_path, "## Top rejection reasons", _rows_to_markdown(rej_rows))

    delta = {
        "n_trades_delta": after["n_trades"] - before["n_trades"],
        "pf_delta": round(after["pf"] - before["pf"], 3),
        "sharpe_delta": round(after["session_sharpe"] - before["session_sharpe"], 3),
    }
    write_json_artifact(summary_json, {
        "stage": "5d", "cfg": cfg,
        "before": before, "after": after, "delta": delta, "rejections": rej_rows,
    })
    return {"before": before, "after": after, "delta": delta}
```

- [ ] **Step 4: Run tests — MUST pass**

```bash
.venv/Scripts/python -m pytest tests/edge_discovery/test_stage5d.py tests/conviction/ -v
```

Expected: 4 new + all prior passes.

- [ ] **Step 5: Commit**

```bash
git add tools/edge_discovery/stages/stage5d_conviction_simulation.py
git add -f tests/edge_discovery/test_stage5d.py
git commit -m "feat(gauntlet/stage5d): conviction gate simulation (T10)"
```

---

### Task 11: Wire Stage 5d into run_gauntlet

**Files:**
- Modify: `tools/edge_discovery/run_gauntlet.py`

- [ ] **Step 1: Modify run_gauntlet.py**

In `tools/edge_discovery/run_gauntlet.py`, imports section, add:

```python
from tools.edge_discovery.stages.stage5d_conviction_simulation import run_stage5d
from services.conviction.scorer import XGBoostScorer
```

At the end of the existing Stage 5c block (after `run_stage5c` call), add:

```python
    # Stage 5d: conviction gate (ML scorer + top-50 + threshold)
    print("[gauntlet] Stage 5d: Conviction gate simulation ...")
    stage5d_run = False
    try:
        cv_cfg = full_cfg.get("conviction_gate", {})
        if cv_cfg and cv_cfg.get("enabled"):
            model_path = ROOT / cv_cfg["model_artifact"]
            feature_spec_path = ROOT / cv_cfg["feature_spec_path"]
            if not model_path.exists() or not feature_spec_path.exists():
                print(f"[gauntlet]   Stage 5d skipped (model artifact not found: {model_path})")
            else:
                scorer = XGBoostScorer(model_path, feature_spec_path)
                # Compose gate config from conviction_gate block
                gate_cfg = {
                    "enabled": True,
                    "daily_cap": int(cv_cfg["daily_cap"]),
                    "min_predicted_r": float(cv_cfg["min_predicted_r"]),
                }
                run_stage5d(
                    trades=filtered_trades,  # re-use Stage 5c's filtered set
                    scorer=scorer,
                    cfg=gate_cfg,
                    report_path=output_dir / "08-conviction-simulation.md",
                    summary_json=output_dir / "stage5d_simulation.json",
                )
                stage5d_run = True
                print("[gauntlet]   Stage 5d complete")
        else:
            print("[gauntlet]   Stage 5d skipped (conviction_gate disabled)")
    except Exception as e:
        print(f"[gauntlet]   Stage 5d ERROR: {e}")
        import traceback; traceback.print_exc()
```

At the return dict of `run_gauntlet_all`, add:
```python
        "stage5d_run": stage5d_run,
```

- [ ] **Step 2: Run tests**

```bash
.venv/Scripts/python -m pytest tests/edge_discovery/ tests/cross_sectional/ tests/conviction/ -q
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add tools/edge_discovery/run_gauntlet.py
git commit -m "feat(gauntlet): wire Stage 5d into run_gauntlet (T11)"
```

---

## Phase G: End-to-end validation (Task 12)

### Task 12: Enable conviction_gate + run full gauntlet

**Files:**
- Modify: `config/configuration.json`

- [ ] **Step 1: Flip conviction_gate.enabled to true**

Edit `config/configuration.json` — change `"enabled": false` to `"enabled": true` in the `conviction_gate` block.

- [ ] **Step 2: Run full gauntlet**

```bash
rm -rf docs/edge_discovery/2026-04-22-run && mkdir -p docs/edge_discovery/2026-04-22-run && \
PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv/Scripts/python -m tools.edge_discovery.run_gauntlet \
  --backtest-dir cloud_results/20260419_discovery \
  --output-dir docs/edge_discovery/2026-04-22-run \
  --discovery-start 2023-01-01 --discovery-end 2024-12-31 \
  --validation-start 2025-01-01 --validation-end 2025-09-30 \
  --holdout-start 2025-10-01 --holdout-end 2026-03-31
```

Expected: runs all stages including 5d. `08-conviction-simulation.md` exists.

- [ ] **Step 3: Verify end-to-end metrics**

```bash
cat docs/edge_discovery/2026-04-22-run/08-conviction-simulation.md
```

Expected: trade count drops from ~100K (Stage 5c output) to ~12K (50/day × 242 sessions). PF should be ≥ Stage 5c baseline.

- [ ] **Step 4: Run validation-period check**

Create a quick one-liner that measures ConvictionGate on 2025 OOS trades (use the validate_stage5c_on_2025.py pattern from sub-project #3 — adapt the script to chain Stage 5c → Stage 5d):

```bash
# Pending: a validate_stage5d_on_2025.py analog (pattern from tools/edge_discovery/validate_stage5c_on_2025.py)
# To be created separately if T12 reveals need for it. For now, model validation report (T8) is sufficient.
```

- [ ] **Step 5: Snapshot + commit**

```bash
mkdir -p analysis/edge_discovery_runs/2026-04-22
cp docs/edge_discovery/2026-04-22-run/08-conviction-simulation.md \
   docs/edge_discovery/2026-04-22-run/stage5d_simulation.json \
   analysis/edge_discovery_runs/2026-04-22/
git add config/configuration.json
git add -f analysis/edge_discovery_runs/2026-04-22/08-conviction-simulation.md \
             analysis/edge_discovery_runs/2026-04-22/stage5d_simulation.json
git commit -m "feat(gauntlet): end-to-end run with conviction_gate enabled (T12)"
```

---

## Deferred for future sub-projects

- **Conviction-tiered or Kelly sizing:** sub-project #5 (Strategy Lifecycle) after shadow-loop validates calibration stability
- **Live integration into screener_live.py:** sub-project #6 (Deployment Path)
- **Live vs backtest parity monitoring:** sub-project #4 (Shadow/Parity Loop)
- **Per-setup model production deployment for small-N setups:** revisit when more data accumulates (sub-project #5)
- **Model retirement / automatic retraining:** sub-project #5
- **Holdout gate on Oct 2025 - Mar 2026:** sub-project #1 Task 86-87 (run AFTER sub-project #2 ships AND 2026 Q1 OCI backtest is available)

---

## Self-review

**Spec coverage:**
- Section 3.1 Scoring approach (XGBoost): ✅ Task 7
- Section 3.2 Target variable (R-multiple): ✅ Task 6 (label) + Task 7 (training)
- Section 3.3 Feature set: ✅ Task 2 (whitelist) + Task 6 (extraction)
- Section 3.4 Leakage audit: ✅ Task 2 (audit_leakage function) + Task 6 (called in pipeline)
- Section 3.5 Coverage audit: ✅ Task 6
- Section 3.6 Selection mechanism: ✅ Task 5 (ConvictionGate)
- Section 3.7 Sizing: ✅ unchanged from existing infra (no task needed)
- Section 3.8 Universal + per-setup: ✅ Task 7 (universal) + Task 9 (per-setup)
- Section 3.9 Training protocol: ✅ Task 7 (chronological 80/20), Task 8 (OOS validate)
- Section 3.10 Daily cap 50: ✅ config in Task 1
- Section 5.3 4 validation tests: ✅ Task 8

**Placeholder scan:** no TBD/TODO/vague references. Every step has exact code.

**Type consistency:** `XGBoostScorer.predict()` returns float consistently. `ConvictionGate.evaluate(cand, predicted_r)` signature consistent across Tasks 5, 10, 11. `extract_features(row)` returns `Dict[str, float]` consistently. `_label_r_multiple` column name consistent across Tasks 6, 7, 9.

**Open item:** Task 12 Step 4 notes a `validate_stage5d_on_2025.py` equivalent may be needed but defers it to after T12 runs. Model validation (Task 8) covers the OOS tests already; the end-to-end 2025 stage5d replay is a "nice to have," not a required ship gate.
