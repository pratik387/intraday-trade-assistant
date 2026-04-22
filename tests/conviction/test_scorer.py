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
    """A missing feature must produce the same prediction as one explicitly set to 0.0."""
    model_path, feature_spec_path, features = tiny_model_artifacts
    scorer = XGBoostScorer(model_path, feature_spec_path)
    incomplete = {
        "momentum_3bar_pct": 0.5,
        "setup_type_premium_zone_short": 1,
        # deliberately missing: vol_z, pdz_confluence_count
    }
    explicit_zeros = {
        "momentum_3bar_pct": 0.5,
        "vol_z": 0.0,
        "pdz_confluence_count": 0.0,
        "setup_type_premium_zone_short": 1,
    }
    pred_incomplete = scorer.predict(incomplete)
    pred_explicit = scorer.predict(explicit_zeros)
    assert isinstance(pred_incomplete, float)
    assert pred_incomplete == pytest.approx(pred_explicit, abs=1e-6)


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


def test_predict_batch_matches_per_call_predict_within_tolerance(tiny_model_artifacts):
    """Batched and per-call predictions must agree numerically."""
    model_path, feature_spec_path, features = tiny_model_artifacts
    scorer = XGBoostScorer(model_path, feature_spec_path)
    feat_dicts = [
        {"momentum_3bar_pct": 0.5, "vol_z": 0.0, "pdz_confluence_count": 2, "setup_type_premium_zone_short": 1},
        {"momentum_3bar_pct": -0.2, "vol_z": 1.5, "pdz_confluence_count": 0, "setup_type_premium_zone_short": 0},
        {"momentum_3bar_pct": 1.0, "vol_z": -0.5, "pdz_confluence_count": 1, "setup_type_premium_zone_short": 1},
    ]
    batched = scorer.predict_batch(feat_dicts)
    individual = np.array([scorer.predict(f) for f in feat_dicts])
    assert batched.shape == (3,)
    np.testing.assert_allclose(batched, individual, atol=1e-5)


def test_predict_batch_empty_input_returns_empty(tiny_model_artifacts):
    model_path, feature_spec_path, _ = tiny_model_artifacts
    scorer = XGBoostScorer(model_path, feature_spec_path)
    out = scorer.predict_batch([])
    assert out.shape == (0,)


def test_predict_batch_handles_missing_features_per_dict(tiny_model_artifacts):
    """Each dict's missing features become 0.0, independently per candidate."""
    model_path, feature_spec_path, _ = tiny_model_artifacts
    scorer = XGBoostScorer(model_path, feature_spec_path)
    incomplete = [
        {"momentum_3bar_pct": 0.5, "setup_type_premium_zone_short": 1},
        {"vol_z": 0.3, "pdz_confluence_count": 2},
    ]
    explicit = [
        {"momentum_3bar_pct": 0.5, "vol_z": 0.0, "pdz_confluence_count": 0.0, "setup_type_premium_zone_short": 1},
        {"momentum_3bar_pct": 0.0, "vol_z": 0.3, "pdz_confluence_count": 2, "setup_type_premium_zone_short": 0},
    ]
    np.testing.assert_allclose(
        scorer.predict_batch(incomplete),
        scorer.predict_batch(explicit),
        atol=1e-6,
    )
