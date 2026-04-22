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
