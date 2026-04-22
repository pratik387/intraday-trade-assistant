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
