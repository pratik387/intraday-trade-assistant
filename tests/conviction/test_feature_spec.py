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
