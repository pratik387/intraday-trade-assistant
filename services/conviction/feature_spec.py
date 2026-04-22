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
            f"leakage detected: frame contains {len(present)} outcome columns "
            f"that must not be training features: {present}"
        )
