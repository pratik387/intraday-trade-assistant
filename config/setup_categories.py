# config/setup_categories.py
"""
Setup Category Registry - Single source of truth for setup → category mapping.

Categories define how quality metrics are calculated for different setup types:
- BREAKOUT: Momentum-based quality (volume * breakout_strength / risk)
- LEVEL: Level acceptance quality (retest_ok + hold_ok)
- REVERSION: Mean reversion quality (extension from VWAP + exhaustion)
- MOMENTUM: Trend strength quality (ADX + trend alignment)

This allows category-specific:
1. Quality calculations in planner
2. Gate logic in trade_decision_gate
3. Ranking weights in ranking_engine
"""

from typing import Dict, Any, Optional
from enum import Enum


class SetupCategory(Enum):
    """Setup categories with distinct quality metrics."""
    BREAKOUT = "BREAKOUT"      # Momentum breaks of levels
    LEVEL = "LEVEL"            # Bounce/rejection at levels
    REVERSION = "REVERSION"    # Mean reversion plays
    MOMENTUM = "MOMENTUM"      # Trend continuation


# Setup type → Category mapping
# Note: Base name without _long/_short suffix is mapped
SETUP_CATEGORIES: Dict[str, SetupCategory] = {
    # BREAKOUT: Momentum breaks of levels
    # Quality = volume_ratio * breakout_strength / normalized_risk
    "breakout": SetupCategory.BREAKOUT,
    "orb_breakout": SetupCategory.BREAKOUT,
    "orb_breakdown": SetupCategory.BREAKOUT,
    "flag_continuation": SetupCategory.BREAKOUT,
    "squeeze_release": SetupCategory.BREAKOUT,
    "momentum_breakout": SetupCategory.BREAKOUT,
    "range_breakout": SetupCategory.BREAKOUT,
    "range_breakdown": SetupCategory.BREAKOUT,
    "gap_breakout": SetupCategory.BREAKOUT,
    "resistance_breakout": SetupCategory.BREAKOUT,
    "support_breakdown": SetupCategory.BREAKOUT,
    "break_of_structure": SetupCategory.BREAKOUT,
    "change_of_character": SetupCategory.BREAKOUT,
    "level_breakout": SetupCategory.BREAKOUT,

    # LEVEL: Bounce/rejection at levels
    # Quality = retest_ok + hold_ok (level acceptance)
    "support_bounce": SetupCategory.LEVEL,
    "resistance_bounce": SetupCategory.LEVEL,
    "range_bounce": SetupCategory.LEVEL,
    "range_rejection": SetupCategory.LEVEL,
    "vwap_reclaim": SetupCategory.LEVEL,
    "vwap_lose": SetupCategory.LEVEL,
    "premium_zone": SetupCategory.LEVEL,
    "discount_zone": SetupCategory.LEVEL,
    "orb_pullback": SetupCategory.LEVEL,
    "order_block": SetupCategory.LEVEL,

    # REVERSION: Mean reversion plays
    # Quality = extension_from_vwap + exhaustion_signals
    "failure_fade": SetupCategory.REVERSION,
    "volume_spike_reversal": SetupCategory.REVERSION,
    "gap_fill": SetupCategory.REVERSION,
    "vwap_mean_reversion": SetupCategory.REVERSION,
    "liquidity_sweep": SetupCategory.REVERSION,
    "fair_value_gap": SetupCategory.REVERSION,

    # MOMENTUM: Trend continuation
    # Quality = ADX (trend strength) + trend alignment
    "trend_continuation": SetupCategory.MOMENTUM,
    "trend_pullback": SetupCategory.MOMENTUM,
    "trend_reversal": SetupCategory.MOMENTUM,
    "momentum_trend": SetupCategory.MOMENTUM,
    "first_hour_momentum": SetupCategory.MOMENTUM,  # FHM - catches big movers early in first hour
}


# Category-specific configuration
CATEGORY_CONFIG: Dict[SetupCategory, Dict[str, Any]] = {
    SetupCategory.BREAKOUT: {
        "quality_factors": ["volume_ratio", "breakout_strength", "adx"],
        "entry_zone_mult": 0.15,  # 0.15 ATR for tight breakout entries
        "min_volume_ratio": 1.2,
        "min_adx": 20,
        "description": "Momentum breaks through key levels",
    },
    SetupCategory.LEVEL: {
        "quality_factors": ["retest_ok", "hold_ok", "level_respect"],
        "entry_zone_mult": 0.10,  # Tighter zones at established levels
        "description": "Bounce/rejection at support/resistance",
    },
    SetupCategory.REVERSION: {
        "quality_factors": ["extension_pct", "exhaustion_volume", "vwap_distance"],
        "entry_zone_mult": 0.20,  # Wider zones for mean reversion
        "min_extension_pct": 1.0,  # Minimum % extension from mean
        "description": "Mean reversion after overextension",
    },
    SetupCategory.MOMENTUM: {
        "quality_factors": ["adx", "trend_alignment", "ema_stack"],
        "entry_zone_mult": 0.12,
        "min_adx": 25,  # Higher ADX required for momentum plays
        "description": "Trend continuation with momentum",
    },
}


def get_base_setup_name(setup_type: str) -> str:
    """
    Extract base setup name by removing _long/_short suffix.

    Examples:
        'orb_breakout_long' -> 'orb_breakout'
        'support_bounce_short' -> 'support_bounce'
        'breakout_long' -> 'breakout'
    """
    if setup_type.endswith('_long'):
        return setup_type[:-5]
    elif setup_type.endswith('_short'):
        return setup_type[:-6]
    return setup_type


def get_category(setup_type: str) -> Optional[SetupCategory]:
    """
    Get the category for a setup type.

    Args:
        setup_type: Full setup type (e.g., 'orb_breakout_long', 'support_bounce_short')

    Returns:
        SetupCategory enum or None if not found
    """
    base_name = get_base_setup_name(setup_type)
    return SETUP_CATEGORIES.get(base_name)


def get_category_config(category: SetupCategory) -> Dict[str, Any]:
    """
    Get configuration for a category.

    Args:
        category: SetupCategory enum

    Returns:
        Category configuration dict
    """
    return CATEGORY_CONFIG.get(category, {})


def get_setup_category_config(setup_type: str) -> Dict[str, Any]:
    """
    Get category config for a setup type (convenience function).

    Args:
        setup_type: Full setup type (e.g., 'orb_breakout_long')

    Returns:
        Category configuration dict or empty dict if not found
    """
    category = get_category(setup_type)
    if category:
        return get_category_config(category)
    return {}


def is_breakout_category(setup_type: str) -> bool:
    """Check if setup type belongs to BREAKOUT category."""
    return get_category(setup_type) == SetupCategory.BREAKOUT


def is_level_category(setup_type: str) -> bool:
    """Check if setup type belongs to LEVEL category."""
    return get_category(setup_type) == SetupCategory.LEVEL


def is_reversion_category(setup_type: str) -> bool:
    """Check if setup type belongs to REVERSION category."""
    return get_category(setup_type) == SetupCategory.REVERSION


def is_momentum_category(setup_type: str) -> bool:
    """Check if setup type belongs to MOMENTUM category."""
    return get_category(setup_type) == SetupCategory.MOMENTUM


# For backward compatibility - list of breakout strategies
BREAKOUT_STRATEGIES = [
    name for name, cat in SETUP_CATEGORIES.items()
    if cat == SetupCategory.BREAKOUT
]


if __name__ == "__main__":
    # Test the category registry
    test_setups = [
        "orb_breakout_long",
        "support_bounce_short",
        "failure_fade_long",
        "trend_continuation_long",
        "premium_zone_short",
        "unknown_setup_type",
    ]

    print("Setup Category Registry Test")
    print("=" * 50)

    for setup in test_setups:
        cat = get_category(setup)
        config = get_setup_category_config(setup)
        print(f"{setup:30s} -> {cat.value if cat else 'UNKNOWN':12s} | factors: {config.get('quality_factors', [])}")
