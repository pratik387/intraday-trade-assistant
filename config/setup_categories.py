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
# Note: Base name without _long/_short suffix is mapped.
# Post-cleanup: only the 5 surviving sub7+sub8 setups remain. The vestigial
# sub-project #1 entries (ICT, level_breakout, failure_fade, squeeze_release,
# flag_continuation, momentum_breakout, vwap_reclaim/lose, gap_fill,
# orb_breakout/breakdown/pullback, support/resistance bounce/breakdown/breakout,
# trend_pullback/continuation/reversal, volume_*, range_*, first_hour_momentum,
# fair_value_gap, premium/discount_zone, order_block, etc.) were removed
# alongside their detector source files — those setups never validated through
# Stage 3 / Phase 6 / Phase 7. The 3 sub8 setups (cpr_mean_revert,
# narrow_cpr_breakout, vwap_first_pullback) were also removed after Phase 7
# OOS confirmed no edge.
SETUP_CATEGORIES: Dict[str, SetupCategory] = {
    # Sub-project #7 — Indian-native setups (surviving)
    "gap_fade": SetupCategory.REVERSION,

    # Sub-project #8 — Extended Indian-native setups (surviving)
    "orb_15": SetupCategory.MOMENTUM,
    "pdh_pdl_reject": SetupCategory.REVERSION,
    "pdh_pdl_sweep_reclaim": SetupCategory.REVERSION,
    "gap_and_go_continuation": SetupCategory.MOMENTUM,
    "ema5_alert_pullback": SetupCategory.MOMENTUM,
    "camarilla_l3_reversal": SetupCategory.REVERSION,
    # expiry_pin_strike_reversal entry removed 2026-05-14 (retired — see docs/retired_setups.md)

    # 2026-05-15: LONG mean-revert on deep small/mid panic gap-downs. Ships
    # disabled pending universe-wide regime-guard orchestrator wiring.
    "long_panic_gap_down": SetupCategory.REVERSION,

    # 2026-05-21: LONG mean-revert on stocks >=2% below session VWAP with 10x+
    # same-time-of-day volume in afternoon. Paper-trade validation phase; ships
    # live-disabled with paper_enabled=true. See specs/2026-05-21-below_vwap_volume_revert_long-paper-trade-spec.md.
    "below_vwap_volume_revert": SetupCategory.REVERSION,

    # 2026-06-03: deep illiquid intraday over-extension fades (both directions).
    # LONG bounce off a >=7% crash; SHORT fade of a >=5.5% spike on a >5x volume
    # blowoff. Both ship disabled pending OCI-coverage + paper. Base names (suffix
    # stripped by get_base_setup_name). Specs: specs/2026-06-03-brief-*.md.
    "panic_crash_revert": SetupCategory.REVERSION,
    "up_spike_fade": SetupCategory.REVERSION,

    # 2026-06-15: multi-day CNC/MTF cross-sectional capitulation-reversion batch
    # (2-3 day delivery hold). Deep-loser / near-52w-low / z-score-oversold / 2d-
    # crash variants, all on the MTF-eligible illiquid tier-1 cohort. Ship live-
    # disabled with paper_enabled=true (paper-validation batch). Base names (suffix
    # stripped by get_base_setup_name). Spec: specs/2026-06-14-brief-mtf_capitulation_revert_long.md.
    "mtf_capitulation_revert": SetupCategory.REVERSION,
    "low52_capitulation_revert": SetupCategory.REVERSION,
    "zscore_oversold_revert": SetupCategory.REVERSION,
    "crash2d_revert": SetupCategory.REVERSION,
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
